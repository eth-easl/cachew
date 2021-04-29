#include "tensorflow/core/kernels/data/experimental/easl_service/service_cache_util.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/protobuf/service_cache.pb.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/kernels/data/experimental/easl_service/arrow/arrow_async_writer.h"
#include "tensorflow/core/kernels/data/experimental/easl_service/arrow/arrow_async_reader.h"



namespace tensorflow {
namespace data {
namespace easl{
namespace service_cache_util {

namespace { // anonymous namespace => declared functions only visible within this file
static constexpr const char *const kCacheLocation = "";

std::string GetFileName(const std::string& shard_directory,
                                uint64 file_id, uint64 split_id = 0) {
  return io::JoinPath(shard_directory, strings::Printf("%08llu.easl",
                      static_cast<unsigned long long>(file_id)));
}

}

constexpr const char* const kMetadataFilename = "service_cache.metadata";
const int64 kWriterVersion = 2; // 0 --> ArrowWriter; 2 --> TFRecordWriter
const char kCompression[] = ""; // can be SNAPPY, GZIP, ZLIB, "" for none.

Writer::Writer(Env* env,
    const std::string& target_dir, const DataTypeVector& output_dtypes,
    const std::vector<PartialTensorShape>& output_shapes, 
    const int writer_count) : env_(env), target_dir_(target_dir), 
    output_dtypes_(output_dtypes), output_shapes_(output_shapes), 
    writer_count_(writer_count) {}  // Constructor, store references in object

Writer::~Writer() {}  // ~ Destructor

Status Writer::Initialize(){
  // TODO (damien-aymon) add constant for writer version.

  if(kWriterVersion == 0) { // 0 -> arrow
    async_writer_ = std::make_unique<arrow_async_writer::ArrowAsyncWriter>(writer_count_);
  } else {
    async_writer_ = std::make_unique<MultiThreadedAsyncWriter>(writer_count_);
  }

  async_writer_->Initialize(env_, /*file_index*/ 0, target_dir_, /*checkpoint_id*/ 0,
                            kCompression, kWriterVersion, output_dtypes_,
          /*done*/ [this](Status s){
              // TODO (damien-aymon) check and propagate errors here!
              if (!s.ok()) {
                VLOG(0) << "EASL - writer error: "<< s.ToString();
              }
              //LOG(ERROR) << "MultiThreadedAsyncWriter in snapshot writer failed: " << s;
              //mutex_lock l(writer_status_mu_);
              //writer_status_ = s;
              return;
          });

  return WriteMetadataFile(env_, target_dir_, output_dtypes_, output_shapes_);
}

Status Writer::Write(const std::vector<Tensor>& tensors){
  async_writer_->Write(tensors);
  // TODO (damien-aymon) check for errors in the async writer
  return Status::OK();
}

Status Writer::Close(){
  // Will call the destructor and block until done writing.
  async_writer_->SignalEOF();
  async_writer_.reset();
  // TODO(damien-aymon) check status in the async writer.
  return Status::OK();
}

Status Writer::WriteMetadataFile(
    Env* env, const std::string& path, const DataTypeVector& output_dtypes,
    const std::vector<PartialTensorShape>& output_shapes){
  experimental::CacheMetadataRecord metadata;
  metadata.set_creation_timestamp(EnvTime::NowMicros());
  metadata.set_version(kWriterVersion);
  for (const auto& output_dtype : output_dtypes) {
    metadata.add_dtype(output_dtype);
  }
  for (const auto& output_shape : output_shapes){
    TensorShapeProto* shape_proto = metadata.add_tensor_shape();
    output_shape.AsProto(shape_proto);
  }
  metadata.set_num_writers(writer_count_);

  string metadata_filename = io::JoinPath(target_dir_, kMetadataFilename);
  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(target_dir_));
  std::string tmp_filename =
      absl::StrCat(metadata_filename, "-tmp-", random::New64());
  TF_RETURN_IF_ERROR(WriteBinaryProto(env, tmp_filename, metadata));
  return env->RenameFile(tmp_filename, metadata_filename);
}

// -----------------------------------------------------------------------------
// MultiThreadedAsyncWriter
// -----------------------------------------------------------------------------

MultiThreadedAsyncWriter::MultiThreadedAsyncWriter(const int writer_count) : writer_count_(writer_count) {}

void MultiThreadedAsyncWriter::Initialize(Env *env, int64 file_index, const std::string &shard_directory,
        uint64 checkpoint_id, const std::string &compression, int64 version,
        const DataTypeVector &output_types, std::function<void (Status)> done) {
  thread_pool_ = absl::make_unique<thread::ThreadPool>(env, ThreadOptions(),
           absl::StrCat("thread_pool_", file_index), writer_count_, false);

  LOG(INFO) << "(MultiThreadedAsyncWriter) Starting ThreadPool";
  for (int i = 0; i < writer_count_; ++i) {
    thread_pool_->Schedule(
            [this, env, shard_directory, checkpoint_id, compression, version,
                    &output_types, done = std::move(done), i] {
                // Note that `done` is not used since it causes a bug here
                WriterThread(env, shard_directory, i, compression, version,
                             output_types);
            }
    );

  }
  LOG(INFO) << "(MultiThreadedAsyncWriter) Finished Starting ThreadPool";
}

void MultiThreadedAsyncWriter::Write(const std::vector<Tensor>& tensors) {
  mutex_lock l(mu_);
  snapshot_util::ElementOrEOF element;
  element.value = tensors;
  deque_.push_back(std::move(element));
}

void MultiThreadedAsyncWriter::SignalEOF() {
  mutex_lock l(mu_);
  
  for (int i = 0; i < writer_count_; ++i) {
    snapshot_util::ElementOrEOF be;
    be.end_of_sequence = true;
    deque_.push_back(std::move(be));
  }
}

void MultiThreadedAsyncWriter::Consume(snapshot_util::ElementOrEOF* be) {
  mutex_lock l(mu_);
  mu_.Await(tensorflow::Condition(this, 
      &MultiThreadedAsyncWriter::ElementAvailable));
  *be = deque_.front();
  deque_.pop_front();
}

bool MultiThreadedAsyncWriter::ElementAvailable() { return !deque_.empty(); }

Status MultiThreadedAsyncWriter::WriterThread(Env* env, 
                                 const std::string& shard_directory,
                                 uint64 writer_id,
                                 const std::string& compression, int64 version,
                                 DataTypeVector output_types) {

  // TODO (damien-aymon) Push this to the specific writers, so that we can make
  // the async writer more general (e.g. different file system, gs://, etc...)
  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(shard_directory));
  LOG(INFO) << "(Writer_" << writer_id << ") Created Dir ";

  std::unique_ptr<snapshot_util::Writer> writer;

  TF_RETURN_IF_ERROR(snapshot_util::Writer::Create(
          env, GetFileName(shard_directory, writer_id),
          compression, version, std::move(output_types), &writer));

  int count = 0;
  LOG(INFO) << "(Writer_" << writer_id << ") Starting to write "; 

  while (true) {
    snapshot_util::ElementOrEOF be;
    Consume(&be);

    LOG(INFO) << "(Writer_" << writer_id << ") Read - " 
      << be.end_of_sequence << " - Total: " << ++count;
    if (be.end_of_sequence) {
      writer->Close();
      LOG(INFO) << "(Writer_" << writer_id << ") Closed w/ total read " 
                << count;
      break;
    }

    TF_RETURN_IF_ERROR(writer->WriteTensors(be.value));
  }
  return Status::OK();
}

// -----------------------------------------------------------------------------
// Reader
// -----------------------------------------------------------------------------

Reader::Reader(Env *env, const std::string &target_dir, const DataTypeVector& output_dtypes,
        const std::vector<PartialTensorShape>& output_shapes, const int reader_count)
        : target_dir_(target_dir), env_(env), output_dtypes_(output_dtypes), reader_count_(reader_count) {}

Status Reader::Initialize() {

  if(kWriterVersion == 0) { // 0 -> arrow
    async_reader_ = std::make_unique<arrow_async_reader::ArrowAsyncReader>(env_, target_dir_, output_dtypes_,
                                                               output_shapes_, reader_count_);
  } else {
    async_reader_ = std::make_unique<MultiThreadedAsyncReader>(env_, target_dir_, output_dtypes_,
                                                               output_shapes_, reader_count_);
  }

  return async_reader_->Initialize();
}

Status Reader::Read(std::vector<Tensor>* &read_tensors, bool* end_of_sequence) {
  return async_reader_->Read(read_tensors, end_of_sequence);
}


// -----------------------------------------------------------------------------
// MultiThreadedAsyncReader (Base Class for ArrowAsyncReader)
// -----------------------------------------------------------------------------



MultiThreadedAsyncReader::MultiThreadedAsyncReader(Env *env, const std::string &target_dir,
                                                   const DataTypeVector &output_dtypes,
                                                   const std::vector<PartialTensorShape> &output_shapes,
                                                   const int reader_count)
    : env_(env), output_dtypes_(output_dtypes), output_shapes_(output_shapes),
    target_dir_(target_dir), reader_count_(reader_count), tensors_() {
  this->num_readers_done_ = 0;
  // TODO (damien-aymon) add constant for writer version.
}

Status MultiThreadedAsyncReader::Initialize() {
  // Don't use metadata file at the moment...
  // TF_RETURN_IF_ERROR(ReadAndParseMetadataFile());
  
  // Find all the files of this dataset
  std::vector<string> files;
  TF_CHECK_OK(env_->GetMatchingPaths(io::JoinPath(target_dir_, "*\\.easl"), 
      &files));
  file_count_ = files.size();

  { 
    mutex_lock l(mu_);
    for (const auto& f : files) {
      file_names_.push_back(f);
    }
  }

  // Spawn the threadpool, and start reading from the files
  thread_pool_ = absl::make_unique<thread::ThreadPool>(env_, ThreadOptions(),  
      absl::StrCat("reader_thread_pool", reader_count_), reader_count_, false);

  LOG(INFO) << "(Reader) Starting ThreadPool"; 
  for (int i = 0; i < reader_count_; ++i) {
    thread_pool_->Schedule(
      [this, i] {
        // ReaderThread(env_, i, cache_file_version_, output_dtypes_);
        ReaderThread(env_, i, kWriterVersion, output_dtypes_, output_shapes_);
        }
    );
  }
  LOG(INFO) << "(Reader) Finished Starting ThreadPool";
}

void MultiThreadedAsyncReader::Consume(string* s, bool* end_of_sequence) {
  mutex_lock l(mu_);
  if (file_names_.empty()) {
    *s = ""; 
    *end_of_sequence = true;
  } else {
    *s = file_names_.front();
    file_names_.pop_front();
    *end_of_sequence = false;
  }
}

void MultiThreadedAsyncReader::Add(std::vector<Tensor>& tensors) {
  mutex_lock l(mu_add_);
  for (const auto& t : tensors)
    tensors_.push_back(t);

  read_cv_.notify_one();
}

Status MultiThreadedAsyncReader::ReaderThread(Env *env, uint64 writer_id, int64 version,
  DataTypeVector output_types, std::vector<PartialTensorShape> output_shapes) {

  // Debugging
  std:string d_string = DataTypeVectorString(output_types);
  LOG(INFO) << "(Reader_" << writer_id << ") Starting reading task\n\tREADING D_TYPE:\t" << d_string;


  tensorflow::profiler::TraceMe activity(
          "EASLReaderThread", tensorflow::profiler::TraceMeLevel::kVerbose);

  bool end_of_sequence = false; 

  while (!end_of_sequence) {
    std::string file_path;
    Consume(&file_path, &end_of_sequence);
    LOG(INFO) << "(Reader_" << writer_id << ") Got file " << file_path;

    if (!end_of_sequence) {
      LOG(INFO) << "(Reader_" << writer_id << ") Reading file " << file_path;

      std::unique_ptr<snapshot_util::Reader> reader;

      snapshot_util::Reader::Create(env, file_path, io::compression::kNone,
                                    version, output_types, &reader);


      LOG(INFO) << "(Reader_" << writer_id << ") Starting to read file " << file_path;
      int64 count = 0;
      bool eof = false;
      while (!eof) {
        std::string t_str = "Reading Tensors:";
        std::vector<Tensor> tensors;
        Status s = reader->ReadTensors(&tensors);
        if (errors::IsOutOfRange(s)) {
          eof = true;  // can't break because of TFRecordReader.
        } else if(s != Status::OK()) {
          LOG(INFO) << "Internal error in TFRecordReader. " << s.ToString();
          return s;
        }

        if(!tensors.empty()) {
          Add(tensors);
        }
      }
      LOG(INFO) << "(Reader_" << writer_id << ") Finished reading file " << file_path
      << " with " << count << " elements.";
    }
  }

  mutex_lock l(mu_add_);
  num_readers_done_++;
  read_cv_.notify_one();

  LOG(INFO) << "(Reader_" << writer_id << ") Finishing reading task";
  return Status::OK();
}

Status MultiThreadedAsyncReader::Read(std::vector<Tensor>* &read_tensors, bool* end_of_sequence) {
  mutex_lock l(mu_add_);
  *end_of_sequence = false;
  int64 n = output_dtypes_.size();

  VLOG(0) << "(Reader) Task is getting invoked... Reading " << n;
  while(true){
    if(!tensors_.empty()){
      while (n > 0) {
        n--;
        read_tensors->push_back(tensors_.front());
        tensors_.pop_front();
      }
      LOG(INFO) << "(Reader) Task have read " << n;
      return Status::OK();
    } else {
      if(num_readers_done_ == reader_count_){
        *end_of_sequence = true;
        
        LOG(INFO) << "(Reader) End of sequence reached, returning empty.";
        return Status::OK();
      }
      // Readers are not done, waiting on data...
      LOG(INFO) << "(Reader) Task could not read, waiting... ";
      read_cv_.wait(l);
    }
  }
  
  /*
  if (num_readers_done_ == reader_count_) {
    *end_of_sequence = true;
    LOG(INFO) << "(Reader) End of sequence reached, returning last tensors.";
    return Status::OK();
  }
  return Status::OK();*/
}

MultiThreadedAsyncReader::~MultiThreadedAsyncReader(){}

Status MultiThreadedAsyncReader::ReadAndParseMetadataFile() {
  string metadata_filename = io::JoinPath(target_dir_, kMetadataFilename);
  TF_RETURN_IF_ERROR(env_->FileExists(metadata_filename));

  experimental::CacheMetadataRecord metadata;
  TF_RETURN_IF_ERROR(ReadBinaryProto(env_, metadata_filename, &metadata));

  cache_file_version_ = metadata.version();

  output_dtypes_ = DataTypeVector();
  for(auto dtype : metadata.dtype()){
    output_dtypes_.push_back(static_cast<DataType>(dtype));
  }

  output_shapes_ = std::vector<PartialTensorShape>();
  for(auto &shape : metadata.tensor_shape()){
    output_shapes_.emplace_back(PartialTensorShape(shape));
  }

  return Status::OK();
}


} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow