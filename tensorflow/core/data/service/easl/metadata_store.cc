#include "tensorflow/core/data/service/easl/metadata_store.h"

#include <memory>
#include <ostream>
#include <fstream>

#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace data {
namespace easl {

namespace {
  const uint8 kMaxClientMetricsHistory = 20;
  const uint8 kMaxPerWorkerClientMetricHistory = 12;
}


// Metrics from the Client
ModelMetrics::Metrics::Metrics() {}


ModelMetrics::Metrics::Metrics(int64 worker_count,
    double last_x_batch_time_ms, double relative_wait_fraction,
    double result_queue_size):
      worker_count_(worker_count),
      last_x_batch_time_ms_(last_x_batch_time_ms),
      relative_wait_fraction_(relative_wait_fraction),
      result_queue_size_(result_queue_size){}

ModelMetrics::Metrics::Metrics(int64 worker_count,
                               int64 local_worker_count,
                               double last_x_batch_time_ms, double relative_wait_fraction,
                               double result_queue_size):
        worker_count_(worker_count),
        remote_worker_count_(worker_count - local_worker_count), // worker_count is from tasks_.size(), (includes local + remote)
        local_worker_count_(local_worker_count),
        last_x_batch_time_ms_(last_x_batch_time_ms),
        relative_wait_fraction_(relative_wait_fraction),
        result_queue_size_(result_queue_size){}

ModelMetrics::Metrics::Metrics(ModelMetrics::Metrics& other) :
    worker_count_(other.worker_count_),
    remote_worker_count_(other.remote_worker_count_),
    local_worker_count_(other.local_worker_count_),
    last_x_batch_time_ms_(other.last_x_batch_time_ms_),
    relative_wait_fraction_(other.relative_wait_fraction_),
    result_queue_size_(other.result_queue_size_){}

Status ModelMetrics::UpdateClientMetrics(
  const int64 client_id,
  ModelMetrics::Metrics& metrics) {
  //using MetricsCollection =
  //  absl::flat_hash_map<int64, std::deque<std::shared_ptr<ModelMetrics::Metrics>>>;

  auto metrics_ptr = std::make_shared<Metrics>(metrics);

//  VLOG(0) << "EASL-MUYU: UpdateClientMetrics: "
//    << "remote_worker_count: " << metrics_ptr->remote_worker_count()
//    << "; local_worker_count: " << metrics_ptr->local_worker_count();

  int64 worker_count = metrics.worker_count();
  // Level 1 - worker_count
  auto worker_count_it = metrics_.find(worker_count);
  if(worker_count_it == metrics_.end()){
    auto metrics_collection = std::make_shared<MetricsCollection>();
    metrics_.insert({worker_count, metrics_collection});
    worker_count_it = metrics_.find(worker_count);
  }
  auto metrics_collection = worker_count_it->second;

  // Level 2 - client_id
  auto it = metrics_collection->find(client_id);
  if (it == metrics_collection->end()) {
    auto entry = std::deque<std::shared_ptr<Metrics>>();
    entry.push_back(metrics_ptr);
    metrics_collection->insert({client_id, entry});
    VLOG(2) << "Created model metrics for client " << client_id;
  } else {
    if (it->second.size() >= kMaxPerWorkerClientMetricHistory){
      it->second.pop_front();
    }
    it->second.push_back(metrics_ptr);
    VLOG(2) << "Appended model metrics for client " << client_id;
  }

  if (metrics_history_.size() >= kMaxClientMetricsHistory){
    metrics_history_.pop_front();
  }
  metrics_history_.push_back(metrics_ptr);

  return Status::OK();
}

Status ModelMetrics::GetAllWorkerCountMetrics(std::shared_ptr<MetricsByWorkerCount>& metrics){
  metrics = std::make_shared<MetricsByWorkerCount>(metrics_);
  return Status::OK();
}

Status ModelMetrics::GetAllClientMetrics(
    const int64 worker_count, std::shared_ptr<MetricsCollection>& metrics){
  auto worker_count_it = metrics_.find(worker_count);
  if(worker_count_it != metrics_.end()) {
    metrics = worker_count_it->second;
  }
  return Status::OK();
}

Status ModelMetrics::GetClientMetrics(
    const int64 worker_count, const int64 client_id,
    std::deque<std::shared_ptr<Metrics>>& metrics) {
  auto worker_count_it = metrics_.find(worker_count);
  if(worker_count_it != metrics_.end()){
    auto it = worker_count_it->second->find(client_id);
    if (it != worker_count_it->second->end()) {
      metrics = it->second;
      return Status::OK();
    }
  }
  return errors::NotFound("No metrics under worker_count ", worker_count, " and client with id ", client_id);
}

// TODO (Damien) update to print latest metrics.
void ModelMetrics:: DumpToStream(std::stringstream& ss){
  ss << "{ " << std::endl;
  bool first_w_count = true;
  for(auto pair : metrics_){
    if(!first_w_count){
      ss << ", \n";
      first_w_count = false;
    }
    ss << "{ \"worker_count\": " << std::to_string(pair.first) << ", \"client_metrics\": { ";

    bool first = true;
    for ( auto client_metrics_pair : *(pair.second)){
      if(!first){ ss << "," << std::endl; first = false; } // Add comma before element, not for first though.
      ss << "\"" << std::to_string(client_metrics_pair.first) << "\" : ";
      ss << "{ \"get_next_time_ms\" : " << 0 << " , ";
      ss << "\"inter_arrival_time_ms\" : " << 0 << " }";
      ss << std::endl;
    }
    ss << " }";
  }
  ss << "}";
}


// Metrics from the Worker Nodes
NodeMetrics::Metrics::Metrics(NodeMetrics::Metrics& other) 
  : bytes_consumed_(other.bytes_consumed()),
    bytes_produced_(other.bytes_produced()),
    num_elements_(other.num_elements()),
    bytes_per_s_(other.bytes_per_s()),
    // computation_time_(other.computation_time()),
    in_node_time_ms_(other.in_node_time_ms()),
    in_prefix_time_ms_(other.in_prefix_time_ms()),
    active_time_ms_(other.active_time_ms()),
    working_time_ms_(other.working_time_ms()){}

NodeMetrics::Metrics::Metrics(int64 bytes_consumed, int64 bytes_produced, 
  int64 num_elements, int64 bytes_per_s,
  // int64 computation_time, 
  double in_node_time_ms, double in_prefix_time_ms, double active_time_ms,
  double working_time_ms)
  : bytes_consumed_(bytes_consumed),
    bytes_produced_(bytes_produced),
    num_elements_(num_elements),
    bytes_per_s_(bytes_per_s),
    // computation_time_(computation_time),
    in_node_time_ms_(in_node_time_ms),
    in_prefix_time_ms_(in_prefix_time_ms),
    active_time_ms_(active_time_ms),
    working_time_ms_(working_time_ms){}

void NodeMetrics::Metrics::Update(NodeMetrics::Metrics& other) {
  bytes_consumed_ = other.bytes_consumed_;
  bytes_produced_ = other.bytes_produced_;
  num_elements_ = other.num_elements_;
  // computation_time_ = other.computation_time_;
  in_node_time_ms_ = other.in_node_time_ms_;
  in_prefix_time_ms_ = other.in_prefix_time_ms_;
  active_time_ms_ = other.active_time_ms_;
  working_time_ms_ = other.working_time_ms_;
}

Status NodeMetrics::UpdateWorkerMetrics(string worker_address, 
  NodeMetrics::Metrics& metrics) {
  auto it = metrics_.find(worker_address);
  if (it != metrics_.end()) {
    it->second->Update(metrics);
  } else {
    auto entry = std::make_shared<NodeMetrics::Metrics>(metrics);
    metrics_.insert({worker_address, entry});
  }
  return Status::OK();
}

Status NodeMetrics::GetWorkerMetrics(string worker_address, 
  std::shared_ptr<Metrics>& metrics) {
  auto it = metrics_.find(worker_address);
  if (it != metrics_.end()) {
    metrics = it->second;
    return Status::OK();
  }
  return errors::NotFound("No metrics under the worker with address ", 
    worker_address);
}

void NodeMetrics::DumpToStream(std::stringstream &ss) {
  ss << "{ " << std::endl;
  bool first = true;
  for(auto pair : metrics_){
    // Add comma before element, not for first though.
    if(!first){ ss << "," << std::endl; }
    else { first = false; }
    ss << "\"" << pair.first << "\" : ";
    ss << "{ \"bytes_consumed\" : " << std::to_string(pair.second->bytes_consumed()) << " , ";
    ss << "\"bytes_produced\" : " << std::to_string(pair.second->bytes_produced()) << " , ";
    ss << "\"num_elements\" : " << std::to_string(pair.second->num_elements()) << " , ";
    // ss << "\"computation_time\" : " << std::to_string(pair.second->computation_time()) << " , ";
    ss << "\"in_node_time_ms\" : " << std::to_string(pair.second->in_node_time_ms()) << " , ";
    ss << "\"in_prefix_time_ms\" : " << std::to_string(pair.second->in_prefix_time_ms()) << " , ";
    ss << "\"working_time_ms\" : " << std::to_string(pair.second->working_time_ms()) << " , ";
    ss << "\"active_time_ms\" : " << std::to_string(pair.second->active_time_ms()) << " }";

    ss << std::endl;
  }
  ss << "}";
}

// Input pipeline metrics
Status InputPipelineMetrics::GetNodeMetrics(string long_name, 
  std::shared_ptr<NodeMetrics>& metrics) {
  auto it = metrics_.find(long_name);
  if (it != metrics_.end()) {
    metrics = it->second;
    return Status::OK();
  }
  return errors::NotFound("No metrics for node ", long_name); 
}

Status InputPipelineMetrics::GetLastNodeMetrics(
  std::shared_ptr<NodeMetrics>& metrics) {
  if (last_node_name_ == "") {
    return errors::NotFound("Last node was not given a name");
  }
  GetNodeMetrics(last_node_name_, metrics);
  return Status::OK();
}

Status InputPipelineMetrics::GetLastTFNodeMetrics(
  std::shared_ptr<NodeMetrics>& metrics) {
  if (last_tf_node_name_ == "") {
    return errors::Unavailable("Last TF node was not given a name");
  }
  GetNodeMetrics(last_tf_node_name_, metrics);
  return Status::OK();
}

Status InputPipelineMetrics::GetMarkerNodeMetrics(
  std::shared_ptr<NodeMetrics>& metrics) {
  if (maker_node_name_ == "") {
    return errors::Unavailable("Marker node was not given a name");
  }
  GetNodeMetrics(maker_node_name_, metrics);
  return Status::OK();
}

Status InputPipelineMetrics::GetWorkerMetrics(string worker_address, 
  NodeMetrics::MetricsCollection& metrics) {
  for (auto& entry : metrics_) {
    std::shared_ptr<NodeMetrics::Metrics> node_metrics;
    Status s = entry.second->GetWorkerMetrics(worker_address, node_metrics);
    if (s.ok()) {
      metrics.insert({entry.first, node_metrics});
    }
  }
  return Status::OK();
}

Status InputPipelineMetrics::UpdateNodeMetrics(string long_name,
  string worker_address, NodeMetrics::Metrics& metrics) {
  auto it = metrics_.find(long_name); 
  if (it == metrics_.end()) {
    auto node_metrics = std::make_shared<NodeMetrics>();
    node_metrics->UpdateWorkerMetrics(worker_address, metrics);
    metrics_.insert({long_name, node_metrics});
    VLOG(2) << "Created node " << long_name << "'s metrics for worker " 
            << worker_address;
  } else {
    it->second->UpdateWorkerMetrics(worker_address, metrics);
    VLOG(2) << "Updated node " << long_name << "'s metrics for worker " 
            << worker_address;
  }
  return Status::OK();
}

std::string InputPipelineMetrics::GetLastNodeName() { return last_node_name_; }
void InputPipelineMetrics::SetLastNodeName(std::string last_node_name) {
  last_node_name_ = last_node_name;
}

std::string InputPipelineMetrics::GetLastTFNodeName() { return last_tf_node_name_; }
void InputPipelineMetrics::SetLastTFNodeName(std::string last_tf_node_name) {
  last_tf_node_name_ = last_tf_node_name;
}

std::string InputPipelineMetrics::GetMarkerNodeName() { return maker_node_name_; }
void InputPipelineMetrics::SetMarkerNodeName(std::string maker_node_name) {
  maker_node_name_ = maker_node_name;
}

void InputPipelineMetrics::DumpToStream(std::stringstream& ss){
  ss << "{ " << std::endl;
  ss << "\"last_node_name\" : \"" << last_node_name_ << "\"";

  bool first = true;
  for(auto pair : metrics_){
    ss << ", " << std::endl;
    ss << "\"" << pair.first << "\" : ";
    pair.second->DumpToStream(ss);
  }
  ss << std::endl << "}";
}


// Job metrics
JobMetrics::JobMetrics(int64 job_id,
                       std::string& job_type,
                       int64 dataset_id,
                       uint64 dataset_fingerprint,
                       std::string& dataset_key,
                       bool is_scaling,
                       const string& name)
      : job_id_(job_id),
        job_type_(job_type),
        dataset_id_(dataset_id),
        dataset_fingerprint_(dataset_fingerprint),
        dataset_key_(dataset_key),
        name_(name),
        model_metrics_(), 
        input_pipeline_metrics_(),
        is_scaling_(is_scaling),
        scaling_state_(JobScalingState::ONLY_REMOTE),
        target_worker_count_(1),
        target_remote_worker_count_(1),
        target_local_worker_count_(0),
        same_scale_counter_(0),
        last_performance_(Performance::NA) {
          model_metrics_ = std::make_shared<ModelMetrics>();
          input_pipeline_metrics_ = std::make_shared<InputPipelineMetrics>();
        }

// For cases that start from local worker
JobMetrics::JobMetrics(
        int64 job_id,
                       std::string& job_type,
                       int64 dataset_id,
                       uint64 dataset_fingerprint,
                       std::string& dataset_key,
                       bool is_scaling,
                       const string& name,
                       int64 target_remote_worker_count,
                       int64 target_local_worker_count,
                       JobScalingState scaling_state
                       )
        : job_id_(job_id),
          job_type_(job_type),
          dataset_id_(dataset_id),
          dataset_fingerprint_(dataset_fingerprint),
          dataset_key_(dataset_key),
          name_(name),
          model_metrics_(),
          input_pipeline_metrics_(),
          is_scaling_(is_scaling),
          scaling_state_(scaling_state),
          target_worker_count_(1),
          target_remote_worker_count_(target_remote_worker_count),
          target_local_worker_count_(target_local_worker_count),
          same_scale_counter_(0),
          last_performance_(Performance::NA) {
  model_metrics_ = std::make_shared<ModelMetrics>();
  input_pipeline_metrics_ = std::make_shared<InputPipelineMetrics>();
}


// JobMetrics
void JobMetrics::DumpToFile(const std::string& path){
  // Start constructing json string:
  std::stringstream ss;
  DumpToStream(ss);

  std::ofstream fstream;
  fstream.open(
      path + "/metrics_job_" + std::to_string(job_id_) + "_ds_key_" + dataset_key_  + ".json",
      std::ofstream::out | std::ofstream::app);
  fstream << ss.rdbuf();
  fstream.close();
}

void JobMetrics::DumpToStream(std::stringstream& ss){
  ss << "{" << std::endl;
  ss << "\"ModelMetrics\" : ";
  model_metrics_->DumpToStream(ss);
  ss << "," << std::endl;
  ss << "\"PipelineMetrics\" : ";
  input_pipeline_metrics_->DumpToStream(ss);
  ss << "}" << std::endl;
}


// Metadata store 
MetadataStore::MetadataStore() 
  : job_metadata_(),
    fingerprint_key_metadata_(),
    fingerprint_name_metadata_() {}

Status MetadataStore::CreateJob(int64 job_id, string& job_type,
  int64 dataset_id, uint64 dataset_fingerprint, std::string& dataset_key,
  bool trigger_rescale) {
  auto it = fingerprint_key_metadata_.find(dataset_fingerprint);
  if ( it == fingerprint_key_metadata_.end()) {
    // We've never seen this input pipeline; it's expected to be a PROFILING job
    CHECK_EQ(job_type, "PROFILE");
    std::string ds_key = dataset_key;
    auto job_metrics = std::make_shared<JobMetrics>(
        job_id, job_type, dataset_id, dataset_fingerprint, ds_key,
        false);
    job_metadata_.insert_or_assign(job_id, job_metrics);

    return Status::OK();
  }

  // TODO FIXME This is not a deep copy of the JobMetrics object
  // Multiple clients could copy the same object and share it, the second client would overwrite the job_id_, .. fields.
  std::shared_ptr<JobMetrics> job_metrics = it->second;
  job_metrics->job_id_ = job_id;
  job_metrics->job_type_ = job_type;
  job_metrics->dataset_id_ = dataset_id;
  job_metrics->dataset_key_ = dataset_key;

  if (trigger_rescale) {
    job_metrics->is_scaling_ = true;
    job_metrics->same_scale_counter_ = 0;
    job_metrics->target_worker_count_ = 1;
    // TODO MUYU has never seen this logic triggering
    job_metrics->model_metrics_->metrics_history_.clear();
  }

  job_metadata_.insert_or_assign(job_id, job_metrics);

  return Status::OK();
}

Status MetadataStore::CreateJobName(int64 job_id, string& job_name,
  string& job_type, int64 dataset_id, uint64 dataset_fingerprint,
  std::string& dataset_key, bool trigger_rescale) {
  string key = CreateFingerprintNameKey(dataset_fingerprint, job_name);
  auto it = fingerprint_name_metadata_.find(key);
  if ( it == fingerprint_name_metadata_.end()) {
    // We've never seen this input pipeline; it's expected to be a PROFILING job
//    CHECK_EQ(job_type, "PROFILE");
    bool is_scaing = job_type != "PROFILE";
    std::string ds_key = dataset_key;
    auto job_metrics = std::make_shared<JobMetrics>(
        job_id, job_type, dataset_id, dataset_fingerprint, ds_key,
        is_scaing, job_name);
    job_metadata_.insert_or_assign(job_id, job_metrics);

    return Status::OK();
  }

    // TODO FIXME This is not a deep copy of the JobMetrics object
    // Multiple clients could copy the same object and share it, the second client would overwrite the job_id_, .. fields.
    std::shared_ptr<JobMetrics> job_metrics = it->second;
    job_metrics->job_id_ = job_id;
    job_metrics->job_type_ = job_type;
    job_metrics->dataset_id_ = dataset_id;
    job_metrics->dataset_key_ = dataset_key;

    if (trigger_rescale) {
      job_metrics->is_scaling_ = true;
      job_metrics->same_scale_counter_ = 0;
      job_metrics->target_worker_count_ = 1;
      job_metrics->model_metrics_->metrics_history_.clear();
    }

    job_metadata_.insert_or_assign(job_id, job_metrics);

    return Status::OK();
}

Status MetadataStore::CreateJobName(int64 job_id, string& job_name,
                                    string& job_type, int64 dataset_id, uint64 dataset_fingerprint,
                                    std::string& dataset_key, bool trigger_rescale,
                                    int64 policy) {
  string key = CreateFingerprintNameKey(dataset_fingerprint, job_name);
  auto it = fingerprint_name_metadata_.find(key);
  if ( it == fingerprint_name_metadata_.end()) {
    // We've never seen this input pipeline; it's expected to be a PROFILING job
//    CHECK_EQ(job_type, "PROFILE");
    VLOG(0) << "CreateJobName:Muyu - Profile";
    bool is_scaing = job_type != "PROFILE";
    std::string ds_key = dataset_key;
    int64 target_remote_worker_count, target_local_worker_count;
    JobScalingState scaling_state;
    if (policy == 3) {
      target_remote_worker_count = 1;
      target_local_worker_count = 0;
      scaling_state = JobScalingState::ONLY_REMOTE;
    }
    else if (policy == 4) {
      target_remote_worker_count = 0;
      target_local_worker_count = 1;
      scaling_state = JobScalingState::INCREASING_LOCAL;
    }
    else {
      VLOG(0) << "CreateJobName with wrong policy option";
    }
    auto job_metrics = std::make_shared<JobMetrics>(
            job_id, job_type, dataset_id, dataset_fingerprint, ds_key,
            is_scaing, job_name, target_remote_worker_count,
            target_local_worker_count, scaling_state);
    job_metadata_.insert_or_assign(job_id, job_metrics);

    return Status::OK();
  }

  // TODO FIXME This is not a deep copy of the JobMetrics object
  // Multiple clients could copy the same object and share it, the second client would overwrite the job_id_, .. fields.
  std::shared_ptr<JobMetrics> job_metrics = it->second;
  job_metrics->job_id_ = job_id;
  job_metrics->job_type_ = job_type;
  job_metrics->dataset_id_ = dataset_id;
  job_metrics->dataset_key_ = dataset_key;

  if (trigger_rescale) {
    job_metrics->is_scaling_ = true;
    job_metrics->same_scale_counter_ = 0;
    job_metrics->target_worker_count_ = 1;
    job_metrics->model_metrics_->metrics_history_.clear();
  }

  job_metadata_.insert_or_assign(job_id, job_metrics);

  return Status::OK();
}

//Find a the job metric, delete it and add it to the dataset_key keyed metrics for persistence
Status MetadataStore::RemoveJob(int64 job_id) {
  // Update datasetKey indexed store with new JobMetrics.
  auto it = job_metadata_.find(job_id);
  if (it == job_metadata_.end()) {
    return errors::NotFound("#1 Job with id ", job_id, " does not have metrics");
  }
  auto job_metrics = it->second;

  // Properly erase job.
  job_metadata_.erase(job_id);
  return Status::OK();
}

Status MetadataStore::GetJobMetrics(int64 job_id, 
  std::shared_ptr<JobMetrics>& metrics) const {
  auto it = job_metadata_.find(job_id);
  if (it == job_metadata_.end()) {
    return errors::NotFound("#2 Job with id ", job_id, " does not have metrics");
  }
  metrics = it->second;
  return Status::OK();
}

Status MetadataStore::GetModelMetrics(int64 job_id, 
  std::shared_ptr<ModelMetrics>& metrics) const {
  std::shared_ptr<JobMetrics> job_metrics;
  Status s = GetJobMetrics(job_id, job_metrics); 
  if (s.ok()) {
    metrics = job_metrics->model_metrics_;
  }
  return s;
}

Status MetadataStore::GetInputPipelineMetrics(int64 job_id, 
  std::shared_ptr<InputPipelineMetrics>& metrics) const {
  std::shared_ptr<JobMetrics> job_metrics;
  Status s = GetJobMetrics(job_id, job_metrics);
  if (s.ok()) {
    metrics = job_metrics->input_pipeline_metrics_;
  }
  return s;
}

Status MetadataStore::GetLastNodeMetrics(int64 job_id, 
  std::shared_ptr<NodeMetrics>& metrics) const {
  std::shared_ptr<JobMetrics> job_metrics;
  Status s = GetJobMetrics(job_id, job_metrics);
  if (s.ok()) {
    return job_metrics->input_pipeline_metrics_->GetLastNodeMetrics(metrics);
  }
  return s;
}

Status MetadataStore::GetLastTFNodeMetrics(int64 job_id, 
  std::shared_ptr<NodeMetrics>& metrics) const {
  std::shared_ptr<JobMetrics> job_metrics;
  Status s = GetJobMetrics(job_id, job_metrics);
  if (s.ok()) {
    return job_metrics->input_pipeline_metrics_->GetLastTFNodeMetrics(metrics);
  }
  return s;
}

Status MetadataStore::GetMarkerNodeMetrics(int64 job_id, 
  std::shared_ptr<NodeMetrics>& metrics) const {
  std::shared_ptr<JobMetrics> job_metrics;
  Status s = GetJobMetrics(job_id, job_metrics);
  if (s.ok()) {
    return job_metrics->input_pipeline_metrics_->GetMarkerNodeMetrics(metrics);
  }
  return s;
}

Status MetadataStore::GetLastNodeMetricsByDatasetFingerprint(
    const uint64 dataset_fingerprint, std::shared_ptr<NodeMetrics>& metrics) const {
  std::shared_ptr<JobMetrics> job_metrics;
  Status s = GetJobMetricsByDatasetFingerprint(dataset_fingerprint, job_metrics);
  if (s.ok()) {
    return job_metrics->input_pipeline_metrics_->GetLastNodeMetrics(metrics);
  }
  return s;
}

Status MetadataStore::GetLastTFNodeMetricsByDatasetFingerprint(
    const uint64 dataset_fingerprint, std::shared_ptr<NodeMetrics>& metrics) const {
  std::shared_ptr<JobMetrics> job_metrics;
  Status s = GetJobMetricsByDatasetFingerprint(dataset_fingerprint, job_metrics);
  if (s.ok()) {
    return job_metrics->input_pipeline_metrics_->GetLastTFNodeMetrics(metrics);
  }
  return s;
}

Status MetadataStore::GetMarkerNodeMetricsByDatasetFingerprint(
    const uint64 dataset_fingerprint, std::shared_ptr<NodeMetrics>& metrics) const {
  std::shared_ptr<JobMetrics> job_metrics;
  Status s = GetJobMetricsByDatasetFingerprint(dataset_fingerprint, job_metrics);
  if (s.ok()) {
    return job_metrics->input_pipeline_metrics_->GetMarkerNodeMetrics(metrics);
  }
  return s;
}

Status MetadataStore::GetJobMetricsByDatasetFingerprint(
    const uint64 dataset_fingerprint, std::shared_ptr<JobMetrics>& metrics) const {
  auto it = fingerprint_key_metadata_.find(dataset_fingerprint);
  if (it == fingerprint_key_metadata_.end()) {
    return errors::NotFound("Dataset ", dataset_fingerprint, " does not (yet) have metrics");
  }
  metrics = it->second;
  return Status::OK();
}

Status MetadataStore::GetJobMetricsByDatasetFingerprintAndName(
    const uint64 dataset_fingerprint, const string& job_name,
  std::shared_ptr<JobMetrics>& metrics) const {
  string key = CreateFingerprintNameKey(dataset_fingerprint, job_name);
  auto it = fingerprint_name_metadata_.find(key);
  if (it == fingerprint_name_metadata_.end()) {
    return errors::NotFound("Dataset ", key, " does not (yet) have metrics");
  }
  metrics = it->second;
  return Status::OK();
}

Status MetadataStore::GetModelMetricsByDatasetFingerprint(
    const uint64 dataset_fingerprint, std::shared_ptr<ModelMetrics>& metrics) const {
  std::shared_ptr<JobMetrics> job_metrics;
  Status s = GetJobMetricsByDatasetFingerprint(dataset_fingerprint, job_metrics);
  if (s.ok()) {
    metrics = job_metrics->model_metrics_;
  }
  return s;
}

Status MetadataStore::GetInputPipelineMetricsByDatasetFingerprint(
    const uint64 dataset_fingerprint, std::shared_ptr<InputPipelineMetrics>& metrics) const {
  std::shared_ptr<JobMetrics> job_metrics;
  Status s = GetJobMetricsByDatasetFingerprint(dataset_fingerprint, job_metrics);
  if (s.ok()) {
    metrics = job_metrics->input_pipeline_metrics_;
  }
  return s;
}

Status MetadataStore::UpdateModelMetrics(
  int64 job_id, int64 client_id,
  ModelMetrics::Metrics& metrics) {
  std::shared_ptr<ModelMetrics> model_metrics;
  Status s = GetModelMetrics(job_id, model_metrics);
  // Test if we are getting correct batch times metrics (i.e. the duration is not negative)
  double l_batch_time = metrics.last_x_batch_time_ms_;
  if (l_batch_time <= 0) {
    VLOG(0) << "Incorrect batch time metrics: " << l_batch_time;
  }
  if (s.ok()) {
    model_metrics->UpdateClientMetrics(client_id, metrics);
  } 
  // if s != ok --> no such job exists
  return s;
}

Status MetadataStore::UpdateInputPipelineMetrics(int64 job_id, 
  string node_long_name, string worker_address, NodeMetrics::Metrics& metrics) {
  std::shared_ptr<InputPipelineMetrics> pipeline_metrics;
  Status s = GetInputPipelineMetrics(job_id, pipeline_metrics);
  if (s.ok()) {
    pipeline_metrics->update_counter_++;
    pipeline_metrics->UpdateNodeMetrics(node_long_name, worker_address, 
      metrics);
  } 
  // if s != ok --> no such job exists
  return s;
}

Status MetadataStore::UpdateFingerprintKeyJobMetrics(int64 job_id) {
  auto it = job_metadata_.find(job_id);
  if (it == job_metadata_.end()) {
    return errors::NotFound("#3 Job with id ", job_id, " does not have metrics");
  }
  auto job_metrics = it->second;
  fingerprint_key_metadata_.insert_or_assign(job_metrics->dataset_fingerprint_, job_metrics);

  return Status::OK();
}

Status MetadataStore::UpdateFingerprintNameKeyJobMetrics(int64 job_id) {
  auto it = job_metadata_.find(job_id);
  if (it == job_metadata_.end()) {
    return errors::NotFound("#4 Job with id ", job_id, " does not have metrics");
  }

  auto job_metrics = it->second;
  if (job_metrics->name_ == "") {
    return errors::NotFound("Job with id ", job_id, " and name '",
      job_metrics->name_, "' has empty name.");
  }
  string key = CreateFingerprintNameKey(job_metrics->dataset_fingerprint_,
                                        job_metrics->name_);
  fingerprint_name_metadata_.insert_or_assign(key, job_metrics);

  return Status::OK();
}

Status MetadataStore::UpdateNodeNames(int64 job_id, string last_node_name, 
  string last_tf_node_name, string marker_node_name) {
  std::shared_ptr<InputPipelineMetrics> pipeline_metrics;
  Status s = GetInputPipelineMetrics(job_id, pipeline_metrics);
  if (s.ok()) {
    pipeline_metrics->SetLastNodeName(last_node_name);
    pipeline_metrics->SetLastTFNodeName(last_tf_node_name);
    pipeline_metrics->SetMarkerNodeName(marker_node_name);
  } 
  // if s != ok --> no such job exists
  return s;
}

Status MetadataStore::SetJobType(uint64 fingerprint, string job_type) {
  auto it = fingerprint_key_metadata_.find(fingerprint);
  if (it == fingerprint_key_metadata_.end()) {
    return errors::NotFound("Could not find the dataset");
  }
  // it->second should be of std::shared_ptr<JobMetrics> type
  it->second->job_type_ = job_type;
  return Status::OK();
}

  Status MetadataStore::SetJobType(uint64 fingerprint, const string& job_name,
    string job_type) {
    string key = CreateFingerprintNameKey(fingerprint, job_name);
    auto it = fingerprint_name_metadata_.find(key);
    if (it == fingerprint_name_metadata_.end()) {
      return errors::NotFound("Could not find the dataset");
    }
    // it->second should be of std::shared_ptr<JobMetrics> type
    it->second->job_type_ = job_type;
    return Status::OK();
  }

Status MetadataStore::GetJobType(uint64 fingerprint, string& job_type) {
  auto it = fingerprint_key_metadata_.find(fingerprint);
  if (it == fingerprint_key_metadata_.end()) {
    return errors::NotFound("Could not find the dataset");
  }
  // it->second should be of std::shared_ptr<JobMetrics> type
  job_type = it->second->job_type_;
  return Status::OK();
}

string MetadataStore::CreateFingerprintNameKey(uint64 fingerprint,
  const string& job_name) const {
  return std::to_string(fingerprint) + job_name;
}

Status MetadataStore::GetJobType(uint64 fingerprint, const string& job_name,
    string& job_type) {
    string key = CreateFingerprintNameKey(fingerprint, job_name);
    auto it = fingerprint_name_metadata_.find(key);
    if (it == fingerprint_name_metadata_.end()) {
      return errors::NotFound("Could not find the dataset");
    }
    // it->second should be of std::shared_ptr<JobMetrics> type
    job_type = it->second->job_type_;
    return Status::OK();
}


Status MetadataStore::SetJobTypeByJobId(int64 job_id, string job_type) {
  std::shared_ptr<JobMetrics> jobMetrics;
  TF_RETURN_IF_ERROR(GetJobMetrics(job_id, jobMetrics));
  jobMetrics->job_type_ = job_type;
  return Status::OK();
}

Status MetadataStore::GetJobTypeByJobId(int64 job_id, string& job_type) {
  std::shared_ptr<JobMetrics> jobMetrics;
  TF_RETURN_IF_ERROR(GetJobMetrics(job_id, jobMetrics));
  job_type = jobMetrics->job_type_;
  return Status::OK();
}

Status MetadataStore::SetJobIsScaling(int64 job_id) {
  std::shared_ptr<JobMetrics> jobMetrics;
  TF_RETURN_IF_ERROR(GetJobMetrics(job_id, jobMetrics));
  jobMetrics->is_scaling_ = true;
  return Status::OK();
}

Status MetadataStore::UnsetJobIsScaling(int64 job_id) {
  std::shared_ptr<JobMetrics> jobMetrics;
  TF_RETURN_IF_ERROR(GetJobMetrics(job_id, jobMetrics));
  jobMetrics->is_scaling_ = false;
  return Status::OK();
}

Status MetadataStore::IsJobScaling(int64 job_id, bool& is_scaling) {
  std::shared_ptr<JobMetrics> jobMetrics;
  TF_RETURN_IF_ERROR(GetJobMetrics(job_id, jobMetrics));
  is_scaling = jobMetrics->is_scaling_;
  return Status::OK();
}

Status MetadataStore::GetJobScalingState(int64 job_id, JobScalingState& scaling_state) {
  std::shared_ptr<JobMetrics> jobMetrics;
  TF_RETURN_IF_ERROR(GetJobMetrics(job_id, jobMetrics));
  scaling_state = jobMetrics->scaling_state_;
  return Status::OK();
}

Status MetadataStore::SetJobScalingState(int64 job_id, JobScalingState scaling_state) {
  std::shared_ptr<JobMetrics> jobMetrics;
  TF_RETURN_IF_ERROR(GetJobMetrics(job_id, jobMetrics));
  jobMetrics->scaling_state_ = scaling_state;
  return Status::OK();
}

Status MetadataStore::GetJobStateInitialWorkerCount(int64 job_id, int64_t& worker_count) {
  std::shared_ptr<JobMetrics> jobMetrics;
  TF_RETURN_IF_ERROR(GetJobMetrics(job_id, jobMetrics));
  worker_count = jobMetrics->state_initial_worker_count_;
  return Status::OK();
}

Status MetadataStore::SetJobStateInitialWorkerCount(int64 job_id, int64_t worker_count) {
  std::shared_ptr<JobMetrics> jobMetrics;
  TF_RETURN_IF_ERROR(GetJobMetrics(job_id, jobMetrics));
  jobMetrics->state_initial_worker_count_ = worker_count;
  return Status::OK();
}

Status MetadataStore::GetLastPerformance(int64 job_id,
                                         Performance& last_performance) {
  std::shared_ptr<JobMetrics> jobMetrics;
  TF_RETURN_IF_ERROR(GetJobMetrics(job_id, jobMetrics));
  last_performance = jobMetrics->last_performance_;
  return Status::OK();
}


Status MetadataStore::SetLastPerformance(int64 job_id,
                                         Performance last_performance) {
  std::shared_ptr<JobMetrics> jobMetrics;
  TF_RETURN_IF_ERROR(GetJobMetrics(job_id, jobMetrics));
  jobMetrics->last_performance_ = last_performance;
  return Status::OK();
}

bool MetadataStore::JobSeenBefore(uint64 fingerprint) {
  return fingerprint_key_metadata_.contains(fingerprint);
}

Status MetadataStore::GetWorkerUpdateCounter(int64 job_id,
  uint64& heartbeat_counter) {
  std::shared_ptr<JobMetrics> jobMetrics;
  TF_RETURN_IF_ERROR(GetJobMetrics(job_id, jobMetrics));
  heartbeat_counter = jobMetrics->input_pipeline_metrics_->update_counter_;
  return Status::OK();
}

Status MetadataStore::GetNumberOfProducedElements(int64 job_id,
  uint64& element_count) {
  std::shared_ptr<NodeMetrics> last_tf_node_metrics;
  Status s = GetLastTFNodeMetrics(job_id, last_tf_node_metrics);

  // It might be that the metrics have been created but the first heartbeat
  // didn't arrive. In this case, s will not be OK
  element_count = 0;
  if (s.ok()) {
    for (auto& e : last_tf_node_metrics->metrics_) {
      element_count += e.second->num_elements();
    }
  }
  // Return OK if s.ok or name of last TF node is still unknown
  return (s.ok() || errors::IsUnavailable(s)) ? Status::OK() : s;
}

Status MetadataStore::GetSameScaleCounter(int64 job_id, uint64& counter) {
  std::shared_ptr<JobMetrics> jobMetrics;
  TF_RETURN_IF_ERROR(GetJobMetrics(job_id, jobMetrics));
  counter = jobMetrics->same_scale_counter_;
  return Status::OK();
}

Status MetadataStore::IncrementSameScaleCounter(int64 job_id, uint64& counter) {
  std::shared_ptr<JobMetrics> jobMetrics;
  TF_RETURN_IF_ERROR(GetJobMetrics(job_id, jobMetrics));
  jobMetrics->same_scale_counter_++;
  counter = jobMetrics->same_scale_counter_;
  return Status::OK();
}

Status MetadataStore::ResetSameScaleCounter(int64 job_id) {
  std::shared_ptr<JobMetrics> jobMetrics;
  TF_RETURN_IF_ERROR(GetJobMetrics(job_id, jobMetrics));
  jobMetrics->same_scale_counter_ = 0;
  return Status::OK();
}

Status MetadataStore::SetJobTargetRemoteWorkerCount(int64 job_id, int64 target_remote_worker_count) {
  std::shared_ptr<JobMetrics> jobMetrics;
  TF_RETURN_IF_ERROR(GetJobMetrics(job_id, jobMetrics));
  jobMetrics->target_remote_worker_count_ = target_remote_worker_count;
  return Status::OK();
}

Status MetadataStore::SetJobTargetLocalWorkerCount(int64 job_id, int64 target_local_worker_count) {
  std::shared_ptr<JobMetrics> jobMetrics;
  TF_RETURN_IF_ERROR(GetJobMetrics(job_id, jobMetrics));
  jobMetrics->target_local_worker_count_ = target_local_worker_count;
  return Status::OK();
}

Status MetadataStore::SetJobTargetWorkerCount(int64 job_id, int64 target_worker_count) {
  std::shared_ptr<JobMetrics> jobMetrics;
  TF_RETURN_IF_ERROR(GetJobMetrics(job_id, jobMetrics));
  jobMetrics->target_worker_count_ = target_worker_count;
  return Status::OK();
}

Status MetadataStore::GetJobTargetWorkerCount(int64 job_id, int64& target_worker_count) {
  std::shared_ptr<JobMetrics> jobMetrics;
  TF_RETURN_IF_ERROR(GetJobMetrics(job_id, jobMetrics));
  target_worker_count = jobMetrics->target_worker_count_;
  return Status::OK();
}

Status MetadataStore::GetJobTargetWorkerCount(int64 job_id,
                                              int64& target_remote_worker_count,
                                              int64& target_local_worker_count) {
  std::shared_ptr<JobMetrics> jobMetrics;
  TF_RETURN_IF_ERROR(GetJobMetrics(job_id, jobMetrics));
  target_remote_worker_count = jobMetrics->target_remote_worker_count_;
  target_local_worker_count = jobMetrics->target_local_worker_count_;
  return Status::OK();
}

Status MetadataStore::DumpJobMetricsToFile(int64 job_id, const std::string& path){
  std::shared_ptr<JobMetrics> jobMetrics;
  TF_RETURN_IF_ERROR(GetJobMetrics(job_id, jobMetrics));
  jobMetrics->DumpToFile(path);
  return Status::OK();
}

Status MetadataStore::AppendJobMetricsDumps(Env* env, const std::string& path) {
  for(auto pair: job_metadata_ ){
    std::string fname = path + "/metrics_updates_job_" + std::to_string(pair.first) + ".json";
    std::stringstream ss;

    Status s = env->FileExists(fname);
    if(!s.ok()){ // Initialize file if not present.
      if (!errors::IsNotFound(s)) {
        return s;
      }
      ss << "[" << std::endl;
    } else {
      ss << ", " << std::endl;
    }

    pair.second->DumpToStream(ss);
    // Append to file
    std::ofstream fstream;
    fstream.open(
        fname,
        std::ofstream::out | std::ofstream::app);
    fstream << ss.rdbuf();
    fstream.close();
  }
  return Status::OK();
}

/*
Status MetadataStore::TransferModelMetricsToNewJob(std::string dataset_key, int64 job_id){
  std::shared_ptr<JobMetrics> old_job_metrics;
  TF_RETURN_IF_ERROR(GetJobMetrics(job_id, old_job_metrics));

  std::shared_ptr<JobMetrics> job_metrics;
  TF_RETURN_IF_ERROR(GetJobMetricsByDatasetKey(dataset_key, job_metrics));

  job_metrics->model_metrics_ = old_job_metrics->model_metrics_;
};
*/

void TerminateJobMetricsAppendDumps(int64 job_id, const std::string& path){
  std::string fname = path + "/metrics_updates_job_" + std::to_string(job_id) + ".json";
  std::stringstream ss;

  ss << " ]" << std::endl;

  std::ofstream fstream;
  fstream.open(
      fname,
      std::ofstream::out | std::ofstream::app);
  fstream << ss.rdbuf();
  fstream.close();
}



} // namespace easl
} // namespace data
} // namespace tensorflow
