//
// Created by Muyu Li on 23.05.22.
//

#include "tensorflow/core/kernels/data/experimental/split/split_second_half.h"

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/data/name_utils.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace split {

constexpr char kSplitSecondHalf[] = "SplitSecondHalf";

SplitSecondHalfOp::Dataset::Dataset(OpKernelContext* ctx, const DatasetBase* input)
  : DatasetBase(DatasetContext(ctx)), input_(input) {
  input_->Ref();
}

SplitSecondHalfOp::Dataset::Dataset(DatasetContext::Params params, const DatasetBase* input)
  : DatasetBase(DatasetContext(std::move(params))), input_(input) {
  input_->Ref();
}

SplitSecondHalfOp::Dataset::~Dataset() {
  input_->Unref();
}

const DataTypeVector& SplitSecondHalfOp::Dataset::output_dtypes() const {
  static DataTypeVector* dtypes = new DataTypeVector({DT_INT64});
  return *dtypes;
}

const std::vector<PartialTensorShape>& SplitSecondHalfOp::Dataset::output_shapes() const {
  std::vector<PartialTensorShape>* shapes = new std::vector<PartialTensorShape>();
  shapes->push_back(TensorShape());
  return *shapes;
}

string SplitSecondHalfOp::Dataset::DebugString() const {
  return name_utils::DatasetDebugString(SplitSecondHalfOp::kDatasetType);
}

Status SplitSecondHalfOp::Dataset::CheckExternalState() const {
  return Status::OK();
}

std::unique_ptr<IteratorBase> SplitSecondHalfOp::Dataset::MakeIteratorInternal(const string& prefix) const {
  return absl::make_unique<Iterator>(Iterator::Params{this, name_utils::IteratorPrefix(kSplitSecondHalf, prefix)});
}

Status SplitSecondHalfOp::Dataset::AsGraphDefInternal(SerializationContext* ctx,
                                           DatasetGraphDefBuilder* b,
                                           Node** output) const {
  Node* input_graph_node = nullptr;
  TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
  TF_RETURN_IF_ERROR(b->AddDataset(this, {input_graph_node}, output));
  return Status::OK();
}

void SplitSecondHalfOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input, DatasetBase** output) {
  *output = new Dataset(ctx, input);
}

Status SplitSecondHalfOp::Dataset::Iterator::GetNextInternal(
        IteratorContext* ctx, std::vector<Tensor>* out_tensors, bool* end_of_sequence) {
  out_tensors->clear();
  int64 val = 10086;
  out_tensors->push_back(Tensor(val));
  return Status::OK();
}

namespace {
    REGISTER_KERNEL_BUILDER(Name("SplitSecondHalf").Device(DEVICE_CPU), SplitSecondHalfOp);
}

} // split
} // experimental
} // data
} // tensorflow