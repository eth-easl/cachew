//
// Created by Muyu Li on 27.06.22.
//

#include "split_marker_op.h"

#include "tensorflow/core/platform/tstring.h"
#include "absl/memory/memory.h"
#include "tensorflow/core/data/name_utils.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace easl{

/* static */ constexpr const char* const SplitMarkerOp::kDatasetType;

class SplitMarkerOp::Dataset : public DatasetBase {
public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input);

    ~Dataset() override;

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
            const string& prefix) const override;

    const DataTypeVector& output_dtypes() const override;

    const std::vector<PartialTensorShape>& output_shapes() const override;

    string DebugString() const override;

    int64_t CardinalityInternal() const override;

    Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override;

    Status CheckExternalState() const override;

protected:

    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override;

private:
    class Iterator;

    const DatasetBase* const input_;
};

class SplitMarkerOp::Dataset::Iterator : public DatasetIterator<Dataset> {
public:
    explicit Iterator(const Params& params);

    Status Initialize(IteratorContext* ctx) override;

    Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override;

protected:
    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override;

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override;

    std::shared_ptr<model::Node> CreateNode(IteratorContext* ctx,
                                            model::Node::Args args) const override;

private:
    // mutex mu_;
    std::unique_ptr<IteratorBase> input_impl_;
};

// -----------------------------------------------------------------------------
// DatasetOp
// -----------------------------------------------------------------------------

SplitMarkerOp::SplitMarkerOp(OpKernelConstruction* ctx)
        : UnaryDatasetOpKernel(ctx) {}

void SplitMarkerOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                           DatasetBase** output) {

  *output = new Dataset(ctx, input);
}

// -----------------------------------------------------------------------------
// Dataset
// -----------------------------------------------------------------------------

SplitMarkerOp::Dataset::Dataset(
        OpKernelContext* ctx,
        const DatasetBase* input)
        : DatasetBase(DatasetContext(ctx)), input_(input){
  input_->Ref();
}

SplitMarkerOp::Dataset::~Dataset() { input_->Unref(); }

std::unique_ptr<IteratorBase>
SplitMarkerOp::Dataset::MakeIteratorInternal(const string& prefix) const {
  VLOG(3) << "EASL - prefix to SplitMarker op: " << prefix;
  return absl::make_unique<Iterator>(
          Iterator::Params{this, name_utils::IteratorPrefix(kDatasetType, prefix)});
}

const DataTypeVector& SplitMarkerOp::Dataset::output_dtypes() const {
  return input_->output_dtypes();
}

const std::vector<PartialTensorShape>&
SplitMarkerOp::Dataset::output_shapes() const {
  return input_->output_shapes();
}

string SplitMarkerOp::Dataset::DebugString() const {
  return name_utils::DatasetDebugString(kDatasetType);
}

int64_t SplitMarkerOp::Dataset::CardinalityInternal() const {
  return input_->Cardinality();
}

Status SplitMarkerOp::Dataset::InputDatasets(
        std::vector<const DatasetBase*>* inputs) const {
  inputs->push_back(input_);
  return Status::OK();
}

Status SplitMarkerOp::Dataset::CheckExternalState() const {
  return input_->CheckExternalState();
}

Status SplitMarkerOp::Dataset::AsGraphDefInternal(
        SerializationContext* ctx, DatasetGraphDefBuilder* b, Node** output) const {
  Node* input_graph_node = nullptr;
  TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

  return b->AddDataset(
          this,
          /*inputs=*/
          {input_graph_node},
          /*attr*/{},
          output);
}

// -----------------------------------------------------------------------------
// Iterator
// -----------------------------------------------------------------------------

SplitMarkerOp::Dataset::Iterator::Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {};

Status SplitMarkerOp::Dataset::Iterator::Initialize(IteratorContext* ctx) {
  return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
}

Status SplitMarkerOp::Dataset::Iterator::SaveInternal(
        SerializationContext* ctx, IteratorStateWriter* writer) {
  return errors::Unimplemented("Checkpointing is currently not supported.");
}

Status SplitMarkerOp::Dataset::Iterator::RestoreInternal(
        IteratorContext* ctx, IteratorStateReader* reader) {
  return errors::Unimplemented("Checkpointing is currently not supported.");
}

Status SplitMarkerOp::Dataset::Iterator::GetNextInternal(
        IteratorContext* ctx, std::vector<Tensor>* out_tensors,
        bool* end_of_sequence) {
  // mutex_lock l(mu_);
  return input_impl_->GetNext(ctx, out_tensors, end_of_sequence);
}

std::shared_ptr<model::Node>
SplitMarkerOp::Dataset::Iterator::CreateNode(IteratorContext* ctx,
                                        model::Node::Args args) const {
  return model::MakeKnownRatioNode(args, 1);
}

namespace {
    REGISTER_KERNEL_BUILDER(Name("SplitMarkerDataset").Device(DEVICE_CPU), SplitMarkerOp);
}  // namespace


} // namespace easl
} // namespace experimental
} // namespace data
} // namespace tensorflow