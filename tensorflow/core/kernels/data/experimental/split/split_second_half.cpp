//
// Created by Muyu Li on 23.05.22.
//

#include "tensorflow/core/kernels/data/experimental/split/split_second_half.h"

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace split_second_half {


namespace {
    constexpr char kSplitSecondHalf[] = "SplitSecondHalf";
    constexpr char kSplitNodeIndex[] = "split_node_index";
    constexpr char kDataServiceDataset[] = "DataServiceDatasetV3"
}


//SplitSecondHalfOp::Dataset::Dataset(OpKernelContext* ctx, const DatasetBase* input)
//  : DatasetBase(DatasetContext(ctx)), input_(input) {
//  input_->Ref();
//}
//
//SplitSecondHalfOp::Dataset::Dataset(DatasetContext::Params params, const DatasetBase* input)
//  : DatasetBase(DatasetContext(std::move(params))), input_(input) {
//  input_->Ref();
//}
//
//SplitSecondHalfOp::Dataset::~Dataset() {
//  input_->Unref();
//}
//
//const DataTypeVector& SplitSecondHalfOp::Dataset::output_dtypes() const {
//  static DataTypeVector* dtypes = new DataTypeVector({DT_INT64});
//  return *dtypes;
//}
//
//const std::vector<PartialTensorShape>& SplitSecondHalfOp::Dataset::output_shapes() const {
//  std::vector<PartialTensorShape>* shapes = new std::vector<PartialTensorShape>();
//  shapes->push_back(TensorShape());
//  return *shapes;
//}
//
//string SplitSecondHalfOp::Dataset::DebugString() const {
//  return name_utils::DatasetDebugString(SplitSecondHalfOp::kDatasetType);
//}
//
//Status SplitSecondHalfOp::Dataset::CheckExternalState() const {
//  return Status::OK();
//}
//
//std::unique_ptr<IteratorBase> SplitSecondHalfOp::Dataset::MakeIteratorInternal(const string& prefix) const {
//  return absl::make_unique<Iterator>(Iterator::Params{this, name_utils::IteratorPrefix(kSplitSecondHalf, prefix)});
//}

//Status SplitSecondHalfOp::Dataset::AsGraphDefInternal(SerializationContext* ctx,
//                                           DatasetGraphDefBuilder* b,
//                                           Node** output) const {
//  Node* input_graph_node = nullptr;
//  TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
//  TF_RETURN_IF_ERROR(b->AddDataset(this, {input_graph_node}, output));
//  return Status::OK();
//}

//void ApplyRewrites()

Status PrintChainOfGraph(std::string sink_node_name,
                         GraphDef* output,
                         std::string prefix) {
  VLOG(0) << "(SplitSecondHalfOp::PrintGraphChain) start with prefix: "
          << prefix << "; with a sink node: " << sink_node_name;

  // start from sink node
  NodeDef* current_node = NULL;
  for (const auto& node : graph_def->node()) {
    if (node.op() == sink_node_name) {
      // TODO: what is the type here? Is that needed to use shared ptr here?
      current_node = &node;
    }
  }
  if (!current_node) {
    VLOG(0) << "Sink Node Name: " << sink_node_name << " not found, print all nodes";
    for (const auto& node : graph_def->node()) {
      VLOG(0) << "--- Node: " << node.op() << " " << node.name();
    }
    return Status::OK();
  }

  while (current_node->input_size() == 1) {
    VLOG(0) << " <- (" << current_node->name() << ")";

    int idx = tensorflow::grappler::graph_utils::FindGraphNodeWithName(current_node->input(0), *output);
    current_node = output->mutable_node(idx);
  }

  VLOG(0) << "(PrintGraphChain) end ----";
}

void SplitSecondHalfOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input_first, DatasetBase** output) {
  DatasetBase* input_second;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(2), &input_second));

  int64 split_node_index;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kSplitNodeIndex, &split_node_index));

  // create graph_def from tensors
  std::vector<std::pair<string, Tensor>> input_list_first, input_list_second;
  GraphDef graph_def_first, graph_def_second;
  string output_node_first, output_node_second;

  TF_RETURN_IF_ERROR(AsGraphDefForRewrite(ctx, input_first, &input_list_first,
                                          &graph_def_first, &output_node_first));
  TF_RETURN_IF_ERROR(AsGraphDefForRewrite(ctx, input_second, &input_list_second,
                                          &graph_def_second, &output_node_second));

  PrintChainOfGraph(output_node_first, &graph_def_first, "First Half");
  PrintChainOfGraph(output_node_second, &graph_def_second, "Second Half");

  *output = input_first;
}

//Status SplitSecondHalfOp::Dataset::Iterator::GetNextInternal(
//        IteratorContext* ctx, std::vector<Tensor>* out_tensors, bool* end_of_sequence) {
//  out_tensors->clear();
//  int64 val = 10086;
//  out_tensors->push_back(Tensor(val));
//  return Status::OK();
//}

namespace {
    REGISTER_KERNEL_BUILDER(Name("SplitSecondHalf").Device(DEVICE_CPU), SplitSecondHalfOp);
}

} // split
} // experimental
} // data
} // tensorflow