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

Status BFSGraph(std::string sink_node_name,
                         GraphDef* output,
                         std::string prefix) {

  VLOG(0) << "(SplitSecondHalfOp::MakeDataset::BFSGraph) start with prefix: "
          << prefix << "; with a sink node: " << sink_node_name;

  // start from sink node
  NodeDef* sink_node = NULL;
  for (const auto& node : graph_def->node()) {
    if (node.op() == sink_node_name || node.name() == sink_node_name) {
      // TODO: what is the type here? Is that needed to use shared ptr here?
      sink_node = &node;
    }
  }
  if (!sink_node) {
    VLOG(0) << "Sink Node Name: " << sink_node_name << " not found, print all nodes";
    for (const auto& node : graph_def->node()) {
      VLOG(0) << "--- Node: " << node.op() << " " << node.name();
    }
    return Status::OK();
  }

  std::queue<NodeDef*> q;
  absl::flat_hash_set<NodeDef*> visited;
  q.push(sink_node);
  visited.insert(sink_node);

  while (!q.empty()) {
    NodeDef* current_node = q.front();
    q.pop();

    for (int i = 0; i < current_node->input_size(); ++i) {
      int idx = graph_utils::FindGraphNodeWithName(current_node->input(i), *output);
      NodeDef* next_node = output->mutable_node(idx);
      VLOG(0) << "EDGE: [" << current_node->name()
              << ", " << current_node->op() << "] -> ["
              << next_node->name() << ", " << next_node->op() << "]";

      if (!visited.contains(next_node)) {
        visited.insert(next_node);
        q.push(next_node);
      }
    }
  }

  VLOG(0) << "(SplitSecondHalfOp::MakeDataset::BFSGraph) end ----";
}

NodeDef* getNodeDefFromName(std::string name, GraphDef* graph) {
  int idx = tensorflow::grappler::graph_utils::FindGraphNodeWithName(name, *graph);
  return graph->mutable_node(idx);
}

NodeDef* addNodeToGraph(NodeDef* node, MutableGraphView& graph_view) {
  // type probably wrong
  VLOG(0) << "Add Node: [" << node->name() << ", " << node->op() << "] to graph";
  return tensorflow::grappler::graph_utils::AddNode(
          node->name(),
          node->op(),
          node->input(),
          node->attr(),
          &graph_view);
}

void rewrite(GraphDef* dsdo_graph, // data_service_dataset_op
             std::string dsdo_sink_node_name,
             GraphDef* second_half_graph,
             std::string second_half_sink_node_name,
             int64 split_node_index) {
  // the split node index is coming from dispatcher side, so it's guaranteed to be feasible

  NodeDef* dsdo_sink_node = getNodeDefFromName(dsdo_sink_node_name, dsdo_graph);
  NodeDef* second_half_sink_node = getNodeDefFromName(second_half_sink_node_name, second_half_graph);

  tensorflow::grappler::MutableGraphView dsdo_graph_view(dsdo_graph);

  NodeDef* current_node_sh = second_half_sink_node;
  NodeDef* prev_node;
  int64 cur_pos_from_back = 0; // sink node position

  for (; cur_pos_from_back < split_node_index; cur_pos_from_back++) {
    int non_const_prefix_count = 0;
    NodeDef* next_node_sh = NULL; // TODO: handle this edge case
    for (int i = 0; i < current_node_sh->input_size(); i++) {
      NodeDef* tmp_node_sh = getNodeDefFromName(current_node->input(i), second_half_graph);
      // Add this node to the dsdo_graph
      NodeDef* added_node_dsdo = addNodeToGraph(tmp_node, dsdo_graph_view);
      current_node_dsdo->mutable_input().insert(added_node_dsdo->name());
      if (tmp_node->op() != "Const") {
        next_node_sh = tmp_node_sh;
      }
    }

    if (non_const_prefix_count > 1) {
      VLOG(0) << "(DeleteNodesAfter::ApplyOptimization) see multiple non-const nodes from ["
              << current_node->name() << ", " << current_node->op() << "]";
      break;
    }

    prev_node = current_node;
    current_node_sh = next_node_sh;
  }

  // replace current_node_dsdo with dsdo_sink_node
  
  return;
}

// refer to add_put_op_at_marker.cc
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

  BFSGraph(output_node_first, &graph_def_first, "First Half");
  BFSGraph(output_node_second, &graph_def_second, "Second Half");

  rewrite(graph_def_first, output_node_first,
          graph_def_second, output_node_second,
          split_node_index);

  // TODO: run graph_def_first to get the dataset

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