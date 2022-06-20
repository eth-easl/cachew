//
// Created by Muyu Li on 23.05.22.
//

#include <queue>

#include "tensorflow/core/kernels/data/experimental/split/split_second_half.h"

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/data/dataset_utils.h"

#include "tensorflow/core/data/service/easl/split_pipeline_state.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace split_second_half {


namespace {
    constexpr char kSplitSecondHalf[] = "SplitSecondHalf";
    constexpr char kSplitNodeIndex[] = "split_node_index";
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

SplitSecondHalfOp::SplitSecondHalfOp(OpKernelConstruction* ctx): UnaryDatasetOpKernel(ctx) {}

NodeDef* getNodeDefFromName(std::string name, GraphDef* graph) {
  int idx = tensorflow::grappler::graph_utils::FindGraphNodeWithName(name, *graph);
  if (idx == -1) return NULL;
  return graph->mutable_node(idx);
}

Status BFSGraph(NodeDef* sink_node,
                         GraphDef* output,
                         std::string prefix) {

  VLOG(0) << "(SplitSecondHalfOp::MakeDataset::BFSGraph) start with prefix: "
          << prefix << "; with a sink node: " << sink_node->name();

  // start from sink node
//  NodeDef* sink_node = getNodeDefFromName(sink_node_name, output);
  VLOG(0) << sink_node->name() << " " << sink_node->op();
//  std::shared_ptr<NodeDef> sink_node = std::make_shared<NodeDef>();
//  bool sink_node_found = false;
//  for (const auto& node : output->node()) {
//    if (node.op() == sink_node_name || node.name() == sink_node_name) {
//      // TODO: what is the type here? Is that needed to use shared ptr here?
//      sink_node = &node;
//      sink_node_found = true;
//    }
//  }
//  if (!sink_node_found) {
//    VLOG(0) << "Sink Node Name: " << sink_node_name << " not found, print all nodes";
//    for (const auto& node : output->node()) {
//      VLOG(0) << "--- Node: " << node.op() << " " << node.name();
//    }
//    return Status::OK();
//  }

  std::queue<const NodeDef*> q;
  absl::flat_hash_set<const NodeDef*> visited;
  q.push(sink_node);
  visited.insert(sink_node);

  while (!q.empty()) {
    const NodeDef* current_node = q.front();
    q.pop();

    for (int i = 0; i < current_node->input_size(); ++i) {
      int idx = tensorflow::grappler::graph_utils::FindGraphNodeWithName(current_node->input(i), *output);
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

  VLOG(0) << "(SplitSecondHalfOp::MakeDataset::BFSGraph) end with prefix: " << prefix;
}

NodeDef* addNodeToGraph(NodeDef* node, tensorflow::grappler::MutableGraphView& graph_view) {
  // type probably wrong
  VLOG(0) << "Add Node: [" << node->name() << ", " << node->op() << "] to graph";
  return tensorflow::grappler::graph_utils::AddNode(
          node->name(),
          node->op(),
          {node->mutable_input()->begin(), node->mutable_input()->end()},
          {node->mutable_attr()->begin(), node->mutable_attr()->end()},
          &graph_view);
}

NodeDef* addSinkNodeToGraph(std::string output_node, GraphDef* graph_def) {
  NodeDef* sink = graph_def->mutable_node()->Add();
  tensorflow::grappler::graph_utils::SetUniqueGraphNodeName("Sink", graph_def,
                                                            sink);
  sink->set_op("Identity");
  sink->add_input(output_node);
  (*sink->mutable_attr())["T"].set_type(DT_VARIANT);

  return sink;
}

NodeDef* rewrite(GraphDef* dsdo_graph, // data_service_dataset_op
             NodeDef* dsdo_sink_node,
             GraphDef* second_half_graph,
             NodeDef* second_half_sink_node,
             int64 split_node_index) {
  // the split node index is coming from dispatcher side, so it's guaranteed to be feasible

//  NodeDef* dsdo_sink_node = getNodeDefFromName(dsdo_sink_node_name, dsdo_graph);
//  NodeDef* second_half_sink_node = getNodeDefFromName(second_half_sink_node_name, second_half_graph);

  tensorflow::grappler::MutableGraphView dsdo_graph_view(dsdo_graph);

  // sh: second_half; dsdo: data_service_dataset_op
  NodeDef* current_node_sh = second_half_sink_node, *prev_node_sh;
  NodeDef* current_node_dsdo = addNodeToGraph(current_node_sh, dsdo_graph_view);
  NodeDef* res = current_node_dsdo;
  int64 cur_pos_from_back = 0; // sink node position

  for (; cur_pos_from_back < split_node_index; cur_pos_from_back++) {
    NodeDef* next_node_sh = NULL; // TODO: handle this edge case
    for (int i = 0; i < current_node_sh->input_size(); i++) {
      NodeDef* tmp_node_sh = getNodeDefFromName(current_node_sh->input(i), second_half_graph);
      // Add this node to the dsdo_graph
      NodeDef* added_node_dsdo = getNodeDefFromName(tmp_node_sh->name(), dsdo_graph);
      if (added_node_dsdo == NULL) {
        added_node_dsdo = addNodeToGraph(tmp_node_sh, dsdo_graph_view);
      }
      else {
        VLOG(0) << "Found Node: [" << added_node_dsdo->name()
          << ", " << added_node_dsdo->op() << "in DSDO Graph";
      }
      if (!(tmp_node_sh->op() != "Const" && cur_pos_from_back == split_node_index - 1)) {
        current_node_dsdo->mutable_input()->Add(std::string(added_node_dsdo->name()));
      }
      else {
        VLOG(0) << "Skipping node: [" << current_node_sh->name() << ", "
          << current_node_sh->op() << "]'s non_const upstream node: ["
          << tmp_node_sh->name() << ", " << tmp_node_sh->op()
          << "]";
      }

      if (tmp_node_sh->op() != "Const") {
        // guaranteed to be only happening once
        next_node_sh = tmp_node_sh;
        if (cur_pos_from_back < split_node_index - 1) {
          current_node_dsdo = added_node_dsdo;
          /*
           * A -> B -> C -> D, split_node_index = 2
           * current_node_dsdo is going to stop at B, but we need to stop at C
           */
        }
      }
    }
    current_node_sh = next_node_sh;
  }

  // replace upstream of current_node_dsdo with dsdo_sink_node
  current_node_dsdo->mutable_input()->Add(std::string(dsdo_sink_node->name()));
  return res;
}

// refer to add_put_op_at_marker.cc
void SplitSecondHalfOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input_first, DatasetBase** output) {
  VLOG(0) << "SplitSecondHalfOp::MakeDataset";

  DatasetBase* input_second;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(2), &input_second));
//  int64 split_node_index;
//  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kSplitNodeIndex, &split_node_index));
//
//  // create graph_def from tensors
  std::vector<std::pair<string, Tensor>> input_list_first, input_list_second;
  GraphDef graph_def_first, graph_def_second;
  std::string output_node_first, output_node_second;
//
//  AsGraphDefForRewrite(ctx, input_first, &input_list_first,
//                                          &graph_def_first, &output_node_first);
  Status s = AsGraphDefForRewrite(ctx, input_second, &input_list_second,
                                          &graph_def_second, &output_node_second);
//
  if (!s.ok()) {
    VLOG(0) << "AsGraphDefForRewrite Fails: " << s.ToString();
  }
  VLOG(0) << "output_node_second: " << output_node_second;

  tensorflow::data::service::easl::split_state::SplitOriginalGraph::AddJob("job_nameAAA", graph_def_second);

  VLOG(0) << "Create Second Graph Node";

//  NodeDef *sink_node_dsdo = addSinkNodeToGraph(output_node_first, &graph_def_first);
//  NodeDef *sink_node_second_half = addSinkNodeToGraph(output_node_second, &graph_def_second);

//  NodeDef *sink_node_dsdo = getNodeDefFromName(output_node_first, &graph_def_first);
//  NodeDef *sink_node_second_half = getNodeDefFromName(output_node_second, &graph_def_second);
//  BFSGraph(sink_node_second_half, &graph_def_second, "Second Half");
//  BFSGraph(sink_node_dsdo, &graph_def_first, "First Half");

//  NodeDef* new_sink_node = rewrite(&graph_def_first, sink_node_dsdo,
//          &graph_def_second, sink_node_second_half,
//          split_node_index);

//  BFSGraph(new_sink_node, &graph_def_first, "First Half After Rewrite");

  // TODO: run graph_def_first to get the dataset
  // See Rewrite Dataset in rewrite_utils.cc
//  FunctionLibraryRuntime* flr = nullptr;
//  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr = nullptr;
//  std::unique_ptr<FunctionLibraryDefinition> lib_def = nullptr;
//  ctx->function_library()->Clone(&lib_def, &pflr, &flr, true);
//
//  AddToFunctionLibrary(lib_def.get(), graph_def_first.library());
//
//  Graph graph(OpRegistry::Global());
//  ImportGraphDef({}, graph_def_first, &graph, nullptr);
//  std::vector<Tensor> outputs;
//  GraphRunner graph_runner(flr->device());
//
//  DatasetBase** rewritten_input;
//
//  graph_runner.Run(&graph, flr, input_list_first, {output_node_first}, &outputs);
//  GetDatasetFromVariantTensor(outputs[0], rewritten_input);
//  (*rewritten_input)->Ref();
//
//  *output = *rewritten_input;

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
