//
// Created by aymond on 16.07.21.
//

#include "tensorflow/core/grappler/optimizers/data/easl_optimizers/add_put_op_at_marker.h"

#include <queue>
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {
namespace easl {

namespace {
// Define constants here
constexpr char kPutOpDataset[] = "ServiceCachePutDataset";
constexpr char kOutputShapes[] = "output_shapes";
constexpr char kOutputTypes[] = "output_types";
// constexpr char kTargetNode[] = "ModelDataset";
constexpr char kTargetNode[] = "MarkerDataset";
constexpr char kSourceCache[] = "source_cache";
constexpr char kMarkerType[] = "marker_type";



} // namespace

NodeDef AddPutOpAtMarker::CreatePutOpNode(MutableGraphView* graph, NodeDef* input) {
  NodeDef put_op_node;

  // Give a unique name to the op
  graph_utils::SetUniqueGraphNodeName("put_op_dataset",
                                      graph->graph(), &put_op_node);

  // Set the node's operation and inputs.
  put_op_node.set_op(kPutOpDataset);
  put_op_node.add_input(input->name());

  NodeDef* location_node = graph_utils::AddScalarConstNode<StringPiece>(
      config_.parameter_map().at("path").placeholder(), graph);
  put_op_node.add_input(location_node->name());

  NodeDef* cache_format_node = graph_utils::AddScalarConstNode<int32>(
      config_.parameter_map().at("cache_format").i(), graph);
  put_op_node.add_input(cache_format_node->name());

  NodeDef* cache_compression = graph_utils::AddScalarConstNode<int32>(
      config_.parameter_map().at("cache_compression").i(), graph);
  put_op_node.add_input(cache_compression->name());

  NodeDef* parallelism_node = graph_utils::AddScalarConstNode<int32>(
      config_.parameter_map().at("cache_ops_parallelism").i(), graph);
  put_op_node.add_input(parallelism_node->name());

  // Copy over the relevant attributes from root of the prefix
  // TODO cleanup dirty hack: output_types can also be Toutput_types.
  VLOG(0) << "before copying attribures";
  graph_utils::CopyAttribute(kOutputShapes, *input, &put_op_node);
  auto it = input->attr().find(kOutputTypes);
  if (it != input->attr().end()){
    (*put_op_node.mutable_attr())[kOutputTypes] = it->second;
  } else {
    it = input->attr().find("Toutput_types");
    (*put_op_node.mutable_attr())[kOutputTypes] = it->second;
  }

  /*
      for (auto key : {kOutputShapes, kOutputTypes}){
        VLOG(0) << "before copying attribures";
        VLOG(0) << "copying from " << input->name();
        graph_utils::CopyAttribute(key, *input, &put_op_node);

        *to_node->mutable_attr())[attribute_name] = from.attr().at(attribute_name);*/
  VLOG(3) << "after copying attribures";

  return put_op_node;
}

Status AddPutOpAtMarker::ApplyOptimization(MutableGraphView &graph, NodeDef *sink_node,
                                   GraphDef *output) {
  VLOG(0) << "In AddPutOpAtMarker optimizer";

  // Define a filtering function which identifies target node
  std::string marker_type = config_.parameter_map().at(kMarkerType).placeholder();
  auto is_target_node = [marker_type](const NodeDef* node) -> bool {
    return node->op() == kTargetNode && node->attr().at(kMarkerType).s() == marker_type;
  };

  // Find the first target op by applying BFS
  absl::flat_hash_set<std::string> visited;
  std::queue<NodeDef*> bfs_queue;
  bfs_queue.push(sink_node);
  NodeDef* target = nullptr;

  while (!bfs_queue.empty()) {
    NodeDef* current_node = bfs_queue.front();
    bfs_queue.pop();
    visited.insert(current_node->name());

    VLOG(1) << "@ current_node: " << current_node->op();

    // TODO(DanGraur): Add logic here to skip certain nodes (e.g. control)

    // Check to see if this node is a target op
    if (is_target_node(current_node)) {
      target = current_node;
      break;
    }

    // Iterate throught the neighbors
    for (int i = 0; i < current_node->input_size(); ++i) {
      if (!visited.contains(current_node->input(i))) {
        int idx = graph_utils::FindGraphNodeWithName(current_node->input(i),
                                                     *output);
        NodeDef* neighbor_node = output->mutable_node(idx);
        bfs_queue.push(neighbor_node);
      }
    }
  }

  // We return if we found no target op
  if (!target) {
    VLOG(0) << "Could not find target node " << kTargetNode
            << " with marker_type " << marker_type;
    return Status::OK();
  }

  // Find the input of the target node
  NodeDef* target_input = graph_utils::GetInputNode(*target, graph);
  if(!target_input){
    return errors::Unknown("The target has no inputs.");
  }

  // Create the put_op_node op node, then add it to the graph
  NodeDef put_op_node = CreatePutOpNode(&graph, target_input);

  // Copy over the relevant attributes
  (*target->mutable_input())[0] = put_op_node.name();
  // graph_utils::CopyAttribute(kOutputTypes, put_op_node, target);

  // Add the node to the graph
  graph.AddNode(std::move(put_op_node));

  return Status::OK();
}

Status AddPutOpAtMarker::OptimizeAndCollectStats(Cluster* cluster,
                                         const GrapplerItem& item,
                                         GraphDef* output,
                                         OptimizationStats* stats) {
  // Initializations
  *output = item.graph;
  MutableGraphView graph(output);

  // Get the sink node
  NodeDef* sink_node;
  TF_RETURN_IF_ERROR(graph_utils::GetFetchNode(graph, item, &sink_node));

  // Apply the transformation
  return ApplyOptimization(graph, sink_node, output);
}


REGISTER_GRAPH_OPTIMIZER_AS(AddPutOpAtMarker, "add_put_op_at_marker");

}  // namespace easl
}  // namespace grappler
}  // namespace tensorflow
