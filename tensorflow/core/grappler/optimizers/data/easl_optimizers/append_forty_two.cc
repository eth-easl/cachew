
#include <queue>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/easl_optimizers/append_forty_two.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {
namespace easl {
namespace {
  // Define constants here
  constexpr char kFortyTwoDataset[] = "FortyTwoDataset";
  constexpr char kOutputShapes[] = "output_shapes";
  constexpr char kOutputTypes[] = "output_types";

  NodeDef CreateFortyTwoOpNode(MutableGraphView* graph, NodeDef* input) {
    NodeDef forty_two_node;

    // Give a unique name to our forty_two node and store it for later use
    graph_utils::SetUniqueGraphNodeName("forty_two_dataset",
        graph->graph(), &forty_two_node);

    // Set its operation and input.
    forty_two_node.set_op(kFortyTwoDataset);
    forty_two_node.add_input(input->name());

    // Add output_type and empty output_shape attributes
    (*forty_two_node.mutable_attr())[kOutputTypes].mutable_list()->add_type(
            tensorflow::DataType::DT_INT32);
    (*forty_two_node.mutable_attr())[kOutputShapes].mutable_list()->add_shape();

    return forty_two_node;
  }

}

Status AppendFortyTwo::OptimizeAndCollectStats(Cluster* cluster,
                                               const GrapplerItem& item,
                                               GraphDef* output,
                                               OptimizationStats* stats) {
  VLOG(0) << "In AppendFortyTwo optimizer";
  *output = item.graph;
  MutableGraphView graph(output);

  // Define a filtering function which identifies batch ops
  auto is_batch_op = [](const NodeDef* node) -> bool {
    // VLOG(1) << "The name of the node is " << node->op() << " and inp size is " << node->input_size();
    return node->op() == "BatchDataset" && node->input_size() == 2 
        || node->op() == "BatchDatasetV2" && node->input_size() == 3;
  };

  // Get the output of the graph
  NodeDef* sink_node;
  TF_RETURN_IF_ERROR(graph_utils::GetFetchNode(graph, item, &sink_node));

  // Find the first batch op by applying BFS
  absl::flat_hash_set<std::string> visited;
  std::queue<NodeDef*> bfs_queue;
  bfs_queue.push(sink_node);
  NodeDef* target = nullptr;

  while (!bfs_queue.empty()) {
    NodeDef* current_node = bfs_queue.front();
    bfs_queue.pop();
    visited.insert(current_node->name());

    // TODO(DanGraur): Add logic here to skip certain nodes (e.g. control)

    // Check to see if this node is a batch op
    if (is_batch_op(current_node)) {
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

  // We return if we found no batch op
  if (!target) {
    VLOG(1) << "Could not find target";
    return Status::OK();
  }
  // Find the input of the target node
  NodeDef* forty_two_input = graph_utils::GetInputNode(*target, graph);
  if(!forty_two_input){
    return errors::Unknown("The target has no inputs.");
  }
  
  // Create the forty_two op node, then add it to the graph
  NodeDef forty_two_node = CreateFortyTwoOpNode(&graph, forty_two_input);

  // Copy over the relevant attributes
  (*target->mutable_input())[0] = forty_two_node.name();
  graph_utils::CopyAttribute(kOutputTypes, forty_two_node, target);

  // Add the node to the graph
  graph.AddNode(std::move(forty_two_node));

  return Status::OK();
}

REGISTER_GRAPH_OPTIMIZER_AS(AppendFortyTwo, "append_forty_two");

}  // namespace easl
}  // namespace grappler
}  // namespace tensorflow
