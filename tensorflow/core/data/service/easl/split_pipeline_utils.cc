//
// Created by Muyu Li on 09.06.22.
//

#include "split_pipeline_utils.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/grappler/optimizers/data/easl_optimizers/delete_nodes_after.h"

namespace tensorflow {
namespace data {
namespace service {
namespace easl{
namespace split_utils {

void BFSGraph(NodeDef* sink_node,
              GraphDef* output,
              std::string prefix) {
  VLOG(0) << "(SplitPipelineUtils::BFSGraph)::" << prefix << " start from sink node";
  std::queue<NodeDef*> q;
  absl::flat_hash_set<NodeDef*> visited;
  q.push(sink_node);
  visited.insert(sink_node);

  while (!q.empty()) {
    NodeDef* current_node = q.front();
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

  VLOG(0) << "(SplitPipelineUtils::BFSGraph) ends";
}

/* split_node_index: split after this node
A -> B -> C -> D, index: 2
A -> B | C -> D
Same on client side
*/
Status DeleteAfterNode(const DatasetDef& dataset,
                         const experimental::DispatcherConfig& dispatcher_config,
                         int64 split_node_index,
                         DatasetDef& updated_dataset) {
  VLOG(0) << "(SplitPipelineUtils::DeleteAfterNode): truncating after index " << split_node_index
    << " (from the end).";
  updated_dataset = dataset;

  tensorflow::grappler::easl::DeleteNodesAfter optimizer;
  tensorflow::RewriterConfig_CustomGraphOptimizer config;

  (*(config.mutable_parameter_map()))["split_node_index"].set_i(
          split_node_index);

  optimizer.Init(&config);

  GraphDef* graph_def = updated_dataset.mutable_graph();
  std::string output_node;

  for (const auto& node : graph_def->node()) {
    if (node.op() == "_Retval") {
      output_node = node.input(0);
    }
  }

  NodeDef* sink = graph_def->mutable_node()->Add();
  tensorflow::grappler::graph_utils::SetUniqueGraphNodeName("Sink", graph_def,
                                                            sink);
  sink->set_op("Identity");
  sink->add_input(output_node);
  (*sink->mutable_attr())["T"].set_type(DT_VARIANT);

  tensorflow::grappler::MutableGraphView graph(graph_def);
  optimizer.ApplyOptimization(graph, sink, graph_def);

  VLOG(0) << "(SplitPipelineUtils::DeleteAfterNode): function ended";

  BFSGraph(sink, graph_def, "DeleteAfterNode");
  return Status::OK();
}


} // split_utils
} // easl
} // service
} // data
} // tensorflow
