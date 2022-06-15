//
// Created by Muyu Li on 15.06.22.
//

#include "append_nodes_after_dsdo.h"

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

  NodeDef* getNodeDefFromName(GraphDef* graph, std::string name) {
    int idx = tensorflow::grappler::graph_utils::FindGraphNodeWithName(name, *graph);
    if (idx == -1) {
      VLOG(0) << "Node: " << name << " not found in graph";
      return NULL;
    }
    return graph->mutable_node(idx);
  }

  std::string getNodeRepr(NodeDef* sink) {
    return "[" + sink->name() + ", " + sink->op() + "]";
  }

  void BFSFromSink(GraphDef* graph, NodeDef* sink, absl::flat_hash_set<const NodeDef*>& visited) {
    VLOG(0) << "AppendNodesAfter::BFSFromSink from 'sink' node: " << getNodeRepr(sink);

    std::queue<const NodeDef*> q;
    visited.insert(sink);
    q.push(sink);

    while (!q.empty()) {
      const NodeDef* current_node = q.front();
      q.pop();

      for (int i = 0; i < current_node->input_size(); i++) {
        NodeDef* next_node = getNodeDefFromName(graph, current_node->input(i));
        VLOG(0) << "EDGE: " << getNodeRepr(current_node)
          << " -> " << getNodeRepr(next_node);

        if (!visited.contains(next_node)) {
          visited.insert(next_node);
          q.push(next_node);
        }
      }
    }
  }

  void BFSWholeGraph(GraphDef* graph) {
    VLOG(0) << "AppendNodesAfter::BFSWholeGraph";


    absl::flat_hash_set<const NodeDef*> visited;

    for (const auto& node: output->node()) {
      std::string node_name = node.name();
      NodeDef* node = getNodeDefFromName(graph, node_name);
      if (!visited.contains(node)) {
        BFSFromSink(graph, node, visited);
      }
    }
  }

}


Status AppendNodesAfterDSDO::OptimizeAndCollectStats(
        Cluster* *cluster, const GrapplerItem& item,
        GraphDef *output, OptimizationStats *stats) {

  VLOG(0) << "In AppendNodesAfterDSDO Optimizer";
  BFSWholeGraph(output);
  *output = item.graph;

  return Status::OK();

}

REGISTER_GRAPH_OPTIMIZER_AS(AppendNodesAfterDSDO, "append_nodes_after_dsdo");

} // easl
} // grappler
} // tensorflow