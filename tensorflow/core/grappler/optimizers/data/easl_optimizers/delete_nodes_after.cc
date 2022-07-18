//
// Created by Muyu Li on 09.06.22.
//

#include "delete_nodes_after.h"
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
    constexpr char kSplitNodeIndex[] = "split_node_index";
}

void DeleteNodesAfter::BFSGraph(NodeDef* sink_node,
                                GraphDef* output) {
  VLOG(0) << "(DelteNodesAfter::BFSGraph) start from sink node";
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

  VLOG(0) << "(DelteNodesAfter::BFSGraph) ends";
}

Status DeleteNodesAfter::ApplyOptimization(MutableGraphView &graph,
                                    NodeDef* sink_node,
                                    GraphDef* output) {
  VLOG(0) << "(DeleteNodesAfter::ApplyOptimization) start";

  int64 split_node_index = config_.parameter_map()
          .at(kSplitNodeIndex)
          .i();

  // iterate throught the graph from sink node
  absl::flat_hash_set<std::string> visited;
  NodeDef* current_node = sink_node;
  NodeDef* prev_node;
  int64 cur_pos_from_back = 0; // sink node position

  if (split_node_index <= 0) {
    VLOG(0) << "split node index equal to zero, no need to split";
    return Status::OK();
  }

  absl::flat_hash_set<std::string> nodes_to_delete;

  // to simplify, we only consider splitting the "tail" of the graph
  /*
   * A -> B \
   * C -> D -> E -> F -> G -> H -> sink
   *
   * we iterate from sink to Node "E"
   */
  for (; cur_pos_from_back < split_node_index;) {
    int non_const_prefix_count = 0;
    NodeDef* next_node = NULL; // TODO: handle this edge case
    for (int i = 0; i < current_node->input_size(); i++) {
      int idx = graph_utils::FindGraphNodeWithName(current_node->input(i), *output);
      NodeDef* tmp_node = output->mutable_node(idx);
      if (tmp_node->op() != "Const") {
        next_node = tmp_node;
        non_const_prefix_count++;
        if (non_const_prefix_count > 1) {
          // we've seen another non-const prefix before
          break;
        }
      }
    }

    if (non_const_prefix_count > 1) {
      VLOG(0) << "(DeleteNodesAfter::ApplyOptimization) see multiple non-const nodes from ["
        << current_node->name() << ", " << current_node->op() << "]";
      break;
    }

    if (current_node->op() == "SplitMarkerDataset") {
      VLOG(0) << "This node is SplitMarkerDataset";
      cur_pos_from_back++;
      if (cur_pos_from_back >= split_node_index) {
        break;
      }
    }
    prev_node = current_node;
    if (prev_node->name() != "Sink") {
      VLOG(0) << "Going to delete Node: " << prev_node->name();
      nodes_to_delete.insert(prev_node->name());
    }
    current_node = next_node;
  }

  Status s = graph.DeleteNodes(nodes_to_delete);
  if (!s.ok()) {
    VLOG(0) << "Delete Nodes failed: " << s.error_message();
  }

  if (cur_pos_from_back < split_node_index) {
    VLOG(0) << "(DeleteNodesAfter::ApplyOptimization): Chain not long enough "
      << "cur_pos_from_back: " << cur_pos_from_back
      << "; split_node_index: " << split_node_index;
  }
  // current_node is the right node to split

  // current_node -> prev_node -> ... -> sink

  // whether this one is needed
  sink_node->mutable_input()->Clear();
  sink_node->add_input(current_node->name());

  return Status::OK();
}

Status DeleteNodesAfter::OptimizeAndCollectStats(Cluster* cluster,
                               const GrapplerItem& item,
                               GraphDef* output,
                               OptimizationStats* stats) {
  *output = item.graph;
  MutableGraphView graph(output);

  // Get the sink node
  NodeDef* sink_node;
  TF_RETURN_IF_ERROR(graph_utils::GetFetchNode(graph, item, &sink_node));

  return ApplyOptimization(graph, sink_node, output);
}

REGISTER_GRAPH_OPTIMIZER_AS(DeleteNodesAfter, "delete_nodes_after");

} // easl
} // grappler
} // easl

