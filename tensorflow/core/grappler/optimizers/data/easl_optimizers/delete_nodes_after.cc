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

void DeleteNodesAfter::PrintChainOfGraph(NodeDef* sink_node,
                       GraphDef* output,
                       int64 split_node_index) {
  VLOG(0) << "(DeleteNodesAfter::PrintGraphChain) start ----";
  NodeDef* current_node = sink_node;
  int64 cur_pos_from_back = 0;

  // reserve some space for printing
  while (cur_pos_from_back < split_node_index + 5 &&
         current_node->input_size() == 1 // more than one: branches; fewer than one: start point
          ) {
    if (cur_pos_from_back == split_node_index) {
      VLOG(0) << " <- **(" << current_node->name() << ")**";
    }
    else {
      VLOG(0) << " <- (" << current_node->name() << ")";
    }
    cur_pos_from_back++;
    int idx = graph_utils::FindGraphNodeWithName(current_node->input(0), *output);
    current_node = output->mutable_node(idx);
  }

  VLOG(0) << "(PrintGraphChain) end ----";
}

Status DeleteNodesAfter::ApplyOptimization(MutableGraphView &graph,
                                    NodeDef* sink_node,
                                    GraphDef* output) {
  VLOG(0) << "(DeleteNodesAfter::ApplyOptimization) start";

  int64 split_node_index = config_.parameter_map()
          .at(kSplitNodeIndex)
          .i();

  PrintChainOfGraph(sink_node, output, split_node_index);

  // iterate throught the graph from sink node
  absl::flat_hash_set<std::string> visited;
  NodeDef* current_node = sink_node;
  NodeDef* prev_node;
  int64 cur_pos_from_back = 0; // sink node position

  // TODO: standard needs to be the same between client and dispatcher
  if (split_node_index == 0) {
    VLOG(0) << "split node index equal to zero, no need to split";
    return Status::OK();
  }

  // TODO: add more print_outs
  // to simplify, we only consider splitting the "tail" of the graph
  /*
   * A -> B \
   * C -> D -> E -> F -> G -> H -> sink
   *
   * we iterate from sink to Node "E"
   */
  while (cur_pos_from_back < split_node_index &&
    current_node->input_size() == 1 // more than one: branches; fewer than one: start point
  ) {
    cur_pos_from_back++;
    int idx = graph_utils::FindGraphNodeWithName(current_node->input(0), *output);
    prev_node = current_node;
    current_node = output->mutable_node(idx);
  }

  if (cur_pos_from_back < split_node_index) {
    // TODO: Split Not Successful
    VLOG(0) << "(DeleteNodesAfter::ApplyOptimization): Chain not long enough "
      << "cur_pos_from_back: " << cur_pos_from_back
      << "; split_node_index: " << split_node_index;
  }
  // current_node is the right node to split

  // current_node -> prev_node -> ... -> sink

  // whether this one is needed
  prev_node->mutable_input()->Clear();
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

