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
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"

#include "tensorflow/core/data/service/easl/split_pipeline_state.h"


namespace tensorflow {
namespace grappler {
namespace easl {
namespace {

  constexpr char kDSDO[] = "DataServiceDatasetV3";

  NodeDef* getNodeDefFromName(GraphDef* graph, std::string name) {
    int idx = tensorflow::grappler::graph_utils::FindGraphNodeWithName(name, *graph);
    if (idx == -1) {
      VLOG(0) << "Node: " << name << " not found in graph";
      return NULL;
    }
    return graph->mutable_node(idx);
  }

  std::string getNodeRepr(const NodeDef* sink) {
    return "[" + sink->name() + ", " + sink->op() + "]";
  }

  void BFSFromSink(GraphDef* graph, const NodeDef* sink, absl::flat_hash_set<const NodeDef*>& visited) {
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

  void BFSWholeGraph(GraphDef* graph, std::string prefix = "123") {
    VLOG(0) << "AppendNodesAfter::BFSWholeGraph " << prefix;


    absl::flat_hash_set<const NodeDef*> visited;

    for (const auto& _node: graph->node()) {
      std::string node_name = _node.name();
      NodeDef* node = getNodeDefFromName(graph, node_name);
      if (!visited.contains(node)) {
        BFSFromSink(graph, node, visited);
      }
    }
  }

  NodeDef* getGraphDSDO(GraphDef* graph) {
    for (const auto& _node: graph->node()) {
      if (_node.op() == kDSDO) {
        return getNodeDefFromName(graph, _node.name());
      }
    }
    return NULL;
  }

  NodeDef* getDownstreamNode(GraphDef* graph, NodeDef* node) {
    VLOG(0) << "In getDownstreamNode";
    for (const auto& _node: graph->node()) {
      for (int i = 0; i < _node.input_size(); i++) {
        if (_node.input(i) == node->name()) {
          VLOG(0) << "Downstream Node of " << getNodeRepr(node) << " is "
            << getNodeRepr(&_node);
          return getNodeDefFromName(graph, _node.name());
        }
      }
    }
    return NULL;
  }

  NodeDef* getSinkNode(GraphDef* graph) {
    for (const auto& _node: graph->node()) {
      if (_node.op() == "_Retval") {
        NodeDef* result = getNodeDefFromName(graph, _node.input(0));
        VLOG(0) << "Possible Sink Node is: " << result->name();
        return result;
      }
    }

    return NULL;
  }


//  NodeDef* getSinkNode(GraphDef*  ssgraph) {
//    absl::flat_hash_set<std::string> hset;
//
//    for (const auto& _node: graph->node()) {
//      for (int i = 0; i < _node.input_size(); i++) {
//        hset.insert(_node.input(i));
//      }
//    }
//
//    NodeDef* result = 0;
//
//    for (const auto& _node: graph->node()) {
//      if (!hset.contains(_node.name())) {
//        result = getNodeDefFromName(graph, _node.name());
//        VLOG(0) << "Possible Sink Node is: " << result->name();
//      }
//    }
//
//    return result;
//  }

  std::string shNodeName(std::string name) {
    return "SH_" + name;
  }

  NodeDef* addNodeToGraph(NodeDef* node, tensorflow::grappler::MutableGraphView& graph_view) {
    VLOG(0) << "Add Node: " << getNodeRepr(node) << " to graph";
    std::vector<std::pair<std::string, AttrValue>> attrs(
            {node->mutable_attr()->begin(), node->mutable_attr()->end()});
    std::vector<std::string> inputs;

//    VLOG(0) << "Print attrs: ";
//    for (const auto& p: attrs) {
//      VLOG(0) << p.first;
//    }

//    // Do i need to add functions into the library?
//    FunctionLibraryDefinition function_library(OpRegistry::Global(),
//                                               graph_view.graph()->library());
//    const FunctionDef* fdef =
//            function_library.Find(node->attr().at("f").func().name());
//    // deep copy needed?
//    VLOG(0) << "Print Function Def " << fdef->signature().name();
//    for (const auto& p: fdef->attr()) {
//      VLOG(0) << p.first;
//    }

    NodeDef* new_node =  tensorflow::grappler::graph_utils::AddNode(
            shNodeName(node->name()),
            node->op(),
//            {node->mutable_input()->begin(), node->mutable_input()->end()},
//            {node->mutable_attr()->begin(), node->mutable_attr()->end()},
            {},
            {},
            &graph_view);

    for (const auto& p: attrs) {
//      VLOG(0) << "Copying attribute name: " << p.first;
      graph_utils::CopyAttribute(p.first, *node, new_node);
    }

    return new_node;

  }
}

Status AppendNodesAfterDSDO::OptimizeAndCollectStats(
        Cluster *cluster, const GrapplerItem& item,
        GraphDef *output, OptimizationStats *stats) {

  VLOG(0) << "In AppendNodesAfterDSDO Optimizer";
  *output = item.graph;
  BFSWholeGraph(output);

  int64 split_node_index =
            tensorflow::data::service::easl::split_state::SplitIndexes::GetSplitIndex();

  VLOG(0) << "ApplyNodesAfterDSDO: split_node_index: " << split_node_index;

  if (split_node_index <= 0) {
    VLOG(0) << "ApplyNodesAfterDSDO: split_node_index is smaller equal to zero, do nothing "
      << split_node_index;
    return Status::OK();
  }

  tensorflow::data::service::easl::split_state::SplitIndexes::Print();
  tensorflow::data::service::easl::split_state::SplitOriginalGraph::Print();

  NodeDef* dsdo_node = getGraphDSDO(output);

  if (dsdo_node == NULL) {
    VLOG(0) << "ApplyNodesAfterDSDO: GraphNode DSDO doesn't exist";
    return Status::OK();
  }

  GraphDef second_half_graph_ = tensorflow::data::service::easl::split_state::SplitOriginalGraph::GetGraph();
  GraphDef* second_half_graph = &second_half_graph_;

  // extend function library
  const auto& sh_library = second_half_graph->library();
  for (auto func: sh_library.function()) {
    VLOG(0) << "Add function def: " << func.signature().name();
    output->mutable_library()->mutable_function()->Add(std::move(func));
  }
  tensorflow::grappler::MutableGraphView dsdo_graph_view(output);

  VLOG(0) << "Print all functions";
  for (const auto & func: output->library().function()) {
    VLOG(0) << func.signature().name();
  }

  // Apply Rewrite
  NodeDef* sink_node_sh = getSinkNode(second_half_graph);
  NodeDef* current_node_sh = sink_node_sh;
  VLOG(0) << "Sink Node of Second Half Graph: " << getNodeRepr(current_node_sh);
  NodeDef* current_node_dsdo = addNodeToGraph(current_node_sh, dsdo_graph_view);

  // 0. join second half with dsdo downstream node
  NodeDef* dsdo_next_node = getDownstreamNode(output, dsdo_node);

  std::vector<std::string> vs;
  vs.push_back(std::string(current_node_dsdo->name()));
  for (int i = 0; i < dsdo_next_node->input_size(); i++) {
    std::string up_node_name = dsdo_next_node->input(i);
    NodeDef* up_node = getNodeDefFromName(output, up_node_name);
    if (up_node->op() == "Const") {
      vs.push_back(up_node_name);
    }
  }

  *(dsdo_next_node->mutable_input()) = {vs.begin(), vs.end()};
  BFSWholeGraph(output, "After Step 0");

  // 1. append second half graph nodes to first half
  int64 cur_pos_from_back = 0;
  for (; cur_pos_from_back < split_node_index; ) {
    NodeDef* next_node_sh = NULL;

    bool end_of_chain = (current_node_sh->op() == "SplitMarkerDataset" &&
            cur_pos_from_back == split_node_index - 1);

    for (int i = 0; i < current_node_sh->input_size(); i++) {
      NodeDef* tmp_node_sh = getNodeDefFromName(second_half_graph,
                                                current_node_sh->input(i));
      // Add this node to the dsdo_graph
      NodeDef* added_node_dsdo = NULL;

      if (!end_of_chain || tmp_node_sh->op() == "Const") {
        added_node_dsdo = addNodeToGraph(tmp_node_sh, dsdo_graph_view);
        current_node_dsdo->mutable_input()->Add(std::string(added_node_dsdo->name()));
      }
//      else if (tmp_node_sh->op() != "SplitMarkerDataset") {
//        VLOG(0) << "Skipping node: [" << current_node_sh->name() << ", "
//                << current_node_sh->op() << "]'s non_const upstream node: ["
//                << tmp_node_sh->name() << ", " << tmp_node_sh->op()
//                << "]";
//      }

      if (tmp_node_sh->op() != "Const") {
        // guaranteed to be only happening once
        next_node_sh = tmp_node_sh;
        if (!end_of_chain) {
          current_node_dsdo = added_node_dsdo;
          /*
           * A -> B -> C -> D, split_node_index = 2
           * current_node_dsdo is going to stop at B, but we need to stop at C
           */
        }
      }

    }
    if (current_node_sh->op() == "SplitMarkerDataset") {
      cur_pos_from_back++;
    }
    current_node_sh = next_node_sh;
  }
  BFSWholeGraph(output, "After Step 1");

  // 2. join two sub-graphs
  current_node_dsdo->mutable_input()->Add(std::string(dsdo_node->name()));

  BFSWholeGraph(output, "After Step 3");

  return Status::OK();

}

REGISTER_GRAPH_OPTIMIZER_AS(AppendNodesAfterDSDO, "append_nodes_after_dsdo");

} // easl
} // grappler
} // tensorflow