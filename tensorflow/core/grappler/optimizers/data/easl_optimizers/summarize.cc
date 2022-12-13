#include <queue>
#include <algorithm>
#include <vector>

#include "tensorflow/core/grappler/optimizers/data/easl_optimizers/summarize.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/fusion_utils.h"
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {
namespace easl {
namespace {

int GetOrderCost(const GraphDef& suggested_order, MutableGraphView &graph, std::vector<std::string> &op_types) {
  std::string last_seen;

  for (const NodeDef& node : suggested_order.node()) {
    auto op_name = node.op();
    op_types.push_back(op_name);
  }
  return 0;
}

}  // namespace

Status Summarize::ApplyOptimization(MutableGraphView &graph, GraphDef &sorted_old_graph) {

    return Status::OK();
}

Status Summarize::OptimizeAndCollectStats(Cluster* cluster,
                                          const GrapplerItem& item,
                                          GraphDef* output,
                                          OptimizationStats* stats) {
    VLOG(0) << "SUMMARZING OPTIMIZED GRAPH !!!!!!!!!";
    GraphDef sorted_old_graph = item.graph;
    TF_RETURN_IF_ERROR(TopologicalSort(&sorted_old_graph));
    *output = sorted_old_graph;

    VLOG(0) << "Sorted graph";

    MutableGraphView graph(output);
    absl::flat_hash_set<string> nodes_to_delete;
    FunctionLibraryDefinition function_library(OpRegistry::Global(), output->library());

    std::vector<std::string> op_types;
    auto cost = GetOrderCost(sorted_old_graph, graph, op_types);
    VLOG(0) << (std::find(op_types.begin(), op_types.end(), "MapDataset") != op_types.end());
    VLOG(0) << (std::find(op_types.begin(), op_types.end(), "FilterDataset") != op_types.end());

    // Only proceed with optimization if we are in the 'right' pipeline (we see a Filter or Map op)
    if (std::find(op_types.begin(), op_types.end(), "MapDataset") == op_types.end() && std::find(op_types.begin(), op_types.end(), "FilterDataset") == op_types.end()) {
        VLOG(0) << "No reorderable ops found! Not running summarization on this pipeline.";
        return Status::OK();
    }

    // Get the output of the graph
    VLOG(0) << "Searching for sink node";
    NodeDef* sink_node;
    if (item.fetch.size() == 1) {
    TF_RETURN_IF_ERROR(graph_utils::GetFetchNode(graph, item, &sink_node));

    // Find the first batch op by applying BFS
    absl::flat_hash_set<std::string> visited;
    std::queue<NodeDef*> bfs_queue;
    VLOG(0) << "Searching for wanted node";
    bfs_queue.push(sink_node);
    NodeDef* target = nullptr;

    std::vector<NodeDef*> changing_dtype = {};
    std::vector<NodeDef*> changing_shape = {};

    //std::vector<bool> node_changed_dtype = {false, true}; // remember we are going bottom up in the pipeline
    std::vector<bool> node_changed_dtype = {false, false}; // remember we are going bottom up in the pipeline
    std::vector<bool> node_changed_shape = {false, false};
    
    while (!bfs_queue.empty()) {
        //VLOG(0) << "Trying another one";
        //VLOG(0) << bfs_queue.size();
        NodeDef* current_node = bfs_queue.front();
        VLOG(0) << "Visiting node " << current_node->name();
        VLOG(0) << "It has op type " << current_node->op();
        VLOG(0) << "Summary of node: " << SummarizeNodeDef(*current_node, 100);

        NodeDef_ExperimentalDebugInfo debug_i = current_node->experimental_debug_info();
        int num_org_nodes = debug_i.original_node_names_size();
        VLOG(0) << "The the current NodeDef was made up from " << num_org_nodes << " nodes.";
        std::vector<std::string> org_names;
        for (int i = 0 ; i < num_org_nodes; ++i) {
            std::string name = debug_i.original_node_names(i);
            VLOG(0) << "Original node " << i << " is called " << name;
            org_names.push_back(name);
        }

        int num_org_funcs = debug_i.original_func_names_size();
        VLOG(0) << "Originally the node is made up of " << num_org_funcs << " functions";
        std::vector<std::string> org_funcs;
        for (int i = 0 ; i < num_org_funcs; ++i) {
            std::string func = debug_i.original_func_names(i);
            VLOG(0) << "Original func " << i << " is called " << func;
            org_funcs.push_back(func);
        }

        bfs_queue.pop();
        VLOG(0) << "popped elem";
        visited.insert(current_node->name());

        // Iterate through the neighbors
        for (int i = 0; i < current_node->input_size(); ++i) {
            if (!visited.contains(current_node->input(i))) {
                int idx = graph_utils::FindGraphNodeWithName(current_node->input(i), 
                *output);
                NodeDef* neighbor_node = output->mutable_node(idx);
                bfs_queue.push(neighbor_node);
                }
            }
        }
    }
    VLOG(0) << "Graph summarization complete!";

    return ApplyOptimization(graph, sorted_old_graph);
}

REGISTER_GRAPH_OPTIMIZER_AS(Summarize, "summarize");

}  // namespace easl
}  // namespace grappler
}  // namespace tensorflow
