#include <queue>

#include "tensorflow/core/grappler/optimizers/data/easl_optimizers/auto_order.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/fusion_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {
namespace easl {
namespace {

NodeDef MakeNewNode(const NodeDef& new_input_node,
                    const NodeDef& org_node,
                    MutableGraphView* graph) {
    NodeDef new_f_node;
    graph_utils::SetUniqueGraphNodeName("n_filter", graph->graph(),
                                      &new_f_node);

    new_f_node.set_op(org_node.op());
    new_f_node.add_input(new_input_node.input(0));

    //auto attr = second_filter_node.attr().at("predicate");
    //*attr.mutable_func()->mutable_name() = fused_function.signature().name();
    //(*new_f_node.mutable_attr())["predicate"] = std::move(attr);
    VLOG(0) << "making new filter predicate";
    (*new_f_node.mutable_attr())["predicate"] = org_node.attr().at("predicate");

    graph_utils::CopyAttribute("Targuments", org_node, &new_f_node);

    for (auto key : {"output_shapes", "output_types"})
        graph_utils::CopyAttribute(key, org_node, &new_f_node);
    //graph_utils::MaybeSetFusedMetadata(first_filter_node, org_node, &new_f_node);

    return new_f_node;
}

// Given a suggested pipeline check that the output shapes/sizes match
Status IsPipelineOk(const GraphDef& suggested_order, MutableGraphView &graph) {
    if (false) {
        return errors::Internal("The suggested graph fails.");
    }
    return Status::OK();
}

// Calculate the cost of a proposed op ordering
// Cost = sum(Shape(op) * DTypeBytes(op))
int GetOrderCost(const GraphDef& suggested_order, MutableGraphView &graph) {
    double cost = 0;

    const NodeDef* m_op = nullptr;
    const NodeDef* b_op = nullptr;
    const NodeDef* f_op = nullptr;
    const NodeDef* next_op = nullptr;
    std::string last_seen;
    
    bool batch_present = false;
    bool map_present = false;
    bool filter_present = false;
    for (const NodeDef& node : suggested_order.node()) {
        //auto dt
        //NodeDef* n_ptr = &node;
        auto op_name = node.op();
        //auto output_s = node.output_size();
        auto input_s = node.input_size();
        double ret_factor = 1.0;
        //double inf_factor = output_s/input_s;
        if (op_name.find("BatchDataset") != std::string::npos) {
            batch_present = true;
            b_op = &node;
        }
        if (op_name.find("MapDataset") != std::string::npos) {
            map_present = true;
            m_op = &node;
        }
        if (op_name.find("FilterDataset") != std::string::npos) {
            filter_present = true;
            f_op = &node;

            
        }
        last_seen = op_name;

        VLOG(0) << op_name;
        //VLOG(0) << output_s;
        VLOG(1) << input_s;
        //VLOG(0) << inf_factor;
        //VLOG(0) << ret_factor;

        //cost+=output_s;
        cost+=input_s*ret_factor;

        
        last_seen = op_name;
    }

    if (batch_present && map_present) { // Should be correct graph
        VLOG(0) << "Found map & batch";
        
    }

    if (filter_present) {
        // For now just rip out the filter node (and see if graph is rewired correctly)
        VLOG(0) << "Filter present";
    }

    return cost;
}

}  // namespace

Status AutoOrder::ApplyOptimization(MutableGraphView &graph, GraphDef &sorted_old_graph) {
    VLOG(0) << "In AutoOrder::ApplyOptimization";

    VLOG(0) << "Original pipline:";
    auto cost = GetOrderCost(sorted_old_graph, graph);
    VLOG(0) << "Total cost:";
    VLOG(0) << cost;

    VLOG(0) << "Updated graph cost:";

    auto new_cost = GetOrderCost(sorted_old_graph, graph);
    VLOG(0) << "Total cost:";
    VLOG(0) << new_cost;
    
    Status s = IsPipelineOk(sorted_old_graph, graph);
    while (!s.ok()) {
        // Choose next best suggestion
        VLOG(0) << "Updating suggestion";
        s = IsPipelineOk(sorted_old_graph, graph);
    }


    return Status::OK();
}

Status AutoOrder::OptimizeAndCollectStats(Cluster* cluster,
                                          const GrapplerItem& item,
                                          GraphDef* output,
                                          OptimizationStats* stats) {
    VLOG(0) << "OTO: AUTO ORDER BEING APPLIED (Should come before filter fusion)!!!!!!!!!";
    GraphDef sorted_old_graph = item.graph;
    TF_RETURN_IF_ERROR(TopologicalSort(&sorted_old_graph));
    *output = sorted_old_graph;

    VLOG(0) << "Sorted graph";

    MutableGraphView graph(output);
    absl::flat_hash_set<string> nodes_to_delete;
    FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                               output->library());

    auto cost = GetOrderCost(sorted_old_graph, graph);
    VLOG(0) << "Now we try to optimize";

    auto get_filter_node = [](const NodeDef& node) -> const NodeDef* {
        // TODO(b/148614315): Support captured inputs.
        if (node.op() == "FilterDataset" && node.input_size() == 1) return &node;
        return nullptr;
    };

    auto make_fused_function = [&](const NodeDef* first_filter_node,
                                   const NodeDef* second_filter_node) -> FunctionDef* {
        const auto& parent_fun = first_filter_node->attr().at("predicate");
        const FunctionDef* first_func =
            function_library.Find(parent_fun.func().name());
        const auto& fun = second_filter_node->attr().at("predicate");
        const FunctionDef* second_func = function_library.Find(fun.func().name());

        if (!fusion_utils::HasSameSignature(first_func->signature(),
                                            second_func->signature())) {
            VLOG(1) << "Can't fuse Filters because they have different signature\n";
            return nullptr;
        }

        return fusion_utils::FuseFunctions(
            *first_func, *second_func, "fused_predicate",
            fusion_utils::SameSignature, fusion_utils::SameInput,
            fusion_utils::LazyConjunctionOutput, fusion_utils::LazyConjunctionNodes,
            output->mutable_library());
    };

    // NEW STUFF
        
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

    auto first_dtype = 

    while (!bfs_queue.empty()) {
        //VLOG(0) << "Trying another one";
        //VLOG(0) << bfs_queue.size();
        NodeDef* current_node = bfs_queue.front();
        VLOG(0) << "Visiting " << current_node->op();
        VLOG(0) << "Curent output_type is " << (*new_f_node.mutable_attr())["output_types"];
        VLOG(0) << "Curent output_type is " << (*new_f_node.mutable_attr())["output_shapes"];
        bfs_queue.pop();
        //VLOG(0) << "poped elem";
        visited.insert(current_node->name());

        // Iterate throught the neighbors
        for (int i = 0; i < current_node->input_size(); ++i) {
            if (!visited.contains(current_node->input(i))) {
                int idx = graph_utils::FindGraphNodeWithName(current_node->input(i), 
                *output);
                NodeDef* neighbor_node = output->mutable_node(idx);
                if (neighbor_node->op().find("FilterDataset") != std::string::npos) {
                    VLOG(0) << "Found node with Filter input";
                    VLOG(0) << current_node->op();
                    target = current_node;
                    bfs_queue.push(neighbor_node);
                    
                } else if (current_node->op().find("FilterDataset") != std::string::npos) {
                    VLOG(0) << "Found Filter node";
                    VLOG(0) << current_node->op();
                    VLOG(0) << current_node->input(0);
                    (*target->mutable_input())[0] = current_node->input(0);

                    bfs_queue.push(neighbor_node);

                    absl::flat_hash_set<string> nodes_to_delete;
                    VLOG(0) << "Deleting filter node";
                    NodeDef* const parent = graph_utils::GetInputNode(*current_node, graph);
                    VLOG(0) << "Input node is " << parent->op();
                    VLOG(0) << "Cur node is " << current_node->op();
                    VLOG(0) << "Target Node is " << target->op();

                    const auto* new_filter_node = graph.AddNode(MakeNewNode(
                    *parent, *current_node, &graph));
                    TF_RETURN_IF_ERROR(graph.UpdateFanouts(parent->name(), new_filter_node->name()));

                    VLOG(0) << "New node is " << new_filter_node->op();
                    VLOG(0) << "New node's input is " << new_filter_node->input(0);
                    VLOG(0) << "New node's parent is " << graph_utils::GetInputNode(*new_filter_node, graph)->op();
                    
                    (*parent->mutable_input())[0] = new_filter_node->name();
                    TF_RETURN_IF_ERROR(graph.UpdateFanouts(current_node->name(), parent->name()));
                    VLOG(0) << "Old nodes test!!!!!!!!";
                    VLOG(0) << "(original) Parent is " << parent->op();
                    VLOG(0) << "Parent's input is " << parent->input(0);

                    VLOG(0) << "Target node is " << target->op();
                    VLOG(0) << "Target node's input " << target->input(0);


                    nodes_to_delete.insert(current_node->name());
                    graph.DeleteNodes(nodes_to_delete);
                    
                    // We've reordered some nodes, now jump out of the process
                    return ApplyOptimization(graph, sorted_old_graph);
                } else {
                    bfs_queue.push(neighbor_node);
                }
            }
        }
    }





    return ApplyOptimization(graph, sorted_old_graph);

    // TODO: Find a way to update num_changes
    // TODO: Update metadata (see inject_prefectch.cc line 128)
    // stats->num_changes++;

    }
    return Status::OK();

}

REGISTER_GRAPH_OPTIMIZER_AS(AutoOrder, "auto_order");

}  // namespace easl
}  // namespace grappler
}  // namespace tensorflow
