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

NodeDef MakeNewFilterNode(const NodeDef& first_filter_node,
                            const NodeDef& second_filter_node,
                            MutableGraphView* graph) {
    NodeDef new_f_node;
    graph_utils::SetUniqueGraphNodeName("n_filter", graph->graph(),
                                      &new_f_node);

    new_f_node.set_op(second_filter_node.op());
    new_f_node.add_input(first_filter_node.input(0));

    //auto attr = second_filter_node.attr().at("predicate");
    //*attr.mutable_func()->mutable_name() = fused_function.signature().name();
    //(*new_f_node.mutable_attr())["predicate"] = std::move(attr);
    VLOG(0) << "making new filter predicate";
    (*new_f_node.mutable_attr())["predicate"] = second_filter_node.attr().at("predicate");

    graph_utils::CopyAttribute("Targuments", second_filter_node, &new_f_node);

    for (auto key : {"output_shapes", "output_types"})
        graph_utils::CopyAttribute(key, second_filter_node, &new_f_node);
    //graph_utils::MaybeSetFusedMetadata(first_filter_node, second_filter_node, &new_f_node);

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
        /*if (last_seen == "FilterDataset") { // We've found the next fixed op
            next_op = &node;


            absl::flat_hash_set<string> nodes_to_delete;
            VLOG(0) << "Start to rip out filter node";
            VLOG(0) << f_op->input_size();
            NodeDef* const parent = graph_utils::GetInputNode(*f_op, graph);
            VLOG(0) << "Got parent node";
            //(*node.mutable_input())[0] = parent->name();
            //TF_RETURN_IF_ERROR(graph.UpdateFanouts(node.name(), parent->name()));
            graph.UpdateFanouts(node.name(), parent->name());
            VLOG(0) << "Updated fanouts";
            //TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));
            nodes_to_delete.insert(f_op->name());
            graph.DeleteNodes(nodes_to_delete);
            VLOG(0) << "Deleted Nodes";
        }*/
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

        /*
        // GIVES SEG FAULT
        // Reorder the map and batch
        NodeDef* map_input = graph_utils::GetInputNode(*m_op, graph);
        NodeDef* batch_input = graph_utils::GetInputNode(*b_op, graph); // should be the map_op
        VLOG(0) << "Batch's input is " << batch_input->op();
        if (!map_input || !batch_input) {
            return -1;
            //return errors::Unknown("The one of the target nodes has no inputs.");
        }

        */

        /*NodeDef forty_two_node;

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

        // Copy over the relevant attributes
        (*target->mutable_input())[0] = forty_two_node.name();
        graph_utils::CopyAttribute(kOutputTypes, forty_two_node, target);*/


        // My stuff
        //(*b_op->mutable_input())[0] = map_input->name();
        //(*m_op->mutable_input())[0] = b_op->name();
        //(*next_op->mutable_input())[0] = m_op->name();
        
    }

    if (filter_present) {
        // For now just rip out the filter node (and see if graph is rewired correctly)
        VLOG(0) << "Filter present";
        /*absl::flat_hash_set<string> nodes_to_delete;
        VLOG(0) << "Start to rip out filter node";
        VLOG(0) << f_op.input_size();
        NodeDef* const parent = graph_utils::GetInputNode(*f_op, graph);
        VLOG(0) << "Got parent node";
        //TF_RETURN_IF_ERROR(graph.UpdateFanouts(node.name(), parent->name()));
        graph.UpdateFanouts(f_op->name(), parent->name());
        VLOG(0) << "Updated fanouts";
        //TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));
        nodes_to_delete.insert(f_op->name());
        graph.DeleteNodes(nodes_to_delete);
        VLOG(0) << "Deleted Nodes";*/

        
        // TODO: make wheel & docker img for CPU, all for TPU
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

    while (!bfs_queue.empty()) {
        //VLOG(0) << "Trying another one";
        //VLOG(0) << bfs_queue.size();
        NodeDef* current_node = bfs_queue.front();
        //VLOG(0) << current_node->op();
        bfs_queue.pop();
        //VLOG(0) << "poped elem";
        visited.insert(current_node->name());
        //VLOG(0) << "Getting the cur input";
        // This gives a seg fault
        //NodeDef* cur_input = graph_utils::GetInputNode(*target, graph);
        /*if (cur_input->op().find("FilterDataset") != std::string::npos) {
            VLOG(0) << "Found node with Filter Input!";
            target = current_node;
            (*target->mutable_input())[0] = cur_input->name();

            absl::flat_hash_set<string> nodes_to_delete;
            VLOG(0) << "Start to rip out filter node";
            VLOG(0) << target->input_size();

            //NodeDef* const parent = graph_utils::GetInputNode(*f_op, graph);
            //VLOG(0) << "Got parent node";
            //(*node.mutable_input())[0] = parent->name();
            //TF_RETURN_IF_ERROR(graph.UpdateFanouts(node.name(), parent->name()));
            
            
            // Maybe we need this??????????
            //graph.UpdateFanouts(node.name(), parent->name());
            //VLOG(0) << "Updated fanouts";
            //TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));
            
            nodes_to_delete.insert(cur_input->name());
            graph.DeleteNodes(nodes_to_delete);
            VLOG(0) << "Deleted Nodes";


            break;
        }*/

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
                    // Update Fanouts between filter node & parent ??? (As in noop_elimination)
                    NodeDef* const parent = graph_utils::GetInputNode(*current_node, graph);
                    VLOG(0) << "Input node is " << parent->op();
                    VLOG(0) << "Cur node is " << current_node->op();
                    VLOG(0) << "Target Node is " target->op();

                    const auto* new_filter_node = graph.AddNode(MakeNewFilterNode(
                    *parent, *current_node, &graph));
                    TF_RETURN_IF_ERROR(graph.UpdateFanouts(parent->name(),
                                           new_filter_node->name()));

                    VLOG(0) << "New node is " << new_filter_node->op();
                    VLOG(0) << "New node's input is " << new_filter_node->input(0);
                    VLOG(0) << "New node's parent is " << graph_utils::GetInputNode(*new_filter_node, graph)->op();
                    
                    
                    TF_RETURN_IF_ERROR(graph.UpdateFanouts(current_node->name(), parent->name()));
                    VLOG(0) << "Old nodes test!!!!!!!!";
                    VLOG(0) << "(original) Parent is " << parent->op();
                    VLOG(0) << "Parent's input is " << parent->input(0);

                    VLOG(0) << "Target node is " << target->op();
                    VLOG(0) << "Target node's input" << target->input(0);
                    

                    nodes_to_delete.insert(current_node->name());
                    graph.DeleteNodes(nodes_to_delete);
                    
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
