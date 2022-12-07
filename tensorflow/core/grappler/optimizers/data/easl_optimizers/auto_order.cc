#include <queue>
#include <algorithm>

#include "tensorflow/core/grappler/optimizers/data/easl_optimizers/auto_order.h"

#include "absl/container/flat_hash_set.h"
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

std::string GetOutputType(const std::string node_str){
  std::string delimiter = "output_types=";

  //{{node FilterDataset/_5}} = FilterDataset[Targuments=[], _cardinality=-2, metadata="\n\01"\n\017FilterDataset:4", output_shapes=[[]], output_types=[DT_INT64], predicate=__inference_Dataset_filter_lambda_28[]](MapDataset/_4)

  if (node_str.find(delimiter) != std::string::npos) {
    std::string dt = node_str.substr(node_str.find(delimiter), node_str.find("], "));
    dt = dt.erase(0, delimiter.size());
    dt = dt.substr(0, dt.find("], "));
    dt = dt + "]";
    return dt;
  } else {
    return "";
  }
}

NodeDef MakeNewNode(const NodeDef& org_position_node,
                    const NodeDef& org_node,
                    MutableGraphView* graph,
                    FunctionLibraryDefinition function_library,
                    FunctionDefLibrary* library,
                    bool changes_dtype = false,
                    bool changes_shape = false) {
    NodeDef new_f_node;
    VLOG(0) << "Inside MakeNewNode";
    VLOG(0) << "Node is a " << org_node.op();
    graph_utils::SetUniqueGraphNodeName("new_node", graph->graph(),
                                      &new_f_node);

    new_f_node.set_op(org_node.op());
    VLOG(0) << "Set op: " << org_node.op();
    new_f_node.add_input(org_position_node.input(0));
    VLOG(0) << "Set input: " << org_position_node.input(0);

    // new_f_node doesn't work as arg for next line for some reason! Maybe new_f_node not set up correctly yet?
    NodeDef* in_node = graph_utils::GetInputNode(org_position_node, *graph);
    VLOG(0) << "Got the input node";

    //auto attr = second_filter_node.attr().at("predicate");
    //*attr.mutable_func()->mutable_name() = fused_function.signature().name();
    //(*new_f_node.mutable_attr())["predicate"] = std::move(attr);
    //VLOG(0) << "making new filter predicate";

    // Add predicates if present
    // TODO: Check what else different op types contain
    std::string summary = SummarizeNodeDef(org_node, 100);
    VLOG(0) << "Summarized org node";
    VLOG(0) << summary;
    VLOG(0) << "Summarized org_position node";
    VLOG(0) << SummarizeNodeDef(org_position_node, 100);
    VLOG(0) << "Summarized org_position input node";
    VLOG(0) << SummarizeNodeDef(*in_node, 100);

    // Add corresponding predicate if present in the original node (i.e. it was a filter node)
    if (summary.find("predicate=") != std::string::npos) {
        (*new_f_node.mutable_attr())["predicate"] = org_node.attr().at("predicate");
        VLOG(0) << "Set predicate (a predicate existed)";

        // We now have to adjust the input type of this predicate/FunctionDef (later on, do the same for map ops)
        if (!changes_dtype) {
            // NEW parent's output type
            std::string parent_out_type_string = GetOutputType(SummarizeNodeDef(*in_node, 100));
            // Remove the '[' and ']' chars
            parent_out_type_string = parent_out_type_string.substr(1,parent_out_type_string.length()-2);
            VLOG(0) << "parent_out_type_string: " << parent_out_type_string;

            std::vector<string> out_type_strings;
            std::stringstream ss(parent_out_type_string);
            std::string type;
            while (getline(ss, type, ',')) {
                out_type_strings.push_back(type);
            }

            const AttrValue& filter_pred = new_f_node.attr().at("predicate");
            AttrValue non_const_filter_pred = (*new_f_node.mutable_attr())["predicate"];
            //FunctionDef func_def_direct = (*new_f_node.mutable_attr())["predicate"].func();
            std::string func_name = (*new_f_node.mutable_attr())["predicate"].func().name();
            VLOG(0) << "Name of filter pred function " << func_name;
            VLOG(0) << "Adjusting filter input dtype!";
            const FunctionDef* filter_func = function_library.Find(func_name);



            // THIS MUST GO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            FunctionDef* mutable_filter_func = const_cast<FunctionDef*>(filter_func);
            OpDef* mutable_ff_sig = mutable_filter_func->mutable_signature();


            tensorflow::grappler::fusion_utils::StringCollection filter_inputs = fusion_utils::GetFunctionInputs(*filter_func);
            // filter_func->signature().input_arg() is of type: const OpDef
            const OpDef ff_sig_const = filter_func->signature();
            // Another BAD ONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            OpDef ff_sig = (OpDef)ff_sig_const;

            auto filter_args = filter_func->signature().input_arg();
            int in_arg_size = ff_sig.input_arg_size();
            VLOG(0) << "There are " << in_arg_size << " arguments";

            //VLOG(0) << "ORIGINAL ArgDef summary:";
            //std::string arg_sum = SummarizeArgs(filter_args);
            //VLOG(0) << arg_sum;

            VLOG(0) << "ORIGINAL OpDef summary:";
            std::string op_sum = SummarizeOpDef(ff_sig_const);
            VLOG(0) << op_sum;

            VLOG(0) << "ORIGINAL Non-const OpDef summary:";
            std::string nc_op_sum = SummarizeOpDef(ff_sig);
            VLOG(0) << nc_op_sum;

            // Fixing an existing function didn't work. Try to construct a brand-new function (based of fusion_utils)

            // This function will be used as a clone of second function, having unique
            // names.
            const FunctionDef* org_func = function_library.Find(org_node->attr().at("predicate").func().name());
            OpDef org_func_sig = org_func->signature();
            FunctionDef setup_function = org_func;

            // TODO: Make a new 'GetUniqueSignature' for our purposes
            /**setup_function.mutable_signature() = GetUniqueSignature(
                first_function.signature(), setup_function.signature(),
                setup_function.mutable_ret(), setup_function.mutable_node_def());*/

            // We aren't fusing anything, so let's just keep the names the same as much as possible
            OpDef signature;
            signature.set_name(setup_function.signature().name());

            for (int i = 0; i < in_arg_size; ++i) {
                // Make a new (mutable) input arg
                const OpDef_ArgDef& input_arg = org_func_sig.input_arg(i);
                input = input_arg;
                input.set_name(input.name());

                // Figure out the CORRECT input type
                DataType dt;
                std::string substr_to_remove = "DT_";
                std::size_t substr_loc = out_type_strings[i].find(substr_to_remove);
                if (substr_loc !=std::string::npos) {
                    out_type_strings[i].erase(substr_loc,substr_to_remove.size());
                }

                //out_type_strings[i].erase(std::remove(out_type_strings[i].begin(), out_type_strings[i].end(), '_'), out_type_strings[i].end());
                std::transform(out_type_strings[i].begin(), out_type_strings[i].end(),out_type_strings[i].begin(), ::tolower);
                VLOG(0) << "Output " << i << " is of type " << out_type_strings[i];
                bool worked = DataTypeFromString(out_type_strings[i], &dt);
                VLOG(0) << "Output has 'DataType' " << dt;

                // Set the right type
                input.set_type(dt);
                VLOG(0) << "Type has been adjusted to " << input.type() << "!";
            }

            for (const auto& output_arg : org_func_sig.output_arg()) {
                auto& output = *signature.add_output_arg();
                output = output_arg;
                output.set_name(output.name()); // Last 2 lines are probably useless??
            }

            *setup_function.mutable_signature() = signature;

            FunctionDef* new_function = library->add_function();

            //fusion_utils::SameSignature(org_func.signature(), setup_function.signature(),
            //              new_function->mutable_signature());
            new_function->mutable_signature() = setup_function.mutable_signature();

            StringPiece func_name_prefix = "reordered_func";
            graph_utils::SetUniqueGraphFunctionName(fused_name_prefix, library,
                                                    new_function);

            //set_output(first_function.ret(), setup_function.ret(),
            //           fused_function->mutable_ret());
            new_function->mutable_ret() = org_func.ret(); // TODO: CHECK THIS ONE!

            auto attr = new_f_node.attr().at("predicate");
            *attr.mutable_func()->mutable_name() = new_function.signature().name();
            (*new_f_node.mutable_attr())["predicate"] = std::move(attr);

            /*

            // TODO: an arg count matching test would be good...

            for (int i = 0; i < in_arg_size; ++i) {
                // First figure out the target data type
                DataType dt;

                std::string substr_to_remove = "DT_";
                std::size_t substr_loc = out_type_strings[i].find(substr_to_remove);
                if (substr_loc !=std::string::npos) {
                    out_type_strings[i].erase(substr_loc,substr_to_remove.size());
                }

                //out_type_strings[i].erase(std::remove(out_type_strings[i].begin(), out_type_strings[i].end(), '_'), out_type_strings[i].end());
                std::transform(out_type_strings[i].begin(), out_type_strings[i].end(),out_type_strings[i].begin(), ::tolower);
                VLOG(0) << "Output " << i << " is of type " << out_type_strings[i];
                bool worked = DataTypeFromString(out_type_strings[i], &dt);
                VLOG(0) << "Output has 'DataType' " << dt;

                // Then get the respective OpDef_ArgDef* and set it
                OpDef_ArgDef* mutable_in_arg = ff_sig.mutable_input_arg(i);
                VLOG(0) << "Input " << i << " was of type " << mutable_in_arg->type();
                VLOG(0) << mutable_in_arg->name();
                VLOG(0) << mutable_in_arg->type();
                //VLOG(0) << mutable_in_arg->description();

                mutable_in_arg->set_type(dt);
                VLOG(0) << "Type has been adjusted to " << mutable_in_arg->type() << "!";

                VLOG(0) << "Second try";
                OpDef_ArgDef* in_arg_mutable = mutable_ff_sig->mutable_input_arg(i);
                in_arg_mutable->set_type(dt);

                function_library.ReplaceFunction(func_name, *mutable_filter_func);



            }

            //VLOG(0) << "EDITED ArgDef summary:";
            //std::string arg_sum_new = SummarizeArgs(filter_args);
            //VLOG(0) << arg_sum_new;

            VLOG(0) << "EDITED OpDef summary:";
            std::string op_sum_new = SummarizeOpDef(ff_sig_const);
            VLOG(0) << op_sum_new;

            VLOG(0) << "EDITED Non-const OpDef summary:";
            std::string nc_op_sum_new = SummarizeOpDef(ff_sig);
            VLOG(0) << nc_op_sum_new;


            VLOG(0) << "Refetching the signature (AKA OpDef)";
            const OpDef ff_sig_const_new = filter_func->signature();
            std::string op_sum_const_new = SummarizeOpDef(ff_sig_const);
            VLOG(0) << op_sum_const_new;


            */
        }
    }
    // Add user-defined function if present in the original node (i.e. it was a map node)
    if (summary.find("f=") != std::string::npos) {
        (*new_f_node.mutable_attr())["f"] = org_node.attr().at("f");
        VLOG(0) << "Set user-defined function (f)";
    }
    // TODO: What about _cardinality, metadata, preserve_cardinality, use_inter_op_parallelism, POSSIBLY MORE

    // Targs should stay the same
    graph_utils::CopyAttribute("Targuments", org_node, &new_f_node);
    VLOG(0) << "Coppied Targs";

    // most nodes don't change dtype/shape (then follow the one from the previous node)
    // otherwise use the dtype/shape of the original node
    if (summary.find("output_types=") != std::string::npos) {
        if (!changes_dtype) {
            graph_utils::CopyAttribute("output_types", *in_node, &new_f_node);
            VLOG(0) << "Used output type of input node";
        } else {
            graph_utils::CopyAttribute("output_types", org_node, &new_f_node);
            VLOG(0) << "Used output type of org node";
        }
    }
    if (summary.find("output_shapes=") != std::string::npos) {
        VLOG(0) << "Setting shape";
        if (!changes_shape) {
            VLOG(0) << "Using shape of input node";
            graph_utils::CopyAttribute("output_shapes", *in_node, &new_f_node);
            VLOG(0) << "Used shape of input node";
        } else {
            VLOG(0) << "Using shape of org node";
            graph_utils::CopyAttribute("output_shapes", org_node, &new_f_node);
            VLOG(0) << "Used shape of org node";
        }
    }

    VLOG(0) << "New node's summary is:";
    VLOG(0) << SummarizeNodeDef(new_f_node, 100);;

    //for (auto key : {"output_shapes", "output_types"})
    //    graph_utils::CopyAttribute(key, org_position_node, &new_f_node);
    //graph_utils::MaybeSetFusedMetadata(first_filter_node, org_node, &new_f_node);

    return new_f_node;
}

std::string GetOutputShapes(const std::string node_str){
    std::string delimiter = "output_shapes=";
    if (node_str.find(delimiter) != std::string::npos) {
        std::string sh = node_str.substr(node_str.find(delimiter), node_str.find("], "));
        sh = sh.erase(0, delimiter.size());
        sh = sh.substr(0, sh.find("], "));
        sh = sh + "]";
        return sh;
    } else {
        return "";
    }
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
int GetOrderCost(const GraphDef& suggested_order, MutableGraphView &graph, std::vector<std::string> &op_types) {
    double cost = 0;

    const NodeDef* m_op = nullptr;
    const NodeDef* b_op = nullptr;
    const NodeDef* f_op = nullptr;
    const NodeDef* next_op = nullptr;
    std::string last_seen;
    
    bool batch_present = false;
    bool map_present = false;
    bool filter_present = false;

    std::vector<NodeDef> changing_dtype = {};
    std::vector<NodeDef> changing_shape = {};

    std::string prev_dtype = "";
    std::string prev_shape = "";

    bool first_one = true;

    for (const NodeDef& node : suggested_order.node()) {
        /*
        VLOG(0) << "########### NODE SUMMARY START ########";
        std::string summary = SummarizeNodeDef(node, 100);
        VLOG(0) << summary;
        std::string dt = GetOutputType(summary);
        std::string sh = GetOutputShapes(summary);
        VLOG(0) << "Output type is: " << dt;
        VLOG(0) << "Output shape is: " << sh;
        VLOG(0) << "########### NODE SUMMARY END ########";

        // Get the node's input and check if dtype/shape is different
        if (!first_one) {
            NodeDef * input_node = graph_utils::GetInputNode(node, graph);
            std::string in_n_sum = SummarizeNodeDef(*input_node, 100);
            std::string in_n_dt = GetOutputType(in_n_sum);
            std::string in_n_sh = GetOutputShapes(in_n_sum);
            if (dt != in_n_dt) {
                changing_dtype.push_back(node);
                VLOG(0) << "Node " << node.name() << " changed dtype!";
            }
            if (sh != in_n_sh) {
                changing_shape.push_back(node);
                VLOG(0) << "Node " << node.name() << " changed shape!";
            }
        } else { // getting input node might not work on edge nodes
            VLOG(0) << "We're probably at an edge node";
        }
        first_one = false;
        */

        //auto dt
        //NodeDef* n_ptr = &node;
        auto op_name = node.op();
        op_types.push_back(op_name);
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
    std::vector<std::string> op_types;
    auto cost = GetOrderCost(sorted_old_graph, graph, op_types);
    VLOG(0) << "Total cost:";
    VLOG(0) << cost;

    VLOG(0) << "Updated graph cost:";

    std::vector<std::string> new_op_types;
    auto new_cost = GetOrderCost(sorted_old_graph, graph, new_op_types);
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

    std::vector<std::string> op_types;
    auto cost = GetOrderCost(sorted_old_graph, graph, op_types);
    VLOG(0) << (std::find(op_types.begin(), op_types.end(), "MapDataset") != op_types.end());
    VLOG(0) << (std::find(op_types.begin(), op_types.end(), "FilterDataset") != op_types.end());

    // Only proceed with optimization if we are in the 'right' pipeline (we see a Filter or Map op)
    if (std::find(op_types.begin(), op_types.end(), "MapDataset") == op_types.end() && std::find(op_types.begin(), op_types.end(), "FilterDataset") == op_types.end()) {
        VLOG(0) << "No reorderable ops found! Not running AutoOrder optimization on this pipeline.";
        return Status::OK();
    }

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

    //auto first_dtype = (*sink_node->mutable_attr())["output_types"];
    //auto first_shape = (*sink_node->mutable_attr())["output_shapes"];

    //VLOG(0) << first_dtype;
    //VLOG(0) << first_shape;

    std::vector<NodeDef*> changing_dtype = {};
    std::vector<NodeDef*> changing_shape = {};

    std::vector<bool> node_changed_dtype = {false, true}; // remember we are going bottom up in the pipeline
    std::vector<bool> node_changed_shape = {false, false};
    
    while (!bfs_queue.empty()) {
        //VLOG(0) << "Trying another one";
        //VLOG(0) << bfs_queue.size();
        NodeDef* current_node = bfs_queue.front();
        VLOG(0) << "Visiting " << current_node->op();
        //VLOG(0) << "Current output_type is " << (*current_node->mutable_attr())["output_types"];
        //VLOG(0) << "Current output_shape is " << (*current_node->mutable_attr())["output_shapes"];
        bfs_queue.pop();
        //VLOG(0) << "popped elem";
        visited.insert(current_node->name());

        // Iterate through the neighbors
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

                    VLOG(0) << "########### NODE SUMMARY START ########";
                    std::string summary = SummarizeNodeDef(*current_node, 100);
                    VLOG(0) << summary;
                    std::string dt = GetOutputType(summary);
                    std::string sh = GetOutputShapes(summary);
                    VLOG(0) << "Output type is: " << dt;
                    VLOG(0) << "Output shape is: " << sh;
                    VLOG(0) << "########### NODE SUMMARY END ########";

                    std::string in_n_sum = SummarizeNodeDef(*neighbor_node, 100);
                    std::string in_n_dt = GetOutputType(in_n_sum);
                    std::string in_n_sh = GetOutputShapes(in_n_sum);
                    if (dt != in_n_dt) {
                        changing_dtype.push_back(current_node);
                        VLOG(0) << "Node " << current_node->name() << " changed dtype!";
                    }
                    if (sh != in_n_sh) {
                        changing_shape.push_back(current_node);
                        VLOG(0) << "Node " << current_node->name() << " changed shape!";
                    }
                    
                } else if (current_node->op().find("FilterDataset") != std::string::npos) {
                    VLOG(0) << "Found Filter node";
                    VLOG(0) << current_node->op();
                    VLOG(0) << current_node->input(0);
                    //(*target->mutable_input())[0] = current_node->input(0);

                    bfs_queue.push(neighbor_node);

                    VLOG(0) << "########### NODE SUMMARY START ########";
                    std::string summary = SummarizeNodeDef(*current_node, 100);
                    VLOG(0) << summary;
                    std::string dt = GetOutputType(summary);
                    std::string sh = GetOutputShapes(summary);
                    VLOG(0) << "Output type is: " << dt;
                    VLOG(0) << "Output shape is: " << sh;
                    VLOG(0) << "########### NODE SUMMARY END ########";

                    // Look into the FunctionDef
                    VLOG(0) << "########### FUNCTION SUMMARY START ########";
                    const auto& filter_pred = current_node->attr().at("predicate");
                    VLOG(0) << "Function name: " << filter_pred.func().name();
                    const FunctionDef* filter_func =
                        function_library.Find(filter_pred.func().name());
                    const auto filter_inputs = fusion_utils::GetFunctionInputs(*filter_func);
                    auto filter_args = filter_func->signature().input_arg();
                    int arg_size = filter_func->signature().input_arg_size();
                    VLOG(0) << "Function has: " << arg_size << " arguments.";
                    for (auto& arg : filter_args) {
                        VLOG(0) << arg.name();
                        VLOG(0) << arg.type();
                    }
                    // Creating a double loop ??
                    /*for (int i = 0; i < arg_size; ++i) {
                        for (auto& arg : filter_args) {
                            VLOG(0) << arg.name();
                        }
                    }*/
                    //VLOG(0) << filter_inputs;

                    VLOG(0) << "########### FUNCTION SUMMARY END ########";

                    std::string in_n_sum = SummarizeNodeDef(*neighbor_node, 100);
                    std::string in_n_dt = GetOutputType(in_n_sum);
                    std::string in_n_sh = GetOutputShapes(in_n_sum);
                    if (dt != in_n_dt) {
                        changing_dtype.push_back(current_node);
                        VLOG(0) << "Node " << current_node->name() << " changed dtype!";
                    }
                    if (sh != in_n_sh) {
                        changing_shape.push_back(current_node);
                        VLOG(0) << "Node " << current_node->name() << " changed shape!";
                    }

                    VLOG(0) << "No. of nodes changing dtype: " << changing_dtype.size();
                    VLOG(0) << "No. of nodes changing shape: " << changing_shape.size();


                    absl::flat_hash_set<string> nodes_to_delete;
                    //VLOG(0) << "Deleting filter node";
                    NodeDef* const parent = graph_utils::GetInputNode(*current_node, graph);
                    VLOG(0) << "Input node is " << parent->op();
                    VLOG(0) << "Cur node is " << current_node->op();
                    VLOG(0) << "Target Node is " << target->op();

                    // Now we update the nodes in the graph
                    // To avoid issues, we make all new from scratch nodes

                    // TODO: Later on the policy should find this order,
                    // for now we just change the order of 2 consecutive ops
                    //std::vector<int> new_order = {1, 0};
                    std::vector<int> new_order = {0, 1}; // Note that this is reverse order (we traverse tree in opposite direction)
                    std::vector<NodeDef*> org_nodes = {current_node, parent}; // Filter, then Map
                    std::vector<NodeDef> new_nodes = {};

                    //for (int j = new_order.size()-1; j >= 0; --j) { // We have to move backwards (each node must be bound with its input)
                    for (int j = 0; j < new_order.size(); ++j) {
                        VLOG(0) << "Making a new " << org_nodes[new_order[j]]->op() << " node";
                        auto* new_node = graph.AddNode(MakeNewNode(*org_nodes[org_nodes.size()-1-j], *org_nodes[new_order[j]], &graph,
                                                                   function_library, output->mutable_library(),
                                                                   node_changed_dtype[new_order[j]], node_changed_shape[new_order[j]]));
                        VLOG(0) << "Added the new node to the graph";
                        new_nodes.insert(new_nodes.begin(), *new_node);
                    }

                    // Link the target node (the one after the last reordered interval) back to the right input
                    (*target->mutable_input())[0] = new_nodes.back().name();

                    VLOG(0) << "Constructed new nodes.";

                    // Update fanouts for each position
                    for (int j = 0; j < org_nodes.size(); ++j) {
                        VLOG(0) << "Updating the Fanout of NEW node " << new_nodes[j].op();
                        TF_RETURN_IF_ERROR(graph.UpdateFanouts(org_nodes[j]->name(), new_nodes[j].name()));
                    }

                    VLOG(0) << "Updated Fanouts.";

                    for (int j = 0; j < org_nodes.size(); ++j) {
                        nodes_to_delete.insert(org_nodes[i]->name());
                    }
                    graph.DeleteNodes(nodes_to_delete);
                    VLOG(0) << "Deleted nodes";


                    VLOG(0) << "(Original) Target node is " << target->op();
                    VLOG(0) << "Target node's input " << target->input(0);
                    VLOG(0) << "Target node's input's input " << graph_utils::GetInputNode(*target, graph)->input(0);
                    VLOG(0) << "Target node's input's input's input " << graph_utils::GetInputNode(*graph_utils::GetInputNode(*target, graph), graph)->input(0); // Should be a map (the next non reordered op)


                    /*
                    auto* new_filter_node = graph.AddNode(MakeNewNode(*parent, *current_node, &graph));
                    TF_RETURN_IF_ERROR(graph.UpdateFanouts(parent->name(), new_filter_node->name()));

                    VLOG(0) << "New node is " << new_filter_node->op();
                    VLOG(0) << "New node's input is " << new_filter_node->input(0);
                    VLOG(0) << "New node's parent is " << graph_utils::GetInputNode(*new_filter_node, graph)->op();

                    // Update output type of Filter node output of 2nd to last map
                    for (auto key : {"output_shapes", "output_types"})
                        graph_utils::CopyAttribute(key, *(graph_utils::GetInputNode(*new_filter_node, graph)), new_filter_node);
                    
                    (*parent->mutable_input())[0] = new_filter_node->name();
                    TF_RETURN_IF_ERROR(graph.UpdateFanouts(current_node->name(), parent->name()));
                    VLOG(0) << "Old nodes test!!!!!!!!";
                    VLOG(0) << "(original) Parent is " << parent->op();
                    VLOG(0) << "Parent's input is " << parent->input(0);

                    VLOG(0) << "Target node is " << target->op();
                    VLOG(0) << "Target node's input " << target->input(0);

                    nodes_to_delete.insert(current_node->name());
                    graph.DeleteNodes(nodes_to_delete);
                    */



                    
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
