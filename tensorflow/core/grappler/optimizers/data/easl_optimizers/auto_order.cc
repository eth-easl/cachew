#include <queue>
#include <algorithm>
#include <vector>
#include <numeric>

#include "tensorflow/core/grappler/optimizers/data/easl_optimizers/auto_order.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/data/service/easl/dispatcher_order_state.h"
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

std::string GetOutputType(const std::string node_str){
  std::string delimiter = "output_types=";

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

    // Add predicates if present
    // TODO: Check what else different op types contain
    std::string summary = SummarizeNodeDef(org_node, 100);
    VLOG(0) << "Summarized org node";
    VLOG(0) << summary;
    VLOG(0) << "Summarized org_position node";
    VLOG(0) << SummarizeNodeDef(org_position_node, 100);
    VLOG(0) << "Summarized org_position input node";
    VLOG(0) << SummarizeNodeDef(*in_node, 100);

    // Check if the node should be kept in its org position
    if (summary.find("keep_position=") != std::string::npos) {
        const AttrValue& filter_pred = org_node.attr().at("keep_position");
        VLOG(0) << SummarizeAttrValue(filter_pred);
        graph_utils::CopyAttribute("keep_position", org_node, &new_f_node);
    }

    // Check which nodes, the org node actually came from
    NodeDef_ExperimentalDebugInfo debug_i = org_node.experimental_debug_info();
    int num_org_nodes = debug_i.original_node_names_size();
    VLOG(0) << "The original NodeDef was made up from " << num_org_nodes << " nodes.";
    std::vector<std::string> org_names;
    for (int i = 0 ; i < num_org_nodes; ++i) {
        std::string name = debug_i.original_node_names(i);
        VLOG(0) << "Org node " << i << " is called " << name;
        org_names.push_back(name);
    }

    // Add corresponding predicate if present in the original node (i.e. it was a filter node)
    if (summary.find("predicate=") != std::string::npos) {
        VLOG(0) << "Set predicate (a predicate existed)";

        // For now focus on ops that don't change the d_type and merely copy over the predicate as is
        changes_dtype = true;
        if (changes_dtype) {  // The node will figure the types out for itself (hopefully)
            (*new_f_node.mutable_attr())["predicate"] = org_node.attr().at("predicate");
        }
        else {
            // There may have been a change in type, follow the output types of the previous node

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

            const AttrValue& filter_pred = org_node.attr().at("predicate");
            //FunctionDef func_def_direct = (*org_node.mutable_attr())["predicate"].func();
            std::string func_name = filter_pred.func().name();
            VLOG(0) << "Name of filter pred function " << func_name;
            VLOG(0) << "Adjusting filter input dtype!";
            const FunctionDef* filter_func = function_library.Find(func_name);

            // THIS MUST GO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            //FunctionDef* mutable_filter_func = const_cast<FunctionDef*>(filter_func);
            //OpDef* mutable_ff_sig = mutable_filter_func->mutable_signature();

            tensorflow::grappler::fusion_utils::StringCollection filter_inputs = fusion_utils::GetFunctionInputs(*filter_func);
            // filter_func->signature().input_arg() is of type: const OpDef
            const OpDef ff_sig_const = filter_func->signature();
            // Another BAD ONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            //OpDef ff_sig = (OpDef)ff_sig_const;

            auto filter_args = filter_func->signature().input_arg();
            int in_arg_size = ff_sig_const.input_arg_size();
            VLOG(0) << "There are " << in_arg_size << " arguments.";

            //VLOG(0) << "ORIGINAL ArgDef summary:";
            //std::string arg_sum = SummarizeArgs(filter_args);
            //VLOG(0) << arg_sum;

            VLOG(0) << "ORIGINAL OpDef summary:";
            std::string op_sum = SummarizeOpDef(ff_sig_const);
            VLOG(0) << op_sum;

            const FunctionDef* org_func = function_library.Find(org_node.attr().at("predicate").func().name());
            OpDef org_func_sig = org_func->signature();

            // NEW TRY (WILDCARD COPY)
            VLOG(0) << "Performing WildCard copy";
            FunctionDef setup_ff = *org_func;

            FunctionDef* real_f = library->add_function();

            // Give the function a new name (to avoid conflicts)
            StringPiece wc_func_prefix = "wc_reo_func";
            graph_utils::SetUniqueGraphFunctionName(wc_func_prefix, library, &setup_ff);
            VLOG(0) << "Set new function's name to " << setup_ff.signature().name();



            // Set the 'real_f' function's signature
            *real_f->mutable_signature() = setup_ff.signature();

            // Name the 'real_f' function
            StringPiece wc_real_f_prefix = "wc_real_reo_func";
            graph_utils::SetUniqueGraphFunctionName(wc_real_f_prefix, library, real_f);
            VLOG(0) << "Set REAL new function's name to " << real_f->signature().name();

            // RenameFunctionNodes ??

            *real_f->mutable_ret() = setup_ff.ret();

            // Other stuff from fusion_utils ??

            // Setting construction ctx OR metadata
            auto get_construction_context = [](const FunctionDef& func) {
              auto iter = func.attr().find("_construction_context");
              if (iter == func.attr().cend()) return std::string();
              return iter->second.s();
            };
            std::string org_construction_context = get_construction_context(*org_func);
            if (!org_construction_context.empty()) {
                VLOG(0) << "The orignial function had a construction context, setting it to the 'real_f' function.";
                (*real_f->mutable_attr())["_construction_context"].set_s(
                    org_construction_context);
            }

            graph_utils::MaybeSetFusedMetadata(org_node, org_node,
                                               &new_f_node);

            DataType dt; // This might be dangerous

            // Fix the input arg types here !!!
            for (int i = 0; i < in_arg_size; ++i) {
                // Figure out the CORRECT input type

                std::string substr_to_remove = "DT_";
                std::size_t substr_loc =
                    out_type_strings[i].find(substr_to_remove);
                if (substr_loc != std::string::npos) {
                    out_type_strings[i].erase(substr_loc,
                                            substr_to_remove.size());
                }

                // out_type_strings[i].erase(std::remove(out_type_strings[i].begin(), out_type_strings[i].end(), '_'), out_type_strings[i].end());
                std::transform(out_type_strings[i].begin(),
                               out_type_strings[i].end(),
                               out_type_strings[i].begin(), ::tolower);
                VLOG(0) << "Output " << i << " is of type "
                        << out_type_strings[i];
                bool worked = DataTypeFromString(out_type_strings[i], &dt);
                VLOG(0) << "Output has 'DataType' " << dt;

                // Set dt to the respective input arg
                //OpDef_ArgDef& mutable_in_arg = *setup_ff.mutable_signature()->mutable_input_arg(i);
                OpDef_ArgDef& mutable_in_arg = *real_f->mutable_signature()->mutable_input_arg(i);


                VLOG(0) << "Original arg name (shouldn't change this): " << mutable_in_arg.name();
                VLOG(0) << "Original arg type (to be changed): " << mutable_in_arg.type();
                mutable_in_arg.set_type(dt);
                VLOG(0) << "New arg type is: " << mutable_in_arg.type();
            }

            // From FilterFusion (This will cause errors earlier on) DO NOT USE!
            //(*real_f->mutable_attr())[data::kTFDataFunction].set_b(true);

            // All inputs of real_f should be set now. Add the fake sink nodes
            //AddFakeSinksV2(real_f, org_func, dt);

            AttrValue org_attr = org_node.attr().at("predicate");
            VLOG(0) << "Original summary of attr " << SummarizeAttrValue(org_attr);
            VLOG(0) << "Original node used function " << org_attr.func().name();
            //*org_attr.mutable_func()->mutable_name() = setup_ff.signature().name();
            *org_attr.mutable_func()->mutable_name() = real_f->signature().name();
            VLOG(0) << "New summary of attr ";
            VLOG(0) << SummarizeAttrValue(org_attr);

            (*new_f_node.mutable_attr())["predicate"] = org_attr;
            VLOG(0) << "Summary of new node's 'predicate' attribute:";
            VLOG(0) << SummarizeAttrValue(new_f_node.attr().at("predicate"));

            VLOG(0) << "Summary of NEW OpDef";
            VLOG(0) << SummarizeOpDef(real_f->signature());

            //AttrValue attr = new_f_node.attr().at("predicate");
            //VLOG(0) << "Previously function used was " << org_func->signature().name();
            //*attr.mutable_func()->mutable_name() = setup_ff.signature().name();
            //(*setup_ff.mutable_attr())["predicate"] = std::move(attr);
            //VLOG(0) << "Now we use function " << setup_ff.attr().at("predicate").func().name();

            //function_library.AddFunctionDef(setup_ff);
            function_library.AddFunctionDef(*real_f);

            VLOG(0) << "Summary of 'predicate attribute':";
            VLOG(0) << SummarizeAttrValue(new_f_node.attr().at("predicate"));
            // END TRY (WILDCARD COPY)


            if (false) {
                // This function will be used as a clone of the original function, having unique names.
                FunctionDef setup_function = *org_func;

                // We aren't fusing anything, so let's just keep the names the same as much as possible
                OpDef signature;
                signature.set_name(setup_function.signature().name());

                for (int i = 0; i < in_arg_size; ++i) {
                  // Make a new (mutable) input arg
                  const OpDef_ArgDef& input_arg = org_func_sig.input_arg(i);
                  auto& input = *signature.add_input_arg();
                  input = input_arg;
                  input.set_name(input.name());

                  // Figure out the CORRECT input type
                  DataType dt;
                  std::string substr_to_remove = "DT_";
                  std::size_t substr_loc =
                      out_type_strings[i].find(substr_to_remove);
                  if (substr_loc != std::string::npos) {
                    out_type_strings[i].erase(substr_loc,
                                              substr_to_remove.size());
                  }

                  // out_type_strings[i].erase(std::remove(out_type_strings[i].begin(), out_type_strings[i].end(), '_'), out_type_strings[i].end());
                  std::transform(out_type_strings[i].begin(),
                                 out_type_strings[i].end(),
                                 out_type_strings[i].begin(), ::tolower);
                  VLOG(0) << "Output " << i << " is of type "
                          << out_type_strings[i];
                  bool worked = DataTypeFromString(out_type_strings[i], &dt);
                  VLOG(0) << "Output has 'DataType' " << dt;

                  // Set the right type
                  VLOG(0) << "Type was " << input.type() << "!";
                  input.set_type(dt);
                  VLOG(0) << "Type has been adjusted to " << input.type()
                          << "!";
                }

                for (const auto& output_arg : org_func_sig.output_arg()) {
                  auto& output = *signature.add_output_arg();
                  output = output_arg;
                  output.set_name(
                      output.name());  // Last 2 lines are probably useless??
                }

                *setup_function.mutable_signature() = signature;

                FunctionDef* new_function = library->add_function();

                // fusion_utils::SameSignature(org_func.signature(), setup_function.signature(),
                //               new_function->mutable_signature());
                *new_function->mutable_signature() =
                    *setup_function.mutable_signature();

                StringPiece func_name_prefix = "reordered_func";
                graph_utils::SetUniqueGraphFunctionName(func_name_prefix,
                                                        library, new_function);

                // set_output(first_function.ret(), setup_function.ret(),
                //            fused_function->mutable_ret());
                *new_function->mutable_ret() =
                    org_func->ret();  // TODO: CHECK THIS ONE!

                auto attr = new_f_node.attr().at("predicate");
                *attr.mutable_func()->mutable_name() =
                    new_function->signature().name();
                (*new_f_node.mutable_attr())["predicate"] = std::move(attr);

                VLOG(0) << "New summary of AttrValue ('predicate') ";
                std::string pred_sum = SummarizeAttrValue(attr);
                VLOG(0) << pred_sum;
            }

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

NodeDef MakeNewNodeV2(const NodeDef& org_position_node,
                      const NodeDef& org_node,
                      MutableGraphView* graph) {
    return org_position_node;
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
            b_op = &node;
        }
        if (op_name.find("MapDataset") != std::string::npos) {
            m_op = &node;
        }
        if (op_name.find("FilterDataset") != std::string::npos) {
            f_op = &node;
        }
        last_seen = op_name;

        if (op_name != "Const" && op_name != "SparseToDense" && op_name != "Identity" &&
            op_name != "StridedSlice" && op_name != "GatherV2" && op_name != "Pack" &&
            op_name != "AddV2" && op_name != "Reshape") {
            VLOG(0) << op_name;
        }

        //VLOG(0) << output_s;
        VLOG(1) << input_s;
        //VLOG(0) << inf_factor;
        //VLOG(0) << ret_factor;

        //cost+=output_s;
        cost+=input_s*ret_factor;
        
        last_seen = op_name;
    }

    return cost;
}

Status GetGraphNodesOfInterest(const GraphDef& sorted_old_graph, std::vector<std::string> graph_nodes_of_interest) {
    VLOG(0) << "Collecting nodes of interest from the computation graph";
    for (const NodeDef& node : sorted_old_graph.node()) {
        std::string op_name = node.op();
        if (op_name == "MapDataset" || op_name == "ParallelMapDatasetV2" || op_name == "ParallelMapDataset") {
            VLOG(0) << "Found an interesting node in the graph " << node.name();
            graph_nodes_of_interest.push_back(node.name());
        }
    }
    return Status::OK();
}

bool NodeChangesTypeOrDims(NodeDef node, GraphDef &sorted_old_graph, MutableGraphView &graph) {
    std::string prev_node_name = node.input(0);
    int prev_idx = graph_utils::FindGraphNodeWithName(prev_node_name, sorted_old_graph);
    NodeDef* prev_node = sorted_old_graph.mutable_node(prev_idx);

    std::string prev_out_types = GetOutputType(SummarizeNodeDef(*prev_node, 100));
    VLOG(0) << "Previous output types were " << prev_out_types;
    std::string new_out_types = GetOutputType(SummarizeNodeDef(node, 100));
    VLOG(0) << "New output types were " << new_out_types;

    if (prev_out_types == new_out_types) {
        return false;
    } else {
        return true;
    }
}

Status GetReorderableIntervals(std::vector<std::string> graph_nodes_of_interest, std::vector<std::vector<std::string>> reorderable_intervals,
                               std::vector<float> inflation_factors, std::vector<std::vector<float>> reorderable_interval_inf_factors,
                               GraphDef &sorted_old_graph, MutableGraphView &graph) {
    VLOG(0) << "Collecting nodes of interest from the computation graph";
    std::vector<std::string> cur_interval;
    std::vector<float> cur_if;
    for (int i = 0; i < graph_nodes_of_interest.size(); ++i) {
        int idx = graph_utils::FindGraphNodeWithName(graph_nodes_of_interest[i], sorted_old_graph);
        NodeDef* cur_node = sorted_old_graph->mutable_node(idx);
        //NodeDef cur_node = graph_nodes_of_interest[i];
        std::string keep_pos_attr = SummarizeAttrValue(cur_node->attr().at("keep_position"));
        VLOG(0) << "Current node is " << cur_node->name();
        VLOG(0) << "keep_position is " << keep_pos_attr;

        bool changes_type_or_dimensions = NodeChangesTypeOrDims(*cur_node, sorted_old_graph, graph);

        if (keep_pos_attr == "true") {
            VLOG(0) << "keep_position was true, do not reorder";
            if (cur_interval.size() > 1) {
                std::vector<std::string> new_interval = cur_interval;
                std::vector<float> new_inf_factors = cur_if;
                reorderable_intervals.push_back(new_interval);
                reorderable_interval_inf_factors.push_back(new_inf_factors);
            }
            cur_interval.clear();
            cur_if.clear();
        } else if (changes_type_or_dimensions) {
            VLOG(0) << "The node changes the type or dimensions, do not reorder";
            if (cur_interval.size() > 1) {
                std::vector<std::string> new_interval = cur_interval;
                reorderable_intervals.push_back(new_interval);
            }
            cur_interval.clear();
            cur_if.clear();
        } else {
            VLOG(0) << "We may reorder this node";
            cur_interval.push_back(cur_node->name());
            cur_if.push_back(inflation_factors[i]);
        }
    }
    if (cur_interval.size() > 1) {
        reorderable_intervals.push_back(cur_interval);
        reorderable_interval_inf_factors.push_back(cur_if);
    }
    VLOG(0) << "In total there are " << reorderable_intervals.size() << " reorderable intervals, "
            << reorderable_interval_inf_factors.size() << " inflation factor intervals";
    return Status::OK();
}

Status GetIdealIntervalOrders(std::vector<std::vector<float>> reorderable_interval_inf_factors,
                              std::vector<std::vector<int>> ideal_interval_orders) {

    VLOG(0) << "Getting the ideal interval orders";
    for (int i = 0; i < reorderable_interval_inf_factors.size(); ++i) {
        for (int j = 0; j < reorderable_interval_inf_factors[i].size(); ++j) {
            // Ignore slightly noisy metrics, where inflation/deflation is < 5 %
            if (reorderable_interval_inf_factors[i][j] > 0.95 && reorderable_interval_inf_factors[i][j] < 1.05) {
                reorderable_interval_inf_factors[i][j] = 1;
            }
        }
        std::vector<float> cur_ifs = reorderable_interval_inf_factors[i];

        std::vector<int> V(cur_ifs.size());
        std::iota(V.begin(), V.end(), 0);
        std::sort( V.begin(), V.end(), [&](int i,int j) {return cur_ifs[i]<cur_ifs[j];} );

        ideal_interval_orders.push_back(V);

        VLOG(0) << "Ideal pipeline for this interval is:";
        for (int j = 0; j < V.size(); ++j) {
            VLOG(0) << V[j];
        }
    }

    return Status::OK();

}

Status FixIntervalOrder(std::vector<std::string> node_names, std::vector<int> desired_order,
                        MutableGraphView &graph, GraphDef &sorted_old_graph) {
    VLOG(0) << "Fixing new interval";

    for (int i = 0; i < node_names.size(); ++i) {
        VLOG(0) << "Moving the " << i << ". (new order) node in interval";
        int idx = graph_utils::FindGraphNodeWithName(node_names[i], sorted_old_graph);
        NodeDef* org_position_node = sorted_old_graph->mutable_node(idx);
        int idx2 = graph_utils::FindGraphNodeWithName(node_names[desired_order[i]], sorted_old_graph);
        NodeDef* org_node = sorted_old_graph->mutable_node(idx2);
        NodeDef new_node = MakeNewNodeV2(*org_position_node, *org_node, graph);
    }

    return Status::OK();
}

}  // namespace

Status AutoOrder::ApplyOptimization(MutableGraphView &graph, GraphDef &sorted_old_graph) {
    VLOG(0) << "In AutoOrder::ApplyOptimization";

    VLOG(0) << "Original pipline:";
    std::vector<std::string> op_types;
    auto cost = GetOrderCost(sorted_old_graph, graph, op_types);
    VLOG(0) << "Total cost:";
    VLOG(0) << cost;

    //std::vector<NodeDef> graph_nodes_of_interest;
    std::vector<std::string> graph_nodes_of_interest;
    Status s1 = GetGraphNodesOfInterest(sorted_old_graph, graph_nodes_of_interest);

    // TODO: Fetch metrics from database
    std::vector<std::string> metrics_nodes_of_interest = {"ParallelMapV2(id:2)",
                "ParallelMapV2(id:3)", "Map(id:4)", "ParallelMapV2(id:6)",
                "ParallelMapV2(id:9)"};
    std::vector<float> inflation_factors = {0.319971, 0.0649351, 1.0, 0.988674,
                                            1.33259};

    VLOG(0) << "There were " << graph_nodes_of_interest.size() << " interesting nodes in the computation graph, "
        << metrics_nodes_of_interest.size() << " interesting nodes with collected metrics";

    std::vector<std::vector<std::string>> reorderable_intervals;
    std::vector<std::vector<float>> reorderable_interval_inf_factors;
    Status s2 = GetReorderableIntervals(graph_nodes_of_interest, reorderable_intervals,
                                        inflation_factors, reorderable_interval_inf_factors,
                                        sorted_old_graph, graph);

    std::vector<std::vector<int>> ideal_interval_orders;
    Status s3 = GetIdealIntervalOrders(reorderable_interval_inf_factors, ideal_interval_orders);

    VLOG(0) << "Computed ideal interval orders: " << ideal_interval_orders.size() << " in total";

    for (int i = 0; i < ideal_interval_orders.size(); ++i) {
        FixIntervalOrder(reorderable_intervals[i], ideal_interval_orders[i], graph, sorted_old_graph);
    }

    /*VLOG(0) << "Updated graph cost:";

    std::vector<std::string> new_op_types;
    auto new_cost = GetOrderCost(sorted_old_graph, graph, new_op_types);
    VLOG(0) << "Total cost:";
    VLOG(0) << new_cost;
    
    Status s = IsPipelineOk(sorted_old_graph, graph);
    while (!s.ok()) {
        // Choose next best suggestion
        VLOG(0) << "Updating suggestion";
        s = IsPipelineOk(sorted_old_graph, graph);
    }*/

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
    FunctionLibraryDefinition function_library(OpRegistry::Global(), output->library());

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

    std::vector<NodeDef> nodes_of_interest;
    
    while (!bfs_queue.empty()) {
        //VLOG(0) << "Trying another one";
        //VLOG(0) << bfs_queue.size();
        NodeDef* current_node = bfs_queue.front();
        std::string op = current_node->op();
        //VLOG(0) << "Visiting " << op;

        if ((op.find("ParallelMap") != std::string::npos) ||
            (op.find("Filter") != std::string::npos) ||
            (op.find("Prefetch") != std::string::npos) ||
            (op.find("ParallelInterleave") != std::string::npos)) {
                nodes_of_interest.push_back(*current_node);
        }

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

    for (int i = 0; i < nodes_of_interest.size(); ++i) {
        VLOG(0) << nodes_of_interest[i].name();
    }
    //tensorflow::data::OrderState::AddOrgPipeline(nodes_of_interest);

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
