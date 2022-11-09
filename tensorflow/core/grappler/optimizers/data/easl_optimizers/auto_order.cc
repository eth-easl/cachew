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

NodeDef MakeFusedFilterNode(const NodeDef& first_filter_node,
                            const NodeDef& second_filter_node,
                            const FunctionDef& fused_function,
                            MutableGraphView* graph) {
    NodeDef fused_node;
    graph_utils::SetUniqueGraphNodeName("fused_filter", graph->graph(),
                                      &fused_node);

    fused_node.set_op("FilterDataset");
    fused_node.add_input(first_filter_node.input(0));

    auto attr = first_filter_node.attr().at("predicate");
    *attr.mutable_func()->mutable_name() = fused_function.signature().name();
    (*fused_node.mutable_attr())["predicate"] = std::move(attr);

    graph_utils::CopyAttribute("Targuments", first_filter_node, &fused_node);

    for (auto key : {"output_shapes", "output_types"})
        graph_utils::CopyAttribute(key, second_filter_node, &fused_node);
    graph_utils::MaybeSetFusedMetadata(first_filter_node, second_filter_node, &fused_node);

    return fused_node;
}

// Calculate the cost of a proposed op ordering
// Cost = sum(Shape(op) * DTypeBytes(op))
int GetOrderCost(const GraphDef& suggested_order) {
    double cost = 0;

    for (const NodeDef& node : suggested_order.node()) {
        //auto dt
        auto op_name = node.op();
        //auto output_s = node.output_size();
        auto input_s = node.input_size();
        double ret_factor = 1.0;
        //double inf_factor = output_s/input_s;


        VLOG(0) << op_name;
        //VLOG(0) << output_s;
        VLOG(0) << input_s;
        //VLOG(0) << inf_factor;
        VLOG(0) << ret_factor;

        //cost+=output_s;
        cost+=input_s*ret_factor;
    }

    return cost;
}

}  // namespace

Status AutoOrder::OptimizeAndCollectStats(Cluster* cluster,
                                          const GrapplerItem& item,
                                          GraphDef* output,
                                          OptimizationStats* stats) {
    VLOG(0) << "OTO: AUTO ORDER BEING APPLIED (Should come before filter fusion)!!!!!!!!!";
    GraphDef sorted_old_graph = item.graph;
    TF_RETURN_IF_ERROR(TopologicalSort(&sorted_old_graph));
    *output = sorted_old_graph;

    MutableGraphView graph(output);
    absl::flat_hash_set<string> nodes_to_delete;
    FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                               output->library());

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

    VLOG(0) << "Original pipline:";
    auto cost = GetOrderCost(sorted_old_graph.node());
    VLOG(0) << "Total cost:";
    VLOG(0) << cost;  

    for (const NodeDef& node : sorted_old_graph.node()) {
        const NodeDef* second_filter_node = get_filter_node(node);
        if (!second_filter_node) continue;

        const NodeDef* first_filter_node =
            get_filter_node(*graph_utils::GetInputNode(*second_filter_node, graph));
        if (!first_filter_node) continue;

        const auto* fused_predicate =
            make_fused_function(first_filter_node, second_filter_node);
        if (!fused_predicate) continue;
        const auto* fused_filter_node = graph.AddNode(MakeFusedFilterNode(
            *first_filter_node, *second_filter_node, *fused_predicate, &graph));

        TF_RETURN_IF_ERROR(graph.UpdateFanouts(second_filter_node->name(),
                                               fused_filter_node->name()));

        TF_RETURN_IF_ERROR(function_library.AddFunctionDef(*fused_predicate));
        nodes_to_delete.insert(first_filter_node->name());
        nodes_to_delete.insert(second_filter_node->name());
        stats->num_changes++;
    }  

    TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));
    return Status::OK();
}

REGISTER_GRAPH_OPTIMIZER_AS(AutoOrder, "auto_order");

}  // namespace easl
}  // namespace grappler
}  // namespace tensorflow
