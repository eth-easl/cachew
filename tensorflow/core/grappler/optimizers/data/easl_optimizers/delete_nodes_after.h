//
// Created by Muyu Li on 09.06.22.
//

#ifndef ML_INPUT_DATA_SERVICE_DELETE_NODES_AFTER_H
#define ML_INPUT_DATA_SERVICE_DELETE_NODES_AFTER_H

#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/optimizers/data/optimizer_base.h"


namespace tensorflow {
namespace grappler {
namespace easl {

// used for pipeline split
class DeleteNodesAfter: public TFDataOptimizerBase {
public:
    DeleteNodesAfter() = default;
    ~DeleteNodesAfter() override = default;

    string name() const override { return "delete_nodes_after"; }

    bool UsesFunctionLibrary() const override { return false; }

    Status Init(
            const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
      config_ = *config;
      return Status::OK();
    }
// Deprecated
//    void PrintChainOfGraph(NodeDef* sink_node,
//                           GraphDef* output,
//                           int64 split_node_index);

    void BFSGraph(NodeDef* sink_node,
                 GraphDef* output);

    Status ApplyOptimization(MutableGraphView &graph,
                             NodeDef* sink_node,
                             GraphDef *output);

    Status OptimizeAndCollectStats(Cluster* cluster,
                                   const GrapplerItem& item,
                                   GraphDef* output,
                                   OptimizationStats* stats) override;

private:
    tensorflow::RewriterConfig_CustomGraphOptimizer config_;
};

} // easl
} // grappler
} // easl

#endif //ML_INPUT_DATA_SERVICE_DELETE_NODES_AFTER_H
