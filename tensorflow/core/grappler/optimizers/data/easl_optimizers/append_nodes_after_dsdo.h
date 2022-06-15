//
// Created by Muyu Li on 15.06.22.
//

#ifndef ML_INPUT_DATA_SERVICE_APPEND_NODES_AFTER_DSDO_H
#define ML_INPUT_DATA_SERVICE_APPEND_NODES_AFTER_DSDO_H

#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/optimizers/data/optimizer_base.h"

namespace tensorflow {
namespace grappler {
namespace easl {

class AppendNodesAfterDSDO: public TFDataOptimizerBase {
public:
    AppendNodesAfterDSDO() = default;

    ~AppendNodesAfterDSDO() override = default;

    string name() const override { return "append_nodes_after_dsdo"; };

    bool UsesFunctionLibrary() const override { return false; };

    Status Init(
            const tensorflow::RewriterConfig_CustomGraphOptimizer *config) override {
      return Status::OK();
    }

    Status OptimizeAndCollectStats(Cluster *cluster, const GrapplerItem &item,
                                   GraphDef *output,
                                   OptimizationStats *stats) override;
}
};

} // namespace easl
} // namespace grappler
} // namespace tensorflow

#endif //ML_INPUT_DATA_SERVICE_APPEND_NODES_AFTER_DSDO_H
