#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_SUMMARIZE_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_SUMMARIZE_H_

#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/optimizers/data/optimizer_base.h"

namespace tensorflow {
namespace grappler {
namespace easl {

// This optimizer optimizes the order of operations in the input pipeline.
class Summarize : public TFDataOptimizerBase {
    public:
        Summarize() = default;
        ~Summarize() override = default;

        string name() const override { return "summarize"; };

        bool UsesFunctionLibrary() const override { return false; }

        Status Init(
            const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
                return Status::OK();
            }

        Status ApplyOptimization(MutableGraphView &graph, GraphDef &sorted_old_graph);

        Status OptimizeAndCollectStats(Cluster* cluster, const GrapplerItem& item,
                                       GraphDef* output,
                                       OptimizationStats* stats) override;
};

} // namespace easl
} // namespace grappler
} // namespace tensorflow

#endif // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_SUMMARIZE_H_