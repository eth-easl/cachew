//
// Created by otmraz on 11.11.22.
//

#include "tensorflow/core/data/service/easl/ordering_utils.h"

#include <queue>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/data/service/easl/cache_model.h"


namespace tensorflow {
namespace data {
namespace service {
namespace easl {
namespace ordering_utils {

namespace {
int MAX_WORKERS_PER_JOB = 100;

double kMinBatchTimeRelativeImprovementDown = 0.03;
uint32 kInStabilityBeforeScaling = 20;
double kMinQueueSizeRelativeGrowth = 1.5; // +50%
double kMinBatchTimeRelativeGrowth = 1.5; // +50%
}


Status OpOrderUpdate(
    const std::string& job_type,
    const int64 job_id,
    const experimental::DispatcherConfig& dispatcher_config,
    ::tensorflow::data::easl::MetadataStore& metadata_store,
    int64& worker_count) {
  using NodeMetrics = ::tensorflow::data::easl::NodeMetrics;
  using ModelMetrics = ::tensorflow::data::easl::ModelMetrics;

  // Ordering policy
  // 0 == Fixed pipeline (no reordering)
  // 1 == AutoOrder policy

  if(dispatcher_config.order_policy() == 0) {
    VLOG(0) << "Not using AutoOrder Policy."
    metadata_store.UnsetJobIsOrdering(job_id);
    return Status::OK();
  } else if (dispatcher_config.order_policy() == 1)
  {
    VLOG(0) << "Using AutoOrder Policy.";
    metadata_store.SetJobIsOrdering(job_id);

    
    bool is_ordering;
    TF_RETURN_IF_ERROR(metadata_store.IsJobOrdering(job_id, is_ordering));

    std::shared_ptr<ModelMetrics> model_metrics;
    TF_RETURN_IF_ERROR(metadata_store.GetModelMetrics(job_id, model_metrics));

    ModelMetrics::MetricsHistory metrics_history = model_metrics->metrics_history_;
    std::shared_ptr<ModelMetrics::Metrics> last_metrics = metrics_history[metrics_history.size() - 1];
    VLOG(0) << "EASL (AutoOrder policy) - Worker count for last metrics: "
                << metrics_history[metrics_history.size()-1]->worker_count(); // Guaranteed to succeed.

    rwc = std::shared_ptr<ModelMetrics::Metrics> last_metrics = metrics_history[metrics_history.size() - 1];
    if (rwc == 0) {
      // Later we might want to try adding reorder even in full local mode (for higher throughput)
      VLOG(0) << "Already at 0 remote workers, no need to reoder.";
      return Status::OK();
    }
    else{
      VLOG(0) << "Currently using " << rwc << " remote workers.";
      return Status::OK();
    }
  } else {
    VLOG(0) << "Invalid AutoOrder Policy!";
    return errors::InvalidArgument("Order policy ", dispatcher_config.order_policy(), " is invalid. Only order policy 0 and 1 and is valid.");
  }
  
}

} // namespace ordering_utils
} // namespace easl
} // namespace service
} // namespace data
} // namespace tensorflow


