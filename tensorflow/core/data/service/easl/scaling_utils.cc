//
// Created by aymond on 26.11.21.
//

#include "tensorflow/core/data/service/easl/scaling_utils.h"

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
namespace scaling_utils {

namespace {
int MAX_WORKERS_PER_JOB = 100;

// These costs are per hour; reducing them to ms is constant so we can leave them like this
double WORKER_COST = 0.31;
double CLIENT_COST = 8.0;

double kMinBatchTimeRelativeImprovementUp = 0; // This should be in the range [0, inf)
double kMinBatchTimeRelativeImprovementDown = 0; // This should be in the range (-inf, 0]
uint32 kInStabilityBeforeScaling = 20;
double kMinQueueSizeRelativeGrowth = 1.5; // +50%
double kMinBatchTimeRelativeGrowth = 1.5; // +50%
double kScaleChangeFraction = 0.4; // 10%
}


Status DynamicWorkerCountUpdate(
    const std::string& job_type,
    const int64 job_id,
    const experimental::DispatcherConfig& dispatcher_config,
    ::tensorflow::data::easl::MetadataStore& metadata_store,
    int64& worker_count) {
  using NodeMetrics = ::tensorflow::data::easl::NodeMetrics;
  using ModelMetrics = ::tensorflow::data::easl::ModelMetrics;

  // Give out max number of workers
  if(dispatcher_config.scaling_policy() == 2){
    metadata_store.UnsetJobIsScaling(job_id);
    worker_count = MAX_WORKERS_PER_JOB;
    return Status::OK();
  }

  VLOG(0) << "EASL (DynamicWorkerCountUpdate) - Entering.";

  bool is_scaling;
  TF_RETURN_IF_ERROR(metadata_store.IsJobScaling(job_id, is_scaling));

  std::shared_ptr<ModelMetrics> model_metrics;
  TF_RETURN_IF_ERROR(metadata_store.GetModelMetrics(job_id, model_metrics));

  ModelMetrics::MetricsHistory metrics_history = model_metrics->metrics_history_;
  VLOG(0) << "EASL (DynamicWorkerCountUpdate) - Worker count for last metrics: "
               << metrics_history[metrics_history.size()-1]->worker_count(); // Guaranteed to succeed.


  std::shared_ptr<ModelMetrics::Metrics> last_metrics =
      metrics_history[metrics_history.size() - 1];
  int64 current_target_worker_count;
  TF_RETURN_IF_ERROR(metadata_store.GetJobTargetWorkerCount(job_id,
    current_target_worker_count));
  if (last_metrics->worker_count() != current_target_worker_count) {
    VLOG(0) << "EASL (DynamicWorkerCountUpdate) - Target metrics count not fulfilled:\n"
            << " > target: " << current_target_worker_count << "\n"
            << " > actual: " << last_metrics->worker_count();
    worker_count = current_target_worker_count;
    return Status::OK();
  }

  if(is_scaling) {
    VLOG(0) << "EASL (DynamicWorkerCountUpdate) - is_scaling is true";
    if (metrics_history.size() == 1) { // Cannot be smaller than 1
      VLOG(0) << "EASL (DynamicWorkerCountUpdate) - no metrics_history -> "
                   << "increasing worker count";
      worker_count = metrics_history.back()->worker_count() + 1;
      metadata_store.SetJobTargetWorkerCount(job_id, worker_count);
      return Status::OK();
    }

    // Also ensures the right metrics are picked for the performance code
    int second_to_last_index = metrics_history.size() - 2;
    std::shared_ptr<ModelMetrics::Metrics> second_to_last_metrics =
      metrics_history[second_to_last_index];
    while(second_to_last_metrics->worker_count() == last_metrics->worker_count()) {
      if (second_to_last_index == 0) {
        VLOG(0) << "EASL (DynamicWorkerCountUpdate) - Should not enter here!"
                << "This leads to an infinite loop!\n"
                << " > Converging here since scaling is not justified.";

        worker_count = last_metrics->worker_count();
        model_metrics->converged_metrics_ = last_metrics;
        metadata_store.UnsetJobIsScaling(job_id);
        metadata_store.ResetSameScaleCounter(job_id);

        metadata_store.SetJobTargetWorkerCount(job_id, worker_count);
        return Status::OK();
      }
      second_to_last_metrics = metrics_history[--second_to_last_index];
    }

    // We are going to compute the cost values for the policy
    uint32 client_count = model_metrics->GetClientCount();

    if (client_count == 0) {
      VLOG(0) << "(DynamicWorkerCountUpdate) Client count cannot be 0!";
      return errors::FailedPrecondition(
          "(DynamicWorkerCountUpdate) Client count cannot be 0!");
    }

    // The cost of the last scale
    double l_cost = last_metrics->last_x_batch_time_ms() * (client_count
        * CLIENT_COST + last_metrics->worker_count() * WORKER_COST);
    // The cost of the second to last scale
    double stl_cost = second_to_last_metrics->last_x_batch_time_ms()
        * (client_count * CLIENT_COST + second_to_last_metrics->worker_count()
        * WORKER_COST);

    // See if there is any improvement
    double relative_improvement = stl_cost - l_cost;

    if (stl_cost <= 0 || l_cost <= 0) {
      VLOG(0) << "(DynamicWorkerCountUpdate) Instability in scaling "
                   << "cost was negative\n"
                   << " > stl_cost = " << stl_cost
                   << " > l_cost = " << l_cost;
      worker_count = current_target_worker_count;
      return Status::OK();
    }

    if (second_to_last_metrics->worker_count() < last_metrics->worker_count()) {
      // We are scaling up
      if (relative_improvement > kMinBatchTimeRelativeImprovementUp) {
        worker_count = last_metrics->worker_count() + 1;
        VLOG(0) << "(EASL::DynamicWorkerCountUpdate::ScalingUp) "
                << "Improvement large enough:\n"
                << " > improvement [$]: " << relative_improvement << "\n"
                << " > next worker count: " << worker_count;
      } else {
        worker_count = second_to_last_metrics->worker_count();
        model_metrics->converged_metrics_ = second_to_last_metrics;
        metadata_store.UnsetJobIsScaling(job_id);
        metadata_store.ResetSameScaleCounter(job_id);
        VLOG(0) << "(EASL::DynamicWorkerCountUpdate::ScalingUp) "
                << "Improvement NOT large enough:\n"
                << " > improvement: " << relative_improvement << "\n"
                << " > next worker count: " << worker_count;
      }
    } else {
      // We are scaling down; cost should not increase
      if (relative_improvement > kMinBatchTimeRelativeImprovementDown
        && last_metrics->worker_count() > 1) {
        worker_count = last_metrics->worker_count() - 1;
        VLOG(0) << "(EASL::DynamicWorkerCountUpdate::ScalingDown) "
                << "Improvement loss ok:\n"
                << " > improvement: " << relative_improvement << "\n"
                << " > next worker count: " << worker_count;
      } else {
        // Case I: last_metrics = n - 1; stl_metrics = n; n != 2 (lowering scale is bad for perf)
        // Case II: last_metrics = 1; stl_metrics = 2 (here it's not clear if 2 is too much); should check relative perf
        auto relevant_metrics = relative_improvement
            > kMinBatchTimeRelativeImprovementDown ?
            last_metrics : second_to_last_metrics;

        worker_count = relevant_metrics->worker_count();
        model_metrics->converged_metrics_ = relevant_metrics;
        metadata_store.UnsetJobIsScaling(job_id);
        metadata_store.ResetSameScaleCounter(job_id);
        VLOG(0) << "(EASL::DynamicWorkerCountUpdate::ScalingDown) "
                << "Improvement loss NOT ok (or only 1 worker left):\n"
                << " > improvement [$]: " << relative_improvement << "\n"
                << " > next worker count: " << worker_count;
      }
    }
    metadata_store.SetJobTargetWorkerCount(job_id, worker_count);
    return Status::OK();
  } else {
    // We're not scaling --> We're in a period of stability
    uint64 same_scale_counter;
    TF_RETURN_IF_ERROR(metadata_store.IncrementSameScaleCounter(job_id,
      same_scale_counter));

    if (same_scale_counter > kInStabilityBeforeScaling) {
      VLOG(0) << "(EASL::DynamicWorkerCountUpdate::StablePeriod) - "
          << "Checking for potential rescaling after stable period.";
      metadata_store.ResetSameScaleCounter(job_id);
      uint32 client_count = model_metrics->GetClientCount();
      std::shared_ptr<ModelMetrics::Metrics> converged_metrics = model_metrics->converged_metrics_;
      std::shared_ptr<ModelMetrics::Metrics> last_metrics = metrics_history.back();

      // The cost of the last scale
      double l_cost = last_metrics->last_x_batch_time_ms() * (client_count
          * CLIENT_COST + last_metrics->worker_count() * WORKER_COST);
      // The cost of the converged scale
      double c_cost = converged_metrics->last_x_batch_time_ms()
          * (client_count * CLIENT_COST + converged_metrics->worker_count()
          * WORKER_COST);
      double relative_improvement = c_cost - l_cost;
      double threshold = c_cost * kScaleChangeFraction;

      double queue_size_converged = converged_metrics->result_queue_size();
      double queue_size_last_metrics = last_metrics->result_queue_size();
      double relative_queue_size = queue_size_last_metrics / queue_size_converged;

      VLOG(0) << "(EASL::DynamicWorkerCountUpdate::StablePeriod) - "
          << "relative_queue_size: " << relative_queue_size << "\n"
          << "queue_size_converged: " << queue_size_converged << "\n"
          << "queue_size_last_metrics: " << queue_size_last_metrics << "\n";

      double converged_batch_time = converged_metrics->last_x_batch_time_ms();
      double l_batch_time = last_metrics->last_x_batch_time_ms();
      double relative_batch_time = l_batch_time / converged_batch_time;

      VLOG(0) << "(EASL::DynamicWorkerCountUpdate::StablePeriod) - "
              << "relative_batch_time: " << relative_batch_time << "\n"
              << "converged_batch_time: " << converged_batch_time << "\n"
              << "l_batch_time: " << l_batch_time << "\n";

      // Current deployment is much cheaper than before; maybe we can go down
      // Going up here can only increase the costs (as only batch time could have changed)
      if (relative_improvement > threshold
          && converged_metrics->worker_count() > 1) {
        VLOG(0) << "Triggering downscale";
        worker_count = converged_metrics->worker_count() - 1;
        metadata_store.SetJobTargetWorkerCount(job_id, worker_count);
        metadata_store.SetJobIsScaling(job_id);
        return Status::OK();
      }


      // Current deployment is more expensive than before; maybe we can go up
      if (-relative_improvement > threshold) {
        VLOG(0) << "Triggering upscale";
        worker_count = converged_metrics->worker_count() + 1;
        metadata_store.SetJobTargetWorkerCount(job_id, worker_count);
        metadata_store.SetJobIsScaling(job_id);
        return Status::OK();
      }

      VLOG(0) << "No rescale triggered";

      worker_count = last_metrics->worker_count();
      return Status::OK();
    }
  }

  // TODO: Consider setting the reference metrics for the next scale when moving
  //       up or down. This ensures the analysis is uniform.

  // Note: One will end up here in the iteration immediately after convergence
  //       when it's highly likely that the metrics pertain to the scale prior
  //       to convergence. You also end up here when in stability without trying
  //       to rescale.
  metadata_store.GetJobTargetWorkerCount(job_id, worker_count);
  return Status::OK();
}



} // namespace scaling_utils
} // namespace easl
} // namespace service
} // namespace data
} // namespace tensorflow


