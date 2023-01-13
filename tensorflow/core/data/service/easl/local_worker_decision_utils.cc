//
// Created by Muyu Li on 29.05.22.
//

#include "local_worker_decision_utils.h"
#include <queue>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/data/service/easl/cache_model.h"
#include "tensorflow/core/data/service/dispatcher_state.h"

namespace tensorflow {
namespace data {
namespace service {
namespace easl {
namespace local_worker_decision {

namespace {
    int MAX_WORKERS_PER_JOB = 100;
    double kMinBatchTimeRelativeImprovementDown = 0.03;
    uint32 kInStabilityBeforeScaling = 20;
    double kMinQueueSizeRelativeGrowth = 1.5; // +50%
    double kMinBatchTimeRelativeGrowth = 1.5; // +50%

    // Why do we have these worker count limits??
    int MAX_LOCAL_WORKERS_PER_JOB = 100; //5;
    int MAX_REMOTE_WORKERS_PER_JOB = 100; //8;
    double kPerformanceErrorBar = 0.10;
    // combine with costs
    double kPerformanceDecreaseTolerance = 0.10;
}

void debug_print_local_remote(std::string debug_string, int64 remote_worker_count, int64 local_worker_count) {
  VLOG(0) << "(EASL::DynamicWorkerCountUpdateWithLocal)::" << debug_string
          << "; Current Remote: " << remote_worker_count
          << "; Current Local: " << local_worker_count;
}

Status DynamicWorkerCountUpdateWithLocal_INCDEC(
        const std::string& job_type,
        const int64 job_id,
        const experimental::DispatcherConfig& dispatcher_config,
        ::tensorflow::data::easl::MetadataStore& metadata_store,
        int64& remote_worker_count,
        int64& local_worker_count,
        const int64 available_workers) {
  // Entering this function means we're choosing the right policy
  using NodeMetrics = ::tensorflow::data::easl::NodeMetrics;
  using ModelMetrics = ::tensorflow::data::easl::ModelMetrics;
  using JobScalingState = ::tensorflow::data::easl::JobScalingState;

  VLOG(1) << "MUYU (DynamicWorkerCountUpdateWithLocal_INCDEC) - Entering.";

  std::shared_ptr<ModelMetrics> model_metrics;
  TF_RETURN_IF_ERROR(metadata_store.GetModelMetrics(job_id, model_metrics));

  ModelMetrics::MetricsHistory metrics_history = model_metrics->metrics_history_;
  std::shared_ptr<ModelMetrics::Metrics> last_metrics = metrics_history[metrics_history.size() - 1];

  VLOG(0) << "MUYU (DynamicWorkerCountUpdateWithLocal_INCDEC) - Worker count for last metrics: "
          << "Remote: "
          << last_metrics->remote_worker_count()
          << "; Local: "
          << last_metrics->local_worker_count(); // Guaranteed to succeed.

  int64 current_target_remote_worker_count, current_target_local_worker_count;
  TF_RETURN_IF_ERROR(metadata_store.GetJobTargetWorkerCount(job_id,
                                                            current_target_remote_worker_count,
                                                            current_target_local_worker_count));
  // TODO: fix case when no more REM workers available (continue with AutoLocal policy)
  if (last_metrics->local_worker_count() != current_target_local_worker_count
    || last_metrics->remote_worker_count() != current_target_remote_worker_count
  ) {
    VLOG(0) << "MUYU (DynamicWorkerCountUpdateWithLocal_INCDEC) - Target metrics count not fulfilled:\n"
            << " > target: " << current_target_remote_worker_count << ", " << current_target_local_worker_count <<  "\n"
            << " > actual: " << last_metrics->remote_worker_count() << ", " << last_metrics->local_worker_count() << "\n"
            << " > available workers: " << available_workers;
    if (current_target_remote_worker_count != last_metrics->remote_worker_count() && available_workers > 0) {
      remote_worker_count = current_target_remote_worker_count;
      local_worker_count = current_target_local_worker_count;
      return Status::OK();
    }
    else { // If we don't have enough rem workers, move to AutoLocal policy
      // We should also have a similar clause for when not enough local workers are present
      // (for now assume there are always enough local workers, since they are free)
      VLOG(0) << "No more remote workers available! Moving on to AutoLocal policy.";
      metadata_store.SetJobScalingState(job_id, JobScalingState::INCREASING_LOCAL);
    }
  }

  JobScalingState scaling_state;
  TF_RETURN_IF_ERROR(metadata_store.GetJobScalingState(job_id, scaling_state));

  if (metrics_history.size() == 1) { // Cannot be smaller than 1
    VLOG(0) << "MUYU (DynamicWorkerCountUpdateWithLocal_INCDEC) - no metrics_history -> increasing worker count";
    remote_worker_count = metrics_history.back()->remote_worker_count() + 1;
    local_worker_count = metrics_history.back()->local_worker_count();
    metadata_store.SetJobTargetRemoteWorkerCount(job_id, remote_worker_count);
    metadata_store.SetJobTargetLocalWorkerCount(job_id, local_worker_count);
    return Status::OK();
  }

  int second_to_last_index = metrics_history.size() - 2;
  std::shared_ptr<ModelMetrics::Metrics> second_to_last_metrics =
          metrics_history[second_to_last_index];
  while(second_to_last_metrics->remote_worker_count() == last_metrics->remote_worker_count() &&
          second_to_last_metrics->local_worker_count() == last_metrics->local_worker_count()
  ) {
    if (second_to_last_index == 0) {
      VLOG(0) << "MUYU (DynamicWorkerCountUpdateWithLocal_INCDEC) - Not Scaling";

      remote_worker_count = last_metrics->remote_worker_count();
      local_worker_count = last_metrics->local_worker_count();
      model_metrics->converged_metrics_ = last_metrics;
      metadata_store.UnsetJobIsScaling(job_id);
      metadata_store.ResetSameScaleCounter(job_id);

      metadata_store.SetJobTargetRemoteWorkerCount(job_id, remote_worker_count);
      metadata_store.SetJobTargetLocalWorkerCount(job_id, local_worker_count);

      return Status::OK();
    }
    second_to_last_metrics = metrics_history[--second_to_last_index];
  }

  double stl_batch_time = second_to_last_metrics->last_x_batch_time_ms();
  double l_batch_time = last_metrics->last_x_batch_time_ms();
  double relative_improvement = 1.0 - l_batch_time / stl_batch_time;

  VLOG(0) << "Relative Improvement: " << relative_improvement;

  // How can this be > 1 ??
  if (relative_improvement > 1.3 || relative_improvement < -1.3) {
    VLOG(0) << "(EASL::DynamicWorkerCountUpdateWithLocal_INCDEC) Relative improvement "
            << "was unstable: " << relative_improvement
            << "; discarding it...";
    remote_worker_count = current_target_remote_worker_count;
    local_worker_count = current_target_local_worker_count;
    return Status::OK();
  }

  switch (scaling_state) {
    case JobScalingState::ONLY_REMOTE: {
      local_worker_count = last_metrics->local_worker_count();
      if (second_to_last_metrics->remote_worker_count() > last_metrics->remote_worker_count()
      ) {
        VLOG(0) << "(EASL::DynamicWorkerCountUpdateWithLocal_INCDEC)::ONLY_REMOTE"
                << "We are scaling down, which is a weird behavior!";
      }
      else {
        // we're scaling up, which is a normal behavior
        if (relative_improvement > dispatcher_config.scaling_threshold_up() &&
            last_metrics->remote_worker_count() < MAX_REMOTE_WORKERS_PER_JOB &&
            available_workers > 0) {
          remote_worker_count = last_metrics->remote_worker_count() + 1;
          VLOG(0) << "(EASL::DynamicWorkerCountUpdateWithLocal_INCDEC::ONLY_REMOTE) "
                  << "Improvement large enough:\n"
                  << " > improvement: " << relative_improvement << "\n"
                  << " > next remote worker count: " << remote_worker_count << "\n"
                  << " > available workers: " << available_workers;
        } else {
          // What's the point of this?
          if (last_metrics->remote_worker_count() == MAX_REMOTE_WORKERS_PER_JOB) {
            remote_worker_count = MAX_REMOTE_WORKERS_PER_JOB;
          }
          else {
            remote_worker_count = second_to_last_metrics->remote_worker_count();
          }
          local_worker_count = second_to_last_metrics->local_worker_count() + 1;
          model_metrics->converged_metrics_ = second_to_last_metrics;
          metadata_store.SetJobScalingState(job_id, JobScalingState::INCREASING_LOCAL);
          metadata_store.SetJobStateInitialWorkerCount(job_id, 0); // initial local worker count = 0
          metadata_store.ResetSameScaleCounter(job_id);
          VLOG(0) << "(EASL::DynamicWorkerCountUpdateWithLocal_INCDEC::ONLY_REMOTE) "
                  << "Improvement NOT large enough:\n"
                  << " > improvement: " << relative_improvement << "\n"
                  << " > next remote worker count: " << remote_worker_count
                  << " Switch to INCREASING_LOCAL mode";
        }
      }
    } break;
    case JobScalingState::DECREASING_REMOTE: {
      int64_t state_initial_worker_count;
      metadata_store.GetJobStateInitialWorkerCount(job_id, state_initial_worker_count);
      if (relative_improvement < -kPerformanceDecreaseTolerance ||
        last_metrics->remote_worker_count()  == 0
      ) {
        VLOG(0) << "(EASL::DynamicWorkerCountUpdateWithLocal_INCDEC::DECREASING_REMOTE::jump_out)";
        // jump out
        if (relative_improvement < -kPerformanceDecreaseTolerance) {
          remote_worker_count = last_metrics->remote_worker_count() + 1;
          local_worker_count = last_metrics->local_worker_count();
        } else {
          remote_worker_count = last_metrics->remote_worker_count();
          local_worker_count = last_metrics->local_worker_count();
        }

        if (remote_worker_count == state_initial_worker_count
              || local_worker_count >= MAX_LOCAL_WORKERS_PER_JOB) {
          metadata_store.SetJobScalingState(job_id, JobScalingState::STABLE);
          VLOG(0) << "(EASL::DynamicWorkerCountUpdateWithLocal_INCDEC::DECREASING_REMOTE::jump out to stable mode)";
        } else {
          metadata_store.SetJobScalingState(job_id, JobScalingState::INCREASING_LOCAL);
          metadata_store.SetJobStateInitialWorkerCount(job_id, local_worker_count);
          local_worker_count += 1;
          debug_print_local_remote("DECREASING_REMOTE::jump out to increase local", remote_worker_count, local_worker_count);
        }
      } else {
        // try reduce remote worker count further
        remote_worker_count = last_metrics->remote_worker_count()  - 1;
        local_worker_count = last_metrics->local_worker_count();
      }
    } break;
    case JobScalingState::INCREASING_LOCAL: {
      int64_t state_initial_worker_count;
      metadata_store.GetJobStateInitialWorkerCount(job_id, state_initial_worker_count);
      if (relative_improvement < -kPerformanceErrorBar ||
          last_metrics->local_worker_count() == MAX_LOCAL_WORKERS_PER_JOB
              ) {
        VLOG(0) << "(EASL::DynamicWorkerCountUpdateWithLocal_INCDEC::INCREASING_LOCAL::jump_out)";

        // decide next stage worker count
        if (relative_improvement < -kPerformanceErrorBar) {
          // set to previous state
          remote_worker_count = last_metrics->remote_worker_count();
          local_worker_count = last_metrics->local_worker_count() - 1;
        } else {
          // max local worker reached, keep everything the same
          remote_worker_count = last_metrics->remote_worker_count();
          local_worker_count = last_metrics->local_worker_count();
          // state_initial_worker_count cannot be greater than MAX_LOCAL_WORKERS
        }

        if (local_worker_count == state_initial_worker_count || remote_worker_count == 0) {
          metadata_store.SetJobScalingState(job_id, JobScalingState::STABLE);
          VLOG(0) << "(EASL::DynamicWorkerCountUpdateWithLocal_INCDEC::INCREASING_LOCAL::jump out to stable mode)"
            << state_initial_worker_count << " " << remote_worker_count;
        } else {
          metadata_store.SetJobScalingState(job_id, JobScalingState::DECREASING_REMOTE);
          metadata_store.SetJobStateInitialWorkerCount(job_id, remote_worker_count);
          remote_worker_count--;
          debug_print_local_remote("INCREASING_LOCAL::jump out to decrease remote", remote_worker_count,
                                   local_worker_count);
        }
      } else {
        remote_worker_count = last_metrics->remote_worker_count();
        local_worker_count = last_metrics->local_worker_count() + 1;
      }
    } break;
    case JobScalingState::STABLE: {
      VLOG(0) << "(EASL::DynamicWorkerCountUpdateWithLocal_INCDEC): Stable Mode, do nothing";
      remote_worker_count = last_metrics->remote_worker_count();
      local_worker_count = last_metrics->local_worker_count();
      break;
    }
    default: {
      VLOG(0) << "(EASL::DynamicWorkerCountUpdateWithLocal_INCDEC): Something wrong happening, entering default branch";
      remote_worker_count = last_metrics->remote_worker_count(); 
      local_worker_count = last_metrics->local_worker_count();
      break;
    }
  }
  metadata_store.SetJobTargetRemoteWorkerCount(job_id, remote_worker_count);
  metadata_store.SetJobTargetLocalWorkerCount(job_id, local_worker_count);

  return Status::OK();
}


//  Start From Local Policy, deprecated
//
//Status DynamicWorkerCountUpdateWithLocal_INCINC(
//        const std::string& job_type,
//        const int64 job_id,
//        const experimental::DispatcherConfig& dispatcher_config,
//        ::tensorflow::data::easl::MetadataStore& metadata_store,
//        int64& remote_worker_count,
//        int64& local_worker_count) {
//  // Entering this function means we're choosing the right policy
//  using NodeMetrics = ::tensorflow::data::easl::NodeMetrics;
//  using ModelMetrics = ::tensorflow::data::easl::ModelMetrics;
//  using JobScalingState = ::tensorflow::data::easl::JobScalingState;
//
//  VLOG(0) << "MUYU (DynamicWorkerCountUpdateWithLocal_INCINC) - Entering.";
//
//  std::shared_ptr<ModelMetrics> model_metrics;
//  TF_RETURN_IF_ERROR(metadata_store.GetModelMetrics(job_id, model_metrics));
//
//  ModelMetrics::MetricsHistory metrics_history = model_metrics->metrics_history_;
//  std::shared_ptr<ModelMetrics::Metrics> last_metrics = metrics_history[metrics_history.size() - 1];
//
//  VLOG(0) << "MUYU (DynamicWorkerCountUpdateWithLocal_INCINC) - Worker count for last metrics: "
//          << "Remote: "
//          << last_metrics->remote_worker_count()
//          << "; Local: "
//          << last_metrics->local_worker_count(); // Guaranteed to succeed.
//
//  int64 current_target_remote_worker_count, current_target_local_worker_count;
//  TF_RETURN_IF_ERROR(metadata_store.GetJobTargetWorkerCount(job_id,
//                                                            current_target_remote_worker_count,
//                                                            current_target_local_worker_count));
//  if (last_metrics->local_worker_count() != current_target_local_worker_count
//      || last_metrics->remote_worker_count() != current_target_remote_worker_count
//          ) {
//    VLOG(0) << "MUYU (DynamicWorkerCountUpdateWithLocal_INCINC) - Target metrics count not fulfilled:\n"
//            << " > target: " << current_target_remote_worker_count << ", " << current_target_local_worker_count <<  "\n"
//            << " > actual: " << last_metrics->remote_worker_count() << ", " << last_metrics->local_worker_count();
//    remote_worker_count = current_target_remote_worker_count;
//    local_worker_count = current_target_local_worker_count;
//    return Status::OK();
//  }
//
//  JobScalingState scaling_state;
//  TF_RETURN_IF_ERROR(metadata_store.GetJobScalingState(job_id, scaling_state));
//
//  if (metrics_history.size() == 1) { // Cannot be smaller than 1
//    VLOG(0) << "MUYU (DynamicWorkerCountUpdateWithLocal_INCINC) - no metrics_history -> increasing local worker count";
//    remote_worker_count = metrics_history.back()->remote_worker_count();
//    local_worker_count = metrics_history.back()->local_worker_count() + 1;
//    metadata_store.SetJobTargetRemoteWorkerCount(job_id, remote_worker_count);
//    metadata_store.SetJobTargetLocalWorkerCount(job_id, local_worker_count);
//    return Status::OK();
//  }
//
//  int second_to_last_index = metrics_history.size() - 2;
//  std::shared_ptr<ModelMetrics::Metrics> second_to_last_metrics =
//          metrics_history[second_to_last_index];
//  while(second_to_last_metrics->remote_worker_count() == last_metrics->remote_worker_count() &&
//        second_to_last_metrics->local_worker_count() == last_metrics->local_worker_count()
//          ) {
//    // TODO: MUYU understand the logic here????
//    if (second_to_last_index == 0) {
//      VLOG(0) << "MUYU (DynamicWorkerCountUpdateWithLocal_INCINC) - Should not enter here!"
//              << "This leads to an infinite loop!\n"
//              << " > Converging here since scaling is not justified.";
//
//      remote_worker_count = last_metrics->remote_worker_count();
//      local_worker_count = 0;
//      model_metrics->converged_metrics_ = last_metrics;
//      metadata_store.UnsetJobIsScaling(job_id);
//      metadata_store.ResetSameScaleCounter(job_id);
//
//      metadata_store.SetJobTargetRemoteWorkerCount(job_id, remote_worker_count);
//      metadata_store.SetJobTargetLocalWorkerCount(job_id, local_worker_count);
//      return Status::OK();
//    }
//    second_to_last_metrics = metrics_history[--second_to_last_index];
//  }
//
//  double stl_batch_time = second_to_last_metrics->last_x_batch_time_ms();
//  double l_batch_time = last_metrics->last_x_batch_time_ms();
//  double relative_improvement = 1.0 - l_batch_time / stl_batch_time;
//
//  // TODO: check the possibility of entering this branch
////  if (relative_improvement > 1.2 || relative_improvement < -1.2) {
////    VLOG(0) << "(EASL::DynamicWorkerCountUpdate) Relative improvement "
////            << "was unstable: " << relative_improvement
////            << "; discarding it...";
////    worker_count = current_target_worker_count;
////    return Status::OK();
////  }
//
//  switch (scaling_state) {
//    case JobScalingState::ONLY_REMOTE: {
//      local_worker_count = last_metrics->local_worker_count();
//      if (second_to_last_metrics->remote_worker_count()  > last_metrics->remote_worker_count()
//              ) {
//        VLOG(0) << "(EASL::DynamicWorkerCountUpdateWithLocal_INCINC)::ONLY_REMOTE"
//                << "We are scaling down, which is a weird behavior!";
//      }
//      else {
//        // we're scaling up, which is a normal behavior
//        if (relative_improvement > dispatcher_config.scaling_threshold_up()) {
//          remote_worker_count = last_metrics->remote_worker_count() + 1;
//          VLOG(0) << "(EASL::DynamicWorkerCountUpdateWithLocal_INCINC::ONLY_REMOTE) "
//                  << "Improvement large enough:\n"
//                  << " > improvement: " << relative_improvement << "\n"
//                  << " > next remote worker count: " << remote_worker_count;
//        } else {
//          // TODO: Should set to second_to_last_metrics here!! But early end of tasks still not working
//          remote_worker_count = last_metrics->remote_worker_count();
//          model_metrics->converged_metrics_ = second_to_last_metrics;
//          metadata_store.SetJobScalingState(job_id, JobScalingState::STABLE);
//          metadata_store.ResetSameScaleCounter(job_id);
//          VLOG(0) << "(EASL::DynamicWorkerCountUpdateWithLocal_INCINC::ONLY_REMOTE) "
//                  << "Improvement NOT large enough:\n"
//                  << " > improvement: " << relative_improvement << "\n"
//                  << " > next remote worker count: " << remote_worker_count
//                  << " Switch to STABLE mode";
//        }
//      }
//    } break;
//    case JobScalingState::INCREASING_LOCAL: {
//      int64_t state_initial_worker_count;
//      metadata_store.GetJobStateInitialWorkerCount(job_id, state_initial_worker_count);
//      if (last_metrics->local_worker_count() == state_initial_worker_count &&
//          last_metrics->local_worker_count() < MAX_LOCAL_WORKERS_PER_JOB
//              ) {
//        remote_worker_count = last_metrics->remote_worker_count();
//        local_worker_count = last_metrics->local_worker_count() + 1;
//        debug_print_local_remote("INCREASING_LOCAL::trial", remote_worker_count, local_worker_count);
//      }
//      else {
//        if (relative_improvement < -kPerformanceErrorBar ||
//            last_metrics->local_worker_count() == MAX_LOCAL_WORKERS_PER_JOB
//                ) {
//          VLOG(0) << "(EASL::DynamicWorkerCountUpdateWithLocal_INCINC::INCREASING_LOCAL::jump_out)";
//
//          if (relative_improvement < -kPerformanceErrorBar) {
//            // TODO: here should be second_to_last_metrics
////            remote_worker_count = second_to_last_metrics->remote_worker_count();
////            local_worker_count = second_to_last_metrics->local_worker_count();
//            remote_worker_count = last_metrics->remote_worker_count();
//            local_worker_count = last_metrics->local_worker_count();
//          } else {
//            remote_worker_count = last_metrics->remote_worker_count();
//            local_worker_count = last_metrics->local_worker_count();
//          }
//
//          metadata_store.SetJobScalingState(job_id, JobScalingState::ONLY_REMOTE);
//          metadata_store.SetJobStateInitialWorkerCount(job_id, remote_worker_count);
//          debug_print_local_remote("INCREASING_LOCAL::jump out to decrease remote", remote_worker_count,
//                                   local_worker_count);
//
//        } else {
//          // try reduce remote worker count further
//          remote_worker_count = last_metrics->remote_worker_count();
//          local_worker_count = last_metrics->local_worker_count() + 1;
//        }
//      }
//    } break;
//    case JobScalingState::STABLE: {
//      // TODO: change this branch!!
//      remote_worker_count = last_metrics->remote_worker_count();
//      local_worker_count = last_metrics->local_worker_count();
//    }
//    default: {
//      VLOG(0) << "(EASL::DynamicWorkerCountUpdateWithLocal_INCINC): Something wrong happening, entering default branch";
//      remote_worker_count = last_metrics->remote_worker_count();
//      local_worker_count = last_metrics->local_worker_count();
//    }
//      break;
//  }
//  metadata_store.SetJobTargetRemoteWorkerCount(job_id, remote_worker_count);
//  metadata_store.SetJobTargetLocalWorkerCount(job_id, local_worker_count);
//
//  // TODO: Consider setting the reference metrics for the next scale when moving
//  //       up or down. This ensures the analysis is uniform.
//
//  // Note: One will end up here in the iteration immediately after convergence
//  //       when it's highly likely that the metrics pertain to the scale prior
//  //       to convergence. You also end up here when in stability without trying
//  //       to rescale.
////  metadata_store.GetJobTargetWorkerCount(job_id, worker_count);
//  return Status::OK();
//}

} // namespace local_decision
} // namespace easl
} // namespace service
} // namespace data
} // namespace tensorflow