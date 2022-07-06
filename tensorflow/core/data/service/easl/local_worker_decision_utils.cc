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


namespace tensorflow {
namespace data {
namespace service {
namespace easl {
namespace local_worker_decision {
/*
Status DecideIfLocal(
        const experimental::DispatcherConfig& dispatcher_config,
        const ::tensorflow::data::easl::MetadataStore& metadata_store,
        const std::string& dataset_key,
        bool& want_to_use_local_workers) {
  using NodeMetrics = ::tensorflow::data::easl::NodeMetrics;
  using ModelMetrics = ::tensorflow::data::easl::ModelMetrics;

  // Check if we have any metrics for this dataset
  std::shared_ptr<data::easl::InputPipelineMetrics> job_metrics;
  Status s = metadata_store.GetInputPipelineMetricsByDatasetKey(
          dataset_key, job_metrics);

  // We do not yet have the metrics for this dataset --> use 1 worker
  if(errors::IsNotFound(s)) {
    VLOG(0) << "DSL (DecideIfLocal) No metrics found for dataset, will use local workers (optimistic)!";
    want_to_use_local_workers = true;
    return Status::OK();
  } else if (!s.ok()) {
    VLOG(0) << "DSL (DecideIfLocal) Another error has been thrown: " << s;
    return s;
  }

  // Pipeline stats: last TF node metrics
  std::shared_ptr<NodeMetrics> last_tf_node_metrics;

  s = metadata_store.GetLastNodeMetricsByDatasetKey(dataset_key, last_tf_node_metrics);
  if (!s.ok()) {
    VLOG(0) << "DSL (DecideIfLocal) Failed to get the last TF node metrics";
    return s;
  }

  int64_t total_bytes_produced = 0, total_num_elements = 0;
  for (std::pair<string, std::shared_ptr<NodeMetrics::Metrics>> e :
          last_tf_node_metrics->metrics_) {
    std::shared_ptr<NodeMetrics::Metrics> node_metrics = e.second;
    total_bytes_produced += node_metrics->bytes_produced();
    total_num_elements += node_metrics->num_elements();
  }

  double avg_bytes_per_element = (double)total_bytes_produced / total_num_elements;
  VLOG(0) << "DSL (DecideIfLocal) Total bytes produced: " << total_bytes_produced << "\n"
          << "Total num elements: " << total_num_elements << "\n"
          << "Avg bytes produced per element: " << avg_bytes_per_element << "\n"
          << "Decision Threshold: " << dispatcher_config.avg_bytes_per_element_local_thres() << "\n";

  if (avg_bytes_per_element > dispatcher_config.avg_bytes_per_element_local_thres()) {
    want_to_use_local_workers = true;
    VLOG(0) << "DSL (DecideIfLocal) Using local workers! (because avg. bytes per element > threshold) \n";
  }
  else {
    want_to_use_local_workers = false;
    VLOG(0) << "DSL (DecideIfLocal) NOT using local workers! (because avg. bytes per element < threshold) \n";
  }

  return Status::OK();
}

std::vector<int64> records;

void grid_search(int64 num_worker_remote_avail, int64 num_worker_local_avail,
                 int64& num_worker_remote_target, int64& num_worker_local_target) {
  std::vector<std::pair<int64, int64>> test_set = std::vector<std::pair<int64, int64>>();
  for(int64 n_r = 0; n_r <= num_worker_remote_avail; n_r++) {
    for(int64 n_l = 0; n_l <= num_worker_local_avail; n_l++) {
      if(n_r + n_l <= 0) {
        continue;
      }
      test_set.emplace_back(n_r, n_l);
    }
  }
  std::vector<int64> epoch_times;
  for(int i = 1; i < records.size(); i++) {
    epoch_times.push_back(records[i] - records[i-1]);
  }
  int index;
  if(epoch_times.size() < test_set.size()) {
    index = epoch_times.size();
  } else {
    index = std::min_element(epoch_times.begin(), epoch_times.begin() + test_set.size()) - epoch_times.begin();
  }
  auto p = test_set[index];
  num_worker_remote_target = p.first;
  num_worker_local_target = p.second;
}

Status DecideTargetWorkersGridSearch(
        const experimental::DispatcherConfig& dispatcher_config,
        const ::tensorflow::data::easl::MetadataStore& metadata_store,
        const std::string& dataset_key,
        int64 num_worker_remote_avail,
        int64 num_worker_local_avail,
        int64& num_worker_remote_target,
        int64& num_worker_local_target) {
  std::time_t t = std::time(nullptr);
  records.push_back(t);
  grid_search(num_worker_remote_avail, num_worker_local_avail, num_worker_remote_target, num_worker_local_target);
  VLOG(0) << "DSL (DecideTargetWorkers) Available remote: " << num_worker_remote_avail << "\n"
          << "Available local: " << num_worker_local_avail << "\n"
          << "Decided remote: " << num_worker_remote_target << "\n"
          << "Decided local: " << num_worker_local_target << "\n";
  return Status::OK();
}
 */

//TODO: Implement this based on EASL 2.7 autoscaling
Status DecideTargetWorkersAutoscaling(
        const experimental::DispatcherConfig& dispatcher_config,
        const ::tensorflow::data::easl::MetadataStore& metadata_store,
        const std::string& dataset_key,
        int64 num_worker_remote_avail,
        int64 num_worker_local_avail,
        int64& num_worker_remote_target,
        int64& num_worker_local_target) {
  num_worker_remote_target = num_worker_remote_avail / 2;
  num_worker_local_target = num_worker_local_avail / 2;

  VLOG(0) << "DSL (DecideTargetWorkers) Available remote: " << num_worker_remote_avail << "\n"
          << "Available local: " << num_worker_local_avail << "\n"
          << "Decided remote: " << num_worker_remote_target << "\n"
          << "Decided local: " << num_worker_local_target << "\n";
  return Status::OK();
}

namespace {
    int MAX_WORKERS_PER_JOB = 100;
    double kMinBatchTimeRelativeImprovementDown = 0.03;
    uint32 kInStabilityBeforeScaling = 20;
    double kMinQueueSizeRelativeGrowth = 1.5; // +50%
    double kMinBatchTimeRelativeGrowth = 1.5; // +50%

    int MAX_LOCAL_WORKERS_PER_JOB = 5;
    int MAX_REMOTE_WORKERS_PER_JOB = 8;
    double kPerformanceErrorBar = 0.1;
    // combine with costs
    double kPerformanceDecreaseTolerance = 0.1;
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
        int64& local_worker_count) {
  // Entering this function means we're choosing the right policy
  using NodeMetrics = ::tensorflow::data::easl::NodeMetrics;
  using ModelMetrics = ::tensorflow::data::easl::ModelMetrics;
  using JobScalingState = ::tensorflow::data::easl::JobScalingState;

  VLOG(0) << "MUYU (DynamicWorkerCountUpdateWithLocal_INCDEC) - Entering.";

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
  if (last_metrics->local_worker_count() != current_target_local_worker_count
    || last_metrics->remote_worker_count() != current_target_remote_worker_count
  ) {
    VLOG(0) << "MUYU (DynamicWorkerCountUpdateWithLocal_INCDEC) - Target metrics count not fulfilled:\n"
            << " > target: " << current_target_remote_worker_count << ", " << current_target_local_worker_count <<  "\n"
            << " > actual: " << last_metrics->remote_worker_count() << ", " << last_metrics->local_worker_count();
    remote_worker_count = current_target_remote_worker_count;
    local_worker_count = current_target_local_worker_count;
    return Status::OK();
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
    // TODO: MUYU understand the logic here????
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

  // TODO: check the possibility of entering this branch
//  if (relative_improvement > 1.2 || relative_improvement < -1.2) {
//    VLOG(0) << "(EASL::DynamicWorkerCountUpdate) Relative improvement "
//            << "was unstable: " << relative_improvement
//            << "; discarding it...";
//    worker_count = current_target_worker_count;
//    return Status::OK();
//  }

  switch (scaling_state) {
    case JobScalingState::ONLY_REMOTE: {
      local_worker_count = last_metrics->local_worker_count();
      if (second_to_last_metrics->remote_worker_count()  > last_metrics->remote_worker_count()
      ) {
        VLOG(0) << "(EASL::DynamicWorkerCountUpdateWithLocal_INCDEC)::ONLY_REMOTE"
                << "We are scaling down, which is a weird behavior!";
      }
      else {
        // we're scaling up, which is a normal behavior
        if (relative_improvement > dispatcher_config.scaling_threshold_up() &&
          last_metrics->remote_worker_count() < MAX_REMOTE_WORKERS_PER_JOB) {
          remote_worker_count = last_metrics->remote_worker_count() + 1;
          VLOG(0) << "(EASL::DynamicWorkerCountUpdateWithLocal_INCDEC::ONLY_REMOTE) "
                  << "Improvement large enough:\n"
                  << " > improvement: " << relative_improvement << "\n"
                  << " > next remote worker count: " << remote_worker_count;
        } else {
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

  // TODO: Consider setting the reference metrics for the next scale when moving
  //       up or down. This ensures the analysis is uniform.

  // Note: One will end up here in the iteration immediately after convergence
  //       when it's highly likely that the metrics pertain to the scale prior
  //       to convergence. You also end up here when in stability without trying
  //       to rescale.
//  metadata_store.GetJobTargetWorkerCount(job_id, worker_count);
  return Status::OK();
}


Status DynamicWorkerCountUpdateWithLocal_INCINC(
        const std::string& job_type,
        const int64 job_id,
        const experimental::DispatcherConfig& dispatcher_config,
        ::tensorflow::data::easl::MetadataStore& metadata_store,
        int64& remote_worker_count,
        int64& local_worker_count) {
  // Entering this function means we're choosing the right policy
  using NodeMetrics = ::tensorflow::data::easl::NodeMetrics;
  using ModelMetrics = ::tensorflow::data::easl::ModelMetrics;
  using JobScalingState = ::tensorflow::data::easl::JobScalingState;

  VLOG(0) << "MUYU (DynamicWorkerCountUpdateWithLocal_INCINC) - Entering.";

  std::shared_ptr<ModelMetrics> model_metrics;
  TF_RETURN_IF_ERROR(metadata_store.GetModelMetrics(job_id, model_metrics));

  ModelMetrics::MetricsHistory metrics_history = model_metrics->metrics_history_;
  std::shared_ptr<ModelMetrics::Metrics> last_metrics = metrics_history[metrics_history.size() - 1];

  VLOG(0) << "MUYU (DynamicWorkerCountUpdateWithLocal_INCINC) - Worker count for last metrics: "
          << "Remote: "
          << last_metrics->remote_worker_count()
          << "; Local: "
          << last_metrics->local_worker_count(); // Guaranteed to succeed.

  int64 current_target_remote_worker_count, current_target_local_worker_count;
  TF_RETURN_IF_ERROR(metadata_store.GetJobTargetWorkerCount(job_id,
                                                            current_target_remote_worker_count,
                                                            current_target_local_worker_count));
  if (last_metrics->local_worker_count() != current_target_local_worker_count
      || last_metrics->remote_worker_count() != current_target_remote_worker_count
          ) {
    VLOG(0) << "MUYU (DynamicWorkerCountUpdateWithLocal_INCINC) - Target metrics count not fulfilled:\n"
            << " > target: " << current_target_remote_worker_count << ", " << current_target_local_worker_count <<  "\n"
            << " > actual: " << last_metrics->remote_worker_count() << ", " << last_metrics->local_worker_count();
    remote_worker_count = current_target_remote_worker_count;
    local_worker_count = current_target_local_worker_count;
    return Status::OK();
  }

  JobScalingState scaling_state;
  TF_RETURN_IF_ERROR(metadata_store.GetJobScalingState(job_id, scaling_state));

  if (metrics_history.size() == 1) { // Cannot be smaller than 1
    VLOG(0) << "MUYU (DynamicWorkerCountUpdateWithLocal_INCINC) - no metrics_history -> increasing local worker count";
    remote_worker_count = metrics_history.back()->remote_worker_count();
    local_worker_count = metrics_history.back()->local_worker_count() + 1;
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
    // TODO: MUYU understand the logic here????
    if (second_to_last_index == 0) {
      VLOG(0) << "MUYU (DynamicWorkerCountUpdateWithLocal_INCINC) - Should not enter here!"
              << "This leads to an infinite loop!\n"
              << " > Converging here since scaling is not justified.";

      remote_worker_count = last_metrics->remote_worker_count();
      local_worker_count = 0;
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

  // TODO: check the possibility of entering this branch
//  if (relative_improvement > 1.2 || relative_improvement < -1.2) {
//    VLOG(0) << "(EASL::DynamicWorkerCountUpdate) Relative improvement "
//            << "was unstable: " << relative_improvement
//            << "; discarding it...";
//    worker_count = current_target_worker_count;
//    return Status::OK();
//  }

  switch (scaling_state) {
    case JobScalingState::ONLY_REMOTE: {
      local_worker_count = last_metrics->local_worker_count();
      if (second_to_last_metrics->remote_worker_count()  > last_metrics->remote_worker_count()
              ) {
        VLOG(0) << "(EASL::DynamicWorkerCountUpdateWithLocal_INCINC)::ONLY_REMOTE"
                << "We are scaling down, which is a weird behavior!";
      }
      else {
        // we're scaling up, which is a normal behavior
        if (relative_improvement > dispatcher_config.scaling_threshold_up()) {
          remote_worker_count = last_metrics->remote_worker_count() + 1;
          VLOG(0) << "(EASL::DynamicWorkerCountUpdateWithLocal_INCINC::ONLY_REMOTE) "
                  << "Improvement large enough:\n"
                  << " > improvement: " << relative_improvement << "\n"
                  << " > next remote worker count: " << remote_worker_count;
        } else {
          // TODO: Should set to second_to_last_metrics here!! But early end of tasks still not working
          remote_worker_count = last_metrics->remote_worker_count();
          model_metrics->converged_metrics_ = second_to_last_metrics;
          metadata_store.SetJobScalingState(job_id, JobScalingState::STABLE);
          metadata_store.ResetSameScaleCounter(job_id);
          VLOG(0) << "(EASL::DynamicWorkerCountUpdateWithLocal_INCINC::ONLY_REMOTE) "
                  << "Improvement NOT large enough:\n"
                  << " > improvement: " << relative_improvement << "\n"
                  << " > next remote worker count: " << remote_worker_count
                  << " Switch to STABLE mode";
        }
      }
    } break;
    case JobScalingState::INCREASING_LOCAL: {
      int64_t state_initial_worker_count;
      metadata_store.GetJobStateInitialWorkerCount(job_id, state_initial_worker_count);
      if (last_metrics->local_worker_count() == state_initial_worker_count &&
          last_metrics->local_worker_count() < MAX_LOCAL_WORKERS_PER_JOB
              ) {
        remote_worker_count = last_metrics->remote_worker_count();
        local_worker_count = last_metrics->local_worker_count() + 1;
        debug_print_local_remote("INCREASING_LOCAL::trial", remote_worker_count, local_worker_count);
      }
      else {
        if (relative_improvement < -kPerformanceErrorBar ||
            last_metrics->local_worker_count() == MAX_LOCAL_WORKERS_PER_JOB
                ) {
          VLOG(0) << "(EASL::DynamicWorkerCountUpdateWithLocal_INCINC::INCREASING_LOCAL::jump_out)";

          if (relative_improvement < -kPerformanceErrorBar) {
            // TODO: here should be second_to_last_metrics
//            remote_worker_count = second_to_last_metrics->remote_worker_count();
//            local_worker_count = second_to_last_metrics->local_worker_count();
            remote_worker_count = last_metrics->remote_worker_count();
            local_worker_count = last_metrics->local_worker_count();
          } else {
            remote_worker_count = last_metrics->remote_worker_count();
            local_worker_count = last_metrics->local_worker_count();
          }

          metadata_store.SetJobScalingState(job_id, JobScalingState::ONLY_REMOTE);
          metadata_store.SetJobStateInitialWorkerCount(job_id, remote_worker_count);
          debug_print_local_remote("INCREASING_LOCAL::jump out to decrease remote", remote_worker_count,
                                   local_worker_count);

        } else {
          // try reduce remote worker count further
          remote_worker_count = last_metrics->remote_worker_count();
          local_worker_count = last_metrics->local_worker_count() + 1;
        }
      }
    } break;
    case JobScalingState::STABLE: {
      // TODO: change this branch!!
      remote_worker_count = last_metrics->remote_worker_count();
      local_worker_count = last_metrics->local_worker_count();
    }
    default: {
      VLOG(0) << "(EASL::DynamicWorkerCountUpdateWithLocal_INCINC): Something wrong happening, entering default branch";
      remote_worker_count = last_metrics->remote_worker_count();
      local_worker_count = last_metrics->local_worker_count();
    }
      break;
  }
  metadata_store.SetJobTargetRemoteWorkerCount(job_id, remote_worker_count);
  metadata_store.SetJobTargetLocalWorkerCount(job_id, local_worker_count);

  // TODO: Consider setting the reference metrics for the next scale when moving
  //       up or down. This ensures the analysis is uniform.

  // Note: One will end up here in the iteration immediately after convergence
  //       when it's highly likely that the metrics pertain to the scale prior
  //       to convergence. You also end up here when in stability without trying
  //       to rescale.
//  metadata_store.GetJobTargetWorkerCount(job_id, worker_count);
  return Status::OK();
}

} // namespace local_decision
} // namespace easl
} // namespace service
} // namespace data
} // namespace tensorflow