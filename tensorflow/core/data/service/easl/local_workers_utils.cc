//
// Created by Muyu Li on 16.11.21.
// Edited by the DSL group HS21 (Theodor Amariucai, Jiale Chen, Muyu Li) throughout November 2021 - February 2022
//

#include "local_workers_utils.h"

namespace tensorflow {
namespace data {
namespace service {
namespace easl {
namespace local_workers_utils {

Status ShouldUseLocalWorkers(
        const experimental::DispatcherConfig& dispatcher_config,
        const ::tensorflow::data::easl::MetadataStore& metadata_store,
        const int64 dataset_fingerprint,
        bool& should_use_local_workers) {
    using NodeMetrics = ::tensorflow::data::easl::NodeMetrics;
    using ModelMetrics = ::tensorflow::data::easl::ModelMetrics;

    // Check if we have any metrics for this dataset
    std::shared_ptr<data::easl::InputPipelineMetrics> job_metrics;
    Status s = metadata_store.GetInputPipelineMetricsByDatasetFingerprint(
            dataset_fingerprint, job_metrics);

    // We do not yet have the metrics for this dataset --> use 1 worker
    if(errors::IsNotFound(s)) {
        VLOG(0) << "DSL (ShouldUseLocalWorkers) No metrics found for dataset, will use local workers (optimistic)!";
        should_use_local_workers = true;
        return Status::OK();
    } else if (!s.ok()) {
        VLOG(0) << "DSL (ShouldUseLocalWorkers) Another error has been thrown: " << s;
        return s;
    }

    // Pipeline stats: last TF node metrics
    std::shared_ptr<NodeMetrics> last_tf_node_metrics;
    s = metadata_store.GetLastNodeMetricsByDatasetFingerprint(dataset_fingerprint, last_tf_node_metrics);
    if (!s.ok()) {
        VLOG(0) << "DSL (ShouldUseLocalWorkers) Failed to get the last TF node metrics";
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
    VLOG(0) << "DSL (ShouldUseLocalWorkers) Total bytes produced: " << total_bytes_produced << "\n"
            << "Total num elements: " << total_num_elements << "\n"
            << "Avg bytes produced per element: " << avg_bytes_per_element << "\n"
            << "Decision Threshold: " << dispatcher_config.avg_bytes_per_element_local_workers_threshold() << "\n";

    if (avg_bytes_per_element > dispatcher_config.avg_bytes_per_element_local_workers_threshold()) {
        should_use_local_workers = true;
        VLOG(0) << "DSL (ShouldUseLocalWorkers) Using local workers! (because avg. bytes per element > threshold) \n";
    }
    else {
        should_use_local_workers = false;
        VLOG(0) << "DSL (ShouldUseLocalWorkers) NOT using local workers! (because avg. bytes per element < threshold) \n";
    }

    return Status::OK();
}

std::vector<int64> records;

Status DecideTargetWorkersGridSearch(
        int64 num_worker_remote_avail,
        int64 num_worker_local_avail,
        int64& num_worker_remote_target,
        int64& num_worker_local_target) {
  std::time_t t = std::time(nullptr);
  records.push_back(t);

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

  VLOG(0) << "DSL (DecideTargetWorkersGridSearch)" << "\n"
          << "Available remote: " << num_worker_remote_avail << "\n"
          << "Available local: " << num_worker_local_avail << "\n"
          << "Decided remote: " << num_worker_remote_target << "\n"
          << "Decided local: " << num_worker_local_target << "\n";
  return Status::OK();
}

} // namespace local_workers_utils
} // namespace easl
} // namespace service
} // namespace data
} // namespace tensorflow
