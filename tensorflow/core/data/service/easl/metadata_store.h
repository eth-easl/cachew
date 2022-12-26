/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_CORE_DATA_SERVICE_EASL_METADATA_STORE_H_
#define TENSORFLOW_CORE_DATA_SERVICE_EASL_METADATA_STORE_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"


namespace tensorflow {
namespace data {
namespace easl {

// This enum is used to register the last change in metrics
enum Performance { UP, DOWN, NA };
// for tracking local workers
enum JobScalingState { ONLY_REMOTE, DECREASING_REMOTE, INCREASING_LOCAL, STABLE};

class ModelMetrics {
  public:
    class Metrics {
      public:
        Metrics();
        Metrics(Metrics& other);
        Metrics(int64 worker_count, double last_x_batch_time_ms,
          double relative_wait_fraction, double result_queue_size);
        Metrics(int64 worker_count, int64 local_worker_count, double last_x_batch_time_ms,
                double relative_wait_fraction, double result_queue_size);

        void set_last_x_batch_time_ms(double x) { last_x_batch_time_ms_ = x; }
        void set_relative_wait_fraction(double x) { relative_wait_fraction_ = x; }
        void set_result_queue_size(double x) { result_queue_size_ = x; }
        void set_worker_count(int64 x) { worker_count_ = x; }

        bool has_scalability_metrics() { return has_scalability_metrics_; }
        int64 worker_count() { return worker_count_; }
        int64 remote_worker_count() { return remote_worker_count_; }
        int64 local_worker_count() { return local_worker_count_; }
        double last_x_batch_time_ms() { return last_x_batch_time_ms_; }
        double relative_wait_fraction() { return relative_wait_fraction_; }
        double result_queue_size() { return result_queue_size_; }

      private:
        bool has_scalability_metrics_;
        int64 remote_worker_count_;
        int64 local_worker_count_;
        int64 worker_count_;
        double last_x_batch_time_ms_;
        double relative_wait_fraction_;
        double result_queue_size_;
    };

    // Keys are client_id.
    using MetricsCollection =
      absl::flat_hash_map<int64, std::deque<std::shared_ptr<ModelMetrics::Metrics>>>;

    using MetricsByWorkerCount =
      absl::flat_hash_map<int64, std::shared_ptr<MetricsCollection>>;

    using MetricsHistory = std::deque<std::shared_ptr<ModelMetrics::Metrics>>;

    ModelMetrics() {}

    // Update the values for a worker_count, client_id pair.
    Status UpdateClientMetrics(const int64 client_id, Metrics& metrics);
    Status GetClientMetrics(const int64 worker_count, const int64 client_id, std::deque<std::shared_ptr<Metrics>>& metrics);
    Status GetAllWorkerCountMetrics(std::shared_ptr<MetricsByWorkerCount>& metrics);
    Status GetAllClientMetrics(const int64 worker_count, std::shared_ptr<MetricsCollection>&);
    Status GetMetricsHistory(MetricsHistory& metrics_history);

    // Dump metrics to a string stream
    void DumpToStream(std::stringstream& ss);

    // The keys are the worker count
    MetricsByWorkerCount metrics_;
    // Metrics history, stored in order of arrival
    MetricsHistory metrics_history_;
    // Metrics for converged state
    std::shared_ptr<Metrics> converged_metrics_;
};

class NodeMetrics {
  public: 
    class Metrics {
      public:
        explicit Metrics(Metrics& other);
        explicit Metrics(int64 bytes_consumed, int64 bytes_produced, 
                          int64 num_elements, int64 bytes_per_s,
                          // int64 computation_time, 
                          double in_node_time_ms, double in_prefix_time_ms,
                          double active_time, double working_time);
        
        void Update(Metrics& other);
        
        void set_bytes_consumed(int64 x)   { bytes_consumed_ = x; }
        void set_bytes_produced(int64 x)   { bytes_produced_ = x; }
        void set_num_elements(int64 x)     { num_elements_ = x; }
        void set_bytes_per_s(int64 x)       { bytes_per_s_ = x; }
        void set_in_node_time_ms(double x)    { in_node_time_ms_ = x; }
        void set_in_prefix_time_ms(double x)  { in_prefix_time_ms_ = x; }
        void set_active_time_ms(double x) { active_time_ms_ = x; }
        void set_working_time_ms(double x) { working_time_ms_ = x; }

        int64 bytes_consumed()   { return bytes_consumed_; }
        int64 bytes_produced()   { return bytes_produced_; }
        int64 num_elements()     { return num_elements_; }
        int64 bytes_per_s()     { return bytes_per_s_; }
        double in_node_time_ms()     { return in_node_time_ms_; }
        double in_prefix_time_ms()   { return in_prefix_time_ms_; }
        double active_time_ms() { return active_time_ms_; }
        double working_time_ms() { return working_time_ms_; }

        void log_metrics() {
          VLOG(3) << "(MetadataStore::NodeMetrics) Metrics:\n"
                  << " > bytes_consumed = " << bytes_consumed_ << "\n"
                  << " > bytes_produced = " << bytes_produced_ << "\n"
                  << " > bytes_per_s = " << bytes_per_s_ << "\n"
                  << " > num_elements = " << num_elements_ << "\n"
                  << " > in_node_time = " << in_node_time_ms_ << "\n"
                  << " > in_prefix_time = " << in_prefix_time_ms_;
        }

      private:
        int64 bytes_consumed_;
        int64 bytes_produced_;
        int64 num_elements_;
        int64 bytes_per_s_;
        double in_node_time_ms_;
        double in_prefix_time_ms_;
        double active_time_ms_;
        double working_time_ms_;
    };

    using MetricsCollection =
      absl::flat_hash_map<string, std::shared_ptr<NodeMetrics::Metrics>>;

    NodeMetrics() {}

    // Get or update the metrics of a worker
    Status UpdateWorkerMetrics(string worker_address, Metrics& metrics);
    Status GetWorkerMetrics(string worker_address, 
      std::shared_ptr<Metrics>& metrics);

    // Dump metrics to string stream
    void DumpToStream(std::stringstream& ss);

  // The key here is the worker address
    MetricsCollection metrics_;
};


class InputPipelineMetrics {
  public:
    using MetricsCollection = 
      absl::flat_hash_map<string, std::shared_ptr<NodeMetrics>>;

    InputPipelineMetrics() 
      : last_node_name_(""),
        last_tf_node_name_(""),
        maker_node_name_("") {}
    InputPipelineMetrics(std::string last_node_name, 
      std::string last_tf_node_name, std::string maker_node_name) 
      : last_node_name_(last_node_name), 
        last_tf_node_name_(last_tf_node_name),
        maker_node_name_(maker_node_name) {}

    // Get the metrics for a single node
    Status GetNodeMetrics(string long_name, 
      std::shared_ptr<NodeMetrics>& metrics);
    Status GetLastNodeMetrics(std::shared_ptr<NodeMetrics>& metrics);
    Status GetLastTFNodeMetrics(std::shared_ptr<NodeMetrics>& metrics);
    Status GetMarkerNodeMetrics(std::shared_ptr<NodeMetrics>& metrics);

    // Get the metrics from the same worker for each node in the graph 
    Status GetWorkerMetrics(string worker_address, 
      NodeMetrics::MetricsCollection& metrics);

    // Methods for setting data
    Status UpdateNodeMetrics(string long_name, string worker_address, 
      NodeMetrics::Metrics& metrics);

    // Methods for managing the last node name
    std::string GetLastNodeName();
    void SetLastNodeName(std::string last_node_name);

    std::string GetLastTFNodeName();
    void SetLastTFNodeName(std::string last_tf_node_name);
    
    std::string GetMarkerNodeName();
    void SetMarkerNodeName(std::string marker_node_name);

    void DumpToStream(std::stringstream& ss);

    // Last user node name
    std::string last_node_name_;
    std::string last_tf_node_name_;
    std::string maker_node_name_;
    // The keys are the long name of the node
    MetricsCollection metrics_;
    // Counter to understand how many worker heartbeats we had
    uint64 update_counter_;
};

class JobMetrics {
  public:
    JobMetrics(int64 job_id,
               std::string& job_type,
               int64 dataset_id,
               uint64 dataset_fingerprint,
               std::string& dataset_key,
               bool is_scaling = true,
               const string& name = string(),
               bool is_ordering = true,
               bool is_ordered = false);

    JobMetrics(int64 job_id,
               std::string& job_type,
               int64 dataset_id,
               uint64 dataset_fingerprint,
               std::string& dataset_key,
               bool is_scaling,
               const string& name,
               int64 target_remote_worker_count,
               int64 target_local_worker_count,
               JobScalingState scaling_state,
               bool is_ordering = true,
               bool is_ordered = false);

    void DumpToFile(const std::string& path);
    void DumpToStream(std::stringstream& ss);

    bool is_scaling_;
    bool is_ordering_;
    bool is_ordered_;

    /* When scaling_state_ is DECREASING_REMOTE: worker_count_ is remote
     * When scaling_state_ is INCREASING_LOCAL: worker_count_ is local
     * */
    JobScalingState scaling_state_;
    int64_t state_initial_worker_count_;

    Performance last_performance_;
    string job_type_;
    string name_;
    uint64 same_scale_counter_;
    int64 target_worker_count_;
    int64 target_remote_worker_count_;
    int64 target_local_worker_count_;
    int64 job_id_;
    int64 dataset_id_;
    int64 dataset_fingerprint_;
    std::string dataset_key_;
    std::shared_ptr<ModelMetrics> model_metrics_;
    std::shared_ptr<InputPipelineMetrics> input_pipeline_metrics_;
};

class MetadataStore {
 public:
  MetadataStore();
  MetadataStore(const MetadataStore &) = delete;
  MetadataStore &operator=(const MetadataStore &) = delete;

  // Create a job entry
  Status CreateJob(int64 job_id,
                   string& job_type,
                   int64 dataset_id,
                   uint64 dataset_fingerprint,
                   std::string& dataset_key,
                   bool trigger_rescale = false);


  Status CreateJobName(int64 job_id,
                       string& job_name,
                       string& job_type,
                       int64 dataset_id,
                       uint64 dataset_fingerprint,
                       std::string& dataset_key,
                       bool trigger_rescale);

  // overloading for scaling policy 3 & 4
  Status CreateJobName(int64 job_id,
                       string& job_name,
                       string& job_type,
                       int64 dataset_id,
                       uint64 dataset_fingerprint,
                       std::string& dataset_key,
                       bool trigger_rescale,
                       int64 policy
                       );

  // Remove job
  Status RemoveJob(int64 job_id);

  // Get metrics
  Status GetJobMetrics(int64 job_id, std::shared_ptr<JobMetrics>& metrics) const;

  Status GetModelMetrics(int64 job_id,
    std::shared_ptr<ModelMetrics>& metrics) const;

  Status GetInputPipelineMetrics(int64 job_id,
    std::shared_ptr<InputPipelineMetrics>& metrics) const;

  Status GetJobMetricsByDatasetFingerprint(const uint64 dataset_fingerprint,
    std::shared_ptr<JobMetrics>& metrics) const;
  Status GetJobMetricsByDatasetFingerprintAndName(
    const uint64 dataset_fingerprint, const string& job_name,
    std::shared_ptr<JobMetrics>& metrics) const;

  Status GetModelMetricsByDatasetFingerprint(const uint64 dataset_fingerprint,
    std::shared_ptr<ModelMetrics>& metrics) const;

  Status GetInputPipelineMetricsByDatasetFingerprint(const uint64 dataset_fingerprint,
    std::shared_ptr<InputPipelineMetrics>& metrics) const;

  Status GetLastNodeMetrics(int64 job_id,
    std::shared_ptr<NodeMetrics>& metrics) const;
  Status GetLastTFNodeMetrics(int64 job_id,
    std::shared_ptr<NodeMetrics>& metrics) const;
  Status GetMarkerNodeMetrics(int64 job_id,
    std::shared_ptr<NodeMetrics>& metrics) const;

  Status GetLastNodeMetricsByDatasetFingerprint(const uint64 dataset_fingerprint,
    std::shared_ptr<NodeMetrics>& metrics) const;
  Status GetLastTFNodeMetricsByDatasetFingerprint(const uint64 dataset_fingerprint,
    std::shared_ptr<NodeMetrics>& metrics) const;
  Status GetMarkerNodeMetricsByDatasetFingerprint(const uint64 dataset_fingerprint,
    std::shared_ptr<NodeMetrics>& metrics) const;

  // Update or create the metrics for a client
  Status UpdateModelMetrics(int64 job_id, int64 client_id,
    ModelMetrics::Metrics& metrics);

  // Update or create the metrics for a client
  Status UpdateInputPipelineMetrics(int64 job_id, string node_long_name,
    string worker_address, NodeMetrics::Metrics& metrics);

  Status UpdateNodeNames(int64 job_id, string last_node_name,
    string last_tf_node_name, string marker_node_name);

  string CreateFingerprintNameKey(uint64 fingerprint, const string& job_name) const;

  Status SetJobType(uint64 fingerprint, string job_type);
  Status SetJobType(uint64 fingerprint, const string& name, string job_type);
  Status GetJobType(uint64 fingerprint, string& job_type);
  Status GetJobType(uint64 fingerprint, const string& name, string& job_type);
  Status SetJobTypeByJobId(int64 job_id, string job_type);
  Status GetJobTypeByJobId(int64 job_id, string& job_type);

  Status SetJobIsScaling(int64 job_id);
  Status UnsetJobIsScaling(int64 job_id);
  Status IsJobScaling(int64 job_id, bool& is_scaling);

  Status SetJobIsOrdering(int64 job_id);
  Status UnsetJobIsOrdering(int64 job_id);
  Status IsJobOrdering(int64 job_id, bool& is_ordering);

  Status IsJobOrdered(int64 job_id, bool& is_ordering);
  Status SetJobIsOrdered(int64 job_id);

  Status GetJobScalingState(int64 job_id, JobScalingState& scaling_state);
  Status SetJobScalingState(int64 job_id, JobScalingState scaling_state);

  Status GetJobStateInitialWorkerCount(int64 job_id, int64_t& worker_count);
  Status SetJobStateInitialWorkerCount(int64 job_id, int64_t worker_count);

  Status GetLastPerformance(int64 job_id, Performance& last_performance);
  Status SetLastPerformance(int64 job_id, Performance last_performance);

  bool JobSeenBefore(uint64 fingerprint);

  Status GetWorkerUpdateCounter(int64 job_id, uint64& heartbeat_counter);
  Status GetNumberOfProducedElements(int64 job_id, uint64& element_count);

  // These are required since looking up in metrics history is both expensive
  // and history can be trimmed
  Status GetSameScaleCounter(int64 job_id, uint64& counter);
  Status IncrementSameScaleCounter(int64 job_id, uint64& counter);
  Status ResetSameScaleCounter(int64 job_id);

  Status SetJobTargetWorkerCount(int64 job_id, int64 target_worker_count);
  Status GetJobTargetWorkerCount(int64 job_id, int64& target_worker_count);
  Status GetJobTargetWorkerCount(int64 job_id, int64& target_remote_worker_count, int64& target_local_worker_count);

  Status SetJobTargetRemoteWorkerCount(int64 job_id, int64 target_remote_worker_count);
  Status SetJobTargetLocalWorkerCount(int64 job_id, int64 target_local_worker_count);

  // Update or create the metrics for the dataset key from the given job.
  Status UpdateFingerprintKeyJobMetrics(int64 job_id);
  Status UpdateFingerprintNameKeyJobMetrics(int64 job_id);

  // Dumps the job metrics in a file named after its id at the given path.
  Status DumpJobMetricsToFile(int64 job_id, const std::string& path);

  // Appends json representation of current status to a separate file for each job
  Status AppendJobMetricsDumps(Env* env, const std::string& path);

  // Transfers the model metrics history previously collected to a new job
  //Status TransferModelMetricsToNewJob(std::string dataset_key, int64 job_id);


 private:
  // Key is job id
  absl::flat_hash_map<int64, std::shared_ptr<JobMetrics>> job_metadata_;
  // Key is fingerprint
  absl::flat_hash_map<int64, std::shared_ptr<JobMetrics>> fingerprint_key_metadata_;
  // Key is fingerprint+job_name
  absl::flat_hash_map<string, std::shared_ptr<JobMetrics>> fingerprint_name_metadata_;
};

// Utils function to append the missing trailing "]"
// at the end of a metrics update dump file.
void TerminateJobMetricsAppendDumps(int64 job_id, const std::string& path);
} // namespace easl
} // namespace data
} // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_EASL_METADATA_STORE_H_
