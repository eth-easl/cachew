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

#include "tensorflow/core/data/service/dispatcher_impl.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef PLATFORM_GOOGLE
#include "file/logging/log_lines.h"
#endif
#include "grpcpp/create_channel.h"
#include "grpcpp/impl/codegen/server_context.h"
#include "grpcpp/security/credentials.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/hash_utils.h"
#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/credentials_factory.h"
#include "tensorflow/core/data/service/dataset_store.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/dispatcher_state.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/journal.h"
#include "tensorflow/core/data/service/worker.grpc.pb.h"
#include "tensorflow/core/data/service/easl/cache_utils.h"
#include "tensorflow/core/data/service/easl/scaling_utils.h"
#include "tensorflow/core/data/service/easl/ordering_utils.h"
#include "tensorflow/core/data/service/easl/metadata_store.h"
#include "tensorflow/core/data/service/easl/local_worker_decision_utils.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/service_config.pb.h"
#include "tensorflow/core/public/session_options.h"

#include <fstream>

namespace tensorflow {
namespace data {
namespace {

using ::tensorflow::protobuf::util::MessageDifferencer;

// The name of the journal directory inside the dispatcher's working directory.
// This name is load-bearing; do not change.
constexpr char kJournalDir[] = "tf_data_dispatcher_journal";
// The name of the datasets directory inside the dispatcher's working directory.
constexpr char kDatasetsDir[] = "datasets";
constexpr int64_t kDefaultJobGcCheckIntervalMs = 10 * 60 * 1000;  // 10 minutes.
constexpr int64_t kDefaultJobGcTimeoutMs = 5 * 60 * 1000;         // 5 minutes.
constexpr int64_t kDefaultClientTimeoutMs = 2 * 60 * 1000;        // 2 minutes.

constexpr std::array<const char*, 8> kNodeNameSharingOps = {
    "HashTable",
    "HashTableV2",
    "MutableHashTable",
    "MutableHashTableV2",
    "MutableDenseHashTable",
    "MutableDenseHashTableV2",
    "MutableHashTableOfTensors",
    "MutableHashTableOfTensorsV2",
};

// EASL: Worker metrics names
constexpr const char kBytesConsumed[] = "bytes_consumed";
constexpr const char kBytesProduced[] = "bytes_produced";
constexpr const char kNumElements[] = "num_elements";
constexpr const char kInNodeTime[] = "in_node_time";
constexpr const char kInPrefixTime[] = "in_prefix_time";
constexpr const char kBytesPerS[] = "bytes_per_s";
constexpr const char kActiveTime[] = "active_time";
constexpr const char kWorkingTime[] = "working_time";

const uint64 kElementThreshold = 300;
const bool kEnableEventLogging = true;

using DispatcherConfig = experimental::DispatcherConfig;
using Dataset = DispatcherState::Dataset;
using Worker = DispatcherState::Worker;
using NamedJobKey = DispatcherState::NamedJobKey;
using Job = DispatcherState::Job;
using Task = DispatcherState::Task;

std::string JournalDir(const std::string& work_dir) {
  return io::JoinPath(work_dir, kJournalDir);
}

std::string DatasetsDir(const std::string& work_dir) {
  return io::JoinPath(work_dir, kDatasetsDir);
}

std::string DatasetKey(int64_t id, uint64 fingerprint) {
  return absl::StrCat("id_", id, "_fp_", fingerprint);
}

Status CreateWorkerStub(const std::string& address, const std::string& protocol,
                        std::unique_ptr<WorkerService::Stub>& stub) {
  ::grpc::ChannelArguments args;
  args.SetMaxReceiveMessageSize(-1);
  std::shared_ptr<::grpc::ChannelCredentials> credentials;
  TF_RETURN_IF_ERROR(
      CredentialsFactory::CreateClientCredentials(protocol, &credentials));
  auto channel = ::grpc::CreateCustomChannel(address, credentials, args);
  stub = WorkerService::NewStub(channel);
  return Status::OK();
}

void PrepareGraph(GraphDef* graph) {
  for (NodeDef& node : *graph->mutable_node()) {
    for (const auto& op : kNodeNameSharingOps) {
      // Set `use_node_name_sharing` to `true` so that resources aren't deleted
      // prematurely. Otherwise, resources may be deleted when their ops are
      // deleted at the end of the GraphRunner::Run used by standalone::Dataset.
      if (node.op() == op) {
        (*node.mutable_attr())["use_node_name_sharing"].set_b(true);
      }
      if (!node.device().empty()) {
        *node.mutable_device() = "";
      }
    }
  }
  StripDevicePlacement(graph->mutable_library());
}

// EASL: Recording events
constexpr const char kEventFileLocation[] = "events.csv";
void RecordEvent(const uint64 fingerprint, const int64 dataset_id,
  const string& job_name, const int64 job_id, const string& event_type,
  const string& additional_info = "") {
  uint64 time_now = Env::Default()->NowMicros();

  std::ifstream in(kEventFileLocation);
  bool file_exists = in.good();
  in.close();

  std::ofstream o(kEventFileLocation, std::ios_base::app);
  if (!file_exists) {
    o << "time,fingerprint,dataset_id,job_name,job_id,event_type,additional\n";
  }

  o << time_now << "," << fingerprint << "," << dataset_id << "," << job_name
    << "," << job_id << "," << event_type << "," << additional_info << "\n";

  o.flush();
  o.close();
}

DispatcherConfig ApplyConfigDefaults(const DispatcherConfig& config) {
  DispatcherConfig new_config(config);
  if (new_config.job_gc_check_interval_ms() == 0) {
    new_config.set_job_gc_check_interval_ms(kDefaultJobGcCheckIntervalMs);
  }
  if (new_config.job_gc_timeout_ms() == 0) {
    new_config.set_job_gc_timeout_ms(kDefaultJobGcTimeoutMs);
  }
  if (new_config.client_timeout_ms() == 0) {
    new_config.set_client_timeout_ms(kDefaultClientTimeoutMs);
  }
  return new_config;
}

}  // namespace

DataServiceDispatcherImpl::DataServiceDispatcherImpl(
    const DispatcherConfig& config)
    : config_(ApplyConfigDefaults(config)),
      env_(Env::Default()),
      state_(config_) {
  if (config_.work_dir().empty()) {
    dataset_store_ = absl::make_unique<MemoryDatasetStore>();
  } else {
    dataset_store_ = absl::make_unique<FileSystemDatasetStore>(
        DatasetsDir(config_.work_dir()));
  }
}

DataServiceDispatcherImpl::~DataServiceDispatcherImpl() {
  {
    mutex_lock l(mu_);
    cancelled_ = true;
    job_gc_thread_cv_.notify_all();
  }
  job_gc_thread_.reset();
}

Status DataServiceDispatcherImpl::Start() {
  mutex_lock l(mu_);

  // EASL - Enable logging if a logging directory is provided:
  if(!config_.log_dir().empty()){
    env_->RecursivelyCreateDir(config_.log_dir());
    log_dumps_enabled_ = true;
    log_dumps_thread_ = absl::WrapUnique(
        env_->StartThread({}, "log-dumps-thread", [&] { LogDumpsThread(); }));
  }

  if (config_.job_gc_timeout_ms() >= 0) {
    job_gc_thread_ = absl::WrapUnique(
        env_->StartThread({}, "job-gc-thread", [&] { JobGcThread(); }));
  }

  if (config_.work_dir().empty()) {
    if (config_.fault_tolerant_mode()) {
      return errors::InvalidArgument(
          "fault_tolerant_mode is True, but no work_dir is configured.");
    }
  } else {
    TF_RETURN_IF_ERROR(
        env_->RecursivelyCreateDir(DatasetsDir(config_.work_dir())));
  }
  if (!config_.fault_tolerant_mode()) {
    LOG(INFO) << "Running with fault_tolerant_mode=False. The dispatcher will "
                 "not be able to recover its state on restart.";
    started_ = true;
    return Status::OK();
  }
  journal_writer_ = absl::make_unique<FileJournalWriter>(
      env_, JournalDir(config_.work_dir()));
  LOG(INFO) << "Attempting to restore dispatcher state from journal in "
            << JournalDir(config_.work_dir());
  Update update;
  bool end_of_journal = false;
  FileJournalReader reader(env_, JournalDir(config_.work_dir()));
  Status s = reader.Read(update, end_of_journal);
  if (errors::IsNotFound(s)) {
    LOG(INFO) << "No journal found. Starting dispatcher from new state.";
  } else if (!s.ok()) {
    return s;
  } else {
    while (!end_of_journal) {
      TF_RETURN_IF_ERROR(ApplyWithoutJournaling(update));
      TF_RETURN_IF_ERROR(reader.Read(update, end_of_journal));
    }
  }
  for (const auto& job : state_.ListJobs()) {
    if (IsDynamicShard(job->processing_mode)) {
      TF_RETURN_IF_ERROR(
          RestoreSplitProviders(*job, split_providers_[job->job_id]));
    }
  }
  for (const auto& client_id : state_.ListActiveClientIds()) {
    // Conservatively pretend we just received a heartbeat from all clients, so
    // that we don't garbage collect jobs too early.
    latest_client_heartbeats_time_[client_id] =
        absl::FromUnixMicros(env_->NowMicros());
  }
  // Initialize the journal writer in `Start` so that we fail fast in case it
  // can't be initialized.
  TF_RETURN_IF_ERROR(journal_writer_.value()->EnsureInitialized());
  started_ = true;
  return Status::OK();
}

size_t DataServiceDispatcherImpl::NumActiveJobs() TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  int64 count = 0;
  for (const auto& job : state_.ListJobs()) {
    if (!job->finished) {
      count++;
    }
  }
  return count;
}

Status DataServiceDispatcherImpl::RestoreSplitProviders(
    const Job& job, std::vector<std::unique_ptr<SplitProvider>>& restored)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  const std::vector<int64_t>& indices =
      job.distributed_epoch_state.value().indices;
  std::vector<std::unique_ptr<SplitProvider>> split_providers;
  TF_RETURN_IF_ERROR(MakeSplitProviders(job.dataset_id, job.job_type, split_providers));
  for (int provider_index = 0; provider_index < indices.size();
       ++provider_index) {
    int index = indices[provider_index];
    VLOG(1) << "Restoring split provider " << provider_index << " for job "
            << job.job_id << " to index " << index;
    Tensor unused_tensor;
    bool unused_end_of_splits;
    for (int i = 0; i < index; ++i) {
      TF_RETURN_IF_ERROR(split_providers[provider_index]->GetNext(
          &unused_tensor, &unused_end_of_splits));
    }
  }
  restored = std::move(split_providers);
  return Status::OK();
}

Status DataServiceDispatcherImpl::FindTasksToDelete(
    const absl::flat_hash_set<int64_t>& current_tasks,
    const std::vector<std::shared_ptr<const Task>> assigned_tasks,
    WorkerHeartbeatResponse* response) {
  absl::flat_hash_set<int64_t> assigned_ids;
  for (const auto& assigned : assigned_tasks) {
    assigned_ids.insert(assigned->task_id);
  }
  for (int64_t current_task : current_tasks) {
    if (!assigned_ids.contains(current_task)) {
      response->add_tasks_to_delete(current_task);
    }
  }
  return Status::OK();
}

Status DataServiceDispatcherImpl::FindNewTasks(
    const std::string& worker_address,
    const absl::flat_hash_set<int64_t>& current_tasks,
    std::vector<std::shared_ptr<const Task>>& assigned_tasks,
    WorkerHeartbeatResponse* response) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  // Check for round-robin jobs that had tasks on the worker removed. Now that
  // the worker is back, we create a new pending task for the worker.
  absl::flat_hash_set<int64_t> assigned_job_ids;
  for (const auto& task : assigned_tasks) {
    assigned_job_ids.insert(task->job->job_id);
  }
  for (const auto& job : state_.ListJobsForWorker(worker_address)) {
    if (!assigned_job_ids.contains(job->job_id) && job->IsRoundRobin() &&
        !job->finished) {
      if(job->IsRoundRobin()) {
        VLOG(1) << "Creating pending task for reconnected worker "
                << worker_address;
        TF_RETURN_IF_ERROR(CreatePendingTask(job, worker_address));
      }
      /*} else {
        VLOG(0) << "EASL - Creating Task for free worker";
        TF_RETURN_IF_ERROR(CreateTask(job, worker_address));
      }*/
    }
  }
  // Refresh assigned_tasks to include newly added pending tasks.
  TF_RETURN_IF_ERROR(state_.TasksForWorker(worker_address, assigned_tasks));
  for (const auto& task : assigned_tasks) {
    if (current_tasks.contains(task->task_id)) {
      continue;
    }
    TaskDef* task_def = response->add_new_tasks();
    TF_RETURN_IF_ERROR(PopulateTaskDef(task, task_def));
  }
  return Status::OK();
}

Status DataServiceDispatcherImpl::ReassignFreeWorkersAndCreateTasks() TF_LOCKS_EXCLUDED(mu_) {
  std::vector<std::shared_ptr<const Task>> tasks;
  // Get list of free workers
  {
    mutex_lock l(mu_);

    std::vector<std::shared_ptr<const DispatcherState::Worker>>old_avail_workers =
        state_.ListAvailableWorkers();
    // Reassign workers
    Update reassign_update;
    reassign_update.mutable_reassign_free_workers()->set_set(true);
    TF_RETURN_IF_ERROR(state_.Apply(reassign_update));
    // Create tasks if needed.
    for (const auto& worker : old_avail_workers) {
      // Create list of already assigned tasks
      std::vector<std::shared_ptr<const Task>> tasks_for_worker;
      Status s = state_.TasksForWorker(worker->address, tasks_for_worker);
      absl::flat_hash_set<int64> assigned_job_ids;
      if (!errors::IsNotFound(s)) {
        for (const auto& task : tasks_for_worker) {
          assigned_job_ids.insert(task->job->job_id);
        }
      }
      // Create task if worker is assigned to job (but has no task for it yet)
      for (const auto& new_job : state_.ListJobsForWorker(worker->address)){
        if (!assigned_job_ids.contains(new_job->job_id) && !new_job->finished){
          // CreateTask
          std::shared_ptr<const Task> task;
          TF_RETURN_IF_ERROR(CreateTask(new_job, worker->address, task));
          tasks.push_back(task);
        }
      }
    }
  }
  TF_RETURN_IF_ERROR(AssignTasks(tasks));
  VLOG(0) << "EASL - Reassigned free workers and created " << tasks.size() << " new tasks";

  return Status::OK();
};

Status DataServiceDispatcherImpl::WorkerHeartbeat(
    const WorkerHeartbeatRequest* request, WorkerHeartbeatResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  VLOG(0) << "Received worker heartbeat request from worker "
          << request->worker_address();
  mutex_lock l(mu_);
  const std::string& worker_address = request->worker_address();
  // Assigned tasks from the perspective of the dispatcher.
  std::vector<std::shared_ptr<const Task>> assigned_tasks;
  Status s = state_.TasksForWorker(worker_address, assigned_tasks);
  if (!s.ok()) {
    if (!errors::IsNotFound(s)) {
      return s;
    }
    VLOG(0) << "Registering new worker at address " << worker_address;
    TF_RETURN_IF_ERROR(state_.ValidateWorker(worker_address));
    Update update;
    update.mutable_register_worker()->set_worker_address(worker_address);
    update.mutable_register_worker()->set_transfer_address(
        request->transfer_address());
    *update.mutable_register_worker()->mutable_worker_tags() =
        request->worker_tags();
    update.mutable_register_worker()->set_worker_uid(request->worker_uid());
    TF_RETURN_IF_ERROR(Apply(update));
    TF_RETURN_IF_ERROR(CreateTasksForWorker(worker_address));
    TF_RETURN_IF_ERROR(state_.TasksForWorker(worker_address, assigned_tasks));
  }
  absl::flat_hash_set<int64_t> current_tasks;
  current_tasks.insert(request->current_tasks().cbegin(),
                       request->current_tasks().cend());
  TF_RETURN_IF_ERROR(
      FindTasksToDelete(current_tasks, assigned_tasks, response));
  TF_RETURN_IF_ERROR(
      FindNewTasks(worker_address, current_tasks, assigned_tasks, response));

  // EASL - Update the metadata with the incoming metrics
  for (int i = 0; i < request->tasks_size(); ++i) {
    VLOG(0) << "Updating metadata";
    auto task = request->tasks(i);
    
    // Get the job for this task
    std::shared_ptr<const Task> task_object;
    Status s = state_.TaskFromId(task.id(), task_object);

    if (s.ok() && !task_object->finished) {
      auto job_id = task_object->job->job_id;
      std::string last_node_name = task.last_node_name();
      std::string last_tf_node_name = task.last_tf_node_name();
      std::string marker_node_name = task.marker_node_name();
      s = metadata_store_.UpdateNodeNames(job_id, last_node_name, 
        last_tf_node_name, marker_node_name);

      // VLOG(0) << "(WorkerHeartbeat) For job with id " 
      //         << task_object->job->job_id << " we have the following relevant "
      //         << "node names\n"
      //         << " > last_node_name = " << last_node_name << "\n"
      //         << " > last_tf_node_name = " << last_tf_node_name << "\n"
      //         << " > marker_node_name = " << marker_node_name;

      if(!s.ok()) {
        // Ignore metrics if job has already been removed from metadata store.
        // Otherwise return status error.
        VLOG(0) << "(DataServiceDispatcherImpl::WorkerHeartbeat) "
                     << "Could not update node names " << s.error_message();
        if(!errors::IsNotFound(s)){ return s; }
      } else {
        for (int j = 0; j < task.nodes_size(); ++j) {
          auto metrics = task.mutable_nodes(j)->mutable_metrics();
          easl::NodeMetrics::Metrics node_metrics((*metrics)[kBytesConsumed], 
            (*metrics)[kBytesProduced], (*metrics)[kNumElements], 
            (*metrics)[kBytesPerS], (*metrics)[kInNodeTime],
            (*metrics)[kInPrefixTime], (*metrics)[kActiveTime],
            (*metrics)[kWorkingTime]);

          // VLOG(0) << "(Dispatcher::WorkerHeartbeat) Metrics for node " 
          //         << task.mutable_nodes(j)->name();
          // node_metrics.log_metrics();

//          TF_RETURN_IF_ERROR(metadata_store_.UpdateInputPipelineMetrics(job_id,
//            task.mutable_nodes(j)->name(), request->worker_address(),
//            node_metrics));
          s = metadata_store_.UpdateInputPipelineMetrics(job_id,
            task.mutable_nodes(j)->name(), request->worker_address(),
            node_metrics);

          if (!s.ok()) {
            VLOG(0) << "(DataServiceDispatcherImpl::WorkerHeartbeat) "
                    << "Could not update pipeline metrics " << s.error_message();
          }
        }

        // Try to see if we need to decide on the execution mode
        string job_type;
        uint64 element_count;
        Status s1 = metadata_store_.GetJobTypeByJobId(job_id, job_type);
        Status s2 = metadata_store_.GetNumberOfProducedElements(job_id,
                                                                element_count);

        if (s1.ok() && s2.ok() && job_type == "PROFILE" &&
            element_count >= kElementThreshold) {


          VLOG(0) << "(WorkerHeartbeat) At least "
                  << kElementThreshold << " elements have been produced";
          // Will change the job_type of job with job_id to something else
          service::easl::cache_utils::DetermineJobTypeUpdated(config_,
                                                              cache_state_, metadata_store_, job_id);

          // We will allow the job to start scaling
          // Note that job is expected to start at 1 worker
          VLOG(0) << "(WorkerHeartbeat) Enabling scaling";
          metadata_store_.ResetSameScaleCounter(job_id);
          metadata_store_.SetJobIsScaling(job_id);

          // Logging stuff
          if (kEnableEventLogging) {
            std::shared_ptr<const Dataset> dataset;
            state_.DatasetFromId(task_object->job->dataset_id, dataset);

            string job_type;
            string job_name = task_object->job->named_job_key->name;
            Status s3 = metadata_store_.GetJobTypeByJobId(job_id, job_type);

            if (s3.ok()) {
              RecordEvent(dataset->fingerprint, dataset->dataset_id, job_name,
                          task_object->job->job_id, "execution_policy_decision",
                          job_type);
            }
          }
        }

        // If the job produced enough elements calculate the metrics for the AutoOrder policy
        bool is_ordered;
        metadata_store_.IsJobOrdered(job_id, is_ordered);
        if (element_count >= kElementThreshold && !is_ordered) {
          metadata_store_.SetJobIsOrdered(job_id);

          // Update the info for the AutoOrder policy
          std::vector<std::string> pipeline_nodes;
          std::vector<float> inflation_factors;
          VLOG(0) << "About to determine inflation factors based on the first " << element_count << " elements.";
          Status s = service::easl::ordering_utils::DetermineInflationFactors(
              metadata_store_, pipeline_nodes, inflation_factors, job_id);
          if (!s.ok()) {
            VLOG(0) << "Determining inflation factors failed";
          }
          std::shared_ptr<const Dataset> ds;
          TF_RETURN_IF_ERROR(state_.DatasetFromId(task_object->job->dataset_id, ds));
          VLOG(0) << "dispatcher_impl: Got " << inflation_factors.size() << " inflation factors";
          VLOG(0) << "Check: pipeline_nodes now has " << pipeline_nodes.size();
          VLOG(0) << "Going to update the 'order_state_";
          order_state_.UpdateLatestInfFactors(ds->fingerprint, pipeline_nodes,
                                              inflation_factors);
          VLOG(0) << "Updated order state";
        }
        else {
          VLOG(0) << "So far " << element_count << " elements produced";
        }
      }
    }
  }

  VLOG(0) << "Finished worker heartbeat for worker at address "
          << request->worker_address();
  return Status::OK();
}

Status DataServiceDispatcherImpl::WorkerUpdate(
    const WorkerUpdateRequest* request, WorkerUpdateResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  bool do_reassign_free_workers = false;
  {
    mutex_lock l(mu_);

    for (auto& update : request->updates()) {
      int64 task_id = update.task_id();
      std::shared_ptr<const Task> task;
      TF_RETURN_IF_ERROR(state_.TaskFromId(task_id, task));
      if (update.completed()) {
        if (task->finished) {
          VLOG(1) << "Received completion update for already-finished task "
                  << task->task_id << " on worker " << task->worker_address;
          continue;
        }
        Update update;
        update.mutable_finish_task()->set_task_id(task_id);
        TF_RETURN_IF_ERROR(Apply(update));

        // EASL - Set dataset as cached if this was a caching job and the job is finished.
        std::shared_ptr<Job> job = task->job;
        if(job->job_type == "PUT" && job->finished) {
          std::shared_ptr<const Dataset> dataset;
          TF_RETURN_IF_ERROR(state_.DatasetFromId(task->job->dataset_id, dataset));
          cache_state_.SetDatasetCached(dataset->fingerprint);

          VLOG(0) << "Dataset with fingerprint " << dataset->fingerprint
                  << "has been added to cache.";
        } else if(job->job_type == "PUT_SOURCE" && job->finished){
          std::shared_ptr<const Dataset> dataset;
          TF_RETURN_IF_ERROR(state_.DatasetFromId(task->job->dataset_id, dataset));
          cache_state_.SetDatasetSourceCached(dataset->fingerprint);

          VLOG(0) << "Dataset with fingerprint " << dataset->fingerprint
                  << "has been added to source cache.";
        }
        // Update metadata store directly, quicker than waiting for the GCOldJobs to run.
        if(job->finished){
          do_reassign_free_workers = true;
          TF_RETURN_IF_ERROR(metadata_store_.UpdateFingerprintNameKeyJobMetrics(
              job->job_id));
          if(log_dumps_enabled_){
            TF_RETURN_IF_ERROR(metadata_store_.DumpJobMetricsToFile(job->job_id, config_.log_dir()));
            easl::TerminateJobMetricsAppendDumps(job->job_id, config_.log_dir());
          }
          TF_RETURN_IF_ERROR(metadata_store_.RemoveJob(job->job_id));

        }

        // TODO revert to 3
        VLOG(0) << "Task " << task_id << " from job " << task->job->job_id
                << " completed";
      }
    }
  }
  if (do_reassign_free_workers) {
    TF_RETURN_IF_ERROR(ReassignFreeWorkersAndCreateTasks());
  }
  return Status::OK();
}

Status DataServiceDispatcherImpl::GetDatasetDef(
    const GetDatasetDefRequest* request, GetDatasetDefResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  mutex_lock l(mu_);
  std::shared_ptr<const Dataset> dataset;
  TF_RETURN_IF_ERROR(state_.DatasetFromId(request->dataset_id(), dataset));

  // TODO (damien-aymon) The request should have the dataset key instead of the dataset_id.
  return errors::PermissionDenied("EASL - dispatcher_impl.cc:411: Should not enter here for now...");
//  std::shared_ptr<const DatasetDef> dataset_def;
//  TF_RETURN_IF_ERROR(GetDatasetDef(*dataset, dataset_def));
//  *response->mutable_dataset_def() = *dataset_def;
//  return Status::OK();
}

Status DataServiceDispatcherImpl::GetSplit(const GetSplitRequest* request,
                                           GetSplitResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  mutex_lock l(mu_);
  int64_t job_id = request->job_id();
  int64_t task_id = request->task_id();
  int64_t repetition = request->repetition();
  int64_t provider_index = request->split_provider_index();
  VLOG(3) << "Received GetSplit request for job " << job_id << ", repetition "
          << repetition << ", split provider index " << provider_index;
  std::shared_ptr<const Job> job;
  TF_RETURN_IF_ERROR(state_.JobFromId(job_id, job));
  if (!job->distributed_epoch_state.has_value()) {
    return errors::FailedPrecondition(
        "Cannot get split for job ", job_id,
        ", since it is not a distributed_epoch job.");
  }
  int64_t current_repetition =
      job->distributed_epoch_state.value().repetitions[provider_index];
  if (repetition < current_repetition) {
    response->set_end_of_splits(true);
    VLOG(0) << "Returning end_of_splits since current repetition "
            << current_repetition
            << " is greater than the requested repetition " << repetition;
    return Status::OK();
  }

  // EASL - Check if this is not a "early ended" task
  bool is_early_ended;
  TF_RETURN_IF_ERROR(state_.IsEarlyEndedTask(job_id, task_id, is_early_ended));
  if (is_early_ended) {
    VLOG(0) << "EASL - Split provider returning eos for early terminated task " << task_id;
    response->set_end_of_splits(true);
    return Status::OK();
  }

  SplitProvider* split_provider =
      split_providers_[job_id][provider_index].get();
  DCHECK(split_provider != nullptr);
  Tensor split;
  bool end_of_splits = false;
  TF_RETURN_IF_ERROR(split_provider->GetNext(&split, &end_of_splits));
  bool scaling;
  string execution_mode;
  TF_RETURN_IF_ERROR(metadata_store_.IsJobScaling(job_id, scaling));
  TF_RETURN_IF_ERROR(metadata_store_.GetJobTypeByJobId(job_id, execution_mode));
  // FIXME: Make sure to keep an eye out for the 2nd part of this condition
  //        It should not block scaling for a new client's job if the data
  //        is cached; still make sure this makes sense
  if (end_of_splits && scaling && execution_mode != "PUT"
    && execution_mode != "PUT_SOURCE") {
    split_provider->Reset();
    split_provider->GetNext(&split, &end_of_splits);
    VLOG(0) << "(GetSplit) Reached EOS while still scaling in " << job_id
                 << " at provider index " << provider_index;

    // EASL: Logging stuff
    if (kEnableEventLogging) {
      std::shared_ptr<const Dataset> dataset;
      state_.DatasetFromId(job->dataset_id, dataset);
      string job_name = job->named_job_key->name;
      RecordEvent(dataset->fingerprint, dataset->dataset_id, job_name, job_id,
                  "extended_epoch");
    }
  }

  TF_RETURN_IF_ERROR(RecordSplitProduced(
      job_id, repetition, request->split_provider_index(), end_of_splits));
  response->set_end_of_splits(end_of_splits);
  if (end_of_splits) {
    VLOG(0) << "EASL - (GetSplit) split provider reached eos for job " << job_id
    << " and task " << task_id;
    // Reset the split provider to prepare for the next repetition.
    TF_RETURN_IF_ERROR(split_providers_[job_id][provider_index]->Reset());
  } else {
    split.AsProtoTensorContent(response->mutable_split());
  }
  VLOG(3) << "Returning from GetSplit, end_of_splits=" << end_of_splits;
  return Status::OK();
}

Status DataServiceDispatcherImpl::MakeSplitProviders(
    int64_t dataset_id,
    const std::string& job_type,
    std::vector<std::unique_ptr<SplitProvider>>& split_providers)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::shared_ptr<const Dataset> dataset;
  TF_RETURN_IF_ERROR(state_.DatasetFromId(dataset_id, dataset));
  std::shared_ptr<const DatasetDef> dataset_def;
  TF_RETURN_IF_ERROR(GetDatasetDef(*dataset, job_type, dataset_def));
  standalone::Dataset::Params params;
  std::unique_ptr<standalone::Dataset> standalone_dataset;
  TF_RETURN_IF_ERROR(standalone::Dataset::FromGraph(
      params, dataset_def->graph(), &standalone_dataset));
  TF_RETURN_IF_ERROR(standalone_dataset->MakeSplitProviders(&split_providers));
  return Status::OK();
}

Status DataServiceDispatcherImpl::GetVersion(const GetVersionRequest* request,
                                             GetVersionResponse* response) {
  response->set_version(kDataServiceVersion);
  return Status::OK();
}

Status DataServiceDispatcherImpl::GetOrRegisterDataset(
    const GetOrRegisterDatasetRequest* request,
    GetOrRegisterDatasetResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  uint64 fingerprint;
  DatasetDef dataset_def = request->dataset();
  GraphDef* graph = dataset_def.mutable_graph();
  PrepareGraph(graph);
  TF_RETURN_IF_ERROR(HashGraph(*graph, &fingerprint));

  mutex_lock l(mu_);
#if defined(PLATFORM_GOOGLE)
  VLOG_LINES(4,
             absl::StrCat("Registering dataset graph: ", graph->DebugString()));
#else
  VLOG(4) << "Registering dataset graph: " << graph->DebugString();
#endif
  std::shared_ptr<const Dataset> dataset;
  Status s = state_.DatasetFromFingerprint(fingerprint, dataset);
  if (s.ok()) {
    int64_t id = dataset->dataset_id;
    VLOG(3) << "Received duplicate RegisterDataset request with fingerprint "
            << fingerprint << ". Returning id " << id;
    response->set_dataset_id(id);
    return Status::OK();
  } else if (!errors::IsNotFound(s)) {
    return s;
  }

  int64_t id;
  TF_RETURN_IF_ERROR(
      RegisterDataset(fingerprint, dataset_def, request->metadata(), id));
  if (!request->metadata().element_spec().empty()) {
    TF_RETURN_IF_ERROR(SetElementSpec(id, request->metadata().element_spec()));
  }

  response->set_dataset_id(id);
  VLOG(3) << "Registered new dataset with id " << id;

  return Status::OK();
}

Status DataServiceDispatcherImpl::RegisterDataset(
    uint64 fingerprint, const DatasetDef& dataset,
    const DataServiceMetadata& metadata, int64_t& dataset_id)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  dataset_id = state_.NextAvailableDatasetId();
  Update update;
  RegisterDatasetUpdate* register_dataset = update.mutable_register_dataset();
  register_dataset->set_dataset_id(dataset_id);
  register_dataset->set_fingerprint(fingerprint);
  *register_dataset->mutable_metadata() = metadata;
  TF_RETURN_IF_ERROR(
      dataset_store_->Put(DatasetKey(dataset_id, fingerprint), dataset));

  // EASL - Create and store put/get versions of this dataset def.
  DatasetDef put_dataset;
  TF_RETURN_IF_ERROR(
      service::easl::cache_utils::AddPutOperator(
          dataset, fingerprint, config_, put_dataset));
  TF_RETURN_IF_ERROR(dataset_store_->Put(
      service::easl::cache_utils::DatasetPutKey(dataset_id, fingerprint),
          put_dataset));
  DatasetDef get_dataset;
  TF_RETURN_IF_ERROR(
      service::easl::cache_utils::AddGetOperator(
          dataset, fingerprint, config_, get_dataset));
  TF_RETURN_IF_ERROR(dataset_store_->Put(
      service::easl::cache_utils::DatasetGetKey(dataset_id, fingerprint),
          get_dataset));

  // EASL - Create and store put/get for source data of this dataset
  DatasetDef put_source_dataset;
  TF_RETURN_IF_ERROR(
      service::easl::cache_utils::AddPutOperatorAtMarker(
          dataset, fingerprint, "source_cache", config_, put_source_dataset));
  TF_RETURN_IF_ERROR(dataset_store_->Put(
      service::easl::cache_utils::DatasetPutSourceKey(dataset_id, fingerprint),
          put_source_dataset));
      
  DatasetDef get_source_dataset;
  TF_RETURN_IF_ERROR(
      service::easl::cache_utils::AddGetOperatorAtMarker(
          dataset, fingerprint, "source_cache", config_, get_source_dataset));
  TF_RETURN_IF_ERROR(dataset_store_->Put(
      service::easl::cache_utils::DatasetGetSourceKey(dataset_id, fingerprint),
          get_source_dataset));
  VLOG(0) << "Added put/get versions for dataset fingerprint " << fingerprint;

  // // TODO: Add ggeneration of all possible pipeline orderings here and store them
  // DatasetDef reordered_dataset;
  // TF_RETURN_IF_ERROR(
  //     service::easl::ordering_utils::AddGetOperatorAtMarker(
  //         dataset, fingerprint, "source_cache", config_, reordered_dataset));
  // TF_RETURN_IF_ERROR(dataset_store_->Put(
  //     service::easl::ordering_utils::DatasetGetSourceKey(dataset_id, fingerprint),
  //         reordered_dataset));

  return Apply(update);
}

Status DataServiceDispatcherImpl::SetElementSpec(
    int64_t dataset_id, const std::string& element_spec)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  Update update;
  SetElementSpecUpdate* set_element_spec = update.mutable_set_element_spec();
  set_element_spec->set_dataset_id(dataset_id);
  set_element_spec->set_element_spec(element_spec);
  TF_RETURN_IF_ERROR(Apply(update));
  return Status::OK();
}

Status DataServiceDispatcherImpl::GetElementSpec(
    const GetElementSpecRequest* request, GetElementSpecResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  mutex_lock l(mu_);
  VLOG(4) << "Read the element spec.";
  int64_t dataset_id = request->dataset_id();

  std::string element_spec;
  TF_RETURN_IF_ERROR(state_.GetElementSpec(dataset_id, element_spec));
  VLOG(3) << "Get the `element_spec` for registered dataset with dataset id: "
          << dataset_id << ".";
  *response->mutable_element_spec() = element_spec;
  return Status::OK();
}

Status DataServiceDispatcherImpl::GetDataServiceMetadata(
    const GetDataServiceMetadataRequest* request,
    GetDataServiceMetadataResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  int64_t dataset_id = request->dataset_id();
  std::shared_ptr<const Dataset> dataset;

  mutex_lock l(mu_);
  TF_RETURN_IF_ERROR(state_.DatasetFromId(dataset_id, dataset));
  VLOG(3) << "Get the data service metadata for dataset id: " << dataset_id
          << ".";
  *response->mutable_metadata() = dataset->metadata;
  return Status::OK();
}

Status DataServiceDispatcherImpl::GetOrCreateJob(
    const GetOrCreateJobRequest* request, GetOrCreateJobResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  VLOG(3) << "GetOrCreateJob(" << request->DebugString() << ")";
  absl::optional<NamedJobKey> key;
  if (request->has_job_key()) {
    key.emplace(request->job_key().job_name(),
                request->job_key().job_name_index());
  }
  std::shared_ptr<const Job> job;
  std::vector<std::shared_ptr<const Task>> tasks;
  {
    mutex_lock l(mu_);
    if (key.has_value()) {
      Status s = state_.NamedJobByKey(key.value(), job);
      if (s.ok()) {
        TF_RETURN_IF_ERROR(ValidateMatchingJob(job, *request));
        // If the matching job was already garbage-collected, we fall through to
        // re-create the job.
        if (!job->garbage_collected) {
          int64_t job_client_id;
          TF_RETURN_IF_ERROR(AcquireJobClientId(job, job_client_id));
          response->set_job_client_id(job_client_id);
          VLOG(3) << "Found existing job for name=" << key.value().name
                  << ", index=" << key.value().index
                  << ". job_id: " << job->job_id;
          return Status::OK();
        }
      } else if (!errors::IsNotFound(s)) {
        return s;
      }
    }
    TF_RETURN_IF_ERROR(CreateJob(*request, job));
    int64_t job_client_id;
    TF_RETURN_IF_ERROR(AcquireJobClientId(job, job_client_id));
    response->set_job_client_id(job_client_id);
    TF_RETURN_IF_ERROR(CreateTasksForJob(job, tasks));
  }
  TF_RETURN_IF_ERROR(AssignTasks(tasks));
  VLOG(3) << "Created job " << job->job_id << " for CreateJob("
          << request->DebugString() << ")";
  return Status::OK();
}

Status DataServiceDispatcherImpl::MaybeRemoveTask(
    const MaybeRemoveTaskRequest* request, MaybeRemoveTaskResponse* response) {
  VLOG(1) << "Attempting to remove task. Request: " << request->DebugString();
  std::shared_ptr<TaskRemover> remover;
  std::shared_ptr<const Task> task;
  {
    mutex_lock l(mu_);
    Status s = state_.TaskFromId(request->task_id(), task);
    if (errors::IsNotFound(s)) {
      // Task is already removed.
      response->set_removed(true);
      return Status::OK();
    }
    TF_RETURN_IF_ERROR(s);
    auto& remover_ref = remove_task_requests_[task->task_id];
    if (remover_ref == nullptr) {
      if (!task->job->IsRoundRobin()) {
        return errors::FailedPrecondition(
            "MaybeRemoveTask called on a non-round-robin task.");
      }
      remover_ref =
          std::make_shared<TaskRemover>(task->job->num_consumers.value());
    }
    remover = remover_ref;
  }
  bool removed =
      remover->RequestRemoval(request->consumer_index(), request->round());
  response->set_removed(removed);
  if (!removed) {
    VLOG(1) << "Failed to remove task " << task->task_id;
    return Status::OK();
  }
  mutex_lock l(mu_);
  if (!task->removed) {
    Update update;
    RemoveTaskUpdate* remove_task = update.mutable_remove_task();
    remove_task->set_task_id(request->task_id());
    TF_RETURN_IF_ERROR(Apply(update));
  }
  VLOG(1) << "Task " << task->task_id << " successfully removed";
  return Status::OK();
}

Status DataServiceDispatcherImpl::ReleaseJobClient(
    const ReleaseJobClientRequest* request,
    ReleaseJobClientResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  std::shared_ptr<const Job> job;
  {
    mutex_lock l(mu_);
    int64_t job_client_id = request->job_client_id();

    // TODO(DanGraur): Changed the scope of the lock from easl_2.7
    TF_RETURN_IF_ERROR(state_.JobForJobClientId(job_client_id, job));
    Update update;
    ReleaseJobClientUpdate* release_job_client =
        update.mutable_release_job_client();
    release_job_client->set_job_client_id(job_client_id);
    release_job_client->set_time_micros(env_->NowMicros());
    TF_RETURN_IF_ERROR(Apply(update));

    if (job->num_clients <=0) {
      Update update;
      update.mutable_garbage_collect_job()->set_job_id(job->job_id);
      TF_RETURN_IF_ERROR(state_.Apply(update));
      VLOG(0) << "EASL - (ReleaseJobClient): Overwrite job_gc_timeout_ms and garbage collect job "
              << job->DebugString();
    }
  }
  if (job->num_clients <= 0) {
    TF_RETURN_IF_ERROR(ReassignFreeWorkersAndCreateTasks());
  }

  return Status::OK();
}

// Validates that the job matches the requested processing mode.
Status DataServiceDispatcherImpl::ValidateMatchingJob(
    std::shared_ptr<const Job> job, const GetOrCreateJobRequest& request)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  DCHECK(job->named_job_key.has_value());
  std::string job_name = job->named_job_key->name;

  if (!MessageDifferencer::Equals(job->processing_mode,
                                  request.processing_mode_def())) {
    return errors::FailedPrecondition(
        "Tried to create a job with name ", job_name, " and processing_mode <",
        request.processing_mode_def().ShortDebugString(),
        "> but there is already an existing job with that name using "
        "processing mode <",
        job->processing_mode.ShortDebugString(), ">");
  }

  if (job->target_workers != request.target_workers()) {
    return errors::InvalidArgument(
        "Tried to create job with name ", job_name, " and target_workers <",
        TargetWorkersToString(request.target_workers()),
        ">, but there is already an existing job "
        "with that name using target_workers <",
        TargetWorkersToString(job->target_workers), ">.");
  }
  return Status::OK();
}

Status DataServiceDispatcherImpl::CreateJob(
    const GetOrCreateJobRequest& request, std::shared_ptr<const Job>& job)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  TF_RETURN_IF_ERROR(ValidateProcessingMode(request.processing_mode_def()));
  int64 job_id = state_.NextAvailableJobId();

  // EASL - Caching decision: should the job compute, write or read from cache?
  int64 dataset_id = request.dataset_id();
  int64 worker_count;
  std::string job_type;

  // EASL: Note that jobs are discarded if they are not named
  if (!request.has_job_key()) {
    return errors::FailedPrecondition("Jobs must be named");
  }

  string job_name = request.job_key().job_name();
  std::shared_ptr<const Dataset> dataset;
  TF_RETURN_IF_ERROR(state_.DatasetFromId(dataset_id, dataset));
  uint64 dataset_fingerprint = dataset->fingerprint;
  std::string compute_dataset_key = DatasetKey(dataset_id, dataset_fingerprint);

  service::easl::cache_utils::DetermineJobType(config_, cache_state_,
    metadata_store_, dataset_fingerprint, job_name, job_type);
  VLOG(0) << "(CreateJob) Caching decision for dataset_key "
               << compute_dataset_key << ": " << job_type;

  // Check Local Workers from Client
  absl::flat_hash_set<std::string> local_workers;
  local_workers.insert(request.local_workers().cbegin(),
    request.local_workers().cend());

  // EASL: Logging stuff
  if (kEnableEventLogging) {
    RecordEvent(dataset_fingerprint, dataset_id, job_name, job_id,
                "execution_mode_change", job_type);
  }

  // Check to see what the previous execution type for this job was
  string existing_job_type;
  Status s = metadata_store_.GetJobType(dataset_fingerprint, job_name,
    existing_job_type);

  // Forcefully trigger rescale if:
  //  * we've transitioned to a new execution type
  //  * we're putting anything into cache (this can only happen once after profiling)
  //  * TODO: we've reordered the input pipeline
  bool trigger_scaling = s.ok() && (existing_job_type != job_type ||
    job_type == "PUT" || job_type == "PUT_SOURCE");

  VLOG(0) << "Deciding whether to trigger the AutoOrder policy";
  //bool new_old_job = metadata_store_.JobSeenBefore();
  // EASL perform the AutoOrder mechanism
  std::shared_ptr<easl::JobMetrics> job_metrics;
  s = metadata_store_.GetJobMetrics(job_id, job_metrics);
  VLOG(0) << "Got the JobMetrics";
  if (!s.ok()) {
    VLOG(0) << "Fetching JobMetrics failed!";
  }

  // TODO: Find a way to figure out that an epoch has already run (or that that the AutoPlacement policy has finished)
  //std::shared_ptr<data::easl::JobMetrics> job_metrics;

  Status s_seen = metadata_store_.GetJobMetricsByDatasetFingerprintAndName(dataset_fingerprint, job_name, job_metrics);
  bool tmp_is_ordering;
  if (!s_seen.ok()){
    VLOG(0) << "We haven't seen this job/dataset before";
    metadata_store_.UnsetJobIsOrdering(job_id);
    tmp_is_ordering = false;
  } else {
    tmp_is_ordering = true;
    VLOG(0) << "Found historical metrics for this job";
    metadata_store_.SetJobIsOrdering(job_id);
    VLOG(0) << "In ordering state";
    auto mm = job_metrics->model_metrics_;
    VLOG(0) << "Got model metrics";
    auto mh = job_metrics->model_metrics_->metrics_history_;
    VLOG(0) << "Got metrics history";
    //auto mh_l = job_metrics->model_metrics_->metrics_history_.back();
    //VLOG(0) << "Got last metrics from history";
    //float l_b_time = job_metrics->model_metrics_->metrics_history_.back()->last_x_batch_time_ms();
    //VLOG(0) << "Last batch time" << job_metrics->model_metrics_->metrics_history_.back()->last_x_batch_time_ms();
    //VLOG(0) << "History length" << job_metrics->model_metrics_->metrics_history_.size();

    // Check whether ordering policy is on
    int o_policy = config_.order_policy();
    VLOG(0) << "Ordering policy is " << o_policy;
  }
  // For now just order after the 1st epoch passes (later on place AutoOrder after AutoPlacement policy)
  //easl::MetricsHistory metrics_history;
  //easl::ModelMetrics model_metrics;
  //Status s_hist = job_metrics->GetMetricsHistory(metrics_history);
  //auto model_metrics = job_metrics->model_metrics_;
  //VLOG(0) << "(Test) Got model metics!";
  //auto metrics_history = job_metrics->model_metrics_->metrics_history_;
  //VLOG(0) << "Got MetricsHistory!";
  //int hist_size = metrics_history.size();
  //VLOG(0) << "History has length (should correspond to no. of epochs): " << hist_size;
  bool is_ordering;
  metadata_store_.IsJobOrdering(job_id, is_ordering);
  VLOG(0) << "Job is ordering: " << is_ordering;
  if (tmp_is_ordering) {
  //if (trigger_scaling = s.ok() && is_ordering && existing_job_type == job_type){
    trigger_scaling = true;
    VLOG(0) << "DISPATCHER TRIGGERING AUTOORDER POLICY (should happen max 1x)!";
    std::vector<std::string> latest_pipeline;
    std::vector<float> inflation_factors;
    order_state_.GetLatestInfFactors(dataset_fingerprint, latest_pipeline, inflation_factors);
    // TODO: Peform actual reordering HERE!!!
    std::shared_ptr<const DatasetDef> dataset_def;
    TF_RETURN_IF_ERROR(GetDatasetDef(*dataset, job_type, dataset_def));
    DatasetDef reorderedDataset;
    service::easl::ordering_utils::OpOrderUpdate(job_type, job_id, config_, metadata_store_,
                                                 job_metrics->target_remote_worker_count_, *dataset_def,
                                                 latest_pipeline, inflation_factors, dataset_fingerprint,
                                                 reorderedDataset);

    // Only reorder once
    metadata_store_.UnsetJobIsOrdering(job_id);
  }

  // EASL add job entry to metadata store
  std::string dataset_key = service::easl::cache_utils::DatasetKey(
    dataset->dataset_id, dataset->fingerprint, job_type);
  if (config_.scaling_policy() == 3 || config_.scaling_policy() == 4) {
    TF_RETURN_IF_ERROR(metadata_store_.CreateJobName(job_id, job_name, job_type,
                                                     dataset->dataset_id,
                                                     dataset->fingerprint,
                                                     dataset_key, trigger_scaling,
                                                     config_.scaling_policy()));
  } else {
    TF_RETURN_IF_ERROR(metadata_store_.CreateJobName(job_id, job_name, job_type,
                                                     dataset->dataset_id,
                                                     dataset->fingerprint,
                                                     dataset_key, trigger_scaling));
  }


  //std::shared_ptr<easl::JobMetrics> job_metrics;
  s = metadata_store_.GetJobMetrics(job_id, job_metrics);
  worker_count = job_metrics->target_worker_count_;

  int64_t remote_worker_count = job_metrics->target_remote_worker_count_;
  int64_t local_worker_count = job_metrics->target_local_worker_count_;
  VLOG(0) << "EASL-MUYU (GetOrCreateJob): "
    << "remote_worker_count: " << remote_worker_count
    << "; local_worker_count: " << local_worker_count;

  if (config_.scaling_policy() == 2) {
    worker_count = 100;
  }

  // Increase the number of workers when puttiong to cache
//  if (job_type == "PUT" || job_type == "PUT_SOURCE") {
//    std::shared_ptr<easl::JobMetrics> dataset_fingerprint_metrics;
//    s = metadata_store_.GetJobMetricsByDatasetFingerprintAndName(
//        dataset_fingerprint, job_name, dataset_fingerprint_metrics);
//    if (s.ok()) {
//      worker_count = std::ceil(std::max(1.0,
//          dataset_fingerprint_metrics->target_worker_count_ * 1.5));
//    }
//    job_metrics->target_worker_count_ = worker_count;
//  }

  // EASL: Logging stuff
  if (kEnableEventLogging) {
    last_scale_[job_name] = worker_count;
    RecordEvent(dataset_fingerprint, dataset_id, job_name, job_id,
                "starting_worker_count", std::to_string(worker_count));
  }

  int64 num_split_providers = 0;
  if (IsDynamicShard(request.processing_mode_def())) {
    TF_RETURN_IF_ERROR(
        MakeSplitProviders(request.dataset_id(), job_type,
            split_providers_[job_id]));
    num_split_providers = split_providers_[job_id].size();
  }

  // TODO: is the second job going to inherit the previous epoch job metrics?
  Update update;
  CreateJobUpdate* create_job = update.mutable_create_job();
  create_job->set_job_id(job_id);
  create_job->set_dataset_id(dataset_id);
  *create_job->mutable_processing_mode_def() = request.processing_mode_def();
  create_job->set_job_type(job_type);
  create_job->set_num_split_providers(num_split_providers);
  create_job->set_target_worker_count(worker_count);
  // only used by policy 3, 4
  create_job->set_target_remote_worker_count(remote_worker_count);
  create_job->set_target_local_worker_count(local_worker_count);
  *create_job->mutable_local_workers() = {local_workers.begin(), local_workers.end()};

  for (auto worker: local_workers) {
    VLOG(0) << "EASL-MUYU (CreateJob) local_workers: " << worker;
  }

  if (request.has_job_key()) {
    NamedJobKeyDef* key = create_job->mutable_named_job_key();
    key->set_name(request.job_key().job_name());
    key->set_index(request.job_key().job_name_index());
  }
  const bool is_coordinated_read = (request.optional_num_consumers_case() ==
                                    GetOrCreateJobRequest::kNumConsumers);
  if (is_coordinated_read) {
    create_job->set_num_consumers(request.num_consumers());
  }
  create_job->set_target_workers(request.target_workers());
  TF_RETURN_IF_ERROR(Apply(update));
  TF_RETURN_IF_ERROR(state_.JobFromId(job_id, job));
  tensorflow::metrics::RecordTFDataServiceJobsCreated(
      request.processing_mode_def(), is_coordinated_read);
  return Status::OK();
}

Status DataServiceDispatcherImpl::CreateTasksForWorker(
    const std::string& worker_address) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  // EASL - reassign workers first
  Update reassign_update;
  reassign_update.mutable_reassign_free_workers()->set_set(true);
  TF_RETURN_IF_ERROR(state_.Apply(reassign_update));

  // Then create tasks
  std::vector<std::shared_ptr<const Job>> jobs = state_.ListJobsForWorker(
    worker_address);
  for (const auto& job : jobs) {
    if (job->finished) {
      continue;
    }
    if (job->num_consumers.has_value()) {
      TF_RETURN_IF_ERROR(CreatePendingTask(job, worker_address));
      continue;
    }
    std::shared_ptr<const Task> task;
    TF_RETURN_IF_ERROR(CreateTask(job, worker_address, task));
    VLOG(0) << "EASL - New task (job " << job->job_id <<
    ") created for joining worker at address " << worker_address;
  }

  return Status::OK();
}

Status DataServiceDispatcherImpl::AcquireJobClientId(
    const std::shared_ptr<const Job>& job, int64_t& job_client_id)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  job_client_id = state_.NextAvailableJobClientId();
  Update update;
  AcquireJobClientUpdate* acquire_job_client =
      update.mutable_acquire_job_client();
  acquire_job_client->set_job_client_id(job_client_id);
  acquire_job_client->set_job_id(job->job_id);
  TF_RETURN_IF_ERROR(Apply(update));
  // Does not release clients before they start to read from the dataset.
  latest_client_heartbeats_time_[job_client_id] = absl::InfiniteFuture();
  return Status::OK();
}

Status DataServiceDispatcherImpl::CreateTasksForJob(
    std::shared_ptr<const Job> job,
    std::vector<std::shared_ptr<const Task>>& tasks)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {

  std::vector<std::shared_ptr<const Worker>> workers;
  if (config_.scaling_policy() == 3 || config_.scaling_policy() == 4) {
    workers = state_.ReserveWorkers(job->job_id,
                                    job->target_remote_worker_count,
                                    job->target_local_worker_count);
  }
  else {
    // original branch
    workers = state_.ReserveWorkers(job->job_id, job->target_worker_count);
  }

  if (workers.size() < job->target_worker_count){
    VLOG(0)
    << "EASL - Not enough workers for job. Elasticity policy requires "
    << job->target_worker_count << " but got " << workers.size();
  }
  tasks.clear();
  tasks.reserve(workers.size());
  for (auto& worker : workers) {
    std::shared_ptr<const Task> task;
    TF_RETURN_IF_ERROR(CreateTask(job, worker->address, task));
    tasks.push_back(task);
  }
  return Status::OK();
}

Status DataServiceDispatcherImpl::CreatePendingTask(
    std::shared_ptr<const Job> job, const std::string& worker_address)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  int64_t task_id = state_.NextAvailableTaskId();
  Update update;
  CreatePendingTaskUpdate* create_task = update.mutable_create_pending_task();
  create_task->set_task_id(task_id);
  create_task->set_job_id(job->job_id);
  create_task->set_worker_address(worker_address);
  create_task->set_starting_round(round_robin_rounds_[job->job_id] + 1);
  std::shared_ptr<const Worker> worker;
  TF_RETURN_IF_ERROR(state_.WorkerFromAddress(worker_address, worker));
  create_task->set_transfer_address(worker->transfer_address);
  *create_task->mutable_worker_tags() = {worker->tags.begin(),
                                         worker->tags.end()};
  create_task->set_worker_uid(worker->uid);

  // TODO (damien-aymon) This is not entirely valid, we do not support cache with round-robin jobs yet.
  std::shared_ptr<const Dataset> dataset;
  TF_RETURN_IF_ERROR(state_.DatasetFromId(job->dataset_id, dataset));
  std::string dataset_key =
      service::easl::cache_utils::DatasetKey(dataset->dataset_id, dataset->fingerprint, job->job_type);
  create_task->set_dataset_key(dataset_key);

  TF_RETURN_IF_ERROR(Apply(update));
  return Status::OK();
}

Status DataServiceDispatcherImpl::CreateTask(std::shared_ptr<const Job> job,
                                             const std::string& worker_address,
                                             std::shared_ptr<const Task>& task)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  int64_t task_id = state_.NextAvailableTaskId();
  Update update;
  CreateTaskUpdate* create_task = update.mutable_create_task();
  create_task->set_task_id(task_id);
  create_task->set_job_id(job->job_id);
  create_task->set_worker_address(worker_address);
  std::shared_ptr<const Worker> worker;
  TF_RETURN_IF_ERROR(state_.WorkerFromAddress(worker_address, worker));
  create_task->set_transfer_address(worker->transfer_address);
  *create_task->mutable_worker_tags() = {worker->tags.begin(),
                                         worker->tags.end()};
  create_task->set_worker_uid(worker->uid);

  // EASL - Get the dataset_key depending on the job type:
  std::shared_ptr<const Dataset> dataset;
  TF_RETURN_IF_ERROR(state_.DatasetFromId(job->dataset_id, dataset));
  std::string dataset_key =
      service::easl::cache_utils::DatasetKey(dataset->dataset_id, dataset->fingerprint, job->job_type);
  create_task->set_dataset_key(dataset_key);

  TF_RETURN_IF_ERROR(Apply(update));
  TF_RETURN_IF_ERROR(state_.TaskFromId(task_id, task));

  return Status::OK();
}

Status DataServiceDispatcherImpl::AssignTasks(
    std::vector<std::shared_ptr<const Task>> tasks) TF_LOCKS_EXCLUDED(mu_) {
  for (const auto& task : tasks) {
    TF_RETURN_IF_ERROR(AssignTask(task));
  }
  return Status::OK();
}

Status DataServiceDispatcherImpl::GetOrCreateWorkerStub(
    const std::string& worker_address, WorkerService::Stub*& out_stub)
    TF_LOCKS_EXCLUDED(mu_) {
  {
    mutex_lock l(mu_);
    auto it = worker_stubs_.find(worker_address);
    if (it != worker_stubs_.end()) {
      out_stub = it->second.get();
      return Status::OK();
    }
  }
  std::unique_ptr<WorkerService::Stub> stub;
  TF_RETURN_IF_ERROR(
      CreateWorkerStub(worker_address, config_.protocol(), stub));
  {
    mutex_lock l(mu_);
    // A concurrent call could have already created the stub.
    auto& worker = worker_stubs_[worker_address];
    if (worker == nullptr) {
      worker = std::move(stub);
    }
    out_stub = worker.get();
  }
  return Status::OK();
}

Status DataServiceDispatcherImpl::AssignTask(std::shared_ptr<const Task> task)
    TF_LOCKS_EXCLUDED(mu_) {
  VLOG(0) << "Started assigning task " << task->task_id << " to worker "
          << task->worker_address;
  grpc::ClientContext client_ctx;
  ProcessTaskRequest req;
  TaskDef* task_def = req.mutable_task();
  {
    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(PopulateTaskDef(task, task_def));
  }
  ProcessTaskResponse resp;
  WorkerService::Stub* stub;
  TF_RETURN_IF_ERROR(GetOrCreateWorkerStub(task->worker_address, stub));
  grpc::Status s = stub->ProcessTask(&client_ctx, req, &resp);
  if (!s.ok()) {
    if (s.error_code() == grpc::StatusCode::UNAVAILABLE ||
        s.error_code() == grpc::StatusCode::ABORTED ||
        s.error_code() == grpc::StatusCode::CANCELLED) {
      // Worker is presumably preempted. We will assign the task to the worker
      // when it reconnects.
      return Status::OK();
    }
    return grpc_util::WrapError(
        absl::StrCat("Failed to submit task to worker ", task->worker_address),
        s);
  }
  VLOG(0) << "Finished assigning task " << task->task_id << " to worker "
          << task->worker_address;
  return Status::OK();
}

Status DataServiceDispatcherImpl::ClientHeartbeat(
    const ClientHeartbeatRequest* request, ClientHeartbeatResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  bool do_reassign_workers = false;
  mutex_lock l(mu_);
  VLOG(0) << "Received heartbeat from client id " << request->job_client_id();

  std::shared_ptr<const Job> job;
  Status s = state_.JobForJobClientId(request->job_client_id(), job);

  if (errors::IsNotFound(s) && !config_.fault_tolerant_mode()) {
    return errors::NotFound(
        "Unknown job client id ", request->job_client_id(),
        ". The dispatcher is not configured to be fault tolerant, so this "
        "could be caused by a dispatcher restart.");
  }
  TF_RETURN_IF_ERROR(s);

  // EASL: Update the client metrics
  int64 job_target_worker_count;
  string job_type;
  string job_name = job->named_job_key->name;
  metadata_store_.GetJobTypeByJobId(job->job_id, job_type);
  // FIXME: Note that we're only checking the first split provider
  if (config_.scaling_policy() == 1 &&
      job_type != "PROFILE" &&
      request->has_scalability_metrics() &&
      job->distributed_epoch_state.value().repetitions[0] == 0) {
    easl::ModelMetrics::Metrics metrics(
      request->worker_count(),
      request->last_x_batch_time_ms(),
      request->relative_wait_fraction(),
      request->result_queue_size());

    s = metadata_store_.UpdateModelMetrics(
        job->job_id, request->job_client_id(), metrics);
    // Ignore metrics for jobs which do not have metrics anymore
    // report error otherwise.
    if (!s.ok()){
      VLOG(0) << "EASL (ClientHeartbeat) - metadatastore error code " << s.code();
      VLOG(0) << s.ToString();
      VLOG(0) << errors::IsNotFound(s);
    }
    if (!s.ok() && !errors::IsNotFound(s)) { return s; }

    // EASL - Determine updated target number of workers
    int64 target_worker_count;
    TF_RETURN_IF_ERROR(
        service::easl::scaling_utils::DynamicWorkerCountUpdate(
            job->job_type, job->job_id, config_, metadata_store_, target_worker_count));
    do_reassign_workers = target_worker_count > job->current_worker_count;
    VLOG(0) << "(ClientHeartbeat) After DynamicWorkerCountUpdate; "
            << "target_worker_count = " << target_worker_count
            << "; job->target_worker_count = " << job->target_worker_count;

    // Re-adjust the worker count in the metrics is not equal to what we
    // want it to be
    if (target_worker_count != metrics.worker_count()) {
      Update update;
      JobTargetWorkerCountUpdate *job_target_worker_count_update =
          update.mutable_job_target_worker_count_update();
      job_target_worker_count_update->set_job_id(job->job_id);
      job_target_worker_count_update->set_target_worker_count(target_worker_count);
      state_.Apply(update);

      // EASL: Logging stuff
      if (kEnableEventLogging &&
        last_scale_[job_name] != target_worker_count) {
        string scale_type = target_worker_count > last_scale_[job_name] ?
                            "scale_up" : "scale_down";
        last_scale_[job_name] = target_worker_count;
        std::shared_ptr<const Dataset> dataset;
        TF_RETURN_IF_ERROR(state_.DatasetFromId(job->dataset_id, dataset));
        RecordEvent(dataset->fingerprint, dataset->dataset_id,
          job->named_job_key->name, job->job_id, scale_type,
          std::to_string(target_worker_count));
      }
    }
  } else if (config_.scaling_policy() == 2) {
    metadata_store_.UnsetJobIsScaling(job->job_id);
    int64 target_worker_count = state_.ListWorkers().size();
    if (job->target_worker_count != target_worker_count) {
      Update update;
      JobTargetWorkerCountUpdate *job_target_worker_count_update =
              update.mutable_job_target_worker_count_update();
      job_target_worker_count_update->set_job_id(job->job_id);
      job_target_worker_count_update->set_target_worker_count(
              target_worker_count);
      state_.Apply(update);
    }
  } else if ((config_.scaling_policy() == 3 || config_.scaling_policy() == 4) &&
              request->has_scalability_metrics() &&
              job->distributed_epoch_state.value().repetitions[0] == 0) {
      easl::ModelMetrics::Metrics metrics(
              request->worker_count(),
              request->local_worker_count(),
              request->last_x_batch_time_ms(),
              request->relative_wait_fraction(),
              request->result_queue_size());

//      VLOG(0) << "EASL-MUYU: ClientHeartbeat: "
//              << "remote_worker_count: " << metrics.remote_worker_count()
//              << "; local_worker_count: " << metrics.local_worker_count();

      s = metadata_store_.UpdateModelMetrics(
              job->job_id, request->job_client_id(), metrics);
      if(!s.ok()){
        VLOG(0) << "EASL (ClientHeartbeat) - metadatastore error code " << s.code();
        VLOG(0) << s.ToString();
        VLOG(0) << errors::IsNotFound(s);
      }
      if (!s.ok() && !errors::IsNotFound(s)) { return s; }
      int64 available_workers = state_.ListAvailableWorkers().size();
      std::vector<std::shared_ptr<const Worker>> free_workers = state_.ListAvailableWorkers();
      int64 loc_w_available = 0;
      int64 rem_w_available = 0;
      /*for (int i = 0; i < available_workers; ++i) {
        auto address = free_workers[i].address;
        if (address.find("localhost") != std::string::npos) {
          loc_w_available++;
        } else {
          rem_w_available++;
        }
      }
      VLOG(0) << "Remote workers available: " << rem_w_available << " Loc workers available: " << loc_w_available;*/
      int64 target_remote_worker_count, target_local_worker_count;
      if (config_.scaling_policy() == 3) {
        TF_RETURN_IF_ERROR(service::easl::local_worker_decision::DynamicWorkerCountUpdateWithLocal_INCDEC(
                job->job_type, job->job_id, config_, metadata_store_,
                target_remote_worker_count, target_local_worker_count, available_workers));
      }
//      else if (config_.scaling_policy() == 4){
//        TF_RETURN_IF_ERROR(service::easl::local_worker_decision::DynamicWorkerCountUpdateWithLocal_INCINC(
//                job->job_type, job->job_id, config_, metadata_store_,
//                target_remote_worker_count, target_local_worker_count));
//      }


      if (target_remote_worker_count != metrics.remote_worker_count() ||
        target_local_worker_count != metrics.local_worker_count()) {
        Update update;
        JobTargetWorkerCountUpdateRemoteAndLocal *job_target_worker_count_update =
                update.mutable_job_target_worker_count_update_remote_and_local();
        job_target_worker_count_update->set_job_id(job->job_id);
        job_target_worker_count_update->set_target_remote_worker_count(target_remote_worker_count);
        job_target_worker_count_update->set_target_local_worker_count(target_local_worker_count);
        state_.Apply(update);

        // state_.Apply only sets the job attributes, still need to directly create tasks
        // assign new workers
        absl::flat_hash_map<std::string, std::shared_ptr<Worker>> candidate_workers;
        state_.AssignWorkersToJob(job->job_id, candidate_workers);

        std::vector<std::shared_ptr<const Task>> tasks;
        for (auto it = candidate_workers.begin(); it != candidate_workers.end(); it++) {
          std::shared_ptr<const Task> task;
          TF_RETURN_IF_ERROR(CreateTask(job, it->first, task));
          tasks.push_back(task);
        }
        l.mutex()->unlock();
        TF_RETURN_IF_ERROR(AssignTasks(tasks));
        l.mutex()->lock();
      }
    }

  if (job->garbage_collected) {
    return errors::FailedPrecondition(
        "The requested job has been garbage collected due to inactivity. "
        "Consider configuring the dispatcher with a higher "
        "`job_gc_timeout_ms`.");
  }
  if (request->optional_current_round_case() ==
      ClientHeartbeatRequest::kCurrentRound) {
    round_robin_rounds_[request->job_client_id()] =
        std::max(round_robin_rounds_[request->job_client_id()],
                 request->current_round());
  }
  if (!job->pending_tasks.empty()) {
    const auto &task = job->pending_tasks.front();
    Update update;
    ClientHeartbeatUpdate *client_heartbeat = update.mutable_client_heartbeat();
    bool apply_update = false;
    client_heartbeat->set_job_client_id(request->job_client_id());
    absl::optional<int64> blocked_round;
    if (request->optional_blocked_round_case() ==
        ClientHeartbeatRequest::kBlockedRound) {
      blocked_round = request->blocked_round();
    }
    VLOG(1) << "Handling pending task in job client heartbeat. job_client_id: "
            << request->job_client_id()
            << ". current_round: " << request->current_round()
            << ". blocked_round: " << blocked_round.value_or(-1)
            << ". target_round: " << task.target_round;
    if (request->current_round() >= task.target_round) {
      TaskRejected *rejected = client_heartbeat->mutable_task_rejected();
      // Exponentially try later and later rounds until consumers all agree.
      int64 round_offset = 2;
      for (int i = 0; i < task.failures; ++i) {
        round_offset *= 2;
      }
      rejected->set_new_target_round(
          round_robin_rounds_[request->job_client_id()] + round_offset);
      apply_update = true;
    }
    if (blocked_round.has_value() &&
        blocked_round.value() <= task.target_round &&
        !task.ready_consumers.contains(request->job_client_id())) {
      client_heartbeat->set_task_accepted(true);
      apply_update = true;
    }
    if (apply_update) {
      TF_RETURN_IF_ERROR(Apply(update));
    }
  }
  if (!job->pending_tasks.empty()) {
    response->set_block_round(job->pending_tasks.front().target_round);
  }
  
  // TODO(DanGraur): Could the deadlock stem from here?
  if (do_reassign_workers) {
    l.mutex()->unlock();
    ReassignFreeWorkersAndCreateTasks();
    l.mutex()->lock();
  }

  std::vector<std::shared_ptr<const Task>> tasks;
  TF_RETURN_IF_ERROR(state_.TasksForJob(job->job_id, tasks));
  VLOG(3) << "The number of tasks for this job (" << job->job_id << ") is: "
    << tasks.size();
  for (const auto& task : tasks) {
    TaskInfo* task_info = response->mutable_task_info()->Add();
    task_info->set_worker_address(task->worker_address);
    task_info->set_transfer_address(task->transfer_address);
    *task_info->mutable_worker_tags() = {task->worker_tags.begin(),
                                         task->worker_tags.end()};
    task_info->set_task_id(task->task_id);
    task_info->set_job_id(job->job_id);
    task_info->set_worker_uid(task->worker_uid);
    task_info->set_starting_round(task->starting_round);
  }
  response->set_job_finished(job->finished);
  response->set_deployment_mode(config_.deployment_mode());
  VLOG(3) << "Found " << response->task_info_size()
          << " tasks for job client id " << request->job_client_id();

  // Collect inflation factor metrics if any
  std::shared_ptr<const Dataset> dataset;
  TF_RETURN_IF_ERROR(state_.DatasetFromId(job->dataset_id, dataset));
  uint64 fingerprint = dataset->fingerprint;
  std::vector<std::string> pipeline_nodes;
  std::vector<float> inflation_factors;
  s = order_state_.GetLatestInfFactors(fingerprint, pipeline_nodes, inflation_factors);
  VLOG(1) << "Pipeline nodes obtained: " << pipeline_nodes.size() << " "
          << "Inflation factors obtained: " << inflation_factors.size();
  bool inf_factors_exist;
  InfFactorMetrics inf_factors;
  if (s.ok()) {
    VLOG(0) << "Found inflation metrics";
    //response->set_has_inf_factors(true);
    for (int i = 0; i < pipeline_nodes.size(); ++i) {
      VLOG(0) << "Node " << pipeline_nodes[i] << " inflation factor " << inflation_factors[i];

      // Avoid reordering, when de/inflation just appears by noise
      if (inflation_factors[i] < 1.05 && inflation_factors[i] > 0.95) {
        inflation_factors[i] = 1;
      }
      std::string s = pipeline_nodes[i];
      s.erase(std::remove(s.begin(), s.end(), '('), s.end());
      s.erase(std::remove(s.begin(), s.end(), ')'), s.end());
      s.erase(std::remove(s.begin(), s.end(), ':'), s.end());
      (*response->mutable_node_inf_factors())[s] = inflation_factors[i];
    }
  } else {
    VLOG(1) << "No inflation metrics available";
  }
  VLOG(1) << "Done";

  return Status::OK();
}

Status DataServiceDispatcherImpl::GetWorkers(const GetWorkersRequest* request,
                                             GetWorkersResponse* response) {
  TF_RETURN_IF_ERROR(CheckStarted());
  mutex_lock l(mu_);
  VLOG(3) << "Enter GetWorkers";
  std::vector<std::shared_ptr<const Worker>> workers = state_.ListWorkers();
  for (const auto& worker : workers) {
    WorkerInfo* info = response->add_workers();
    info->set_address(worker->address);
  }
  VLOG(3) << "Returning list of " << response->workers_size()
          << " workers from GetWorkers";
  return Status::OK();
}

Status DataServiceDispatcherImpl::PopulateTaskDef(
    std::shared_ptr<const Task> task, TaskDef* task_def) const
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  task_def->set_dataset_id(task->job->dataset_id);
  task_def->set_job_id(task->job->job_id);
  task_def->set_worker_address(task->worker_address);
  task_def->set_task_id(task->task_id);
  *task_def->mutable_processing_mode_def() = task->job->processing_mode;
  if (IsStaticShard(task->job->processing_mode)) {
    task_def->set_num_workers(config_.worker_addresses_size());
    TF_ASSIGN_OR_RETURN(int64_t worker_index,
                        state_.GetWorkerIndex(task->worker_address));
    task_def->set_worker_index(worker_index);
  }
  if (task->job->distributed_epoch_state.has_value()) {
    task_def->set_num_split_providers(
        task->job->distributed_epoch_state.value().indices.size());
  }
  if (task->job->num_consumers.has_value()) {
    task_def->set_num_consumers(task->job->num_consumers.value());
  }
  std::shared_ptr<const Dataset> dataset;
  TF_RETURN_IF_ERROR(state_.DatasetFromId(task->job->dataset_id, dataset));

  // EASL - Task was assigned COMPUTE, GET, PUT. We should not take the generic one.
  std::string dataset_key = task->dataset_key;
  //std::string dataset_key =
      //DatasetKey(dataset->dataset_id, dataset->fingerprint);
  if (config_.work_dir().empty()) {
    std::shared_ptr<const DatasetDef> dataset_def;
    TF_RETURN_IF_ERROR(dataset_store_->Get(dataset_key, dataset_def));
    *task_def->mutable_dataset_def() = *dataset_def;
  } else {
    std::string path =
        io::JoinPath(DatasetsDir(config_.work_dir()), dataset_key);
    task_def->set_path(path);
  }
  return Status::OK();
}

Status DataServiceDispatcherImpl::CheckStarted() TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  if (!started_) {
    return errors::Unavailable("Dispatcher has not started yet.");
  }
  return Status::OK();
}

Status DataServiceDispatcherImpl::RecordSplitProduced(
    int64_t job_id, int64_t repetition, int64_t split_provider_index,
    bool finished) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  Update update;
  ProduceSplitUpdate* produce_split = update.mutable_produce_split();
  produce_split->set_job_id(job_id);
  produce_split->set_repetition(repetition);
  produce_split->set_split_provider_index(split_provider_index);
  produce_split->set_finished(finished);
  return Apply(update);
}

Status DataServiceDispatcherImpl::ApplyWithoutJournaling(const Update& update)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  return state_.Apply(update);
}

Status DataServiceDispatcherImpl::Apply(const Update& update)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  if (journal_writer_.has_value()) {
    TF_RETURN_IF_ERROR(journal_writer_.value()->Write(update));
  }
  return state_.Apply(update);
}

void DataServiceDispatcherImpl::JobGcThread() {
  int64_t next_check_micros = 0;
  while (true) {
    {
      mutex_lock l(mu_);
      while (!cancelled_ && env_->NowMicros() < next_check_micros) {
        int64 remaining_micros = next_check_micros - env_->NowMicros();
        job_gc_thread_cv_.wait_for(l,
                                   std::chrono::microseconds(remaining_micros));
      }
      if (cancelled_) {
        return;
      }
      {
        Status s = ReleaseMissingClients();
        if (!s.ok()) {
          LOG(WARNING) << "Error releasing missing clients: " << s;
        }
      }
      {
        Status s = GcOldJobs();
        if (!s.ok()) {
          LOG(WARNING) << "Error garbage collecting old jobs: " << s;
        }
      }
        next_check_micros =
            env_->NowMicros() + (config_.job_gc_check_interval_ms() * 1000);
    }
    Status s = ReassignFreeWorkersAndCreateTasks();
    if (!s.ok()) {
      LOG(WARNING) << "Error reassigning free workers in gc thread: " << s;
    }
  }
}

Status DataServiceDispatcherImpl::ReleaseMissingClients()
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  int64_t now = env_->NowMicros();
  for (const auto& client_id : state_.ListActiveClientIds()) {
    if (absl::FromUnixMicros(now) >
        latest_client_heartbeats_time_[client_id] +
            absl::Milliseconds(config_.client_timeout_ms())) {
      LOG(INFO) << "Releasing timed-out client with id " << client_id;
      Update update;
      ReleaseJobClientUpdate* release_client =
          update.mutable_release_job_client();
      release_client->set_job_client_id(client_id);
      release_client->set_time_micros(now);
      TF_RETURN_IF_ERROR(Apply(update));
    }
  }
  return Status::OK();
}

Status DataServiceDispatcherImpl::GcOldJobs() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::vector<std::shared_ptr<const Job>> jobs = state_.ListJobs();
  int64_t now = env_->NowMicros();
  for (const auto& job : jobs) {
    if (job->finished || job->num_clients > 0 ||
        job->last_client_released_micros < 0 ||
        now < job->last_client_released_micros +
                  (config_.job_gc_timeout_ms() * 1000)) {
      continue;
    }
    Update update;
    update.mutable_garbage_collect_job()->set_job_id(job->job_id);
    TF_RETURN_IF_ERROR(state_.Apply(update));
    LOG(INFO) << "Garbage collected job " << job->DebugString();

  }
  return Status::OK();
}

void DataServiceDispatcherImpl::LogDumpsThread() {
  int64 next_check_micros = 0;
  while (true) {
    mutex_lock l(mu_);
    while (!cancelled_ && env_->NowMicros() < next_check_micros) {
      int64 remaining_micros = next_check_micros - env_->NowMicros();
      log_dumps_thread_cv_.wait_for(l,
                                 std::chrono::microseconds(remaining_micros));
    }
    if (cancelled_) {
      return;
    }
    Status s = metadata_store_.AppendJobMetricsDumps(env_, config_.log_dir());
    if (!s.ok()) {
      LOG(WARNING) << "Error garbage collecting old jobs: " << s;
    }
    next_check_micros =
        env_->NowMicros() + (config_.log_dumps_interval_ms() * 1000);
  }
}

/* EASL - never used
Status DataServiceDispatcherImpl::GetDatasetDef(
    int64_t dataset_id, std::shared_ptr<const DatasetDef>& dataset_def)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::shared_ptr<const Dataset> dataset;
  TF_RETURN_IF_ERROR(state_.DatasetFromId(dataset_id, dataset));
  return GetDatasetDef(*dataset, dataset_def);
}*/

Status DataServiceDispatcherImpl::GetDatasetDef(
    const Dataset& dataset,
    const std::string& job_type,
    std::shared_ptr<const DatasetDef>& dataset_def)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::string key = service::easl::cache_utils::DatasetKey(
      dataset.dataset_id, dataset.fingerprint, job_type);

  //return errors::PermissionDenied("Should not enter here for now...");
  return dataset_store_->Get(key, dataset_def);
}

}  // namespace data
}  // namespace tensorflow
