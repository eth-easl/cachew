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
#include "tensorflow/core/data/service/dispatcher_state.h"

#include <memory>
#include <string>
#include <vector>
#include <algorithm>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/data/service/data_service.h"
#include "tensorflow/core/data/service/journal.h"
#include "tensorflow/core/data/service/journal.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace data {

DispatcherState::DispatcherState() {}

Status DispatcherState::Apply(const Update& update) {
  switch (update.update_type_case()) {
    case Update::kRegisterDataset:
      RegisterDataset(update.register_dataset());
      break;
    case Update::kRegisterWorker:
      RegisterWorker(update.register_worker());
      break;
    case Update::kCreateJob:
      CreateJob(update.create_job());
      break;
    case Update::kProduceSplit:
      ProduceSplit(update.produce_split());
      break;
    case Update::kAcquireJobClient:
      AcquireJobClient(update.acquire_job_client());
      break;
    case Update::kReleaseJobClient:
      ReleaseJobClient(update.release_job_client());
      break;
    case Update::kGarbageCollectJob:
      GarbageCollectJob(update.garbage_collect_job());
      break;
    case Update::kRemoveTask:
      RemoveTask(update.remove_task());
      break;
    case Update::kCreatePendingTask:
      CreatePendingTask(update.create_pending_task());
      break;
    case Update::kClientHeartbeat:
      ClientHeartbeat(update.client_heartbeat());
      break;
    case Update::kCreateTask:
      CreateTask(update.create_task());
      break;
    case Update::kFinishTask:
      FinishTask(update.finish_task());
      break;
    case Update::kSetElementSpec:
      SetElementSpec(update.set_element_spec());
      break;
    case Update::kReassignFreeWorkers:
      ReassignFreeWorkers();
      break;
    case Update::kJobTargetWorkerCountUpdate:
      UpdateJobTargetWorkerCount(update.job_target_worker_count_update());
      break;
    case Update::UPDATE_TYPE_NOT_SET:
      return errors::Internal("Update type not set.");
  }

  return Status::OK();
}

void DispatcherState::RegisterDataset(
    const RegisterDatasetUpdate& register_dataset) {
  int64 id = register_dataset.dataset_id();
  int64 fingerprint = register_dataset.fingerprint();
  auto dataset = std::make_shared<Dataset>(id, fingerprint);
  DCHECK(!datasets_by_id_.contains(id));
  datasets_by_id_[id] = dataset;
  DCHECK(!datasets_by_fingerprint_.contains(fingerprint));
  datasets_by_fingerprint_[fingerprint] = dataset;
  next_available_dataset_id_ = std::max(next_available_dataset_id_, id + 1);
}

void DispatcherState::RegisterWorker(
    const RegisterWorkerUpdate& register_worker) {
  std::string address = register_worker.worker_address();
  DCHECK(!workers_.contains(address));
  DCHECK(!avail_workers_.contains(address));
  workers_[address] =
      std::make_shared<Worker>(address, register_worker.transfer_address());
  avail_workers_[address] =
      std::make_shared<Worker>(address, register_worker.transfer_address());
  tasks_by_worker_[address] =
      absl::flat_hash_map<int64, std::shared_ptr<Task>>();
  jobs_by_worker_[address] =
      absl::flat_hash_map<int64, std::shared_ptr<Job>>();
}

void DispatcherState::CreateJob(const CreateJobUpdate& create_job) {
  int64 job_id = create_job.job_id();
  absl::optional<NamedJobKey> named_job_key;
  if (create_job.has_named_job_key()) {
    named_job_key.emplace(create_job.named_job_key().name(),
                          create_job.named_job_key().index());
  }
  absl::optional<int64> num_consumers;
  if (create_job.optional_num_consumers_case() ==
      CreateJobUpdate::kNumConsumers) {
    num_consumers = create_job.num_consumers();
  }
  auto job = std::make_shared<Job>(job_id, create_job.dataset_id(),
                                   ProcessingMode(create_job.processing_mode()),
                                   create_job.num_split_providers(),
                                   named_job_key, num_consumers,
                                   create_job.job_type(), create_job.target_worker_count());
  DCHECK(!jobs_.contains(job_id));
  jobs_[job_id] = job;
  tasks_by_job_[job_id] = TasksById();
  ending_tasks_by_job_[job_id] = TasksById();

  if (named_job_key.has_value()) {
    DCHECK(!named_jobs_.contains(named_job_key.value()) ||
           named_jobs_[named_job_key.value()]->garbage_collected);
    named_jobs_[named_job_key.value()] = job;
  }
  next_available_job_id_ = std::max(next_available_job_id_, job_id + 1);
}

void DispatcherState::ProduceSplit(const ProduceSplitUpdate& produce_split) {
  std::shared_ptr<Job> job = jobs_[produce_split.job_id()];
  DCHECK(job->distributed_epoch_state.has_value());
  DistributedEpochState& state = job->distributed_epoch_state.value();
  int64 provider_index = produce_split.split_provider_index();
  DCHECK_EQ(produce_split.repetition(), state.repetitions[provider_index]);
  if (produce_split.finished()) {
    state.repetitions[provider_index]++;
    state.indices[provider_index] = 0;
    return;
  }
  state.indices[provider_index]++;
}

void DispatcherState::AcquireJobClient(
    const AcquireJobClientUpdate& acquire_job_client) {
  int64 job_client_id = acquire_job_client.job_client_id();
  std::shared_ptr<Job>& job = jobs_for_client_ids_[job_client_id];
  DCHECK(!job);
  job = jobs_[acquire_job_client.job_id()];
  DCHECK(job);
  job->num_clients++;
  next_available_job_client_id_ =
      std::max(next_available_job_client_id_, job_client_id + 1);
}

void DispatcherState::ReleaseJobClient(
    const ReleaseJobClientUpdate& release_job_client) {
  int64 job_client_id = release_job_client.job_client_id();
  std::shared_ptr<Job>& job = jobs_for_client_ids_[job_client_id];
  DCHECK(job);
  job->num_clients--;
  DCHECK_GE(job->num_clients, 0);
  job->last_client_released_micros = release_job_client.time_micros();
  jobs_for_client_ids_.erase(job_client_id);
}

void DispatcherState::GarbageCollectJob(
    const GarbageCollectJobUpdate& garbage_collect_job) {
  int64 job_id = garbage_collect_job.job_id();
  for(auto it=tasks_by_job_[job_id].begin(); it!=tasks_by_job_[job_id].end(); it++){
    it->second->finished = true;
    tasks_by_worker_[it->second->worker_address].erase(it->second->task_id);
  }
  jobs_[job_id]->finished = true;
  jobs_[job_id]->garbage_collected = true;

  // EASL - Update available workers.
  for (auto it = workers_by_job_[job_id].begin(); it != workers_by_job_[job_id].end(); ++it) {
    VLOG(0) << "(GarbageCollectJob) Releasing worker at address " << it->first
            << " for job " << job_id;
    avail_workers_[it->first] = it->second;
    jobs_by_worker_[it->first].erase(job_id);
  }
  workers_by_job_[job_id].clear();
}

void DispatcherState::RemoveTask(const RemoveTaskUpdate& remove_task) {
  std::shared_ptr<Task>& task = tasks_[remove_task.task_id()];
  DCHECK(task);
  task->removed = true;
  tasks_by_job_[task->job->job_id].erase(task->task_id);
  tasks_by_worker_[task->worker_address].erase(task->task_id);
  avail_workers_[task->worker_address] = workers_[task->worker_address];
  ending_tasks_by_job_[task->job->job_id].erase(task->task_id);
  tasks_.erase(task->task_id);
  VLOG(1) << "Removed task " << remove_task.task_id() << " from worker "
          << task->worker_address;
}

void DispatcherState::CreatePendingTask(
    const CreatePendingTaskUpdate& create_pending_task) {
  int64 task_id = create_pending_task.task_id();
  auto& task = tasks_[task_id];
  DCHECK_EQ(task, nullptr);
  auto& job = jobs_[create_pending_task.job_id()];
  DCHECK_NE(job, nullptr);
  task =
      std::make_shared<Task>(task_id, job, create_pending_task.worker_address(),
                             create_pending_task.transfer_address(), create_pending_task.dataset_key());
  job->pending_tasks.emplace(task, create_pending_task.starting_round());
  tasks_by_worker_[create_pending_task.worker_address()][task->task_id] = task;
  next_available_task_id_ = std::max(next_available_task_id_, task_id + 1);
}

void DispatcherState::ClientHeartbeat(
    const ClientHeartbeatUpdate& client_heartbeat) {
  int64 job_client_id = client_heartbeat.job_client_id();
  auto& job = jobs_for_client_ids_[job_client_id];
  DCHECK(!job->pending_tasks.empty());
  auto& task = job->pending_tasks.front();
  if (client_heartbeat.has_task_rejected()) {
    task.failures++;
    task.ready_consumers.clear();
    task.target_round = client_heartbeat.task_rejected().new_target_round();
  }
  if (client_heartbeat.task_accepted()) {
    task.ready_consumers.insert(job_client_id);
    if (task.ready_consumers.size() == job->num_consumers.value()) {
      VLOG(1) << "Promoting task " << task.task->task_id
              << " from pending to active";
      task.task->starting_round = task.target_round;
      tasks_by_job_[job->job_id][task.task->task_id] = task.task;
      job->pending_tasks.pop();
    }
  }
}

void DispatcherState::CreateTask(const CreateTaskUpdate& create_task) {
  int64 task_id = create_task.task_id();
  auto& task = tasks_[task_id];
  DCHECK_EQ(task, nullptr);
  auto& job = jobs_[create_task.job_id()];
  DCHECK_NE(job, nullptr);
  task = std::make_shared<Task>(task_id, job, create_task.worker_address(),
                                create_task.transfer_address(), create_task.dataset_key());
  job->current_worker_count++;
  tasks_by_job_[create_task.job_id()][task->task_id] = task;
  tasks_by_worker_[create_task.worker_address()][task->task_id] = task;
  next_available_task_id_ = std::max(next_available_task_id_, task_id + 1);
}

void DispatcherState::FinishTask(const FinishTaskUpdate& finish_task) {
  VLOG(2) << "Marking task " << finish_task.task_id() << " as finished";
  int64 task_id = finish_task.task_id();
  auto& task = tasks_[task_id];
  DCHECK(task != nullptr);
  task->finished = true;
  tasks_by_worker_[task->worker_address].erase(task->task_id);
  jobs_[task->job->job_id]->current_worker_count--;
  // Do not remove ended tasks because it's used as a reference for already ended tasks.
  ending_tasks_by_job_[task->job->job_id].erase(task_id);
  tasks_by_job_[task->job->job_id].erase(task_id);

  std::shared_ptr<Worker> worker = workers_[task->worker_address];
  avail_workers_[worker->address] = worker;
  jobs_by_worker_[worker->address].erase(task->job->job_id);
  workers_by_job_[task->job->job_id].erase(task->worker_address);
  VLOG(0) << "(FinishTask) Releasing worker at address " << worker->address
          << " for job " << task->job->job_id;

  bool all_finished = true;
  for(auto it = tasks_by_job_[task->job->job_id].begin(); it != tasks_by_job_[task->job->job_id].end(); ++it){
    if (!it->second->finished) {
      all_finished = false;
    }
  }
  jobs_[task->job->job_id]->finished = all_finished;
  // When a job completes, mark its workers as available
  if (all_finished) {
    VLOG(0) << "(FinishTask) Job " << task->job->job_id << " finished: "
            << all_finished;
    workers_by_job_[task->job->job_id].clear();
    ending_tasks_by_job_[task->job->job_id].clear(); // Or erase?
  }
}

void DispatcherState::SetElementSpec(
    const SetElementSpecUpdate& set_element_spec) {
  int64 dataset_id = set_element_spec.dataset_id();
  std::string element_spec = set_element_spec.element_spec();
  DCHECK(!id_element_spec_info_.contains(dataset_id));
  id_element_spec_info_[dataset_id] = element_spec;
}

Status DispatcherState::GetElementSpec(int64 dataset_id,
                                       std::string& element_spec) const {
  auto it = id_element_spec_info_.find(dataset_id);
  if (it == id_element_spec_info_.end()) {
    return errors::NotFound("Element_spec with key ", dataset_id, " not found");
  }
  element_spec = it->second;
  return Status::OK();
}

int64 DispatcherState::NextAvailableDatasetId() const {
  return next_available_dataset_id_;
}

Status DispatcherState::DatasetFromId(
    int64 id, std::shared_ptr<const Dataset>& dataset) const {
  auto it = datasets_by_id_.find(id);
  if (it == datasets_by_id_.end()) {
    return errors::NotFound("Dataset id ", id, " not found");
  }
  dataset = it->second;
  return Status::OK();
}

Status DispatcherState::DatasetFromFingerprint(
    uint64 fingerprint, std::shared_ptr<const Dataset>& dataset) const {
  auto it = datasets_by_fingerprint_.find(fingerprint);
  if (it == datasets_by_fingerprint_.end()) {
    return errors::NotFound("Dataset fingerprint ", fingerprint, " not found");
  }
  dataset = it->second;
  return Status::OK();
}

Status DispatcherState::WorkerFromAddress(
    const std::string& address, std::shared_ptr<const Worker>& worker) const {
  auto it = workers_.find(address);
  if (it == workers_.end()) {
    return errors::NotFound("Worker with address ", address, " not found.");
  }
  worker = it->second;
  return Status::OK();
}

std::vector<std::shared_ptr<const DispatcherState::Worker>>
DispatcherState::ListWorkers() const {
  std::vector<std::shared_ptr<const Worker>> workers;
  workers.reserve(workers_.size());
  for (const auto& it : workers_) {
    workers.push_back(it.second);
  }
  return workers;
}

std::vector<std::shared_ptr<const DispatcherState::Worker>>
DispatcherState::ListAvailableWorkers() const {
  std::vector<std::shared_ptr<const Worker>> workers;
  workers.reserve(avail_workers_.size());
  for (const auto& it : avail_workers_) {
    workers.push_back(it.second);
  }
  return workers;
}

std::vector<std::shared_ptr<const DispatcherState::Worker>>
DispatcherState::ReserveWorkers(
    int64 job_id, int64 target_num_workers) {
  // DCHECK(num_workers <= avail_workers_.size()); 
  jobs_[job_id]->target_worker_count = target_num_workers;
  // If the number of required workers is below those available, we just assign
  // as many as there are available at this epoch's scheduling time.
  int64 num_workers = target_num_workers <= 0 
    || target_num_workers > avail_workers_.size() ? avail_workers_.size() 
    : target_num_workers;
  std::vector<std::shared_ptr<const Worker>> workers;
  workers.reserve(num_workers);
  VLOG(0) << "(ReserveWorkers) User got " << num_workers << " workers from " 
          << "target " << target_num_workers << " workers";
  for (auto it = avail_workers_.begin(); it != avail_workers_.end(); ) {
    num_workers--;
    workers.push_back(it->second);
    VLOG(0) << "(ReserveWorkers) Assigning worker at address " 
            << it->second->address << " to job " << job_id;
    workers_by_job_[job_id][it->second->address] = it->second;
    jobs_by_worker_[it->second->address][job_id] = jobs_[job_id];
    avail_workers_.erase(it++);
    if (num_workers == 0)
      break;
  }
  VLOG(0) << "(ReserveWorkers) Number of workers for job " << job_id << " is: "
          << workers_by_job_[job_id].size();
  return workers;
}


// Go through jobs linearly and reassign free workers to jobs that miss workers.
void DispatcherState::ReassignFreeWorkers() {
  auto job_iter = jobs_.begin();
  if(job_iter == jobs_.end()){
    // Went through all jobs, can return
    return;
  }
  VLOG(0) << "EASL (ReassignFreeWorkers) - avail_workers_.size() "
      << avail_workers_.size();
  for(auto it = avail_workers_.begin(); it != avail_workers_.end(); it++){
    // Get a job in need of workers (i.e. num_assigned_workers < job->target_worker_count)
    std::shared_ptr<Job> job = job_iter->second;
    int64 num_assigned_workers = workers_by_job_[job->job_id].size();
    while (job->finished || num_assigned_workers >= job->target_worker_count){
      job_iter++;
      if(job_iter == jobs_.end()){
        // Went through all jobs, can return
        return;
      }
      job = job_iter->second;
      num_assigned_workers = workers_by_job_[job->job_id].size();
    }
    VLOG(0) << "EASL - (ReassignFreeWorkers) Reassigned worker "
            << it->second->address << " to job " << job->job_id;

    // Assign one worker to the job
    workers_by_job_[job->job_id][it->second->address] = it->second;
    jobs_by_worker_[it->second->address][job->job_id] = jobs_[job->job_id];
    avail_workers_.erase(it);
  }
}

void DispatcherState::UpdateJobTargetWorkerCount(
    const JobTargetWorkerCountUpdate job_target_worker_count_update) {
  const int64 job_id = job_target_worker_count_update.job_id();
  DCHECK(jobs_.contains(job_id));
  std::shared_ptr<Job> job = jobs_[job_id];

  VLOG(3) << "Got request for worker count change:\n"
               << " > job_target: " << job->target_worker_count << "\n"
               << " > current: " << job->current_worker_count << "\n"
               << " > request: " << job_target_worker_count_update.target_worker_count();

  if (job->target_worker_count < job_target_worker_count_update.target_worker_count()){
    VLOG(0) << "EASL (UpdateJobTargetWorkerCount) - Increased worker count from "
    << job->target_worker_count << " to target " << job_target_worker_count_update.target_worker_count();
  } else if (job->current_worker_count > job_target_worker_count_update.target_worker_count()){
    // Remove only created tasks..
    VLOG(0) << "EASL (UpdateJobTargetWorkerCount) - Decreased worker count from "
            << job->current_worker_count << " to target " << job_target_worker_count_update.target_worker_count();
    uint64 num_tasks_to_end  =
        job->current_worker_count - job_target_worker_count_update.target_worker_count();

    // Find tasks to end early
    DCHECK(tasks_by_job_.contains(job_id));
    TasksById current_tasks = tasks_by_job_[job_id];
    auto it = current_tasks.begin();
    for (int i=0; it!=current_tasks.end() && num_tasks_to_end>0; i++){
      auto task = it->second;
      // Only add to list if not already there.
      if (!ending_tasks_by_job_[job_id].contains(task->task_id)){
        ending_tasks_by_job_[job_id][task->task_id] = task;
        num_tasks_to_end--;
        VLOG(0) << "EASL - (UpdateJobTargetWorkerCount) - ending task " << task->task_id;
      }
      it++;
    }
    if(num_tasks_to_end > 0){
      VLOG(0) << "EASL (UpdateJobTargetWorkerCount) - not able to end enough tasks.";
    }
  }
  job->target_worker_count = job_target_worker_count_update.target_worker_count();

}

std::vector<std::shared_ptr<const DispatcherState::Job>>
DispatcherState::ListJobs() {
  std::vector<std::shared_ptr<const DispatcherState::Job>> jobs;
  jobs.reserve(jobs_.size());
  for (const auto& it : jobs_) {
    jobs.push_back(it.second);
  }
  return jobs;
}

std::vector<std::shared_ptr<const DispatcherState::Job>>
DispatcherState::ListJobsForWorker(const absl::string_view worker_address) {
  std::vector<std::shared_ptr<const DispatcherState::Job>> jobs;
  auto it = jobs_by_worker_.find(worker_address);
  if (it == jobs_by_worker_.end()) {
    VLOG(4) << "Worker at address " << worker_address
            << " is not yet assigned to any jobs.";
  }

  // FIXME(DanGraur): This will throw a nullptr exception if `it` is at end.
  const absl::flat_hash_map<int64, std::shared_ptr<Job>>& worker_jobs =
      it->second;
  jobs.reserve(worker_jobs.size());
  for (const auto& job : worker_jobs) {
    jobs.push_back(job.second);
  }
  return jobs;
}

Status DispatcherState::JobFromId(int64 id,
                                  std::shared_ptr<const Job>& job) const {
  auto it = jobs_.find(id);
  if (it == jobs_.end()) {
    return errors::NotFound("Job id ", id, " not found");
  }
  job = it->second;
  return Status::OK();
}

Status DispatcherState::NamedJobByKey(NamedJobKey named_job_key,
                                      std::shared_ptr<const Job>& job) const {
  auto it = named_jobs_.find(named_job_key);
  if (it == named_jobs_.end()) {
    return errors::NotFound("Named job key (", named_job_key.name, ", ",
                            named_job_key.index, ") not found");
  }
  job = it->second;
  return Status::OK();
}

int64 DispatcherState::NextAvailableJobId() const {
  return next_available_job_id_;
}

Status DispatcherState::JobForJobClientId(int64 job_client_id,
                                          std::shared_ptr<const Job>& job) {
  job = jobs_for_client_ids_[job_client_id];
  if (!job) {
    return errors::NotFound("Job client id not found: ", job_client_id);
  }
  return Status::OK();
}

int64 DispatcherState::NextAvailableJobClientId() const {
  return next_available_job_client_id_;
}

Status DispatcherState::TaskFromId(int64 id,
                                   std::shared_ptr<const Task>& task) const {
  auto it = tasks_.find(id);
  if (it == tasks_.end()) {
    return errors::NotFound("Task ", id, " not found");
  }
  task = it->second;
  return Status::OK();
}

Status DispatcherState::TasksForJob(
    int64 job_id, std::vector<std::shared_ptr<const Task>>& tasks) const {
  auto it = tasks_by_job_.find(job_id);
  if (it == tasks_by_job_.end()) {
    return errors::NotFound("Job ", job_id, " not found");
  }
  tasks.clear();
  tasks.reserve(it->second.size());
  for(auto task_it=it->second.begin(); task_it!=it->second.end(); task_it++) {
    tasks.push_back(task_it->second);
  }
  return Status::OK();
}

Status DispatcherState::TasksForWorker(
    absl::string_view worker_address,
    std::vector<std::shared_ptr<const Task>>& tasks) const {
  tasks.clear();
  auto it = tasks_by_worker_.find(worker_address);
  if (it == tasks_by_worker_.end()) {
    return errors::NotFound("Worker ", worker_address, " not found");
  }
  const absl::flat_hash_map<int64, std::shared_ptr<Task>>& worker_tasks =
      it->second;
  tasks.reserve(worker_tasks.size());
  for (const auto& task : worker_tasks) {
    tasks.push_back(task.second);
  }
  return Status::OK();
}

Status DispatcherState::IsEarlyEndedTask(const int64 job_id, const int64 task_id, bool& is_early_ended_task){
  DCHECK(ending_tasks_by_job_.contains(job_id));
  is_early_ended_task = ending_tasks_by_job_[job_id].contains(task_id);
  return Status::OK();
}

void DispatcherState::AddFutureEndedJob(int64 job_id,
  int32 split_provider_index) {
  std::string key = std::to_string(job_id) + "_" + std::to_string(
    split_provider_index);
  future_terminated_split_providers_.insert(key);
}
bool DispatcherState::IsFutureEndedJob(int64 job_id,
  int32 split_provider_index) {
  std::string key = std::to_string(job_id) + "_" + std::to_string(
    split_provider_index);
  auto loc = future_terminated_split_providers_.find(key);
  return loc != future_terminated_split_providers_.end();
}

int64 DispatcherState::NextAvailableTaskId() const {
  return next_available_task_id_;
}

}  // namespace data
}  // namespace tensorflow
