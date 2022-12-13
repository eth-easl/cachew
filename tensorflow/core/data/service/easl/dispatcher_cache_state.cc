#include "tensorflow/core/data/service/easl/dispatcher_cache_state.h"

#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace data {
namespace easl {

OldCacheState::OldCacheState() {}

bool OldCacheState::IsDatasetCached(
    const uint64 fingerprint, const std::string& worker_address) const {
  auto worker_it = is_cached_at_worker_.find(fingerprint);
  if(worker_it == is_cached_at_worker_.end()){
    return false;
  }

  auto is_cached_it = worker_it->second.find(worker_address);
  if(is_cached_it == worker_it->second.end()){
    return false;
  }

  return is_cached_it->second;
}

void OldCacheState::SetDatasetCached(const uint64 fingerprint,
                                  const std::string& worker_address){
  is_cached_at_worker_[fingerprint][worker_address] = true;
}

Status OldCacheState::GetCachingTaskId(const uint64 fingerprint,
                                    const std::string &worker_address,
                                    int64 &task_id) const {
  auto worker_it = caching_task_id_for_worker_.find(fingerprint);
  if(worker_it == caching_task_id_for_worker_.end()){
    return errors::NotFound("No task responsible for caching this dataset");
  }

  auto task_it = worker_it->second.find(worker_address);
  if(task_it == worker_it->second.end()) {
    return errors::NotFound("No task responsible for caching this dataset");
  }

  task_id = task_it->second;
  return Status::OK();
}

void OldCacheState::RegisterCachingTask(const uint64 fingerprint,
                                     const std::string &worker_address,
                                     const int64 task_id) {
  // TODO (damien-aymon) Check if already present??
  caching_task_id_for_worker_[fingerprint][worker_address] = task_id;
}

} // namespace easl

CacheState::CacheState(){}

bool CacheState::IsDatasetCached(const uint64 fingerprint) const {
  auto is_cached_it = is_cached_.find(fingerprint);
  if(is_cached_it == is_cached_.end()){
    return false;
  }

  return is_cached_it->second;
}

void CacheState::SetDatasetCached(const uint64 fingerprint) {
  is_cached_[fingerprint] = true;
}

bool CacheState::IsDatasetSourceCached(const uint64 fingerprint) const {
  auto is_cached_it = is_source_cached_.find(fingerprint);
  if(is_cached_it == is_source_cached_.end()){
    return false;
  }

  return is_cached_it->second;
}

void CacheState::SetDatasetSourceCached(const uint64 fingerprint) {
  is_source_cached_[fingerprint] = true;
}

Status CacheState::GetCachingJobId(const uint64 fingerprint, int64 &job_id) const {
  auto it = fingerprint_to_caching_job_.find(fingerprint);
  if(it == fingerprint_to_caching_job_.end()){
    return errors::NotFound(
        "There is no job responsible for caching the dataset with fingerprint "
        + fingerprint);
  }
  job_id = it->second;
  return Status::OK();
}

void CacheState::RegisterCachingJob(const uint64 fingerprint, const int64 job_id) {
  fingerprint_to_caching_job_[fingerprint] = job_id;
}
} // namespace data
} // namespace tensorflow

