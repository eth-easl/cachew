//
// Created by damien-aymon on 14.12.20.
//

#ifndef TENSORFLOW_CORE_DATA_SERVICE_EASL_DISPATCHER_CACHE_STATE_H_
#define TENSORFLOW_CORE_DATA_SERVICE_EASL_DISPATCHER_CACHE_STATE_H_

#include "tensorflow/core/lib/core/status.h"
#include "absl/container/flat_hash_map.h"

namespace tensorflow {
namespace data {
namespace easl {

// This class holds the state of the tf.data service cache.
// This implementation assumes that every worker holds its own cache.
// We leave it here as a reference.
class OldCacheState {
 public:
  OldCacheState();
  OldCacheState(const OldCacheState &) = delete;
  OldCacheState &operator=(const OldCacheState &) = delete;

  bool IsDatasetCached( const uint64 fingerprint,
                        const std::string& worker_address) const;

  void SetDatasetCached(const uint64 fingerprint,
                        const std::string& worker_address);

  // Sets the task_id responsible for caching the dataset with this
  // fingerprint at this worker.
  // Returns an error if the task_id is not found.
  Status GetCachingTaskId(const uint64 fingerprint,
                          const std::string& worker_address,
                          int64& task_id) const;

  void RegisterCachingTask(const uint64 fingerprint,
                           const std::string& worker_address,
                           const int64 task_id);

 private:
  absl::flat_hash_map<uint64, absl::flat_hash_map<std::string, bool>>
      is_cached_at_worker_;

  absl::flat_hash_map<uint64, absl::flat_hash_map<std::string, int64>>
      caching_task_id_for_worker_;

};

} // namespace easl

class CacheState {
 public:
  CacheState();
  CacheState(const CacheState &) = delete;
  CacheState &operator=(const CacheState &) = delete;

  bool IsDatasetCached( const uint64 fingerprint) const;

  void SetDatasetCached(const uint64 fingerprint);

  bool IsDatasetSourceCached( const uint64 fingerprint) const;

  void SetDatasetSourceCached(const uint64 fingerprint);

  void PrintContents();

  // Returns an error if the jo is not found.
  Status GetCachingJobId(const uint64 fingerprint,
                          int64& job_id) const;

  //Sets the job_id responsible for caching the dataset with this
  // fingerprint
  void RegisterCachingJob(const uint64 fingerprint,
                           const int64 job_id);

 private:
  // For materialized dataset caching
  // keyed by fingerprint
  absl::flat_hash_map<uint64, bool> is_cached_;
  // keyed by fingerprint -> job_id
  absl::flat_hash_map<uint64, int64> fingerprint_to_caching_job_;

  // For source data caching
  // keyed by fingerprint
  absl::flat_hash_map<uint64, bool> is_source_cached_;
  // keyed by fingerprint -> job_id
  absl::flat_hash_map<uint64, int64> fingerprint_to_source_caching_job_;

};

} // namespace data
} // namespace tensorflow

#endif //TENSORFLOW_CORE_DATA_SERVICE_EASL_DISPATCHER_CACHE_STATE_H_
