#ifndef TENSORFLOW_CORE_DATA_SERVICE_EASL_CACHE_UTILS_H_
#define TENSORFLOW_CORE_DATA_SERVICE_EASL_CACHE_UTILS_H_

#include <string>
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/data/service/easl/dispatcher_cache_state.h"
#include "tensorflow/core/data/service/easl/metadata_store.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/dispatcher_state.h"
#include "tensorflow/core/protobuf/service_config.pb.h"


namespace tensorflow {
namespace data {
namespace service {
namespace easl{
namespace cache_utils {

std::string DatasetPutKey(const int64 id, const uint64 fingerprint);
std::string DatasetPutSourceKey(const int64 id, const uint64 fingerprint);

std::string DatasetGetKey(const int64 id, const uint64 fingerprint);
std::string DatasetGetSourceKey(const int64 id, const uint64 fingerprint);

std::string DatasetKey(const int64 id, const uint64 fingerprint, const std::string& job_type);

// TODO (damien-aymon) deprecated, left here for reference.
// Sets the dataset_key to either ".._put", ".._get" or standard depending on
// the cache state.
// If the dataset is cached, the ".._get" version is set.
// Otherwise, if this task is responsible for caching the dataset at this
// worker, the ".._put" version is set.
// Finally, if none of the above match, the standard (plain compute) version
// is set.
// WARNING, a lock must be protecting cache_state, which is not thread safe!!!
/*
Status DatasetKeyOld(const ::tensorflow::data::easl::CacheState& cache_state,
                  const int64 dataset_id,
                  const uint64 fingerprint,
                  const std::string& worker_address,
                  const int64 task_id,
                  std::string& dataset_key);
*/

Status DetermineJobType(
        const experimental::DispatcherConfig& dispatcher_config,
        ::tensorflow::data::CacheState& cache_state,
        const ::tensorflow::data::easl::MetadataStore& metadata_store,
        const uint64 fingerprint,
        const string& job_name,
        std::string& job_type);

Status DetermineJobTypeUpdated(
                      const experimental::DispatcherConfig& dispatcher_config,
                      ::tensorflow::data::CacheState& cache_state,
                      ::tensorflow::data::easl::MetadataStore& metadata_store,
                      const int64 job_id);

Status AddPutOperator(const DatasetDef& dataset,
                      const uint64 fingerprint,
                      const experimental::DispatcherConfig& dispatcher_config,
                      DatasetDef& updated_dataset);

Status AddGetOperator(const DatasetDef& dataset,
                      const uint64 fingerprint,
                      const experimental::DispatcherConfig& dispatcher_config,
                      DatasetDef& updated_dataset);

Status AddGetOperatorAtMarker(
    const DatasetDef& dataset,
    const uint64 fingerprint,
    const std::string& marker_type,
    const experimental::DispatcherConfig& dispatcher_config,
    DatasetDef& updated_dataset);

Status AddPutOperatorAtMarker(
    const DatasetDef& dataset,
    const uint64 fingerprint,
    const std::string& marker_type,
    const experimental::DispatcherConfig& dispatcher_config,
    DatasetDef& updated_dataset);




} // namespace cache_utils
} // namespace easl
} // namespace service
} // namespace data
} // namespace tensorflow

#endif //TENSORFLOW_CORE_DATA_SERVICE_EASL_CACHE_UTILS_H_
