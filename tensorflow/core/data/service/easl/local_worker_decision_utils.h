//
// Created by Muyu Li on 29.05.22.
//

#ifndef ML_INPUT_DATA_SERVICE_LOCAL_WORKER_DECISION_UTILS_H
#define ML_INPUT_DATA_SERVICE_LOCAL_WORKER_DECISION_UTILS_H

#include <string>
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/data/service/easl/metadata_store.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/dispatcher_state.h"
#include "tensorflow/core/protobuf/service_config.pb.h"

namespace tensorflow {
namespace data {
namespace service {
namespace easl {
namespace local_worker_decision {

Status DecideIfLocal(
        const experimental::DispatcherConfig& dispatcher_config,
        const ::tensorflow::data::easl::MetadataStore& metadata_store,
        const std::string& dataset_key,
        bool& using_local_workers);

Status DecideTargetWorkersGridSearch(
        const experimental::DispatcherConfig& dispatcher_config,
        const ::tensorflow::data::easl::MetadataStore& metadata_store,
        const std::string& dataset_key,
        int64 num_worker_remote_avail,
        int64 num_worker_local_avail,
        int64& num_worker_remote_target,
        int64& num_worker_local_target);

Status DecideTargetWorkersAutoscaling(
        const experimental::DispatcherConfig& dispatcher_config,
        const ::tensorflow::data::easl::MetadataStore& metadata_store,
        const std::string& dataset_key,
        int64 num_worker_remote_avail,
        int64 num_worker_local_avail,
        int64& num_worker_remote_target,
        int64& num_worker_local_target);

Status DynamicWorkerCountUpdateWithLocal(
        const std::string& job_type,
        const int64 job_id,
        const experimental::DispatcherConfig& dispatcher_config,
        ::tensorflow::data::easl::MetadataStore& metadata_store,
        int64& remote_worker_count,
        int64& local_worker_count)

} // namespace local_worker_decision
} // namespace easl
} // namespace service
} // namespace data
} // namespace tensorflow

#endif //ML_INPUT_DATA_SERVICE_LOCAL_WORKER_DECISION_UTILS_H
