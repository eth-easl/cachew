//
// Created by Muyu Li on 16.11.21.
// Edited by the DSL group HS21 (Theodor Amariucai, Jiale Chen, Muyu Li) throughout November 2021 - February 2022
//

#ifndef ML_INPUT_DATA_SERVICE_LOCAL_WORKERS_UTILS_H
#define ML_INPUT_DATA_SERVICE_LOCAL_WORKERS_UTILS_H

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
namespace local_workers_utils {

Status ShouldUseLocalWorkers(
        const experimental::DispatcherConfig& dispatcher_config,
        const ::tensorflow::data::easl::MetadataStore& metadata_store,
        const int64 dataset_key,
        bool& should_use_local_workers);

Status DecideTargetWorkersGridSearch(
        int64 num_worker_remote_avail,
        int64 num_worker_local_avail,
        int64& num_worker_remote_target,
        int64& num_worker_local_target);

} // namespace local_workers_utils
} // namespace easl
} // namespace service
} // namespace data
} // namespace tensorflow

#endif //ML_INPUT_DATA_SERVICE_LOCAL_WORKERS_UTILS_H
