//
// Created by otmraz on 11.11.22.
//

#ifndef ML_INPUT_DATA_SERVICE_TENSORFLOW_CORE_DATA_SERVICE_EASL_ORDERING_UTILS_H_
#define ML_INPUT_DATA_SERVICE_TENSORFLOW_CORE_DATA_SERVICE_EASL_ORDERING_UTILS_H_


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
namespace ordering_utils {


Status OpOrderUpdate(
    const std::string& job_type,
    const int64 job_id,
    const experimental::DispatcherConfig& dispatcher_config,
    ::tensorflow::data::easl::MetadataStore& metadata_store,
    int64& worker_count
    );


} // namespace ordering_utils
} // namespace easl
} // namespace service
} // namespace data
} // namespace tensorflow

#endif //ML_INPUT_DATA_SERVICE_TENSORFLOW_CORE_DATA_SERVICE_EASL_ORDERING_UTILS_H_
