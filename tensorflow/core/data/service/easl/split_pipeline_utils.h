//
// Created by Muyu Li on 09.06.22.
//

#ifndef ML_INPUT_DATA_SERVICE_SPLIT_PIPELINE_UTILS_H
#define ML_INPUT_DATA_SERVICE_SPLIT_PIPELINE_UTILS_H

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
namespace easl{
namespace split_utils {

/* split_node_index: split after this node
  A -> B -> C -> D, index: 2
  A -> B | C -> D
  Same on client side
 */
Status DeleteAfterNode(const DatasetDef& dataset,
                         const experimental::DispatcherConfig& dispatcher_config,
                         int64 split_node_index,
                         DatasetDef& updated_dataset);


/*
 * Static variable used on the client side
 * Updated after getting the get_or_create_job_request
 *
 */

class SplitIndexes {
public:
    static void AddJob(std::string job_name);

    static void Print();

    static int64 GetSplitIndexFromJob(std::string job_name);

private:
    using JobToIndexMap = absl::flat_hash_map<std::string, int64>;
    static mutex mu_;
    static JobToIndexMap* split_index_ TF_GUARDED_BY(mu_);
};



} // split_utils
} // easl
} // service
} // data
} // tensorflow

#endif //ML_INPUT_DATA_SERVICE_SPLIT_PIPELINE_UTILS_H
