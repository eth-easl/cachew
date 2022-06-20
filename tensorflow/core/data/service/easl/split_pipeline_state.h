//
// Created by Muyu Li on 20.06.22.
//

#include <string>
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/data/service/easl/metadata_store.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/protobuf/service_config.pb.h"

#ifndef ML_INPUT_DATA_SERVICE_SPLIT_PIPELINE_STATE_H
#define ML_INPUT_DATA_SERVICE_SPLIT_PIPELINE_STATE_H

namespace tensorflow {
namespace data {
namespace service {
namespace easl {
namespace split_state {

class SplitIndexes {
public:
    static void AddJob(std::string job_name, int64 split_node_index);

    static void Print();

    static int64 GetSplitIndexFromJob(std::string job_name);
    static int64 GetSplitIndex();

private:
    using JobToIndexMap = absl::flat_hash_map<std::string, int64>;
    static mutex mu_;
    static JobToIndexMap* split_index_ TF_GUARDED_BY(mu_);
};

class SplitOriginalGraph {
public:
    static void AddJob(std::string job_name, GraphDef graph);
    static void Print();

    static GraphDef GetGraphFromJob(std::string job_name);
    static GraphDef GetGraph();

private:
    using JobToGraphMap = absl::flat_hash_map<std::string, GraphDef>;
    static mutex mu_;
    static JobToGraphMap* graphs_ TF_GUARDED_BY(mu_);
};

} // split_state
}
}
}
}

#endif //ML_INPUT_DATA_SERVICE_SPLIT_PIPELINE_STATE_H
