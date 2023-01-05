//
// Created by Oto Mraz on 10.12.22.
//

#ifndef TENSORFLOW_CORE_DATA_SERVICE_EASL_DISPATCHER_ORDER_STATE_H_
#define TENSORFLOW_CORE_DATA_SERVICE_EASL_DISPATCHER_ORDER_STATE_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/node_def.pb.h"
//#include "tensorflow/core/framework/node_def_util.h"
#include "absl/container/flat_hash_map.h"
#include <vector>
#include <map>

namespace tensorflow {
namespace data {
namespace easl {

} // namespace easl

class OrderState {
  public:
    OrderState();
    OrderState(const OrderState &) = delete;
    OrderState &operator=(const OrderState &) = delete;

    bool IsPipelineReordered( const uint64 fingerprint) const;

    void SetPipelineReordered(const uint64 fingerprint);

    // Returns an error if the job is not found.
    Status GetOrderingJobId(const uint64 fingerprint,
                            int64& job_id) const;

    void UpdateLatestInfFactors(const uint64 fingerprint,
                                std::vector<std::string> pipeline_nodes,
                                std::vector<float> inflation_factors);

    Status GetLatestInfFactors(const uint64 fingerprint,
                               std::vector<std::string> &pipeline_nodes,
                               std::vector<float> &inflation_factors);

    void AddOrgPipeline(std::vector<NodeDef> org_nodes);

    void AddFinalPipeline(const uint64 fingerprint,
                          std::vector<NodeDef> final_nodes);

    //Sets the job_id responsible for ordering the pipeline with this fingerprint
    void RegisterOrderingJob(const uint64 fingerprint,
                             const int64 job_id);

  private:
    // keyed by fingerprint
    absl::flat_hash_map<uint64, bool> is_ordered_;
    // keyed by fingerprint
    //absl::flat_hash_map<uint64, absl::flat_hash_map<std::string, std::shared_ptr<std::vector<std::string>>> latest_pipeline_order;
    std::map<uint64, std::vector<std::string>> latest_pipeline_order;
    // keyed by fingerprint
    //absl::flat_hash_map<uint64, absl::flat_hash_map<std::string, std::shared_ptr<std::vector<float>>> latest_inflation_factors;
    std::map<uint64, std::vector<float>> latest_inflation_factors;

    // keyed by fingerprint
    std::map<uint64, std::vector<NodeDef>> final_graphs_;

    // keyed by fingerprint
    std::map<uint64, std::vector<NodeDef>> org_graphs_;

    // We don't have access to the key at this stage, so we just have to order
    // pipelines greedily (one-by-one) in the multi-tenancy case
    std::vector<NodeDef> org_graph_;

    // keyed by fingerprint -> job_id
    absl::flat_hash_map<uint64, int64> fingerprint_to_ordering_job_;

};

} // namespace data
} // namespace tensorflow

#endif //TENSORFLOW_CORE_DATA_SERVICE_EASL_DISPATCHER_ORDER_STATE_H_
