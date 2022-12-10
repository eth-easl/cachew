#include "tensorflow/core/data/service/easl/dispatcher_order_state.h"

#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace data {
namespace easl {

} // namespace easl

OrderState::OrderState(){}

bool OrderState::IsPipelineReordered(const uint64 fingerprint) const {
  auto is_ordered_it = is_ordered_.find(fingerprint);
  if(is_ordered_it == is_ordered_.end()){
    return false;
  }

  return is_ordered_it->second;
}

void OrderState::SetPipelineReordered(const uint64 fingerprint) {
  is_ordered_[fingerprint] = true;
}

Status OrderState::GetOrderingJobId(const uint64 fingerprint, int64 &job_id) const {
  auto it = fingerprint_to_ordering_job_.find(fingerprint);
  if (it == fingerprint_to_ordering_job_.end()) {
    return errors::NotFound(
        "There is no job responsible for ordering the dataset with fingerprint " + fingerprint);
  }
  job_id = it->second;
  return Status::OK();
}

void OrderState::UpdateLatestInfFactors(const uint64 fingerprint, std::vector<std::string> pipeline_nodes, std::vector<float> inflation_factors) {
  latest_pipeline_order[fingerprint] = pipeline_nodes;
  latest_inflation_factors[fingerprint] = inflation_factors;
}

Status OrderState::GetLatestInfFactors(const uint64 fingerprint, std::vector<std::string> pipeline_nodes, std::vector<float> inflation_factors) {
  auto it = is_ordered_.find(fingerprint);
  if (it == is_ordered_.end()) {
    return errors::NotFound(
        "Dataset with fingerprint " + fingerprint + std::string(" does not jet have data for inlaftion factors!"));
  }
  pipeline_nodes = latest_pipeline_order[fingerprint];
  inflation_factors = latest_inflation_factors[fingerprint];
  return Status::OK();
}

void OrderState::RegisterOrderingJob(const uint64 fingerprint, const int64 job_id) {
  fingerprint_to_ordering_job_[fingerprint] = job_id;
}
} // namespace data
} // namespace tensorflow

