#include "tensorflow/core/data/service/easl/dispatcher_order_state.h"

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/framework/node_def.pb.h"

#include <iostream>
#include <fstream>

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

    // Write the metrics to file
    std::string fingerprint_str = std::to_string(fingerprint);
    ofstream metrics_file("metrics" + fingerprint_str + ".csv");

    for (int i = 0; i < pipeline_nodes.size(); ++i) {
        metrics_file << pipeline_nodes[i] << "," << inflation_factors[i];
    }

    metrics_file.close();
}

Status OrderState::GetLatestInfFactors(const uint64 fingerprint, std::vector<std::string> pipeline_nodes, std::vector<float> inflation_factors) {
    auto it = is_ordered_.find(fingerprint);
    if (it == is_ordered_.end()) {
        return errors::NotFound(
            "Dataset with fingerprint " + fingerprint + std::string(" does not jet have data for inflation factors!"));
    }
    pipeline_nodes = latest_pipeline_order[fingerprint];
    inflation_factors = latest_inflation_factors[fingerprint];
    return Status::OK();
}

void OrderState::AddOrgPipeline(std::vector<NodeDef> org_nodes) {
    if (org_graph_.size() == 0) {
        org_graph_.insert(std::end(org_graph_), std::begin(org_nodes), std::end(org_nodes));
        VLOG(0) << "Added original pipeline to the store.";
    } else {
        VLOG(0) << "(Will retry) Busy reordering another pipeline.";
    }
}

// Here there should be some check that we are matching the correct 2 org & final pipelines
void OrderState::AddFinalPipeline(const uint64 fingerprint, std::vector<NodeDef> final_nodes) {
    auto it = final_graphs_.find(fingerprint);
    if (org_graph_.size() == 0) {
        VLOG(0) << "Couldn't find the original graph";
        return;
    } else {
        org_graphs_[fingerprint] = org_graph_;
        org_graph_.clear();
    }
    if (it == final_graphs_.end()) {
        final_graphs_[fingerprint] = final_nodes;
    }
}

void OrderState::RegisterOrderingJob(const uint64 fingerprint, const int64 job_id) {
    fingerprint_to_ordering_job_[fingerprint] = job_id;
}

} // namespace data
} // namespace tensorflow

