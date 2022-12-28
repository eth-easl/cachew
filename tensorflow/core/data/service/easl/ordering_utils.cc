//
// Created by otmraz on 11.11.22.
//

#include "tensorflow/core/data/service/easl/ordering_utils.h"

#include <queue>
#include <vector>
#include <numeric>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/data/service/easl/cache_model.h"
#include "tensorflow/core/grappler/optimizers/data/easl_optimizers/auto_order.h"

namespace tensorflow {
namespace data {
namespace service {
namespace easl {
namespace ordering_utils {

namespace {
int MAX_WORKERS_PER_JOB = 100;

double kMinBatchTimeRelativeImprovementDown = 0.03;
uint32 kInStabilityBeforeScaling = 20;
double kMinQueueSizeRelativeGrowth = 1.5; // +50%
double kMinBatchTimeRelativeGrowth = 1.5; // +50%
}

Status FindReorderableIntervals(std::vector<std::string> pipeline_nodes,
                                std::vector<float> inflationFactors,
                                std::vector<std::vector<std::string>> reorderable_intervals,
                                std::vector<std::vector<float>> inf_f_intervals
                                ) {
  // For now just return the whole pipeline
  reorderable_intervals.push_back(pipeline_nodes);
  inf_f_intervals.push_back(inflationFactors);
  return Status::OK();
}

Status GetIntervalOrders(std::vector<std::vector<std::string>> reorderable_intervals,
                         std::vector<std::vector<float>> inf_f_intervals,
                         std::vector<std::vector<std::string>> target_interval_orders) {
  VLOG(0) << "There are " << reorderable_intervals.size() << "Reorderable interval in this pipeline.";
  for (int i = 0; i < reorderable_intervals.size(); ++i) {
    // Get the wanted order of the reorderable intervals

    std::vector<std::string> cur_interval = reorderable_intervals[i];
    std::vector<int> idxs(cur_interval.size());
    std::iota(idxs.begin(),idxs.end(),0); //Initializing
    std::sort(idxs.begin(), idxs.end(), [&](int j,int k){return inf_f_intervals[i][j]<inf_f_intervals[i][k];});

    VLOG(0) << "New order: ";
    std::vector<std::string> new_interval;
    for (int j = 0; j < cur_interval.size(); ++j) {
      new_interval.push_back(cur_interval[idxs[j]]);
      VLOG(0) << new_interval[j];
    }
    target_interval_orders.push_back(new_interval);
  }
  return Status::OK();
}

// Big assumption (TBC), this happens after the completion of an epoch (i.e. each op will process the same no. of elems (except for filtering))
// TODO: Check what happens with batch!
Status DetermineInflationFactors(::tensorflow::data::easl::MetadataStore& metadata_store,
                                 std::vector<std::string> pipeline_nodes,
                                 std::vector<float> inflationFactors,
                                 int64 job_id) {
  VLOG(0) << "Calculating inflation factors";
  std::shared_ptr<::tensorflow::data::easl::JobMetrics> job_metrics;
  TF_RETURN_IF_ERROR(metadata_store.GetJobMetrics(job_id, job_metrics));
  VLOG(0) << "Got job metrics";

  std::shared_ptr<::tensorflow::data::easl::InputPipelineMetrics> i_p_metrics;
  metadata_store.GetInputPipelineMetrics(job_id, i_p_metrics);
  VLOG(0) << "Got input pipeline metrics";

  std::vector<std::string> worker_ips;
  std::shared_ptr<::tensorflow::data::easl::NodeMetrics> final_node_metrics;
  TF_RETURN_IF_ERROR(metadata_store.GetLastNodeMetrics(job_id, final_node_metrics));
  VLOG(0) << "Got LastNodeMetrics";
  for (auto e : final_node_metrics->metrics_) {
    worker_ips.push_back(e.first);
  }
  int num_workers = worker_ips.size();
  VLOG(0) << "In total " << num_workers << " workers worked on this job";

  int nodes_in_pipeline = 0;
  for (auto e : i_p_metrics->metrics_) {
    nodes_in_pipeline++;
    pipeline_nodes.push_back(e.first);
    //inflationFactors.push_back(0);
  }
  VLOG(0) << "In total the pipeline has " << nodes_in_pipeline << " nodes";

  // 1. Sort the pipeline nodes by id
  std::vector<std::string> pipeline_nodes_sorted(nodes_in_pipeline);
  for (auto n : pipeline_nodes) {
    int pos = n.substr(n.find(":"), s.length() - s.find(delimiter) - 1);
    pipeline_nodes_sorted[pos] = n;
  }

  // 2. Remove any nodes after 1st TFRecord node
  tf_rec_pos = 0
  for (int i = 0; i < nodes_in_pipeline; ++i) {
    if (pipeline_nodes_sorted[i].find("TFRecord") != std::string::npos) {
      tf_rec_pos = i;
      break;
    }
  }
  pipeline_nodes_sorted_filtered = pipeline_nodes_sorted[0:tf_rec_pos+1];
  VLOG(0) << "The main pipeline has " << pipeline_nodes_sorted_filtered.length() << " nodes";

  // 3. Remove any Prefetch, MemoryCache, MemoryCacheImpl, AssertCardinality, TensorSlice, ParallelInterleaveV4 nodes
  //    (we aren't interested in those)
  for (int i = pipeline_nodes_sorted_filtered.length() - 1; i >= 0; --i) {
    std::string cur_node = pipeline_nodes_sorted_filtered[i];
    if (
        pipeline_nodes_sorted_filtered[i].find("TFRecord") != std::string::npos ||
        pipeline_nodes_sorted_filtered[i].find("Prefetch") != std::string::npos ||
        pipeline_nodes_sorted_filtered[i].find("MemoryCache") != std::string::npos ||
        pipeline_nodes_sorted_filtered[i].find("MemoryCacheImpl") != std::string::npos ||
        pipeline_nodes_sorted_filtered[i].find("AssertCardinality") != std::string::npos ||
        pipeline_nodes_sorted_filtered[i].find("ParallelInterleaveV4") != std::string::npos ||
        pipeline_nodes_sorted_filtered[i].find("TensorSlice") != std::string::npos

    ) {
      pipeline_nodes_sorted_filtered.erase(pipeline_nodes_sorted_filtered.begin() + i);
    }
  }
  VLOG(0) << "The main pipeline has " << pipeline_nodes_sorted_filtered.length() << " nodes of interest";

  // Use the num elems produced by a specific worker's last node as a weighting means
  std::vector<int> elems_produced_final;
  int total_elems_produced = 0;
  for (int i = 0; i < num_workers; ++i) {
    tensorflow::data::easl::NodeMetrics::MetricsCollection final_node_worker_metrics;
    TF_RETURN_IF_ERROR(i_p_metrics->GetWorkerMetrics(worker_ips[i], final_node_worker_metrics));
    auto it = final_node_worker_metrics.find(pipeline_nodes_sorted_filtered[pipeline_nodes_sorted_filtered.size()-1]);
    if (it != final_node_worker_metrics.end()) {
      elems_produced_final.push_back(it->second->num_elements());
    } else {
      elems_produced_final.push_back(0);
    }
    //elems_produced_final.push_back(final_node_worker_metrics.find(pipeline_nodes_sorted_filtered[pipeline_nodes_sorted_filtered.size()-1]).num_elements());
    total_elems_produced += elems_produced_final[i];
    VLOG(0) << "Worker " << worker_ips[i] << " produced " << elems_produced_final[i] << " elements";
  }
  VLOG(0) << "In total " << total_elems_produced << " elements were produced by the input pipeline.";

  // Examine the metric for each worker 1 by 1
  for (int i = 0; i < num_workers; ++i) {
    ::tensorflow::data::easl::NodeMetrics::MetricsCollection worker_metrics;
    Status s = i_p_metrics->GetWorkerMetrics(worker_ips[i], worker_metrics);
    for (int j = 0; j < nodes_in_pipeline; ++j) {
      // TODO: Use the elements produeced by the worker on the current node (otherwise filter nodes may be problematic)
      VLOG(1) << "Node " << pipeline_nodes_sorted_filtered[j];
      auto it = worker_metrics.find(pipeline_nodes_sorted_filtered[j]);
      if (it != worker_metrics.end()) {
        int bc = it->second->bytes_consumed();
        int bp = it->second->bytes_produced();
        VLOG(1) << "consumed " << bc << " bytes, produced " << bp << " bytes";
        if (bc == 0) {
          float inflation_f = -1.0; // -1 can be a special placeholder if no bytes were consumed
        } else {
          float inflation_f = 1.0 * bp / bc;
          inflationFactors[j] += elems_produced_final[i] * inflation_f;
        }
      }
    }
  }
  VLOG(0) << "Calculated all inflation factors";

  // Divide inflation factors by the no. of elems
  for (int i = 0; i < nodes_in_pipeline; ++i) {
    inflationFactors[i] /= 1.0 * total_elems_produced;
    VLOG(0) << "Node " << pipeline_nodes_sorted_filtered[i] << " has inflation factor " << inflationFactors[i];
  }

  // 4. Filter out node with inflation factor 0 (clearly input nodes)
  for (int i = inflationFactors.length() - 1; i >= 0; --i) {
    if (inflationFactors[i] == 0) {
      inflationFactors.erase(inflationFactors.begin() + i);
      pipeline_nodes_sorted_filtered.erase(pipeline_nodes_sorted_filtered.begin() + i);
    }
  }

  for (int i = 0; i < pipeline_nodes_sorted_filtered.length(); ++i) {
    VLOG(0) << "Node " << pipeline_nodes_sorted_filtered[i] << " has inflation factor " << inflationFactors[i];
  }

  return Status::OK();
}

Status OpOrderUpdate(
    const std::string& job_type,
    const int64 job_id,
    const experimental::DispatcherConfig& dispatcher_config,
    ::tensorflow::data::easl::MetadataStore& metadata_store,
    int64& worker_count,
    const DatasetDef& dataset,
    std::vector<std::string> latest_pipeline,
    std::vector<float> inflation_factors,
    const uint64 fingerprint,
    DatasetDef& reordered_dataset) {
  using NodeMetrics = ::tensorflow::data::easl::NodeMetrics;
  using ModelMetrics = ::tensorflow::data::easl::ModelMetrics;

  // Ordering policy
  // 0 == Fixed pipeline (no reordering)
  // 1 == AutoOrder policy

  // Print out collected data
  VLOG(0) << "Summary: Pipeline has " << latest_pipeline.size() << " nodes";
  for (int i = 0; i < latest_pipeline.size(); ++i) {
    VLOG(0) << "Node: " << latest_pipeline[i] << " Inflation factor: " << inflation_factors[i];
  }

  if (dispatcher_config.order_policy() == 0) {
    VLOG(0) << "Not using AutoOrder Policy.";
    metadata_store.UnsetJobIsOrdering(job_id);
    return Status::OK();
  } else if (dispatcher_config.order_policy() == 1) {
    VLOG(0) << "Using AutoOrder Policy.";
    metadata_store.SetJobIsOrdering(job_id);

    
    bool is_ordering;
    TF_RETURN_IF_ERROR(metadata_store.IsJobOrdering(job_id, is_ordering));

    std::shared_ptr<ModelMetrics> model_metrics;
    TF_RETURN_IF_ERROR(metadata_store.GetModelMetrics(job_id, model_metrics));

    ModelMetrics::MetricsHistory metrics_history = model_metrics->metrics_history_;
    std::shared_ptr<ModelMetrics::Metrics> last_metrics = metrics_history[metrics_history.size() - 1];
    VLOG(0) << "EASL (AutoOrder policy) - Worker count for last metrics: "
                << metrics_history[metrics_history.size()-1]->worker_count(); // Guaranteed to succeed.

    int rwc = last_metrics->remote_worker_count();
    if (rwc == 0) {
      // Later we might want to try adding reorder even in full local mode (for higher throughput)
      VLOG(0) << "Already at 0 remote workers, no need to reorder.";
      return Status::OK();
    } else {
      VLOG(0) << "Currently using " << rwc << " remote workers.";
      return Status::OK();

      // Initialize the optimizer
      reordered_dataset = dataset;
      GraphDef* graph_def = reordered_dataset.mutable_graph();
      tensorflow::grappler::easl::AutoOrder optimizer;

      //std::vector<float> inflation_factors;
      //std::vector<std::string> pipeline_nodes;
      //Status s1 = DetermineInflationFactors(metadata_store, pipeline_nodes, inflation_factors, job_id);
      VLOG(0) << "Fetching inflation factors ...";
      //GetLatestInfFactors(fingerprint, pipeline_nodes, inflation_factors);

      // TODO: Add logic to mark some nodes with flags
      std::vector<std::vector<std::string>> reorderable_intervals;
      std::vector<std::vector<float>> inf_f_intervals;
      Status s = FindReorderableIntervals(latest_pipeline, inflation_factors, reorderable_intervals, inf_f_intervals);

      std::vector<std::vector<std::string>> target_interval_orders;
      Status s1 = GetIntervalOrders(reorderable_intervals, inf_f_intervals, target_interval_orders);




      tensorflow::grappler::MutableGraphView graph(graph_def);
      Status s2 = optimizer.ApplyOptimization(graph, *graph_def);
      if(s2.ok()) {
        VLOG(0) << "AutoOrder policy succeeded!";
        return s2;
      } else {
        VLOG(0) << "AutoOrder policy failed!";
        return s2;
      }
    }
  } else {
    VLOG(0) << "Invalid AutoOrder Policy!";
    return errors::InvalidArgument("Order policy ", dispatcher_config.order_policy(), " is invalid. Only order policy 0 and 1 and is valid.");
  }
  
}

} // namespace ordering_utils
} // namespace easl
} // namespace service
} // namespace data
} // namespace tensorflow


