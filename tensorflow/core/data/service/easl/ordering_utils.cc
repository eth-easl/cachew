//
// Created by otmraz on 11.11.22.
//

#include "tensorflow/core/data/service/easl/ordering_utils.h"

#include <queue>
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

// Big assumption (TBC), this happens after the completion of an epoch (i.e. each op will process the same no. of elems (except for filtering))
// TODO: Check what happens with batch!
Status DetermineInflationFactors(::tensorflow::data::easl::MetadataStore& metadata_store, std::vector<float> inflationFactors, int64 job_id) {
  std::shared_ptr<::tensorflow::data::easl::JobMetrics> job_metrics;
  TF_RETURN_IF_ERROR(GetJobMetrics(job_id, job_metrics))


  std::shared_ptr<::tensorflow::data::easl::InputPipelineMetrics> i_p_metrics;
  metadata_store.GetInputPipelineMetrics(job_id, i_p_metrics);

  std::vector<std::string> worker_ips;
  std::shared_ptr<::tensorflow::data::easl::NodeMetrics> final_node_metrics;
  TF_RETURN_IF_ERROR(metadata_store.GetLastNodeMetrics(job_id, final_node_metrics));
  for (auto e : final_node_metrics->metrics_) {
    worker_ips.push_back(e.first);
  }
  int num_workers = worker_ips.size();
  VLOG(0) << "In total " << num_workers << " workers worked on this job";

  int nodes_in_pipeline = 0;
  std::vector<std::string> node_names;
  for (auto e : i_p_metrics->metrics_) {
    nodes_in_pipeline++;
    node_names.push_back(e.first);
    inflationFactors.push_back(0);
  }
  VLOG(0) << "In total the pipeline has " << nodes_in_pipeline << " nodes";

  // Use the num elems produced by a specific worker's last node as a weighting means
  std::vector<int> elems_produced_final;
  int total_elems_produced = 0;
  for (int i = 0; i < num_workers; ++i) {
    tensorflow::data::easl::NodeMetrics final_node_worker_metrics;
    TF_RETURN_IF_ERROR(i_p_metrics->GetWorkerMetrics(worker_ips[i], final_node_metrics));
    elems_produced_final.push_back(final_node_worker_metrics->num_elements());
    total_elems_produced += elems_produced_final[i];
    VLOG(0) << "Worker " << worker_ips[i] << " produced " << elems_produced_final[i] << " elements";
  }
  VLOG(0) << "In total " << total_elems_produced << " elements were produced by the input pipeline.";

  // Examine the metric for each worker 1 by 1
  for (int i = 0; i < num_workers; ++i) {
    ::tensorflow::data::easl::NodeMetrics::MetricsCollection worker_metrics;
    Status s = i_p_metrics.GetWorkerMetrics(worker_ips[i], worker_metrics);
    for (int j = 0; j < nodes_in_pipeline; ++j) {
      // TODO: Use the elements produeced by the worker on the current node (otherwise filter nodes may be problematic)
      float inflation_f = worker_metrics->metrics_.find(node_names[j]).bytes_produced() / worker_metrics.metrics_.find(node_names[j]).bytes_consumed()
      inflationFactors[j] += elems_produced_final[i] * inflation_f;
    }
  }

  // Divide inflation factors by the no. of elems
  for (int i = 0; i < nodes_in_pipeline; ++i) {
    inflationFactors[i] /= total_elems_produced;
    VLOG(0) << "Node " << node_names[i] << " has inflation factor " << inflationFactors[i];
  }




  /*std::shared_ptr<NodeMetrics> final_node_metrics;
  TF_RETURN_IF_ERROR(metadata_store.GetLastNodeMetrics(job_id, final_node_metrics));
  size_t num_workers = (final_node_metrics->metrics_).size();
  int num_ops = 5; // Figure out from the node metrics
  //for (int i)
  int total_elems =
  for (std::pair<std::string, std::shared_ptr<NodeMetrics::Metrics>> e : final_node_metrics->metrics_) {
    VLOG(0) << "NodeMetrics first (string) is: " << e.first;
    std::shared_ptr<NodeMetrics::Metrics> worker_metrics = e.second;

    for (int j = 0 ; j < num_ops; ++j) {

    }
  }*/

  return Status::OK();
}

Status OpOrderUpdate(
    const std::string& job_type,
    const int64 job_id,
    const experimental::DispatcherConfig& dispatcher_config,
    ::tensorflow::data::easl::MetadataStore& metadata_store,
    int64& worker_count,
    const DatasetDef& dataset,
    DatasetDef& reordered_dataset) {
  using NodeMetrics = ::tensorflow::data::easl::NodeMetrics;
  using ModelMetrics = ::tensorflow::data::easl::ModelMetrics;

  // Ordering policy
  // 0 == Fixed pipeline (no reordering)
  // 1 == AutoOrder policy

  if(dispatcher_config.order_policy() == 0) {
    VLOG(0) << "Not using AutoOrder Policy.";
    metadata_store.UnsetJobIsOrdering(job_id);
    return Status::OK();
  } else if (dispatcher_config.order_policy() == 1)
  {
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
    }
    else{
      VLOG(0) << "Currently using " << rwc << " remote workers.";
      return Status::OK();

      // Initialize the optimizer
      reordered_dataset = dataset;
      GraphDef* graph_def = reordered_dataset.mutable_graph();
      tensorflow::grappler::easl::AutoOrder optimizer;

      std::vector<float> inflationFactors;
      Status s1 = DetermineInflationFactors(metadata_store, inflationFactors, job_id);
      
      tensorflow::grappler::MutableGraphView graph(graph_def);
      Status s = optimizer.ApplyOptimization(graph, *graph_def);
      if(s.ok()) {
        VLOG(0) << "AutoOrder policy succeeded!";
        return s;
      } else {
        VLOG(0) << "AutoOrder policy failed!";
        return s;
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


