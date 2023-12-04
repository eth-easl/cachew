#include "tensorflow/core/data/service/easl/cache_utils.h"
#include <queue>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/optimizers/data/easl_optimizers/add_put_op.h"
#include "tensorflow/core/grappler/optimizers/data/easl_optimizers/add_get_op.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/data/service/easl/cache_model.h"
#include "tensorflow/core/grappler/optimizers/data/easl_optimizers/add_put_op_at_marker.h"
#include "tensorflow/core/grappler/optimizers/data/easl_optimizers/add_get_op_at_marker.h"

namespace tensorflow {
namespace data {
namespace service {
namespace easl {
namespace cache_utils {

Status DoBFS(NodeDef* sink_node, GraphDef& graph_def, string prefix) {
  absl::flat_hash_set<std::string> visited;
  std::queue<NodeDef*> bfs_queue;
  bfs_queue.push(sink_node);

  VLOG(0) << "(" << prefix << ") BFS @ current_node: "
          << "Root --> " << sink_node->op();

  while (!bfs_queue.empty()) {
    NodeDef* current_node = bfs_queue.front();
    bfs_queue.pop();
    visited.insert(current_node->name());

    // Iterate throught the neighbors
    for (int i = 0; i < current_node->input_size(); ++i) {
      if (!visited.contains(current_node->input(i))) {
        int idx = tensorflow::grappler::graph_utils::FindGraphNodeWithName(
            current_node->input(i), graph_def);
        NodeDef* neighbor_node = graph_def.mutable_node(idx);
        bfs_queue.push(neighbor_node);

        VLOG(0) << "(" << prefix << ") BFS @ current_node: "
                << SummarizeNodeDef(*current_node) << " --> "
                << SummarizeNodeDef(*neighbor_node);
      }
    }
  }

  return Status::OK();
}

std::string DatasetPutKey(const int64 id, const uint64 fingerprint) {
  return absl::StrCat("id_", id, "_fp_", fingerprint, "_put");
}

std::string DatasetPutSourceKey(const int64 id, const uint64 fingerprint) {
  return absl::StrCat("id_", id, "_fp_", fingerprint, "_put_source");
}

std::string DatasetGetKey(const int64 id, const uint64 fingerprint) {
  return absl::StrCat("id_", id, "_fp_", fingerprint, "_get");
}

std::string DatasetGetSourceKey(const int64 id, const uint64 fingerprint) {
  return absl::StrCat("id_", id, "_fp_", fingerprint, "_get_source");
}

std::string DatasetKey(
    const int64 id, const uint64 fingerprint, const std::string& job_type){
  if(job_type=="COMPUTE" || job_type == "PROFILE"){
    return absl::StrCat("id_", id, "_fp_", fingerprint);
  } else if (job_type=="GET"){
    return DatasetGetKey(id, fingerprint);
  } else if (job_type=="PUT"){
    return DatasetPutKey(id, fingerprint);
  } else if (job_type=="GET_SOURCE") {
    return DatasetGetSourceKey(id, fingerprint);
  } else if (job_type=="PUT_SOURCE") {
    return DatasetPutSourceKey(id, fingerprint);
  }
  return "";
}

Status DetermineJobType(const experimental::DispatcherConfig& dispatcher_config,
                          ::tensorflow::data::CacheState& cache_state,
                          const ::tensorflow::data::easl::MetadataStore& metadata_store,
                          const uint64 fingerprint,
                          const string& job_name,
                          std::string& job_type) {

    // First check if we should use a "fixed" cache policy:
    // 1 == EASL
    // 2 == compute
    // 3 == full cache(put, then get from 2nd epoch)
    // 4 == source cache(put, then get from 2nd epoch)
    // 30 == Force cache
    // 31 == Force read cache
    // 40 == Force source cache
    // 41 == Force read source cache

    // We force the pure compute mode; essentially this disables any caching
    job_type = "COMPUTE";
    return Status::OK();

    // Compute -------------------------------------------------------------------
    if(dispatcher_config.cache_policy() == 2) {
      job_type = "COMPUTE";
      return Status::OK();
      // Caching -------------------------------------------------------------------
    } else if(dispatcher_config.cache_policy() == 3){
      if(cache_state.IsDatasetCached(fingerprint)){
        job_type = "GET";
      } else {
        job_type = "PUT";
      }
      return Status::OK();
    } else if (dispatcher_config.cache_policy() == 30) {
      job_type = "PUT";
      return Status::OK();
    } else if (dispatcher_config.cache_policy() == 31) {
      job_type = "GET";
      return Status::OK();
      // Source Caching ------------------------------------------------------------
    } else if(dispatcher_config.cache_policy() == 4) {
      if (cache_state.IsDatasetSourceCached(fingerprint)) {
        job_type = "GET_SOURCE";
      } else {
        job_type = "PUT_SOURCE";
      }
      return Status::OK();
    } else if (dispatcher_config.cache_policy() == 40) {
      job_type = "PUT_SOURCE";
      return Status::OK();
    } else if (dispatcher_config.cache_policy() == 41) {
      job_type = "GET_SOURCE";
      return Status::OK();
    }
    // -------------------------------------------------------------------------
    // Cache policy = EASL (cache_policy==1)
    // -------------------------------------------------------------------------

    // If fingerprint is cached --> just get that cache. Prefer full cache over source
    // If fingerprint is not cached, but has metrics --> job exists, just copy over the last thing
    //  Only issue here: When job changes type during first epoch to put / put_source make sure to still allow the epoch extension
    // When job does not exist, set it to PROFILE

    if (cache_state.IsDatasetCached(fingerprint)) {
      job_type = "GET";
      return Status::OK();
    }

    if (cache_state.IsDatasetSourceCached(fingerprint)) {
      job_type = "GET_SOURCE";
      return Status::OK();
    }
    std::shared_ptr<data::easl::JobMetrics> job_metrics;
    Status s = metadata_store.GetJobMetricsByDatasetFingerprintAndName(fingerprint,
      job_name, job_metrics);

    // We've never seen this fingerprint --> PROFILE
    if (!s.ok()) {
      job_type = "PROFILE";
      return Status::OK();
    }

    // We have metrics, and it can only be COMPUTE, PUT, or PUT_SOURCE
    job_type = job_metrics->job_type_;
    return Status::OK();
}


Status DetermineJobTypeUpdated(
    const experimental::DispatcherConfig& dispatcher_config,
    ::tensorflow::data::CacheState& cache_state,
    ::tensorflow::data::easl::MetadataStore& metadata_store,
    const int64 job_id) {
  // Compute metrics
  using NodeMetrics = ::tensorflow::data::easl::NodeMetrics;
  std::shared_ptr<NodeMetrics> final_node_metrics;
  TF_RETURN_IF_ERROR(metadata_store.GetLastNodeMetrics(job_id,
    final_node_metrics));

  size_t num_workers = (final_node_metrics->metrics_).size();
  DCHECK(num_workers > 0);

  double compute_time_total_ms;
  double compute_working_time_total_ms;

  double compute_row_size = 0;
  double compute_time_per_row_ms = 0;
  double compute_working_time_per_row_ms = 0;

  double compute_total_processed_instances = 0.0;
  for(std::pair<std::string, std::shared_ptr<NodeMetrics::Metrics>> e :
      final_node_metrics->metrics_) {
    compute_total_processed_instances += e.second->num_elements();
  }

  for(std::pair<std::string, std::shared_ptr<NodeMetrics::Metrics>> e :
      final_node_metrics->metrics_) {
    std::shared_ptr<NodeMetrics::Metrics> worker_metrics = e.second;
    if (worker_metrics->num_elements() <= 0) {
      continue;
    }

    double weight = worker_metrics->num_elements() /
                    compute_total_processed_instances;
    compute_row_size += weight * worker_metrics->bytes_produced() /
                        worker_metrics->num_elements();
    compute_time_per_row_ms += worker_metrics->active_time_ms() * weight;
    compute_working_time_per_row_ms += worker_metrics->working_time_ms() * weight;
    VLOG(0) << "(DetermineJobTypeUpdated) Per worker row size last node:\n"
            << " > bytes_produced: " << worker_metrics->bytes_produced() << "\n"
            << " > num_elements: " << worker_metrics->num_elements() << "\n"
            << " > weight: " << weight << "\n"
            << " > compute_row_size component: " << (weight * worker_metrics->bytes_produced() /
                                                     worker_metrics->num_elements()) << "\n"
            << " > compute_row_size: " << compute_row_size << "\n"
            << " > compute_time_per_row_ms component: " << (worker_metrics->active_time_ms() * weight) << "\n"
            << " > compute_time_per_row_ms: " << compute_time_per_row_ms << "\n"
            << " > compute_working_time_per_row_ms component: " << (worker_metrics->working_time_ms() * weight) << "\n"
            << " > compute_working_time_per_row_ms: " << compute_working_time_per_row_ms;
  }

  // For the next values need 'normalization', since you might have batching
  // after the marker, hence you need to compare the total times for a
  // fair comparison
  compute_time_total_ms = compute_time_per_row_ms *
                          compute_total_processed_instances;
  compute_working_time_total_ms = compute_working_time_per_row_ms *
                                  compute_total_processed_instances;

  // For the next values need 'normalization', since you might have batching
  // after the marker, hence you need to compare the total times for a
  // fair comparison
  double cache_read_time_per_row_ms = data::cache_model::GetTimePerRow(
      compute_row_size);
  double cache_read_time_total_ms = cache_read_time_per_row_ms *
                                    compute_total_processed_instances;

  VLOG(0) << "(DetermineJobTypeUpdated) Compute values:\n"
          << " > total_elements: " << compute_total_processed_instances << "\n"
          << " > row_size: " << compute_row_size << "\n"
          << " > compute_time_per_row_ms: " << compute_time_per_row_ms << "\n"
          << " > compute_working_time_per_row_ms: " << compute_working_time_per_row_ms << "\n"
          << " > (M) compute_time_total_ms: " << compute_time_total_ms << "\n"
          << " > compute_working_time_total_ms: " << compute_working_time_total_ms << "\n"
          << " > cache_read_time_per_row_ms: " << cache_read_time_per_row_ms << "\n"
          << " > (M) cache_read_time_total_ms: " << cache_read_time_total_ms;

  // IO metrics
  std::shared_ptr<data::easl::InputPipelineMetrics> input_pipeline_metrics;
  metadata_store.GetInputPipelineMetrics(job_id, input_pipeline_metrics);

  if(!input_pipeline_metrics->GetMarkerNodeName().empty()) {
    VLOG(0) << "Found marker node name: "
                 << input_pipeline_metrics->GetMarkerNodeName();
    std::shared_ptr<data::easl::NodeMetrics> marker_node_metrics;
    metadata_store.GetMarkerNodeMetrics(job_id, marker_node_metrics);

    double io_row_size = 0.0;
    double avg_io_bytes_per_s = 0.0; // Will not be used.
    double avg_io_time_total_ms = 0.0; // == avg_gcs_source_time_ms.

    double marker_cache_total_processed_instances = 0.0;
    for(std::pair<std::string, std::shared_ptr<NodeMetrics::Metrics>> e :
        marker_node_metrics->metrics_) {
      marker_cache_total_processed_instances += e.second->num_elements();
    }

    for (auto& e : marker_node_metrics->metrics_) {
      std::shared_ptr<NodeMetrics::Metrics> node_metrics = e.second;
      if (node_metrics->num_elements() <= 0) {
        continue;
      }
      double weight = node_metrics->num_elements() /
                      marker_cache_total_processed_instances;

      avg_io_bytes_per_s += node_metrics->bytes_per_s() * weight;
      io_row_size += weight * node_metrics->bytes_produced() /
                     node_metrics->num_elements();
      avg_io_time_total_ms += node_metrics->active_time_ms() * weight;

      VLOG(0) << "(DetermineJobTypeUpdated) Per worker row size marker node:\n"
              << " > bytes_produced: " << node_metrics->bytes_produced() << "\n"
              << " > num_elements: " << node_metrics->num_elements() << "\n"
              << " > weight: " << weight << "\n"
              << " > io_row_size component: " << (weight * node_metrics->bytes_produced() /
                                             node_metrics->num_elements()) << "\n"
              << " > io_row_size: " << io_row_size << "\n"
              << " > avg_io_time_total_ms component: " << (node_metrics->active_time_ms() * weight) << "\n"
              << " > avg_io_time_total_ms: " << avg_io_time_total_ms;
    }

    // The next values need 'normalization', since you might have batching
    // after the marker, hence you need to compare the total times for a
    // fair comparison
    avg_io_time_total_ms = avg_io_time_total_ms *
                           marker_cache_total_processed_instances;
    double avg_io_bytes_per_active_time_ms = io_row_size *
                                             marker_cache_total_processed_instances;

    // Derive source caching values
    double post_maker_compute_overhead = compute_time_total_ms -
      avg_io_time_total_ms;
    double source_cache_io_time_per_row_ms = data::cache_model::GetTimePerRow(
        io_row_size);
    double source_cache_io_time_total_ms = source_cache_io_time_per_row_ms *
        marker_cache_total_processed_instances;

    // TODO(Dan): Here it was source_cache_io_time_per_row_ms instead of
    //            source_cache_io_time_total_ms; I think that was a mistake
    double source_cache_compute_time_total_ms = post_maker_compute_overhead +
      source_cache_io_time_total_ms;
//    double source_cache_compute_time_total_ms = std::max(
//        source_cache_io_time_total_ms, compute_working_time_total_ms);
//    double source_cache_compute_time_total_ms =
//        std::max(source_cache_io_time_per_row_ms, compute_working_time_total_ms);

    // FIXME(Dan): The decision does not seem to be right. Here we compare the
    //             last node's working time with the read time from cache
    //             but we never consider the addition of the compute overhead
    //             that comes after reading from cache

    VLOG(0) << "(DetermineJobTypeUpdated) Mark cache values:\n"
            << " > total_elements: " << marker_cache_total_processed_instances << "\n"
            << " > row_size: " << io_row_size << "\n"
            << " > avg_io_bytes_per_s: " << avg_io_bytes_per_s << "\n"
            << " > avg_io_time_total_ms: " << avg_io_time_total_ms << "\n"
            << " > (M) source_cache_compute_time_total_ms: " << source_cache_compute_time_total_ms << "\n"
            << " > avg_io_bytes_per_active_time_ms: " << avg_io_bytes_per_active_time_ms << "\n"
            << " > source_cache_io_time_per_row_ms: " << source_cache_io_time_per_row_ms  << "\n"
            << " > source_cache_io_time_total_ms: " << source_cache_io_time_total_ms << "\n"
            << " > compute_working_time_total_ms: " << compute_working_time_total_ms;

    // Take a decision
    std::vector<double> v = {source_cache_compute_time_total_ms,
                             cache_read_time_total_ms, compute_time_total_ms};
    int minElementIndex = std::min_element(v.begin(), v.end()) - v.begin();

    switch (minElementIndex) {
      case 0: // This is the source cache
        metadata_store.SetJobTypeByJobId(job_id, "PUT_SOURCE");
        VLOG(0) << "Cache decision: SOURCE CACHING";
        break;
      case 1: // This is the full cache
        metadata_store.SetJobTypeByJobId(job_id, "PUT");
        VLOG(0) << "Cache decision: FULL CACHING";
        break;
      case 2: // This is the compute
        metadata_store.SetJobTypeByJobId(job_id, "COMPUTE");
        VLOG(0) << "Cache decision: COMPUTE";
        break;
      default:
        VLOG(0) << "Cache decision: In DEFAULT... Will throw error...";
        return errors::Unimplemented("Caching decision defaulted to last option... "
                                     "See DetermineJobType!");
    }
  } else {
    VLOG(0) << "No marker node found, choosing between put or compute";
    if (compute_time_per_row_ms < cache_read_time_per_row_ms) {
      metadata_store.SetJobTypeByJobId(job_id, "COMPUTE");
    } else {
      metadata_store.SetJobTypeByJobId(job_id, "PUT");
    }
  }
  return Status::OK();
}

Status AddPutOperator(const DatasetDef& dataset,
                      const uint64 fingerprint,
                      const experimental::DispatcherConfig& dispatcher_config,
                      DatasetDef& updated_dataset) {
  // TODO remove this.
  //updated_dataset = dataset;
  //return Status::OK();
  VLOG(1) << "(AddPutOperator) At the start of the method";
  // Copy over the original dataset
  updated_dataset = dataset;

  // Initialize the optimizer
  tensorflow::grappler::easl::AddPutOp optimizer;
  // Transfer arguments from dispatcher config to optimizer config.
  tensorflow::RewriterConfig_CustomGraphOptimizer config;

  // TODO - set path where to store graph.
  (*(config.mutable_parameter_map()))["path"].set_placeholder(
      absl::StrCat(dispatcher_config.cache_path(), "/materialized/", fingerprint));
  (*(config.mutable_parameter_map()))["cache_format"].set_i(
      dispatcher_config.cache_format());
  (*(config.mutable_parameter_map()))["cache_compression"].set_i(
      dispatcher_config.cache_compression());
  (*(config.mutable_parameter_map()))["cache_ops_parallelism"].set_i(
      dispatcher_config.cache_ops_parallelism());

  optimizer.Init(&config);

  // Get the graph def and wrap it in a GrapplerItem
  GraphDef* graph_def = updated_dataset.mutable_graph();
  std::string output_node;

  // Find the output node; the one before '_Retval'
  for (const auto& node : graph_def->node()) {
    if (node.op() == "_Retval") {
      output_node = node.input(0);
    }
  }

  // Create a 'Sink' node and attatch it to the real output
  NodeDef* sink = graph_def->mutable_node()->Add();
  tensorflow::grappler::graph_utils::SetUniqueGraphNodeName("Sink", graph_def,
                                                            sink);
  sink->set_op("Identity");
  sink->add_input(output_node);
  (*sink->mutable_attr())["T"].set_type(DT_VARIANT);

  // Do BFS
  //DoBFS(sink, *graph_def, "AddPutOperator");

  // Create the MuttableGraphView
  tensorflow::grappler::MutableGraphView graph(graph_def);
  optimizer.ApplyOptimization(graph, sink, graph_def);

  // Do BFS
  //DoBFS(sink, *graph_def, "AfterAddPutOperator");

  // Disconnect the 'Sink' node
  // sink->mutable_input()->Clear();
  VLOG(1) << "(AddPutOperator) At the end of the method";

  return Status::OK();
}


Status AddGetOperator(const DatasetDef& dataset,
                      const uint64 fingerprint,
                      const experimental::DispatcherConfig& dispatcher_config,
                      DatasetDef& updated_dataset){
  // TODO remove this.
  //updated_dataset = dataset;
  //return Status::OK();

  VLOG(1) << "(AddGetOperator) At the start of the method";
  // Copy over the original dataset
  updated_dataset = dataset;

  // Initialize the optimizer  
  tensorflow::grappler::easl::AddGetOp optimizer;
  // Transfer arguments from dispatcher config to optimizer config.
  tensorflow::RewriterConfig_CustomGraphOptimizer config;

  // TODO - set path where to store graph.
  (*(config.mutable_parameter_map()))["path"].set_placeholder(
      absl::StrCat(dispatcher_config.cache_path(), "/materialized/", fingerprint));
  (*(config.mutable_parameter_map()))["cache_format"].set_i(
      dispatcher_config.cache_format());
  (*(config.mutable_parameter_map()))["cache_compression"].set_i(
      dispatcher_config.cache_compression());
  (*(config.mutable_parameter_map()))["cache_ops_parallelism"].set_i(
      dispatcher_config.cache_ops_parallelism());

  optimizer.Init(&config);

  // Get the graph def and wrap it in a GrapplerItem
  GraphDef* graph_def = updated_dataset.mutable_graph();
  std::string output_node;

  // Find the output node; the one before '_Retval'
  for (const auto& node : graph_def->node()) {
    if (node.op() == "_Retval") {
      output_node = node.input(0);
    }
  }

  // Create a 'Sink' node and attatch it to the real output
  NodeDef* sink = graph_def->mutable_node()->Add();
  tensorflow::grappler::graph_utils::SetUniqueGraphNodeName("Sink", graph_def,
                                                            sink);
  sink->set_op("Identity");
  sink->add_input(output_node);
  (*sink->mutable_attr())["T"].set_type(DT_VARIANT);

  // Do BFS
  //DoBFS(sink, *graph_def, "AddGetOperator");

  // Create the MuttableGraphView
  tensorflow::grappler::MutableGraphView graph(graph_def);
  optimizer.ApplyOptimization(graph, sink, graph_def);

  // Do BFS
  //DoBFS(sink, *graph_def, "AfterAddGetOperator");

  // Disconnect the 'Sink' node
  // sink->mutable_input()->Clear();
  VLOG(1) << "(AddGetOperator) At the end of the method";

  return Status::OK();
}


Status AddPutOperatorAtMarker(const DatasetDef& dataset,
                              const uint64 fingerprint,
                              const std::string& marker_type,
                              const experimental::DispatcherConfig& dispatcher_config,
                              DatasetDef& updated_dataset) {
  VLOG(0) << "(AddPutOperatorAtMarker) At the beginning of the method";

  // Copy over the original dataset
  updated_dataset = dataset;

  // Initialize the optimizer
  tensorflow::grappler::easl::AddPutOpAtMarker optimizer;
  // Transfer arguments from dispatcher config to optimizer config.
  tensorflow::RewriterConfig_CustomGraphOptimizer config;

  // Set correct path
  std::string cache_path;
  if(marker_type == "source_cache"){
    cache_path = absl::StrCat(dispatcher_config.cache_path(), "/source/", fingerprint);
  } else {
    absl::StrCat(dispatcher_config.cache_path(), "/materialized/", fingerprint);
  }

  // TODO - set path where to store graph.
  (*(config.mutable_parameter_map()))["path"].set_placeholder(cache_path);
  (*(config.mutable_parameter_map()))["cache_format"].set_i(
      dispatcher_config.cache_format());
  (*(config.mutable_parameter_map()))["cache_compression"].set_i(
      dispatcher_config.cache_compression());
  (*(config.mutable_parameter_map()))["cache_ops_parallelism"].set_i(
      dispatcher_config.cache_ops_parallelism());
  (*(config.mutable_parameter_map()))["marker_type"].set_placeholder(
      marker_type);

  optimizer.Init(&config);

  // Get the graph def and wrap it in a GrapplerItem
  GraphDef* graph_def = updated_dataset.mutable_graph();
  std::string output_node;

  // Find the output node; the one before '_Retval'
  for (const auto& node : graph_def->node()) {
    if (node.op() == "_Retval") {
      output_node = node.input(0);
    }
  }

  // Create a 'Sink' node and attatch it to the real output
  NodeDef* sink = graph_def->mutable_node()->Add();
  tensorflow::grappler::graph_utils::SetUniqueGraphNodeName("Sink", graph_def,
                                                            sink);
  sink->set_op("Identity");
  sink->add_input(output_node);
  (*sink->mutable_attr())["T"].set_type(DT_VARIANT);

  // Do BFS for debugging
  //DoBFS(sink, *graph_def, "AddPutAtMarkerOperator");

  // Create the MuttableGraphView
  tensorflow::grappler::MutableGraphView graph(graph_def);
  optimizer.ApplyOptimization(graph, sink, graph_def);

  // Do BFS
  //DoBFS(sink, *graph_def, "AfterAddPutAtMarkerOperator");

  // Disconnect the 'Sink' node
  // sink->mutable_input()->Clear();
  VLOG(0) << "(AddPutOperatorAtMarker) At the end of the method";

  return Status::OK();
}


Status AddGetOperatorAtMarker(
    const DatasetDef& dataset,
    const uint64 fingerprint,
    const std::string& marker_type,
    const experimental::DispatcherConfig& dispatcher_config,
    DatasetDef& updated_dataset){
  // TODO remove this.
  //updated_dataset = dataset;
  //return Status::OK();

  VLOG(0) << "(AddGetOperator) At the start of the method";
  // Copy over the original dataset
  updated_dataset = dataset;

  // Initialize the optimizer
  tensorflow::grappler::easl::AddGetOpAtMarker optimizer;
  // Transfer arguments from dispatcher config to optimizer config.
  tensorflow::RewriterConfig_CustomGraphOptimizer config;

  // Set correct path
  std::string cache_path;
  if(marker_type == "source_cache"){
    cache_path = absl::StrCat(dispatcher_config.cache_path(), "/source/", fingerprint);
  } else {
    absl::StrCat(dispatcher_config.cache_path(), "/materialized/", fingerprint);
  }

  // TODO - set path where to store graph.
  (*(config.mutable_parameter_map()))["path"].set_placeholder(cache_path);
  (*(config.mutable_parameter_map()))["cache_format"].set_i(
      dispatcher_config.cache_format());
  (*(config.mutable_parameter_map()))["cache_compression"].set_i(
      dispatcher_config.cache_compression());
  (*(config.mutable_parameter_map()))["cache_ops_parallelism"].set_i(
      dispatcher_config.cache_ops_parallelism());
  (*(config.mutable_parameter_map()))["marker_type"].set_placeholder(
      marker_type);

  optimizer.Init(&config);

  // Get the graph def and wrap it in a GrapplerItem
  GraphDef* graph_def = updated_dataset.mutable_graph();
  std::string output_node;

  // Find the output node; the one before '_Retval'
  for (const auto& node : graph_def->node()) {
    if (node.op() == "_Retval") {
      output_node = node.input(0);
    }
  }

  // Create a 'Sink' node and attatch it to the real output
  NodeDef* sink = graph_def->mutable_node()->Add();
  tensorflow::grappler::graph_utils::SetUniqueGraphNodeName("Sink", graph_def,
                                                            sink);
  sink->set_op("Identity");
  sink->add_input(output_node);
  (*sink->mutable_attr())["T"].set_type(DT_VARIANT);

  // Do BFS
  //DoBFS(sink, *graph_def, "AddGetOperator");

  // Create the MuttableGraphView
  tensorflow::grappler::MutableGraphView graph(graph_def);
  optimizer.ApplyOptimization(graph, sink, graph_def);

  // Do BFS
  //DoBFS(sink, *graph_def, "AfterAddGetOperator");

  // Disconnect the 'Sink' node
  // sink->mutable_input()->Clear();
  // VLOG(0) << "(AddGetOperatorAtMarker) At the end of the method";

  return Status::OK();
}



} // namespace cache_utils
} // namespace easl
} // namespace service
} // namespace data
} // namespace tensorflow
