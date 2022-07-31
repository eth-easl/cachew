//
// Created by Muyu Li on 09.06.22.
//

#include "split_pipeline_utils.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/grappler/optimizers/data/easl_optimizers/delete_nodes_after.h"

namespace tensorflow {
namespace data {
namespace service {
namespace easl{
namespace split_utils {

NodeDef* getNodeDefFromName(GraphDef* graph, std::string name) {
  int idx = tensorflow::grappler::graph_utils::FindGraphNodeWithName(name, *graph);
  if (idx == -1) {
    VLOG(0) << "Node: " << name << " not found in graph";
    return NULL;
  }
  return graph->mutable_node(idx);
}

void BFSGraph(NodeDef* sink_node,
              GraphDef* output,
              std::string prefix) {
  VLOG(0) << "(SplitPipelineUtils::BFSGraph)::" << prefix << " start from sink node";
  std::queue<NodeDef*> q;
  absl::flat_hash_set<NodeDef*> visited;
  q.push(sink_node);
  visited.insert(sink_node);

  while (!q.empty()) {
    NodeDef* current_node = q.front();
    q.pop();

    for (int i = 0; i < current_node->input_size(); ++i) {
      int idx = tensorflow::grappler::graph_utils::FindGraphNodeWithName(current_node->input(i), *output);
      NodeDef* next_node = output->mutable_node(idx);
      VLOG(0) << "EDGE: [" << current_node->name()
              << ", " << current_node->op() << "] -> ["
              << next_node->name() << ", " << next_node->op() << "]";

      if (!visited.contains(next_node)) {
        visited.insert(next_node);
        q.push(next_node);
      }
    }
  }

  VLOG(0) << "(SplitPipelineUtils::BFSGraph) ends";
}

/* split_node_index: split after this node
A -> B -> C -> D, index: 2
A -> B | C -> D
Same on client side
*/
Status DeleteAfterNode(const DatasetDef& dataset,
                         const experimental::DispatcherConfig& dispatcher_config,
                         int64 split_node_index,
                         DatasetDef& updated_dataset) {
  VLOG(0) << "(SplitPipelineUtils::DeleteAfterNode): truncating after index " << split_node_index
    << " (from the end).";
  updated_dataset = dataset;

  tensorflow::grappler::easl::DeleteNodesAfter optimizer;
  tensorflow::RewriterConfig_CustomGraphOptimizer config;

  (*(config.mutable_parameter_map()))["split_node_index"].set_i(
          split_node_index);

  optimizer.Init(&config);

  GraphDef* graph_def = updated_dataset.mutable_graph();
  std::string output_node;

  std::string retval_nodename = "";
  for (const auto& node : graph_def->node()) {
    if (node.op() == "_Retval") {
      output_node = node.input(0);
      retval_nodename = node.name();
    }
  }

  NodeDef* sink = graph_def->mutable_node()->Add();
  tensorflow::grappler::graph_utils::SetUniqueGraphNodeName("Sink", graph_def,
                                                            sink);
  sink->set_op("Identity");
  sink->add_input(output_node);
  (*sink->mutable_attr())["T"].set_type(DT_VARIANT);

  NodeDef* retval_node = getNodeDefFromName(graph_def, retval_nodename);
  retval_node->mutable_input()->Clear();
  retval_node->add_input("Sink");

  BFSGraph(retval_node, graph_def, "DeleteAfterNode: Before");

  tensorflow::grappler::MutableGraphView graph(graph_def);
  optimizer.ApplyOptimization(graph, sink, graph_def);

  VLOG(0) << "(SplitPipelineUtils::DeleteAfterNode): function ended";

  BFSGraph(retval_node, graph_def, "DeleteAfterNode");
  return Status::OK();
}

std::string SplitDatasetKey(const int64 id, const uint64 fingerprint,
                            const int64 split_node_index) {
  return absl::StrCat("id_", id, "_fp_", fingerprint, "_sni_", split_node_index);
}

bool ifLocal(std::string address) {
  // hacky
  return address[0] == 'l';
}

Status LogNodeMetrics(string worker_address,
                      ::tensorflow::data::easl::NodeMetrics::MetricsCollection& metrics) {
  VLOG(0) << "LogNodeMetrics: -- start logging metrics for worker_address: "
    << worker_address;
  for(auto pair : metrics){
    VLOG(0) << "-- " << pair.first << ": "
       << "\"bytes_produced\" : " << std::to_string(pair.second->bytes_produced()) << "; "
       << "\"active_time_ms\" : " << std::to_string(pair.second->active_time_ms());
  }
}

Status LogSplitMetrics(const experimental::DispatcherConfig& dispatcher_config,
                       ::tensorflow::data::easl::MetadataStore& metadata_store,
                       const std::vector<std::string> workers,
                       const int64 job_id) {
  std::string local_worker_addr = "", remote_worker_addr = "";
  for (const auto& worker: workers) {
    VLOG(0) << worker;
    if (ifLocal(worker)) {
      local_worker_addr = worker;
    }
    else {
      remote_worker_addr = worker;
    }
  }

  if (local_worker_addr == "") {
    VLOG(0) << "LogSplitMetrics: local worker metrics not collected";
    return Status::OK();
  }
  if (remote_worker_addr == "") {
    VLOG(0) << "LogSplitMetrics: remote worker metrics not collected";
    return Status::OK();
  }

  VLOG(0) << "LogSplitMetrics: local_worker_addr: " << local_worker_addr
    << " remote_worker_addr: " << remote_worker_addr;

  using NodeMetrics = ::tensorflow::data::easl::NodeMetrics;
  using InputPipelineMetrics = ::tensorflow::data::easl::InputPipelineMetrics;

  std::shared_ptr<InputPipelineMetrics> input_pipeline_metrics;
  metadata_store.GetInputPipelineMetrics(job_id, input_pipeline_metrics);

  if (input_pipeline_metrics == NULL) {
    VLOG(0) << "Metrics not ready yet";
    return Status::OK();
  }

  double active_time_after_marker_node, active_time_marker_node, active_time_last_node;
  int64 bytes_produced_marker_node, bytes_produced_last_node;

  input_pipeline_metrics->GetWorkerMetricsSplitLocal(local_worker_addr,
                                                     active_time_after_marker_node,
                                                     bytes_produced_marker_node,
                                                     bytes_produced_last_node);

  input_pipeline_metrics->GetWorkerMetricsSplitRemote(remote_worker_addr,
                                                      active_time_marker_node,
                                                      active_time_last_node);

  VLOG(0) << "LogSplitMetrics: active_time_after_marker_node: " << active_time_after_marker_node
          << "; active_time_marker_node: " << active_time_marker_node
          << "; active_time_last_node: " << active_time_last_node
          << "; bytes_produced_last_node_per_ms: " << bytes_produced_last_node
          << "; bytes_produced_marker_node_per_ms: " << bytes_produced_marker_node;

  if (active_time_after_marker_node <= 1e-5 ||
      active_time_marker_node <= 1e-5 ||
      active_time_last_node <= 1e-5 ||
      bytes_produced_marker_node <= 5.0 ||
      bytes_produced_last_node <= 5.0
      ) {
    // When these metrics are not ready
    return Status::OK();
  }

  /*
   * bytes_per_s is actually measured in ms, so should be bytes_per_ms
   *
   * nw_speed = 10Gbps = 1.25 * 1000 * 1000 * 1000 Bps = 1.25 * 1e6  bpms
   */

  double nw_speed = 1.25 * 1e6;
  double tran_split = active_time_marker_node * bytes_produced_marker_node / nw_speed;
  double tran_remote = active_time_last_node * bytes_produced_last_node / nw_speed;

  VLOG(0) << "LogSplitMetrics: Local: \n"
      << "  LWSH: " << active_time_after_marker_node << "\n"
      << "  RWFH: " << active_time_marker_node << "\n"
      << "  TRAN: " << tran_split << "\n";

  VLOG(0) << "LogSplitMetrics: Remote: \n"
          << "  LN: " << active_time_last_node << "\n"
          << "  TRAN: " << tran_remote << "\n";

  if (active_time_after_marker_node + active_time_marker_node + tran_split
        < active_time_last_node + tran_remote) {
    VLOG(0) << "LogSplitMetrics: Split!!!";
  } else {
    VLOG(0) << "LogSplitMetrics: Remote!!!";
  }

  return Status::OK();
}

Status GetSplitNodeIndex(::tensorflow::data::easl::MetadataStore& metadata_store,
                         const int64 job_id,
                         int64& split_node_index) {
  int64 ret;
  Status s = metadata_store.GetJobSplitNodeIndex(job_id, ret);
  if (!s.ok()) {
    split_node_index = 0;
    VLOG(0) << "split_utils::GetSplitNodeIndex Error, Index is: " << split_node_index;
    return Status::OK();
  }
  split_node_index = ret;
  VLOG(0) << "split_utils::GetSplitNodeIndex Index is: " << split_node_index;
  return Status::OK();
}

Status GetSplitNodeIndex(::tensorflow::data::easl::MetadataStore& metadata_store,
                         const uint64 fingerprint,
                         const string job_name,
                         int64& split_node_index) {
  int64 ret;
  Status s = metadata_store.GetJobSplitNodeIndex(fingerprint, job_name, ret);
  if (!s.ok()) {
    split_node_index = 0;
    VLOG(0) << "split_utils::GetSplitNodeIndex Error, Index is: " << split_node_index;
    return Status::OK();
  }
  split_node_index = ret;
  VLOG(0) << "split_utils::GetSplitNodeIndex Index is: " << split_node_index;
  return Status::OK();
}

} // split_utils
} // easl
} // service
} // data
} // tensorflow
