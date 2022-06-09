//
// Created by Muyu Li on 20.06.22.
//

#include "split_pipeline_state.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

#include <queue>

namespace tensorflow {
namespace data {
namespace service {
namespace easl {
namespace split_state {

// SplitIndex
mutex SplitIndexes::mu_(LINKER_INITIALIZED);
SplitIndexes::JobToIndexMap* SplitIndexes::split_index_ =
        new JobToIndexMap();

void SplitIndexes::Print() {
  VLOG(0) << "SplitIndexes::Print";
  tf_shared_lock l(mu_);
  for (const auto& it: (*split_index_)) {
    VLOG(0) << it.first << " " << it.second;
  }
  VLOG(0) << "SplitIndexes::Print End";
}

void SplitIndexes::AddJob(std::string job_name, int64 split_node_index) {
  VLOG(0) << "SplitIndexes::AddJob";
  mutex_lock l(mu_);
  (*split_index_)[job_name] = split_node_index;
}

int64 SplitIndexes::GetSplitIndexFromJob(std::string job_name) {
  tf_shared_lock l(mu_);
  JobToIndexMap::const_iterator it = split_index_->find(job_name);
  if (it == split_index_->end()) {
    return 0;
  }
  return it->second;
}

int64 SplitIndexes::GetSplitIndex() {
  VLOG(0) << "In GetSplitIndex";
  tf_shared_lock l(mu_);
  if (split_index_->empty()) {
    return 0;
  }
  VLOG(0) << "SplitIndexes::GetSplitIndex: job-" << split_index_->begin()->first;
  return split_index_->begin()->second;
}

// SplitOriginalGraph
mutex SplitOriginalGraph::mu_(LINKER_INITIALIZED);
SplitOriginalGraph::JobToGraphMap* SplitOriginalGraph::graphs_ =
    new JobToGraphMap();

NodeDef* getNodeDefFromName(GraphDef* graph, std::string name) {
  int idx = tensorflow::grappler::graph_utils::FindGraphNodeWithName(name, *graph);
  if (idx == -1) {
    VLOG(0) << "Node: " << name << " not found in graph";
    return NULL;
  }
  return graph->mutable_node(idx);
}

std::string getNodeRepr(const NodeDef* sink) {
  return "[" + sink->name() + ", " + sink->op() + "]";
}

void BFSFromSink(GraphDef* graph, const NodeDef* sink, absl::flat_hash_set<const NodeDef*>& visited) {
  VLOG(0) << "SplitOriginalGraph::BFSFromSink from 'sink' node: " << getNodeRepr(sink);

  std::queue<const NodeDef*> q;
  visited.insert(sink);
  q.push(sink);

  while (!q.empty()) {
    const NodeDef* current_node = q.front();
    q.pop();

    for (int i = 0; i < current_node->input_size(); i++) {
      const NodeDef* next_node = getNodeDefFromName(graph, current_node->input(i));
      VLOG(0) << "EDGE: " << getNodeRepr(current_node)
              << " -> " << getNodeRepr(next_node);

      if (!visited.contains(next_node)) {
        visited.insert(next_node);
        q.push(next_node);
      }
    }
  }
}

void BFSPrint(GraphDef* graph) {
  VLOG(0) << "SplitOriginalGraph::BFSPrint---";
  absl::flat_hash_set<const NodeDef*> visited;

  for (const auto& _node: graph->node()) {
    std::string node_name = _node.name();
    const NodeDef* node = getNodeDefFromName(graph, node_name);
    if (!visited.contains(node)) {
      BFSFromSink(graph, node, visited);
    }
  }
}

void SplitOriginalGraph::Print() {
  VLOG(0) << "SplitOriginalGraph::Print";
  tf_shared_lock l(mu_);
  for (auto& it: (*graphs_)) {
    VLOG(0) << "JobName: " << it.first;
    BFSPrint(&(it.second));
  }
  VLOG(0) << "SplitIndexes::Print End";
}

void SplitOriginalGraph::AddJob(std::string job_name, GraphDef graph) {
  mutex_lock l(mu_);
  (*graphs_)[job_name] = graph;
}

GraphDef SplitOriginalGraph::GetGraphFromJob(std::string job_name) {
  tf_shared_lock l(mu_);
  JobToGraphMap::const_iterator it = graphs_->find(job_name);
  if (it == graphs_->end()) {
    VLOG(0) << "GetGraphFromJob not found";
  }
  return it->second;
}

GraphDef SplitOriginalGraph::GetGraph() {
  tf_shared_lock l(mu_);
  VLOG(0) << "SplitOriginalGraph::GetGraph job-" << graphs_->begin()->first;
  return graphs_->begin()->second;
}

} // split_state
}
}
}
}