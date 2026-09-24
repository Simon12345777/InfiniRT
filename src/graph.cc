#include "graph.h"

#include <algorithm>
#include <utility>

namespace infini::rt::graph {
namespace {

void MaybeAppendRegion(std::size_t begin, std::size_t end,
                       double submission_cost,
                       const CapturePartitionOptions& options,
                       CapturePlan* plan) {
  const std::size_t node_count = end - begin;
  const double saving = submission_cost - options.graph_launch_cost_us -
                        options.boundary_cost_us;
  if (node_count < options.min_nodes ||
      saving < options.min_estimated_saving_us) {
    return;
  }

  plan->regions.push_back({begin, end, saving});
  plan->captured_node_count += node_count;
}

}  // namespace

bool IsCaptureEligible(CaptureCapability capability,
                       const CaptureEnvironment& environment) noexcept {
  switch (capability) {
    case CaptureCapability::kAlways:
      return true;
    case CaptureCapability::kShapeSpecialized:
      return environment.shapes_specialized;
    case CaptureCapability::kParameterPatchable:
      return environment.supports_parameter_update;
    case CaptureCapability::kUnsupported:
    case CaptureCapability::kHostDependent:
      return false;
  }

  return false;
}

CapturePlan PlanCaptureRegions(const std::vector<NodeDesc>& nodes,
                               const CaptureEnvironment& environment,
                               const CapturePartitionOptions& options) {
  CapturePlan plan;
  std::size_t region_begin = 0;
  double submission_cost = 0.0;
  bool in_region = false;

  for (std::size_t index = 0; index < nodes.size(); ++index) {
    if (IsCaptureEligible(nodes[index].capture, environment)) {
      if (!in_region) {
        region_begin = index;
        submission_cost = 0.0;
        in_region = true;
      }
      submission_cost += std::max(0.0, nodes[index].submission_cost_us);
      continue;
    }

    if (in_region) {
      MaybeAppendRegion(region_begin, index, submission_cost, options, &plan);
      in_region = false;
    }
  }

  if (in_region) {
    MaybeAppendRegion(region_begin, nodes.size(), submission_cost, options,
                      &plan);
  }

  plan.uncaptured_node_count = nodes.size() - plan.captured_node_count;
  return plan;
}

PartitionedGraphExecutor::~PartitionedGraphExecutor() { Reset(); }

bool PartitionedGraphExecutor::Prepare(
    std::vector<NodeDesc> nodes, std::vector<NodeFunction> node_functions,
    const CaptureEnvironment& environment,
    const CapturePartitionOptions& options, NativeGraphCallbacks callbacks) {
  if (!Reset()) {
    return false;
  }
  if (nodes.size() != node_functions.size()) {
    return false;
  }

  nodes_ = std::move(nodes);
  node_functions_ = std::move(node_functions);
  callbacks_ = std::move(callbacks);
  const CapturePlan candidate_plan =
      PlanCaptureRegions(nodes_, environment, options);
  bool requires_parameter_update = false;
  for (const auto& region : candidate_plan.regions) {
    for (std::size_t index = region.begin; index < region.end; ++index) {
      requires_parameter_update |=
          nodes_[index].capture == CaptureCapability::kParameterPatchable;
    }
  }
  if (!candidate_plan.regions.empty() &&
      !HasCompleteCallbacks(requires_parameter_update)) {
    Reset();
    return false;
  }

  capture_failure_count_ = 0;
  active_plan_.regions.reserve(candidate_plan.regions.size());
  native_regions_.reserve(candidate_plan.regions.size());

  for (const auto& region : candidate_plan.regions) {
    if (!callbacks_.begin_capture()) {
      ++capture_failure_count_;
      continue;
    }

    bool record_success = true;
    for (std::size_t index = region.begin; index < region.end; ++index) {
      if (!node_functions_[index]()) {
        record_success = false;
        break;
      }
    }

    void* graph = nullptr;
    const bool end_success = callbacks_.end_capture(&graph);
    if (!record_success || !end_success || graph == nullptr) {
      if (graph != nullptr) {
        callbacks_.destroy_graph(graph);
      }
      ++capture_failure_count_;
      continue;
    }

    void* graph_exec = nullptr;
    if (!callbacks_.instantiate(&graph_exec, graph) || graph_exec == nullptr) {
      callbacks_.destroy_graph(graph);
      ++capture_failure_count_;
      continue;
    }

    const bool region_requires_parameter_update = std::any_of(
        nodes_.begin() + static_cast<std::ptrdiff_t>(region.begin),
        nodes_.begin() + static_cast<std::ptrdiff_t>(region.end),
        [](const NodeDesc& node) {
          return node.capture == CaptureCapability::kParameterPatchable;
        });
    native_regions_.push_back(
        {region, graph, graph_exec, region_requires_parameter_update});
    active_plan_.regions.push_back(region);
    active_plan_.captured_node_count += region.size();
  }

  active_plan_.uncaptured_node_count =
      nodes_.size() - active_plan_.captured_node_count;
  prepared_ = true;
  return true;
}

bool PartitionedGraphExecutor::Execute() {
  if (!prepared_) {
    return false;
  }

  std::size_t region_index = 0;
  std::size_t node_index = 0;
  while (node_index < nodes_.size()) {
    if (region_index < native_regions_.size() &&
        native_regions_[region_index].region.begin == node_index) {
      const auto& native_region = native_regions_[region_index];
      if (native_region.requires_parameter_update &&
          !callbacks_.update_graph_exec(
              native_region.graph_exec, native_region.region.begin,
              native_region.region.end)) {
        return false;
      }
      if (!callbacks_.launch(native_region.graph_exec)) {
        return false;
      }
      node_index = native_region.region.end;
      ++region_index;
      continue;
    }

    if (!node_functions_[node_index]()) {
      return false;
    }
    ++node_index;
  }

  return true;
}

bool PartitionedGraphExecutor::Reset() noexcept {
  bool success = true;
  for (auto region = native_regions_.rbegin();
       region != native_regions_.rend(); ++region) {
    if (region->graph_exec != nullptr && callbacks_.destroy_graph_exec &&
        !callbacks_.destroy_graph_exec(region->graph_exec)) {
      success = false;
    }
    if (region->graph != nullptr && callbacks_.destroy_graph &&
        !callbacks_.destroy_graph(region->graph)) {
      success = false;
    }
  }

  native_regions_.clear();
  nodes_.clear();
  node_functions_.clear();
  callbacks_ = {};
  active_plan_ = {};
  capture_failure_count_ = 0;
  prepared_ = false;
  return success;
}

bool PartitionedGraphExecutor::HasCompleteCallbacks(
    bool requires_parameter_update) const noexcept {
  return callbacks_.begin_capture && callbacks_.end_capture &&
         callbacks_.instantiate &&
         (!requires_parameter_update || callbacks_.update_graph_exec) &&
         callbacks_.launch && callbacks_.destroy_graph &&
         callbacks_.destroy_graph_exec;
}

}  // namespace infini::rt::graph
