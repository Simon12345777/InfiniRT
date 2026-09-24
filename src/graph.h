#ifndef INFINI_RT_GRAPH_H_
#define INFINI_RT_GRAPH_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

namespace infini::rt::graph {

enum class NodeKind {
  kOperator,
  kLayoutConvert,
  kMemcpy,
  kBarrier,
  kExternalCall,
};

enum class CaptureCapability {
  kAlways,
  kShapeSpecialized,
  kParameterPatchable,
  kUnsupported,
  kHostDependent,
};

struct NodeDesc {
  NodeKind kind{NodeKind::kOperator};

  CaptureCapability capture{CaptureCapability::kUnsupported};

  // Estimated host-side submission cost removed when this node is captured.
  double submission_cost_us{0.0};

  std::string debug_name;
};

struct CaptureEnvironment {
  bool shapes_specialized{false};

  bool supports_parameter_update{false};
};

struct CapturePartitionOptions {
  std::size_t min_nodes{2};

  double graph_launch_cost_us{0.0};

  double boundary_cost_us{0.0};

  double min_estimated_saving_us{0.0};
};

struct CaptureRegion {
  std::size_t begin{0};

  std::size_t end{0};

  double estimated_saving_us{0.0};

  std::size_t size() const noexcept { return end - begin; }
};

struct CapturePlan {
  std::vector<CaptureRegion> regions;

  std::size_t captured_node_count{0};

  std::size_t uncaptured_node_count{0};
};

bool IsCaptureEligible(CaptureCapability capability,
                       const CaptureEnvironment& environment) noexcept;

// Partitions submission-ordered nodes into maximal eligible regions. Regions
// that cannot amortize their launch/boundary cost remain on the ordinary path;
// one unsupported node does not invalidate the other regions.
CapturePlan PlanCaptureRegions(
    const std::vector<NodeDesc>& nodes, const CaptureEnvironment& environment,
    const CapturePartitionOptions& options = {});

// Backend-neutral callbacks used to bind the partitioned executor to the
// existing `runtime::*` graph API. Graph handles use the same opaque
// representation as the public runtime API.
struct NativeGraphCallbacks {
  std::function<bool()> begin_capture;

  std::function<bool(void**)> end_capture;

  std::function<bool(void**, void*)> instantiate;

  // Called immediately before launch for regions containing parameter-
  // patchable nodes. The range is `[begin, end)` in the original node list.
  std::function<bool(void*, std::size_t, std::size_t)> update_graph_exec;

  std::function<bool(void*)> launch;

  std::function<bool(void*)> destroy_graph;

  std::function<bool(void*)> destroy_graph_exec;
};

template <typename BackendRuntime>
NativeGraphCallbacks MakeNativeGraphCallbacks(
    typename BackendRuntime::Stream stream) {
  static_assert(std::is_pointer_v<typename BackendRuntime::Graph>,
                "Native graph handles must be pointer-like.");
  static_assert(std::is_pointer_v<typename BackendRuntime::GraphExec>,
                "Native graph executable handles must be pointer-like.");

  NativeGraphCallbacks callbacks;
  callbacks.begin_capture = [stream] {
    return BackendRuntime::StreamBeginCapture(
               stream, BackendRuntime::kStreamCaptureModeRelaxed) ==
           BackendRuntime::kSuccess;
  };
  callbacks.end_capture = [stream](void** graph) {
    typename BackendRuntime::Graph native_graph = nullptr;
    const auto status = BackendRuntime::StreamEndCapture(stream, &native_graph);
    *graph = reinterpret_cast<void*>(native_graph);
    return status == BackendRuntime::kSuccess;
  };
  callbacks.instantiate = [](void** graph_exec, void* graph) {
    typename BackendRuntime::GraphExec native_graph_exec = nullptr;
    const auto status = BackendRuntime::GraphInstantiate(
        &native_graph_exec,
        reinterpret_cast<typename BackendRuntime::Graph>(graph));
    *graph_exec = reinterpret_cast<void*>(native_graph_exec);
    return status == BackendRuntime::kSuccess;
  };
  callbacks.launch = [stream](void* graph_exec) {
    return BackendRuntime::GraphLaunch(
               reinterpret_cast<typename BackendRuntime::GraphExec>(graph_exec),
               stream) == BackendRuntime::kSuccess;
  };
  callbacks.destroy_graph = [](void* graph) {
    return BackendRuntime::GraphDestroy(
               reinterpret_cast<typename BackendRuntime::Graph>(graph)) ==
           BackendRuntime::kSuccess;
  };
  callbacks.destroy_graph_exec = [](void* graph_exec) {
    return BackendRuntime::GraphExecDestroy(
               reinterpret_cast<typename BackendRuntime::GraphExec>(
                   graph_exec)) == BackendRuntime::kSuccess;
  };
  return callbacks;
}

// Records profitable regions once and replays them while ordinary nodes and
// failed capture regions continue through their original functions. Node
// functions used in captured regions are invoked during `Prepare()` to submit
// their work to the active backend capture stream. Captured addresses and
// parameters must remain valid until the next `Reset()` or `Prepare()`.
class PartitionedGraphExecutor {
 public:
  using NodeFunction = std::function<bool()>;

  PartitionedGraphExecutor() = default;

  PartitionedGraphExecutor(const PartitionedGraphExecutor&) = delete;

  PartitionedGraphExecutor& operator=(const PartitionedGraphExecutor&) =
      delete;

  ~PartitionedGraphExecutor();

  bool Prepare(std::vector<NodeDesc> nodes,
               std::vector<NodeFunction> node_functions,
               const CaptureEnvironment& environment,
               const CapturePartitionOptions& options,
               NativeGraphCallbacks callbacks);

  bool Execute();

  bool Reset() noexcept;

  bool prepared() const noexcept { return prepared_; }

  const CapturePlan& plan() const noexcept { return active_plan_; }

  std::size_t capture_failure_count() const noexcept {
    return capture_failure_count_;
  }

 private:
  struct NativeRegion {
    CaptureRegion region;

    void* graph{nullptr};

    void* graph_exec{nullptr};

    bool requires_parameter_update{false};
  };

  bool HasCompleteCallbacks(bool requires_parameter_update) const noexcept;

  std::vector<NodeDesc> nodes_;

  std::vector<NodeFunction> node_functions_;

  NativeGraphCallbacks callbacks_;

  std::vector<NativeRegion> native_regions_;

  CapturePlan active_plan_;

  std::size_t capture_failure_count_{0};

  bool prepared_{false};
};

}  // namespace infini::rt::graph

#endif
