#include <infini/rt.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "test_helper.h"

namespace {

namespace graph = infini::rt::graph;

struct FakeRuntime {
  using Error = int;
  using Stream = void*;
  using Graph = void*;
  using GraphExec = void*;

  static constexpr Error kSuccess = 0;
  static constexpr int kStreamCaptureModeRelaxed = 2;

  static Error StreamBeginCapture(Stream, int) { return kSuccess; }

  static Error StreamEndCapture(Stream, Graph* graph) {
    *graph = reinterpret_cast<Graph>(std::uintptr_t{1});
    return kSuccess;
  }

  static Error GraphInstantiate(GraphExec* graph_exec, Graph) {
    *graph_exec = reinterpret_cast<GraphExec>(std::uintptr_t{2});
    return kSuccess;
  }

  static Error GraphLaunch(GraphExec, Stream) { return kSuccess; }

  static Error GraphDestroy(Graph) { return kSuccess; }

  static Error GraphExecDestroy(GraphExec) { return kSuccess; }
};

void TestPartialCapture(infini::rt::test::TestContext* context) {
  using Capability = graph::CaptureCapability;
  const std::vector<graph::NodeDesc> nodes{
      {graph::NodeKind::kOperator, Capability::kAlways, 5.0, "a"},
      {graph::NodeKind::kOperator, Capability::kAlways, 5.0, "b"},
      {graph::NodeKind::kExternalCall, Capability::kHostDependent, 8.0,
       "host"},
      {graph::NodeKind::kOperator, Capability::kShapeSpecialized, 6.0, "c"},
      {graph::NodeKind::kOperator, Capability::kAlways, 6.0, "d"},
      {graph::NodeKind::kOperator, Capability::kAlways, 6.0, "e"},
  };

  graph::CapturePartitionOptions options;
  options.min_nodes = 2;
  options.graph_launch_cost_us = 2.0;
  options.boundary_cost_us = 1.0;
  options.min_estimated_saving_us = 5.0;

  const auto unspecialized = graph::PlanCaptureRegions(nodes, {}, options);
  context->ExpectEqual(unspecialized.regions.size(), std::size_t{2},
                       "Expected independent capture regions.");
  context->ExpectEqual(unspecialized.regions[0].begin, std::size_t{0},
                       "Unexpected first region start.");
  context->ExpectEqual(unspecialized.regions[0].end, std::size_t{2},
                       "Unexpected first region end.");
  context->ExpectEqual(unspecialized.regions[1].begin, std::size_t{4},
                       "Unexpected second region start.");
  context->ExpectEqual(unspecialized.regions[1].end, std::size_t{6},
                       "Unexpected second region end.");
  context->ExpectEqual(unspecialized.captured_node_count, std::size_t{4},
                       "Unexpected captured node count.");
  context->ExpectEqual(unspecialized.uncaptured_node_count, std::size_t{2},
                       "Unexpected uncaptured node count.");

  graph::CaptureEnvironment specialized;
  specialized.shapes_specialized = true;
  const auto specialized_plan =
      graph::PlanCaptureRegions(nodes, specialized, options);
  context->ExpectEqual(specialized_plan.regions.size(), std::size_t{2},
                       "Shape specialization should extend one region.");
  context->ExpectEqual(specialized_plan.regions[1].begin, std::size_t{3},
                       "Specialized region should include shape-bound node.");
  context->ExpectEqual(specialized_plan.regions[1].end, std::size_t{6},
                       "Unexpected specialized region end.");
}

void TestCaptureProfitability(infini::rt::test::TestContext* context) {
  using Capability = graph::CaptureCapability;
  const std::vector<graph::NodeDesc> nodes{
      {graph::NodeKind::kOperator, Capability::kAlways, 1.0, "a"},
      {graph::NodeKind::kOperator, Capability::kAlways, 1.0, "b"},
  };

  graph::CapturePartitionOptions options;
  options.min_nodes = 2;
  options.graph_launch_cost_us = 3.0;
  options.min_estimated_saving_us = 0.0;

  const auto plan = graph::PlanCaptureRegions(nodes, {}, options);
  context->Expect(plan.regions.empty(),
                  "An unprofitable region should remain uncaptured.");
  context->ExpectEqual(plan.uncaptured_node_count, nodes.size(),
                       "Rejected regions should remain ordinary nodes.");

  const graph::NodeDesc patchable{graph::NodeKind::kOperator,
                                  Capability::kParameterPatchable, 5.0,
                                  "patchable"};
  context->Expect(
      !graph::IsCaptureEligible(patchable.capture, {}),
      "Parameter-dependent nodes need backend update support.");
  graph::CaptureEnvironment update_environment;
  update_environment.supports_parameter_update = true;
  context->Expect(
      graph::IsCaptureEligible(patchable.capture, update_environment),
      "Parameter update support should make the node capture eligible.");
}

void TestPartitionedExecution(infini::rt::test::TestContext* context) {
  using Capability = graph::CaptureCapability;
  const std::vector<graph::NodeDesc> nodes{
      {graph::NodeKind::kOperator, Capability::kAlways, 5.0, "a"},
      {graph::NodeKind::kOperator, Capability::kAlways, 5.0, "b"},
      {graph::NodeKind::kExternalCall, Capability::kHostDependent, 1.0,
       "host"},
      {graph::NodeKind::kOperator, Capability::kParameterPatchable, 5.0, "c"},
      {graph::NodeKind::kOperator, Capability::kAlways, 5.0, "d"},
  };

  std::vector<std::size_t> node_calls(nodes.size(), 0);
  std::vector<graph::PartitionedGraphExecutor::NodeFunction> functions;
  for (std::size_t index = 0; index < nodes.size(); ++index) {
    functions.push_back([&, index] {
      ++node_calls[index];
      return true;
    });
  }

  std::size_t begin_calls = 0;
  std::size_t update_calls = 0;
  std::size_t launch_calls = 0;
  std::size_t destroyed_graphs = 0;
  std::size_t destroyed_execs = 0;
  graph::NativeGraphCallbacks callbacks;
  callbacks.begin_capture = [&] {
    ++begin_calls;
    return true;
  };
  callbacks.end_capture = [&](void** handle) {
    *handle = reinterpret_cast<void*>(begin_calls * 2 - 1);
    return true;
  };
  callbacks.instantiate = [](void** graph_exec, void* graph_handle) {
    *graph_exec = reinterpret_cast<void*>(
        reinterpret_cast<std::uintptr_t>(graph_handle) + 1);
    return true;
  };
  callbacks.update_graph_exec = [&](void*, std::size_t begin,
                                    std::size_t end) {
    ++update_calls;
    return begin == 3 && end == 5;
  };
  callbacks.launch = [&](void*) {
    ++launch_calls;
    return true;
  };
  callbacks.destroy_graph = [&](void*) {
    ++destroyed_graphs;
    return true;
  };
  callbacks.destroy_graph_exec = [&](void*) {
    ++destroyed_execs;
    return true;
  };

  graph::CapturePartitionOptions options;
  options.min_nodes = 2;
  graph::CaptureEnvironment environment;
  environment.supports_parameter_update = true;
  graph::PartitionedGraphExecutor executor;
  context->Expect(executor.Prepare(nodes, std::move(functions), environment,
                                   options, std::move(callbacks)),
                  "Partitioned executor preparation failed.");
  context->ExpectEqual(executor.plan().regions.size(), std::size_t{2},
                       "Expected two recorded regions.");
  context->ExpectEqual(executor.plan().captured_node_count, std::size_t{4},
                       "Unexpected recorded node count.");
  context->ExpectEqual(begin_calls, std::size_t{2},
                       "Each region should begin one capture.");

  context->Expect(executor.Execute(), "Partitioned execution failed.");
  context->ExpectEqual(launch_calls, std::size_t{2},
                       "Each recorded region should launch once.");
  context->ExpectEqual(update_calls, std::size_t{1},
                       "Only the parameter-dependent region should update.");
  context->ExpectEqual(node_calls[0], std::size_t{1},
                       "Captured nodes should only submit during recording.");
  context->ExpectEqual(node_calls[1], std::size_t{1},
                       "Captured nodes should only submit during recording.");
  context->ExpectEqual(node_calls[2], std::size_t{1},
                       "Ordinary nodes should execute during replay.");
  context->ExpectEqual(node_calls[3], std::size_t{1},
                       "Captured nodes should only submit during recording.");
  context->ExpectEqual(node_calls[4], std::size_t{1},
                       "Captured nodes should only submit during recording.");

  context->Expect(executor.Reset(), "Destroying native regions failed.");
  context->ExpectEqual(destroyed_graphs, std::size_t{2},
                       "Every native graph should be destroyed.");
  context->ExpectEqual(destroyed_execs, std::size_t{2},
                       "Every executable graph should be destroyed.");
}

void TestCaptureFailureFallback(infini::rt::test::TestContext* context) {
  using Capability = graph::CaptureCapability;
  const std::vector<graph::NodeDesc> nodes{
      {graph::NodeKind::kOperator, Capability::kAlways, 5.0, "a"},
      {graph::NodeKind::kOperator, Capability::kAlways, 5.0, "b"},
  };
  std::size_t ordinary_calls = 0;
  std::vector<graph::PartitionedGraphExecutor::NodeFunction> functions;
  for (std::size_t index = 0; index < nodes.size(); ++index) {
    functions.push_back([&] {
      ++ordinary_calls;
      return true;
    });
  }

  graph::NativeGraphCallbacks callbacks;
  callbacks.begin_capture = [] { return false; };
  callbacks.end_capture = [](void**) { return false; };
  callbacks.instantiate = [](void**, void*) { return false; };
  callbacks.launch = [](void*) { return false; };
  callbacks.destroy_graph = [](void*) { return true; };
  callbacks.destroy_graph_exec = [](void*) { return true; };

  graph::CapturePartitionOptions options;
  options.min_nodes = 2;
  graph::PartitionedGraphExecutor executor;
  context->Expect(executor.Prepare(nodes, std::move(functions), {}, options,
                                   std::move(callbacks)),
                  "Capture failure should preserve an ordinary fallback.");
  context->ExpectEqual(executor.capture_failure_count(), std::size_t{1},
                       "Capture failure should be reported.");
  context->Expect(executor.plan().regions.empty(),
                  "A failed native capture must not become active.");
  context->Expect(executor.Execute(), "Ordinary fallback execution failed.");
  context->ExpectEqual(ordinary_calls, nodes.size(),
                       "Every failed-region node should execute normally.");
}

void TestNativeRuntimeAdapter(infini::rt::test::TestContext* context) {
  auto callbacks = graph::MakeNativeGraphCallbacks<FakeRuntime>(nullptr);
  void* graph_handle = nullptr;
  void* graph_exec = nullptr;

  context->Expect(callbacks.begin_capture(),
                  "Runtime adapter failed to begin capture.");
  context->Expect(callbacks.end_capture(&graph_handle),
                  "Runtime adapter failed to end capture.");
  context->Expect(callbacks.instantiate(&graph_exec, graph_handle),
                  "Runtime adapter failed to instantiate a graph.");
  context->Expect(callbacks.launch(graph_exec),
                  "Runtime adapter failed to launch a graph.");
  context->Expect(callbacks.destroy_graph_exec(graph_exec),
                  "Runtime adapter failed to destroy a graph executable.");
  context->Expect(callbacks.destroy_graph(graph_handle),
                  "Runtime adapter failed to destroy a graph.");
}

}  // namespace

int main() {
  infini::rt::test::TestContext context;

  TestPartialCapture(&context);
  TestCaptureProfitability(&context);
  TestPartitionedExecution(&context);
  TestCaptureFailureFallback(&context);
  TestNativeRuntimeAdapter(&context);

  return context.ExitCode();
}
