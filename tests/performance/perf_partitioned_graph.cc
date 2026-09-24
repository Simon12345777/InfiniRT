#include <infini/rt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "perf_common.h"

#ifndef INFINI_RT_PERF_DEVICE_TYPE
#define INFINI_RT_PERF_DEVICE_TYPE infini::rt::Device::Type::kCpu
#endif

#ifndef INFINI_RT_PERF_SUPPORTS_GRAPH_CAPTURE
#define INFINI_RT_PERF_SUPPORTS_GRAPH_CAPTURE 0
#endif

namespace {

namespace graph = infini::rt::graph;
namespace perf = infini::rt::perf;
namespace runtime = infini::rt::runtime;

using BackendRuntime = runtime::Runtime<INFINI_RT_PERF_DEVICE_TYPE>;
using NodeFunction = graph::PartitionedGraphExecutor::NodeFunction;

struct Config {
  std::size_t bytes{4096};
  std::size_t region_nodes{64};
  std::size_t iterations{500};
  std::size_t samples{7};
  std::size_t warmup_iterations{50};
  double min_speedup{1.0};
};

struct Measurements {
  std::vector<double> samples_us;
  double mean_us{0.0};
  double median_us{0.0};
};

std::size_t ReadSizeEnvironment(std::string_view name,
                                std::size_t fallback) {
  const std::string key(name);
  const char* value = std::getenv(key.c_str());
  if (value == nullptr || *value == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const auto parsed = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0' || parsed == 0 ||
      parsed > std::numeric_limits<std::size_t>::max()) {
    std::cerr << "Invalid positive integer in " << name << ": " << value
              << std::endl;
    std::exit(2);
  }
  return static_cast<std::size_t>(parsed);
}

double ReadDoubleEnvironment(std::string_view name, double fallback) {
  const std::string key(name);
  const char* value = std::getenv(key.c_str());
  if (value == nullptr || *value == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const auto parsed = std::strtod(value, &end);
  if (end == value || *end != '\0' || !std::isfinite(parsed) || parsed < 0.0) {
    std::cerr << "Invalid non-negative number in " << name << ": " << value
              << std::endl;
    std::exit(2);
  }
  return parsed;
}

Config ReadConfig() {
  Config config;
  config.bytes = ReadSizeEnvironment("INFINI_RT_PERF_GRAPH_BYTES", config.bytes);
  config.region_nodes = ReadSizeEnvironment(
      "INFINI_RT_PERF_GRAPH_REGION_NODES", config.region_nodes);
  config.iterations = ReadSizeEnvironment("INFINI_RT_PERF_GRAPH_ITERATIONS",
                                          config.iterations);
  config.samples =
      ReadSizeEnvironment("INFINI_RT_PERF_GRAPH_SAMPLES", config.samples);
  config.warmup_iterations = ReadSizeEnvironment(
      "INFINI_RT_PERF_GRAPH_WARMUP", config.warmup_iterations);
  config.min_speedup = ReadDoubleEnvironment(
      "INFINI_RT_PERF_MIN_GRAPH_SPEEDUP", config.min_speedup);
  return config;
}

bool Check(runtime::Error status, std::string_view operation) {
  if (status == runtime::kSuccess) {
    return true;
  }
  std::cerr << operation << " failed with status " << static_cast<int>(status)
            << std::endl;
  return false;
}

std::vector<graph::NodeDesc> BuildNodes(std::size_t region_nodes) {
  std::vector<graph::NodeDesc> nodes;
  nodes.reserve(region_nodes * 2 + 1);
  for (std::size_t index = 0; index < region_nodes; ++index) {
    nodes.push_back({graph::NodeKind::kOperator,
                     graph::CaptureCapability::kAlways, 1.0,
                     "capture-safe-prefix-" + std::to_string(index)});
  }

  // This represents one InfiniLM host-dependent or unsupported boundary.
  // Under the legacy all-or-nothing policy it makes the entire forward pass
  // eager. The partitioned executor leaves only this node eager.
  nodes.push_back({graph::NodeKind::kExternalCall,
                   graph::CaptureCapability::kHostDependent, 1.0,
                   "host-dependent-boundary"});

  for (std::size_t index = 0; index < region_nodes; ++index) {
    nodes.push_back({graph::NodeKind::kOperator,
                     graph::CaptureCapability::kAlways, 1.0,
                     "capture-safe-suffix-" + std::to_string(index)});
  }
  return nodes;
}

std::vector<NodeFunction> BuildFunctions(
    std::size_t region_nodes, std::size_t bytes, void* source,
    void* destination, void* boundary_destination,
    typename BackendRuntime::Stream stream, std::size_t* host_boundary_calls) {
  std::vector<NodeFunction> functions;
  functions.reserve(region_nodes * 2 + 1);

  const auto append_copy = [&] {
    functions.emplace_back([=] {
      return BackendRuntime::MemcpyAsync(
                 destination, source, bytes,
                 BackendRuntime::kMemcpyDeviceToDevice, stream) ==
             BackendRuntime::kSuccess;
    });
  };

  for (std::size_t index = 0; index < region_nodes; ++index) {
    append_copy();
  }

  functions.emplace_back([=] {
    ++(*host_boundary_calls);
    // The branch/control part runs on the host while its selected eager work is
    // submitted to the same stream, preserving ordering with both graphs.
    return BackendRuntime::MemcpyAsync(
               boundary_destination, source, bytes,
               BackendRuntime::kMemcpyDeviceToDevice, stream) ==
           BackendRuntime::kSuccess;
  });

  for (std::size_t index = 0; index < region_nodes; ++index) {
    append_copy();
  }
  return functions;
}

template <typename Invoke>
bool Measure(Invoke&& invoke, runtime::Stream stream, const Config& config,
             Measurements* result) {
  for (std::size_t index = 0; index < config.warmup_iterations; ++index) {
    if (!invoke()) {
      return false;
    }
  }
  if (!Check(runtime::StreamSynchronize(stream), "warmup synchronization")) {
    return false;
  }

  result->samples_us.reserve(config.samples);
  for (std::size_t sample = 0; sample < config.samples; ++sample) {
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < config.iterations; ++index) {
      if (!invoke()) {
        return false;
      }
    }
    if (!Check(runtime::StreamSynchronize(stream),
               "measurement synchronization")) {
      return false;
    }
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_us =
        std::chrono::duration<double, std::micro>(end - start).count();
    result->samples_us.push_back(elapsed_us /
                                 static_cast<double>(config.iterations));
  }

  result->mean_us = perf::Mean(result->samples_us);
  result->median_us = perf::Median(result->samples_us);
  return true;
}

void PrintComparison(const Config& config, const graph::CapturePlan& plan,
                     const Measurements& eager,
                     const Measurements& partitioned) {
  const auto speedup = eager.median_us / partitioned.median_us;
  const auto latency_reduction =
      (eager.median_us - partitioned.median_us) / eager.median_us * 100.0;

  std::cout << std::setprecision(12)
            << "{\"backend\":\"" << perf::JsonEscape(INFINI_RT_PERF_BACKEND_NAME)
            << "\",\"benchmark\":\"partitioned_graph_comparison\""
            << ",\"params\":{"
            << "\"model_pattern\":\"infinilm_unsupported_boundary\""
            << ",\"baseline\":\"legacy_whole_graph_fallback_eager\""
            << ",\"candidate\":\"partitioned_graph\""
            << ",\"bytes_per_node\":" << config.bytes
            << ",\"region_nodes\":" << config.region_nodes << "}"
            << ",\"iterations\":" << config.iterations
            << ",\"unit\":\"x\""
            << ",\"mean\":" << eager.mean_us / partitioned.mean_us
            << ",\"median\":" << speedup
            << ",\"captured_regions\":" << plan.regions.size()
            << ",\"captured_nodes\":" << plan.captured_node_count
            << ",\"eager_nodes\":" << plan.uncaptured_node_count
            << ",\"baseline_median_us\":" << eager.median_us
            << ",\"partial_median_us\":" << partitioned.median_us
            << ",\"speedup\":" << speedup
            << ",\"latency_reduction_percent\":" << latency_reduction
            << ",\"minimum_required_speedup\":" << config.min_speedup
            << ",\"passed\":"
            << (speedup >= config.min_speedup ? "true" : "false") << "}"
            << std::endl;
}

}  // namespace

int main() {
  if constexpr (!INFINI_RT_PERF_SUPPORTS_GRAPH_CAPTURE) {
    perf::SkipBenchmark("perf_partitioned_graph",
                        "selected backend has no native graph capture");
    return 0;
  }

  const Config config = ReadConfig();
  infini::rt::set_runtime_device_type(INFINI_RT_PERF_DEVICE_TYPE);

  int device_count = 0;
  if (!Check(runtime::GetDeviceCount(&device_count), "device count query")) {
    return 1;
  }
  if (device_count == 0) {
    perf::SkipBenchmark("perf_partitioned_graph", "no device is available");
    return 0;
  }
  if (!Check(runtime::SetDevice(0), "device selection")) {
    return 1;
  }

  void* source = nullptr;
  void* destination = nullptr;
  void* boundary_destination = nullptr;
  runtime::Stream stream = nullptr;
  graph::PartitionedGraphExecutor executor;
  int exit_code = 0;

  if (!Check(runtime::Malloc(&source, config.bytes), "source allocation") ||
      !Check(runtime::Malloc(&destination, config.bytes),
             "destination allocation") ||
      !Check(runtime::Malloc(&boundary_destination, config.bytes),
             "boundary allocation") ||
      !Check(runtime::StreamCreate(&stream), "stream creation")) {
    exit_code = 1;
  }

  std::vector<std::uint8_t> input(config.bytes);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<std::uint8_t>((index * 37 + 11) & 0xff);
  }
  if (exit_code == 0 &&
      !Check(runtime::Memcpy(source, input.data(), input.size(),
                            runtime::kMemcpyHostToDevice),
             "source initialization")) {
    exit_code = 1;
  }

  const auto native_stream =
      reinterpret_cast<typename BackendRuntime::Stream>(stream);
  std::size_t eager_boundary_calls = 0;
  std::size_t partitioned_boundary_calls = 0;
  auto eager_functions = BuildFunctions(
      config.region_nodes, config.bytes, source, destination,
      boundary_destination, native_stream, &eager_boundary_calls);

  graph::CapturePartitionOptions options;
  options.min_nodes = 2;
  options.graph_launch_cost_us = 1.0;
  options.boundary_cost_us = 0.0;
  options.min_estimated_saving_us = 0.0;

  if (exit_code == 0) {
    auto partitioned_functions = BuildFunctions(
        config.region_nodes, config.bytes, source, destination,
        boundary_destination, native_stream, &partitioned_boundary_calls);
    if (!executor.Prepare(
            BuildNodes(config.region_nodes), std::move(partitioned_functions),
            {}, options,
            graph::MakeNativeGraphCallbacks<BackendRuntime>(native_stream))) {
      std::cerr << "Failed to prepare partitioned graph executor." << std::endl;
      exit_code = 1;
    } else if (executor.plan().regions.size() != 2 ||
               executor.plan().captured_node_count != config.region_nodes * 2 ||
               executor.plan().uncaptured_node_count != 1) {
      std::cerr << "Unexpected partition plan: regions="
                << executor.plan().regions.size()
                << ", captured_nodes=" << executor.plan().captured_node_count
                << ", eager_nodes=" << executor.plan().uncaptured_node_count
                << std::endl;
      exit_code = 1;
    } else if (!Check(runtime::StreamSynchronize(stream),
                      "capture synchronization")) {
      exit_code = 1;
    }
  }

  Measurements eager_measurements;
  Measurements partitioned_measurements;
  if (exit_code == 0) {
    const auto eager_invoke = [&] {
      return std::all_of(eager_functions.begin(), eager_functions.end(),
                         [](const NodeFunction& function) { return function(); });
    };
    if (!Measure(eager_invoke, stream, config, &eager_measurements) ||
        !Measure([&] { return executor.Execute(); }, stream, config,
                 &partitioned_measurements)) {
      std::cerr << "Benchmark execution failed." << std::endl;
      exit_code = 1;
    }
  }

  if (exit_code == 0) {
    const auto expected_calls =
        config.warmup_iterations + config.samples * config.iterations;
    if (eager_boundary_calls != expected_calls ||
        partitioned_boundary_calls != expected_calls) {
      std::cerr << "Host boundary execution count mismatch: eager="
                << eager_boundary_calls
                << ", partitioned=" << partitioned_boundary_calls
                << ", expected=" << expected_calls << std::endl;
      exit_code = 1;
    }
  }

  if (exit_code == 0) {
    std::vector<std::uint8_t> output(config.bytes, 0);
    if (!Check(runtime::Memcpy(output.data(), destination, output.size(),
                              runtime::kMemcpyDeviceToHost),
               "output validation copy") ||
        output != input) {
      std::cerr << "Partitioned graph output validation failed." << std::endl;
      exit_code = 1;
    }
  }

  if (exit_code == 0) {
    const std::vector<perf::Param> params{
        perf::StringParam("model_pattern", "infinilm_unsupported_boundary"),
        perf::NumberParam("bytes_per_node", config.bytes),
        perf::NumberParam("region_nodes", config.region_nodes)};
    perf::PrintResult("partitioned_graph_legacy_eager", params,
                      config.iterations, "us", eager_measurements.mean_us,
                      eager_measurements.median_us);
    perf::PrintResult("partitioned_graph_partial_capture", params,
                      config.iterations, "us",
                      partitioned_measurements.mean_us,
                      partitioned_measurements.median_us);
    PrintComparison(config, executor.plan(), eager_measurements,
                    partitioned_measurements);

    const auto speedup = eager_measurements.median_us /
                         partitioned_measurements.median_us;
    if (speedup < config.min_speedup) {
      std::cerr << "Observed speedup " << speedup
                << "x is below required " << config.min_speedup << "x."
                << std::endl;
      exit_code = 1;
    }
  }

  if (!executor.Reset()) {
    std::cerr << "Failed to destroy graph resources." << std::endl;
    exit_code = 1;
  }
  if (stream != nullptr && !Check(runtime::StreamDestroy(stream),
                                  "stream destruction")) {
    exit_code = 1;
  }
  if (boundary_destination != nullptr &&
      !Check(runtime::Free(boundary_destination), "boundary free")) {
    exit_code = 1;
  }
  if (destination != nullptr &&
      !Check(runtime::Free(destination), "destination free")) {
    exit_code = 1;
  }
  if (source != nullptr && !Check(runtime::Free(source), "source free")) {
    exit_code = 1;
  }
  return exit_code;
}
