// Google Benchmark for the C++ kernels alone, without the binding layer. Informational: the
// published tables come from the Python harness (bench/py, tools/qkdev bench).
//
//   ~/build/quant-cpp-python-bindings/bench/bench/qkern_bench --benchmark_filter=Rolling
#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "qkern/qkern.hpp"

namespace {

std::vector<double> series(std::size_t n) {
  std::mt19937_64 rng(20261030);
  std::normal_distribution<double> dist(0.0, 1.0);
  std::vector<double> v(n);
  for (auto& x : v) {
    x = dist(rng);
  }
  return v;
}

qkern::StridedView<double> view(const std::vector<double>& v) { return {v.data(), v.size(), 1}; }

qkern::OutView view(std::vector<double>& v) { return {v.data(), v.size(), 1}; }

template <class Kernel>
void run(benchmark::State& state, Kernel kernel) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto w = static_cast<std::size_t>(state.range(1));
  const auto x = series(n);
  std::vector<double> out(n);
  const qkern::RollingConfig cfg{w, w};
  for (auto _ : state) {
    kernel(view(x), view(out), cfg);
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

// Benchmark names are CamelCase so `--benchmark_filter=Rolling` selects the rolling kernels.
// NOLINTBEGIN(readability-identifier-naming)
void RollingMean(benchmark::State& s) {
  run(s, [](auto x, auto o, const auto& c) { qkern::rolling_mean(x, o, c); });
}
void RollingStd(benchmark::State& s) {
  run(s, [](auto x, auto o, const auto& c) { qkern::rolling_std(x, o, c); });
}
void RollingZscore(benchmark::State& s) {
  run(s, [](auto x, auto o, const auto& c) { qkern::rolling_zscore(x, o, c, true); });
}
void RollingMax(benchmark::State& s) {
  run(s, [](auto x, auto o, const auto& c) { qkern::rolling_max(x, o, c); });
}
void EwmMean(benchmark::State& s) {
  run(s, [](auto x, auto o, const auto&) { qkern::ewm_mean(x, o, qkern::EwmConfig{0.05}); });
}
void EwmVar(benchmark::State& s) {
  run(s, [](auto x, auto o, const auto&) { qkern::ewm_var(x, o, qkern::EwmConfig{0.05}); });
}

void RollingOls(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto w = static_cast<std::size_t>(state.range(1));
  const auto x = series(n);
  const auto y = series(n);
  std::vector<double> a(n);
  std::vector<double> b(n);
  std::vector<double> r(n);
  std::vector<double> q(n);
  for (auto _ : state) {
    qkern::rolling_ols(view(y), view(x), view(a), view(b), view(r), view(q),
                       qkern::RollingConfig{w, w}, true);
    benchmark::DoNotOptimize(b.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

// NOLINTEND(readability-identifier-naming)

void bench_args(benchmark::Benchmark* b) {
  for (std::int64_t n : {100000, 10000000}) {
    for (std::int64_t w : {20, 252, 2520}) {
      b->Args({n, w});
    }
  }
  b->Unit(benchmark::kMillisecond);
}

}  // namespace

BENCHMARK(RollingMean)->Apply(bench_args);
BENCHMARK(RollingStd)->Apply(bench_args);
BENCHMARK(RollingZscore)->Apply(bench_args);
BENCHMARK(RollingMax)->Apply(bench_args);
BENCHMARK(RollingOls)->Apply(bench_args);
BENCHMARK(EwmMean)->Apply(bench_args);
BENCHMARK(EwmVar)->Apply(bench_args);
