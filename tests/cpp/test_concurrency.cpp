// Run under ThreadSanitizer (ctest --preset tsan -R concurrency): 8 threads run batch kernels
// on one shared read-only input with separate outputs, and separate streaming objects per
// thread. The kernels keep no global state, so TSan must report nothing.
#include <gtest/gtest.h>

#include <cstddef>
#include <thread>
#include <vector>

#include "qkern/ewm.hpp"
#include "qkern/rolling_minmax.hpp"
#include "qkern/rolling_moments.hpp"
#include "qkern/rolling_ols.hpp"
#include "test_helpers.hpp"

namespace {

using qkern::RollingConfig;
using qkern_test::in;
using qkern_test::out;

constexpr std::size_t kThreads = 8;

TEST(Threads, BatchKernelsOnSharedInput) {
  const auto x = qkern_test::gaussian(20000, 81);
  const auto y = qkern_test::gaussian(20000, 82);
  std::vector<double> reference(x.size());
  qkern::rolling_var(in(x), out(reference), RollingConfig{50, 10});

  std::vector<std::vector<double>> outputs(kThreads, std::vector<double>(x.size()));
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (std::size_t k = 0; k < kThreads; ++k) {
    threads.emplace_back([&, k] {
      std::vector<double>& o = outputs[k];
      std::vector<double> scratch(x.size());
      std::vector<double> scratch2(x.size());
      std::vector<double> scratch3(x.size());
      qkern::rolling_mean(in(x), out(scratch), RollingConfig{50, 10});
      qkern::rolling_zscore(in(x), out(scratch), RollingConfig{50, 10}, false);
      qkern::ewm_var(in(x), out(scratch), qkern::EwmConfig{0.1});
      qkern::rolling_max(in(x), out(scratch), RollingConfig{50, 1});
      qkern::rolling_ols(in(y), in(x), out(scratch), out(scratch2), out(scratch3), out(o),
                         RollingConfig{50, 10}, true);
      qkern::rolling_var(in(x), out(o), RollingConfig{50, 10});
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  for (const auto& o : outputs) {
    qkern_test::expect_identical(o, reference);
  }
}

TEST(Threads, SeparateStreamingObjects) {
  const auto x = qkern_test::gaussian(10000, 83);
  std::vector<double> results(kThreads);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (std::size_t k = 0; k < kThreads; ++k) {
    threads.emplace_back([&, k] {
      qkern::RollingMoments moments(RollingConfig{100, 100});
      qkern::Ewm ewm(qkern::EwmConfig{0.05});
      qkern::RollingMinMax minmax(RollingConfig{100, 1});
      for (double v : x) {
        moments.push(v);
        ewm.push(v);
        minmax.push(v);
      }
      results[k] = moments.var() + ewm.var() + minmax.max();
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  for (double r : results) {
    EXPECT_EQ(r, results[0]);
  }
}

}  // namespace
