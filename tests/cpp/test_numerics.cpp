// Numerical accuracy targets NFR-06 (a) and (b), plus a shortened drift check for (c); the
// full 1e8-update drift test runs in Python's slow suite.
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "qkern/rolling_moments.hpp"
#include "test_helpers.hpp"

namespace {

using qkern::RollingConfig;
using qkern_test::in;
using qkern_test::out;

// 1,000 window end points spread evenly over the series.
std::vector<std::size_t> sample_points(std::size_t n, std::size_t w) {
  std::vector<std::size_t> pts;
  for (std::size_t k = 0; k < 1000; ++k) {
    pts.push_back(w - 1 + (k * (n - w)) / 999);
  }
  return pts;
}

TEST(Numerics, GaussianMeanAndVarianceAgainstExactTwoPass) {
  // NFR-06 (a): n = 1e6, |mean error| <= 1e-12 max(1, |exact|), var/std relative <= 1e-10.
  const std::size_t n = 1000000;
  const auto x = qkern_test::gaussian(n, 71);
  for (std::size_t w : {20U, 252U, 2520U}) {
    std::vector<double> mean(n);
    std::vector<double> var(n);
    qkern::rolling_mean(in(x), out(mean), RollingConfig{w, w});
    qkern::rolling_var(in(x), out(var), RollingConfig{w, w});
    double worst_mean = 0.0;
    double worst_var = 0.0;
    for (std::size_t t : sample_points(n, w)) {
      const auto e = qkern_test::exact_moments(qkern_test::window_values(x, t, w), 1);
      worst_mean =
          std::max(worst_mean, std::fabs(mean[t] - e.mean) / std::max(1.0, std::fabs(e.mean)));
      worst_var = std::max(worst_var, std::fabs(var[t] - e.var) / e.var);
    }
    EXPECT_LE(worst_mean, 1e-12) << "w=" << w;
    EXPECT_LE(worst_var, 1e-10) << "w=" << w;
  }
}

TEST(Numerics, OffsetDataVarianceAndTheNaiveMethodFails) {
  // NFR-06 (b): x = 1e9 + N(0, 1); relative variance error <= 1e-6. The textbook
  // (sum x^2 - n mean^2) / (n - 1) over running sums fails the same bound.
  const std::size_t n = 200000;
  const auto x = qkern_test::gaussian(n, 72, 1e9, 1.0);
  for (std::size_t w : {20U, 252U}) {
    std::vector<double> var(n);
    qkern::rolling_var(in(x), out(var), RollingConfig{w, w});
    double s = 0.0;
    double s2 = 0.0;
    double worst = 0.0;
    double worst_naive = 0.0;
    for (std::size_t t = 0; t < n; ++t) {
      s += x[t];
      s2 += x[t] * x[t];
      if (t >= w) {
        s -= x[t - w];
        s2 -= x[t - w] * x[t - w];
      }
      if (t + 1 < w || t % 97 != 0) {
        continue;
      }
      const auto e = qkern_test::exact_moments(qkern_test::window_values(x, t, w), 1);
      const auto wd = static_cast<double>(w);
      const double naive = (s2 - s * s / wd) / (wd - 1.0);
      worst = std::max(worst, std::fabs(var[t] - e.var) / e.var);
      worst_naive = std::max(worst_naive, std::fabs(naive - e.var) / e.var);
    }
    EXPECT_LE(worst, 1e-6) << "w=" << w;
    EXPECT_GT(worst_naive, 1e-6) << "w=" << w << ": the naive method should fail here";
  }
}

TEST(Numerics, StreamingDriftStaysBounded) {
  // NFR-06 (c), shortened: 3e6 streaming updates (about 45 resync periods), w = 100.
  const std::size_t n = 3000000;
  const std::size_t w = 100;
  const auto x = qkern_test::gaussian(n, 73, 3.0, 1.0);
  qkern::RollingMoments st(RollingConfig{w, w});
  for (double v : x) {
    st.push(v);
  }
  const auto e = qkern_test::exact_moments(qkern_test::window_values(x, n - 1, w), 1);
  EXPECT_LE(std::fabs(st.var() - e.var) / e.var, 1e-9);
}

}  // namespace
