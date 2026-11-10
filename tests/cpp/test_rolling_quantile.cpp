#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "qkern/rolling_quantile.hpp"
#include "test_helpers.hpp"

namespace {

using qkern::Interpolation;
using qkern::NanPolicy;
using qkern::RollingConfig;
using qkern_test::in;
using qkern_test::kInf;
using qkern_test::kNaN;
using qkern_test::out;

std::vector<double> run(const std::vector<double>& x, RollingConfig cfg, double q,
                        Interpolation interp = Interpolation::Linear) {
  std::vector<double> o(x.size());
  qkern::rolling_quantile(in(x), out(o), cfg, q, interp);
  return o;
}

// pandas' definition on a sorted window.
double brute(std::vector<double> v, double q, Interpolation interp) {
  std::sort(v.begin(), v.end());
  const double h = q * static_cast<double>(v.size() - 1);
  const double k = std::floor(h);
  const double frac = h - k;
  const auto i = static_cast<std::size_t>(k);
  const double lo = v[i];
  if (frac == 0.0) {
    return lo;
  }
  const double hi = v[i + 1];
  switch (interp) {
    case Interpolation::Lower:
      return lo;
    case Interpolation::Higher:
      return hi;
    case Interpolation::Midpoint:
      return (lo + hi) / 2.0;
    case Interpolation::Nearest:
      if (frac == 0.5) {
        return std::fmod(k, 2.0) == 0.0 ? lo : hi;
      }
      return frac < 0.5 ? lo : hi;
    case Interpolation::Linear:
    default:
      return lo + (hi - lo) * frac;
  }
}

TEST(RollingQuantile, ParseAndValidate) {
  EXPECT_EQ(qkern::parse_interpolation("linear"), Interpolation::Linear);
  EXPECT_EQ(qkern::parse_interpolation("lower"), Interpolation::Lower);
  EXPECT_EQ(qkern::parse_interpolation("higher"), Interpolation::Higher);
  EXPECT_EQ(qkern::parse_interpolation("nearest"), Interpolation::Nearest);
  EXPECT_EQ(qkern::parse_interpolation("midpoint"), Interpolation::Midpoint);
  EXPECT_THROW((void)qkern::parse_interpolation("cubic"), std::invalid_argument);
  EXPECT_THROW((void)qkern::RollingQuantile(RollingConfig{3, 1}, 1.5, Interpolation::Linear),
               std::invalid_argument);
  EXPECT_THROW((void)qkern::RollingQuantile(RollingConfig{3, 1}, kNaN, Interpolation::Linear),
               std::invalid_argument);
}

TEST(RollingQuantile, MedianHandComputed) {
  qkern_test::expect_near_nan(run({5, 1, 4, 2, 3}, {3, 3}, 0.5), {kNaN, kNaN, 4, 2, 3}, 0);
  qkern_test::expect_near_nan(run({5, 1, 4, 2}, {4, 2}, 0.5), {kNaN, 3, 4, 3}, 0);
}

TEST(RollingQuantile, MatchesSortedWindowForEveryInterpolation) {
  std::vector<double> x = qkern_test::gaussian(3000, 101);
  qkern_test::add_nans(x, 0.1, 102);
  for (std::size_t i = 0; i < x.size(); i += 7) {
    x[i] = std::round(x[i]);  // duplicates
  }
  for (Interpolation interp : {Interpolation::Linear, Interpolation::Lower, Interpolation::Higher,
                               Interpolation::Nearest, Interpolation::Midpoint}) {
    for (double q : {0.0, 0.1, 0.25, 0.5, 0.9, 1.0}) {
      for (std::size_t w : {1U, 2U, 5U, 64U}) {
        const auto got = run(x, {w, 1}, q, interp);
        for (std::size_t t = 0; t < x.size(); ++t) {
          const auto vals = qkern_test::window_values(x, t, w);
          if (vals.empty()) {
            EXPECT_TRUE(std::isnan(got[t]));
          } else {
            EXPECT_EQ(got[t], brute(vals, q, interp)) << "q=" << q << " w=" << w << " t=" << t;
          }
        }
      }
    }
  }
}

TEST(RollingQuantile, RefillAfterTheLowHeapEmpties) {
  // Found by hypothesis: once the low heap empties, a new value must still go to the heap
  // that keeps low <= high.
  qkern_test::expect_near_nan(run({0, 0, 0, 1}, {2, 1}, 0.0), {0, 0, 0, 0}, 0);
  qkern_test::expect_near_nan(run({0, 0, 0, 1}, {2, 1}, 1.0), {0, 0, 0, 1}, 0);
}

TEST(RollingQuantile, NaNPoliciesAndInfinities) {
  const std::vector<double> x{1, kNaN, 3, kInf, -kInf, 2};
  qkern_test::expect_near_nan(run(x, {2, 1}, 1.0), {1, 1, 3, kInf, kInf, 2}, 0);
  qkern_test::expect_near_nan(run(x, {2, 1, 1, NanPolicy::Propagate}, 0.0),
                              {1, kNaN, kNaN, 3, -kInf, -kInf}, 0);
  EXPECT_THROW(run(x, {2, 1, 1, NanPolicy::Raise}, 0.5), std::invalid_argument);
  // Equal infinities interpolate to themselves rather than inf - inf = NaN.
  qkern_test::expect_near_nan(run({kInf, kInf}, {2, 2}, 0.5), {kNaN, kInf}, 0);
}

TEST(RollingQuantile, SignedZerosShareOneKey) {
  const std::vector<double> x{0.0, -0.0, 0.0, -0.0, 1.0, 1.0, 1.0};
  qkern_test::expect_near_nan(run(x, {3, 3}, 0.5), {kNaN, kNaN, 0, 0, 0, 1, 1}, 0);
}

TEST(RollingQuantile, HeapsAreRebuiltWhenDeadEntriesPileUp) {
  // Increasing data with q = 0 keeps dead entries deep in the high heap; the rebuild bounds
  // memory without changing results.
  std::vector<double> x(20000);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = static_cast<double>(i % 977) + 0.5 * static_cast<double>(i) / 977.0;
  }
  const auto got = run(x, {50, 50}, 0.0, Interpolation::Lower);
  for (std::size_t t = 49; t < x.size(); t += 97) {
    const auto vals = qkern_test::window_values(x, t, 50);
    EXPECT_EQ(got[t], *std::min_element(vals.begin(), vals.end()));
  }
}

TEST(RollingQuantile, StreamingSaveLoadAndReset) {
  std::vector<double> x = qkern_test::gaussian(500, 103);
  qkern_test::add_nans(x, 0.05, 104);
  const RollingConfig cfg{20, 5};
  qkern::RollingQuantile a(cfg, 0.3, Interpolation::Nearest);
  qkern::RollingQuantile b(cfg, 0.3, Interpolation::Nearest);
  for (std::size_t i = 0; i < 250; ++i) {
    a.push(x[i]);
  }
  b.load(a.save());
  for (std::size_t i = 250; i < x.size(); ++i) {
    a.push(x[i]);
    b.push(x[i]);
    qkern_test::expect_identical({a.value()}, {b.value()});
  }
  EXPECT_EQ(a.quantile(), 0.3);
  EXPECT_EQ(a.interpolation(), Interpolation::Nearest);
  EXPECT_EQ(a.config().window, 20U);
  a.reset();
  EXPECT_EQ(a.count(), 0U);
  EXPECT_FALSE(a.ready());
  qkern::State bad;
  bad.buffer.assign(21, 1.0);
  EXPECT_THROW(b.load(bad), std::invalid_argument);
}

}  // namespace
