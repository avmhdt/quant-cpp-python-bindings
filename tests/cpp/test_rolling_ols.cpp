#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "qkern/rolling_ols.hpp"
#include "test_helpers.hpp"

namespace {

using qkern::NanPolicy;
using qkern::RollingConfig;
using qkern_test::in;
using qkern_test::kInf;
using qkern_test::kNaN;
using qkern_test::out;

struct Ols {
  std::vector<double> alpha, beta, resid, r2;
};

Ols run(const std::vector<double>& y, const std::vector<double>& x, RollingConfig cfg,
        bool include_current = true) {
  Ols o{std::vector<double>(y.size()), std::vector<double>(y.size()), std::vector<double>(y.size()),
        std::vector<double>(y.size())};
  qkern::rolling_ols(in(y), in(x), out(o.alpha), out(o.beta), out(o.resid), out(o.r2), cfg,
                     include_current);
  return o;
}

// Reference fit over the pairwise-complete values of one window.
struct Fit {
  double alpha = kNaN;
  double beta = kNaN;
  double r2 = kNaN;
  double sxx = kNaN;
  double syy = kNaN;
};

Fit fit(const std::vector<double>& y, const std::vector<double>& x, std::size_t t, std::size_t w) {
  std::vector<double> ys;
  std::vector<double> xs;
  for (std::size_t j = (t + 1 >= w ? t + 1 - w : 0); j <= t; ++j) {
    if (!std::isnan(y[j]) && !std::isnan(x[j])) {
      ys.push_back(y[j]);
      xs.push_back(x[j]);
    }
  }
  Fit f;
  if (ys.size() < 2) {
    return f;
  }
  const auto mx = qkern_test::exact_moments(xs, 0).mean;
  const auto my = qkern_test::exact_moments(ys, 0).mean;
  qkern::detail::CompensatedSum sxx;
  qkern::detail::CompensatedSum syy;
  qkern::detail::CompensatedSum sxy;
  for (std::size_t i = 0; i < ys.size(); ++i) {
    sxx.add((xs[i] - mx) * (xs[i] - mx));
    syy.add((ys[i] - my) * (ys[i] - my));
    sxy.add((xs[i] - mx) * (ys[i] - my));
  }
  f.sxx = sxx.value();
  f.syy = syy.value();
  f.beta = sxy.value() / sxx.value();
  f.alpha = my - f.beta * mx;
  f.r2 = sxy.value() * sxy.value() / (sxx.value() * syy.value());
  return f;
}

TEST(RollingOls, ExactLine) {
  const std::vector<double> x{0, 1, 2, 3, 4};
  const std::vector<double> y{2, 5, 8, 11, 14};
  const auto o = run(y, x, {3, 2});
  qkern_test::expect_near_nan(o.beta, {kNaN, 3, 3, 3, 3}, 1e-14);
  qkern_test::expect_near_nan(o.alpha, {kNaN, 2, 2, 2, 2}, 1e-13);
  qkern_test::expect_near_nan(o.resid, {kNaN, 0, 0, 0, 0}, 1e-13);
  qkern_test::expect_near_nan(o.r2, {kNaN, 1, 1, 1, 1}, 1e-14);
}

TEST(RollingOls, ConstantXGivesNaNConstantYGivesZeroBeta) {
  const auto cx = run({1, 2, 3, 4}, {0.1, 0.1, 0.1, 0.1}, {3, 2});
  for (double b : cx.beta) {
    EXPECT_TRUE(std::isnan(b));
  }
  const auto cy = run({0.3, 0.3, 0.3, 0.3}, {1, 2, 4, 8}, {3, 2});
  qkern_test::expect_near_nan(cy.beta, {kNaN, 0, 0, 0}, 0);
  qkern_test::expect_near_nan(cy.alpha, {kNaN, 0.3, 0.3, 0.3}, 0);
  qkern_test::expect_near_nan(cy.resid, {kNaN, 0, 0, 0}, 0);
  for (double r : cy.r2) {
    EXPECT_TRUE(std::isnan(r));
  }
}

TEST(RollingOls, PairwiseCompleteNaN) {
  const std::vector<double> x{0, 1, kNaN, 3, 4, 5};
  const std::vector<double> y{1, 3, 5, kNaN, 9, 11};
  const auto o = run(y, x, {4, 2});
  // t=4 window is rows 1..4; complete pairs (x, y) = (1, 3) and (4, 9): y = 1 + 2x.
  EXPECT_NEAR(o.beta[4], 2.0, 1e-14);
  EXPECT_NEAR(o.alpha[4], 1.0, 1e-14);
  EXPECT_TRUE(std::isnan(o.resid[2]));
  EXPECT_TRUE(std::isnan(o.resid[3]));
  const auto p = run(y, x, {4, 2, 1, NanPolicy::Propagate});
  EXPECT_TRUE(std::isnan(p.beta[4]));
  EXPECT_THROW(run(y, x, {4, 2, 1, NanPolicy::Raise}), std::invalid_argument);
}

TEST(RollingOls, InfinityInWindowGivesNaN) {
  const std::vector<double> x{0, 1, 2, kInf, 4, 5, 6};
  const std::vector<double> y{0, 2, 4, 6, 8, 10, 12};
  const auto o = run(y, x, {2, 2});
  qkern_test::expect_near_nan(o.beta, {kNaN, 2, 2, kNaN, kNaN, 2, 2}, 1e-14);
}

TEST(RollingOls, MatchesReferenceOnNoisyData) {
  auto x = qkern_test::gaussian(3000, 61);
  auto y = qkern_test::gaussian(3000, 62);
  for (std::size_t i = 0; i < x.size(); ++i) {
    y[i] += 0.8 * x[i] + 10.0;
  }
  qkern_test::add_nans(x, 0.05, 63);
  qkern_test::add_nans(y, 0.05, 64);
  for (std::size_t w : {3U, 20U, 252U}) {
    const auto o = run(y, x, {w, 2});
    for (std::size_t t = 0; t < x.size(); ++t) {
      const Fit f = fit(y, x, t, w);
      if (std::isnan(f.beta)) {
        EXPECT_TRUE(std::isnan(o.beta[t]));
        continue;
      }
      // The sliding co-moments carry an absolute error of order 1e-14 (between resyncs), so
      // ratios lose relative accuracy when a window's Sxx or Syy is tiny: the tolerance
      // scales with 1 / Sxx and 1 / Syy.
      const double cond = 1e-12 * (1.0 / f.sxx + 1.0 / f.syy);
      EXPECT_NEAR(o.beta[t], f.beta, 1e-9 * std::max(1.0, std::fabs(f.beta)) + cond) << t;
      EXPECT_NEAR(o.alpha[t], f.alpha, 1e-9 * std::max(1.0, std::fabs(f.alpha)) + 10 * cond)
          << "t=" << t;
      EXPECT_NEAR(o.r2[t], f.r2, 1e-9 + cond) << "t=" << t;
      if (!std::isnan(x[t]) && !std::isnan(y[t])) {
        EXPECT_NEAR(o.resid[t], y[t] - f.alpha - f.beta * x[t], 1e-8 + 10 * cond) << "t=" << t;
      }
    }
  }
}

TEST(RollingOls, ExcludeCurrentUsesPreviousWindow) {
  const auto x = qkern_test::gaussian(500, 65);
  auto y = qkern_test::gaussian(500, 66);
  const auto in_sample = run(y, x, {30, 10}, true);
  const auto oos = run(y, x, {30, 10}, false);
  for (std::size_t t = 1; t < x.size(); ++t) {
    qkern_test::expect_identical({oos.beta[t]}, {in_sample.beta[t - 1]});
    qkern_test::expect_identical({oos.alpha[t]}, {in_sample.alpha[t - 1]});
    if (!std::isnan(oos.beta[t])) {
      EXPECT_NEAR(oos.resid[t], y[t] - oos.alpha[t] - oos.beta[t] * x[t], 1e-12);
    }
  }
}

TEST(RollingOls, ResyncPathKeepsAccuracy) {
  const std::size_t n = qkern::kResyncInterval + 4000;
  const auto x = qkern_test::gaussian(n, 67, 50.0, 1.0);
  auto y = qkern_test::gaussian(n, 68);
  for (std::size_t i = 0; i < n; ++i) {
    y[i] += 1.5 * x[i];
  }
  const std::size_t w = 25;
  const auto o = run(y, x, {w, w});
  for (std::size_t t = n - 200; t < n; ++t) {
    const Fit f = fit(y, x, t, w);
    EXPECT_NEAR(o.beta[t], f.beta, 1e-9) << "t=" << t;
  }
}

TEST(RollingOls, ResyncWithNoFinitePairs) {
  qkern::RollingOls st(RollingConfig{3, 1});
  for (std::uint64_t i = 0; i < qkern::kResyncInterval; ++i) {
    st.push(kNaN, 1.0);
  }
  EXPECT_EQ(st.count(), 0U);
  st.push(1.0, 0.0);
  st.push(3.0, 1.0);
  EXPECT_DOUBLE_EQ(st.beta(), 2.0);
  st.reset();
  EXPECT_EQ(st.count(), 0U);
  EXPECT_TRUE(std::isnan(st.r2()));
  EXPECT_TRUE(std::isnan(st.resid(1.0, 1.0)));
  EXPECT_EQ(st.config().window, 3U);
}

TEST(RollingOls, LengthMismatch) {
  const std::vector<double> y{1, 2, 3};
  const std::vector<double> x{1, 2};
  std::vector<double> a(3);
  std::vector<double> b(3);
  std::vector<double> r(3);
  std::vector<double> q(3);
  EXPECT_THROW(qkern::rolling_ols(in(y), in(x), out(a), out(b), out(r), out(q), {2, 1}, true),
               std::invalid_argument);
}

}  // namespace
