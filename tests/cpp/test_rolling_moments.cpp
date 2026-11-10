#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "qkern/rolling_moments.hpp"
#include "test_helpers.hpp"

namespace {

using qkern::NanPolicy;
using qkern::RollingConfig;
using qkern_test::in;
using qkern_test::kInf;
using qkern_test::kNaN;
using qkern_test::out;

std::vector<double> var_of(const std::vector<double>& x, RollingConfig cfg) {
  std::vector<double> o(x.size());
  qkern::rolling_var(in(x), out(o), cfg);
  return o;
}

std::vector<double> std_of(const std::vector<double>& x, RollingConfig cfg) {
  std::vector<double> o(x.size());
  qkern::rolling_std(in(x), out(o), cfg);
  return o;
}

std::vector<double> z_of(const std::vector<double>& x, RollingConfig cfg, bool include_current) {
  std::vector<double> o(x.size());
  qkern::rolling_zscore(in(x), out(o), cfg, include_current);
  return o;
}

TEST(RollingVar, HandComputed) {
  // Windows of 3: {1,2,4} var 7/3, {2,4,8} var 28/3.
  qkern_test::expect_near_nan(var_of({1, 2, 4, 8}, {3, 3}), {kNaN, kNaN, 7.0 / 3, 28.0 / 3}, 1e-15);
  qkern_test::expect_near_nan(var_of({1, 2, 4, 8}, {3, 3, 0}), {kNaN, kNaN, 14.0 / 9, 56.0 / 9},
                              1e-15);
  qkern_test::expect_near_nan(std_of({1, 2, 4, 8}, {3, 2}),
                              {kNaN, std::sqrt(0.5), std::sqrt(7.0 / 3), std::sqrt(28.0 / 3)},
                              1e-15);
}

TEST(RollingVar, DdofNeedsEnoughObservations) {
  // ddof=1 and one valid value: NaN even with min_periods=1; ddof=0 gives 0.
  qkern_test::expect_near_nan(var_of({5, kNaN}, {2, 1, 1}), {kNaN, kNaN}, 0);
  qkern_test::expect_near_nan(var_of({5, kNaN}, {2, 1, 0}), {0, 0}, 0);
  qkern_test::expect_near_nan(var_of({1, 2, 3}, {3, 1, 3}), {kNaN, kNaN, kNaN}, 0);
}

TEST(RollingVar, NaNPolicies) {
  const std::vector<double> x{1, kNaN, 3, 5, 7, 9};
  qkern_test::expect_near_nan(var_of(x, {3, 2}), {kNaN, kNaN, 2, 2, 4, 4}, 1e-15);
  qkern_test::expect_near_nan(var_of(x, {3, 2, 1, NanPolicy::Propagate}),
                              {kNaN, kNaN, kNaN, kNaN, 4, 4}, 1e-15);
  EXPECT_THROW(var_of(x, {3, 2, 1, NanPolicy::Raise}), std::invalid_argument);
}

TEST(RollingVar, InfinityMakesVarianceNaNUntilItLeaves) {
  const std::vector<double> x{1, 2, kInf, 4, 5, 6};
  qkern_test::expect_near_nan(var_of(x, {2, 2}), {kNaN, 0.5, kNaN, kNaN, 0.5, 0.5}, 1e-15);
}

TEST(RollingVar, ConstantWindowIsExactlyZero) {
  std::vector<double> x = qkern_test::gaussian(500, 3);
  for (std::size_t i = 200; i < 300; ++i) {
    x[i] = 0.1;  // not representable: the add/remove path would leave rounding residue
  }
  const auto v = var_of(x, {20, 20});
  for (std::size_t t = 219; t < 300; ++t) {
    EXPECT_EQ(v[t], 0.0) << "t=" << t;
  }
  RollingConfig cfg{20, 20};
  qkern::RollingMoments st(cfg);
  for (std::size_t t = 0; t < 300; ++t) {
    st.push(x[t]);
  }
  EXPECT_EQ(st.mean(), 0.1);
  EXPECT_TRUE(std::isnan(st.zscore(0.1)));
}

TEST(RollingVar, NaNInsideConstantRunKeepsItConstant) {
  const std::vector<double> x{3, 3, kNaN, 3, 3};
  qkern_test::expect_near_nan(var_of(x, {4, 2}), {kNaN, 0, 0, 0, 0}, 0);
}

TEST(RollingVar, MatchesExactTwoPassWithNaN) {
  std::vector<double> x = qkern_test::gaussian(3000, 11, 5.0, 2.0);
  qkern_test::add_nans(x, 0.1, 12);
  for (std::size_t w : {1U, 2U, 7U, 50U, 400U}) {
    const RollingConfig cfg{w, 1};
    const auto v = var_of(x, cfg);
    std::vector<double> want(x.size());
    for (std::size_t t = 0; t < x.size(); ++t) {
      want[t] = qkern_test::exact_moments(qkern_test::window_values(x, t, w), 1).var;
    }
    qkern_test::expect_near_nan(v, want, 1e-10);
  }
}

TEST(RollingMoments, ResyncPathKeepsAccuracy) {
  // More than two resync intervals with a short window.
  const std::size_t n = 2 * qkern::kResyncInterval + 1234;
  const auto x = qkern_test::gaussian(n, 21, 100.0, 1.0);
  const std::size_t w = 30;
  const auto v = var_of(x, {w, w});
  for (std::size_t t = n - 500; t < n; ++t) {
    const auto e = qkern_test::exact_moments(qkern_test::window_values(x, t, w), 1);
    EXPECT_NEAR(v[t], e.var, 1e-10 * e.var) << "t=" << t;
  }
}

TEST(RollingMoments, ResyncWithAllNaNWindow) {
  RollingConfig cfg{4, 1};
  qkern::RollingMoments st(cfg);
  for (std::uint64_t i = 0; i < qkern::kResyncInterval; ++i) {
    st.push(kNaN);
  }
  EXPECT_EQ(st.count(), 0U);
  EXPECT_TRUE(std::isnan(st.mean()));
  st.push(2.0);
  st.push(4.0);
  EXPECT_EQ(st.mean(), 3.0);
  EXPECT_EQ(st.var(), 2.0);
}

TEST(RollingMoments, BatchMomentsReturnsMeanVarCount) {
  const std::vector<double> x{1, kNaN, 3, kInf, 5};
  std::vector<double> mean(5);
  std::vector<double> var(5);
  std::vector<std::int64_t> count(5);
  qkern::rolling_moments(in(x), out(mean), out(var),
                         qkern::CountOutView{count.data(), count.size(), 1}, {2, 1});
  qkern_test::expect_near_nan(mean, {1, 1, 3, kInf, kInf}, 0);
  qkern_test::expect_near_nan(var, {kNaN, kNaN, kNaN, kNaN, kNaN}, 0);
  EXPECT_EQ(count, (std::vector<std::int64_t>{1, 1, 1, 2, 2}));
}

TEST(RollingMoments, StreamingEqualsBatchBitForBit) {
  std::vector<double> x = qkern_test::gaussian(5000, 31);
  qkern_test::add_nans(x, 0.05, 32);
  x[100] = kInf;
  x[2000] = -kInf;
  const RollingConfig cfg{37, 5};
  const auto batch = var_of(x, cfg);
  qkern::RollingMoments st(cfg);
  std::vector<double> stream(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    st.push(x[i]);
    stream[i] = st.var();
  }
  qkern_test::expect_identical(batch, stream);
  st.reset();
  EXPECT_EQ(st.count(), 0U);
  EXPECT_FALSE(st.ready());
  EXPECT_EQ(st.config().window, 37U);
}

TEST(RollingZscore, IncludeCurrent) {
  // Window {1,2,3}: mean 2, std 1, z(3) = 1.
  const auto z = z_of({1, 2, 3, 3, 3}, {3, 3}, true);
  qkern_test::expect_near_nan(z, {kNaN, kNaN, 1.0, (3 - 8.0 / 3) / std::sqrt(1.0 / 3), kNaN},
                              1e-14);
}

TEST(RollingZscore, ExcludeCurrentUsesPreviousWindow) {
  // z_t against W_{t-1}: t=3 uses {1,2,3}: (6-2)/1 = 4.
  const auto z = z_of({1, 2, 3, 6}, {3, 3}, false);
  qkern_test::expect_near_nan(z, {kNaN, kNaN, kNaN, 4.0}, 1e-15);
}

TEST(RollingZscore, NaNAndInfInputs) {
  const auto z = z_of({1, 2, kNaN, 3, kInf, 5}, {3, 2}, true);
  EXPECT_TRUE(std::isnan(z[2]));  // x_t is NaN
  EXPECT_TRUE(std::isnan(z[4]));  // inf in window: std NaN
  EXPECT_TRUE(std::isnan(z[5]));
  const auto zx = z_of({1, 2, 3, kInf}, {3, 3}, false);
  EXPECT_EQ(zx[3], kInf);  // finite window, infinite x
}

TEST(RollingMoments, ConfigValidation) {
  EXPECT_THROW((void)qkern::RollingMoments(RollingConfig{0, 1}), std::invalid_argument);
  const std::vector<double> x{1, 2};
  std::vector<double> o(1);
  EXPECT_THROW(qkern::rolling_var(in(x), out(o), {2, 1}), std::invalid_argument);
  EXPECT_THROW(qkern::rolling_std(in(x), out(o), {2, 1}), std::invalid_argument);
  EXPECT_THROW(qkern::rolling_zscore(in(x), out(o), {2, 1}, true), std::invalid_argument);
}

}  // namespace
