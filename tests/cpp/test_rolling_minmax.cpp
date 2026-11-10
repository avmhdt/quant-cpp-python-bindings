#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "qkern/rolling_minmax.hpp"
#include "test_helpers.hpp"

namespace {

using qkern::NanPolicy;
using qkern::RollingConfig;
using qkern_test::in;
using qkern_test::kInf;
using qkern_test::kNaN;
using qkern_test::out;

std::vector<double> max_of(const std::vector<double>& x, RollingConfig cfg) {
  std::vector<double> o(x.size());
  qkern::rolling_max(in(x), out(o), cfg);
  return o;
}

std::vector<double> min_of(const std::vector<double>& x, RollingConfig cfg) {
  std::vector<double> o(x.size());
  qkern::rolling_min(in(x), out(o), cfg);
  return o;
}

TEST(RollingMinMax, HandComputed) {
  const std::vector<double> x{3, 1, 4, 1, 5, 9, 2, 6};
  qkern_test::expect_near_nan(max_of(x, {3, 3}), {kNaN, kNaN, 4, 4, 5, 9, 9, 9}, 0);
  qkern_test::expect_near_nan(min_of(x, {3, 3}), {kNaN, kNaN, 1, 1, 1, 1, 2, 2}, 0);
  qkern_test::expect_near_nan(max_of(x, {3, 1}), {3, 3, 4, 4, 5, 9, 9, 9}, 0);
  qkern_test::expect_near_nan(min_of(x, {1, 1}), x, 0);
}

TEST(RollingMinMax, DuplicatesAndTies) {
  const std::vector<double> x{2, 2, 2, 1, 1, 3};
  qkern_test::expect_near_nan(max_of(x, {2, 1}), {2, 2, 2, 2, 1, 3}, 0);
  qkern_test::expect_near_nan(min_of(x, {2, 1}), {2, 2, 2, 1, 1, 1}, 0);
}

TEST(RollingMinMax, NaNPolicies) {
  const std::vector<double> x{1, kNaN, 3, 2, kNaN, kNaN, 0};
  qkern_test::expect_near_nan(max_of(x, {2, 1}), {1, 1, 3, 3, 2, kNaN, 0}, 0);
  qkern_test::expect_near_nan(max_of(x, {2, 1, 1, NanPolicy::Propagate}),
                              {1, kNaN, kNaN, 3, kNaN, kNaN, kNaN}, 0);
  EXPECT_THROW(max_of(x, {2, 1, 1, NanPolicy::Raise}), std::invalid_argument);
}

TEST(RollingMinMax, InfinitiesOrderNormally) {
  const std::vector<double> x{1, kInf, 2, -kInf, 3};
  qkern_test::expect_near_nan(max_of(x, {2, 1}), {1, kInf, kInf, 2, 3}, 0);
  qkern_test::expect_near_nan(min_of(x, {2, 1}), {1, 1, 2, -kInf, -kInf}, 0);
}

TEST(RollingMinMax, MatchesBruteForce) {
  std::vector<double> x = qkern_test::gaussian(4000, 51);
  qkern_test::add_nans(x, 0.15, 52);
  for (std::size_t w : {1U, 2U, 5U, 64U, 1000U}) {
    for (std::size_t mp : {std::size_t{1}, w}) {
      const auto mx = max_of(x, {w, mp});
      const auto mn = min_of(x, {w, mp});
      for (std::size_t t = 0; t < x.size(); ++t) {
        const auto vals = qkern_test::window_values(x, t, w);
        if (vals.size() < mp) {
          EXPECT_TRUE(std::isnan(mx[t]));
          EXPECT_TRUE(std::isnan(mn[t]));
        } else {
          EXPECT_EQ(mx[t], *std::max_element(vals.begin(), vals.end())) << "w=" << w;
          EXPECT_EQ(mn[t], *std::min_element(vals.begin(), vals.end())) << "w=" << w;
        }
      }
    }
  }
}

TEST(RollingMinMax, LinearWorkOnAdversarialMonotoneInput) {
  // Increasing input is the worst case for the max deque's back pops, decreasing input for
  // the min deque's; each element is pushed once and popped at most once.
  const std::size_t n = 100000;
  const std::size_t w = 1000;
  for (int direction : {1, -1}) {
    qkern::RollingMinMax st(RollingConfig{w, 1});
    for (std::size_t i = 0; i < n; ++i) {
      st.push(direction * static_cast<double>(i));
    }
    // Two deques, each at most one push and one pop per element.
    EXPECT_LE(st.ops(), 4 * n) << "direction " << direction;
    EXPECT_EQ(st.max(), direction > 0 ? static_cast<double>(n - 1) : -static_cast<double>(n - w));
    EXPECT_EQ(st.min(), direction > 0 ? static_cast<double>(n - w) : -static_cast<double>(n - 1));
  }
}

TEST(RollingMinMax, StreamingResetAndCount) {
  qkern::RollingMinMax st(RollingConfig{3, 2});
  st.push(5.0);
  EXPECT_FALSE(st.ready());
  EXPECT_TRUE(std::isnan(st.max()));
  st.push(kNaN);
  st.push(7.0);
  EXPECT_EQ(st.count(), 2U);
  EXPECT_EQ(st.max(), 7.0);
  EXPECT_EQ(st.min(), 5.0);
  st.reset();
  EXPECT_EQ(st.count(), 0U);
  EXPECT_EQ(st.config().window, 3U);
  // A max-only state reports NaN for min.
  qkern::RollingMinMax max_only(RollingConfig{2, 1}, false, true);
  max_only.push(1.0);
  EXPECT_TRUE(std::isnan(max_only.min()));
  EXPECT_EQ(max_only.max(), 1.0);
}

TEST(RollingMinMax, LengthMismatch) {
  const std::vector<double> x{1, 2};
  std::vector<double> o(3);
  EXPECT_THROW(qkern::rolling_min(in(x), out(o), {2, 1}), std::invalid_argument);
  EXPECT_THROW(qkern::rolling_max(in(x), out(o), {2, 1}), std::invalid_argument);
}

}  // namespace
