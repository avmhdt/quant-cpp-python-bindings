#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "qkern/rolling_moments.hpp"

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

std::vector<double> run(const std::vector<double>& x, std::size_t window, std::size_t min_periods,
                        qkern::NanPolicy nan = qkern::NanPolicy::Skip) {
  std::vector<double> out(x.size());
  qkern::rolling_mean(qkern::StridedView<double>{x.data(), x.size(), 1},
                      qkern::OutView{out.data(), out.size(), 1},
                      qkern::RollingConfig{window, min_periods, 1, nan});
  return out;
}

void expect_same(const std::vector<double>& got, const std::vector<double>& want) {
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (std::isnan(want[i])) {
      EXPECT_TRUE(std::isnan(got[i])) << "index " << i << " got " << got[i];
    } else {
      EXPECT_DOUBLE_EQ(got[i], want[i]) << "index " << i;
    }
  }
}

TEST(RollingMean, HandComputed) {
  expect_same(run({1, 2, 3, 4, 5}, 3, 3), {kNaN, kNaN, 2, 3, 4});
  expect_same(run({1, 2, 3, 4, 5}, 3, 1), {1, 1.5, 2, 3, 4});
}

TEST(RollingMean, WindowOneIsIdentity) { expect_same(run({3, -1, 7}, 1, 1), {3, -1, 7}); }

TEST(RollingMean, WindowLongerThanInput) {
  expect_same(run({2, 4}, 10, 1), {2, 3});
  expect_same(run({2, 4}, 10, 10), {kNaN, kNaN});
}

TEST(RollingMean, EmptyInput) { EXPECT_TRUE(run({}, 3, 1).empty()); }

TEST(RollingMean, NaNSkipUsesValidValues) {
  expect_same(run({1, kNaN, 3, 5}, 2, 1), {1, 1, 3, 4});
  expect_same(run({1, kNaN, 3, 5}, 2, 2), {kNaN, kNaN, kNaN, 4});
}

TEST(RollingMean, NaNPropagate) {
  expect_same(run({1, kNaN, 3, 5, 7}, 2, 1, qkern::NanPolicy::Propagate), {1, kNaN, kNaN, 4, 6});
}

TEST(RollingMean, NaNRaise) {
  EXPECT_THROW(run({1, kNaN}, 2, 1, qkern::NanPolicy::Raise), std::invalid_argument);
  expect_same(run({1, 3}, 2, 1, qkern::NanPolicy::Raise), {1, 2});
}

TEST(RollingMean, InfinitiesAreCountedApart) {
  // +inf, then both signs, then only -inf, then finite again with clean sums.
  expect_same(run({1, kInf, -kInf, 2, 3, 4}, 2, 1), {1, kInf, kNaN, -kInf, 2.5, 3.5});
}

TEST(RollingMean, RingBufferWraparoundMatchesDirectMean) {
  std::vector<double> x;
  x.reserve(1000);
  for (int i = 0; i < 1000; ++i) {
    x.push_back(std::sin(0.1 * i) * 100.0 + i);
  }
  const std::size_t w = 7;
  const auto got = run(x, w, w);
  for (std::size_t t = w - 1; t < x.size(); ++t) {
    double s = 0.0;
    for (std::size_t j = t + 1 - w; j <= t; ++j) {
      s += x[j];
    }
    EXPECT_NEAR(got[t], s / static_cast<double>(w), 1e-9) << "t=" << t;
  }
}

TEST(RollingMean, StreamingEqualsBatchBitForBit) {
  const std::vector<double> x{0.5, kNaN, 2.25, -1.0, kInf, 3.0, 4.0, kNaN, 8.0};
  const auto batch = run(x, 3, 2);
  qkern::RollingMoments state(qkern::RollingConfig{3, 2});
  for (std::size_t i = 0; i < x.size(); ++i) {
    state.push(x[i]);
    const double m = state.mean();
    if (std::isnan(batch[i])) {
      EXPECT_TRUE(std::isnan(m));
    } else {
      EXPECT_EQ(m, batch[i]);
    }
  }
  state.reset();
  EXPECT_EQ(state.count(), 0U);
  EXPECT_TRUE(std::isnan(state.mean()));
}

TEST(RollingMean, FloatInput) {
  const std::vector<float> x{1.0F, 2.0F, 4.0F};
  std::vector<double> out(3);
  qkern::rolling_mean(qkern::StridedView<float>{x.data(), 3, 1}, qkern::OutView{out.data(), 3, 1},
                      qkern::RollingConfig{2, 1});
  expect_same(out, {1, 1.5, 3});
}

TEST(RollingMean, LengthMismatchThrows) {
  const std::vector<double> x{1, 2, 3};
  std::vector<double> out(2);
  EXPECT_THROW(qkern::rolling_mean(qkern::StridedView<double>{x.data(), 3, 1},
                                   qkern::OutView{out.data(), 2, 1}, qkern::RollingConfig{2, 1}),
               std::invalid_argument);
}

}  // namespace
