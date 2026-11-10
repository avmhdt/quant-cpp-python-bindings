#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "qkern/ewm.hpp"
#include "test_helpers.hpp"

namespace {

using qkern::EwmConfig;
using qkern::NanPolicy;
using qkern_test::in;
using qkern_test::kInf;
using qkern_test::kNaN;
using qkern_test::out;

std::vector<double> mean_of(const std::vector<double>& x, EwmConfig cfg) {
  std::vector<double> o(x.size());
  qkern::ewm_mean(in(x), out(o), cfg);
  return o;
}

std::vector<double> var_of(const std::vector<double>& x, EwmConfig cfg) {
  std::vector<double> o(x.size());
  qkern::ewm_var(in(x), out(o), cfg);
  return o;
}

// Direct weighted form for adjust=False without NaN: observation k of t+1 has weight
// (1-a)^t for k = 0 and a (1-a)^(t-k) otherwise; the weights sum to 1.
struct Direct {
  double mean;
  double biased_var;
  double var;
};

Direct direct(const std::vector<double>& x, std::size_t t, double a) {
  std::vector<double> w(t + 1);
  for (std::size_t k = 0; k <= t; ++k) {
    w[k] = (k == 0 ? 1.0 : a) * std::pow(1.0 - a, static_cast<double>(t - k));
  }
  double m = 0.0;
  double sw2 = 0.0;
  for (std::size_t k = 0; k <= t; ++k) {
    m += w[k] * x[k];
    sw2 += w[k] * w[k];
  }
  double c = 0.0;
  for (std::size_t k = 0; k <= t; ++k) {
    c += w[k] * (x[k] - m) * (x[k] - m);
  }
  return {m, c, sw2 < 1.0 ? c / (1.0 - sw2) : kNaN};
}

TEST(EwmAlpha, FromSpanAndHalflife) {
  EXPECT_DOUBLE_EQ(qkern::alpha_from_span(1.0), 1.0);
  EXPECT_DOUBLE_EQ(qkern::alpha_from_span(9.0), 0.2);
  EXPECT_DOUBLE_EQ(qkern::alpha_from_halflife(1.0), 0.5);
  EXPECT_NEAR(std::pow(1.0 - qkern::alpha_from_halflife(10.0), 10.0), 0.5, 1e-15);
  EXPECT_THROW((void)qkern::alpha_from_span(0.5), std::invalid_argument);
  EXPECT_THROW((void)qkern::alpha_from_span(kNaN), std::invalid_argument);
  EXPECT_THROW((void)qkern::alpha_from_span(kInf), std::invalid_argument);
  EXPECT_THROW((void)qkern::alpha_from_halflife(0.0), std::invalid_argument);
  EXPECT_THROW((void)qkern::alpha_from_halflife(-1.0), std::invalid_argument);
  EXPECT_THROW((void)qkern::alpha_from_halflife(kInf), std::invalid_argument);
}

TEST(EwmConfigValidation, AlphaRange) {
  EXPECT_THROW((void)qkern::Ewm(EwmConfig{0.0}), std::invalid_argument);
  EXPECT_THROW((void)qkern::Ewm(EwmConfig{1.5}), std::invalid_argument);
  EXPECT_THROW((void)qkern::Ewm(EwmConfig{kNaN}), std::invalid_argument);
  EXPECT_NO_THROW((void)qkern::Ewm(EwmConfig{1.0}));
}

TEST(EwmMean, HandComputed) {
  qkern_test::expect_near_nan(mean_of({1, 2, 3}, {0.5}), {1, 1.5, 2.25}, 1e-15);
  qkern_test::expect_near_nan(mean_of({1, 2, 3}, {1.0}), {1, 2, 3}, 0);
}

TEST(EwmMean, NaNSkipKeepsState) {
  qkern_test::expect_near_nan(mean_of({kNaN, 1, kNaN, 3}, {0.5}), {kNaN, 1, 1, 2}, 1e-15);
}

TEST(EwmMean, NaNPropagatePoisonsUntilReset) {
  EwmConfig cfg{0.5};
  cfg.nan = NanPolicy::Propagate;
  qkern_test::expect_near_nan(mean_of({1, kNaN, 3}, cfg), {1, kNaN, kNaN}, 0);
  qkern::Ewm st(cfg);
  st.push(1.0);
  st.push(kNaN);
  EXPECT_TRUE(std::isnan(st.mean()));
  st.reset();
  st.push(4.0);
  EXPECT_EQ(st.mean(), 4.0);
}

TEST(EwmMean, NaNRaise) {
  EwmConfig cfg{0.5};
  cfg.nan = NanPolicy::Raise;
  EXPECT_THROW(mean_of({1, kNaN}, cfg), std::invalid_argument);
}

TEST(EwmMean, MinPeriodsCountsValidObservations) {
  EwmConfig cfg{0.5};
  cfg.min_periods = 2;
  qkern_test::expect_near_nan(mean_of({1, kNaN, 2, 3}, cfg), {kNaN, kNaN, 1.5, 2.25}, 1e-15);
}

TEST(EwmMean, InfinityFollowsTheRecurrence) {
  qkern_test::expect_near_nan(mean_of({1, kInf, 2, -kInf, 3}, {0.5}), {1, kInf, kInf, kNaN, kNaN},
                              0);
}

TEST(EwmVar, MatchesDirectWeightedForm) {
  const auto x = qkern_test::gaussian(200, 41, 3.0, 2.0);
  for (double a : {0.05, 0.3, 0.9}) {
    EwmConfig unbiased{a};
    EwmConfig biased{a, true};
    const auto m = mean_of(x, unbiased);
    const auto v = var_of(x, unbiased);
    const auto vb = var_of(x, biased);
    for (std::size_t t = 0; t < x.size(); ++t) {
      const Direct d = direct(x, t, a);
      EXPECT_NEAR(m[t], d.mean, 1e-12 * std::max(1.0, std::fabs(d.mean))) << "a=" << a;
      EXPECT_NEAR(vb[t], d.biased_var, 1e-11 * std::max(1.0, d.biased_var)) << "a=" << a;
      if (t == 0) {
        EXPECT_TRUE(std::isnan(v[t]));
      } else {
        EXPECT_NEAR(v[t], d.var, 1e-10 * std::max(1.0, d.var)) << "a=" << a << " t=" << t;
      }
    }
  }
}

TEST(EwmVar, ConstantSeriesHasZeroVariance) {
  const auto v = var_of({2.5, 2.5, 2.5, 2.5}, {0.3});
  qkern_test::expect_near_nan(v, {kNaN, 0, 0, 0}, 0);
}

TEST(EwmVar, InfinityMakesVarianceNaN) {
  const auto v = var_of({1, 2, kInf, 3}, {0.5, true});
  EXPECT_FALSE(std::isnan(v[1]));
  EXPECT_TRUE(std::isnan(v[2]));
  EXPECT_TRUE(std::isnan(v[3]));
}

TEST(EwmStd, IsSqrtOfVar) {
  const auto x = qkern_test::gaussian(50, 43);
  const auto v = var_of(x, {0.2});
  std::vector<double> s(x.size());
  qkern::ewm_std(in(x), out(s), {0.2});
  for (std::size_t i = 1; i < x.size(); ++i) {
    EXPECT_EQ(s[i], std::sqrt(v[i]));
  }
}

TEST(Ewm, StreamingEqualsBatchAndCounts) {
  std::vector<double> x = qkern_test::gaussian(1000, 44);
  qkern_test::add_nans(x, 0.1, 45);
  const EwmConfig cfg{0.1, false, 3};
  const auto batch = var_of(x, cfg);
  qkern::Ewm st(cfg);
  std::vector<double> stream(x.size());
  std::size_t valid = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    st.push(x[i]);
    valid += std::isnan(x[i]) ? 0 : 1;
    stream[i] = st.var();
    EXPECT_EQ(st.count(), valid);
  }
  qkern_test::expect_identical(batch, stream);
  EXPECT_EQ(st.config().min_periods, 3U);
  EXPECT_EQ(st.stddev(), std::sqrt(st.var()));
}

TEST(Ewm, LengthMismatch) {
  const std::vector<double> x{1, 2};
  std::vector<double> o(1);
  EXPECT_THROW(qkern::ewm_mean(in(x), out(o), {0.5}), std::invalid_argument);
  EXPECT_THROW(qkern::ewm_var(in(x), out(o), {0.5}), std::invalid_argument);
  EXPECT_THROW(qkern::ewm_std(in(x), out(o), {0.5}), std::invalid_argument);
}

}  // namespace
