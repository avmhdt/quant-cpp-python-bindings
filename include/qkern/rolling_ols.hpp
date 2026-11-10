// Rolling simple regression of y on x with an intercept, by bivariate sliding Welford
// (co-moments Sxx, Syy, Sxy) with the same accuracy measures as the univariate moments:
// shifted data, compensated sliding sums for the means, and an exact resync every
// kResyncInterval pushes or when Sxx or Syy collapses below kCollapseRatio of its peak.
//
//   beta = Sxy / Sxx,  alpha = mean_y - beta * mean_x,  r2 = Sxy^2 / (Sxx * Syy),
//   resid_t = y_t - alpha - beta * x_t
//
// Pairs are pairwise complete: a pair with a NaN on either side is not a valid observation.
// A pair holding an infinity is valid but makes every output NaN while it is in the window
// (DQ-04). Outputs need at least min_periods valid pairs, at least 2 finite pairs and
// Sxx > 0. A window whose x values are all equal has Sxx exactly 0 (outputs NaN); one whose
// y values are all equal has Syy and Sxy exactly 0 (beta 0, r2 NaN).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "qkern/detail.hpp"
#include "qkern/nan_policy.hpp"
#include "qkern/ring_buffer.hpp"
#include "qkern/views.hpp"

namespace qkern {

class RollingOls {
 public:
  explicit RollingOls(const RollingConfig& cfg)
      : cfg_((validate(cfg), cfg)), ring_y_(cfg.window), ring_x_(cfg.window) {}

  void push(double y, double x) noexcept {
    bool collapsed = false;
    if (ring_y_.full()) {
      collapsed = remove(ring_y_.pop_front(), ring_x_.pop_front());
    }
    ring_y_.push_back(y);
    ring_x_.push_back(x);
    add(y, x);
    if (++since_resync_ >= kResyncInterval || collapsed) {
      resync();
    }
  }

  // Valid pairs (no NaN on either side) in the window, infinities included.
  [[nodiscard]] std::size_t count() const noexcept { return n_finite_ + n_inf_; }

  [[nodiscard]] bool ready() const noexcept {
    return !(cfg_.nan == NanPolicy::Propagate && n_nan_ > 0) && count() >= cfg_.min_periods &&
           n_inf_ == 0 && n_finite_ >= 2 && sxx() > 0.0;
  }

  [[nodiscard]] double beta() const noexcept { return ready() ? sxy() / sxx() : kNaN; }

  [[nodiscard]] double alpha() const noexcept {
    return ready() ? mean_y() - (sxy() / sxx()) * mean_x() : kNaN;
  }

  [[nodiscard]] double r2() const noexcept {
    if (!ready()) {
      return kNaN;
    }
    const double syy_v = syy();
    if (!(syy_v > 0.0)) {
      return kNaN;
    }
    const double s = sxy();
    return std::min((s * s) / (sxx() * syy_v), 1.0);
  }

  // Residual of a pair against the current fit (the pair is not pushed).
  [[nodiscard]] double resid(double y, double x) const noexcept {
    if (std::isnan(y) || std::isnan(x) || !ready()) {
      return kNaN;
    }
    if (y_constant()) {
      return y - run_y_.value();  // beta is exactly 0
    }
    // Centred form (y - mean_y) - beta (x - mean_x), computed in the shifted frame.
    const double b = sxy() / sxx();
    return ((y - shift_y_) - shifted_mean_y()) - b * ((x - shift_x_) - shifted_mean_x());
  }

  void reset() noexcept {
    ring_y_.clear();
    ring_x_.clear();
    n_finite_ = n_inf_ = n_nan_ = 0;
    sum_x_.reset();
    sum_y_.reset();
    sxx_ = syy_ = sxy_ = sxx_peak_ = syy_peak_ = shift_x_ = shift_y_ = 0.0;
    run_x_.reset();
    run_y_.reset();
    since_resync_ = 0;
  }

  [[nodiscard]] const RollingConfig& config() const noexcept { return cfg_; }

  [[nodiscard]] State save() const {
    State st;
    st.buffer.reserve(2 * ring_y_.size());
    for (std::size_t i = 0; i < ring_y_.size(); ++i) {
      st.buffer.push_back(ring_y_[i]);
    }
    for (std::size_t i = 0; i < ring_x_.size(); ++i) {
      st.buffer.push_back(ring_x_[i]);
    }
    st.ints = {n_finite_, n_inf_, n_nan_, run_x_.run(), run_y_.run(), since_resync_};
    st.reals = {shift_x_,
                shift_y_,
                sum_x_.raw_sum(),
                sum_x_.raw_comp(),
                sum_y_.raw_sum(),
                sum_y_.raw_comp(),
                sxx_,
                syy_,
                sxy_,
                sxx_peak_,
                syy_peak_,
                run_x_.value(),
                run_y_.value()};
    return st;
  }

  void load(const State& st) {
    const std::size_t n = st.buffer.size() / 2;
    detail::check_state(st.buffer.size() % 2 == 0 && n <= cfg_.window && st.ints.size() == 6 &&
                        st.reals.size() == 13 && st.ints[0] + st.ints[1] + st.ints[2] == n);
    reset();
    for (std::size_t i = 0; i < n; ++i) {
      ring_y_.push_back(st.buffer[i]);
      ring_x_.push_back(st.buffer[n + i]);
    }
    n_finite_ = static_cast<std::size_t>(st.ints[0]);
    n_inf_ = static_cast<std::size_t>(st.ints[1]);
    n_nan_ = static_cast<std::size_t>(st.ints[2]);
    run_x_.load(st.reals[11], static_cast<std::size_t>(st.ints[3]));
    run_y_.load(st.reals[12], static_cast<std::size_t>(st.ints[4]));
    since_resync_ = st.ints[5];
    shift_x_ = st.reals[0];
    shift_y_ = st.reals[1];
    sum_x_.load(st.reals[2], st.reals[3]);
    sum_y_.load(st.reals[4], st.reals[5]);
    sxx_ = st.reals[6];
    syy_ = st.reals[7];
    sxy_ = st.reals[8];
    sxx_peak_ = st.reals[9];
    syy_peak_ = st.reals[10];
  }

 private:
  [[nodiscard]] bool x_constant() const noexcept { return run_x_.covers(n_finite_); }
  [[nodiscard]] bool y_constant() const noexcept { return run_y_.covers(n_finite_); }
  [[nodiscard]] double shifted_mean_x() const noexcept {
    return n_finite_ == 0 ? 0.0 : sum_x_.value() / static_cast<double>(n_finite_);
  }
  [[nodiscard]] double shifted_mean_y() const noexcept {
    return n_finite_ == 0 ? 0.0 : sum_y_.value() / static_cast<double>(n_finite_);
  }
  [[nodiscard]] double mean_x() const noexcept {
    return x_constant() ? run_x_.value() : shift_x_ + shifted_mean_x();
  }
  [[nodiscard]] double mean_y() const noexcept {
    return y_constant() ? run_y_.value() : shift_y_ + shifted_mean_y();
  }
  [[nodiscard]] double sxx() const noexcept { return x_constant() ? 0.0 : std::max(sxx_, 0.0); }
  [[nodiscard]] double syy() const noexcept { return y_constant() ? 0.0 : std::max(syy_, 0.0); }
  [[nodiscard]] double sxy() const noexcept { return (x_constant() || y_constant()) ? 0.0 : sxy_; }

  void add(double y, double x) noexcept {
    if (std::isnan(y) || std::isnan(x)) {
      ++n_nan_;
      return;
    }
    if (std::isinf(y) || std::isinf(x)) {
      ++n_inf_;
      run_x_.push_inf();
      run_y_.push_inf();
      return;
    }
    if (n_finite_ == 0) {
      shift_x_ = x;
      shift_y_ = y;
    }
    const double xs = x - shift_x_;
    const double ys = y - shift_y_;
    const double mx_before = shifted_mean_x();
    const double my_before = shifted_mean_y();
    sum_x_.add(xs);
    sum_y_.add(ys);
    ++n_finite_;
    const double mx = shifted_mean_x();
    const double my = shifted_mean_y();
    sxx_ += (xs - mx_before) * (xs - mx);
    syy_ += (ys - my_before) * (ys - my);
    sxy_ += (xs - mx_before) * (ys - my);
    sxx_peak_ = std::max(sxx_peak_, sxx_);
    syy_peak_ = std::max(syy_peak_, syy_);
    run_x_.push_finite(x);
    run_y_.push_finite(y);
  }

  // Returns true when Sxx or Syy lost most of its magnitude (see kCollapseRatio).
  bool remove(double y, double x) noexcept {
    if (std::isnan(y) || std::isnan(x)) {
      --n_nan_;
      return false;
    }
    if (std::isinf(y) || std::isinf(x)) {
      --n_inf_;
      return false;
    }
    if (n_finite_ == 1) {
      n_finite_ = 0;
      sum_x_.reset();
      sum_y_.reset();
      sxx_ = syy_ = sxy_ = sxx_peak_ = syy_peak_ = 0.0;
      return false;
    }
    const double xs = x - shift_x_;
    const double ys = y - shift_y_;
    const double mx_before = shifted_mean_x();
    const double my_before = shifted_mean_y();
    sum_x_.add(-xs);
    sum_y_.add(-ys);
    --n_finite_;
    const double mx = shifted_mean_x();
    sxx_ = std::max(sxx_ - (xs - mx) * (xs - mx_before), 0.0);
    syy_ = std::max(syy_ - (ys - shifted_mean_y()) * (ys - my_before), 0.0);
    sxy_ -= (xs - mx) * (ys - my_before);
    return sxx_ < kCollapseRatio * sxx_peak_ || syy_ < kCollapseRatio * syy_peak_;
  }

  void resync() noexcept {
    since_resync_ = 0;
    detail::CompensatedSum sum_x;
    detail::CompensatedSum sum_y;
    std::size_t n = 0;
    for (std::size_t i = 0; i < ring_y_.size(); ++i) {
      const double y = ring_y_[i];
      const double x = ring_x_[i];
      if (std::isfinite(y) && std::isfinite(x)) {
        sum_x.add(x);
        sum_y.add(y);
        ++n;
      }
    }
    if (n == 0) {
      sum_x_.reset();
      sum_y_.reset();
      sxx_ = syy_ = sxy_ = sxx_peak_ = syy_peak_ = shift_x_ = shift_y_ = 0.0;
      return;
    }
    const auto nd = static_cast<double>(n);
    shift_x_ = sum_x.value() / nd;
    shift_y_ = sum_y.value() / nd;
    detail::CompensatedSum cxx;
    detail::CompensatedSum cyy;
    detail::CompensatedSum cxy;
    detail::CompensatedSum dx_sum;
    detail::CompensatedSum dy_sum;
    for (std::size_t i = 0; i < ring_y_.size(); ++i) {
      const double y = ring_y_[i];
      const double x = ring_x_[i];
      if (std::isfinite(y) && std::isfinite(x)) {
        const double dx = x - shift_x_;
        const double dy = y - shift_y_;
        cxx.add(dx * dx);
        cyy.add(dy * dy);
        cxy.add(dx * dy);
        dx_sum.add(dx);
        dy_sum.add(dy);
      }
    }
    const double ex = dx_sum.value();
    const double ey = dy_sum.value();
    sum_x_.load(ex, 0.0);
    sum_y_.load(ey, 0.0);
    sxx_ = std::max(cxx.value() - ex * ex / nd, 0.0);
    syy_ = std::max(cyy.value() - ey * ey / nd, 0.0);
    sxy_ = cxy.value() - ex * ey / nd;
    sxx_peak_ = sxx_;
    syy_peak_ = syy_;
  }

  RollingConfig cfg_;
  RingBuffer ring_y_;
  RingBuffer ring_x_;
  std::size_t n_finite_ = 0;
  std::size_t n_inf_ = 0;
  std::size_t n_nan_ = 0;
  double shift_x_ = 0.0;  // the sums are for x - shift_x_ and y - shift_y_
  double shift_y_ = 0.0;
  detail::CompensatedSum sum_x_;
  detail::CompensatedSum sum_y_;
  double sxx_ = 0.0;
  double syy_ = 0.0;
  double sxy_ = 0.0;
  double sxx_peak_ = 0.0;
  double syy_peak_ = 0.0;
  detail::SameRun run_x_;
  detail::SameRun run_y_;
  std::uint64_t since_resync_ = 0;
};

// Batch driver. With include_current the fit at t uses W_t and resid_t is in-sample; without
// it alpha, beta and r2 at t come from W_{t-1} and resid_t is out-of-sample (DQ-01).
template <Real T, Real U>
void rolling_ols(StridedView<T> y, StridedView<U> x, OutView alpha, OutView beta, OutView resid,
                 OutView r2, const RollingConfig& cfg, bool include_current) {
  validate(cfg);
  check_same_size(y.size, x.size);
  check_same_size(y.size, alpha.size);
  check_same_size(y.size, beta.size);
  check_same_size(y.size, resid.size);
  check_same_size(y.size, r2.size);
  check_no_nan(y, cfg.nan);
  check_no_nan(x, cfg.nan);
  RollingOls state(cfg);
  for (std::size_t i = 0; i < y.size; ++i) {
    const double yi = y[i];
    const double xi = x[i];
    if (include_current) {
      state.push(yi, xi);
    }
    alpha[i] = state.alpha();
    beta[i] = state.beta();
    r2[i] = state.r2();
    resid[i] = state.resid(yi, xi);
    if (!include_current) {
      state.push(yi, xi);
    }
  }
}

}  // namespace qkern
