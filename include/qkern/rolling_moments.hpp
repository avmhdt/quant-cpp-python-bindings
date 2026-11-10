// Rolling mean, variance, standard deviation and z-score over a count-based window.
//
// Algorithm: sliding Welford add/remove on a ring buffer, with three accuracy measures.
//   * Shifted data: the sums are kept for x - shift, where the shift is the first value of
//     an empty window and moves to the exact mean at every resync. For data far from zero
//     (prices near 1e9) the running mean is then O(1); x - shift is exact while x is within a
//     factor of two of the shift (Sterbenz).
//   * Compensated mean: the mean comes from a Neumaier sliding sum, so its error stays near
//     eps * |mean| however large the values that already left the window were.
//   * Exact resync: a corrected two-pass recomputation over the ring every kResyncInterval
//     pushes, and as soon as M2 falls below kCollapseRatio of its peak since the last
//     resync (after a large value leaves the window).
// Batch kernels and the streaming class drive this one state machine, so both give
// bit-identical results (FR-08).
//
// Semantics (spec DQ-02 to DQ-04):
//   * NaN is never a valid observation. NanPolicy::Skip leaves it out of the window's valid
//     set; NanPolicy::Propagate makes every output NaN while a NaN is inside the window;
//     NanPolicy::Raise is checked by the batch drivers before computing.
//   * +inf and -inf are valid observations, counted apart from the finite values so the
//     running sums are never polluted after an infinity leaves the window. The mean is
//     +inf or -inf while only one sign is present and NaN when both are; variance, standard
//     deviation and z-score are NaN while any infinity is in the window.
//   * Outputs are NaN while the number of valid observations is below min_periods. The
//     variance also needs n_valid - ddof >= 1.
//   * A window whose finite values are all equal has mean exactly that value and variance
//     exactly 0, so z-scores there are NaN (std = 0) rather than noise.
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

class RollingMoments {
 public:
  explicit RollingMoments(const RollingConfig& cfg)
      : cfg_(cfg), ring_((validate(cfg), cfg.window)) {}

  void push(double x) noexcept {
    bool collapsed = false;
    if (ring_.full()) {
      collapsed = remove(ring_.pop_front());
    }
    ring_.push_back(x);
    add(x);
    if (++since_resync_ >= kResyncInterval || collapsed) {
      resync();
    }
  }

  // Valid (non-NaN) observations in the window, infinities included.
  [[nodiscard]] std::size_t count() const noexcept { return n_finite_ + n_posinf_ + n_neginf_; }

  // True when count() >= min_periods and no NaN blocks the output under Propagate.
  [[nodiscard]] bool ready() const noexcept {
    return !(cfg_.nan == NanPolicy::Propagate && n_nan_ > 0) && count() >= cfg_.min_periods;
  }

  [[nodiscard]] double mean() const noexcept {
    if (!ready() || (n_posinf_ > 0 && n_neginf_ > 0)) {
      return kNaN;
    }
    if (n_posinf_ > 0) {
      return kInf;
    }
    if (n_neginf_ > 0) {
      return -kInf;
    }
    return run_.covers(n_finite_) ? run_.value() : shift_ + shifted_mean();
  }

  [[nodiscard]] double var() const noexcept {
    if (!ready() || n_posinf_ > 0 || n_neginf_ > 0) {
      return kNaN;
    }
    const auto ddof = static_cast<std::size_t>(cfg_.ddof);
    if (n_finite_ <= ddof) {
      return kNaN;
    }
    if (run_.covers(n_finite_)) {
      return 0.0;
    }
    return std::max(m2_, 0.0) / static_cast<double>(n_finite_ - ddof);
  }

  [[nodiscard]] double stddev() const noexcept { return std::sqrt(var()); }

  // z-score of x against the current window (x is not pushed). NaN when x is NaN, when the
  // window's mean or standard deviation is NaN, or when the standard deviation is 0.
  [[nodiscard]] double zscore(double x) const noexcept {
    if (std::isnan(x)) {
      return kNaN;
    }
    const double s = stddev();
    if (std::isnan(s) || s == 0.0) {
      return kNaN;
    }
    // s is finite and non-zero, so the window holds no infinity and is not constant.
    return ((x - shift_) - shifted_mean()) / s;
  }

  void reset() noexcept {
    ring_.clear();
    n_finite_ = n_posinf_ = n_neginf_ = n_nan_ = 0;
    sum_.reset();
    m2_ = m2_peak_ = shift_ = 0.0;
    run_.reset();
    since_resync_ = 0;
  }

  [[nodiscard]] const RollingConfig& config() const noexcept { return cfg_; }

  [[nodiscard]] State save() const {
    State st;
    st.buffer.reserve(ring_.size());
    for (std::size_t i = 0; i < ring_.size(); ++i) {
      st.buffer.push_back(ring_[i]);
    }
    st.ints = {n_finite_, n_posinf_, n_neginf_, n_nan_, run_.run(), since_resync_};
    st.reals = {shift_, sum_.raw_sum(), sum_.raw_comp(), m2_, m2_peak_, run_.value()};
    return st;
  }

  void load(const State& st) {
    detail::check_state(st.buffer.size() <= cfg_.window && st.ints.size() == 6 &&
                        st.reals.size() == 6 &&
                        st.ints[0] + st.ints[1] + st.ints[2] + st.ints[3] == st.buffer.size());
    reset();
    for (double v : st.buffer) {
      ring_.push_back(v);
    }
    n_finite_ = static_cast<std::size_t>(st.ints[0]);
    n_posinf_ = static_cast<std::size_t>(st.ints[1]);
    n_neginf_ = static_cast<std::size_t>(st.ints[2]);
    n_nan_ = static_cast<std::size_t>(st.ints[3]);
    run_.load(st.reals[5], static_cast<std::size_t>(st.ints[4]));
    since_resync_ = st.ints[5];
    shift_ = st.reals[0];
    sum_.load(st.reals[1], st.reals[2]);
    m2_ = st.reals[3];
    m2_peak_ = st.reals[4];
  }

 private:
  // Mean of x - shift_ over the finite values, from the compensated sliding sum.
  [[nodiscard]] double shifted_mean() const noexcept {
    return n_finite_ == 0 ? 0.0 : sum_.value() / static_cast<double>(n_finite_);
  }

  void add(double x) noexcept {
    if (std::isnan(x)) {
      ++n_nan_;
    } else if (std::isinf(x)) {
      ++(x > 0 ? n_posinf_ : n_neginf_);
      run_.push_inf();
    } else {
      if (n_finite_ == 0) {
        shift_ = x;
      }
      const double xs = x - shift_;
      const double mean_before = shifted_mean();
      sum_.add(xs);
      ++n_finite_;
      m2_ += (xs - mean_before) * (xs - shifted_mean());
      m2_peak_ = std::max(m2_peak_, m2_);
      run_.push_finite(x);
    }
  }

  // Returns true when the sums lost most of their magnitude (see kCollapseRatio).
  bool remove(double x) noexcept {
    if (std::isnan(x)) {
      --n_nan_;
    } else if (std::isinf(x)) {
      --(x > 0 ? n_posinf_ : n_neginf_);
    } else if (n_finite_ == 1) {
      n_finite_ = 0;
      sum_.reset();
      m2_ = m2_peak_ = 0.0;
    } else {
      const double xs = x - shift_;
      const double mean_before = shifted_mean();
      sum_.add(-xs);
      --n_finite_;
      m2_ = std::max(m2_ - (xs - shifted_mean()) * (xs - mean_before), 0.0);
      return m2_ < kCollapseRatio * m2_peak_;
    }
    return false;
  }

  // Exact recomputation over the ring (corrected two-pass, Chan, Golub and LeVeque 1983),
  // re-centred on the exact mean.
  void resync() noexcept {
    since_resync_ = 0;
    detail::CompensatedSum sum;
    std::size_t n = 0;
    for (std::size_t i = 0; i < ring_.size(); ++i) {
      const double v = ring_[i];
      if (std::isfinite(v)) {
        sum.add(v);
        ++n;
      }
    }
    if (n == 0) {
      sum_.reset();
      m2_ = m2_peak_ = shift_ = 0.0;
      return;
    }
    const auto nd = static_cast<double>(n);
    shift_ = sum.value() / nd;
    detail::CompensatedSum sq;
    detail::CompensatedSum dev;
    for (std::size_t i = 0; i < ring_.size(); ++i) {
      const double v = ring_[i];
      if (std::isfinite(v)) {
        const double d = v - shift_;
        sq.add(d * d);
        dev.add(d);
      }
    }
    const double corr = dev.value();
    sum_.load(corr, 0.0);
    m2_ = std::max(sq.value() - corr * corr / nd, 0.0);
    m2_peak_ = m2_;
  }

  RollingConfig cfg_;
  RingBuffer ring_;
  std::size_t n_finite_ = 0;
  std::size_t n_posinf_ = 0;
  std::size_t n_neginf_ = 0;
  std::size_t n_nan_ = 0;
  double shift_ = 0.0;          // the sums are for x - shift_
  detail::CompensatedSum sum_;  // sum of x - shift_ over the finite values
  double m2_ = 0.0;             // sum of squared deviations
  double m2_peak_ = 0.0;        // largest m2_ since the last resync
  detail::SameRun run_;
  std::uint64_t since_resync_ = 0;
};

// ---------------------------------------------------------------------------------------
// Batch drivers
// ---------------------------------------------------------------------------------------

template <Real T>
void rolling_mean(StridedView<T> x, OutView out, const RollingConfig& cfg) {
  validate(cfg);
  check_same_size(x.size, out.size);
  check_no_nan(x, cfg.nan);
  RollingMoments state(cfg);
  for (std::size_t i = 0; i < x.size; ++i) {
    state.push(x[i]);
    out[i] = state.mean();
  }
}

template <Real T>
void rolling_var(StridedView<T> x, OutView out, const RollingConfig& cfg) {
  validate(cfg);
  check_same_size(x.size, out.size);
  check_no_nan(x, cfg.nan);
  RollingMoments state(cfg);
  for (std::size_t i = 0; i < x.size; ++i) {
    state.push(x[i]);
    out[i] = state.var();
  }
}

template <Real T>
void rolling_std(StridedView<T> x, OutView out, const RollingConfig& cfg) {
  validate(cfg);
  check_same_size(x.size, out.size);
  check_no_nan(x, cfg.nan);
  RollingMoments state(cfg);
  for (std::size_t i = 0; i < x.size; ++i) {
    state.push(x[i]);
    out[i] = state.stddev();
  }
}

// Mean, variance and valid count in one pass (qkern.rolling_moments).
template <Real T>
void rolling_moments(StridedView<T> x, OutView mean, OutView var, CountOutView count,
                     const RollingConfig& cfg) {
  validate(cfg);
  check_same_size(x.size, mean.size);
  check_same_size(x.size, var.size);
  check_same_size(x.size, count.size);
  check_no_nan(x, cfg.nan);
  RollingMoments state(cfg);
  for (std::size_t i = 0; i < x.size; ++i) {
    state.push(x[i]);
    mean[i] = state.mean();
    var[i] = state.var();
    count[i] = static_cast<std::int64_t>(state.count());
  }
}

// z_t = (x_t - mean) / std. With include_current the window is W_t (x_t included); without
// it the statistics come from W_{t-1}, so z_t never uses x_t itself (DQ-01).
template <Real T>
void rolling_zscore(StridedView<T> x, OutView out, const RollingConfig& cfg, bool include_current) {
  validate(cfg);
  check_same_size(x.size, out.size);
  check_no_nan(x, cfg.nan);
  RollingMoments state(cfg);
  for (std::size_t i = 0; i < x.size; ++i) {
    const double v = x[i];
    if (include_current) {
      state.push(v);
      out[i] = state.zscore(v);
    } else {
      out[i] = state.zscore(v);
      state.push(v);
    }
  }
}

}  // namespace qkern
