// Rolling mean, variance, standard deviation and z-score over a count-based window, by
// sliding Welford add/remove on a ring buffer with an exact two-pass resync every
// kResyncInterval pushes. The running sums are kept for x - shift, where the shift is the
// first value of an empty window and is moved to the exact mean at every resync. For data
// far from zero (prices near 1e9) the running mean is then O(1) and its rounding error tiny,
// instead of drifting by ulp(1e9) per update; x - shift is exact while x is within a factor
// of two of the shift (Sterbenz). Batch kernels and the streaming class drive this one state
// machine, so both give bit-identical results (FR-08).
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
    if (ring_.full()) {
      remove(ring_.pop_front());
    }
    ring_.push_back(x);
    add(x);
    if (++since_resync_ >= kResyncInterval) {
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
    return run_.covers(n_finite_) ? run_.value() : shift_ + mean_;
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
    return ((x - shift_) - mean_) / s;
  }

  void reset() noexcept {
    ring_.clear();
    n_finite_ = n_posinf_ = n_neginf_ = n_nan_ = 0;
    mean_ = m2_ = shift_ = 0.0;
    run_.reset();
    since_resync_ = 0;
  }

  [[nodiscard]] const RollingConfig& config() const noexcept { return cfg_; }

 private:
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
      ++n_finite_;
      const double xs = x - shift_;
      const double delta = xs - mean_;
      mean_ += delta / static_cast<double>(n_finite_);
      m2_ += delta * (xs - mean_);
      run_.push_finite(x);
    }
  }

  void remove(double x) noexcept {
    if (std::isnan(x)) {
      --n_nan_;
    } else if (std::isinf(x)) {
      --(x > 0 ? n_posinf_ : n_neginf_);
    } else if (n_finite_ == 1) {
      n_finite_ = 0;
      mean_ = m2_ = 0.0;
    } else {
      --n_finite_;
      const double xs = x - shift_;
      const double mean_before = mean_;
      mean_ -= (xs - mean_before) / static_cast<double>(n_finite_);
      m2_ -= (xs - mean_) * (xs - mean_before);
      m2_ = std::max(m2_, 0.0);
    }
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
      mean_ = m2_ = shift_ = 0.0;
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
    mean_ = corr / nd;
    m2_ = std::max(sq.value() - corr * corr / nd, 0.0);
  }

  RollingConfig cfg_;
  RingBuffer ring_;
  std::size_t n_finite_ = 0;
  std::size_t n_posinf_ = 0;
  std::size_t n_neginf_ = 0;
  std::size_t n_nan_ = 0;
  double shift_ = 0.0;  // running sums are for x - shift_
  double mean_ = 0.0;   // mean of x - shift_
  double m2_ = 0.0;
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
