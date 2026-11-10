// Rolling moments over a count-based window: sliding Welford add/remove on a ring buffer.
//
// M1 scope: the state machine and the rolling mean. Variance, z-score and the periodic
// exact resync land in M2 on the same class, so batch and streaming share one
// implementation (FR-08).
//
// Semantics (spec DQ-02 to DQ-04):
//   * NaN is never a valid observation. With NanPolicy::Skip it is excluded from the
//     window's valid set; with NanPolicy::Propagate any NaN inside the window makes the
//     output NaN. NanPolicy::Raise is checked by the batch driver before computing.
//   * +inf and -inf are valid observations, counted apart from the finite values so the
//     running sums are never polluted after an infinity leaves the window. The mean is
//     +inf or -inf while only one sign is present, NaN when both are.
//   * The output is NaN while the number of valid observations is below min_periods.
#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

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
  }

  // Number of valid (non-NaN) observations in the window, infinities included.
  [[nodiscard]] std::size_t count() const noexcept { return n_finite_ + n_posinf_ + n_neginf_; }

  [[nodiscard]] double mean() const noexcept {
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    if (cfg_.nan == NanPolicy::Propagate && n_nan_ > 0) {
      return kNaN;
    }
    if (count() < cfg_.min_periods) {
      return kNaN;
    }
    if (n_posinf_ > 0 && n_neginf_ > 0) {
      return kNaN;
    }
    if (n_posinf_ > 0) {
      return std::numeric_limits<double>::infinity();
    }
    if (n_neginf_ > 0) {
      return -std::numeric_limits<double>::infinity();
    }
    return mean_;
  }

  void reset() noexcept {
    ring_.clear();
    n_finite_ = n_posinf_ = n_neginf_ = n_nan_ = 0;
    mean_ = 0.0;
  }

  [[nodiscard]] const RollingConfig& config() const noexcept { return cfg_; }

 private:
  void add(double x) noexcept {
    if (std::isnan(x)) {
      ++n_nan_;
    } else if (std::isinf(x)) {
      ++(x > 0 ? n_posinf_ : n_neginf_);
    } else {
      ++n_finite_;
      mean_ += (x - mean_) / static_cast<double>(n_finite_);
    }
  }

  void remove(double x) noexcept {
    if (std::isnan(x)) {
      --n_nan_;
    } else if (std::isinf(x)) {
      --(x > 0 ? n_posinf_ : n_neginf_);
    } else if (n_finite_ == 1) {
      n_finite_ = 0;
      mean_ = 0.0;
    } else {
      --n_finite_;
      mean_ -= (x - mean_) / static_cast<double>(n_finite_);
    }
  }

  RollingConfig cfg_;
  RingBuffer ring_;
  std::size_t n_finite_ = 0;
  std::size_t n_posinf_ = 0;
  std::size_t n_neginf_ = 0;
  std::size_t n_nan_ = 0;
  double mean_ = 0.0;
};

// Batch driver: feeds the same state machine as the streaming class.
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

}  // namespace qkern
