// Rolling quantile and median over a count-based window: two heaps with lazy deletion
// (FR-06). The low max-heap holds the smallest k + 1 valid values of the window and the high
// min-heap the rest, where k = floor(q (m - 1)) for m valid values, so the two heap tops are
// the order statistics k and k + 1 that every interpolation mode needs. A value leaving the
// window is not searched for: it is recorded in a pending-removal table and dropped when it
// reaches a heap top. Each step costs O(log w) amortised. When dead entries make the heaps
// more than twice the window's size, they are rebuilt from the ring buffer.
//
// Interpolation at h = q (m - 1), matching pandas' rolling quantile:
//   linear   v_k + (v_{k+1} - v_k) (h - k)
//   lower    v_k
//   higher   v_{k+1} when h has a fractional part, else v_k
//   nearest  the closer of the two; exact halves round to the even index
//   midpoint (v_k + v_{k+1}) / 2 when h has a fractional part, else v_k
// NaN follows the NanPolicy rules of the other rolling kernels; infinities are ordered
// normally (pandas treats them as missing).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "qkern/detail.hpp"
#include "qkern/nan_policy.hpp"
#include "qkern/ring_buffer.hpp"
#include "qkern/views.hpp"

namespace qkern {

enum class Interpolation : std::uint8_t { Linear, Lower, Higher, Nearest, Midpoint };

[[nodiscard]] inline Interpolation parse_interpolation(std::string_view name) {
  if (name == "linear") {
    return Interpolation::Linear;
  }
  if (name == "lower") {
    return Interpolation::Lower;
  }
  if (name == "higher") {
    return Interpolation::Higher;
  }
  if (name == "nearest") {
    return Interpolation::Nearest;
  }
  if (name == "midpoint") {
    return Interpolation::Midpoint;
  }
  throw std::invalid_argument(
      "interpolation must be 'linear', 'lower', 'higher', 'nearest' or 'midpoint', got '" +
      std::string(name) + "'");
}

inline void validate_quantile(double q) {
  if (!(q >= 0.0 && q <= 1.0)) {
    throw std::invalid_argument("q must satisfy 0 <= q <= 1");
  }
}

class RollingQuantile {
 public:
  RollingQuantile(const RollingConfig& cfg, double q, Interpolation interp)
      : cfg_((validate(cfg), validate_quantile(q), cfg)),
        q_(q),
        interp_(interp),
        ring_(cfg.window) {}

  void push(double x) {
    if (ring_.full()) {
      const double old = ring_.pop_front();
      if (std::isnan(old)) {
        --n_nan_;
      } else {
        erase(old);
      }
    }
    ring_.push_back(x);
    if (std::isnan(x)) {
      ++n_nan_;
    } else {
      insert(x);
    }
    rebalance();
    if (low_.size() + high_.size() > 2 * cfg_.window + 16) {
      rebuild();
    }
  }

  [[nodiscard]] std::size_t count() const noexcept { return n_low_ + n_high_; }

  [[nodiscard]] bool ready() const noexcept {
    return !(cfg_.nan == NanPolicy::Propagate && n_nan_ > 0) && count() >= cfg_.min_periods;
  }

  [[nodiscard]] double value() const noexcept {
    if (!ready() || count() == 0) {
      return kNaN;
    }
    const double h = q_ * static_cast<double>(count() - 1);
    const double k = std::floor(h);
    const double frac = h - k;
    const double lo = low_.front();
    if (frac == 0.0) {
      return lo;
    }
    const double hi = high_.front();
    switch (interp_) {
      case Interpolation::Lower:
        return lo;
      case Interpolation::Higher:
        return hi;
      case Interpolation::Midpoint:
        return lo == hi ? lo : (lo + hi) / 2.0;
      case Interpolation::Nearest:
        if (frac == 0.5) {
          return std::fmod(k, 2.0) == 0.0 ? lo : hi;
        }
        return frac < 0.5 ? lo : hi;
      case Interpolation::Linear:
      default:
        return lo == hi ? lo : lo + (hi - lo) * frac;
    }
  }

  void reset() noexcept {
    ring_.clear();
    low_.clear();
    high_.clear();
    pending_.clear();
    n_low_ = n_high_ = n_nan_ = 0;
  }

  [[nodiscard]] const RollingConfig& config() const noexcept { return cfg_; }
  [[nodiscard]] double quantile() const noexcept { return q_; }
  [[nodiscard]] Interpolation interpolation() const noexcept { return interp_; }

  // The quantile depends only on the window's contents, so the state is just the ring.
  [[nodiscard]] State save() const {
    State st;
    for (std::size_t i = 0; i < ring_.size(); ++i) {
      st.buffer.push_back(ring_[i]);
    }
    return st;
  }

  void load(const State& st) {
    detail::check_state(st.buffer.size() <= cfg_.window && st.ints.empty() && st.reals.empty());
    reset();
    for (double v : st.buffer) {
      ring_.push_back(v);
    }
    rebuild();
  }

 private:
  // -0.0 and 0.0 compare equal; use one key for both.
  [[nodiscard]] static double key(double v) noexcept { return v == 0.0 ? 0.0 : v; }

  // Keeps every live low value <= every live high value; rebalance() then fixes the counts.
  void insert(double x) {
    const bool to_low = n_low_ > 0 ? x <= low_.front() : (n_high_ == 0 || x <= high_.front());
    if (to_low) {
      low_.push_back(x);
      std::push_heap(low_.begin(), low_.end());
      ++n_low_;
    } else {
      high_.push_back(x);
      std::push_heap(high_.begin(), high_.end(), std::greater<>());
      ++n_high_;
    }
  }

  void erase(double v) {
    if (n_low_ > 0 && v <= low_.front()) {
      --n_low_;
    } else {
      --n_high_;
    }
    ++pending_[key(v)];
    prune_low();
    prune_high();
  }

  void prune_low() {
    while (!low_.empty()) {
      auto it = pending_.find(key(low_.front()));
      if (it == pending_.end()) {
        return;
      }
      if (--it->second == 0) {
        pending_.erase(it);
      }
      std::pop_heap(low_.begin(), low_.end());
      low_.pop_back();
    }
  }

  void prune_high() {
    while (!high_.empty()) {
      auto it = pending_.find(key(high_.front()));
      if (it == pending_.end()) {
        return;
      }
      if (--it->second == 0) {
        pending_.erase(it);
      }
      std::pop_heap(high_.begin(), high_.end(), std::greater<>());
      high_.pop_back();
    }
  }

  [[nodiscard]] std::size_t target_low() const noexcept {
    const std::size_t m = count();
    if (m == 0) {
      return 0;
    }
    return static_cast<std::size_t>(std::floor(q_ * static_cast<double>(m - 1))) + 1;
  }

  void rebalance() {
    const std::size_t target = target_low();
    while (n_low_ > target) {
      const double v = low_.front();
      std::pop_heap(low_.begin(), low_.end());
      low_.pop_back();
      --n_low_;
      high_.push_back(v);
      std::push_heap(high_.begin(), high_.end(), std::greater<>());
      ++n_high_;
      prune_low();
    }
    while (n_low_ < target) {
      const double v = high_.front();
      std::pop_heap(high_.begin(), high_.end(), std::greater<>());
      high_.pop_back();
      --n_high_;
      low_.push_back(v);
      std::push_heap(low_.begin(), low_.end());
      ++n_low_;
      prune_high();
    }
  }

  // Rebuild both heaps from the ring (drops every dead entry).
  void rebuild() {
    low_.clear();
    high_.clear();
    pending_.clear();
    n_low_ = n_high_ = n_nan_ = 0;
    std::vector<double> vals;
    for (std::size_t i = 0; i < ring_.size(); ++i) {
      const double v = ring_[i];
      if (std::isnan(v)) {
        ++n_nan_;
      } else {
        vals.push_back(v);
      }
    }
    std::sort(vals.begin(), vals.end());
    const std::size_t m = vals.size();
    const std::size_t t =
        m == 0 ? 0 : static_cast<std::size_t>(std::floor(q_ * static_cast<double>(m - 1))) + 1;
    low_.assign(vals.begin(), vals.begin() + static_cast<std::ptrdiff_t>(t));
    high_.assign(vals.begin() + static_cast<std::ptrdiff_t>(t), vals.end());
    std::make_heap(low_.begin(), low_.end());
    std::make_heap(high_.begin(), high_.end(), std::greater<>());
    n_low_ = low_.size();
    n_high_ = high_.size();
  }

  RollingConfig cfg_;
  double q_;
  Interpolation interp_;
  RingBuffer ring_;
  std::vector<double> low_;   // max-heap
  std::vector<double> high_;  // min-heap
  std::unordered_map<double, std::size_t> pending_;
  std::size_t n_low_ = 0;  // live entries
  std::size_t n_high_ = 0;
  std::size_t n_nan_ = 0;
};

template <Real T>
void rolling_quantile(StridedView<T> x, OutView out, const RollingConfig& cfg, double q,
                      Interpolation interp) {
  validate(cfg);
  validate_quantile(q);
  check_same_size(x.size, out.size);
  check_no_nan(x, cfg.nan);
  RollingQuantile state(cfg, q, interp);
  for (std::size_t i = 0; i < x.size; ++i) {
    state.push(x[i]);
    out[i] = state.value();
  }
}

}  // namespace qkern
