// Rolling minimum and maximum with monotonic deques of (index, value) pairs (Lemire 2006,
// "Streaming maximum-minimum filter using no more than three comparisons per element").
//
// The max deque holds strictly decreasing values from front to back: a new value first pops
// every back entry that is <= it, so each index is pushed once and popped at most once and
// the total work over n elements is O(n), O(1) amortised per element (FR-04). The front is
// the window's maximum once expired indices (<= t - window) are dropped. The min deque is
// the mirror image. NaN is never inserted (NanPolicy::Skip leaves it out of the valid set);
// infinities are ordered normally.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "qkern/detail.hpp"
#include "qkern/nan_policy.hpp"
#include "qkern/ring_buffer.hpp"
#include "qkern/views.hpp"

namespace qkern {

namespace detail {

// Fixed-capacity double-ended queue of (index, value), sized to the window.
class IndexDeque {
 public:
  struct Entry {
    std::uint64_t index = 0;
    double value = 0.0;
  };

  explicit IndexDeque(std::size_t capacity) : buf_(capacity == 0 ? 1 : capacity) {}

  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] const Entry& front() const noexcept { return buf_[head_]; }
  [[nodiscard]] const Entry& back() const noexcept { return buf_[wrap(head_ + size_ - 1)]; }

  // Precondition: size() < capacity.
  void push_back(Entry e) noexcept {
    buf_[wrap(head_ + size_)] = e;
    ++size_;
  }
  void pop_back() noexcept { --size_; }
  void pop_front() noexcept {
    head_ = wrap(head_ + 1);
    --size_;
  }
  void clear() noexcept { head_ = size_ = 0; }

 private:
  [[nodiscard]] std::size_t wrap(std::size_t i) const noexcept {
    return i >= buf_.size() ? i - buf_.size() : i;
  }

  std::vector<Entry> buf_;
  std::size_t head_ = 0;
  std::size_t size_ = 0;
};

}  // namespace detail

class RollingMinMax {
 public:
  explicit RollingMinMax(const RollingConfig& cfg, bool track_min = true, bool track_max = true)
      : cfg_((validate(cfg), cfg)),
        track_min_(track_min),
        track_max_(track_max),
        ring_(cfg.window),
        min_q_(track_min ? cfg.window : 1),
        max_q_(track_max ? cfg.window : 1) {}

  void push(double x) noexcept {
    if (ring_.full()) {
      if (std::isnan(ring_.pop_front())) {
        --n_nan_;
      } else {
        --n_valid_;
      }
    }
    ring_.push_back(x);
    const std::uint64_t t = next_++;
    // Drop entries that left the window: index <= t - window.
    if (t >= cfg_.window) {
      const std::uint64_t oldest = t - cfg_.window + 1;
      if (track_max_ && !max_q_.empty() && max_q_.front().index < oldest) {
        max_q_.pop_front();
        ++ops_;
      }
      if (track_min_ && !min_q_.empty() && min_q_.front().index < oldest) {
        min_q_.pop_front();
        ++ops_;
      }
    }
    if (std::isnan(x)) {
      ++n_nan_;
      return;
    }
    ++n_valid_;
    if (track_max_) {
      while (!max_q_.empty() && max_q_.back().value <= x) {
        max_q_.pop_back();
        ++ops_;
      }
      max_q_.push_back({t, x});
      ++ops_;
    }
    if (track_min_) {
      while (!min_q_.empty() && min_q_.back().value >= x) {
        min_q_.pop_back();
        ++ops_;
      }
      min_q_.push_back({t, x});
      ++ops_;
    }
  }

  [[nodiscard]] std::size_t count() const noexcept { return n_valid_; }

  [[nodiscard]] bool ready() const noexcept {
    return !(cfg_.nan == NanPolicy::Propagate && n_nan_ > 0) && n_valid_ >= cfg_.min_periods;
  }

  [[nodiscard]] double min() const noexcept {
    return ready() && track_min_ ? min_q_.front().value : kNaN;
  }
  [[nodiscard]] double max() const noexcept {
    return ready() && track_max_ ? max_q_.front().value : kNaN;
  }

  // Deque pushes plus pops since construction: at most 2 per element per deque (FR-04 test).
  [[nodiscard]] std::uint64_t ops() const noexcept { return ops_; }

  void reset() noexcept {
    ring_.clear();
    min_q_.clear();
    max_q_.clear();
    n_valid_ = n_nan_ = 0;
    next_ = 0;
  }

  [[nodiscard]] const RollingConfig& config() const noexcept { return cfg_; }

 private:
  RollingConfig cfg_;
  bool track_min_;
  bool track_max_;
  RingBuffer ring_;
  detail::IndexDeque min_q_;
  detail::IndexDeque max_q_;
  std::size_t n_valid_ = 0;
  std::size_t n_nan_ = 0;
  std::uint64_t next_ = 0;
  std::uint64_t ops_ = 0;
};

template <Real T>
void rolling_min(StridedView<T> x, OutView out, const RollingConfig& cfg) {
  validate(cfg);
  check_same_size(x.size, out.size);
  check_no_nan(x, cfg.nan);
  RollingMinMax state(cfg, /*track_min=*/true, /*track_max=*/false);
  for (std::size_t i = 0; i < x.size; ++i) {
    state.push(x[i]);
    out[i] = state.min();
  }
}

template <Real T>
void rolling_max(StridedView<T> x, OutView out, const RollingConfig& cfg) {
  validate(cfg);
  check_same_size(x.size, out.size);
  check_no_nan(x, cfg.nan);
  RollingMinMax state(cfg, /*track_min=*/false, /*track_max=*/true);
  for (std::size_t i = 0; i < x.size; ++i) {
    state.push(x[i]);
    out[i] = state.max();
  }
}

}  // namespace qkern
