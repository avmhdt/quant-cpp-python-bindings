// Internal helpers shared by the kernels. Not part of the public C++ API.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace qkern {

// Sliding states recompute their sums exactly over the ring buffer after this many pushes,
// which bounds the rounding drift of add/remove updates (spec section 8).
inline constexpr std::uint64_t kResyncInterval = 65536;

// They also recompute when a co-moment falls below this fraction of its largest value since
// the last recomputation. Sliding updates leave an absolute error of a few ulp of that
// largest value, so after a large observation leaves the window the relative error of a
// much smaller variance would otherwise grow without bound (spec NFR-06 (d)).
inline constexpr double kCollapseRatio = 1e-6;

inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
inline constexpr double kInf = std::numeric_limits<double>::infinity();

// Serialised state of a streaming object (pickling, FR-09): the window contents oldest
// first, plus the integer and real fields of the state machine. The layout is private to
// each class and versioned by the Python layer.
struct State {
  std::vector<double> buffer;
  std::vector<std::uint64_t> ints;
  std::vector<double> reals;
};

namespace detail {

inline void check_state(bool ok) {
  if (!ok) {
    throw std::invalid_argument("invalid or incompatible streaming state");
  }
}

// Length of the current run of identical finite values among the most recent finite pushes.
// When every finite value in a window belongs to the run, the window is constant: its mean
// is exactly that value and its variance exactly 0 (pandas behaves the same way). Without
// this, add/remove rounding leaves a variance of ~1e-30 and z-scores explode instead of
// being NaN.
class SameRun {
 public:
  void push_finite(double x) noexcept {
    if (run_ > 0 && x == value_) {
      ++run_;
    } else {
      value_ = x;
      run_ = 1;
    }
  }
  // An infinity breaks the run: values pushed before it have left the window by the time it
  // has, so counting restarts after it.
  void push_inf() noexcept { run_ = 0; }
  void reset() noexcept { run_ = 0; }

  // True when the last `n_finite` finite pushes were all equal (n_finite >= 1).
  [[nodiscard]] bool covers(std::size_t n_finite) const noexcept {
    return n_finite > 0 && run_ >= n_finite;
  }
  [[nodiscard]] double value() const noexcept { return value_; }
  [[nodiscard]] std::size_t run() const noexcept { return run_; }

  void load(double value, std::size_t run) noexcept {
    value_ = value;
    run_ = run;
  }

 private:
  double value_ = 0.0;
  std::size_t run_ = 0;
};

// Neumaier-compensated sum. Supports removal (add(-x)), so it also serves as an accurate
// sliding sum: its error stays near eps * |sum| instead of eps * (largest value seen).
class CompensatedSum {
 public:
  void add(double x) noexcept {
    const double t = sum_ + x;
    if (std::fabs(sum_) >= std::fabs(x)) {
      comp_ += (sum_ - t) + x;
    } else {
      comp_ += (x - t) + sum_;
    }
    sum_ = t;
  }
  [[nodiscard]] double value() const noexcept { return sum_ + comp_; }
  [[nodiscard]] double raw_sum() const noexcept { return sum_; }
  [[nodiscard]] double raw_comp() const noexcept { return comp_; }
  void load(double sum, double comp) noexcept {
    sum_ = sum;
    comp_ = comp;
  }
  void reset() noexcept { sum_ = comp_ = 0.0; }

 private:
  double sum_ = 0.0;
  double comp_ = 0.0;
};

}  // namespace detail
}  // namespace qkern
