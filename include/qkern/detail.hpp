// Internal helpers shared by the kernels. Not part of the public C++ API.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace qkern {

// Sliding states recompute their sums exactly over the ring buffer after this many pushes,
// which bounds the rounding drift of add/remove updates (spec section 8).
inline constexpr std::uint64_t kResyncInterval = 65536;

inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
inline constexpr double kInf = std::numeric_limits<double>::infinity();

namespace detail {

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

// Neumaier-compensated sum, used by the exact resync passes and the tests' references.
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

 private:
  double sum_ = 0.0;
  double comp_ = 0.0;
};

}  // namespace detail
}  // namespace qkern
