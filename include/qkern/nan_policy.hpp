// NaN policy, kernel configuration structs and their validation (DQ-02, DQ-03).
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

#include "qkern/views.hpp"

namespace qkern {

enum class NanPolicy : std::uint8_t { Skip, Propagate, Raise };

[[nodiscard]] inline NanPolicy parse_nan_policy(std::string_view name) {
  if (name == "skip") {
    return NanPolicy::Skip;
  }
  if (name == "propagate") {
    return NanPolicy::Propagate;
  }
  if (name == "raise") {
    return NanPolicy::Raise;
  }
  throw std::invalid_argument("nan_policy must be 'skip', 'propagate' or 'raise', got '" +
                              std::string(name) + "'");
}

struct RollingConfig {
  std::size_t window = 1;
  std::size_t min_periods = 1;
  int ddof = 1;
  NanPolicy nan = NanPolicy::Skip;
};

inline void validate(const RollingConfig& cfg) {
  if (cfg.window < 1) {
    throw std::invalid_argument("window must be >= 1");
  }
  if (cfg.min_periods < 1 || cfg.min_periods > cfg.window) {
    throw std::invalid_argument("min_periods must satisfy 1 <= min_periods <= window");
  }
  if (cfg.ddof < 0) {
    throw std::invalid_argument("ddof must be >= 0");
  }
}

// nan_policy="raise" is checked over the whole input before any output is written.
template <Real T>
void check_no_nan(StridedView<T> x, NanPolicy nan) {
  if (nan != NanPolicy::Raise) {
    return;
  }
  for (std::size_t i = 0; i < x.size; ++i) {
    if (std::isnan(x[i])) {
      throw std::invalid_argument("input contains NaN and nan_policy='raise'");
    }
  }
}

inline void check_same_size(std::size_t n_in, std::size_t n_out) {
  if (n_in != n_out) {
    throw std::invalid_argument("output length " + std::to_string(n_out) +
                                " does not match input length " + std::to_string(n_in));
  }
}

}  // namespace qkern
