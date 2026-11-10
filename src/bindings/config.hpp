// Argument normalisation shared by the batch functions and the streaming classes, so both
// accept and reject exactly the same arguments.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include "qkern/ewm.hpp"
#include "qkern/nan_policy.hpp"

namespace qkern::bindings {

inline RollingConfig make_rolling_config(std::int64_t window,
                                         std::optional<std::int64_t> min_periods, std::int64_t ddof,
                                         const std::string& nan_policy) {
  if (window < 1) {
    throw std::invalid_argument("window must be >= 1, got " + std::to_string(window));
  }
  const std::int64_t mp = min_periods.value_or(window);
  if (mp < 1 || mp > window) {
    throw std::invalid_argument("min_periods must satisfy 1 <= min_periods <= window, got " +
                                std::to_string(mp));
  }
  if (ddof < 0) {
    throw std::invalid_argument("ddof must be >= 0, got " + std::to_string(ddof));
  }
  RollingConfig cfg{static_cast<std::size_t>(window), static_cast<std::size_t>(mp),
                    static_cast<int>(ddof), parse_nan_policy(nan_policy)};
  validate(cfg);
  return cfg;
}

// Exactly one of alpha, span, halflife (FR-03).
inline double resolve_alpha(std::optional<double> alpha, std::optional<double> span,
                            std::optional<double> halflife) {
  const int given = static_cast<int>(alpha.has_value()) + static_cast<int>(span.has_value()) +
                    static_cast<int>(halflife.has_value());
  if (given != 1) {
    throw std::invalid_argument("pass exactly one of alpha, span or halflife");
  }
  if (span.has_value()) {
    return alpha_from_span(*span);
  }
  if (halflife.has_value()) {
    return alpha_from_halflife(*halflife);
  }
  if (alpha.has_value() && *alpha > 0.0 && *alpha <= 1.0) {
    return *alpha;
  }
  throw std::invalid_argument("alpha must satisfy 0 < alpha <= 1");
}

inline EwmConfig make_ewm_config(double alpha, bool bias, std::int64_t min_periods,
                                 const std::string& nan_policy) {
  if (min_periods < 0) {
    throw std::invalid_argument("min_periods must be >= 0, got " + std::to_string(min_periods));
  }
  EwmConfig cfg{alpha, bias, static_cast<std::size_t>(min_periods), parse_nan_policy(nan_policy)};
  validate(cfg);
  return cfg;
}

inline const char* nan_policy_name(NanPolicy p) noexcept {
  switch (p) {
    case NanPolicy::Propagate:
      return "propagate";
    case NanPolicy::Raise:
      return "raise";
    case NanPolicy::Skip:
    default:
      return "skip";
  }
}

}  // namespace qkern::bindings
