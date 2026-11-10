// Shared helpers for the C++ tests: views over vectors, NaN-aware comparisons, references.
#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <random>
#include <vector>

#include "qkern/detail.hpp"
#include "qkern/views.hpp"

namespace qkern_test {

inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
inline constexpr double kInf = std::numeric_limits<double>::infinity();

inline qkern::StridedView<double> in(const std::vector<double>& v) {
  return qkern::StridedView<double>{v.data(), v.size(), 1};
}

inline qkern::OutView out(std::vector<double>& v) { return qkern::OutView{v.data(), v.size(), 1}; }

// Equal, treating NaN == NaN and comparing infinities exactly.
inline void expect_near_nan(const std::vector<double>& got, const std::vector<double>& want,
                            double tol) {
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (std::isnan(want[i])) {
      EXPECT_TRUE(std::isnan(got[i])) << "index " << i << " got " << got[i];
    } else if (std::isinf(want[i])) {
      EXPECT_EQ(got[i], want[i]) << "index " << i;
    } else {
      EXPECT_NEAR(got[i], want[i], tol * std::max(1.0, std::fabs(want[i]))) << "index " << i;
    }
  }
}

// Bit-for-bit equality (NaN matches NaN).
inline void expect_identical(const std::vector<double>& a, const std::vector<double>& b) {
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::isnan(a[i])) {
      EXPECT_TRUE(std::isnan(b[i])) << "index " << i;
    } else {
      EXPECT_EQ(a[i], b[i]) << "index " << i;
    }
  }
}

// Gaussian draws by Box-Muller on std::mt19937_64, whose output the standard fixes, so the
// test data is the same with every standard library (std::normal_distribution is not).
inline std::vector<double> gaussian(std::size_t n, unsigned seed, double loc = 0.0,
                                    double scale = 1.0) {
  std::mt19937_64 rng(seed);
  const auto uniform = [&rng] {
    // 53 random bits in (0, 1].
    return (static_cast<double>(rng() >> 11U) + 1.0) * 0x1.0p-53;
  };
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; i += 2) {
    const double r = std::sqrt(-2.0 * std::log(uniform()));
    const double theta = 2.0 * std::numbers::pi * uniform();
    v[i] = loc + scale * r * std::cos(theta);
    if (i + 1 < n) {
      v[i + 1] = loc + scale * r * std::sin(theta);
    }
  }
  return v;
}

// Sprinkle NaN into a series with the given probability.
inline void add_nans(std::vector<double>& v, double prob, unsigned seed) {
  std::mt19937_64 rng(seed);
  for (auto& x : v) {
    if (static_cast<double>(rng() >> 11U) * 0x1.0p-53 < prob) {
      x = kNaN;
    }
  }
}

// Valid values among positions max(0, t - w + 1) .. t.
inline std::vector<double> window_values(const std::vector<double>& x, std::size_t t,
                                         std::size_t w) {
  std::vector<double> vals;
  const std::size_t start = t + 1 >= w ? t + 1 - w : 0;
  for (std::size_t j = start; j <= t; ++j) {
    if (!std::isnan(x[j])) {
      vals.push_back(x[j]);
    }
  }
  return vals;
}

// Exact-ish mean and variance by compensated two-pass sums.
struct Exact {
  double mean = kNaN;
  double var = kNaN;
};

inline Exact exact_moments(const std::vector<double>& vals, int ddof) {
  Exact e;
  if (vals.empty()) {
    return e;
  }
  qkern::detail::CompensatedSum s;
  for (double v : vals) {
    s.add(v);
  }
  const auto n = static_cast<double>(vals.size());
  e.mean = s.value() / n;
  if (static_cast<int>(vals.size()) - ddof < 1) {
    return e;
  }
  qkern::detail::CompensatedSum sq;
  for (double v : vals) {
    sq.add((v - e.mean) * (v - e.mean));
  }
  e.var = sq.value() / (n - static_cast<double>(ddof));
  return e;
}

}  // namespace qkern_test
