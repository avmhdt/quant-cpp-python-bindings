// Strided, non-owning views over caller memory. The core never sees Python or NumPy:
// the binding layer turns ndarrays (and, later, Arrow buffers) into these views without
// copying.
#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace qkern {

template <class T>
concept Real = std::same_as<T, double> || std::same_as<T, float>;

// A read-only 1-D view. `stride` is in elements and may be negative (x[::-1]).
// `validity` is an optional Arrow-style LSB bitmap; a cleared bit reads as NaN (DQ-06).
template <Real T>
struct StridedView {
  const T* data = nullptr;
  std::size_t size = 0;
  std::ptrdiff_t stride = 1;
  const std::uint8_t* validity = nullptr;
  std::int64_t validity_offset = 0;

  [[nodiscard]] double operator[](std::size_t i) const noexcept {
    if (validity != nullptr) {
      const auto bit = static_cast<std::uint64_t>(validity_offset) + static_cast<std::uint64_t>(i);
      const auto byte = static_cast<unsigned>(validity[bit >> 3U]);
      if (((byte >> (bit & 7U)) & 1U) == 0U) {
        return std::numeric_limits<double>::quiet_NaN();
      }
    }
    return static_cast<double>(data[static_cast<std::ptrdiff_t>(i) * stride]);
  }
};

// A writable 1-D float64 view, stride in elements.
struct OutView {
  double* data = nullptr;
  std::size_t size = 0;
  std::ptrdiff_t stride = 1;

  [[nodiscard]] double& operator[](std::size_t i) const noexcept {
    return data[static_cast<std::ptrdiff_t>(i) * stride];
  }
};

// A writable 1-D int64 view (observation counts), stride in elements.
struct CountOutView {
  std::int64_t* data = nullptr;
  std::size_t size = 0;
  std::ptrdiff_t stride = 1;

  [[nodiscard]] std::int64_t& operator[](std::size_t i) const noexcept {
    return data[static_cast<std::ptrdiff_t>(i) * stride];
  }
};

}  // namespace qkern
