// Binding-layer helpers: ndarray -> view without copying, C++-allocated outputs handed to
// NumPy through a capsule (no copy on return), out= validation, the GIL-release threshold
// and the debug copy counters (FR-10, FR-11, FR-14, NFR-04).
//
// Rule: every Python object is touched before the GIL is released. Code inside
// release_gil_if() sees only raw pointers and views.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "qkern/views.hpp"

namespace qkern::bindings {

namespace nb = nanobind;

// Read-only 1-D input of any stride (positive or negative), float64 or float32.
template <Real T>
using In1D = nb::ndarray<const T, nb::ndim<1>, nb::device::cpu>;

// Writable 1-D float64 out= target of any stride.
using Out1D = nb::ndarray<double, nb::ndim<1>, nb::device::cpu>;

// NumPy array returned to Python; owns a C++ buffer through a capsule.
using NumpyF64 = nb::ndarray<nb::numpy, double, nb::ndim<1>>;

// Batch calls release the GIL when they touch at least this many elements (FR-14).
inline constexpr std::size_t kGilReleaseMinN = 1024;

inline std::atomic<std::size_t>& gil_release_min_n() noexcept {
  static std::atomic<std::size_t> value{kGilReleaseMinN};
  return value;
}

struct DebugCounters {
  std::atomic<std::uint64_t> input_conversions{0};
  std::atomic<std::uint64_t> output_copies{0};
};

inline DebugCounters& debug_counters() noexcept {
  static DebugCounters counters;
  return counters;
}

template <class F>
void release_gil_if(std::size_t n_elements, F&& body) {
  if (n_elements >= gil_release_min_n().load(std::memory_order_relaxed)) {
    const nb::gil_scoped_release release;
    std::forward<F>(body)();
  } else {
    std::forward<F>(body)();
  }
}

template <Real T>
[[nodiscard]] StridedView<T> view_of(const In1D<T>& a) noexcept {
  return StridedView<T>{a.data(), a.shape(0), static_cast<std::ptrdiff_t>(a.stride(0))};
}

[[nodiscard]] inline OutView view_of(const Out1D& a) noexcept {
  return OutView{a.data(), a.shape(0), static_cast<std::ptrdiff_t>(a.stride(0))};
}

// A float64 buffer allocated before the GIL is released and handed to NumPy afterwards.
class OutBuffer {
 public:
  explicit OutBuffer(std::size_t n)
      : n_(n), data_(std::make_unique<double[]>(n == 0 ? 1 : n)) {}  // NOLINT(*-avoid-c-arrays)

  [[nodiscard]] OutView view() const noexcept { return OutView{data_.get(), n_, 1}; }

  // Transfers ownership to a capsule; the returned array's base is that capsule.
  [[nodiscard]] NumpyF64 release_to_numpy() {
    double* raw = data_.release();
    const nb::capsule owner(raw, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    const std::size_t shape[1] = {n_};  // NOLINT(*-avoid-c-arrays)
    return {raw, 1, shape, owner};
  }

 private:
  std::size_t n_;
  std::unique_ptr<double[]> data_;  // NOLINT(*-avoid-c-arrays)
};

// Byte range [lo, hi) spanned by a strided 1-D array.
struct ByteRange {
  std::uintptr_t lo = 0;
  std::uintptr_t hi = 0;
};

[[nodiscard]] inline ByteRange byte_range(const void* data, std::size_t n, std::ptrdiff_t stride,
                                          std::size_t itemsize) noexcept {
  if (n == 0) {
    return {};
  }
  const auto base = reinterpret_cast<std::uintptr_t>(data);  // NOLINT(*-reinterpret-cast)
  const auto span =
      static_cast<std::ptrdiff_t>(n - 1) * stride * static_cast<std::ptrdiff_t>(itemsize);
  const auto first = span < 0 ? base - static_cast<std::uintptr_t>(-span) : base;
  const auto last = span < 0 ? base : base + static_cast<std::uintptr_t>(span);
  return {first, last + itemsize};
}

[[nodiscard]] inline bool overlaps(ByteRange a, ByteRange b) noexcept {
  return a.lo < a.hi && b.lo < b.hi && a.lo < b.hi && b.lo < a.hi;
}

// Validates out= against an input (DQ-11): same length, no memory overlap.
template <Real T>
void check_out(const In1D<T>& x, const Out1D& out) {
  if (out.shape(0) != x.shape(0)) {
    throw std::invalid_argument("out has length " + std::to_string(out.shape(0)) + ", expected " +
                                std::to_string(x.shape(0)));
  }
  const ByteRange in_range =
      byte_range(x.data(), x.shape(0), static_cast<std::ptrdiff_t>(x.stride(0)), sizeof(T));
  const ByteRange out_range = byte_range(
      out.data(), out.shape(0), static_cast<std::ptrdiff_t>(out.stride(0)), sizeof(double));
  if (overlaps(in_range, out_range)) {
    throw std::invalid_argument("out must not overlap the input buffer");
  }
}

}  // namespace qkern::bindings
