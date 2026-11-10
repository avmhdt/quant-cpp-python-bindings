// Binding-layer helpers: ndarray -> view without copying, C++-allocated outputs handed to
// NumPy through a capsule (no copy on return), out= validation, the GIL-release threshold
// and the debug copy counters (FR-10, FR-11, FR-13, FR-14, NFR-04).
//
// Rule: every Python object is touched before the GIL is released. Code inside
// release_gil_if() sees only raw pointers and views.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <array>
#include <atomic>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "qkern/views.hpp"

namespace qkern::bindings {

namespace nb = nanobind;

// Any read-only CPU array; dtype and ndim are checked by resolve_input().
using AnyIn = nb::ndarray<nb::ro, nb::device::cpu>;
// Writable float64 out= target of any stride (1-D or 2-D, checked by resolve_out()).
using AnyOut = nb::ndarray<double, nb::device::cpu>;
// NumPy arrays returned to Python; each owns a C++ buffer through a capsule.
using NumpyF64 = nb::ndarray<nb::numpy, double>;
using NumpyI64 = nb::ndarray<nb::numpy, std::int64_t>;

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

// ---------------------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------------------

enum class Kind : std::uint8_t { F64, F32 };

// A 1-D or 2-D input as float64 or float32 storage. float64 and float32 arrays are read in
// place with their own strides (any sign); integer and bool arrays are converted once to a
// column-major float64 copy, counted in debug_counters().input_conversions (DQ-05).
struct Input {
  const void* data = nullptr;
  Kind kind = Kind::F64;
  std::size_t ndim = 1;
  std::size_t rows = 0;
  std::size_t cols = 1;
  std::ptrdiff_t s0 = 1;  // element strides
  std::ptrdiff_t s1 = 0;
  std::unique_ptr<double[]> owned;  // NOLINT(*-avoid-c-arrays): converted copy, if any
  // Byte range of the caller's buffer, for the out= overlap check.
  std::uintptr_t lo = 0;
  std::uintptr_t hi = 0;

  [[nodiscard]] std::size_t size() const noexcept { return rows * cols; }

  template <Real T>
  [[nodiscard]] StridedView<T> column(std::size_t j) const noexcept {
    const T* base = static_cast<const T*>(data) + static_cast<std::ptrdiff_t>(j) * s1;
    return StridedView<T>{base, rows, s0};
  }

  // Calls f(StridedView<double or float>) for column j.
  template <class F>
  void visit(std::size_t j, F&& f) const {
    if (kind == Kind::F64) {
      std::forward<F>(f)(column<double>(j));
    } else {
      std::forward<F>(f)(column<float>(j));
    }
  }
};

namespace detail {

struct Span {
  std::uintptr_t lo = 0;
  std::uintptr_t hi = 0;
};

// Byte range spanned by a strided array of up to 2 dimensions.
inline Span byte_span(const void* data, std::size_t ndim, const std::size_t* shape,
                      const std::int64_t* strides, std::size_t itemsize) noexcept {
  for (std::size_t d = 0; d < ndim; ++d) {
    if (shape[d] == 0) {
      return {};
    }
  }
  const auto base = reinterpret_cast<std::uintptr_t>(data);  // NOLINT(*-reinterpret-cast)
  std::int64_t lo = 0;
  std::int64_t hi = 0;
  for (std::size_t d = 0; d < ndim; ++d) {
    const std::int64_t extent =
        static_cast<std::int64_t>(shape[d] - 1) * strides[d] * static_cast<std::int64_t>(itemsize);
    (extent < 0 ? lo : hi) += extent;
  }
  return {base + static_cast<std::uintptr_t>(lo),
          base + static_cast<std::uintptr_t>(hi) + itemsize};
}

template <class T>
double read_as_double(const void* data, std::ptrdiff_t offset) noexcept {
  return static_cast<double>(static_cast<const T*>(data)[offset]);
}

using Reader = double (*)(const void*, std::ptrdiff_t);

// Reader for an integer or bool dtype, or nullptr when the dtype is not convertible.
inline Reader reader_for(nb::dlpack::dtype dt) noexcept {
  using Code = nb::dlpack::dtype_code;
  const auto code = static_cast<Code>(dt.code);
  if (dt.lanes != 1) {
    return nullptr;
  }
  if (code == Code::Bool && dt.bits == 8) {
    return &read_as_double<bool>;
  }
  if (code == Code::Int) {
    switch (dt.bits) {
      case 8:
        return &read_as_double<std::int8_t>;
      case 16:
        return &read_as_double<std::int16_t>;
      case 32:
        return &read_as_double<std::int32_t>;
      case 64:
        return &read_as_double<std::int64_t>;
      default:
        return nullptr;
    }
  }
  if (code == Code::UInt) {
    switch (dt.bits) {
      case 8:
        return &read_as_double<std::uint8_t>;
      case 16:
        return &read_as_double<std::uint16_t>;
      case 32:
        return &read_as_double<std::uint32_t>;
      case 64:
        return &read_as_double<std::uint64_t>;
      default:
        return nullptr;
    }
  }
  return nullptr;
}

}  // namespace detail

// Resolves an ndarray to an Input. max_ndim is 1 for streaming update_many and 2 for batch
// kernels (2-D inputs are processed column by column along axis 0, FR-13).
inline Input resolve_input(const AnyIn& a, const char* name, std::size_t max_ndim = 2) {
  const std::size_t ndim = a.ndim();
  if (ndim < 1 || ndim > max_ndim) {
    const std::string want = max_ndim == 1 ? "1-D" : "1-D or 2-D";
    throw nb::type_error(
        (std::string(name) + " must be " + want + ", got ndim=" + std::to_string(ndim)).c_str());
  }
  Input in;
  in.ndim = ndim;
  in.rows = a.shape(0);
  in.cols = ndim == 2 ? a.shape(1) : 1;
  in.s0 = static_cast<std::ptrdiff_t>(a.stride(0));
  in.s1 = ndim == 2 ? static_cast<std::ptrdiff_t>(a.stride(1)) : 0;
  const std::array<std::size_t, 2> shape{a.shape(0), ndim == 2 ? a.shape(1) : 1};
  const std::array<std::int64_t, 2> strides{a.stride(0), ndim == 2 ? a.stride(1) : 0};
  const nb::dlpack::dtype dt = a.dtype();
  const detail::Span span =
      detail::byte_span(a.data(), ndim, shape.data(), strides.data(), a.itemsize());
  in.lo = span.lo;
  in.hi = span.hi;
  if (dt == nb::dtype<double>()) {
    in.kind = Kind::F64;
    in.data = a.data();
    return in;
  }
  if (dt == nb::dtype<float>()) {
    in.kind = Kind::F32;
    in.data = a.data();
    return in;
  }
  const detail::Reader read = detail::reader_for(dt);
  if (read == nullptr) {
    throw nb::type_error((std::string(name) +
                          " has an unsupported dtype; expected a real numeric dtype "
                          "(float64, float32, integer or bool)")
                             .c_str());
  }
  // One conversion to a column-major float64 buffer.
  const std::size_t n = in.size();
  in.owned = std::make_unique<double[]>(n == 0 ? 1 : n);  // NOLINT(*-avoid-c-arrays)
  for (std::size_t j = 0; j < in.cols; ++j) {
    for (std::size_t i = 0; i < in.rows; ++i) {
      const auto offset =
          static_cast<std::ptrdiff_t>(i) * in.s0 + static_cast<std::ptrdiff_t>(j) * in.s1;
      in.owned[j * in.rows + i] = read(a.data(), offset);
    }
  }
  in.kind = Kind::F64;
  in.data = in.owned.get();
  in.s0 = 1;
  in.s1 = static_cast<std::ptrdiff_t>(in.rows);
  debug_counters().input_conversions.fetch_add(1, std::memory_order_relaxed);
  return in;
}

// ---------------------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------------------

// Writable float64 output columns: a caller's out= array or a fresh column-major buffer.
struct Output {
  double* data = nullptr;
  std::size_t rows = 0;
  std::size_t cols = 1;
  std::ptrdiff_t s0 = 1;
  std::ptrdiff_t s1 = 0;

  [[nodiscard]] OutView column(std::size_t j) const noexcept {
    return OutView{data + static_cast<std::ptrdiff_t>(j) * s1, rows, s0};
  }
};

// A column-major buffer allocated before the GIL is released and handed to NumPy afterwards
// through a capsule, so the returned array is not a copy (FR-11).
template <class T>
class OutBuffer {
 public:
  explicit OutBuffer(const Input& shape_of)
      : ndim_(shape_of.ndim),
        rows_(shape_of.rows),
        cols_(shape_of.cols),
        // NOLINTNEXTLINE(*-avoid-c-arrays): a runtime-sized buffer handed to NumPy
        data_(std::make_unique<T[]>(rows_ * cols_ == 0 ? 1 : rows_ * cols_)) {}

  // Column j, contiguous.
  [[nodiscard]] T* column(std::size_t j) const noexcept { return data_.get() + j * rows_; }

  [[nodiscard]] Output as_output() const noexcept
    requires std::same_as<T, double>
  {
    return Output{data_.get(), rows_, cols_, 1, static_cast<std::ptrdiff_t>(rows_)};
  }

  // Transfers ownership to a capsule; the returned array's base is that capsule.
  [[nodiscard]] nb::ndarray<nb::numpy, T> release_to_numpy() {
    T* raw = data_.release();
    const nb::capsule owner(raw, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    const std::array<std::size_t, 2> shape{rows_, cols_};
    const std::array<std::int64_t, 2> strides{1, static_cast<std::int64_t>(rows_)};
    return nb::ndarray<nb::numpy, T>(raw, ndim_, shape.data(), owner, strides.data());
  }

 private:
  std::size_t ndim_;
  std::size_t rows_;
  std::size_t cols_;
  std::unique_ptr<T[]> data_;  // NOLINT(*-avoid-c-arrays)
};

// Validates out= against the input (DQ-11): same shape, float64 and writable (enforced by
// the AnyOut type), no memory overlap with the caller's input buffer.
inline Output resolve_out(const AnyOut& out, const Input& in) {
  if (out.ndim() != in.ndim) {
    throw nb::type_error(("out must have ndim=" + std::to_string(in.ndim)).c_str());
  }
  const std::size_t rows = out.shape(0);
  const std::size_t cols = out.ndim() == 2 ? out.shape(1) : 1;
  if (rows != in.rows || cols != in.cols) {
    throw std::invalid_argument("out has the wrong shape for the input");
  }
  const std::array<std::size_t, 2> shape{rows, cols};
  const std::array<std::int64_t, 2> strides{out.stride(0), out.ndim() == 2 ? out.stride(1) : 0};
  const detail::Span span =
      detail::byte_span(out.data(), out.ndim(), shape.data(), strides.data(), sizeof(double));
  if (span.lo < span.hi && in.lo < in.hi && span.lo < in.hi && in.lo < span.hi) {
    throw std::invalid_argument("out must not overlap the input buffer");
  }
  return Output{out.data(), rows, cols, static_cast<std::ptrdiff_t>(out.stride(0)),
                out.ndim() == 2 ? static_cast<std::ptrdiff_t>(out.stride(1)) : 0};
}

// Runs a single-output kernel `kernel(StridedView<T>, OutView)` over every column, into
// out= when given (returns nullopt) or into a new array (returned). The GIL is released
// once for the whole call (FR-13, FR-14).
template <class Kernel>
std::optional<NumpyF64> run_unary(const AnyIn& x, const std::optional<AnyOut>& out,
                                  const Kernel& kernel) {
  const Input in = resolve_input(x, "x");
  std::optional<OutBuffer<double>> buf;
  Output dst;
  if (out.has_value()) {
    dst = resolve_out(*out, in);
  } else {
    buf.emplace(in);
    dst = buf->as_output();
  }
  release_gil_if(in.size(), [&] {
    for (std::size_t j = 0; j < in.cols; ++j) {
      in.visit(j, [&](auto view) { kernel(view, dst.column(j)); });
    }
  });
  if (out.has_value()) {
    return std::nullopt;
  }
  return buf->release_to_numpy();
}

}  // namespace qkern::bindings
