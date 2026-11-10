// Arrow C Data Interface importer (FR-12, DQ-06, D-07 option a): validates an
// ArrowSchema/ArrowArray pair of a primitive float64 ("g") or float32 ("f") array and exposes
// its values buffer as a StridedView without copying. Null slots are read through the
// validity bitmap and behave exactly like NaN. The caller keeps the producer's capsules alive
// while the view is in use; the importer never frees anything itself.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "qkern/arrow_c_abi.h"
#include "qkern/views.hpp"

namespace qkern {

enum class ArrowKind : std::uint8_t { Float64, Float32 };

struct ArrowValues {
  ArrowKind kind = ArrowKind::Float64;
  const void* values = nullptr;  // already offset to the first element
  std::size_t length = 0;
  const std::uint8_t* validity = nullptr;  // nullptr when the array has no nulls
  std::int64_t validity_offset = 0;

  template <Real T>
  [[nodiscard]] StridedView<T> view() const noexcept {
    return StridedView<T>{static_cast<const T*>(values), length, 1, validity, validity_offset};
  }
};

// Throws std::invalid_argument with a message suitable for a TypeError.
inline ArrowValues import_arrow(const ArrowSchema* schema, const ArrowArray* array) {
  if (schema == nullptr || array == nullptr || schema->release == nullptr ||
      array->release == nullptr) {
    throw std::invalid_argument("Arrow array or schema is released or null");
  }
  const char* fmt = schema->format;
  ArrowValues out;
  std::size_t itemsize = 0;
  if (fmt != nullptr && std::strcmp(fmt, "g") == 0) {
    out.kind = ArrowKind::Float64;
    itemsize = 8;
  } else if (fmt != nullptr && std::strcmp(fmt, "f") == 0) {
    out.kind = ArrowKind::Float32;
    itemsize = 4;
  } else {
    throw std::invalid_argument(std::string("unsupported Arrow format '") +
                                (fmt == nullptr ? "" : fmt) +
                                "'; expected float64 ('g') or float32 ('f')");
  }
  if (schema->n_children != 0 || array->n_children != 0 || array->dictionary != nullptr) {
    throw std::invalid_argument("nested or dictionary-encoded Arrow arrays are not supported");
  }
  if (array->length < 0 || array->offset < 0 || array->n_buffers != 2) {
    throw std::invalid_argument("malformed primitive Arrow array");
  }
  const auto* values = static_cast<const std::uint8_t*>(array->buffers[1]);
  if (values == nullptr && array->length > 0) {
    throw std::invalid_argument("Arrow array has no values buffer");
  }
  out.length = static_cast<std::size_t>(array->length);
  out.values =
      values == nullptr ? nullptr : values + static_cast<std::size_t>(array->offset) * itemsize;
  if (array->null_count != 0 && array->buffers[0] != nullptr) {
    out.validity = static_cast<const std::uint8_t*>(array->buffers[0]);
    out.validity_offset = array->offset;
  }
  return out;
}

}  // namespace qkern
