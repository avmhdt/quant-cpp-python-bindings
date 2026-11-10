// _ArrowInput: a validated, zero-copy view of a float64/float32 Arrow array obtained through
// the Arrow PyCapsule interface (FR-12, D-07). It keeps the producer's capsules (or the one
// chunk it pulled from a stream) alive for as long as the view is used.
#pragma once

#include <nanobind/nanobind.h>

#include <stdexcept>
#include <string>

#include "qkern/arrow_c_abi.h"
#include "qkern/arrow_import.hpp"

namespace qkern::bindings {

namespace nb = nanobind;

class ArrowInput {
 public:
  ArrowInput() = default;
  ArrowInput(const ArrowInput&) = delete;
  ArrowInput& operator=(const ArrowInput&) = delete;
  ArrowInput(ArrowInput&&) = delete;
  ArrowInput& operator=(ArrowInput&&) = delete;

  ~ArrowInput() {
    // Only structs this object pulled from a stream are ours to release.
    if (owned_array_.release != nullptr) {
      owned_array_.release(&owned_array_);
    }
    if (owned_schema_.release != nullptr) {
      owned_schema_.release(&owned_schema_);
    }
  }

  // From __arrow_c_array__(): the capsules keep ownership; we only read.
  void init_from_array(const nb::object& schema_capsule, const nb::object& array_capsule) {
    auto* schema = static_cast<ArrowSchema*>(capsule_pointer(schema_capsule, "arrow_schema"));
    auto* array = static_cast<ArrowArray*>(capsule_pointer(array_capsule, "arrow_array"));
    values_ = checked_import(schema, array);
    keep_alive_ = nb::make_tuple(schema_capsule, array_capsule);
  }

  // From __arrow_c_stream__(): one chunk is viewed in place; more chunks raise TypeError.
  void init_from_stream(const nb::object& stream_capsule) {
    auto* stream =
        static_cast<ArrowArrayStream*>(capsule_pointer(stream_capsule, "arrow_array_stream"));
    if (stream->release == nullptr) {
      throw nb::type_error("the Arrow stream has already been consumed");
    }
    if (stream->get_schema(stream, &owned_schema_) != 0) {
      throw nb::type_error(stream_error(stream).c_str());
    }
    int chunks = 0;
    while (true) {
      ArrowArray next{};
      if (stream->get_next(stream, &next) != 0) {
        throw nb::type_error(stream_error(stream).c_str());
      }
      if (next.release == nullptr) {
        break;  // end of stream
      }
      if (next.length == 0 && chunks > 0) {
        next.release(&next);
        continue;
      }
      if (chunks == 1) {
        next.release(&next);
        throw nb::type_error(
            "Arrow input has more than one chunk; combine them first "
            "(pyarrow: combine_chunks(), Polars: rechunk())");
      }
      if (owned_array_.release != nullptr) {
        owned_array_.release(&owned_array_);  // an earlier empty chunk
      }
      owned_array_ = next;
      if (next.length > 0) {
        chunks = 1;
      }
    }
    if (owned_array_.release == nullptr) {
      // An empty stream: a zero-length array of the stream's type.
      values_ = ArrowValues{};
      values_.kind = std::string(owned_schema_.format == nullptr ? "" : owned_schema_.format) == "f"
                         ? ArrowKind::Float32
                         : ArrowKind::Float64;
      if (!format_ok()) {
        throw nb::type_error("unsupported Arrow format; expected float64 ('g') or float32 ('f')");
      }
    } else {
      values_ = checked_import(&owned_schema_, &owned_array_);
    }
    keep_alive_ = stream_capsule;
  }

  [[nodiscard]] const ArrowValues& values() const noexcept { return values_; }

 private:
  static void* capsule_pointer(const nb::object& capsule, const char* name) {
    void* p = PyCapsule_GetPointer(capsule.ptr(), name);
    if (p == nullptr) {
      PyErr_Clear();
      throw nb::type_error((std::string("expected a PyCapsule named '") + name + "'").c_str());
    }
    return p;
  }

  static std::string stream_error(ArrowArrayStream* stream) {
    const char* msg = stream->get_last_error == nullptr ? nullptr : stream->get_last_error(stream);
    return std::string("Arrow stream error: ") + (msg == nullptr ? "unknown" : msg);
  }

  [[nodiscard]] bool format_ok() const {
    const std::string fmt = owned_schema_.format == nullptr ? "" : owned_schema_.format;
    return fmt == "g" || fmt == "f";
  }

  // Unsupported Arrow arrays are a TypeError in Python (spec section 10 error table).
  static ArrowValues checked_import(const ArrowSchema* schema, const ArrowArray* array) {
    try {
      return import_arrow(schema, array);
    } catch (const std::invalid_argument& e) {
      throw nb::type_error(e.what());
    }
  }

  ArrowValues values_;
  ArrowSchema owned_schema_{};
  ArrowArray owned_array_{};
  nb::object keep_alive_;
};

}  // namespace qkern::bindings
