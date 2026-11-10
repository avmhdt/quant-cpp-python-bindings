// NB_MODULE(_qkern): build metadata, private test hooks and the per-family bindings.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstddef>
#include <cstdint>

#include "bind.hpp"
#include "ndarray_utils.hpp"
#include "qkern/version.hpp"

namespace nb = nanobind;
namespace qb = qkern::bindings;

namespace {

#ifndef QKERN_GIT_SHA
#define QKERN_GIT_SHA "unknown"
#endif
#ifndef QKERN_COMPILER_ID
#define QKERN_COMPILER_ID "unknown"
#endif
#ifndef QKERN_COMPILER_VERSION
#define QKERN_COMPILER_VERSION "unknown"
#endif
#ifndef QKERN_BUILD_TYPE
#define QKERN_BUILD_TYPE "unknown"
#endif
#ifndef QKERN_FP_FLAGS
#define QKERN_FP_FLAGS "unknown"
#endif
#ifndef QKERN_NANOBIND_VERSION
#define QKERN_NANOBIND_VERSION "unknown"
#endif

#if defined(Py_LIMITED_API)
constexpr bool kAbi3 = true;
#else
constexpr bool kAbi3 = false;
#endif

#if defined(Py_GIL_DISABLED)
constexpr bool kFreeThreaded = true;
#else
constexpr bool kFreeThreaded = false;
#endif

nb::typed<nb::dict, nb::str, nb::object> build_info() {
  nb::typed<nb::dict, nb::str, nb::object> d;
  d["version"] = QKERN_VERSION_STRING;
  d["git_sha"] = QKERN_GIT_SHA;
  d["compiler"] = QKERN_COMPILER_ID;
  d["compiler_version"] = QKERN_COMPILER_VERSION;
#if defined(_MSVC_LANG)
  d["cxx_standard"] = static_cast<std::int64_t>(_MSVC_LANG);  // MSVC keeps __cplusplus at 199711
#else
  d["cxx_standard"] = static_cast<std::int64_t>(__cplusplus);
#endif
  d["build_type"] = QKERN_BUILD_TYPE;
  d["fp_flags"] = QKERN_FP_FLAGS;
  d["nanobind_version"] = QKERN_NANOBIND_VERSION;
  d["abi3"] = kAbi3;
  d["python_build_version"] = PY_VERSION;
  d["gil_release_min_n"] =
      static_cast<std::int64_t>(qb::gil_release_min_n().load(std::memory_order_relaxed));
  d["free_threaded"] = kFreeThreaded;
  return d;
}

nb::typed<nb::dict, nb::str, std::uint64_t> debug_counters() {
  nb::typed<nb::dict, nb::str, std::uint64_t> d;
  d["input_conversions"] = qb::debug_counters().input_conversions.load();
  d["output_copies"] = qb::debug_counters().output_copies.load();
  return d;
}

}  // namespace

NB_MODULE(_qkern, m) {  // NOLINT(*-avoid-non-const-global-variables)
  m.doc() = "qkern C++ extension: streaming and rolling time-series kernels.";

  m.def("_build_info", &build_info, "Build metadata used by qkern.build_info().");
  m.def(
      "_set_gil_release_min_n",
      [](std::size_t n) { qb::gil_release_min_n().store(n, std::memory_order_relaxed); },
      nb::arg("n"), "Private test/benchmark hook: GIL release threshold in elements.");
  m.def(
      "_get_gil_release_min_n",
      [] { return qb::gil_release_min_n().load(std::memory_order_relaxed); },
      "Private test/benchmark hook: current GIL release threshold.");
  m.def("_debug_counters", &debug_counters,
        "Private test hook: input conversions and output copies made by the binding layer.");
  m.def(
      "_reset_debug_counters",
      [] {
        qb::debug_counters().input_conversions.store(0);
        qb::debug_counters().output_copies.store(0);
      },
      "Private test hook: zero the debug counters.");

  qb::bind_rolling(m);
  qb::bind_ewm(m);
  qb::bind_minmax(m);
  qb::bind_ols(m);
  qb::bind_streaming(m);
  qb::bind_quantile(m);
}
