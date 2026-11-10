// Rolling minimum and maximum batch kernels.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <optional>
#include <string>

#include "bind.hpp"
#include "config.hpp"
#include "ndarray_utils.hpp"
#include "qkern/rolling_minmax.hpp"

namespace qkern::bindings {

namespace {

using MaybeOut = std::optional<AnyOut>;

std::optional<NumpyF64> min_fn(const AnyIn& x, std::int64_t window,
                               std::optional<std::int64_t> min_periods,
                               const std::string& nan_policy, const MaybeOut& out) {
  const RollingConfig cfg = make_rolling_config(window, min_periods, 1, nan_policy);
  return run_unary(x, out, [&](auto in, OutView o) { rolling_min(in, o, cfg); });
}

std::optional<NumpyF64> max_fn(const AnyIn& x, std::int64_t window,
                               std::optional<std::int64_t> min_periods,
                               const std::string& nan_policy, const MaybeOut& out) {
  const RollingConfig cfg = make_rolling_config(window, min_periods, 1, nan_policy);
  return run_unary(x, out, [&](auto in, OutView o) { rolling_max(in, o, cfg); });
}

}  // namespace

void bind_minmax(nb::module_& m) {
  m.def("rolling_min", &min_fn, nb::arg("x"), nb::arg("window"), nb::arg("min_periods"),
        nb::arg("nan_policy"), nb::arg("out").none(),
        "Rolling minimum (see qkern.rolling_min). Returns None when out is given.");
  m.def("rolling_max", &max_fn, nb::arg("x"), nb::arg("window"), nb::arg("min_periods"),
        nb::arg("nan_policy"), nb::arg("out").none(),
        "Rolling maximum (see qkern.rolling_max). Returns None when out is given.");
}

}  // namespace qkern::bindings
