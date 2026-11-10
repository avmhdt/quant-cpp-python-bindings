// Exponentially weighted batch kernels and the alpha/span/halflife resolution.
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
#include "qkern/ewm.hpp"

namespace qkern::bindings {

namespace {

using MaybeOut = std::optional<AnyOut>;

std::optional<NumpyF64> ewm_mean_fn(nb::handle x, double alpha, std::int64_t min_periods,
                                    const std::string& nan_policy, const MaybeOut& out) {
  const EwmConfig cfg = make_ewm_config(alpha, false, min_periods, nan_policy);
  return run_unary(x, out, [&](auto in, OutView o) { ewm_mean(in, o, cfg); });
}

std::optional<NumpyF64> ewm_var_fn(nb::handle x, double alpha, bool bias, std::int64_t min_periods,
                                   const std::string& nan_policy, const MaybeOut& out) {
  const EwmConfig cfg = make_ewm_config(alpha, bias, min_periods, nan_policy);
  return run_unary(x, out, [&](auto in, OutView o) { ewm_var(in, o, cfg); });
}

std::optional<NumpyF64> ewm_std_fn(nb::handle x, double alpha, bool bias, std::int64_t min_periods,
                                   const std::string& nan_policy, const MaybeOut& out) {
  const EwmConfig cfg = make_ewm_config(alpha, bias, min_periods, nan_policy);
  return run_unary(x, out, [&](auto in, OutView o) { ewm_std(in, o, cfg); });
}

}  // namespace

void bind_ewm(nb::module_& m) {
  m.def("_resolve_alpha", &resolve_alpha, nb::arg("alpha").none(), nb::arg("span").none(),
        nb::arg("halflife").none(),
        "Smoothing factor from exactly one of alpha, span (2 / (span + 1)) or halflife "
        "(1 - exp(-ln 2 / halflife)).");
  m.def("ewm_mean", &ewm_mean_fn, nb::arg("x"), nb::arg("alpha"), nb::arg("min_periods"),
        nb::arg("nan_policy"), nb::arg("out").none(),
        "EWM mean (see qkern.ewm_mean). Returns None when out is given.");
  m.def("ewm_var", &ewm_var_fn, nb::arg("x"), nb::arg("alpha"), nb::arg("bias"),
        nb::arg("min_periods"), nb::arg("nan_policy"), nb::arg("out").none(),
        "EWM variance (see qkern.ewm_var). Returns None when out is given.");
  m.def("ewm_std", &ewm_std_fn, nb::arg("x"), nb::arg("alpha"), nb::arg("bias"),
        nb::arg("min_periods"), nb::arg("nan_policy"), nb::arg("out").none(),
        "EWM standard deviation (see qkern.ewm_std). Returns None when out is given.");
}

}  // namespace qkern::bindings
