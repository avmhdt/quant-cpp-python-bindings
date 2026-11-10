// Rolling-moment batch kernels: mean, var, std, z-score and the combined moments. The public
// Python wrappers in qkern/_api.py normalise arguments and call these once.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>

#include "bind.hpp"
#include "config.hpp"
#include "ndarray_utils.hpp"
#include "qkern/rolling_moments.hpp"

namespace qkern::bindings {

namespace {

using MaybeOut = std::optional<AnyOut>;

std::optional<NumpyF64> mean_fn(nb::handle x, std::int64_t window,
                                std::optional<std::int64_t> min_periods,
                                const std::string& nan_policy, const MaybeOut& out) {
  const RollingConfig cfg = make_rolling_config(window, min_periods, 1, nan_policy);
  return run_unary(x, out, [&](auto in, OutView o) { rolling_mean(in, o, cfg); });
}

std::optional<NumpyF64> var_fn(nb::handle x, std::int64_t window,
                               std::optional<std::int64_t> min_periods, std::int64_t ddof,
                               const std::string& nan_policy, const MaybeOut& out) {
  const RollingConfig cfg = make_rolling_config(window, min_periods, ddof, nan_policy);
  return run_unary(x, out, [&](auto in, OutView o) { rolling_var(in, o, cfg); });
}

std::optional<NumpyF64> std_fn(nb::handle x, std::int64_t window,
                               std::optional<std::int64_t> min_periods, std::int64_t ddof,
                               const std::string& nan_policy, const MaybeOut& out) {
  const RollingConfig cfg = make_rolling_config(window, min_periods, ddof, nan_policy);
  return run_unary(x, out, [&](auto in, OutView o) { rolling_std(in, o, cfg); });
}

std::optional<NumpyF64> zscore_fn(nb::handle x, std::int64_t window,
                                  std::optional<std::int64_t> min_periods, std::int64_t ddof,
                                  bool include_current, const std::string& nan_policy,
                                  const MaybeOut& out) {
  const RollingConfig cfg = make_rolling_config(window, min_periods, ddof, nan_policy);
  return run_unary(x, out,
                   [&](auto in, OutView o) { rolling_zscore(in, o, cfg, include_current); });
}

std::tuple<NumpyF64, NumpyF64, NumpyI64> moments_fn(nb::handle x, std::int64_t window,
                                                    std::optional<std::int64_t> min_periods,
                                                    std::int64_t ddof,
                                                    const std::string& nan_policy) {
  const RollingConfig cfg = make_rolling_config(window, min_periods, ddof, nan_policy);
  const Input in = resolve_input(x, "x");
  OutBuffer<double> mean(in);
  OutBuffer<double> var(in);
  OutBuffer<std::int64_t> count(in);
  release_gil_if(in.size(), [&] {
    for (std::size_t j = 0; j < in.cols; ++j) {
      in.visit(j, [&](auto view) {
        rolling_moments(view, OutView{mean.column(j), in.rows, 1},
                        OutView{var.column(j), in.rows, 1},
                        CountOutView{count.column(j), in.rows, 1}, cfg);
      });
    }
  });
  return {mean.release_to_numpy(), var.release_to_numpy(), count.release_to_numpy()};
}

}  // namespace

void bind_rolling(nb::module_& m) {
  m.def("rolling_mean", &mean_fn, nb::arg("x"), nb::arg("window"), nb::arg("min_periods"),
        nb::arg("nan_policy"), nb::arg("out").none(),
        "Rolling mean (see qkern.rolling_mean). Returns None when out is given.");
  m.def("rolling_var", &var_fn, nb::arg("x"), nb::arg("window"), nb::arg("min_periods"),
        nb::arg("ddof"), nb::arg("nan_policy"), nb::arg("out").none(),
        "Rolling variance (see qkern.rolling_var). Returns None when out is given.");
  m.def("rolling_std", &std_fn, nb::arg("x"), nb::arg("window"), nb::arg("min_periods"),
        nb::arg("ddof"), nb::arg("nan_policy"), nb::arg("out").none(),
        "Rolling standard deviation (see qkern.rolling_std). Returns None when out is given.");
  m.def("rolling_zscore", &zscore_fn, nb::arg("x"), nb::arg("window"), nb::arg("min_periods"),
        nb::arg("ddof"), nb::arg("include_current"), nb::arg("nan_policy"), nb::arg("out").none(),
        "Rolling z-score (see qkern.rolling_zscore). Returns None when out is given.");
  m.def("rolling_moments", &moments_fn, nb::arg("x"), nb::arg("window"), nb::arg("min_periods"),
        nb::arg("ddof"), nb::arg("nan_policy"),
        "Rolling mean, variance and valid count in one pass (see qkern.rolling_moments).");
}

}  // namespace qkern::bindings
