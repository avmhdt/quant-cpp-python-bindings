// Rolling OLS batch kernel. y may be 1-D or 2-D; x is either the same shape as y or 1-D of
// y's length, in which case every column of y is regressed on the same x (for example stock
// returns on one market return).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>

#include "bind.hpp"
#include "config.hpp"
#include "ndarray_utils.hpp"
#include "qkern/rolling_ols.hpp"

namespace qkern::bindings {

namespace {

using OlsArrays = std::tuple<NumpyF64, NumpyF64, NumpyF64, NumpyF64>;

OlsArrays ols_fn(nb::handle y, nb::handle x, std::int64_t window,
                 std::optional<std::int64_t> min_periods, bool include_current,
                 const std::string& nan_policy) {
  const RollingConfig cfg = make_rolling_config(window, min_periods, 1, nan_policy);
  const Input yin = resolve_input(y, "y");
  const Input xin = resolve_input(x, "x");
  const bool broadcast_x = xin.ndim == 1 && yin.ndim == 2;
  if (xin.rows != yin.rows || (!broadcast_x && xin.cols != yin.cols)) {
    throw std::invalid_argument(
        "x must have the same shape as y, or be 1-D with y's number of rows");
  }
  OutBuffer<double> alpha(yin);
  OutBuffer<double> beta(yin);
  OutBuffer<double> resid(yin);
  OutBuffer<double> r2(yin);
  release_gil_if(yin.size(), [&] {
    for (std::size_t j = 0; j < yin.cols; ++j) {
      const std::size_t jx = broadcast_x ? 0 : j;
      yin.visit(j, [&](auto yv) {
        xin.visit(jx, [&](auto xv) {
          rolling_ols(yv, xv, OutView{alpha.column(j), yin.rows, 1},
                      OutView{beta.column(j), yin.rows, 1}, OutView{resid.column(j), yin.rows, 1},
                      OutView{r2.column(j), yin.rows, 1}, cfg, include_current);
        });
      });
    }
  });
  return {alpha.release_to_numpy(), beta.release_to_numpy(), resid.release_to_numpy(),
          r2.release_to_numpy()};
}

}  // namespace

void bind_ols(nb::module_& m) {
  m.def("rolling_ols", &ols_fn, nb::arg("y"), nb::arg("x"), nb::arg("window"),
        nb::arg("min_periods"), nb::arg("include_current"), nb::arg("nan_policy"),
        "Rolling OLS of y on x with intercept: (alpha, beta, resid, r2) (see qkern.rolling_ols).");
}

}  // namespace qkern::bindings
