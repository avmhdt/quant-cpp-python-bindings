// Rolling-window batch kernels. The public Python wrappers live in qkern/_api.py; these
// entry points take already-normalised arguments.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstddef>
#include <string>

#include "bind.hpp"
#include "ndarray_utils.hpp"
#include "qkern/nan_policy.hpp"
#include "qkern/rolling_moments.hpp"

namespace qkern::bindings {

namespace {

RollingConfig make_config(std::size_t window, std::size_t min_periods, const std::string& nan) {
  RollingConfig cfg{window, min_periods, 1, parse_nan_policy(nan)};
  validate(cfg);
  return cfg;
}

template <Real T>
NumpyF64 rolling_mean_alloc(const In1D<T>& x, std::size_t window, std::size_t min_periods,
                            const std::string& nan) {
  const RollingConfig cfg = make_config(window, min_periods, nan);
  const StridedView<T> in = view_of(x);
  OutBuffer buf(in.size);
  release_gil_if(in.size, [&] { rolling_mean(in, buf.view(), cfg); });
  return buf.release_to_numpy();
}

template <Real T>
void rolling_mean_into(const In1D<T>& x, std::size_t window, std::size_t min_periods,
                       const std::string& nan, const Out1D& out) {
  const RollingConfig cfg = make_config(window, min_periods, nan);
  check_out(x, out);
  const StridedView<T> in = view_of(x);
  const OutView dst = view_of(out);
  release_gil_if(in.size, [&] { rolling_mean(in, dst, cfg); });
}

template <Real T>
void def_rolling_mean(nb::module_& m) {
  m.def("rolling_mean", &rolling_mean_alloc<T>, nb::arg("x").noconvert(), nb::arg("window"),
        nb::arg("min_periods"), nb::arg("nan_policy"),
        "Rolling mean over the last `window` observations (see qkern.rolling_mean).");
  m.def("rolling_mean_into", &rolling_mean_into<T>, nb::arg("x").noconvert(), nb::arg("window"),
        nb::arg("min_periods"), nb::arg("nan_policy"), nb::arg("out").noconvert(),
        "Rolling mean written into a caller-supplied float64 array.");
}

}  // namespace

void bind_rolling(nb::module_& m) {
  def_rolling_mean<double>(m);
  def_rolling_mean<float>(m);
}

}  // namespace qkern::bindings
