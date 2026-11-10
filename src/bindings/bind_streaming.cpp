// Streaming classes: the same C++ state machines the batch kernels drive (FR-08), with
// update / update_many / reset, read-only properties and versioned pickling (FR-09).
// update() keeps the GIL; update_many() releases it like a batch call (FR-14). Objects are
// not thread-safe: do not share one between threads.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "bind.hpp"
#include "config.hpp"
#include "ndarray_utils.hpp"
#include "qkern/ewm.hpp"
#include "qkern/rolling_minmax.hpp"
#include "qkern/rolling_moments.hpp"
#include "qkern/rolling_ols.hpp"

namespace qkern::bindings {

namespace {

using namespace nb::literals;

constexpr std::int64_t kStateVersion = 1;
constexpr double kNaNValue = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------------------------------
// Pickle helpers: {"v": 1, "config": {...}, "buffer": [...], "ints": [...], "reals": [...]}
// ---------------------------------------------------------------------------------------

using StateDict = nb::typed<nb::dict, nb::str, nb::object>;

StateDict to_dict(const State& st, const nb::dict& config) {
  StateDict d;
  d["v"] = kStateVersion;
  d["config"] = config;
  d["buffer"] = st.buffer;
  d["ints"] = st.ints;
  d["reals"] = st.reals;
  return d;
}

nb::dict checked(const StateDict& d) {
  if (!d.contains("v") || nb::cast<std::int64_t>(d["v"]) != kStateVersion) {
    throw std::invalid_argument("unsupported qkern state version (expected v=1)");
  }
  return nb::cast<nb::dict>(d["config"]);
}

State from_dict(const StateDict& d) {
  return State{nb::cast<std::vector<double>>(d["buffer"]),
               nb::cast<std::vector<std::uint64_t>>(d["ints"]),
               nb::cast<std::vector<double>>(d["reals"])};
}

RollingConfig rolling_config_from(const nb::dict& c) {
  return make_rolling_config(nb::cast<std::int64_t>(c["window"]),
                             nb::cast<std::int64_t>(c["min_periods"]),
                             c.contains("ddof") ? nb::cast<std::int64_t>(c["ddof"]) : 1,
                             nb::cast<std::string>(c["nan_policy"]));
}

nb::dict rolling_config_dict(const RollingConfig& cfg, bool with_ddof) {
  nb::dict c;
  c["window"] = cfg.window;
  c["min_periods"] = cfg.min_periods;
  if (with_ddof) {
    c["ddof"] = cfg.ddof;
  }
  c["nan_policy"] = nan_policy_name(cfg.nan);
  return c;
}

nb::object result_type(const char* name) {
  return nb::module_::import_("qkern._results").attr(name);
}

// ---------------------------------------------------------------------------------------
// RollingOLS keeps the last pair so that .resid reports its residual.
// ---------------------------------------------------------------------------------------

struct OlsStream {
  RollingOls state;
  double last_y = kNaNValue;
  double last_x = kNaNValue;

  explicit OlsStream(const RollingConfig& cfg) : state(cfg) {}
  void push(double y, double x) noexcept {
    state.push(y, x);
    last_y = y;
    last_x = x;
  }
};

void bind_rolling_moments(nb::module_& m) {
  nb::class_<RollingMoments>(m, "RollingMoments",
                             "Streaming rolling mean, variance, std and z-score over the last "
                             "`window` observations (same semantics as qkern.rolling_moments).")
      .def(
          "__init__",
          [](RollingMoments* self, std::int64_t window, std::optional<std::int64_t> min_periods,
             std::int64_t ddof, const std::string& nan_policy) {
            new (self) RollingMoments(make_rolling_config(window, min_periods, ddof, nan_policy));
          },
          "window"_a, nb::kw_only(), "min_periods"_a = nb::none(), "ddof"_a = 1,
          "nan_policy"_a = "skip")
      .def("update", &RollingMoments::push, "x"_a, "Push one observation.")
      .def(
          "update_many",
          [](RollingMoments& s, nb::handle x) {
            const Input in = resolve_input(x, "x", 1);
            OutBuffer<double> mean(in);
            OutBuffer<double> var(in);
            OutBuffer<std::int64_t> count(in);
            release_gil_if(in.size(), [&] {
              in.visit(0, [&](auto v) {
                for (std::size_t i = 0; i < v.size; ++i) {
                  s.push(v[i]);
                  mean.column(0)[i] = s.mean();
                  var.column(0)[i] = s.var();
                  count.column(0)[i] = static_cast<std::int64_t>(s.count());
                }
              });
            });
            return result_type("Moments")(mean.release_to_numpy(), var.release_to_numpy(),
                                          count.release_to_numpy());
          },
          "x"_a,
          nb::sig("def update_many(self, x: numpy.typing.NDArray[typing.Any]) -> "
                  "qkern._results.Moments"),
          "Push every element of a 1-D array; returns the per-element Moments.")
      .def("zscore", &RollingMoments::zscore, "x"_a,
           "z-score of x against the current window (x is not pushed).")
      .def("reset", &RollingMoments::reset, "Forget every observation.")
      .def_prop_ro("mean", &RollingMoments::mean)
      .def_prop_ro("var", &RollingMoments::var)
      .def_prop_ro("std", &RollingMoments::stddev)
      .def_prop_ro("count", &RollingMoments::count)
      .def_prop_ro("ready", &RollingMoments::ready)
      .def_prop_ro("window", [](const RollingMoments& s) { return s.config().window; })
      .def_prop_ro("min_periods", [](const RollingMoments& s) { return s.config().min_periods; })
      .def_prop_ro("ddof", [](const RollingMoments& s) { return s.config().ddof; })
      .def_prop_ro("nan_policy",
                   [](const RollingMoments& s) { return nan_policy_name(s.config().nan); })
      .def("__getstate__",
           [](const RollingMoments& s) {
             return to_dict(s.save(), rolling_config_dict(s.config(), true));
           })
      .def("__setstate__",
           [](RollingMoments& self, const StateDict& d) {
             new (&self) RollingMoments(rolling_config_from(checked(d)));
             self.load(from_dict(d));
           })
      .def("__repr__", [](const RollingMoments& s) {
        const auto& c = s.config();
        return "RollingMoments(window=" + std::to_string(c.window) +
               ", min_periods=" + std::to_string(c.min_periods) +
               ", ddof=" + std::to_string(c.ddof) + ", nan_policy='" + nan_policy_name(c.nan) +
               "', count=" + std::to_string(s.count()) + ")";
      });
}

void bind_ewm_class(nb::module_& m) {
  nb::class_<Ewm>(m, "EWM",
                  "Streaming exponentially weighted mean, variance and std (same semantics "
                  "as qkern.ewm_mean / ewm_var).")
      .def(
          "__init__",
          [](Ewm* self, std::optional<double> alpha, std::optional<double> span,
             std::optional<double> halflife, bool bias, std::int64_t min_periods,
             const std::string& nan_policy) {
            new (self) Ewm(make_ewm_config(resolve_alpha(alpha, span, halflife), bias, min_periods,
                                           nan_policy));
          },
          nb::kw_only(), "alpha"_a = nb::none(), "span"_a = nb::none(), "halflife"_a = nb::none(),
          "bias"_a = false, "min_periods"_a = 0, "nan_policy"_a = "skip")
      .def("update", &Ewm::push, "x"_a, "Push one observation.")
      .def(
          "update_many",
          [](Ewm& s, nb::handle x) {
            const Input in = resolve_input(x, "x", 1);
            OutBuffer<double> mean(in);
            OutBuffer<double> var(in);
            release_gil_if(in.size(), [&] {
              in.visit(0, [&](auto v) {
                for (std::size_t i = 0; i < v.size; ++i) {
                  s.push(v[i]);
                  mean.column(0)[i] = s.mean();
                  var.column(0)[i] = s.var();
                }
              });
            });
            return std::make_tuple(mean.release_to_numpy(), var.release_to_numpy());
          },
          "x"_a,
          nb::sig("def update_many(self, x: numpy.typing.NDArray[typing.Any]) -> "
                  "tuple[numpy.typing.NDArray[numpy.float64], "
                  "numpy.typing.NDArray[numpy.float64]]"),
          "Push every element of a 1-D array; returns per-element (mean, var).")
      .def("reset", &Ewm::reset, "Forget every observation.")
      .def_prop_ro("mean", &Ewm::mean)
      .def_prop_ro("var", &Ewm::var)
      .def_prop_ro("std", &Ewm::stddev)
      .def_prop_ro("count", &Ewm::count)
      .def_prop_ro("ready", &Ewm::ready)
      .def_prop_ro("alpha", [](const Ewm& s) { return s.config().alpha; })
      .def_prop_ro("bias", [](const Ewm& s) { return s.config().bias; })
      .def_prop_ro("min_periods", [](const Ewm& s) { return s.config().min_periods; })
      .def_prop_ro("nan_policy", [](const Ewm& s) { return nan_policy_name(s.config().nan); })
      .def("__getstate__",
           [](const Ewm& s) {
             nb::dict c;
             c["alpha"] = s.config().alpha;
             c["bias"] = s.config().bias;
             c["min_periods"] = s.config().min_periods;
             c["nan_policy"] = nan_policy_name(s.config().nan);
             return to_dict(s.save(), c);
           })
      .def("__setstate__",
           [](Ewm& self, const StateDict& d) {
             const nb::dict c = checked(d);
             new (&self) Ewm(make_ewm_config(
                 nb::cast<double>(c["alpha"]), nb::cast<bool>(c["bias"]),
                 nb::cast<std::int64_t>(c["min_periods"]), nb::cast<std::string>(c["nan_policy"])));
             self.load(from_dict(d));
           })
      .def("__repr__", [](const Ewm& s) {
        const auto& c = s.config();
        return "EWM(alpha=" + std::to_string(c.alpha) + ", bias=" + (c.bias ? "True" : "False") +
               ", min_periods=" + std::to_string(c.min_periods) + ", nan_policy='" +
               nan_policy_name(c.nan) + "', count=" + std::to_string(s.count()) + ")";
      });
}

void bind_minmax_class(nb::module_& m) {
  nb::class_<RollingMinMax>(m, "RollingMinMax",
                            "Streaming rolling minimum and maximum (monotonic deques; same "
                            "semantics as qkern.rolling_min / rolling_max).")
      .def(
          "__init__",
          [](RollingMinMax* self, std::int64_t window, std::optional<std::int64_t> min_periods,
             const std::string& nan_policy) {
            new (self) RollingMinMax(make_rolling_config(window, min_periods, 1, nan_policy));
          },
          "window"_a, nb::kw_only(), "min_periods"_a = nb::none(), "nan_policy"_a = "skip")
      .def("update", &RollingMinMax::push, "x"_a, "Push one observation.")
      .def(
          "update_many",
          [](RollingMinMax& s, nb::handle x) {
            const Input in = resolve_input(x, "x", 1);
            OutBuffer<double> lo(in);
            OutBuffer<double> hi(in);
            release_gil_if(in.size(), [&] {
              in.visit(0, [&](auto v) {
                for (std::size_t i = 0; i < v.size; ++i) {
                  s.push(v[i]);
                  lo.column(0)[i] = s.min();
                  hi.column(0)[i] = s.max();
                }
              });
            });
            return std::make_tuple(lo.release_to_numpy(), hi.release_to_numpy());
          },
          "x"_a,
          nb::sig("def update_many(self, x: numpy.typing.NDArray[typing.Any]) -> "
                  "tuple[numpy.typing.NDArray[numpy.float64], "
                  "numpy.typing.NDArray[numpy.float64]]"),
          "Push every element of a 1-D array; returns per-element (min, max).")
      .def("reset", &RollingMinMax::reset, "Forget every observation.")
      .def_prop_ro("min", &RollingMinMax::min)
      .def_prop_ro("max", &RollingMinMax::max)
      .def_prop_ro("count", &RollingMinMax::count)
      .def_prop_ro("ready", &RollingMinMax::ready)
      .def_prop_ro("window", [](const RollingMinMax& s) { return s.config().window; })
      .def_prop_ro("min_periods", [](const RollingMinMax& s) { return s.config().min_periods; })
      .def_prop_ro("nan_policy",
                   [](const RollingMinMax& s) { return nan_policy_name(s.config().nan); })
      .def("__getstate__",
           [](const RollingMinMax& s) {
             return to_dict(s.save(), rolling_config_dict(s.config(), false));
           })
      .def("__setstate__",
           [](RollingMinMax& self, const StateDict& d) {
             new (&self) RollingMinMax(rolling_config_from(checked(d)));
             self.load(from_dict(d));
           })
      .def("__repr__", [](const RollingMinMax& s) {
        const auto& c = s.config();
        return "RollingMinMax(window=" + std::to_string(c.window) +
               ", min_periods=" + std::to_string(c.min_periods) + ", nan_policy='" +
               nan_policy_name(c.nan) + "', count=" + std::to_string(s.count()) + ")";
      });
}

void bind_ols_class(nb::module_& m) {
  nb::class_<OlsStream>(m, "RollingOLS",
                        "Streaming rolling OLS of y on x with intercept (same semantics as "
                        "qkern.rolling_ols with include_current=True).")
      .def(
          "__init__",
          [](OlsStream* self, std::int64_t window, std::optional<std::int64_t> min_periods,
             const std::string& nan_policy) {
            new (self) OlsStream(make_rolling_config(window, min_periods, 1, nan_policy));
          },
          "window"_a, nb::kw_only(), "min_periods"_a = nb::none(), "nan_policy"_a = "skip")
      .def("update", &OlsStream::push, "y"_a, "x"_a, "Push one (y, x) pair.")
      .def(
          "update_many",
          [](OlsStream& s, nb::handle y, nb::handle x) {
            const Input yin = resolve_input(y, "y", 1);
            const Input xin = resolve_input(x, "x", 1);
            if (yin.rows != xin.rows) {
              throw std::invalid_argument("y and x must have the same length");
            }
            OutBuffer<double> alpha(yin);
            OutBuffer<double> beta(yin);
            OutBuffer<double> resid(yin);
            OutBuffer<double> r2(yin);
            release_gil_if(yin.size(), [&] {
              yin.visit(0, [&](auto yv) {
                xin.visit(0, [&](auto xv) {
                  for (std::size_t i = 0; i < yv.size; ++i) {
                    const double yi = yv[i];
                    const double xi = xv[i];
                    s.push(yi, xi);
                    alpha.column(0)[i] = s.state.alpha();
                    beta.column(0)[i] = s.state.beta();
                    r2.column(0)[i] = s.state.r2();
                    resid.column(0)[i] = s.state.resid(yi, xi);
                  }
                });
              });
            });
            return result_type("OLSResult")(alpha.release_to_numpy(), beta.release_to_numpy(),
                                            resid.release_to_numpy(), r2.release_to_numpy());
          },
          "y"_a, "x"_a,
          nb::sig("def update_many(self, y: numpy.typing.NDArray[typing.Any], "
                  "x: numpy.typing.NDArray[typing.Any]) -> qkern._results.OLSResult"),
          "Push every pair of two 1-D arrays; returns the per-element OLSResult.")
      .def(
          "reset",
          [](OlsStream& s) {
            s.state.reset();
            s.last_y = s.last_x = kNaNValue;
          },
          "Forget every observation.")
      .def_prop_ro("alpha", [](const OlsStream& s) { return s.state.alpha(); })
      .def_prop_ro("beta", [](const OlsStream& s) { return s.state.beta(); })
      .def_prop_ro("r2", [](const OlsStream& s) { return s.state.r2(); })
      .def_prop_ro("resid", [](const OlsStream& s) { return s.state.resid(s.last_y, s.last_x); })
      .def_prop_ro("count", [](const OlsStream& s) { return s.state.count(); })
      .def_prop_ro("ready", [](const OlsStream& s) { return s.state.ready(); })
      .def_prop_ro("window", [](const OlsStream& s) { return s.state.config().window; })
      .def_prop_ro("min_periods", [](const OlsStream& s) { return s.state.config().min_periods; })
      .def_prop_ro("nan_policy",
                   [](const OlsStream& s) { return nan_policy_name(s.state.config().nan); })
      .def("__getstate__",
           [](const OlsStream& s) {
             StateDict d = to_dict(s.state.save(), rolling_config_dict(s.state.config(), false));
             d["last"] = std::make_tuple(s.last_y, s.last_x);
             return d;
           })
      .def("__setstate__",
           [](OlsStream& self, const StateDict& d) {
             new (&self) OlsStream(rolling_config_from(checked(d)));
             self.state.load(from_dict(d));
             const auto last = nb::cast<std::tuple<double, double>>(d["last"]);
             self.last_y = std::get<0>(last);
             self.last_x = std::get<1>(last);
           })
      .def("__repr__", [](const OlsStream& s) {
        const auto& c = s.state.config();
        return "RollingOLS(window=" + std::to_string(c.window) +
               ", min_periods=" + std::to_string(c.min_periods) + ", nan_policy='" +
               nan_policy_name(c.nan) + "', count=" + std::to_string(s.state.count()) + ")";
      });
}

}  // namespace

void bind_streaming(nb::module_& m) {
  bind_rolling_moments(m);
  bind_ewm_class(m);
  bind_minmax_class(m);
  bind_ols_class(m);
}

}  // namespace qkern::bindings
