// Rolling quantile: batch kernel and the RollingQuantile streaming class (FR-06), plus the
// _ArrowInput holder used by every kernel for Arrow inputs (FR-12).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "arrow_input.hpp"
#include "bind.hpp"
#include "config.hpp"
#include "ndarray_utils.hpp"
#include "qkern/rolling_quantile.hpp"

namespace qkern::bindings {

namespace {

using namespace nb::literals;
using StateDict = nb::typed<nb::dict, nb::str, nb::object>;

const char* interpolation_name(Interpolation i) noexcept {
  switch (i) {
    case Interpolation::Lower:
      return "lower";
    case Interpolation::Higher:
      return "higher";
    case Interpolation::Nearest:
      return "nearest";
    case Interpolation::Midpoint:
      return "midpoint";
    case Interpolation::Linear:
    default:
      return "linear";
  }
}

std::optional<NumpyF64> quantile_fn(nb::handle x, std::int64_t window,
                                    std::optional<std::int64_t> min_periods, double q,
                                    const std::string& interpolation, const std::string& nan_policy,
                                    const std::optional<AnyOut>& out) {
  const RollingConfig cfg = make_rolling_config(window, min_periods, 1, nan_policy);
  validate_quantile(q);
  const Interpolation interp = parse_interpolation(interpolation);
  return run_unary(x, out, [&](auto in, OutView o) { rolling_quantile(in, o, cfg, q, interp); });
}

RollingQuantile make_quantile(std::int64_t window, double q, const std::string& interpolation,
                              std::optional<std::int64_t> min_periods,
                              const std::string& nan_policy) {
  return {make_rolling_config(window, min_periods, 1, nan_policy), (validate_quantile(q), q),
          parse_interpolation(interpolation)};
}

}  // namespace

void bind_quantile(nb::module_& m) {
  nb::class_<ArrowInput>(m, "_ArrowInput",
                         "Zero-copy view of a float64/float32 Arrow array (PyCapsule interface).")
      .def_static(
          "from_array",
          [](const nb::object& schema, const nb::object& array) {
            auto* in = new ArrowInput();  // NOLINT(*-owning-memory): handed to Python
            try {
              in->init_from_array(schema, array);
            } catch (...) {
              delete in;  // NOLINT(*-owning-memory)
              throw;
            }
            return in;
          },
          "schema"_a, "array"_a, nb::rv_policy::take_ownership,
          "Wrap the capsules returned by __arrow_c_array__().")
      .def_static(
          "from_stream",
          [](const nb::object& stream) {
            auto* in = new ArrowInput();  // NOLINT(*-owning-memory): handed to Python
            try {
              in->init_from_stream(stream);
            } catch (...) {
              delete in;  // NOLINT(*-owning-memory)
              throw;
            }
            return in;
          },
          "stream"_a, nb::rv_policy::take_ownership,
          "Wrap the single chunk of the stream returned by __arrow_c_stream__().")
      .def("__len__", [](const ArrowInput& a) { return a.values().length; });

  m.def("rolling_quantile", &quantile_fn, "x"_a, "window"_a, "min_periods"_a, "q"_a,
        "interpolation"_a, "nan_policy"_a, "out"_a.none(),
        "Rolling quantile (see qkern.rolling_quantile). Returns None when out is given.");

  nb::class_<RollingQuantile>(m, "RollingQuantile",
                              "Streaming rolling quantile (two heaps with lazy deletion; same "
                              "semantics as qkern.rolling_quantile).")
      .def(
          "__init__",
          [](RollingQuantile* self, std::int64_t window, double q, const std::string& interpolation,
             std::optional<std::int64_t> min_periods, const std::string& nan_policy) {
            new (self)
                RollingQuantile(make_quantile(window, q, interpolation, min_periods, nan_policy));
          },
          "window"_a, "q"_a, nb::kw_only(), "interpolation"_a = "linear",
          "min_periods"_a = nb::none(), "nan_policy"_a = "skip")
      .def("update", &RollingQuantile::push, "x"_a, "Push one observation.")
      .def(
          "update_many",
          [](RollingQuantile& s, nb::handle x) {
            const Input in = resolve_input(x, "x", 1);
            OutBuffer<double> out(in);
            release_gil_if(in.size(), [&] {
              in.visit(0, [&](auto v) {
                for (std::size_t i = 0; i < v.size; ++i) {
                  s.push(v[i]);
                  out.column(0)[i] = s.value();
                }
              });
            });
            return out.release_to_numpy();
          },
          "x"_a,
          nb::sig("def update_many(self, x: numpy.typing.NDArray[typing.Any]) -> "
                  "numpy.typing.NDArray[numpy.float64]"),
          "Push every element of a 1-D array; returns the per-element quantile.")
      .def("reset", &RollingQuantile::reset, "Forget every observation.")
      .def_prop_ro("value", &RollingQuantile::value)
      .def_prop_ro("count", &RollingQuantile::count)
      .def_prop_ro("ready", &RollingQuantile::ready)
      .def_prop_ro("q", &RollingQuantile::quantile)
      .def_prop_ro("interpolation",
                   [](const RollingQuantile& s) { return interpolation_name(s.interpolation()); })
      .def_prop_ro("window", [](const RollingQuantile& s) { return s.config().window; })
      .def_prop_ro("min_periods", [](const RollingQuantile& s) { return s.config().min_periods; })
      .def_prop_ro("nan_policy",
                   [](const RollingQuantile& s) { return nan_policy_name(s.config().nan); })
      .def("__getstate__",
           [](const RollingQuantile& s) {
             StateDict d;
             nb::dict c;
             c["window"] = s.config().window;
             c["min_periods"] = s.config().min_periods;
             c["nan_policy"] = nan_policy_name(s.config().nan);
             c["q"] = s.quantile();
             c["interpolation"] = interpolation_name(s.interpolation());
             d["v"] = 1;
             d["config"] = c;
             d["buffer"] = s.save().buffer;
             return d;
           })
      .def("__setstate__",
           [](RollingQuantile& self, const StateDict& d) {
             if (!d.contains("v") || nb::cast<std::int64_t>(d["v"]) != 1) {
               throw std::invalid_argument("unsupported qkern state version (expected v=1)");
             }
             const auto c = nb::cast<nb::dict>(d["config"]);
             new (&self) RollingQuantile(make_quantile(
                 nb::cast<std::int64_t>(c["window"]), nb::cast<double>(c["q"]),
                 nb::cast<std::string>(c["interpolation"]),
                 nb::cast<std::int64_t>(c["min_periods"]), nb::cast<std::string>(c["nan_policy"])));
             State st;
             st.buffer = nb::cast<std::vector<double>>(d["buffer"]);
             self.load(st);
           })
      .def("__repr__", [](const RollingQuantile& s) {
        const auto& c = s.config();
        return "RollingQuantile(window=" + std::to_string(c.window) +
               ", q=" + std::to_string(s.quantile()) + ", interpolation='" +
               interpolation_name(s.interpolation()) +
               "', min_periods=" + std::to_string(c.min_periods) + ", nan_policy='" +
               nan_policy_name(c.nan) + "', count=" + std::to_string(s.count()) + ")";
      });
}

}  // namespace qkern::bindings
