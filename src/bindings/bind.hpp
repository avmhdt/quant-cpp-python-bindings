// Registration functions, one per bind_*.cpp file.
#pragma once

#include <nanobind/nanobind.h>

namespace qkern::bindings {

void bind_rolling(nanobind::module_& m);
void bind_ewm(nanobind::module_& m);
void bind_minmax(nanobind::module_& m);
void bind_ols(nanobind::module_& m);
void bind_streaming(nanobind::module_& m);

}  // namespace qkern::bindings
