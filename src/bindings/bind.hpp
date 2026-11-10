// Registration functions, one per bind_*.cpp file.
#pragma once

#include <nanobind/nanobind.h>

namespace qkern::bindings {

void bind_rolling(nanobind::module_& m);

}  // namespace qkern::bindings
