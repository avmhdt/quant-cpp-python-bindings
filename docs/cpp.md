# C++ API

The kernels are a header-only C++20 library, `qkern::core`, usable without Python.

```cmake
find_package(qkern 0.1 CONFIG REQUIRED)   # after `cmake --install`
target_link_libraries(app PRIVATE qkern::core)
```

or through CPM.cmake:

```cmake
CPMAddPackage(NAME qkern GITHUB_REPOSITORY avmhdt/quant-cpp-python-bindings GIT_TAG v0.1.0)
target_link_libraries(app PRIVATE qkern::core)
```

`qkern::core` carries `-ffp-contract=off` (GCC/Clang) or `/fp:precise` (MSVC) as a usage
requirement, so a consumer's results match the Python package's.

```cpp
#include "qkern/qkern.hpp"

std::vector<double> x = {1, 2, 4, 8, 16};
std::vector<double> sd(x.size());
qkern::rolling_std(qkern::StridedView<double>{x.data(), x.size(), 1},
                   qkern::OutView{sd.data(), sd.size(), 1},
                   qkern::RollingConfig{3, 2});

qkern::Ewm ewm(qkern::EwmConfig{qkern::alpha_from_span(20.0)});
for (double v : x) ewm.push(v);
```

| Header | Contents |
|---|---|
| `views.hpp` | `StridedView<T>` (any stride, optional Arrow validity bitmap), `OutView` |
| `nan_policy.hpp` | `NanPolicy`, `RollingConfig`, `EwmConfig`, validation |
| `rolling_moments.hpp` | `RollingMoments`; `rolling_mean/var/std/moments/zscore` |
| `ewm.hpp` | `Ewm`; `ewm_mean/var/std`; `alpha_from_span`, `alpha_from_halflife` |
| `rolling_minmax.hpp` | `RollingMinMax`; `rolling_min/max` |
| `rolling_ols.hpp` | `RollingOls`; `rolling_ols` |
| `rolling_quantile.hpp` | `RollingQuantile`, `Interpolation`; `rolling_quantile` |
| `arrow_import.hpp` | `import_arrow` for Arrow C Data Interface arrays |

Every state class has `push`, `reset`, `save` and `load`. Invalid configurations throw
`std::invalid_argument`. A runnable example is in `examples/cpp_consumer`.
