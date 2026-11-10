# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the
version is 0.x, a minor version bump may break the API; every break is listed under
"Changed" or "Removed".

## [Unreleased]

### Added

- Project scaffolding: header-only C++20 core target `qkern::core`, CMake presets, nanobind
  extension `qkern._qkern` built with scikit-build-core, committed type stubs.
- `qkern.rolling_mean` (sliding Welford mean, NaN/inf/`min_periods` semantics, `out=`,
  float32 and strided inputs without copies, GIL released for n >= 1024).
- `qkern.build_info()` and `qkern.show_config()`.
- `qkern.testing.reference.rolling_mean`, a pure-Python reference kernel.
- Developer CLI `python -m tools.qkdev` (`fixtures make`, `data smoke`, `data import-real`,
  `stubs check`) and seeded synthetic fixtures.
- CI on Linux, macOS and Windows: lint, C++ tests, Python tests on 3.11 to 3.14, and an
  abi3 wheel spike.
- C++ kernel core: rolling mean, variance, standard deviation, moments and z-score (sliding
  Welford on shifted data with exact resync), EWM mean/variance/std (pandas `ewmcov`
  recurrence, adjust=False), rolling min/max (monotonic deques) and rolling OLS (bivariate
  sliding Welford), as streaming state classes that the batch kernels drive.
- `cmake --install` exports `qkern::core` with `qkernConfig.cmake`; `examples/cpp_consumer`
  builds against it with `find_package` or CPM.
- Google Benchmark binary `qkern_bench`; ASan/UBSan and TSan CI jobs; C++ coverage gate.
