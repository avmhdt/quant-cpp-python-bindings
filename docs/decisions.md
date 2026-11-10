# Decisions

Copied at M1 from the owner's `DECISIONS.md` (recorded 2026-10-31, fourth batch 2026-11-09),
which overrides the spec. Agent decisions on points the spec leaves open follow. No secrets
or keys ever go in this file.

## Owner decisions that apply here

- **Project:** #15 `quant-cpp-python-bindings`, started 2026-11-09 in the fourth batch with
  #6, #12, #13 and #14. All 15 projects are now started; the pace is a few at a time with no
  fixed date.
- **Merging:** agents push feature branches and open PRs, one milestone per PR. PRs merge with
  `gh pr merge <n> --auto --squash` as `avmhdt` once CI is green. If a permission check refuses
  the merge, the PR is reported as ready for the owner. Branch protection is turned on once CI
  exists (M1).
- **GitHub:** `gh` active account `avmhdt`, admin on the repo, Actions enabled.
- **D-03 Python support (changed from the spec default):** Python 3.10 is dropped. Wheels are
  cp311 version-specific plus one cp312 abi3 wheel covering 3.12 and later.
- **D-02 licence:** MIT (default; the owner may override). Code only: licensed data is never
  redistributed.
- **Money:** nothing approved for this project (USD 0). Free GitHub-hosted runners only.
- **Machine:** shared PC. Builds use at most 8 threads at `nice 10` and stop if free memory
  drops below 30 GB. Downloads are capped at 10 MB/s. Never restart WSL. Data and build trees
  live in WSL ext4 (`~/build/quant-cpp-python-bindings`, `~/data/quant-cpp-python-bindings`),
  never on `C:` and never in git. D: stays above 150 GB free.
- **Benchmarks:** published benchmark runs wait until no multi-week run is active on this PC
  (after about 2026-11-29: crypto-lob-alpha's recorder and event-driven-backtester's 21-day
  testnet run, which ends 2026-11-28 20:10 UTC).
- **Publishing:** aggregates only. No Claude, Anthropic or AI attribution anywhere.
- **Accepted:** every other owner decision in spec section 3 with a default stands as written.

## Prerequisites

| ID | Item | Status |
|---|---|---|
| P-01 | Toolchain | Done by the owner (`setup/wsl-toolchain.sh`). Recorded 2026-11-09 in WSL2 Ubuntu 22.04.5: g++-13 13.4.0, clang++-18 18.1.8, clang-format-18 and clang-tidy-18 18.1.8, CMake 4.4.4, Ninja 1.13.2, gcovr 8.8, uv 0.12.23, CPython 3.11.17 / 3.12.15 / 3.13.16 from uv. Host: Intel Core i9-14900K, 62 GB RAM visible to WSL, 32 logical CPUs. |
| P-02 | Actions and admin | Done. Main is protected at M1 by a ruleset that requires the aggregate `ci` check (see below). |
| P-03 | PyPI name | Checked 2026-11-09: `https://pypi.org/pypi/qkern/json`, `https://test.pypi.org/pypi/qkern/json`, `https://pypi.org/simple/qkern/` and `https://test.pypi.org/simple/qkern/` all return 404, so `qkern` is unused. The alternates `rollkern` and `quantkern` also return 404 on PyPI. PyPI can still reject a name as too similar to an existing project, so the name is only confirmed by the first TestPyPI upload (M5). |
| P-07 to P-13 | Release setup | Owner steps, needed before M5. |
| P-15, P-16 | Consumer merges, upstream target | Owner steps, needed before M6. |

## Spec decisions

| ID | Decision | Status |
|---|---|---|
| D-01 | Name | `qkern` for both the distribution and the import (P-03 above). |
| D-02 | Licence | MIT. `LICENSE`, `license = "MIT"` and `license-files` in `pyproject.toml`. |
| D-03 | Python support | Owner change: cp311 plus cp312 abi3; CI tests 3.11, 3.12, 3.13 and 3.14. |
| D-04 | Free-threaded wheels | Default: later. |
| D-05 | Extra platforms | Default: skip for 0.x. |
| D-06 | Manual approval before PyPI | Default: required reviewer (the owner) on the `pypi` environment. |
| D-07 | Arrow input | Default (a), C++ PyCapsule consumer; decided at M4. |
| D-08 | Docs | MkDocs Material on GitHub Pages. |
| D-09 | First version | Default: 0.1.0 at the end of M5 via TestPyPI then PyPI; 0.2.0 after M6. |
| D-10 | Upstream target | Default Polars, fallback hftbacktest; the owner picks before M6. |
| D-11 | Reference hardware | Default: the owner's machine, after the multi-week runs end. |
| D-12 | Paid spend | None. |

## Agent decisions

### M1 (2026-11-09)

- **Pinned versions.** `build-system.requires` pins `scikit-build-core==1.1.2` and
  `nanobind==2.15.0`. The spec says to pin the latest nanobind 2.x at M1; 3.0 (2026-08-22)
  and 3.1 exist, but the spec's instruction is explicit and nanobind 2.15 has everything the
  spec uses (ndarray, stable ABI, stubgen). Moving to 3.x is a later, separate decision.
  GoogleTest v1.18.0 through CPM.cmake v0.43.2 (vendored in `cmake/CPM.cmake`). Python tools
  are locked in `uv.lock`.
- **Version source.** `pyproject.toml` holds the version. CMake reads it with a regex and
  generates `qkern/version.hpp` (`QKERN_VERSION_*`); `qkern.__version__` comes from the
  compiled extension, and a test checks it equals the installed distribution's metadata.
- **Floating-point flags.** `-ffp-contract=off` (GCC/Clang) and `/fp:precise` (MSVC) are
  INTERFACE usage requirements of `qkern::core`, because results of the header-only kernels
  depend on how the consumer's translation unit is compiled (DQ-08). Warnings and `-Werror`
  apply only to this repo's own targets; nanobind, Python and GoogleTest headers are system
  headers.
- **Stubs.** `src/qkern/_qkern.pyi` is generated by `python -m nanobind.stubgen -P -m
  qkern._qkern` from the installed extension (`tools.qkdev stubs check [--write]`); `-P`
  keeps the private test hooks. The wheel ships this committed file, which CI checks against
  a fresh generation, instead of installing a build-tree copy: the two would collide in the
  wheel, and cross-compiled wheels (macOS x86_64 on arm64) cannot import the module at build
  time. `nanobind_add_stub` stays available in CMake behind `QKERN_STUBGEN` (off by default).
- **stubtest deferred to M3.** `mypy.stubtest qkern._qkern` reports every nanobind function
  as "not a function" with runtime signature `(*args, **kwargs)`, because nanobind function
  objects do not expose signatures to `inspect`. M1 checks the stub with `mypy --strict` and
  the drift check; how to satisfy AC-W-09's stubtest clause is settled at M3.
- **Required check.** The ruleset requires one aggregate job, `ci`, which fails unless every
  other job in `ci.yml` (lint, 4 C++ jobs, 10 Python jobs, abi3 spike) succeeded. This is the
  spec's "require status checks ci / \*" in a form that does not need editing when jobs
  change.
- **CI runners.** `macos-latest` instead of the spec's `macos-14` for the AppleClang and
  Python jobs, because older macOS images are retired on a schedule; the version in use is
  shown in each job log. gitleaks runs as the pinned v8.30.1 release binary, verified by
  SHA-256, rather than through a third-party action.
- **clang-tidy.** The enabled families are the spec's. Two style-only sub-checks are off:
  `bugprone-easily-swappable-parameters` (flags every `(window, min_periods)` pair) and
  `modernize-use-trailing-return-type`. The `QKERN_VERSION_*` macros carry a NOLINT for
  `modernize-macro-to-enum`, since the spec requires macros. Third-party targets
  (nanobind's static library, GoogleTest) are not linted.
- **Presets.** Beyond the spec's list: `gcc13` and `clang18` (the dev configuration with an
  explicit compiler, used by CI and locally) and `tidy`. `msvc` uses the default Visual
  Studio generator, so it works whichever Visual Studio the runner has.
- **Fixtures.** Written as uncompressed `.npz` with fixed zip timestamps, so regeneration
  is byte-identical across machines (DQ-09); about 0.6 MB in total. The BBO and book_l2
  fixtures come with the order-flow kernels (FR-07, M4).
- **Dependency groups.** `dev` also holds `nanobind` (for stubgen) and `pandas-stubs` (for
  `mypy --strict` on the tests). `[tool.uv] default-groups = ["dev"]`.
- **Input dtypes in M1.** The binding exposes float64 and float32 overloads with
  `noconvert()`, so those are read in place; the Python layer converts other real dtypes
  once to float64. Counting that conversion in `_debug_counters()` arrives with the full
  dtype dispatch in M3.
- **Data root.** The `data/` symlink to `$DATA_ROOT` is created when real-workload data is
  first needed (M4); M1's data smoke test reads nothing from disk except fixtures.

### abi3 spike (M1)

- Local, WSL2 with GCC 13: `uv build --wheel --python 3.12` produced
  `qkern-0.1.0-cp312-abi3-linux_x86_64.whl` containing `qkern/_qkern.abi3.so`;
  `abi3audit --strict` found 0 non-abi3 symbols (baseline 3.12). The 3.11 build produced a
  version-specific `cpython-311` extension, as intended. `twine check --strict` passed for the
  wheel and the sdist, and `pip install .` into a fresh 3.12 venv imported and ran.
- CI, cibuildwheel v4.3.5 on manylinux_2_28 x86_64 (run 42725465145, PR #1): the build
  linked `_qkern.abi3.so`, auditwheel produced
  `qkern-0.1.0-cp312-abi3-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl`, cibuildwheel's
  own `abi3audit --strict` step and the job's filename assert passed, and the test suite
  passed inside the wheel's environment. R-02's fallback (version-specific wheels for every
  Python) is not needed.

### M2 (2026-11-09)

- **Shifted accumulation.** The sliding Welford sums are kept for `x - shift`, where the
  shift is the first value entering an empty window and moves to the exact window mean at
  every resync. The plain form missed NFR-06 (b) on `1e9 + N(0, 1)` data (relative variance
  error 7.4e-6 at w = 252, against 1e-6), because the running mean near 1e9 drifts by
  ulp(1e9) ≈ 1.2e-7 per update between resyncs. With the shift the error is within the
  bound, and `x - shift` is exact (Sterbenz) while x stays within a factor of two of the
  shift. Rolling OLS uses the same scheme for x and y. The algorithm is otherwise the
  spec's: add/remove, M2 clamped at 0, exact two-pass resync every 65,536 pushes.
- **Compensated mean and collapse resync.** The sliding mean comes from a Neumaier
  compensated sum (add and remove), so its error stays near eps × |mean| instead of
  eps × (largest value that passed through the window). Besides the periodic resync, the
  moments (M2) and OLS (Sxx, Syy) recompute exactly as soon as the co-moment falls below
  `kCollapseRatio = 1e-6` of its peak since the last resync: after a large value leaves the
  window, the sliding update keeps an absolute error of a few ulp of the old, larger value,
  which would otherwise dominate a much smaller variance (NFR-06 (d) on data up to 1e6, and
  the first M2 CI run, where differently drawn data made the OLS reference test miss on
  MSVC and AppleClang). The extra O(w) passes happen only after such collapses. The spec's
  periodic resync interval is unchanged.
- **State save/load.** Each streaming class has `State save() const` and `load(const State&)`
  (window contents oldest first, plus integer and real fields). A restored object continues
  bit-identically; malformed states throw `std::invalid_argument`. The Python layer pickles
  this with a version field (M3).
- **Portable test data.** The C++ tests draw Gaussians by Box-Muller on `std::mt19937_64`,
  whose output the standard fixes, instead of `std::normal_distribution`, whose output
  differs between libstdc++, libc++ and the MSVC STL.
- **Constant windows.** A window whose finite values are all equal has mean exactly that
  value and variance exactly 0 (z-score NaN), tracked by a run-length counter. pandas does
  the same; without it add/remove rounding leaves a variance near 1e-30 and z-scores of
  order 1e14. Rolling OLS applies it to x (Sxx = 0, outputs NaN) and y (Syy = Sxy = 0,
  beta 0, r2 NaN).
- **Conditioning of sliding co-moments.** Between resyncs the sliding sums carry an absolute
  error near 1e-14 times the data scale, so ratios (beta, r2) lose relative accuracy in
  windows whose Sxx or Syy is tiny (for example two nearly equal points). The C++ OLS test
  scales its tolerance by 1e-12 (1/Sxx + 1/Syy). The Python property tests against pandas
  (NFR-06 (d), M3) need the same care; any tolerance choice there goes into this log.
- **EWM.** One `Ewm` state drives `ewm_mean`, `ewm_var` and `ewm_std`. The update is pandas'
  `ewmcov` recurrence with x = y, adjust=False and ignore_na=True, in the same operation
  order, so the mean equals pandas' `ewm(...).mean()` update too. `nan_policy="propagate"`
  has no window to leave, so the first NaN makes all later outputs NaN until `reset()`;
  `min_periods = 0` behaves as 1. Infinities follow the recurrence (mean ±inf, then NaN once
  both signs were seen; variance NaN from then on), as in pandas.
- **Rolling OLS with include_current=False.** alpha, beta and r2 at t all come from W_{t-1},
  not only the residual, so no output at t uses row t (DQ-01).
- **Internal header.** `include/qkern/detail.hpp` (beyond the spec's layout) holds shared
  constants and helpers: `kResyncInterval`, `kNaN`, `kInf`, the run-length counter and a
  Neumaier compensated sum used by the resyncs. It is not part of the public API.
- **Coverage and sanitizer scope at M2.** The C++ coverage gate covers `include/qkern` from
  the GoogleTest run (98.8% of lines locally). `src/bindings` and the pytest run join it at
  M3, when the binding layer is complete, as does the ASan/UBSan pytest run. The coverage
  build uses `-fprofile-update=atomic`, because the 8-thread concurrency test otherwise
  corrupts gcov counters (gcovr reported negative hit counts).
- **Required checks.** The ruleset now requires `sanitizers` (aggregate of the `asan` and
  `tsan` jobs) beside `ci`, as AC-W-02 asks for both.
- **Benchmarks.** `qkern_bench` (Google Benchmark v1.10.0) is built by the `dev` preset family
  and the `bench` preset; its numbers are informational. Function names are CamelCase so
  `--benchmark_filter=Rolling` works; they carry a NOLINT for the naming check.
- **TSan locally.** WSL2's kernel uses high-entropy ASLR like the CI runners. Locally the
  concurrency tests run under `setarch -R` (no sudo needed); CI lowers `vm.mmap_rnd_bits`.

### M3 (2026-11-09)

- **Input dispatch.** Every batch function takes one generic read-only ndarray. float64 and
  float32 of any stride are read in place; integer and bool dtypes are converted once to a
  column-major float64 buffer in C++ and counted in `_debug_counters()["input_conversions"]`
  (DQ-05); float16 and long double are converted by the Python layer (not counted: they are
  outside DQ-05's integer/bool rule and rare). Everything else is a TypeError.
- **2-D layout.** 2-D inputs `(n, k)` in any order are processed column by column; results
  are column-major (Fortran order), because each column is computed contiguously. `rolling_ols`
  accepts `y (n, k)` with `x (n, k)` or `x (n,)`, the market-beta case of the consumer table.
- **out= returns.** The extension's `out=` variants return None; the Python wrapper returns
  `out` itself, so `qkern.f(x, ..., out=o) is o`.
- **Argument checks in one place.** Range checks (window, min_periods, ddof, alpha, span,
  halflife, nan_policy) live in C++ (`src/bindings/config.hpp`) and are shared by the batch
  functions and the streaming classes; the Python layer checks only Python types (bool or
  float `window` is a TypeError).
- **Streaming classes.** `RollingMoments`, `EWM`, `RollingMinMax`, `RollingOLS` are the C++
  state classes exported directly. `RollingMinMax.update_many` and `EWM.update_many` return
  `(min, max)` and `(mean, var)` tuples (the spec leaves these open); `RollingOLS.update_many`
  matches `rolling_ols(..., include_current=True)` and `.resid` is the last pushed pair's
  residual against the current fit. Pickling uses `{"v": 1, "config", "buffer", "ints",
  "reals"}` from the C++ `save()`; `copy.copy` and `copy.deepcopy` go through the same path.
- **pandas as a test oracle.** pandas' sliding sums keep rounding residue after a large value
  leaves the window (std 2.8e-7 for the window `[0, 0, 0]` after 46 left it, found by
  hypothesis). NFR-06 (d) is checked as: within 1e-9 · max(1, |pandas|) of pandas, or, where
  pandas is further from the exact `math.fsum` reference than qkern, within the same
  tolerance of that reference. This is not a weaker tolerance; it picks the right oracle when
  pandas is the inaccurate one, and the difference is listed on `docs/semantics.md`.
- **Ill-conditioned OLS windows.** Where a window's x values differ only in their last bits
  (Sxx below 1e-8 of the window's sum of x²), Sxx is rounding noise and no two algorithms,
  the fsum reference included, agree on beta. The property test against the reference skips
  those rows; NaN placement is still checked on every row.
- **pandas covariance in tests.** pandas computes rolling covariance as E[xy] − E[x]E[y]; the
  beta comparison against pandas uses rows where x's rolling variance exceeds 1e-3, where
  that formula is accurate, and tolerance 1e-6. The exact reference test covers OLS fully.
- **Golden files.** 44 cases (7 univariate kernels × 6 fixtures, plus rolling OLS on the pairs
  fixture with and without `include_current`) on the first 1,000 rows of each fixture,
  0.6 MB in total, with pandas cross-checks in `tests/golden/manifest.json`.
- **GIL test timing.** With release forced off, the test raises CPython's switch interval to
  1 s around the measurement so that an interpreter-level switch between the two counter reads
  cannot be mistaken for a release by the extension.
- **Python reference kernels.** `qkern.testing.reference` now covers every Must kernel (window
  recomputation with `math.fsum`, EWM from explicit weights) with qkern's semantics, as an
  oracle independent of pandas for downstream tests too.
- **stubtest (AC-W-09).** `mypy.stubtest qkern._qkern` runs in the lint job with
  `tests/stubtest_allowlist.txt`, which lists only what nanobind objects cannot satisfy:
  module-level functions are `nb_func` objects with no `inspect` signature (stubtest:
  "not a function", runtime `(*args, **kwargs)`), and the four classes use nanobind's
  metaclass and are disjoint bases. The function signatures in the stub come from nanobind's
  stubgen and are drift-checked by `qkdev stubs check`; stubtest still fails on any name
  present on one side only. This settles the M1 deferral.
- **Stub post-processing.** nanobind writes a dtype-less ndarray parameter as bare `NDArray`,
  which `mypy --strict` rejects; `qkdev stubs` rewrites it to `NDArray[Any]` as part of
  generation. The streaming `update_many` methods and the pickling methods carry explicit
  signatures (`nb::sig`, typed dicts) so the stub names `Moments`, `OLSResult` and typed
  tuples instead of `object`.
- **r2 formula.** r2 is computed as (Sxy/Sxx)·(Sxy/Syy) rather than Sxy²/(Sxx·Syy), which
  underflowed to 0/0 on data near 1e-100 (found by hypothesis at the CI profile).
- **Moments.count.** The spec's field name `count` shadows `tuple.count` on `Moments`; kept
  as specified, with a type-ignore on that field.

