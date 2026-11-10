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
- CI, cibuildwheel v4.3.5 on manylinux_2_28 x86_64: see the `abi3-spike` job of the M1 PR.
