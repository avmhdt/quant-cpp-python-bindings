# Python/C++ bindings library

> A small, fast C++ core shipped as a real Python package with wheels, docs and CI.

**Status:** planning

| | |
|---|---|
| Track | Quant Developer |
| Estimated effort | 3–4 weeks |
| Stack | C++20, nanobind or pybind11, CMake, scikit-build-core |
| Data cost | Free |
| Strongest for | Every quant dev seat |

## Requirements

- [ ] A fast C++ core with a small, sharp scope: an order book, rolling statistics, or an options pricing toolkit.
- [ ] Bindings with nanobind or pybind11, zero-copy NumPy or Arrow buffers, and the GIL released during long calls.
- [ ] Builds with scikit-build-core and CMake; wheels for Linux, macOS and Windows via cibuildwheel.
- [ ] Tests in both languages, type stubs, a docs site (MkDocs or Sphinx), semantic versioning and a changelog.
- [ ] Benchmarks against pure Python, NumPy, Numba and Polars equivalents.

## Data

Raw and licensed data live in `data/`, which is git-ignored. Only download scripts and small synthetic fixtures are committed.

| Dataset | Cost | What it gives you |
|---|---|---|
| Data from your other projects | Free | Real workloads for benchmarks. |
| Synthetic fixtures | Free | Deterministic test inputs. |
| [nanobind and cibuildwheel](https://github.com/wjakob/nanobind) | Free | The binding and wheel-building tools themselves. |

## Success criteria

### Works (correct and reproducible)

- [ ] pip install from PyPI works on all three operating systems, and CI is green on every commit.
- [ ] High test coverage on the core and clean sanitizer runs.

### Impresses hirers

- [ ] Published on PyPI with real outside users, issues or stars.
- [ ] A benchmarks page with reproducible scripts.
- [ ] Your other portfolio projects depend on it.
- [ ] A merged pull request to an established project (QuantLib, Polars, nautilus_trader, hftbacktest).

## Questions this project must answer

- What does releasing the GIL actually buy you?
- How do you avoid copies across the language boundary?
- How do you keep ABI compatibility across Python versions?
