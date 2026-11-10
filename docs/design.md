# Design

How qkern answers the three questions its README asks. Measured results are filled in from
the reference benchmark run (see [Benchmarks](benchmarks.md)).

## Copies

The binding layer (`src/bindings/ndarray_utils.hpp`) accepts any read-only array through
nanobind's `ndarray` and turns it into a strided view: a pointer, a length and a stride in
elements, which may be negative. float64 and float32 inputs are never copied, whatever their
stride; integer and bool inputs are converted once and the conversion is counted. Arrow
arrays arrive through the PyCapsule interface (`include/qkern/arrow_import.hpp`): the values
buffer is read in place and nulls are read from the validity bitmap. Results are allocated
in C++ and handed to NumPy with a capsule that frees them, so returning them copies nothing;
`out=` writes into the caller's array after a byte-range overlap check.

Evidence: `tests/python/test_zero_copy.py` checks the binding layer's debug counters for
every layout and Arrow input (AC-W-07).

## GIL

A batch call releases the GIL when it touches at least 1,024 elements
(`kGilReleaseMinN`). All argument conversion happens before the release; the kernel itself
touches no Python object. Smaller calls keep the GIL, because releasing and re-acquiring it
costs more than the work. `update()` on a streaming object always keeps it.

Evidence: `tests/python/test_gil.py` runs a counting thread during a call: it advances
thousands of times with the release on and not at all with it forced off. The thread-scaling
suite measures what the release buys in throughput (NFR-05).

## ABI

One wheel serves CPython 3.12 and later: the extension is compiled against the limited C API
(`STABLE_ABI` in `CMakeLists.txt`, `wheel.py-api = "cp312"` in `pyproject.toml`). CPython 3.11
gets its own version-specific wheel. The CMake call forwards scikit-build-core's
`SKBUILD_SABI_COMPONENT`, without which nanobind silently builds a version-specific
extension inside an abi3-tagged wheel. `wheels.yml` therefore checks the file name, runs
`abi3audit --strict`, and tests the one abi3 wheel on 3.12, 3.13 and 3.14.

## Numerics

The rolling moments use a sliding Welford update on shifted data with a compensated sliding
sum for the mean, and an exact two-pass recomputation every 65,536 rows or whenever the sum
of squares collapses (after a large value leaves the window). See
[Semantics](semantics.md#numerical-accuracy) for the targets and their tests.
