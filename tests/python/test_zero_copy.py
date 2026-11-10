"""Zero-copy in and out (AC-W-07, NFR-04): no input conversion for float64/float32 of any
stride, outputs owned by a capsule (not copied), out= returned as the same object."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import qkern
from qkern import _qkern


@pytest.fixture(autouse=True)
def _reset_counters() -> Iterator[None]:
    _qkern._reset_debug_counters()
    yield
    _qkern._reset_debug_counters()


def _counters() -> dict[str, int]:
    return dict(_qkern._debug_counters())


BASE = np.random.default_rng(0).standard_normal(3000)
LAYOUTS: dict[str, npt.NDArray[Any]] = {
    "f64_contig": BASE,
    "f32_contig": BASE.astype(np.float32),
    "f64_step3": BASE[::3],
    "f64_reversed": BASE[::-1],
    "f32_step2": BASE.astype(np.float32)[::2],
    "f64_column_of_c_order": BASE.reshape(1000, 3)[:, 1],
    "f64_readonly": np.frombuffer(BASE.tobytes(), dtype=np.float64),
}


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_no_input_copy_for_float_layouts(name: str) -> None:
    x = LAYOUTS[name]
    for fn in (qkern.rolling_mean, qkern.rolling_std, qkern.rolling_max, qkern.rolling_zscore):
        fn(x, 20)
    qkern.ewm_var(x, span=20)
    qkern.rolling_ols(x, x[::-1], 20)
    qkern.RollingMoments(20).update_many(x)
    assert _counters() == {"input_conversions": 0, "output_copies": 0}


def test_integer_input_is_converted_once_and_counted() -> None:
    qkern.rolling_mean(np.arange(100), 5)
    assert _counters()["input_conversions"] == 1
    qkern.rolling_mean(np.arange(100, dtype=np.int32)[::-1], 5)
    assert _counters()["input_conversions"] == 2


def test_output_wraps_the_cpp_buffer() -> None:
    x = np.ones(100)
    out = qkern.rolling_mean(x, 3)
    # NumPy does not own the data: it wraps the C++ buffer, kept alive through the base
    # chain (nanobind exports it via the buffer protocol; the capsule frees it).
    assert out.flags.owndata is False
    base = out.base
    assert base is not None
    assert not (isinstance(base, np.ndarray) and base.flags.owndata)
    assert out.flags.writeable
    assert not np.shares_memory(out, x)
    assert _counters()["output_copies"] == 0


def test_out_is_the_same_object_and_written_in_place() -> None:
    x = np.arange(50.0)
    out = np.full(50, -1.0)
    assert qkern.rolling_mean(x, 3, out=out) is out
    np.testing.assert_array_equal(out, qkern.rolling_mean(x, 3))
    strided = np.zeros(100)[::2]
    assert qkern.rolling_std(x, 3, out=strided) is strided
    two_d = np.zeros((25, 2), order="F")
    assert qkern.rolling_max(x.reshape(25, 2), 3, out=two_d) is two_d
    assert _counters() == {"input_conversions": 0, "output_copies": 0}


def test_two_d_input_not_copied() -> None:
    mat = np.random.default_rng(1).standard_normal((500, 4))
    for m in (mat, np.asfortranarray(mat), mat[::2, ::-1]):
        qkern.rolling_var(m, 10)
    assert _counters()["input_conversions"] == 0
