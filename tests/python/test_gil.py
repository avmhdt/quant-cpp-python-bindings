"""The GIL is released in long batch calls (AC-W-08, FR-14): a Python thread keeps running
while the kernel works, and stops when release is forced off."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable, Iterator

import numpy as np
import numpy.typing as npt
import pytest

import qkern
from qkern import _qkern

F64 = npt.NDArray[np.float64]
N = 10_000_000


@pytest.fixture(scope="module")
def big() -> F64:
    return np.random.default_rng(0).standard_normal(N)


@pytest.fixture
def ticker() -> Iterator[list[int]]:
    """A background thread that counts as fast as it can until stopped."""
    count = [0]
    stop = threading.Event()

    def run() -> None:
        while not stop.is_set():
            count[0] += 1

    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(0.05)
    yield count
    stop.set()
    t.join()


def _advance_during(count: list[int], call: Callable[[], object]) -> int:
    before = count[0]
    call()
    return count[0] - before


def test_release_lets_other_threads_run(big: F64, ticker: list[int]) -> None:
    assert _qkern._get_gil_release_min_n() == 1024
    advance = _advance_during(ticker, lambda: qkern.rolling_std(big, 252))
    assert advance >= 1000


def test_forced_off_blocks_other_threads(big: F64, ticker: list[int]) -> None:
    # A 1 s switch interval keeps CPython from handing the GIL to the counter between the
    # two reads unless the extension itself releases it.
    old = sys.getswitchinterval()
    try:
        _qkern._set_gil_release_min_n(2**62)
        sys.setswitchinterval(1.0)
        # Let any switch request made under the old interval complete now; when this thread
        # gets the GIL back it holds it for up to 1 s, far longer than the call.
        time.sleep(0.01)
        advance = _advance_during(ticker, lambda: qkern.rolling_std(big, 252))
    finally:
        sys.setswitchinterval(old)
        _qkern._set_gil_release_min_n(1024)
    assert advance < 50


def test_update_many_releases_too(big: F64, ticker: list[int]) -> None:
    obj = qkern.EWM(span=20)
    advance = _advance_during(ticker, lambda: obj.update_many(big))
    assert advance >= 1000
