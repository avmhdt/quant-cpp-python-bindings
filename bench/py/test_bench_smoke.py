"""Benchmark smoke test for CI (informational): small sizes through pytest-benchmark, and
one end-to-end `bench run --quick` + `bench report` into a temporary directory (AC-W-13
machinery). Fails only on errors, never on timings. Needs the `bench` dependency group."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import qkern
from tools.qkdev import bench
from tools.qkdev.settings import Settings

X = np.random.default_rng(0).standard_normal(100_000)


@pytest.mark.parametrize(
    "fn",
    [
        lambda: qkern.rolling_std(X, 252),
        lambda: qkern.rolling_zscore(X, 252),
        lambda: qkern.ewm_var(X, span=252),
        lambda: qkern.rolling_max(X, 252),
        lambda: qkern.rolling_median(X, 252),
        lambda: qkern.rolling_ols(X, X[::-1].copy(), 252),
    ],
    ids=["std", "zscore", "ewm_var", "max", "median", "ols"],
)
def test_kernels(benchmark: Any, fn: Any) -> None:
    benchmark(fn)


def test_run_and_report_end_to_end(tmp_path: Path) -> None:
    settings = Settings(qkern_bench_max_threads=2)
    path = bench.run(
        settings, "all", Path("bench/config.toml"), tmp_path, quick=True, allow_dirty=True
    )
    data = json.loads(path.read_text())
    assert {r["suite"] for r in data["results"]} >= {"micro", "overhead", "threads"}
    assert not [r for r in data["results"] if r.get("error")], "an impl failed its output check"
    docs = tmp_path / "benchmarks.md"
    bench.report(path, docs, tmp_path / "fig", allow_dirty=True)
    text = docs.read_text()
    assert "Single-thread throughput" in text
    assert (tmp_path / "fig" / "threads_scaling.svg").exists()
    assert (tmp_path / "bench_log.csv").read_text().count("\n") == 2
