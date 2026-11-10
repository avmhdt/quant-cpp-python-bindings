"""Golden regression files (spec section 7b): expected kernel outputs on the committed
fixtures, compared within 1e-12 relative on every platform (DQ-08, NFR-07).

``qkdev fixtures golden`` recomputes and compares; ``--accept`` rewrites the files and the
manifest (with a CHANGELOG note, by policy). Each entry records the largest difference from
pandas at generation time, where pandas uses the same semantics (finite inputs).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from tools.qkdev import fixtures

log = logging.getLogger("qkdev.golden")

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "tests" / "golden"
MANIFEST = GOLDEN_DIR / "manifest.json"
ROWS = 1000  # first rows of each fixture
RTOL = 1e-12

F64 = npt.NDArray[np.float64]
Outputs = dict[str, F64]


@dataclass(frozen=True)
class Case:
    kernel: str
    fixture: str
    params: dict[str, Any]
    run: Callable[[dict[str, F64]], Outputs]
    pandas: Callable[[dict[str, F64]], Outputs] | None

    @property
    def stem(self) -> str:
        digest = hashlib.sha1(json.dumps(self.params, sort_keys=True).encode()).hexdigest()[:8]
        return f"{self.kernel}__{self.fixture}__{digest}"


def _single(fn: Callable[[dict[str, F64]], F64]) -> Callable[[dict[str, F64]], Outputs]:
    def call(d: dict[str, F64]) -> Outputs:
        return {"out": fn(d)}

    return call


def _cases() -> list[Case]:
    import pandas as pd

    import qkern

    def roll(d: dict[str, F64], w: int) -> Any:
        return pd.Series(d["x"]).rolling(w)

    unary: list[tuple[str, dict[str, Any], Callable[..., F64], Callable[..., F64]]] = [
        (
            "rolling_mean",
            {"window": 20},
            lambda d: qkern.rolling_mean(d["x"], 20),
            lambda d: roll(d, 20).mean().to_numpy(),
        ),
        (
            "rolling_std",
            {"window": 20},
            lambda d: qkern.rolling_std(d["x"], 20),
            lambda d: roll(d, 20).std().to_numpy(),
        ),
        (
            "rolling_zscore",
            {"window": 20},
            lambda d: qkern.rolling_zscore(d["x"], 20),
            lambda d: ((pd.Series(d["x"]) - roll(d, 20).mean()) / roll(d, 20).std()).to_numpy(),
        ),
        (
            "rolling_max",
            {"window": 20},
            lambda d: qkern.rolling_max(d["x"], 20),
            lambda d: roll(d, 20).max().to_numpy(),
        ),
        (
            "rolling_min",
            {"window": 20},
            lambda d: qkern.rolling_min(d["x"], 20),
            lambda d: roll(d, 20).min().to_numpy(),
        ),
        (
            "ewm_mean",
            {"span": 20},
            lambda d: qkern.ewm_mean(d["x"], span=20),
            lambda d: (
                pd.Series(d["x"]).ewm(span=20, adjust=False, ignore_na=True).mean().to_numpy()
            ),
        ),
        (
            "ewm_var",
            {"span": 20},
            lambda d: qkern.ewm_var(d["x"], span=20),
            lambda d: pd.Series(d["x"]).ewm(span=20, adjust=False, ignore_na=True).var().to_numpy(),
        ),
    ]
    names = [
        "gauss_10k",
        "offset_1e9_10k",
        "nan_runs_10k",
        "inf_spikes_5k",
        "constant_segments_5k",
        "student_t3_10k",
    ]
    cases: list[Case] = []
    for kernel, params, run, pdf in unary:
        for name in names:
            cases.append(Case(kernel, name, params, _single(run), _single(pdf)))
    for include_current in (True, False):

        def run_ols(d: dict[str, F64], ic: bool = include_current) -> Outputs:
            r = qkern.rolling_ols(d["y"], d["x"], 60, include_current=ic)
            return {"alpha": r.alpha, "beta": r.beta, "resid": r.resid, "r2": r.r2}

        def pd_ols(d: dict[str, F64], ic: bool = include_current) -> Outputs:
            ys, xs = pd.Series(d["y"]), pd.Series(d["x"])
            beta = ys.rolling(60).cov(xs) / xs.rolling(60).var()
            return {"beta": (beta if ic else beta.shift(1)).to_numpy()}

        params = {"window": 60, "include_current": include_current}
        cases.append(Case("rolling_ols", "pairs_corr_10k", params, run_ols, pd_ols))
    return cases


def _load_inputs(name: str) -> dict[str, F64]:
    data = fixtures.load(name)
    return {k: np.asarray(v, dtype=np.float64)[:ROWS] for k, v in data.items() if k != "meta"}


def _max_abs_diff(a: Outputs, b: Outputs) -> float | None:
    diffs = []
    for key, want in b.items():
        got = a[key]
        both = np.isfinite(got) & np.isfinite(want)
        if np.any(np.isnan(got) != np.isnan(want)):
            return None  # semantics differ (inputs with infinities)
        if both.any():
            diffs.append(float(np.max(np.abs(got[both] - want[both]))))
    return max(diffs) if diffs else 0.0


def _close(a: F64, b: F64) -> bool:
    if a.shape != b.shape or np.any(np.isnan(a) != np.isnan(b)):
        return False
    inf = np.isinf(b)
    if np.any(a[inf] != b[inf]):
        return False
    ok = np.isfinite(b)
    return bool(np.all(np.abs(a[ok] - b[ok]) <= RTOL * np.maximum(1.0, np.abs(b[ok]))))


def run(*, accept: bool, kernel: str | None = None) -> int:
    import pandas as pd

    import qkern

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    failures: list[str] = []
    for case in _cases():
        if kernel is not None and case.kernel != kernel:
            continue
        inputs = _load_inputs(case.fixture)
        outputs = case.run(inputs)
        path = GOLDEN_DIR / f"{case.stem}.npz"
        if accept:
            path.write_bytes(fixtures.npz_bytes(dict(outputs)))
            pd_diff = _max_abs_diff(outputs, case.pandas(inputs)) if case.pandas else None
            manifest[case.stem] = {
                "kernel": case.kernel,
                "fixture": case.fixture,
                "params": case.params,
                "rows": ROWS,
                "qkern_version": qkern.__version__,
                "generated_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "pandas_version": pd.__version__,
                "max_abs_diff_vs_pandas": pd_diff,
            }
            continue
        if not path.exists():
            failures.append(f"{case.stem}: missing")
            continue
        with np.load(path) as stored:
            for key, got in outputs.items():
                if not _close(got, stored[key]):
                    failures.append(f"{case.stem}: {key} differs")
    if accept:
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", newline="\n")
        log.info("wrote %d golden files and %s", len(manifest), MANIFEST.name)
        return 0
    for f in failures:
        log.error("%s", f)
    if not failures:
        log.info("golden files match within %g relative", RTOL)
    return 1 if failures else 0
