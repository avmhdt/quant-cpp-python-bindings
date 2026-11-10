"""Deterministic synthetic fixtures (spec section 7, DQ-09).

Each fixture is ``tests/fixtures/series_<name>.npz`` with ``x``, optionally ``y``, and a
0-d string array ``meta`` holding JSON. Files are written as uncompressed zip archives with
fixed timestamps, so regenerating from the same seeds is byte-identical.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

FIXTURE_VERSION = 1
FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)

Arrays = dict[str, npt.NDArray[np.float64]]


@dataclass(frozen=True)
class Fixture:
    name: str
    seed: int
    params: dict[str, float | int]
    build: Callable[[np.random.Generator, dict[str, float | int]], Arrays]


def _gauss(rng: np.random.Generator, p: dict[str, float | int]) -> Arrays:
    return {"x": rng.standard_normal(int(p["n"]))}


def _offset(rng: np.random.Generator, p: dict[str, float | int]) -> Arrays:
    return {"x": float(p["offset"]) + rng.standard_normal(int(p["n"]))}


def _nan_runs(rng: np.random.Generator, p: dict[str, float | int]) -> Arrays:
    n = int(p["n"])
    x = rng.standard_normal(n)
    isolated = rng.choice(n, size=int(n * float(p["isolated_frac"])), replace=False)
    x[isolated] = np.nan
    for _ in range(int(p["runs"])):
        length = int(rng.integers(1, int(p["max_run"]) + 1))
        start = int(rng.integers(0, n - length))
        x[start : start + length] = np.nan
    return {"x": x}


def _inf_spikes(rng: np.random.Generator, p: dict[str, float | int]) -> Arrays:
    n = int(p["n"])
    x = rng.standard_normal(n)
    spikes = rng.choice(n, size=2 * int(p["spikes"]), replace=False)
    half = int(p["spikes"])
    x[spikes[:half]] = np.inf
    x[spikes[half:]] = -np.inf
    # One adjacent +inf/-inf pair so short windows see both signs at once.
    mid = n // 2
    x[mid], x[mid + 1] = np.inf, -np.inf
    return {"x": x}


def _constant_segments(rng: np.random.Generator, p: dict[str, float | int]) -> Arrays:
    n = int(p["n"])
    parts: list[npt.NDArray[np.float64]] = []
    total = 0
    while total < n:
        length = int(rng.integers(1, int(p["max_segment"]) + 1))
        if rng.random() < 0.5:
            seg = np.full(length, float(rng.standard_normal()))
        else:
            seg = rng.standard_normal(length)
        parts.append(seg)
        total += length
    return {"x": np.concatenate(parts)[:n]}


def _student_t(rng: np.random.Generator, p: dict[str, float | int]) -> Arrays:
    # Z / sqrt(chi2_df / df), with chi2_df as a sum of df squared normals. Only sqrt is
    # used, which IEEE 754 rounds exactly, so the bytes do not depend on the platform's libm
    # (numpy's standard_t calls log/exp and differs in the last bit on macOS).
    n, df = int(p["n"]), int(p["df"])
    z = rng.standard_normal(n)
    chi2 = np.square(rng.standard_normal((df, n))).sum(axis=0)
    return {"x": z / np.sqrt(chi2 / df)}


def _pairs(rng: np.random.Generator, p: dict[str, float | int]) -> Arrays:
    n = int(p["n"])
    split = int(p["regime_change_at"])
    x = rng.standard_normal(n)
    noise = rng.standard_normal(n) * float(p["noise"])
    beta = np.where(np.arange(n) < split, float(p["beta_1"]), float(p["beta_2"]))
    return {"x": x, "y": beta * x + noise}


FIXTURES: tuple[Fixture, ...] = (
    Fixture("gauss_10k", 101, {"n": 10_000}, _gauss),
    Fixture("offset_1e9_10k", 102, {"n": 10_000, "offset": 1e9}, _offset),
    Fixture(
        "nan_runs_10k",
        103,
        {"n": 10_000, "isolated_frac": 0.05, "runs": 8, "max_run": 300},
        _nan_runs,
    ),
    Fixture("inf_spikes_5k", 104, {"n": 5_000, "spikes": 10}, _inf_spikes),
    Fixture("constant_segments_5k", 105, {"n": 5_000, "max_segment": 500}, _constant_segments),
    Fixture("student_t3_10k", 106, {"n": 10_000, "df": 3}, _student_t),
    Fixture(
        "pairs_corr_10k",
        107,
        {"n": 10_000, "beta_1": 0.8, "beta_2": 1.5, "noise": 0.3, "regime_change_at": 6_000},
        _pairs,
    ),
)


def _npy_bytes(arr: npt.NDArray[np.generic]) -> bytes:
    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.asanyarray(arr), allow_pickle=False)
    return buf.getvalue()


def npz_bytes(members: dict[str, npt.NDArray[np.generic]]) -> bytes:
    """Deterministic .npz bytes: uncompressed, sorted members, fixed timestamps."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
        for key in sorted(members):
            info = zipfile.ZipInfo(f"{key}.npy", date_time=_ZIP_DATE)
            info.external_attr = 0o644 << 16
            info.create_system = 3  # Unix; ZipInfo defaults to 0 on Windows
            zf.writestr(info, _npy_bytes(members[key]))
    return out.getvalue()


def fixture_bytes(fx: Fixture) -> bytes:
    """Serialise one fixture to deterministic .npz bytes."""
    arrays = fx.build(np.random.default_rng(fx.seed), fx.params)
    meta = json.dumps(
        {
            "generator": f"tools.qkdev.fixtures:{fx.build.__name__}",
            "seed": fx.seed,
            "params": fx.params,
            "qkern_fixture_version": FIXTURE_VERSION,
        },
        sort_keys=True,
    )
    members: dict[str, npt.NDArray[np.generic]] = {**arrays, "meta": np.array(meta)}
    return npz_bytes(members)


def fixture_path(fx: Fixture, directory: Path = FIXTURE_DIR) -> Path:
    return directory / f"series_{fx.name}.npz"


def make(directory: Path = FIXTURE_DIR, *, check: bool = False) -> list[str]:
    """Write all fixtures, or with ``check`` return the names whose files differ."""
    directory.mkdir(parents=True, exist_ok=True)
    mismatched: list[str] = []
    for fx in FIXTURES:
        data = fixture_bytes(fx)
        path = fixture_path(fx, directory)
        if check:
            if not path.exists() or path.read_bytes() != data:
                mismatched.append(fx.name)
        else:
            path.write_bytes(data)
    return mismatched


def load(name: str, directory: Path = FIXTURE_DIR) -> dict[str, npt.NDArray[np.generic]]:
    """Load a fixture by name (without the ``series_`` prefix)."""
    with np.load(directory / f"series_{name}.npz", allow_pickle=False) as data:
        return {k: data[k] for k in data.files}
