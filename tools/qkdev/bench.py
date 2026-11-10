"""Benchmark runner and report generator (FR-24 to FR-26, spec sections 7b and 10).

``bench run`` times qkern against the comparators in ``bench/config.toml``. Every
implementation's output is checked against qkern (rtol 1e-9, atol 1e-12) before it is timed;
a comparator that fails the check is recorded with ``median_s = null`` and an error, never
timed. Results go to ``<out>/<YYYYMMDD>-<host>-<sha7>.json`` and one row is appended to
``bench_log.csv`` for every run (the honest trial count).

``bench report`` turns one results file into ``docs/benchmarks.md`` and SVG figures.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import statistics
import subprocess
import threading
import time
import tomllib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from tools.qkdev.settings import REPO_ROOT, Settings

log = logging.getLogger("qkdev.bench")

F64 = npt.NDArray[np.float64]
SUITES = ("micro", "overhead", "threads", "real")
LOG_FIELDS = [
    "timestamp_utc",
    "git_sha",
    "dirty",
    "host",
    "suite",
    "config_hash",
    "n_results",
    "json_path",
    "notes",
]


# ---------------------------------------------------------------------------------------
# Environment and metadata
# ---------------------------------------------------------------------------------------


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _physical_cores() -> int | None:
    try:
        out = subprocess.run(
            ["lscpu", "-p=core,socket"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    cores = {line for line in out.splitlines() if line and not line.startswith("#")}
    return len(cores) or None


def _ram_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        return None


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def metadata(settings: Settings, config_hash: str) -> dict[str, Any]:
    import qkern

    sha = _git("rev-parse", "--short=7", "HEAD") or "unknown"
    return {
        "host": settings.qkern_bench_host_label or platform.node(),
        "cpu_model": _cpu_model(),
        "physical_cores": _physical_cores(),
        "logical_cores": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "versions": {
            d: _version(d) for d in ("qkern", "numpy", "pandas", "numba", "polars", "bottleneck")
        },
        "build": qkern.build_info(),
        "git_sha": sha,
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_hash": config_hash,
    }


# ---------------------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------------------


@dataclass
class Timing:
    median_s: float
    iqr_s: float
    min_s: float


def _time(fn: Callable[[], object], repeats: int) -> Timing:
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    q = statistics.quantiles(samples, n=4) if len(samples) > 1 else [samples[0]] * 3
    return Timing(statistics.median(samples), q[2] - q[0], min(samples))


def _max_abs_diff(got: F64, want: F64) -> float:
    if got.shape != want.shape or np.any(np.isnan(got) != np.isnan(want)):
        return float("inf")
    ok = np.isfinite(want)
    return float(np.max(np.abs(got[ok] - want[ok]), initial=0.0))


def _agrees(got: F64, want: F64) -> bool:
    if got.shape != want.shape or np.any(np.isnan(got) != np.isnan(want)):
        return False
    ok = np.isfinite(want)
    return bool(np.all(np.abs(got[ok] - want[ok]) <= 1e-12 + 1e-9 * np.abs(want[ok])))


# ---------------------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------------------


def _qkern_kernels() -> dict[str, Callable[..., F64]]:
    import qkern

    return {
        "rolling_mean": lambda x, w: qkern.rolling_mean(x, w),
        "rolling_std": lambda x, w: qkern.rolling_std(x, w),
        "rolling_zscore": lambda x, w: qkern.rolling_zscore(x, w),
        "ewm_mean": lambda x, w: qkern.ewm_mean(x, span=w),
        "ewm_var": lambda x, w: qkern.ewm_var(x, span=w),
        "rolling_max": lambda x, w: qkern.rolling_max(x, w),
        "rolling_ols": lambda y, x, w: qkern.rolling_ols(y, x, w).beta,
        "rolling_quantile": lambda x, w: qkern.rolling_median(x, w),
    }


def _qkern_stream() -> dict[str, Callable[..., F64]]:
    import qkern

    return {
        "rolling_mean": lambda x, w: qkern.RollingMoments(w).update_many(x).mean,
        "rolling_std": lambda x, w: np.sqrt(qkern.RollingMoments(w).update_many(x).var),
        "ewm_mean": lambda x, w: qkern.EWM(span=w).update_many(x)[0],
        "ewm_var": lambda x, w: qkern.EWM(span=w).update_many(x)[1],
        "rolling_max": lambda x, w: qkern.RollingMinMax(w).update_many(x)[1],
        "rolling_ols": lambda y, x, w: qkern.RollingOLS(w).update_many(y, x).beta,
        "rolling_quantile": lambda x, w: qkern.RollingQuantile(w, 0.5).update_many(x),
    }


def _series(n: int, seed: int, layout: str) -> F64:
    rng = np.random.default_rng(seed)
    if layout == "strided2":
        return rng.standard_normal(2 * n)[::2]
    return rng.standard_normal(n)


def run_micro(cfg: dict[str, Any], quick: bool) -> list[dict[str, Any]]:
    from bench.py.baselines import LOADERS

    d, s = cfg["defaults"], cfg["suites"]["micro"]
    limits = s["limits"]
    ns = [n for n in s["n"] if n <= 100_000] if quick else s["n"]
    windows = s["window"][:2] if quick else s["window"]
    repeats = 3 if quick else d["repeats"]
    impls: dict[str, dict[str, Callable[..., F64]]] = {
        "qkern": _qkern_kernels(),
        "qkern_stream": _qkern_stream(),
    }
    for name in s["impls"]:
        if name in LOADERS:
            try:
                impls[name] = LOADERS[name]()
            except ImportError as exc:
                log.warning("skipping %s: %s", name, exc)
    results = []
    for kernel in s["kernels"]:
        for n in ns:
            for w in windows:
                if w > n:
                    continue
                for layout in s["layout"]:
                    x = _series(n, d["seed"], layout)
                    args: tuple[Any, ...] = (x, w)
                    if kernel == "rolling_ols":
                        y = 0.5 * x + _series(n, d["seed"] + 1, layout)
                        args = (y, x, w)
                    want = impls["qkern"][kernel](*args)
                    for impl, table in impls.items():
                        fn = table.get(kernel)
                        if fn is None:
                            continue
                        if impl == "pure_python" and n > limits["pure_python_max_n"]:
                            continue
                        if impl == "qkern_stream" and n > limits["qkern_stream_max_n"]:
                            continue
                        if (
                            impl == "numpy"
                            and kernel in ("rolling_std", "rolling_max", "rolling_quantile")
                            and n * w * 8 > limits["numpy_sliding_view_max_bytes"]
                        ):
                            continue
                        results.append(
                            _measure(
                                kernel, impl, fn, args, want, n, w, layout, repeats, d["warmup"]
                            )
                        )
    return results


def _measure(
    kernel: str,
    impl: str,
    fn: Callable[..., F64],
    args: tuple[Any, ...],
    want: F64,
    n: int,
    w: int,
    layout: str,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "suite": "micro",
        "kernel": kernel,
        "impl": impl,
        "n": n,
        "window": w,
        "dtype": "float64",
        "layout": layout,
        "threads": 1,
        "gil_release": True if impl.startswith("qkern") else None,
        "repeats": repeats,
    }
    t0 = time.perf_counter()
    got = np.asarray(fn(*args), dtype=np.float64)
    first = time.perf_counter() - t0
    for _ in range(max(warmup - 1, 0)):
        fn(*args)
    row["compile_s"] = first if impl == "numba" else None
    row["max_abs_diff_vs_qkern"] = _max_abs_diff(got, want)
    if not _agrees(got, want):
        row.update(
            median_s=None,
            iqr_s=None,
            min_s=None,
            throughput_melem_s=None,
            error="output differs from qkern beyond rtol 1e-9 / atol 1e-12",
        )
        log.warning("%s/%s n=%d w=%d: output check failed", kernel, impl, n, w)
        return row
    t = _time(lambda: fn(*args), repeats)
    row.update(
        median_s=t.median_s, iqr_s=t.iqr_s, min_s=t.min_s, throughput_melem_s=n / t.median_s / 1e6
    )
    return row


def _batch_call(arr: F64) -> Callable[[], object]:
    import qkern

    def call() -> object:
        return qkern.rolling_mean(arr, 1)

    return call


def run_overhead(cfg: dict[str, Any], quick: bool) -> list[dict[str, Any]]:
    import qkern

    iters = 100_000 if quick else int(cfg["suites"]["overhead"]["iterations"])
    obj = qkern.RollingMoments(20)
    update = obj.update
    arrays = {k: np.random.default_rng(0).standard_normal(k) for k in (1, 10, 100)}
    calls: dict[str, Callable[[], object]] = {"update_scalar": lambda: update(1.0)}
    for k, arr in arrays.items():
        calls[f"batch_n{k}"] = _batch_call(arr)
    results = []
    for name in cfg["suites"]["overhead"]["calls"]:
        call = calls[name]
        for _ in range(1000):
            call()
        t0 = time.perf_counter()
        for _ in range(iters):
            call()
        per_call = (time.perf_counter() - t0) / iters
        results.append(
            {
                "suite": "overhead",
                "kernel": name,
                "impl": "qkern",
                "n": iters,
                "window": None,
                "dtype": "float64",
                "layout": "contig",
                "threads": 1,
                "gil_release": None,
                "repeats": 1,
                "median_s": per_call,
                "iqr_s": None,
                "min_s": per_call,
                "throughput_melem_s": None,
                "compile_s": None,
                "max_abs_diff_vs_qkern": 0.0,
            }
        )
    return results


def run_threads(cfg: dict[str, Any], settings: Settings, quick: bool) -> list[dict[str, Any]]:
    import qkern
    from qkern import _qkern

    s = cfg["suites"]["threads"]
    cap = settings.qkern_bench_max_threads or min(_physical_cores() or os.cpu_count() or 1, 8)
    counts = [t for t in s["threads"] if t <= cap]
    if quick:
        counts = [t for t in counts if t <= 2]
    n = 200_000 if quick else int(s["n_per_thread"])
    w = int(s["window"])
    kernels = {
        "rolling_std": lambda a: qkern.rolling_std(a, w),
        "rolling_quantile": lambda a: qkern.rolling_median(a, w),
    }
    data = [np.random.default_rng(i).standard_normal(n) for i in range(max(counts))]
    results = []
    for name in s["kernels"]:
        fn = kernels[name]
        for release in s["gil_release"]:
            _qkern._set_gil_release_min_n(1024 if release else 2**62)
            try:
                for t in counts:
                    barrier = threading.Barrier(t)

                    def work(
                        i: int,
                        b: threading.Barrier = barrier,
                        f: Callable[[F64], object] = fn,
                    ) -> None:
                        b.wait()
                        f(data[i])

                    timing = _time(partial(_run_pool, work, t), 3 if quick else 5)
                    results.append(
                        {
                            "suite": "threads",
                            "kernel": name,
                            "impl": "qkern",
                            "n": n,
                            "window": w,
                            "dtype": "float64",
                            "layout": "contig",
                            "threads": t,
                            "gil_release": release,
                            "repeats": 3,
                            "median_s": timing.median_s,
                            "iqr_s": timing.iqr_s,
                            "min_s": timing.min_s,
                            "throughput_melem_s": t * n / timing.median_s / 1e6,
                            "compile_s": None,
                            "max_abs_diff_vs_qkern": 0.0,
                        }
                    )
            finally:
                _qkern._set_gil_release_min_n(1024)
    return results


def _run_pool(work: Callable[[int], None], t: int) -> None:
    with ThreadPoolExecutor(max_workers=t) as pool:
        list(pool.map(work, range(t)))


def run_real(cfg: dict[str, Any], settings: Settings) -> list[dict[str, Any]]:
    import qkern

    if settings.qkern_real_data_dir is None:
        log.info("real suite skipped: QKERN_REAL_DATA_DIR is not set")
        return []
    processed = settings.data_root / "processed"
    results = []
    for path in sorted(processed.glob("trades__*/log_return.npy")):
        x = np.load(path)
        for w in cfg["suites"]["real"]["window"]:
            for kernel, fn in (
                ("rolling_std", qkern.rolling_std),
                ("rolling_zscore", qkern.rolling_zscore),
            ):
                t = _time(partial(fn, x, w, min_periods=1), 5)
                results.append(
                    {
                        "suite": "real",
                        "kernel": kernel,
                        "impl": "qkern",
                        "n": int(x.size),
                        "window": w,
                        "dtype": "float64",
                        "layout": "contig",
                        "threads": 1,
                        "gil_release": True,
                        "repeats": 5,
                        "median_s": t.median_s,
                        "iqr_s": t.iqr_s,
                        "min_s": t.min_s,
                        "throughput_melem_s": x.size / t.median_s / 1e6,
                        "compile_s": None,
                        "max_abs_diff_vs_qkern": 0.0,
                        "dataset": path.parent.name.split("__")[0],
                    }
                )
    return results


def run(
    settings: Settings, suite: str, config: Path, out: Path, *, quick: bool, allow_dirty: bool
) -> Path:
    raw = config.read_bytes()
    cfg = tomllib.loads(raw.decode())
    for key, value in cfg["defaults"]["single_thread_env"].items():
        os.environ[key] = str(value)
    meta = metadata(settings, hashlib.sha1(raw).hexdigest()[:12])
    if meta["dirty"] and not allow_dirty:
        raise SystemExit("working tree is dirty; commit first or pass --allow-dirty")
    meta["quick"] = quick
    suites = SUITES if suite == "all" else (suite,)
    results: list[dict[str, Any]] = []
    for name in suites:
        log.info("suite %s (quick=%s)", name, quick)
        if name == "micro":
            results += run_micro(cfg, quick)
        elif name == "overhead":
            results += run_overhead(cfg, quick)
        elif name == "threads":
            results += run_threads(cfg, settings, quick)
        elif name == "real":
            results += run_real(cfg, settings)
    out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d")
    path = out / f"{stamp}-{meta['host']}-{meta['git_sha']}.json"
    path.write_text(json.dumps({"meta": meta, "results": results}, indent=1) + "\n")
    log_path = out / "bench_log.csv"
    new = not log_path.exists()
    with log_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": meta["utc"],
                "git_sha": meta["git_sha"],
                "dirty": meta["dirty"],
                "host": meta["host"],
                "suite": suite,
                "config_hash": meta["config_hash"],
                "n_results": len(results),
                "json_path": path.name,
                "notes": "quick" if quick else "",
            }
        )
    log.info("wrote %s (%d results)", path, len(results))
    return path


# ---------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------


def _fmt(v: float | None, digits: int = 1) -> str:
    return "-" if v is None else f"{v:,.{digits}f}"


def report(json_path: Path, docs: Path, figures: Path, *, allow_dirty: bool = False) -> None:
    data = json.loads(json_path.read_text())
    meta, rows = data["meta"], data["results"]
    if meta.get("dirty") and not allow_dirty:
        raise SystemExit(f"{json_path.name} came from a dirty tree; pass --allow-dirty")
    micro = [r for r in rows if r["suite"] == "micro" and r["median_s"] is not None]
    lines = [
        "# Benchmarks",
        "",
        "Generated by `uv run --group bench python -m tools.qkdev bench report` from "
        f"[`{json_path.name}`](https://github.com/avmhdt/quant-cpp-python-bindings/blob/main/results/bench/{json_path.name}).",
        "Do not edit by hand.",
        "",
        "## Environment",
        "",
        "| | |",
        "|---|---|",
    ]
    env = {
        "Host": meta["host"],
        "CPU": meta["cpu_model"],
        "Cores (physical / logical)": f"{meta['physical_cores']} / {meta['logical_cores']}",
        "RAM (GB)": meta["ram_gb"],
        "OS": meta["os"],
        "Python": meta["python"],
        "qkern build": f"{meta['build'].get('compiler')} {meta['build'].get('compiler_version')}, "
        f"{meta['build'].get('build_type')}, {meta['build'].get('fp_flags')}",
        "Git": f"{meta['git_sha']}{' (dirty)' if meta['dirty'] else ''}",
        "UTC": meta["utc"],
        "Comparator threads": "1 (OMP, Numba and Polars pinned to one thread)",
    }
    for k, v in env.items():
        lines.append(f"| {k} | {v} |")
    vers = ", ".join(f"{k} {v}" for k, v in meta["versions"].items() if v)
    lines += ["", f"Versions: {vers}.", "", "## Single-thread throughput (million elements/s)", ""]
    if micro:
        big = max(r["n"] for r in micro)
        w = 252 if any(r["window"] == 252 for r in micro) else micro[0]["window"]
        impls = [
            i
            for i in (
                "qkern",
                "qkern_stream",
                "numpy",
                "pandas",
                "numba",
                "polars",
                "bottleneck",
                "pure_python",
            )
            if any(r["impl"] == i for r in micro)
        ]
        lines.append(
            f"n = {big:,}, window = {w}, contiguous float64. Speed-ups are qkern's "
            "batch time divided into the comparator's."
        )
        lines += [
            "",
            "| kernel | " + " | ".join(impls) + " | vs pandas | vs Numba |",
            "|---|" + "---|" * (len(impls) + 2),
        ]
        speedups: dict[str, float] = {}
        for kernel in dict.fromkeys(r["kernel"] for r in micro):
            sel = {
                r["impl"]: r
                for r in micro
                if r["kernel"] == kernel
                and r["n"] == big
                and r["window"] == w
                and r["layout"] == "contig"
            }
            if "qkern" not in sel:
                continue
            q = sel["qkern"]["median_s"]
            cells = [_fmt(sel[i]["throughput_melem_s"]) if i in sel else "-" for i in impls]
            vs_pd = sel["pandas"]["median_s"] / q if "pandas" in sel else None
            vs_nb = sel["numba"]["median_s"] / q if "numba" in sel else None
            if vs_pd:
                speedups[kernel] = vs_pd
            lines.append(
                f"| {kernel} | " + " | ".join(cells) + f" | {_fmt(vs_pd, 2)}x | {_fmt(vs_nb, 2)}x |"
            )
        _bar_svg(speedups, figures / "micro_speedup.svg", "qkern speed-up over pandas")
    over = [r for r in rows if r["suite"] == "overhead"]
    if over:
        lines += ["", "## Call overhead", "", "| call | time per call (ns) |", "|---|---|"]
        lines += [f"| {r['kernel']} | {_fmt(r['median_s'] * 1e9, 0)} |" for r in over]
        _bar_svg(
            {r["kernel"]: r["median_s"] * 1e9 for r in over},
            figures / "overhead.svg",
            "ns per call",
        )
    thr = [r for r in rows if r["suite"] == "threads"]
    if thr:
        lines += [
            "",
            "## Thread scaling (what releasing the GIL buys)",
            "",
            "| kernel | GIL released | threads | aggregate Melem/s | speed-up vs 1 thread |",
            "|---|---|---|---|---|",
        ]
        for r in thr:
            base = next(
                b
                for b in thr
                if b["kernel"] == r["kernel"]
                and b["gil_release"] == r["gil_release"]
                and b["threads"] == 1
            )
            lines.append(
                f"| {r['kernel']} | {r['gil_release']} | {r['threads']} | "
                f"{_fmt(r['throughput_melem_s'])} | "
                f"{_fmt(r['throughput_melem_s'] / base['throughput_melem_s'], 2)}x |"
            )
        _threads_svg(thr, figures / "threads_scaling.svg")
    failed = [r for r in rows if r.get("error")]
    lines += [
        "",
        "## Output checks",
        "",
        f"{len(rows) - len(failed)} of {len(rows)} measurements passed the pre-timing "
        "equality check against qkern.",
    ]
    lines += [
        f"- {r['kernel']} / {r['impl']} (n={r['n']}, w={r['window']}): {r['error']}" for r in failed
    ]
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run --group bench python -m tools.qkdev bench run --suite all",
        "uv run --group bench python -m tools.qkdev bench report --in "
        f"results/bench/{json_path.name}",
        "```",
        "",
    ]
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    log.info("wrote %s", docs)


def _svg_setup() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "qkern"
    return plt


def _bar_svg(values: dict[str, float], path: Path, ylabel: str) -> None:
    if not values:
        return
    plt = _svg_setup()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(list(values), list(values.values()), color="#3b6ea5")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


def _threads_svg(rows: list[dict[str, Any]], path: Path) -> None:
    plt = _svg_setup()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for kernel in dict.fromkeys(r["kernel"] for r in rows):
        for release in (True, False):
            sel = sorted(
                (r for r in rows if r["kernel"] == kernel and r["gil_release"] == release),
                key=lambda r: r["threads"],
            )
            if sel:
                ax.plot(
                    [r["threads"] for r in sel],
                    [r["throughput_melem_s"] for r in sel],
                    marker="o",
                    linestyle="-" if release else "--",
                    label=f"{kernel}, GIL {'released' if release else 'held'}",
                )
    ax.set_xlabel("Python threads")
    ax.set_ylabel("aggregate Melem/s")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)
