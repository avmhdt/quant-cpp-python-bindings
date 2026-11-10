"""Data access for real-workload benchmarks (spec section 7e). Nothing here is committed."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np

from tools.qkdev import fixtures
from tools.qkdev.settings import Settings

log = logging.getLogger("qkdev.data")

DATASET_COLUMNS: dict[str, tuple[str, ...]] = {
    "trades": ("ts_exchange", "price", "qty", "side"),
    "bbo": ("ts", "bid_px", "bid_qty", "ask_px", "ask_qty"),
    "book_l2": (
        "ts_exchange",
        "ts_recv",
        "batch_id",
        "seq",
        "side",
        "price",
        "qty",
        "is_snapshot",
    ),
}


def list_partitions(root: Path) -> list[Path]:
    """Canonical partitions under ``root``: <dataset>/venue=*/symbol=*/date=*."""
    return sorted(p for p in root.glob("*/venue=*/symbol=*/date=*") if p.is_dir())


def smoke(settings: Settings) -> int:
    """M1 data smoke test: one fixture round trip, then list real partitions if configured."""
    fx = fixtures.FIXTURES[0]
    with tempfile.TemporaryDirectory() as tmp:
        fixtures.make(Path(tmp))
        loaded = fixtures.load(fx.name, Path(tmp))
        regenerated = fx.build(np.random.default_rng(fx.seed), fx.params)
        if not np.array_equal(loaded["x"], regenerated["x"]):
            log.error("fixture round trip failed for %s", fx.name)
            return 1
    log.info("fixture round trip ok: %s (%d values)", fx.name, loaded["x"].shape[0])
    root = settings.qkern_real_data_dir
    if root is None:
        log.info("QKERN_REAL_DATA_DIR is not set: real-workload data skipped")
        return 0
    if not root.is_dir():
        log.error("QKERN_REAL_DATA_DIR=%s is not a directory", root)
        return 1
    parts = list_partitions(root)
    log.info("QKERN_REAL_DATA_DIR=%s: %d partitions", root, len(parts))
    for p in parts[:20]:
        log.info("  %s", p.relative_to(root))
    return 0


def import_real(settings: Settings, dataset: str, venue: str, symbol: str, date: str) -> int:
    """Convert one canonical partition to ``$DATA_ROOT/processed/<...>/<column>.npy``."""
    root = settings.qkern_real_data_dir
    if root is None:
        log.info("skipped: QKERN_REAL_DATA_DIR is not set")
        return 0
    if dataset not in DATASET_COLUMNS:
        log.error("unknown dataset %r; expected one of %s", dataset, sorted(DATASET_COLUMNS))
        return 2
    import pyarrow.parquet as pq  # dev/bench dependency only

    part = root / dataset / f"venue={venue}" / f"symbol={symbol}" / f"date={date}"
    files = sorted(part.glob("part-*.parquet"))
    if not files:
        log.error("no part-*.parquet files in %s", part)
        return 1
    columns = list(DATASET_COLUMNS[dataset])
    table = pq.read_table(files, columns=columns)
    out_dir = settings.data_root / "processed" / f"{dataset}__{venue}__{symbol}__{date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in columns:
        np.save(out_dir / f"{name}.npy", table.column(name).to_numpy())
    if dataset == "trades":
        price = table.column("price").to_numpy().astype(np.float64)
        log_return = np.full(price.shape, np.nan)
        log_return[1:] = np.diff(np.log(price))
        np.save(out_dir / "log_return.npy", log_return)
    log.info("wrote %d rows to %s", table.num_rows, out_dir)
    return 0
