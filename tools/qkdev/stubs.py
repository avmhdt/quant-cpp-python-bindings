"""Type-stub drift check: regenerate _qkern.pyi with nanobind stubgen and compare (FR-19)."""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from tools.qkdev.settings import REPO_ROOT

log = logging.getLogger("qkdev.stubs")

COMMITTED = REPO_ROOT / "src" / "qkern" / "_qkern.pyi"


def generate(output: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "nanobind.stubgen",
            "-q",
            "-P",
            "-m",
            "qkern._qkern",
            "-o",
            str(output),
        ],
        check=True,
    )


def check(*, write: bool) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp) / "_qkern.pyi"
        generate(fresh)
        new = fresh.read_text(encoding="utf-8")
    old = COMMITTED.read_text(encoding="utf-8") if COMMITTED.exists() else ""
    if new == old:
        log.info("stubs up to date: %s", COMMITTED.relative_to(REPO_ROOT))
        return 0
    if write:
        COMMITTED.write_text(new, encoding="utf-8", newline="\n")
        log.info("wrote %s", COMMITTED.relative_to(REPO_ROOT))
        return 0
    log.error("%s is stale; run: uv run python -m tools.qkdev stubs check --write", COMMITTED)
    return 1
