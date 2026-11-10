"""Type-stub drift check: regenerate _qkern.pyi with nanobind stubgen and compare (FR-19).

The generated stub gets one mechanical post-processing step: nanobind writes a dtype-less
ndarray parameter as bare ``NDArray``, which ``mypy --strict`` rejects, so it becomes
``NDArray[Any]``. Everything else is stubgen's output unchanged.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from tools.qkdev.settings import REPO_ROOT

log = logging.getLogger("qkdev.stubs")

COMMITTED = REPO_ROOT / "src" / "qkern" / "_qkern.pyi"

_BARE_NDARRAY = re.compile(r"\bNDArray\b(?!\[)")


def postprocess(text: str) -> str:
    """Give bare ``NDArray`` a type argument and make sure ``Any`` is imported."""
    lines = []
    for line in text.splitlines(keepends=True):
        if not line.startswith(("from ", "import ")):
            line = _BARE_NDARRAY.sub("NDArray[Any]", line)
        lines.append(line)
    out = "".join(lines)
    match = re.search(r"^from typing import (.+)$", out, flags=re.MULTILINE)
    if "NDArray[Any]" in out and (match is None or "Any" not in match.group(1).split(", ")):
        if match is None:
            out = "from typing import Any\n" + out
        else:
            names = sorted({*match.group(1).split(", "), "Any"})
            out = out.replace(match.group(0), "from typing import " + ", ".join(names))
    return out


def generate(output: Path) -> None:
    # -P keeps the private test hooks (_set_gil_release_min_n, ...) in the stub.
    cmd = [sys.executable, "-m", "nanobind.stubgen", "-q", "-P", "-m", "qkern._qkern"]
    subprocess.run([*cmd, "-o", str(output)], check=True)
    output.write_text(postprocess(output.read_text(encoding="utf-8")), encoding="utf-8")


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
