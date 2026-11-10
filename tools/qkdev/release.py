"""Release helpers (FR-22, FR-23, AC-W-12): version and changelog checks, release notes, and
the install-from-index smoke test. Agents prepare release PRs; only the owner pushes tags.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

from tools.qkdev.settings import REPO_ROOT

log = logging.getLogger("qkdev.release")

INDEXES = {
    "pypi": "https://pypi.org/simple/",
    "testpypi": "https://test.pypi.org/simple/",
}


def project_version() -> str:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def changelog_section(version: str) -> str | None:
    """The body of '## [version] - YYYY-MM-DD' in CHANGELOG.md, or None."""
    text = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    pattern = rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}\n(.*?)(?=^## \[|\Z)"
    match = re.search(pattern, text, flags=re.S | re.M)
    return match.group(1).strip() if match else None


def check(tag: str, *, require_clean: bool = True) -> int:
    problems = []
    version = project_version()
    if tag != f"v{version}":
        problems.append(f"tag {tag} does not match pyproject version {version}")
    if changelog_section(version) is None:
        problems.append(f"CHANGELOG.md has no dated section '## [{version}] - YYYY-MM-DD'")
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    if data["project"].get("license") != "MIT" or "LICENSE" not in data["project"].get(
        "license-files", []
    ):
        problems.append("license metadata missing (license = 'MIT', license-files)")
    if require_clean:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if status:
            problems.append("working tree has uncommitted changes")
    for p in problems:
        log.error("%s", p)
    if not problems:
        log.info("release %s: version, changelog and licence metadata consistent", tag)
    return 1 if problems else 0


def notes(version: str, output: Path) -> int:
    body = changelog_section(version)
    if body is None:
        log.error("no CHANGELOG section for %s", version)
        return 1
    output.write_text(body + "\n", encoding="utf-8")
    return 0


def smoke(index: str, version: str) -> int:
    """Install qkern==version from an index into a fresh venv and run tools/qkdev/smoke.py."""
    url = INDEXES[index]
    with tempfile.TemporaryDirectory() as tmp:
        venv = Path(tmp) / "venv"
        subprocess.run(["uv", "venv", "--python", sys.executable, str(venv)], check=True)
        py = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        cmd = [
            "uv",
            "pip",
            "install",
            "--python",
            str(py),
            "--no-cache",
            "--index-url",
            url,
            f"qkern=={version}",
        ]
        if index == "testpypi":
            # NumPy comes from PyPI; only qkern from TestPyPI.
            cmd[-1:-1] = [
                "--extra-index-url",
                INDEXES["pypi"],
                "--index-strategy",
                "unsafe-best-match",
            ]
        subprocess.run(cmd, check=True)
        result = subprocess.run(
            [str(py), str(REPO_ROOT / "tools" / "qkdev" / "smoke.py")], cwd=tmp, check=False
        )
    return result.returncode
