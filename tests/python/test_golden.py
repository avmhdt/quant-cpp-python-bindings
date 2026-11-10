"""Golden regression (spec section 11): kernel outputs on the committed fixtures match the
committed golden files within 1e-12 relative, on every platform and wheel (NFR-07)."""

from __future__ import annotations

import numpy as np
import pytest

from tools.qkdev import golden

CASES = golden._cases()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.stem)
def test_golden(case: golden.Case) -> None:
    path = golden.GOLDEN_DIR / f"{case.stem}.npz"
    assert path.exists(), "run: uv run python -m tools.qkdev fixtures golden --accept"
    outputs = case.run(golden._load_inputs(case.fixture))
    with np.load(path) as stored:
        assert sorted(stored.files) == sorted(outputs)
        for key, got in outputs.items():
            assert golden._close(got, stored[key]), f"{key} differs from {path.name}"


def test_manifest_lists_every_case() -> None:
    import json

    manifest = json.loads(golden.MANIFEST.read_text())
    assert sorted(manifest) == sorted(c.stem for c in CASES)
    for entry in manifest.values():
        assert entry["rows"] == golden.ROWS
