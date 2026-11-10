"""Smoke tests: the extension imports, metadata is sane, the public API is complete."""

from __future__ import annotations

import importlib.metadata
import inspect

import numpy as np
import pytest

import qkern
from qkern import _qkern

BATCH = [
    "rolling_mean",
    "rolling_var",
    "rolling_std",
    "rolling_moments",
    "rolling_zscore",
    "ewm_mean",
    "ewm_var",
    "ewm_std",
    "rolling_min",
    "rolling_max",
    "rolling_ols",
]
OTHER = [
    "RollingMoments",
    "EWM",
    "RollingMinMax",
    "RollingOLS",
    "Moments",
    "OLSResult",
    "BBO",
    "show_config",
    "build_info",
]


def test_version_matches_distribution_metadata() -> None:
    assert qkern.__version__ == importlib.metadata.version("qkern")
    assert qkern.build_info()["version"] == qkern.__version__


def test_build_info_fields() -> None:
    info = qkern.build_info()
    for key in (
        "version",
        "git_sha",
        "compiler",
        "compiler_version",
        "cxx_standard",
        "build_type",
        "fp_flags",
        "nanobind_version",
        "abi3",
        "python_build_version",
        "gil_release_min_n",
        "free_threaded",
    ):
        assert key in info
    assert info["gil_release_min_n"] == 1024
    assert isinstance(info["abi3"], bool)
    assert int(info["cxx_standard"]) >= 202002


def test_show_config_prints(capsys: pytest.CaptureFixture[str]) -> None:
    qkern.show_config()
    out = capsys.readouterr().out
    assert "version" in out
    assert "nanobind_version" in out


def test_public_api_is_exported_and_documented() -> None:
    for name in BATCH + OTHER:
        obj = getattr(qkern, name)
        assert name in qkern.__all__
        assert inspect.getdoc(obj), f"{name} has no docstring"


def test_batch_functions_have_examples() -> None:
    for name in BATCH:
        doc = inspect.getdoc(getattr(qkern, name)) or ""
        assert "Example:" in doc, name
        assert ">>>" in doc, name


def test_gil_threshold_hook_round_trips() -> None:
    assert _qkern._get_gil_release_min_n() == 1024
    try:
        _qkern._set_gil_release_min_n(2**62)
        assert _qkern._get_gil_release_min_n() == 2**62
        np.testing.assert_array_equal(qkern.rolling_mean(np.ones(5000), 3)[2:], 1.0)
    finally:
        _qkern._set_gil_release_min_n(1024)


def test_lists_are_accepted() -> None:
    got = qkern.rolling_mean([1.0, 2.0, 3.0], 2)  # type: ignore[arg-type]
    np.testing.assert_array_equal(got, [np.nan, 1.5, 2.5])


def test_build_info_rejects_unexpected_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_qkern, "_build_info", lambda: {"version": 1.5})
    with pytest.raises(TypeError, match="unexpected build_info value"):
        qkern.build_info()
