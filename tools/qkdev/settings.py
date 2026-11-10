"""Settings read from the environment and the git-ignored .env (see .env.example)."""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    data_root: Path = Path("~/data/quant-cpp-python-bindings")
    qkern_real_data_dir: Path | None = None
    qkern_bench_out: Path = Path("results/bench")
    qkern_bench_max_threads: int | None = None
    qkern_bench_host_label: str | None = None

    @field_validator("qkern_real_data_dir", "qkern_bench_max_threads", mode="before")
    @classmethod
    def _empty_is_none(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("data_root", "qkern_real_data_dir", mode="after")
    @classmethod
    def _expand(cls, value: Path | None) -> Path | None:
        return None if value is None else value.expanduser()


def load_settings() -> Settings:
    return Settings()
