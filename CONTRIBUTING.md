# Contributing

Thanks for your interest in qkern. Issues and pull requests are welcome.

## Development setup

You need a C++20 compiler (GCC ≥ 13, Clang ≥ 17, AppleClang ≥ 15 or MSVC 2022 17.8+),
CMake ≥ 3.28, Ninja and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                        # builds the extension and installs the dev tools
uv run pytest                  # Python tests (not slow, not local_data)
cmake --preset dev             # C++ tests (Debug), build tree under ~/build/
cmake --build --preset dev
ctest --preset dev
```

After changing C++ code, rebuild the extension with `uv sync --reinstall-package qkern`.

## Before you open a pull request

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict src/qkern tests/python tools
uv run python -m tools.qkdev stubs check          # --write after changing bindings
uv run python -m tools.qkdev fixtures make --check
git ls-files '*.cpp' '*.hpp' | xargs clang-format --dry-run --Werror
```

`pre-commit install` runs most of these on each commit.

## Rules

- One implementation per kernel: batch functions drive the same C++ state machine as the
  streaming classes.
- No `-ffast-math`, no `-Ofast`, no FMA contraction. Never loosen a numerical tolerance
  without an entry in `docs/decisions.md` explaining why.
- NumPy is the only runtime dependency.
- Generated files (`src/qkern/_qkern.pyi`, golden files, benchmark pages and figures) are
  changed only by their generator commands.
- Changes to a public name or default need a CHANGELOG entry under "Changed".
- Never commit data other than the small synthetic fixtures in `tests/fixtures/`.
