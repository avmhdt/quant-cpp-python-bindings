# Install

```bash
pip install qkern
```

Wheels exist for CPython 3.11 (version-specific) and 3.12 or later (one stable-ABI wheel,
`cp312-abi3`, that also serves 3.13 and 3.14) on:

| Platform | Architectures |
|---|---|
| Linux (manylinux_2_28) | x86_64, aarch64 |
| macOS 11+ | arm64, x86_64 |
| Windows | AMD64 |

NumPy (1.26 or later) is the only runtime dependency. Python 3.10 and older are not
supported.

## From source

The source distribution builds with any C++20 compiler (GCC 13+, Clang 17+, AppleClang 15+,
MSVC 2022 17.8+) and CMake 3.28+:

```bash
pip install --no-binary qkern qkern
```

The build downloads nothing beyond its build requirements (scikit-build-core and nanobind).

## Check the build

```bash
python -c "import qkern; qkern.show_config()"
```

prints the version, compiler, floating-point flags (`-ffp-contract=off`: no FMA contraction,
so results match across platforms), whether the extension uses the stable ABI, and the GIL
release threshold.
