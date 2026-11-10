"""qkdev: developer CLI (fixtures, data, stubs). Run as ``python -m tools.qkdev``."""

from __future__ import annotations

import logging

import typer

from tools.qkdev import data as data_mod
from tools.qkdev import fixtures as fixtures_mod
from tools.qkdev import golden as golden_mod
from tools.qkdev import stubs as stubs_mod
from tools.qkdev.settings import load_settings

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
fixtures_app = typer.Typer(no_args_is_help=True, help="Synthetic test fixtures.")
data_app = typer.Typer(no_args_is_help=True, help="Real-workload data (local only).")
stubs_app = typer.Typer(no_args_is_help=True, help="Type stubs for the extension module.")
app.add_typer(fixtures_app, name="fixtures")
app.add_typer(data_app, name="data")
app.add_typer(stubs_app, name="stubs")

log = logging.getLogger("qkdev")


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@fixtures_app.command("make")
def fixtures_make(
    check: bool = typer.Option(False, "--check", help="Verify committed files are identical."),
) -> None:
    """Generate all synthetic fixtures into tests/fixtures/."""
    mismatched = fixtures_mod.make(check=check)
    if check:
        if mismatched:
            log.error("fixtures differ from their generator: %s", ", ".join(mismatched))
            raise typer.Exit(1)
        log.info("all %d fixtures are byte-identical", len(fixtures_mod.FIXTURES))
    else:
        log.info("wrote %d fixtures to %s", len(fixtures_mod.FIXTURES), fixtures_mod.FIXTURE_DIR)


@fixtures_app.command("golden")
def fixtures_golden(
    accept: bool = typer.Option(False, "--accept", help="Rewrite golden files and manifest."),
    kernel: str | None = typer.Option(None, "--kernel", help="Only this kernel."),
) -> None:
    """Compare kernel outputs with tests/golden (1e-12 relative); --accept rewrites them."""
    raise typer.Exit(golden_mod.run(accept=accept, kernel=kernel))


@data_app.command("smoke")
def data_smoke() -> None:
    """Fixture round trip, and list real partitions when QKERN_REAL_DATA_DIR is set."""
    raise typer.Exit(data_mod.smoke(load_settings()))


@data_app.command("import-real")
def data_import_real(
    dataset: str = typer.Option(..., help="trades, bbo or book_l2"),
    venue: str = typer.Option(...),
    symbol: str = typer.Option(...),
    date: str = typer.Option(..., help="YYYY-MM-DD"),
) -> None:
    """Convert one canonical Parquet partition to .npy columns under $DATA_ROOT/processed."""
    raise typer.Exit(data_mod.import_real(load_settings(), dataset, venue, symbol, date))


@stubs_app.command("check")
def stubs_check(
    write: bool = typer.Option(False, "--write", help="Rewrite the committed stub."),
) -> None:
    """Fail if src/qkern/_qkern.pyi differs from freshly generated stubs."""
    raise typer.Exit(stubs_mod.check(write=write))


if __name__ == "__main__":
    app()
