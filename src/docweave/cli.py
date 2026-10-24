"""Command line interface (built on click).

Exit codes (stable, documented in the README):

0  success
1  unexpected internal error
2  invalid input (unsupported format, encrypted, malformed, missing file)
3  a configured limit or deadline was reached
4  a required optional dependency is not installed
5  completed, but at least one file failed or produced errors
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

from docweave import __version__
from docweave.config import ConversionConfig
from docweave.converter import (
    JSON,
    JSONL,
    MARKDOWN,
    OUTPUT_FORMATS,
    REPORT,
    BatchItem,
    BatchReport,
    Converter,
    discover,
)
from docweave.diagram.render import DIAGRAM_FORMATS, MERMAID
from docweave.exceptions import (
    DeadlineExceededError,
    DependencyMissingError,
    DocWeaveError,
    InputError,
    LimitExceededError,
)

EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_INPUT = 2
EXIT_LIMIT = 3
EXIT_DEPENDENCY = 4
EXIT_PARTIAL = 5

_EXT_FOR_FORMAT = {
    MARKDOWN: "md",
    JSON: "json",
    JSONL: "jsonl",
    REPORT: "report.json",
}
_EXTENSION_TO_FORMAT = {
    "md": MARKDOWN,
    "markdown": MARKDOWN,
    "json": JSON,
    "jsonl": JSONL,
    "ndjson": JSONL,
    "report.json": REPORT,
}

console = Console(soft_wrap=False)
err_console = Console(stderr=True, soft_wrap=False)


def _exit_code_for(error: BaseException) -> int:
    if isinstance(error, DependencyMissingError):
        return EXIT_DEPENDENCY
    if isinstance(error, (LimitExceededError, DeadlineExceededError)):
        return EXIT_LIMIT
    if isinstance(error, InputError):
        return EXIT_INPUT
    if isinstance(error, DocWeaveError):
        return int(getattr(error, "exit_code", EXIT_INTERNAL))
    return EXIT_INTERNAL


class _Context:
    """Resolved invocation settings shared by the subcommands."""

    def __init__(
        self,
        quiet: bool,
        as_json: bool,
        config: ConversionConfig,
    ) -> None:
        self.quiet = quiet
        self.as_json = as_json
        self.config = config


def _build_config(
    ctx: click.Context | None,
    *,
    extract_assets: bool,
    assets_dir: str | None,
    pdf_tables: str | None,
    include_headers_footers: bool,
    include_images: bool | None,
    table_rendering: str | None,
    image_rendering: str | None,
    max_file_size_mb: int | None,
    deadline: float | None,
    ocr_repair: bool,
    detect_diagrams: bool,
    classify_images: bool,
    config_path: str | None,
    quiet: bool,
    as_json: bool,
) -> _Context:
    try:
        base = ConversionConfig.from_json_file(config_path) if config_path else ConversionConfig()
        overrides = {
            "extract_assets": True if (extract_assets or assets_dir) else None,
            "assets_dir": Path(assets_dir) if assets_dir else None,
            "pdf_tables": pdf_tables,
            "include_headers_footers": True if include_headers_footers else None,
            "include_images": include_images,
            "table_rendering": table_rendering,
            "image_rendering": image_rendering,
            "max_file_size_mb": max_file_size_mb,
            "deadline_seconds": deadline,
            "ocr_repair": True if ocr_repair else None,
            "detect_diagrams": True if detect_diagrams else None,
            "classify_images": True if classify_images else None,
        }
        return _Context(quiet, as_json, base.merged(**overrides))
    except DocWeaveError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise SystemExit(_exit_code_for(exc)) from exc


def _resolve_format(fmt: str | None, output: Path | None) -> str:
    if fmt:
        chosen = fmt.lower()
        if chosen not in OUTPUT_FORMATS:
            raise InputError(
                f"Unknown --format '{fmt}'. Choose one of: {', '.join(OUTPUT_FORMATS)}."
            )
        return chosen
    if output is not None:
        mapped = _EXTENSION_TO_FORMAT.get(output.suffix.lstrip(".").lower())
        if mapped:
            return mapped
    return MARKDOWN


def _print_diagnostics(result, limit: int = 20) -> None:
    diagnostics = result.diagnostics
    if not diagnostics.diagnostics:
        console.print("[green]No diagnostics: nothing suspicious was detected.[/green]")
        return
    table = Table(show_lines=False, box=None, pad_edge=False)
    table.add_column("sev", width=7)
    table.add_column("code", width=30)
    table.add_column("where", width=14)
    table.add_column("detail", overflow="fold")
    colours = {"error": "red", "warning": "yellow", "info": "cyan"}
    for d in diagnostics.sorted()[:limit]:
        table.add_row(
            Text(d.severity.value, style=colours[d.severity.value]),
            Text(d.code),
            Text(d.location()),
            Text(d.message),
        )
    console.print(table)
    remaining = len(diagnostics.diagnostics) - limit
    if remaining > 0:
        console.print(f"[dim]… and {remaining} more[/dim]")
    console.print(f"[dim]{diagnostics.summary()}[/dim]")


# --------------------------------------------------------------------------
# click group
# --------------------------------------------------------------------------


@click.group(
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100},
    invoke_without_command=False,
)
@click.version_option(__version__, "-V", "--version", prog_name="docweave")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output.")
@click.option("--json-output", "as_json", is_flag=True, help="Machine-readable stdout.")
@click.pass_context
def cli(ctx: click.Context, quiet: bool, as_json: bool) -> None:
    """Local-first document conversion with honest diagnostics.

    \b
    docweave convert report.pdf -o report.md
    docweave convert report.pdf --format json -o report.json
    docweave convert ./corpus -r -o ./out --extract-assets
    docweave inspect report.pdf
    docweave validate ./corpus --recursive
    docweave capabilities
    """
    ctx.ensure_object(dict)
    ctx.obj["quiet"] = quiet
    ctx.obj["as_json"] = as_json


# --------------------------------------------------------------------------
# convert
# --------------------------------------------------------------------------


@cli.command("convert")
@click.argument("sources", nargs=-1, required=True, type=click.Path(path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output file or directory.")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(OUTPUT_FORMATS), help="Output format."
)
@click.option("--recursive", "-r", is_flag=True, help="Recurse into directories.")
@click.option("--extract-assets", is_flag=True, help="Write embedded images to disk.")
@click.option("--assets-dir", type=str, help="Asset output directory (implies --extract-assets).")
@click.option(
    "--pdf-tables", type=click.Choice(["auto", "always", "never"]), help="PDF table recovery."
)
@click.option("--include-headers-footers", is_flag=True, help="Keep repeating headers/footers.")
@click.option("--images/--no-images", "include_images", default=None, help="Include image references.")
@click.option(
    "--tables",
    "table_rendering",
    type=click.Choice(["markdown", "html", "both", "list"]),
    help="How to render tables.",
)
@click.option(
    "--image-rendering",
    type=click.Choice(["placeholder", "path", "link", "omit"]),
    help="How to render images.",
)
@click.option("--max-file-size-mb", type=int, help="Reject larger inputs.")
@click.option("--deadline", "deadline_seconds", type=float, help="Per-document time budget.")
@click.option(
    "--ocr-repair",
    is_flag=True,
    help=(
        "Re-read pages that measurement says have no usable text. Uses Indra "
        "OCR when installed, Tesseract otherwise; needs `pip install "
        "'docweave[ocr]'`."
    ),
)
@click.option(
    "--detect-diagrams",
    is_flag=True,
    help="Reconstruct diagrams found inside the document's images.",
)
@click.option(
    "--classify-images",
    is_flag=True,
    help="Classify images before reconstruction and skip the ones that are not diagrams.",
)
@click.option("--concurrency", "-j", type=click.IntRange(1, 32), default=1, show_default=True)
@click.option("--config", "-c", "config_path", type=str, help="JSON config file.")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output.")
@click.option("--json-output", "as_json", is_flag=True, help="Print a machine-readable summary.")
def convert_command(
    sources: Sequence[Path],
    output: Path | None,
    fmt: str | None,
    recursive: bool,
    extract_assets: bool,
    assets_dir: str | None,
    pdf_tables: str | None,
    include_headers_footers: bool,
    include_images: bool | None,
    table_rendering: str | None,
    image_rendering: str | None,
    max_file_size_mb: int | None,
    deadline_seconds: float | None,
    concurrency: int,
    config_path: str | None,
    quiet: bool,
    as_json: bool,
    ocr_repair: bool,
    detect_diagrams: bool,
    classify_images: bool,
) -> None:
    """Convert one or more documents."""
    context = _build_config(
        None,
        extract_assets=extract_assets,
        assets_dir=assets_dir,
        pdf_tables=pdf_tables,
        include_headers_footers=include_headers_footers,
        include_images=include_images,
        table_rendering=table_rendering,
        image_rendering=image_rendering,
        max_file_size_mb=max_file_size_mb,
        deadline=deadline_seconds,
        ocr_repair=ocr_repair,
        detect_diagrams=detect_diagrams,
        classify_images=classify_images,
        config_path=config_path,
        quiet=quiet,
        as_json=as_json,
    )
    quiet = quiet or context.quiet
    as_json = as_json or context.as_json
    # Machine-readable stdout requires silence everywhere else.
    quiet = quiet or as_json

    try:
        output_format = _resolve_format(fmt, output)
    except DocWeaveError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise SystemExit(_exit_code_for(exc)) from exc

    files: list[Path] = []
    for source in sources:
        files.extend(discover(source, recursive=recursive))

    if not files:
        err_console.print("[red]No input files matched.[/red]")
        raise SystemExit(EXIT_INPUT)

    single_input = len(sources) == 1 and Path(sources[0]).is_file()
    single = single_input and len(files) == 1

    if single and output is not None and output.suffix == "":
        output = output / f"{files[0].stem}.{_EXT_FOR_FORMAT[output_format]}"
    if not single and output is not None and output.suffix:
        err_console.print(
            "[red]--output must be a directory when converting more than one file.[/red]"
        )
        raise SystemExit(EXIT_INPUT)

    converter = Converter(context.config)

    if single:
        raise SystemExit(
            _convert_one(
                converter, files[0], output, output_format, quiet, show_diagnostics=not quiet
            )
        )

    report = _convert_batch(converter, files, output, output_format, concurrency, quiet)
    if as_json:
        click.echo(json.dumps(report.summary(), default=str))
    else:
        _render_batch_summary(report, quiet)
    raise SystemExit(EXIT_OK if not report.failed else EXIT_PARTIAL)


def _convert_one(
    converter: Converter,
    path: Path,
    output: Path | None,
    output_format: str,
    quiet: bool,
    show_diagnostics: bool = True,
) -> int:
    try:
        result = converter.convert(path)
    except DocWeaveError as exc:
        err_console.print(f"[red]{path.name}: {exc}[/red]")
        return _exit_code_for(exc)
    except Exception as exc:  # pragma: no cover - unexpected adapter failure
        err_console.print(f"[red]{path.name}: unexpected {type(exc).__name__}: {exc}[/red]")
        return EXIT_INTERNAL

    text = result.render(output_format, converter.config)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        if not quiet:
            console.print(
                f"[green]wrote[/green] {output} "
                f"({len(text)} chars, {result.elapsed_ms:.0f} ms)"
            )
    elif not quiet:
        click.echo(text)

    if show_diagnostics:
        _print_diagnostics(result)
    return EXIT_OK if result.ok else EXIT_PARTIAL


def _unique_names(items: list[BatchItem], output_format: str) -> dict[int, str]:
    """Assign a collision-free output filename to every batch item, in order."""
    taken: set[str] = set()
    assigned: dict[int, str] = {}
    for item in items:
        assigned[id(item)] = _target_name(item.path, output_format, taken)
    return assigned


def _target_name(path: Path, output_format: str, taken: set[str]) -> str:
    """Pick a collision-free output name.

    ``report.pdf`` and ``report.docx`` would both become ``report.md`` and one
    would silently overwrite the other, so fall back to including the source
    extension and then a counter.
    """
    ext = "." + _EXT_FOR_FORMAT[output_format]
    candidates = [f"{path.stem}{ext}", f"{path.stem}_{path.suffix.lstrip('.')}{ext}"]
    for base in candidates:
        name = base.lower()
        if name not in taken:
            taken.add(name)
            return base
    index = 2
    while True:
        base = f"{path.stem}_{index}{ext}"
        if base.lower() not in taken:
            taken.add(base.lower())
            return base
        index += 1


def _convert_batch(
    converter: Converter,
    files: list[Path],
    output: Path | None,
    output_format: str,
    concurrency: int,
    quiet: bool,
) -> BatchReport:
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    if not quiet:
        console.print(f"[dim]Converting {len(files)} file(s), concurrency {concurrency}…[/dim]")

    items = [BatchItem(path=f) for f in files]
    # Names are assigned up front, serially, so concurrency cannot race on them.
    names = _unique_names(items, output_format)

    def run(item: BatchItem) -> None:
        result, error, kind = converter.convert_safe(item.path)
        item.result = result
        item.error = error
        item.error_kind = kind
        if result is not None and output is not None:
            target = output / names[id(item)]
            target.write_text(result.render(output_format, converter.config), encoding="utf-8")
        if quiet:
            return
        if item.ok:
            assert item.result is not None
            console.print(
                f"[green]ok[/green]   {item.path.name} ({item.elapsed_ms:.0f} ms, "
                f"{len(item.result.diagnostics.diagnostics)} diagnostic(s))"
            )
        else:
            console.print(f"[red]fail[/red] {item.path.name}: {item.reason}")

    started = time.perf_counter()
    if concurrency <= 1:
        for item in items:
            run(item)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(run, items))

    return BatchReport(items=items, elapsed_ms=(time.perf_counter() - started) * 1000.0)


def _render_batch_summary(report: BatchReport, quiet: bool) -> None:
    if quiet:
        return
    summary = report.summary()
    console.print()
    console.print(
        f"[bold]{summary['succeeded']}/{summary['total']} succeeded[/bold] "
        f"in {summary['elapsed_ms']:.0f} ms"
    )
    if report.failed:
        table = Table(show_lines=False, box=None)
        table.add_column("file", overflow="fold")
        table.add_column("error", overflow="fold")
        for item in report.failed:
            table.add_row(item.path.name, item.reason)
        console.print(table)
    if summary["codes"]:
        console.print(f"[dim]diagnostic codes: {', '.join(summary['codes'])}[/dim]")


# --------------------------------------------------------------------------
# inspect
# --------------------------------------------------------------------------


@cli.command("inspect")
@click.argument("sources", nargs=-1, required=True, type=click.Path(path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
@click.option("--quiet", "-q", is_flag=True, help="Suppress human-readable output.")
def inspect_command(sources: Sequence[Path], as_json: bool, quiet: bool) -> None:
    """Report format, integrity and cheap facts without converting."""
    converter = Converter()
    payloads: list[dict] = []
    failures = 0

    for source in sources:
        try:
            result = converter.inspect(source)
        except DocWeaveError as exc:
            failures += 1
            err_console.print(f"[red]{source}: {exc}[/red]")
            continue
        d = result.detection
        if not result.readable:
            failures += 1
        payloads.append(
            {
                "name": result.name,
                "bytes": result.byte_size,
                "sha256": result.sha256,
                "detected_format": d.format,
                "confidence": d.confidence,
                "method": d.method,
                "declared_extension": d.declared_extension,
                "extension_mismatch": d.extension_mismatch,
                "encrypted": d.encrypted,
                "structure": result.structure,
                "notes": d.notes,
            }
        )
        if as_json or quiet:
            continue
        if not result.readable:
            console.print(f"[bold]{result.name}[/bold]")
            console.print(f"  [red]unreadable: {result.error}[/red]")
            continue
        console.print(f"[bold]{result.name}[/bold]")
        console.print(f"  format    : {d.format} (confidence {d.confidence:.2f}, via {d.method})")
        if d.declared_extension:
            flag = " [yellow]!= content[/yellow]" if d.extension_mismatch else ""
            console.print(f"  extension : .{d.declared_extension}{flag}")
        console.print(f"  size      : {result.byte_size} bytes")
        console.print(f"  sha256    : {result.sha256[:16]}…")
        for key, value in result.structure.items():
            console.print(f"  {key:<9}: {value}")
        for note in d.notes:
            console.print(f"  [yellow]note[/yellow]: {note}")

    if as_json:
        click.echo(json.dumps(payloads, default=str))
    raise SystemExit(EXIT_OK if not failures else EXIT_PARTIAL)


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------


@cli.command("validate")
@click.argument("sources", nargs=-1, required=True, type=click.Path(path_type=Path))
@click.option("--recursive", "-r", is_flag=True, help="Recurse into directories.")
@click.option("--concurrency", "-j", type=click.IntRange(1, 32), default=1, show_default=True)
@click.option("--config", "-c", "config_path", type=str, help="JSON config file.")
@click.option("--json", "as_json", is_flag=True, help="Emit the full report as JSON.")
@click.option("--quiet", "-q", is_flag=True, help="Suppress human-readable output.")
def validate_command(
    sources: Sequence[Path],
    recursive: bool,
    concurrency: int,
    config_path: str | None,
    as_json: bool,
    quiet: bool,
) -> None:
    """Convert and report extraction diagnostics without writing output."""
    try:
        config = ConversionConfig.from_json_file(config_path) if config_path else ConversionConfig()
    except DocWeaveError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise SystemExit(_exit_code_for(exc)) from exc

    files: list[Path] = []
    for source in sources:
        files.extend(discover(source, recursive=recursive))
    if not files:
        err_console.print("[red]No input files matched.[/red]")
        raise SystemExit(EXIT_INPUT)

    report = Converter(config).convert_many(files, concurrency=concurrency)
    payload = report.summary()
    payload["files"] = [
        {
            "name": item.path.name,
            "ok": item.ok,
            "error": None if item.ok else item.reason,
            "elapsed_ms": round(item.elapsed_ms, 3),
            "diagnostics": [
                d.model_dump(mode="json")
                for d in (item.result.diagnostics.sorted() if item.result else [])
            ],
        }
        for item in report.items
    ]

    if as_json:
        click.echo(json.dumps(payload, default=str))
        raise SystemExit(EXIT_OK if not report.failed else EXIT_PARTIAL)

    if not quiet:
        console.print(f"[bold]{payload['succeeded']}/{payload['total']} converted[/bold]")
        colours = {"error": "red", "warning": "yellow", "info": "cyan"}
        for item in report.items:
            if item.result is None:
                console.print(f"[red]fail[/red] {item.path.name}: {item.reason}")
                continue
            worst = item.result.diagnostics.worst_severity()
            label = worst.value if worst else "clean"
            count = len(item.result.diagnostics.diagnostics)
            colour = colours.get(label, "green")
            console.print(f"[{colour}]{label}[/{colour}] {item.path.name} ({count} diagnostic(s))")
            for d in item.result.diagnostics.sorted()[:5]:
                console.print(f"    [dim]{d.to_line()}[/dim]")

    raise SystemExit(EXIT_OK if not report.failed else EXIT_PARTIAL)


# --------------------------------------------------------------------------
# capabilities
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# diagram
# --------------------------------------------------------------------------


@cli.command("diagram")
@click.argument("source", required=True, type=click.Path(path_type=Path))
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(DIAGRAM_FORMATS),
    default=MERMAID,
    show_default=True,
    help="Diagram output format.",
)
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Write to a file.")
@click.option("--no-ocr", is_flag=True, help="Skip OCR entirely; labels stay unread.")
@click.option("--quiet", "-q", is_flag=True, help="Suppress the diagnostic summary.")
@click.option("--json", "as_json", is_flag=True, help="Emit the full graph as JSON.")
def diagram_command(
    source: Path,
    fmt: str,
    output: Path | None,
    no_ocr: bool,
    quiet: bool,
    as_json: bool,
) -> None:
    """Reconstruct a diagram image into structure, not just text.

    \b
    docweave diagram architecture.png --format mermaid -o architecture.mmd
    docweave diagram flowchart.png --format json -o flowchart.json
    """
    from docweave.diagram import analyze_file, render

    try:
        result = analyze_file(source, reader=_diagram_reader(no_ocr))
    except ValueError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise SystemExit(EXIT_INPUT) from exc
    except DocWeaveError as exc:
        err_console.print(f"[red]{source.name}: {exc}[/red]")
        raise SystemExit(_exit_code_for(exc)) from exc

    chosen = JSON if as_json else fmt
    text = render(result.graph, chosen)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        if not quiet:
            console.print(
                f"[green]wrote[/green] {output} "
                f"({len(text)} chars, {result.elapsed_ms:.0f} ms)"
            )
    else:
        # The diagram itself is the result; --quiet only silences the report.
        click.echo(text)

    if not quiet:
        _print_diagram_diagnostics(result)
    # Exit 5 when the reconstruction is not trustworthy enough to publish.
    raise SystemExit(EXIT_OK if not result.graph.degraded else EXIT_PARTIAL)


def _diagram_reader(no_ocr: bool):
    from docweave.diagram import default_reader
    from docweave.diagram.text import NullLabelReader

    if no_ocr:
        return NullLabelReader()
    return default_reader()


def _print_diagram_diagnostics(result) -> None:
    diagnostics = result.diagnostics
    graph = result.graph

    # An empty graph with a clean bill of health is the worst possible report:
    # it tells the reader the picture was understood when nothing was found.
    # A real diagram that yields no nodes is a failure, and saying so is the
    # whole point of shipping diagnostics at all.
    if not graph.nodes:
        console.print(
            f"[red]No diagram structure was found in "
            f"{graph.source_image or 'the image'}.[/red]"
        )
        console.print(
            "  [dim]The image opened and was analysed; nothing in it matched a "
            "shape this engine recognises.[/dim]"
        )
        console.print(
            "  [dim]Likely: filled or coloured nodes rather than outlines, "
            "illustration rather than a diagram, or a resolution too low for the "
            "contour detector.[/dim]"
        )
        if not diagnostics:
            return

    if not diagnostics:
        console.print("[green]Diagram reconstructed with no uncertainties flagged.[/green]")
        return
    table = Table(show_lines=False, box=None, pad_edge=False)
    table.add_column("sev", width=7)
    table.add_column("code", width=28)
    table.add_column("detail", overflow="fold")
    colours = {"error": "red", "warning": "yellow", "info": "cyan"}
    for diagnostic in diagnostics:
        table.add_row(
            Text(diagnostic.severity.value, style=colours[diagnostic.severity.value]),
            Text(diagnostic.code),
            Text(diagnostic.message),
        )
    console.print(table)
    stats = result.graph.connectivity()
    console.print(
        f"[dim]{result.graph.kind.value}: {stats['node_count']} nodes, "
        f"{stats['edge_count']} edges, {stats['readable_labels']} labels read "
        f"({result.elapsed_ms:.0f} ms)[/dim]"
    )


# --------------------------------------------------------------------------
# capabilities
# --------------------------------------------------------------------------


@cli.command("capabilities")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def capabilities_command(as_json: bool) -> None:
    """Show adapters, optional dependencies, capabilities and limitations."""
    payloads = Converter().capabilities()
    if as_json:
        click.echo(json.dumps(payloads, default=str))
        return

    for entry in payloads:
        if entry["available"]:
            status = "[green]available[/green]"
        else:
            status = "[yellow]missing " + ", ".join(entry["missing_dependencies"]) + "[/yellow]"
        console.print(f"[bold]{entry['adapter']}[/bold] ({', '.join(entry['formats'])}) — {status}")
        console.print(f"  cost      : {entry['cost']}")
        if entry["capabilities"]:
            console.print("  can do    :")
            for item in entry["capabilities"]:
                console.print(f"    • {item}")
        if entry["limitations"]:
            console.print("  cannot do :")
            for item in entry["limitations"]:
                console.print(f"    • {item}")
        console.print()


def main() -> None:  # pragma: no cover - console-script entry point
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
