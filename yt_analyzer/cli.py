"""Command line interface.

    yt-analyzer analyze @mkbhd --output report.md
    yt-analyzer demo
    yt-analyzer providers
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .analytics import consistency_score, duration_buckets, outliers, performance_by_duration
from .config import ConfigError, get_settings
from .health import check_settings, missing_credentials
from .models import ChannelReport, format_duration
from .pipeline import ChannelAnalysisPipeline
from .providers.base import ProviderError, available_analyzers, available_sources


def _init_stdio() -> bool:
    """Make stdout UTF-8 where possible; report whether glyphs are safe.

    Video titles routinely contain emoji and non-Latin script, which a cp1252
    Windows console cannot encode.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - platform specific
                pass

    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "▸✓✗•".encode(encoding)
        return True
    except (LookupError, UnicodeEncodeError):  # pragma: no cover - platform specific
        return False


UNICODE_OK = _init_stdio()

SYMBOLS = {
    "arrow": "▸" if UNICODE_OK else ">",
    "bullet": "•" if UNICODE_OK else "-",
    "check": "✓" if UNICODE_OK else "+",
    "cross": "✗" if UNICODE_OK else "x",
}

console = Console()

STAGE_STYLE = {
    "resolve": "cyan",
    "videos": "magenta",
    "stats": "yellow",
    "transcripts": "blue",
    "analyze": "bright_magenta",
    "done": "green",
}


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
    )


def _progress(stage: str, message: str) -> None:
    style = STAGE_STYLE.get(stage, "white")
    console.print(f"[{style}]{SYMBOLS['arrow']} {stage:<11}[/{style}] {message}")


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def _render_report(report: ChannelReport) -> None:
    channel, stats = report.channel, report.stats

    console.print()
    console.print(
        Panel(
            f"{channel.subscriber_count:,} subscribers  ·  "
            f"{channel.video_count:,} videos  ·  "
            f"{channel.view_count:,} total views\n{channel.url}",
            title=channel.title,
            border_style="red",
        )
    )

    summary = Table(title="Performance", header_style="bold", show_header=False)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value")
    summary.add_row("Videos analyzed", f"{stats.total_videos}")
    summary.add_row("Median views", f"{stats.median_views:,.0f}")
    summary.add_row("Mean views", f"{stats.mean_views:,.0f}")
    summary.add_row("Mean engagement", f"{stats.mean_engagement_rate:.2f}%")
    summary.add_row(
        "Mean duration", format_duration(int(stats.mean_duration_seconds))
    )
    summary.add_row(
        "Shorts / long-form",
        f"{stats.shorts_count} ({stats.shorts_share:.0f}%) / {stats.longform_count}",
    )
    summary.add_row("Upload cadence", f"{stats.uploads_per_month:.1f}/month")
    score = consistency_score(report.videos)
    if score is not None:
        summary.add_row("Consistency", f"{score:.0f}/100")
    if stats.best_weekday:
        summary.add_row("Best upload day", stats.best_weekday)
    console.print()
    console.print(summary)

    top = report.top_videos(5)
    if top:
        table = Table(title="Top Videos by Views", header_style="bold")
        table.add_column("Views", justify="right", style="green")
        table.add_column("Engage", justify="right")
        table.add_column("Length", justify="right", style="dim")
        table.add_column("Title")
        for video in top:
            table.add_row(
                f"{video.view_count:,}",
                f"{video.engagement_rate:.2f}%",
                video.duration_label,
                _truncate(video.title, 52),
            )
        console.print()
        console.print(table)

    over = outliers(report.videos)
    if over:
        console.print(
            f"\n[bold]Outliers[/bold] [dim](more than 2x the median)[/dim]"
        )
        for video in over[:5]:
            console.print(
                f"  {SYMBOLS['bullet']} {video.view_count:,} — "
                f"{_truncate(video.title, 60)}"
            )

    performance = performance_by_duration(report.videos)
    if performance:
        table = Table(title="Mean Views by Length", header_style="bold")
        table.add_column("Bucket", style="cyan")
        table.add_column("Videos", justify="right")
        table.add_column("Mean views", justify="right")
        buckets = duration_buckets(report.videos)
        for label, mean_views in performance.items():
            table.add_row(label, str(buckets.get(label, 0)), f"{mean_views:,.0f}")
        console.print()
        console.print(table)

    if stats.top_tags:
        tags = ", ".join(f"{tag} ({count})" for tag, count in stats.top_tags[:10])
        console.print(f"\n[bold]Top tags[/bold]\n  {tags}")

    if stats.title_word_frequency:
        words = ", ".join(
            f"{word} ({count})" for word, count in stats.title_word_frequency[:10]
        )
        console.print(f"\n[bold]Recurring title words[/bold]\n  {words}")

    insights = report.insights
    if insights:
        console.print()
        if insights.content_format:
            console.print(f"[bold]Format:[/bold] {insights.content_format}")
        if insights.target_audience:
            console.print(f"[bold]Audience:[/bold] {insights.target_audience}")
        for heading, values in (
            ("Themes", insights.themes),
            ("Hook patterns", insights.hook_patterns),
            ("Strengths", insights.strengths),
            ("Opportunities", insights.opportunities),
            ("Recommended next", insights.recommended_topics),
        ):
            if values:
                console.print(f"\n[bold]{heading}[/bold]")
                for value in values:
                    console.print(f"  {SYMBOLS['bullet']} {value}")
    else:
        console.print(
            "\n[dim]No qualitative insights were produced "
            "(analyzer unavailable or failed).[/dim]"
        )


def _require_credentials(settings, command: str) -> None:
    """Stop before the run when a selected provider has no key.

    Exits 1 with the variable name and where to get it, rather than letting the
    failure surface as a ConfigError part way through the pipeline.
    """
    gaps = missing_credentials(settings)
    if not gaps:
        return

    console.print("[red]Missing credentials for this configuration:[/red]")
    for item in gaps:
        where = f" [dim]({item.get_it_at})[/dim]" if item.get_it_at else ""
        console.print(
            f"  {SYMBOLS['cross']} [bold]{item.env_var}[/bold] "
            f"- needed for {item.needed_for}{where}"
        )
    console.print(
        f"\nSet it in your .env file, then re-run [bold]{command} check[/bold] "
        f"to verify. Or switch the provider back to 'mock' to run offline."
    )
    raise SystemExit(1)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="yt-analyzer")
def cli() -> None:
    """Analyze a YouTube channel's content patterns and performance."""


@cli.command()
@click.argument("channel")
@click.option("--source", default=None, help="Override DATA_SOURCE.")
@click.option("--analyzer", default=None, help="Override ANALYZER.")
@click.option("--max-videos", type=int, default=None, help="How many uploads to fetch.")
@click.option(
    "--max-transcripts", type=int, default=None, help="How many transcripts to read."
)
@click.option("--no-transcripts", is_flag=True, help="Skip transcript fetching.")
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the markdown report to this path.",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def analyze(
    channel, source, analyzer, max_videos, max_transcripts, no_transcripts,
    output, verbose,
) -> None:
    """Analyze CHANNEL — a handle (@name), URL, or channel id."""
    _configure_logging(verbose)
    settings = get_settings(refresh=True)
    if source:
        settings.source = source
    if analyzer:
        settings.analyzer = analyzer
    if max_videos:
        settings.max_videos = max_videos
    if max_transcripts is not None:
        settings.max_transcripts = max_transcripts
    if no_transcripts:
        settings.fetch_transcripts = False

    console.print(f"[dim]{settings.describe()}[/dim]")
    _require_credentials(settings, "yt-analyzer")

    try:
        pipeline = ChannelAnalysisPipeline(settings=settings, progress=_progress)
        report = pipeline.run(channel)
    except (ConfigError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    except ProviderError as exc:
        console.print(f"[red]Provider error:[/red] {exc}")
        sys.exit(2)

    _render_report(report)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.to_markdown(), encoding="utf-8")
        console.print(f"\n[dim]Report written to {output}[/dim]")

    console.print(f"\n[dim]Finished in {report.elapsed_seconds:.1f}s[/dim]")


@cli.command()
def providers() -> None:
    """List available providers and the current configuration."""
    settings = get_settings(refresh=True)

    table = Table(title="Providers", header_style="bold")
    table.add_column("Stage", style="cyan")
    table.add_column("Available")
    table.add_column("Selected", style="green")
    table.add_row("source", ", ".join(available_sources()), settings.source)
    table.add_row("analyzer", ", ".join(available_analyzers()), settings.analyzer)
    console.print(table)
    console.print(f"\n[dim]{settings.describe()}[/dim]")


@cli.command()
@click.option("--source", default=None, help="Override DATA_SOURCE.")
@click.option("--analyzer", default=None, help="Override ANALYZER.")
def check(source, analyzer) -> None:
    """Verify the configured API keys actually work."""
    settings = get_settings(refresh=True)
    if source:
        settings.source = source
    if analyzer:
        settings.analyzer = analyzer

    console.print(f"[dim]{settings.describe()}[/dim]\n")
    results = check_settings(settings)

    if not results:
        console.print("[yellow]Nothing to check for this configuration.[/yellow]")
        return

    for result in results:
        icon = (
            f"[green]{SYMBOLS['check']}[/green]"
            if result.ok
            else f"[red]{SYMBOLS['cross']}[/red]"
        )
        console.print(f"  {icon} [bold]{result.provider}[/bold]: {result.message}")

    failed = [result for result in results if not result.ok]
    if failed:
        console.print(f"\n[red]{len(failed)} check(s) failed.[/red]")
        sys.exit(2)
    console.print("\n[green]All checks passed.[/green]")


@cli.command()
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the markdown report to this path.",
)
def demo(output) -> None:
    """Analyze a synthetic channel offline -- no API keys needed."""
    _configure_logging(False)
    settings = get_settings(refresh=True)
    settings.source = settings.analyzer = "mock"

    console.print("[bold]Analyzing the built-in demo channel[/bold]\n")
    report = ChannelAnalysisPipeline(settings=settings, progress=_progress).run("@demo")
    _render_report(report)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.to_markdown(), encoding="utf-8")
        console.print(f"\n[dim]Report written to {output}[/dim]")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
