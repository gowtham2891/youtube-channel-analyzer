"""CLI tests via Click's runner."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from yt_analyzer import __version__
from yt_analyzer.cli import SYMBOLS, _truncate, cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestCli:
    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "YouTube channel" in result.output

    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_providers(self, runner):
        result = runner.invoke(cli, ["providers"])
        assert result.exit_code == 0
        assert "youtube" in result.output
        assert "gemini" in result.output

    def test_demo(self, runner):
        result = runner.invoke(cli, ["demo"])
        assert result.exit_code == 0, result.output
        assert "Performance" in result.output

    def test_demo_writes_a_report(self, runner, tmp_path):
        target = tmp_path / "report.md"
        result = runner.invoke(cli, ["demo", "--output", str(target)])
        assert result.exit_code == 0
        assert "## Performance" in target.read_text(encoding="utf-8")

    def test_analyze(self, runner):
        result = runner.invoke(cli, ["analyze", "@demo", "--max-videos", "15"])
        assert result.exit_code == 0, result.output
        assert "Top Videos by Views" in result.output

    def test_analyze_without_transcripts(self, runner):
        result = runner.invoke(
            cli, ["analyze", "@demo", "--no-transcripts", "--max-videos", "10"]
        )
        assert result.exit_code == 0
        assert "Skipped" in result.output

    def test_bad_source_exits_1(self, runner):
        result = runner.invoke(cli, ["analyze", "@demo", "--source", "nonexistent"])
        assert result.exit_code == 1
        assert "Unknown data source" in result.output

    def test_empty_channel_exits_1(self, runner):
        result = runner.invoke(cli, ["analyze", "   "])
        assert result.exit_code == 1

    def test_live_source_without_a_key_exits_1(self, runner, monkeypatch):
        monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
        result = runner.invoke(cli, ["analyze", "@mkbhd", "--source", "youtube"])
        assert result.exit_code == 1
        assert "YOUTUBE_API_KEY" in result.output


class TestHelpers:
    def test_truncate_leaves_short_text(self):
        assert _truncate("short", 20) == "short"

    def test_truncate_shortens_long_text(self):
        assert len(_truncate("x" * 100, 20)) == 20


class TestSymbols:
    def test_every_symbol_survives_the_active_stdout(self):
        import sys

        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        for value in SYMBOLS.values():
            value.encode(encoding)
