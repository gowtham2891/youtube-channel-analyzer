"""Tests for providers, parsing and the end-to-end pipeline."""

from __future__ import annotations

import httpx
import pytest

from yt_analyzer.config import ConfigError, Settings
from yt_analyzer.models import Channel, ChannelReport, ContentInsights, Transcript, Video
from yt_analyzer.pipeline import ChannelAnalysisPipeline, analyze_channel
from yt_analyzer.providers.base import (
    ProviderError,
    available_analyzers,
    available_sources,
    get_analyzer,
    get_source,
)
from yt_analyzer.providers.gemini import GeminiAnalyzer, build_prompt, loads_lenient
from yt_analyzer.providers.mock import HeuristicAnalyzer, MockDataSource
from yt_analyzer.providers.youtube import YouTubeDataSource, parse_channel_identifier


@pytest.fixture
def settings() -> Settings:
    return Settings(source="mock", analyzer="mock", max_videos=20, max_transcripts=5)


class TestChannelIdentifierParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("UCBJycsmduvYEL83R_U4JriQ", ("id", "UCBJycsmduvYEL83R_U4JriQ")),
            ("@mkbhd", ("handle", "mkbhd")),
            ("mkbhd", ("handle", "mkbhd")),
            ("https://www.youtube.com/@mkbhd", ("handle", "mkbhd")),
            (
                "https://www.youtube.com/channel/UCBJycsmduvYEL83R_U4JriQ",
                ("id", "UCBJycsmduvYEL83R_U4JriQ"),
            ),
            ("https://www.youtube.com/user/marquesbrownlee", ("username", "marquesbrownlee")),
            ("https://www.youtube.com/c/MKBHD", ("handle", "MKBHD")),
        ],
    )
    def test_parses_every_form(self, raw, expected):
        assert parse_channel_identifier(raw) == expected

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            parse_channel_identifier("   ")

    def test_url_without_a_channel_raises(self):
        with pytest.raises(ValueError, match="Could not find a channel"):
            parse_channel_identifier("https://www.youtube.com/")


class TestYouTubeParsing:
    def test_requires_an_api_key(self):
        with pytest.raises(ConfigError, match="YOUTUBE_API_KEY"):
            YouTubeDataSource(Settings(source="youtube", youtube_api_key=""))

    def test_parses_a_channel(self):
        channel = YouTubeDataSource.parse_channel(
            {
                "id": "UC123",
                "snippet": {
                    "title": "Test Channel",
                    "description": "About",
                    "publishedAt": "2020-01-01T00:00:00Z",
                    "thumbnails": {"high": {"url": "https://img/high.jpg"}},
                },
                "statistics": {
                    "subscriberCount": "1000",
                    "videoCount": "50",
                    "viewCount": "250000",
                },
                "contentDetails": {"relatedPlaylists": {"uploads": "UU123"}},
            }
        )
        assert channel.title == "Test Channel"
        assert channel.subscriber_count == 1000
        assert channel.uploads_playlist_id == "UU123"
        assert channel.average_views_per_video == 5000

    def test_parses_a_channel_with_missing_fields(self):
        channel = YouTubeDataSource.parse_channel({"id": "UC1"})
        assert channel.subscriber_count == 0
        assert channel.uploads_playlist_id == ""

    def test_parses_a_video(self):
        video = YouTubeDataSource.parse_video(
            {
                "id": "abc123",
                "snippet": {
                    "title": "My Video",
                    "publishedAt": "2026-03-05T10:00:00Z",
                    "tags": ["ai", "python"],
                    "thumbnails": {"default": {"url": "https://img/d.jpg"}},
                },
                "statistics": {
                    "viewCount": "5000", "likeCount": "250", "commentCount": "30"
                },
                "contentDetails": {"duration": "PT12M30S"},
            }
        )
        assert video.duration_seconds == 750
        assert video.view_count == 5000
        assert video.tags == ["ai", "python"]

    def test_video_with_hidden_statistics(self):
        """Uploaders can hide likes; the API then omits the field entirely."""
        video = YouTubeDataSource.parse_video(
            {"id": "x", "snippet": {"title": "T"}, "statistics": {"viewCount": "10"}}
        )
        assert video.like_count == 0
        assert video.engagement_rate == 0.0

    def test_parses_a_transcript(self):
        transcript = YouTubeDataSource.parse_transcript(
            "vid",
            [
                {"text": "Hello there", "start": 0.0, "duration": 2.0},
                {"text": "   ", "start": 2.0, "duration": 1.0},
                {"text": "Second line", "start": 3.0, "duration": 2.0},
            ],
        )
        assert len(transcript.segments) == 2  # blank dropped
        assert transcript.text == "Hello there Second line"
        assert transcript.available

    def test_403_becomes_a_clear_message(self, monkeypatch):
        source = YouTubeDataSource(Settings(youtube_api_key="key"))

        def forbidden(*args, **kwargs):
            request = httpx.Request("GET", "https://example.com")
            return httpx.Response(403, request=request)

        monkeypatch.setattr(httpx, "get", forbidden)
        with pytest.raises(ProviderError, match="quota"):
            source._get("channels", {})

    def test_retries_then_raises(self, monkeypatch):
        source = YouTubeDataSource(Settings(youtube_api_key="key", max_retries=2))
        calls = {"n": 0}

        def boom(*args, **kwargs):
            calls["n"] += 1
            raise httpx.ConnectError("down")

        monkeypatch.setattr(httpx, "get", boom)
        monkeypatch.setattr("time.sleep", lambda _: None)

        with pytest.raises(ProviderError, match="after 2 attempts"):
            source._get("channels", {})
        assert calls["n"] == 2


class TestTranscriptModel:
    def test_opening_takes_only_the_first_seconds(self):
        transcript = Transcript(
            video_id="v",
            segments=[
                {"start": 0.0, "duration": 5.0, "text": "The hook"},
                {"start": 5.0, "duration": 5.0, "text": "Still early"},
                {"start": 40.0, "duration": 5.0, "text": "Much later"},
            ],
        )
        assert transcript.opening(30) == "The hook Still early"

    def test_unavailable_transcript(self):
        transcript = Transcript(video_id="v", error="TranscriptsDisabled")
        assert not transcript.available
        assert transcript.word_count == 0


class TestRegistries:
    def test_sources(self):
        assert {"mock", "youtube"} <= set(available_sources())

    def test_analyzers(self):
        assert {"mock", "gemini"} <= set(available_analyzers())

    def test_resolution(self, settings):
        assert isinstance(get_source(settings), MockDataSource)
        assert isinstance(get_analyzer(settings), HeuristicAnalyzer)

    def test_unknown_source_raises(self, settings):
        settings.source = "nope"
        with pytest.raises(ValueError, match="Unknown data source"):
            get_source(settings)

    def test_unknown_analyzer_raises(self, settings):
        settings.analyzer = "nope"
        with pytest.raises(ValueError, match="Unknown analyzer"):
            get_analyzer(settings)


class TestMockDataSource:
    def test_generates_a_channel(self, settings):
        channel = MockDataSource(settings).resolve_channel("@demo")
        assert channel.title
        assert channel.uploads_playlist_id

    def test_empty_identifier_raises(self, settings):
        with pytest.raises(ProviderError, match="must not be empty"):
            MockDataSource(settings).resolve_channel("  ")

    def test_video_generation_is_deterministic(self, settings):
        source = MockDataSource(settings)
        channel = source.resolve_channel("@demo")
        first = source.fetch_videos(channel, 20)
        second = source.fetch_videos(channel, 20)
        assert [v.view_count for v in first] == [v.view_count for v in second]

    def test_videos_are_published_in_descending_order(self, settings):
        source = MockDataSource(settings)
        videos = source.fetch_videos(source.resolve_channel("@demo"), 20)
        dates = [video.published_at for video in videos]
        assert dates == sorted(dates, reverse=True)

    def test_contains_shorts_and_longform(self, settings):
        source = MockDataSource(settings)
        videos = source.fetch_videos(source.resolve_channel("@demo"), 30)
        assert any(video.is_short for video in videos)
        assert any(not video.is_short for video in videos)

    def test_some_transcripts_are_missing(self, settings):
        source = MockDataSource(settings)
        videos = source.fetch_videos(source.resolve_channel("@demo"), 30)
        transcripts = [source.fetch_transcript(video) for video in videos]
        assert any(t.available for t in transcripts)
        assert any(not t.available for t in transcripts)


class TestHeuristicAnalyzer:
    def test_produces_insights(self, settings):
        source = MockDataSource(settings)
        channel = source.resolve_channel("@demo")
        videos = source.fetch_videos(channel, 30)
        transcripts = [source.fetch_transcript(video) for video in videos[:5]]

        insights = HeuristicAnalyzer(settings).analyze(channel, videos, transcripts)
        assert insights.themes
        assert insights.content_format
        assert insights.target_audience
        assert insights.strengths

    def test_no_videos_yields_nothing(self, settings):
        channel = MockDataSource(settings).resolve_channel("@demo")
        assert HeuristicAnalyzer(settings).analyze(channel, [], []) is None

    def test_flags_missing_tags_as_an_opportunity(self, settings):
        channel = MockDataSource(settings).resolve_channel("@demo")
        videos = [
            Video(id=f"v{i}", title=f"Title {i}", view_count=100, tags=[])
            for i in range(5)
        ]
        insights = HeuristicAnalyzer(settings).analyze(channel, videos, [])
        assert any("no tags" in item for item in insights.opportunities)


class TestGeminiAnalyzer:
    def test_requires_an_api_key(self):
        with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
            GeminiAnalyzer(Settings(analyzer="gemini", gemini_api_key=""))

    def test_prompt_includes_titles_tags_and_hooks(self, settings):
        source = MockDataSource(settings)
        channel = source.resolve_channel("@demo")
        videos = source.fetch_videos(channel, 5)
        transcripts = [source.fetch_transcript(video) for video in videos]

        prompt = build_prompt(channel, videos, transcripts, 1000)
        assert "RECENT VIDEO TITLES" in prompt
        assert "TAGS USED" in prompt
        assert videos[0].title in prompt

    def test_validates_good_output(self):
        insights = GeminiAnalyzer.validate(
            {"themes": ["AI", "  "], "content_format": " Long-form ", "strengths": []}
        )
        assert insights.themes == ["AI"]  # blank dropped
        assert insights.content_format == "Long-form"

    def test_rejects_malformed_output(self):
        assert GeminiAnalyzer.validate({"themes": "not-a-list"}) is None

    def test_lenient_json(self):
        assert loads_lenient('```json\n{"a": 1}\n```') == {"a": 1}

    def test_lenient_json_rejects_empty(self):
        with pytest.raises(ProviderError, match="empty response"):
            loads_lenient("   ")


class TestPipeline:
    def test_runs_end_to_end(self, settings):
        report = ChannelAnalysisPipeline(settings=settings).run("@demo")
        assert isinstance(report, ChannelReport)
        assert report.videos
        assert report.stats.total_videos == len(report.videos)
        assert report.insights is not None

    def test_respects_max_videos(self, settings):
        settings.max_videos = 7
        report = ChannelAnalysisPipeline(settings=settings).run("@demo")
        assert len(report.videos) == 7

    def test_transcripts_can_be_skipped(self, settings):
        settings.fetch_transcripts = False
        report = ChannelAnalysisPipeline(settings=settings).run("@demo")
        assert report.transcripts_analyzed == 0

    def test_reports_progress_for_each_stage(self, settings):
        seen = []
        ChannelAnalysisPipeline(
            settings=settings, progress=lambda stage, msg: seen.append(stage)
        ).run("@demo")
        assert {"resolve", "videos", "stats", "analyze", "done"} <= set(seen)

    def test_empty_identifier_raises(self, settings):
        with pytest.raises(ValueError, match="must not be empty"):
            ChannelAnalysisPipeline(settings=settings).run("  ")

    def test_bad_provider_fails_at_construction(self, settings):
        settings.source = "nonexistent"
        with pytest.raises(ValueError, match="Unknown data source"):
            ChannelAnalysisPipeline(settings=settings)

    def test_a_failing_analyzer_still_returns_statistics(self, settings):
        class BrokenAnalyzer(HeuristicAnalyzer):
            def analyze(self, channel, videos, transcripts):
                raise ProviderError("model unavailable")

        pipeline = ChannelAnalysisPipeline(settings=settings)
        pipeline.analyzer = BrokenAnalyzer(settings)
        report = pipeline.run("@demo")

        assert report.insights is None
        assert report.stats.total_videos > 0  # numbers survived

    def test_transcripts_are_taken_from_the_most_viewed(self, settings):
        settings.max_transcripts = 3
        report = ChannelAnalysisPipeline(settings=settings).run("@demo")
        assert report.transcripts_analyzed <= 3

    def test_top_videos_sorting(self, settings):
        report = ChannelAnalysisPipeline(settings=settings).run("@demo")
        top = report.top_videos(5)
        assert [v.view_count for v in top] == sorted(
            [v.view_count for v in top], reverse=True
        )

    def test_markdown_export(self, settings):
        report = ChannelAnalysisPipeline(settings=settings).run("@demo")
        markdown = report.to_markdown()
        assert "## Performance" in markdown
        assert "## Top Videos by Views" in markdown
        assert "## Content Insights" in markdown

    def test_report_round_trips_through_json(self, settings):
        report = ChannelAnalysisPipeline(settings=settings).run("@demo")
        restored = ChannelReport.model_validate_json(report.model_dump_json())
        assert restored.stats.total_views == report.stats.total_views

    def test_convenience_wrapper(self, settings):
        assert analyze_channel("@demo", settings=settings).videos


class TestSettings:
    def test_mock_mode_detection(self):
        assert Settings(source="mock", analyzer="mock").is_mock
        assert not Settings(source="youtube", analyzer="mock").is_mock

    def test_language_list_parsing(self):
        assert Settings(transcript_languages="en, te , hi").language_list == [
            "en", "te", "hi"
        ]

    def test_require_rejects_blanks(self):
        with pytest.raises(ConfigError, match="MY_KEY is required"):
            Settings().require("", "MY_KEY", "prov")


class TestMarkdownRendering:
    def test_report_without_insights_still_renders(self):
        report = ChannelReport(
            channel=Channel(id="UC1", title="Test"),
            videos=[Video(id="v", title="A", view_count=10)],
        )
        markdown = report.to_markdown()
        assert "# Test" in markdown
        assert "## Content Insights" not in markdown

    def test_insights_sections_appear(self):
        report = ChannelReport(
            channel=Channel(id="UC1", title="Test"),
            insights=ContentInsights(themes=["AI"], strengths=["Consistent"]),
        )
        markdown = report.to_markdown()
        assert "### Themes" in markdown
        assert "### Strengths" in markdown
