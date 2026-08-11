"""Orchestrates channel resolution, video fetching, stats and LLM analysis."""

from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional

from .analytics import compute_stats
from .config import Settings, get_settings
from .models import ChannelReport, Transcript
from .providers.base import ProviderError, get_analyzer, get_source

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str], None]


def _noop(stage: str, message: str) -> None:
    logger.info("[%s] %s", stage, message)


class ChannelAnalysisPipeline:
    """resolve -> fetch videos -> compute stats -> fetch transcripts -> analyze.

    Statistics are computed before the LLM stage, so a failed or skipped
    analysis still yields a complete quantitative report.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.progress = progress or _noop
        self.source = get_source(self.settings)
        self.analyzer = get_analyzer(self.settings)

    def run(self, identifier: str) -> ChannelReport:
        identifier = (identifier or "").strip()
        if not identifier:
            raise ValueError("Channel identifier must not be empty.")

        started = time.monotonic()

        self.progress("resolve", f"Resolving {identifier!r} via {self.source.name}")
        channel = self.source.resolve_channel(identifier)
        self.progress(
            "resolve",
            f"{channel.title} — {channel.subscriber_count:,} subscribers, "
            f"{channel.video_count:,} videos",
        )

        self.progress("videos", f"Fetching up to {self.settings.max_videos} uploads")
        videos = self.source.fetch_videos(channel, self.settings.max_videos)
        if not videos:
            raise ProviderError(f"No videos found for channel '{channel.title}'.")
        self.progress("videos", f"Fetched {len(videos)} video(s)")

        self.progress("stats", "Computing statistics")
        stats = compute_stats(videos)
        self.progress(
            "stats",
            f"median {stats.median_views:,.0f} views · "
            f"{stats.mean_engagement_rate:.2f}% engagement · "
            f"{stats.uploads_per_month:.1f} uploads/month",
        )

        transcripts = self._fetch_transcripts(videos)

        self.progress("analyze", f"Analyzing content via {self.analyzer.name}")
        try:
            insights = self.analyzer.analyze(channel, videos, transcripts)
        except ProviderError as exc:
            # The numbers are already computed; a failed LLM call must not
            # discard them.
            self.progress("analyze", f"Analysis failed ({exc}); returning stats only")
            insights = None

        elapsed = time.monotonic() - started
        self.progress("done", f"Completed in {elapsed:.1f}s")

        return ChannelReport(
            channel=channel,
            videos=videos,
            stats=stats,
            insights=insights,
            transcripts_analyzed=sum(1 for t in transcripts if t.available),
            elapsed_seconds=elapsed,
        )

    def _fetch_transcripts(self, videos) -> List[Transcript]:
        if not self.settings.fetch_transcripts or self.settings.max_transcripts <= 0:
            self.progress("transcripts", "Skipped")
            return []

        # Most-viewed first: the videos that worked are the ones worth reading.
        ranked = sorted(videos, key=lambda video: -video.view_count)
        targets = ranked[: self.settings.max_transcripts]

        self.progress("transcripts", f"Fetching {len(targets)} transcript(s)")
        transcripts = [self.source.fetch_transcript(video) for video in targets]

        available = sum(1 for transcript in transcripts if transcript.available)
        self.progress(
            "transcripts", f"{available}/{len(targets)} transcript(s) available"
        )
        return transcripts


def analyze_channel(
    identifier: str,
    settings: Optional[Settings] = None,
    progress: Optional[ProgressCallback] = None,
) -> ChannelReport:
    """Convenience wrapper for one-shot use."""
    return ChannelAnalysisPipeline(settings=settings, progress=progress).run(identifier)
