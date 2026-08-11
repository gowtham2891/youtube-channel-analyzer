"""Domain models for channel, video, transcript and analysis data."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

#: ISO-8601 duration as returned by the YouTube Data API, e.g. "PT1H2M10S".
_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?T?(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?$"
)


def parse_duration(iso: str) -> int:
    """Convert an ISO-8601 duration to seconds. Unparseable input yields 0."""
    if not iso:
        return 0
    match = _DURATION_RE.match(iso.strip())
    if not match:
        return 0
    parts = {key: int(value or 0) for key, value in match.groupdict().items()}
    return (
        parts["days"] * 86400
        + parts["hours"] * 3600
        + parts["minutes"] * 60
        + parts["seconds"]
    )


def format_duration(seconds: int) -> str:
    """Render seconds as ``M:SS`` or ``H:MM:SS``."""
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class Channel(BaseModel):
    """Channel-level metadata."""

    id: str
    title: str = Field(default="")
    description: str = Field(default="")
    subscriber_count: int = Field(default=0, ge=0)
    video_count: int = Field(default=0, ge=0)
    view_count: int = Field(default=0, ge=0)
    uploads_playlist_id: str = Field(default="")
    published_at: Optional[datetime] = None
    thumbnail_url: str = Field(default="")

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/channel/{self.id}"

    @property
    def average_views_per_video(self) -> float:
        return self.view_count / self.video_count if self.video_count else 0.0


class Video(BaseModel):
    """One uploaded video with its public statistics."""

    id: str
    title: str = Field(default="")
    description: str = Field(default="")
    published_at: Optional[datetime] = None
    duration_seconds: int = Field(default=0, ge=0)
    view_count: int = Field(default=0, ge=0)
    like_count: int = Field(default=0, ge=0)
    comment_count: int = Field(default=0, ge=0)
    tags: List[str] = Field(default_factory=list)
    thumbnail_url: str = Field(default="")

    @field_validator("published_at", mode="before")
    @classmethod
    def _parse_timestamp(cls, value):
        """Accept the API's ``Z``-suffixed RFC 3339 timestamps."""
        if isinstance(value, str) and value:
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        return value

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.id}"

    @property
    def is_short(self) -> bool:
        """YouTube Shorts are 60 seconds or less."""
        return 0 < self.duration_seconds <= 60

    @property
    def engagement_rate(self) -> float:
        """(likes + comments) / views, as a percentage."""
        if not self.view_count:
            return 0.0
        return (self.like_count + self.comment_count) / self.view_count * 100

    @property
    def like_ratio(self) -> float:
        return self.like_count / self.view_count * 100 if self.view_count else 0.0

    @property
    def duration_label(self) -> str:
        return format_duration(self.duration_seconds)

    @property
    def age_days(self) -> Optional[int]:
        if not self.published_at:
            return None
        now = datetime.now(timezone.utc)
        published = self.published_at
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        return max((now - published).days, 0)

    @property
    def views_per_day(self) -> float:
        age = self.age_days
        if not age:
            return float(self.view_count)
        return self.view_count / age


class TranscriptSegment(BaseModel):
    start: float = Field(default=0.0, ge=0)
    duration: float = Field(default=0.0, ge=0)
    text: str = Field(default="")


class Transcript(BaseModel):
    """A video transcript, if one was available."""

    video_id: str
    segments: List[TranscriptSegment] = Field(default_factory=list)
    language: str = Field(default="")
    is_generated: bool = Field(default=True)
    error: str = Field(default="")

    @property
    def text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments).strip()

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def available(self) -> bool:
        return bool(self.segments)

    def opening(self, seconds: float = 30.0) -> str:
        """The first ``seconds`` of speech — the hook."""
        return " ".join(
            segment.text.strip()
            for segment in self.segments
            if segment.start < seconds
        ).strip()


class ContentInsights(BaseModel):
    """What the LLM concluded from reading the transcripts."""

    themes: List[str] = Field(default_factory=list)
    hook_patterns: List[str] = Field(default_factory=list)
    content_format: str = Field(default="")
    target_audience: str = Field(default="")
    strengths: List[str] = Field(default_factory=list)
    opportunities: List[str] = Field(default_factory=list)
    recommended_topics: List[str] = Field(default_factory=list)


class ChannelStats(BaseModel):
    """Deterministic statistics computed from video metadata."""

    total_videos: int = 0
    total_views: int = 0
    total_likes: int = 0
    total_comments: int = 0
    median_views: float = 0.0
    mean_views: float = 0.0
    mean_engagement_rate: float = 0.0
    mean_duration_seconds: float = 0.0
    shorts_count: int = 0
    longform_count: int = 0
    uploads_per_month: float = 0.0
    best_weekday: str = ""
    weekday_distribution: Dict[str, int] = Field(default_factory=dict)
    weekday_mean_views: Dict[str, float] = Field(default_factory=dict)
    top_tags: List[tuple] = Field(default_factory=list)
    title_word_frequency: List[tuple] = Field(default_factory=list)

    @property
    def shorts_share(self) -> float:
        return self.shorts_count / self.total_videos * 100 if self.total_videos else 0.0


class ChannelReport(BaseModel):
    """The complete analysis of one channel."""

    channel: Channel
    videos: List[Video] = Field(default_factory=list)
    stats: ChannelStats = Field(default_factory=ChannelStats)
    insights: Optional[ContentInsights] = None
    transcripts_analyzed: int = 0
    elapsed_seconds: float = Field(default=0.0, ge=0)

    def top_videos(self, limit: int = 5, key: str = "view_count") -> List[Video]:
        """Top videos by any numeric attribute or property."""
        return sorted(
            self.videos, key=lambda video: getattr(video, key, 0), reverse=True
        )[:limit]

    def to_markdown(self) -> str:
        channel = self.channel
        stats = self.stats
        lines = [f"# {channel.title}", ""]

        lines += [
            f"**Subscribers:** {channel.subscriber_count:,}  |  "
            f"**Videos:** {channel.video_count:,}  |  "
            f"**Total views:** {channel.view_count:,}",
            "",
            f"[{channel.url}]({channel.url})",
            "",
            "## Performance",
            "",
            f"- Analyzed **{stats.total_videos}** recent videos",
            f"- Median views: **{stats.median_views:,.0f}** "
            f"(mean {stats.mean_views:,.0f})",
            f"- Mean engagement rate: **{stats.mean_engagement_rate:.2f}%**",
            f"- Mean duration: **{format_duration(int(stats.mean_duration_seconds))}**",
            f"- Shorts: **{stats.shorts_count}** ({stats.shorts_share:.0f}%)  |  "
            f"Long-form: **{stats.longform_count}**",
            f"- Upload cadence: **{stats.uploads_per_month:.1f}/month**",
        ]
        if stats.best_weekday:
            lines.append(f"- Best performing upload day: **{stats.best_weekday}**")
        lines.append("")

        top = self.top_videos(5)
        if top:
            lines += [
                "## Top Videos by Views",
                "",
                "| Views | Engagement | Duration | Title |",
                "| ---: | ---: | ---: | --- |",
            ]
            for video in top:
                lines.append(
                    f"| {video.view_count:,} | {video.engagement_rate:.2f}% | "
                    f"{video.duration_label} | [{video.title}]({video.url}) |"
                )
            lines.append("")

        if stats.top_tags:
            tags = ", ".join(f"{tag} ({count})" for tag, count in stats.top_tags[:12])
            lines += ["## Most Used Tags", "", tags, ""]

        if stats.title_word_frequency:
            words = ", ".join(
                f"{word} ({count})" for word, count in stats.title_word_frequency[:12]
            )
            lines += ["## Recurring Title Words", "", words, ""]

        if self.insights:
            insights = self.insights
            lines += ["## Content Insights", ""]
            if insights.content_format:
                lines += [f"**Format:** {insights.content_format}", ""]
            if insights.target_audience:
                lines += [f"**Audience:** {insights.target_audience}", ""]
            for heading, values in (
                ("Themes", insights.themes),
                ("Hook patterns", insights.hook_patterns),
                ("Strengths", insights.strengths),
                ("Opportunities", insights.opportunities),
                ("Recommended topics", insights.recommended_topics),
            ):
                if values:
                    lines += [f"### {heading}", ""]
                    lines += [f"- {value}" for value in values]
                    lines.append("")

        lines += [
            "---",
            "",
            f"*Analyzed {stats.total_videos} videos and "
            f"{self.transcripts_analyzed} transcript(s) in "
            f"{self.elapsed_seconds:.1f}s.*",
        ]
        return "\n".join(lines).rstrip() + "\n"
