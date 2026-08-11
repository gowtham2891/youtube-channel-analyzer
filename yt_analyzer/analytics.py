"""Deterministic statistics over video metadata.

Pure Python on purpose: these numbers are the factual backbone of the report,
so they are computed and tested independently of any model. The LLM interprets
them; it never produces them.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .models import ChannelStats, Video

WEEKDAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]

#: Filler words that dominate any title corpus without carrying meaning.
TITLE_STOPWORDS = {
    "the", "and", "for", "with", "you", "your", "how", "why", "what", "this",
    "that", "from", "are", "was", "will", "can", "our", "its", "it's", "all",
    "new", "get", "not", "but", "his", "her", "she", "him", "they", "them",
    "have", "has", "did", "does", "into", "out", "off", "top", "best", "vs",
    "part", "ep", "episode", "full", "video", "shorts", "official",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]{2,}")


def _aware(moment: datetime) -> datetime:
    """Normalise to timezone-aware UTC so comparisons never raise."""
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment


def compute_stats(videos: List[Video]) -> ChannelStats:
    """Compute every deterministic statistic from ``videos``."""
    if not videos:
        return ChannelStats()

    views = [video.view_count for video in videos]
    engagement = [video.engagement_rate for video in videos]
    durations = [video.duration_seconds for video in videos if video.duration_seconds]

    weekday_counts, weekday_views = _weekday_breakdown(videos)
    weekday_means = {
        day: statistics.mean(values) for day, values in weekday_views.items() if values
    }
    best_weekday = (
        max(weekday_means, key=lambda day: weekday_means[day]) if weekday_means else ""
    )

    return ChannelStats(
        total_videos=len(videos),
        total_views=sum(views),
        total_likes=sum(video.like_count for video in videos),
        total_comments=sum(video.comment_count for video in videos),
        median_views=statistics.median(views),
        mean_views=statistics.mean(views),
        mean_engagement_rate=statistics.mean(engagement) if engagement else 0.0,
        mean_duration_seconds=statistics.mean(durations) if durations else 0.0,
        shorts_count=sum(1 for video in videos if video.is_short),
        longform_count=sum(1 for video in videos if not video.is_short),
        uploads_per_month=upload_cadence(videos),
        best_weekday=best_weekday,
        weekday_distribution=weekday_counts,
        weekday_mean_views={day: round(value, 1) for day, value in weekday_means.items()},
        top_tags=top_tags(videos),
        title_word_frequency=title_word_frequency(videos),
    )


def _weekday_breakdown(
    videos: List[Video],
) -> Tuple[Dict[str, int], Dict[str, List[int]]]:
    counts: Dict[str, int] = {}
    views: Dict[str, List[int]] = {}
    for video in videos:
        if not video.published_at:
            continue
        day = WEEKDAYS[_aware(video.published_at).weekday()]
        counts[day] = counts.get(day, 0) + 1
        views.setdefault(day, []).append(video.view_count)
    return counts, views


def upload_cadence(videos: List[Video]) -> float:
    """Average uploads per month across the observed publishing window.

    Counts *intervals*, not videos: n uploads span only n-1 gaps, so dividing
    n by the first-to-last span would overstate the rate by a factor of
    n/(n-1). At weekly cadence that is the difference between reporting 4.9
    and the correct 4.3 uploads per month.

    Returns 0.0 when fewer than two videos carry a date, since a rate cannot
    be inferred from a single point.
    """
    dated = [_aware(video.published_at) for video in videos if video.published_at]
    if len(dated) < 2:
        return 0.0

    span_days = (max(dated) - min(dated)).days
    if span_days <= 0:
        # Every upload landed on one day; report the raw count.
        return float(len(dated))
    return (len(dated) - 1) / (span_days / 30.44)


def top_tags(videos: List[Video], limit: int = 20) -> List[Tuple[str, int]]:
    """Most frequently applied tags, case-normalised."""
    counter: Counter = Counter()
    for video in videos:
        # Count each tag once per video even if repeated in its own list.
        counter.update({tag.strip().lower() for tag in video.tags if tag.strip()})
    return counter.most_common(limit)


def title_word_frequency(
    videos: List[Video], limit: int = 20
) -> List[Tuple[str, int]]:
    """Recurring meaningful words across titles."""
    counter: Counter = Counter()
    for video in videos:
        words = {
            word.lower()
            for word in _WORD_RE.findall(video.title)
            if word.lower() not in TITLE_STOPWORDS
        }
        counter.update(words)
    return [(word, count) for word, count in counter.most_common(limit) if count > 1]


def outliers(videos: List[Video], factor: float = 2.0) -> List[Video]:
    """Videos whose views exceed ``factor`` times the median.

    The median rather than the mean, because a single viral video would drag a
    mean high enough to hide every other overperformer.
    """
    if len(videos) < 3:
        return []
    median = statistics.median([video.view_count for video in videos])
    if median <= 0:
        return []
    return sorted(
        [video for video in videos if video.view_count > median * factor],
        key=lambda video: -video.view_count,
    )


def duration_buckets(videos: List[Video]) -> Dict[str, int]:
    """Group videos into human-meaningful length bands."""
    buckets = {
        "Shorts (≤1m)": 0,
        "Short (1-5m)": 0,
        "Medium (5-15m)": 0,
        "Long (15-30m)": 0,
        "Very long (>30m)": 0,
    }
    for video in videos:
        seconds = video.duration_seconds
        if seconds <= 0:
            continue
        if seconds <= 60:
            buckets["Shorts (≤1m)"] += 1
        elif seconds <= 300:
            buckets["Short (1-5m)"] += 1
        elif seconds <= 900:
            buckets["Medium (5-15m)"] += 1
        elif seconds <= 1800:
            buckets["Long (15-30m)"] += 1
        else:
            buckets["Very long (>30m)"] += 1
    return buckets


def performance_by_duration(videos: List[Video]) -> Dict[str, float]:
    """Mean views per duration bucket — does length correlate with reach?"""
    grouped: Dict[str, List[int]] = {}
    for video in videos:
        if video.duration_seconds <= 0:
            continue
        for label, upper in (
            ("Shorts (≤1m)", 60),
            ("Short (1-5m)", 300),
            ("Medium (5-15m)", 900),
            ("Long (15-30m)", 1800),
            ("Very long (>30m)", float("inf")),
        ):
            if video.duration_seconds <= upper:
                grouped.setdefault(label, []).append(video.view_count)
                break
    return {
        label: round(statistics.mean(values), 1) for label, values in grouped.items()
    }


def monthly_upload_counts(videos: List[Video]) -> Dict[str, int]:
    """Uploads per ``YYYY-MM``, chronologically ordered."""
    counts: Dict[str, int] = {}
    for video in videos:
        if not video.published_at:
            continue
        key = _aware(video.published_at).strftime("%Y-%m")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def consistency_score(videos: List[Video]) -> Optional[float]:
    """0-100 score for how regular the upload gaps are.

    100 means perfectly even spacing. Returns ``None`` when there are too few
    dated videos to say anything.
    """
    dated = sorted(_aware(v.published_at) for v in videos if v.published_at)
    if len(dated) < 3:
        return None

    gaps = [
        (later - earlier).total_seconds() / 86400
        for earlier, later in zip(dated, dated[1:])
    ]
    mean_gap = statistics.mean(gaps)
    if mean_gap <= 0:
        return 100.0

    # Coefficient of variation: lower spread means more consistent uploads.
    spread = statistics.pstdev(gaps) / mean_gap
    return round(max(0.0, min(100.0, (1 - spread) * 100)), 1)
