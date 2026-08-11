"""Tests for the deterministic statistics layer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from yt_analyzer.analytics import (
    compute_stats,
    consistency_score,
    duration_buckets,
    monthly_upload_counts,
    outliers,
    performance_by_duration,
    title_word_frequency,
    top_tags,
    upload_cadence,
)
from yt_analyzer.models import Video, format_duration, parse_duration

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_video(
    index: int = 0,
    views: int = 1000,
    likes: int = 50,
    comments: int = 10,
    duration: int = 600,
    days_ago: int = 0,
    title: str = "A Video Title",
    tags=None,
) -> Video:
    return Video(
        id=f"vid{index}",
        title=title,
        published_at=BASE - timedelta(days=days_ago),
        duration_seconds=duration,
        view_count=views,
        like_count=likes,
        comment_count=comments,
        tags=tags or [],
    )


class TestDurationParsing:
    @pytest.mark.parametrize(
        "iso,expected",
        [
            ("PT1H2M10S", 3730),
            ("PT45S", 45),
            ("PT10M", 600),
            ("PT2H", 7200),
            ("P1DT2H", 93600),
            ("", 0),
            ("garbage", 0),
        ],
    )
    def test_parse(self, iso, expected):
        assert parse_duration(iso) == expected

    @pytest.mark.parametrize(
        "seconds,expected",
        [(45, "0:45"), (600, "10:00"), (3730, "1:02:10"), (0, "0:00"), (-5, "0:00")],
    )
    def test_format(self, seconds, expected):
        assert format_duration(seconds) == expected


class TestVideoProperties:
    def test_engagement_rate(self):
        video = make_video(views=1000, likes=50, comments=10)
        assert video.engagement_rate == pytest.approx(6.0)

    def test_engagement_rate_with_no_views(self):
        assert make_video(views=0).engagement_rate == 0.0

    def test_like_ratio(self):
        assert make_video(views=1000, likes=50).like_ratio == pytest.approx(5.0)

    @pytest.mark.parametrize(
        "duration,expected", [(45, True), (60, True), (61, False), (0, False)]
    )
    def test_is_short(self, duration, expected):
        assert make_video(duration=duration).is_short is expected

    def test_url(self):
        assert make_video(index=7).url == "https://www.youtube.com/watch?v=vid7"

    def test_parses_api_timestamps(self):
        video = Video(id="x", published_at="2026-03-05T10:00:00Z")
        assert video.published_at.year == 2026

    def test_invalid_timestamp_becomes_none(self):
        assert Video(id="x", published_at="not-a-date").published_at is None


class TestComputeStats:
    def test_empty_input(self):
        stats = compute_stats([])
        assert stats.total_videos == 0
        assert stats.median_views == 0.0

    def test_totals_and_averages(self):
        videos = [
            make_video(0, views=100, likes=10, comments=5),
            make_video(1, views=200, likes=20, comments=10),
            make_video(2, views=900, likes=90, comments=45),
        ]
        stats = compute_stats(videos)
        assert stats.total_videos == 3
        assert stats.total_views == 1200
        assert stats.total_likes == 120
        assert stats.total_comments == 60
        assert stats.median_views == 200
        assert stats.mean_views == 400

    def test_median_resists_a_single_outlier(self):
        videos = [make_video(i, views=100) for i in range(9)]
        videos.append(make_video(9, views=1_000_000))
        stats = compute_stats(videos)
        assert stats.median_views == 100
        assert stats.mean_views > stats.median_views

    def test_shorts_split(self):
        videos = [make_video(0, duration=30), make_video(1, duration=600)]
        stats = compute_stats(videos)
        assert stats.shorts_count == 1
        assert stats.longform_count == 1
        assert stats.shorts_share == 50.0

    def test_best_weekday_uses_mean_views_not_count(self):
        # Three low-view Mondays against one high-view Friday.
        videos = [
            make_video(0, views=100, days_ago=0),   # Thursday
            make_video(1, views=100, days_ago=7),
            make_video(2, views=100, days_ago=14),
            make_video(3, views=99_000, days_ago=1),  # Wednesday
        ]
        stats = compute_stats(videos)
        assert stats.weekday_distribution["Thursday"] == 3
        assert stats.best_weekday == "Wednesday"

    def test_handles_videos_without_dates(self):
        videos = [make_video(0), Video(id="undated", view_count=500)]
        stats = compute_stats(videos)
        assert stats.total_videos == 2


class TestUploadCadence:
    def test_needs_at_least_two_dates(self):
        assert upload_cadence([make_video(0)]) == 0.0
        assert upload_cadence([]) == 0.0

    def test_weekly_uploads_are_about_four_per_month(self):
        videos = [make_video(i, days_ago=i * 7) for i in range(9)]
        assert upload_cadence(videos) == pytest.approx(4.3, abs=0.3)

    def test_same_day_uploads_do_not_divide_by_zero(self):
        videos = [make_video(i, days_ago=0) for i in range(3)]
        assert upload_cadence(videos) == 3.0


class TestConsistencyScore:
    def test_needs_at_least_three_dates(self):
        assert consistency_score([make_video(0), make_video(1, days_ago=7)]) is None

    def test_perfectly_even_spacing_scores_100(self):
        videos = [make_video(i, days_ago=i * 7) for i in range(6)]
        assert consistency_score(videos) == 100.0

    def test_erratic_spacing_scores_lower(self):
        videos = [
            make_video(0, days_ago=0),
            make_video(1, days_ago=1),
            make_video(2, days_ago=2),
            make_video(3, days_ago=200),
        ]
        assert consistency_score(videos) < 50


class TestOutliers:
    def test_finds_videos_above_twice_the_median(self):
        videos = [make_video(i, views=100) for i in range(5)]
        videos.append(make_video(5, views=1000))
        found = outliers(videos)
        assert len(found) == 1
        assert found[0].view_count == 1000

    def test_needs_enough_videos(self):
        assert outliers([make_video(0), make_video(1)]) == []

    def test_all_zero_views_yields_nothing(self):
        assert outliers([make_video(i, views=0) for i in range(5)]) == []

    def test_results_are_sorted_descending(self):
        videos = [make_video(i, views=100) for i in range(5)]
        videos += [make_video(5, views=500), make_video(6, views=900)]
        assert [v.view_count for v in outliers(videos)] == [900, 500]


class TestDurationBuckets:
    def test_assigns_each_band(self):
        videos = [
            make_video(0, duration=30),
            make_video(1, duration=200),
            make_video(2, duration=700),
            make_video(3, duration=1200),
            make_video(4, duration=3600),
        ]
        buckets = duration_buckets(videos)
        assert all(count == 1 for count in buckets.values())

    def test_zero_duration_is_skipped(self):
        assert sum(duration_buckets([make_video(0, duration=0)]).values()) == 0

    def test_performance_by_duration_averages_within_bucket(self):
        videos = [
            make_video(0, duration=30, views=100),
            make_video(1, duration=40, views=300),
            make_video(2, duration=700, views=1000),
        ]
        performance = performance_by_duration(videos)
        assert performance["Shorts (≤1m)"] == 200.0
        assert performance["Medium (5-15m)"] == 1000.0


class TestTagsAndTitles:
    def test_top_tags_are_case_normalised(self):
        videos = [
            make_video(0, tags=["Python", "AI"]),
            make_video(1, tags=["python", "rag"]),
        ]
        assert dict(top_tags(videos))["python"] == 2

    def test_a_repeated_tag_counts_once_per_video(self):
        videos = [make_video(0, tags=["python", "python", "PYTHON"])]
        assert dict(top_tags(videos))["python"] == 1

    def test_title_words_drop_stopwords_and_singletons(self):
        videos = [
            make_video(0, title="The Truth About Retrieval"),
            make_video(1, title="The Truth About Agents"),
        ]
        words = dict(title_word_frequency(videos))
        assert words.get("truth") == 2
        assert "the" not in words       # stopword
        assert "retrieval" not in words  # appears once only


class TestMonthlyCounts:
    def test_groups_and_orders_by_month(self):
        videos = [
            make_video(0, days_ago=0),    # 2026-01
            make_video(1, days_ago=40),   # 2025-11
            make_video(2, days_ago=45),   # 2025-11
        ]
        counts = monthly_upload_counts(videos)
        assert list(counts) == sorted(counts)
        assert counts["2025-11"] == 2

    def test_undated_videos_are_skipped(self):
        assert monthly_upload_counts([Video(id="x")]) == {}
