"""Streamlit UI for the YouTube Channel Analyzer.

    streamlit run app.py

Chart notes: every chart here is single-series (one hue, magnitude or
change-over-time), so no categorical palette and no legend is needed. There is
deliberately no dual-axis chart anywhere — views and upload counts live on
separate charts rather than sharing a distorted scale. Each chart ships with a
table view beside it.
"""

from __future__ import annotations

import streamlit as st

from yt_analyzer import __version__, get_settings
from yt_analyzer.analytics import (
    consistency_score,
    duration_buckets,
    monthly_upload_counts,
    outliers,
    performance_by_duration,
)
from yt_analyzer.config import ConfigError
from yt_analyzer.models import ChannelReport, format_duration
from yt_analyzer.pipeline import ChannelAnalysisPipeline
from yt_analyzer.providers.base import (
    ProviderError,
    available_analyzers,
    available_sources,
)

st.set_page_config(
    page_title="YouTube Channel Analyzer", page_icon="📺", layout="wide"
)

#: Single accent hue, validated for contrast against both chart surfaces.
ACCENT = "#2E6BE6"

WEEKDAY_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]


def build_settings():
    settings = get_settings(refresh=True)

    with st.sidebar:
        st.title("📺 Channel Analyzer")
        st.caption(f"v{__version__}")
        st.divider()

        st.subheader("Providers")
        sources = available_sources()
        settings.source = st.selectbox(
            "Data source",
            sources,
            index=sources.index(settings.source) if settings.source in sources else 0,
            help="`mock` generates a synthetic channel — no API key needed.",
        )
        analyzers = available_analyzers()
        settings.analyzer = st.selectbox(
            "Content analyzer",
            analyzers,
            index=analyzers.index(settings.analyzer)
            if settings.analyzer in analyzers
            else 0,
            help="`mock` uses offline heuristics instead of Gemini.",
        )

        st.subheader("Scope")
        settings.max_videos = st.slider("Videos to fetch", 5, 200, settings.max_videos)
        settings.fetch_transcripts = st.toggle(
            "Read transcripts", value=settings.fetch_transcripts
        )
        if settings.fetch_transcripts:
            settings.max_transcripts = st.slider(
                "Transcripts to read", 1, 30, settings.max_transcripts
            )

        st.divider()
        if settings.is_mock:
            st.success("Mock mode — no credentials required.")
        else:
            st.info("Live mode — reading credentials from .env")
            if settings.source == "youtube" and not settings.youtube_api_key:
                st.error("YOUTUBE_API_KEY is not set.")

    return settings


def render_overview(report: ChannelReport) -> None:
    channel, stats = report.channel, report.stats

    header, thumb = st.columns([4, 1])
    with header:
        st.subheader(channel.title)
        st.caption(channel.url)
        if channel.description:
            st.write(channel.description[:400])
    with thumb:
        if channel.thumbnail_url and not channel.thumbnail_url.endswith(
            "mock-channel.jpg"
        ):
            st.image(channel.thumbnail_url, width=120)

    row1 = st.columns(4)
    row1[0].metric("Subscribers", f"{channel.subscriber_count:,}")
    row1[1].metric("Total videos", f"{channel.video_count:,}")
    row1[2].metric("Total views", f"{channel.view_count:,}")
    row1[3].metric("Analyzed", f"{stats.total_videos}")

    row2 = st.columns(4)
    row2[0].metric("Median views", f"{stats.median_views:,.0f}")
    row2[1].metric("Mean engagement", f"{stats.mean_engagement_rate:.2f}%")
    row2[2].metric(
        "Mean duration", format_duration(int(stats.mean_duration_seconds))
    )
    row2[3].metric("Uploads / month", f"{stats.uploads_per_month:.1f}")

    score = consistency_score(report.videos)
    if score is not None:
        st.progress(
            score / 100,
            text=f"Upload consistency: {score:.0f}/100 "
            f"(100 = perfectly even spacing)",
        )


def render_charts(report: ChannelReport) -> None:
    st.markdown("#### Upload cadence over time")
    monthly = monthly_upload_counts(report.videos)
    if monthly:
        chart_col, table_col = st.columns([3, 1])
        with chart_col:
            st.line_chart(
                {"Uploads": list(monthly.values())},
                x=None,
                color=ACCENT,
                height=240,
            )
            st.caption(
                f"Uploads per month, {list(monthly)[0]} to {list(monthly)[-1]}"
            )
        with table_col:
            st.dataframe(
                [{"Month": month, "Uploads": count} for month, count in monthly.items()],
                hide_index=True,
                width="stretch",
                height=240,
            )
    else:
        st.info("No publish dates available for these videos.")

    st.markdown("#### Mean views by upload day")
    weekday_means = report.stats.weekday_mean_views
    if weekday_means:
        ordered = {
            day: weekday_means.get(day, 0.0)
            for day in WEEKDAY_ORDER
            if day in weekday_means
        }
        chart_col, table_col = st.columns([3, 1])
        with chart_col:
            st.bar_chart(
                {"Mean views": list(ordered.values())}, color=ACCENT, height=240
            )
            st.caption(" · ".join(ordered))
        with table_col:
            st.dataframe(
                [
                    {
                        "Day": day,
                        "Videos": report.stats.weekday_distribution.get(day, 0),
                        "Mean views": round(value),
                    }
                    for day, value in ordered.items()
                ],
                hide_index=True,
                width="stretch",
                height=240,
            )
    else:
        st.info("No weekday data available.")

    st.markdown("#### Mean views by video length")
    performance = performance_by_duration(report.videos)
    if performance:
        counts = duration_buckets(report.videos)
        chart_col, table_col = st.columns([3, 1])
        with chart_col:
            st.bar_chart(
                {"Mean views": list(performance.values())}, color=ACCENT, height=240
            )
            st.caption(" · ".join(performance))
        with table_col:
            st.dataframe(
                [
                    {
                        "Length": label,
                        "Videos": counts.get(label, 0),
                        "Mean views": round(value),
                    }
                    for label, value in performance.items()
                ],
                hide_index=True,
                width="stretch",
                height=240,
            )
    else:
        st.info("No duration data available.")


def render_videos(report: ChannelReport) -> None:
    over = outliers(report.videos)
    if over:
        st.success(
            f"{len(over)} video(s) beat the median by more than 2x — "
            f"the strongest signal of what works on this channel."
        )
        for video in over[:5]:
            st.markdown(
                f"- **{video.view_count:,} views** — "
                f"[{video.title}]({video.url}) · {video.duration_label}"
            )

    sort_key = st.selectbox(
        "Sort by",
        ["view_count", "engagement_rate", "like_count", "views_per_day", "duration_seconds"],
        format_func=lambda key: key.replace("_", " ").title(),
    )
    st.dataframe(
        [
            {
                "Views": video.view_count,
                "Likes": video.like_count,
                "Comments": video.comment_count,
                "Engagement %": round(video.engagement_rate, 2),
                "Views/day": round(video.views_per_day, 1),
                "Length": video.duration_label,
                "Published": video.published_at.date().isoformat()
                if video.published_at
                else "—",
                "Title": video.title,
                "URL": video.url,
            }
            for video in report.top_videos(len(report.videos), key=sort_key)
        ],
        width="stretch",
        hide_index=True,
        column_config={"URL": st.column_config.LinkColumn("Link", display_text="open")},
    )


def render_insights(report: ChannelReport) -> None:
    insights = report.insights
    if not insights:
        st.info(
            "No qualitative insights were produced — the analyzer was "
            "unavailable or returned nothing."
        )
        return

    st.caption(
        f"Derived from {report.transcripts_analyzed} transcript(s) and "
        f"{len(report.videos)} video titles. All numbers elsewhere in this "
        f"report are computed directly, not by the model."
    )

    if insights.content_format:
        st.markdown(f"**Format:** {insights.content_format}")
    if insights.target_audience:
        st.markdown(f"**Audience:** {insights.target_audience}")

    left, right = st.columns(2)
    with left:
        for heading, values in (
            ("Themes", insights.themes),
            ("Hook patterns", insights.hook_patterns),
        ):
            if values:
                st.markdown(f"##### {heading}")
                for value in values:
                    st.markdown(f"- {value}")
    with right:
        for heading, values in (
            ("Strengths", insights.strengths),
            ("Opportunities", insights.opportunities),
            ("Recommended next", insights.recommended_topics),
        ):
            if values:
                st.markdown(f"##### {heading}")
                for value in values:
                    st.markdown(f"- {value}")


def main() -> None:
    settings = build_settings()

    st.title("YouTube Channel Analyzer")
    st.caption(
        "Fetches a channel's uploads and transcripts, computes the numbers "
        "directly, and uses an LLM only for the qualitative read."
    )

    identifier = st.text_input(
        "Channel",
        value="@demo" if settings.source == "mock" else "@mkbhd",
        help="A handle (@name), a full URL, or a raw UC… channel id.",
    )

    if st.button("Analyze", type="primary", width="stretch"):
        if not identifier.strip():
            st.warning("Enter a channel first.")
            return

        log_area = st.empty()
        messages: list[str] = []

        def progress(stage: str, message: str) -> None:
            messages.append(f"**{stage}** — {message}")
            log_area.info("\n\n".join(messages[-10:]))

        try:
            with st.spinner("Analyzing…"):
                pipeline = ChannelAnalysisPipeline(settings=settings, progress=progress)
                report = pipeline.run(identifier)
        except (ConfigError, ValueError) as exc:
            st.error(str(exc))
            return
        except ProviderError as exc:
            st.error(f"Provider error: {exc}")
            return

        log_area.empty()
        st.session_state["report"] = report

    report = st.session_state.get("report")
    if not report:
        st.info("Enter a channel and press **Analyze** to begin.")
        return

    overview, charts, videos, insights, export = st.tabs(
        ["Overview", "Trends", "Videos", "Insights", "Export"]
    )

    with overview:
        render_overview(report)
    with charts:
        render_charts(report)
    with videos:
        render_videos(report)
    with insights:
        render_insights(report)
    with export:
        markdown = report.to_markdown()
        st.download_button(
            "Download Markdown", markdown, file_name="channel-report.md",
            mime="text/markdown",
        )
        st.download_button(
            "Download JSON", report.model_dump_json(indent=2),
            file_name="channel-report.json", mime="application/json",
        )
        st.code(markdown, language="markdown")


if __name__ == "__main__":
    main()
