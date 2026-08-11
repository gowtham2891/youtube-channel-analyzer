"""YouTube Channel Analyzer.

Analyzes a channel's uploads and transcripts to surface content patterns,
top-performing topics and posting insights.

Importing this package registers every built-in provider.
"""

from __future__ import annotations

from .analytics import (
    compute_stats,
    consistency_score,
    duration_buckets,
    monthly_upload_counts,
    outliers,
    performance_by_duration,
)
from .config import ConfigError, Settings, get_settings
from .health import (
    CheckResult,
    MissingCredential,
    check_gemini,
    check_settings,
    check_youtube,
    credentials_ready,
    missing_credentials,
)
from .models import (
    Channel,
    ChannelReport,
    ChannelStats,
    ContentInsights,
    Transcript,
    Video,
)
from .pipeline import ChannelAnalysisPipeline, analyze_channel
from .providers.base import ProviderError

# Importing the provider modules populates the registries.
from .providers import mock as _mock  # noqa: F401
from .providers import youtube as _youtube  # noqa: F401
from .providers import gemini as _gemini  # noqa: F401

__version__ = "1.0.0"

__all__ = [
    "Channel",
    "ChannelAnalysisPipeline",
    "ChannelReport",
    "ChannelStats",
    "CheckResult",
    "ConfigError",
    "ContentInsights",
    "MissingCredential",
    "ProviderError",
    "Settings",
    "Transcript",
    "Video",
    "analyze_channel",
    "check_gemini",
    "check_settings",
    "check_youtube",
    "compute_stats",
    "consistency_score",
    "credentials_ready",
    "duration_buckets",
    "get_settings",
    "missing_credentials",
    "monthly_upload_counts",
    "outliers",
    "performance_by_duration",
    "__version__",
]
