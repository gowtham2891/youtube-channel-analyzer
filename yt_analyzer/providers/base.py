"""Provider interfaces and registries for data sources and content analyzers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

from ..config import Settings
from ..models import Channel, ContentInsights, Transcript, Video


class DataSource(ABC):
    """Supplies channel metadata, videos and transcripts."""

    name: str = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def resolve_channel(self, identifier: str) -> Channel:
        """Resolve a handle, URL, username or raw id into a :class:`Channel`."""

    @abstractmethod
    def fetch_videos(self, channel: Channel, limit: int) -> List[Video]:
        """Fetch up to ``limit`` of the channel's most recent uploads."""

    @abstractmethod
    def fetch_transcript(self, video: Video) -> Transcript:
        """Fetch a transcript, returning one with ``error`` set on failure."""


class ContentAnalyzer(ABC):
    """Turns videos and transcripts into qualitative insights."""

    name: str = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def analyze(
        self,
        channel: Channel,
        videos: List[Video],
        transcripts: List[Transcript],
    ) -> Optional[ContentInsights]:
        """Produce insights, or ``None`` if none could be derived."""


class ProviderError(RuntimeError):
    """Raised when a provider fails unrecoverably."""


_SOURCES: Dict[str, Callable[[Settings], DataSource]] = {}
_ANALYZERS: Dict[str, Callable[[Settings], ContentAnalyzer]] = {}


def register_source(name: str, factory: Callable[[Settings], DataSource]) -> None:
    _SOURCES[name] = factory


def register_analyzer(name: str, factory: Callable[[Settings], ContentAnalyzer]) -> None:
    _ANALYZERS[name] = factory


def available_sources() -> List[str]:
    return sorted(_SOURCES)


def available_analyzers() -> List[str]:
    return sorted(_ANALYZERS)


def get_source(settings: Settings) -> DataSource:
    try:
        factory = _SOURCES[settings.source]
    except KeyError:
        raise ValueError(
            f"Unknown data source '{settings.source}'. "
            f"Available: {', '.join(available_sources())}"
        ) from None
    return factory(settings)


def get_analyzer(settings: Settings) -> ContentAnalyzer:
    try:
        factory = _ANALYZERS[settings.analyzer]
    except KeyError:
        raise ValueError(
            f"Unknown analyzer '{settings.analyzer}'. "
            f"Available: {', '.join(available_analyzers())}"
        ) from None
    return factory(settings)
