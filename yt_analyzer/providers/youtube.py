"""YouTube Data API v3 data source, plus transcript retrieval.

Channel identifiers are accepted in every form a user is likely to paste: a
handle (``@mkbhd``), a full URL, a legacy ``/user/`` path, or a raw ``UC...`` id.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import httpx

from ..config import Settings
from ..models import Channel, Transcript, TranscriptSegment, Video, parse_duration
from .base import DataSource, ProviderError, register_source

logger = logging.getLogger(__name__)

#: The API caps a videos.list request at 50 ids.
BATCH_SIZE = 50

_CHANNEL_ID_RE = re.compile(r"^UC[\w-]{22}$")


def parse_channel_identifier(raw: str) -> tuple:
    """Classify a channel identifier.

    Returns ``(kind, value)`` where kind is ``id``, ``handle`` or ``username``.
    """
    value = (raw or "").strip()
    if not value:
        raise ValueError("Channel identifier must not be empty.")

    if value.startswith("http://") or value.startswith("https://"):
        parsed = urlparse(value)
        parts = [part for part in parsed.path.split("/") if part]

        if parts:
            if parts[0] == "channel" and len(parts) > 1:
                return "id", parts[1]
            if parts[0] == "user" and len(parts) > 1:
                return "username", parts[1]
            if parts[0].startswith("@"):
                return "handle", parts[0][1:]
            if parts[0] in {"c", "watch"} and len(parts) > 1:
                # /c/Name is a legacy vanity URL; treat it as a handle lookup.
                return "handle", parts[1]
        query = parse_qs(parsed.query)
        if "channel_id" in query:
            return "id", query["channel_id"][0]
        raise ValueError(f"Could not find a channel in URL: {raw}")

    if value.startswith("@"):
        return "handle", value[1:]
    if _CHANNEL_ID_RE.match(value):
        return "id", value
    return "handle", value


class YouTubeDataSource(DataSource):
    """Reads channels, videos and transcripts from public YouTube APIs."""

    name = "youtube"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.api_key = settings.require(
            settings.youtube_api_key, "YOUTUBE_API_KEY", "youtube"
        )
        self.base_url = settings.youtube_base_url.rstrip("/")

    # -- HTTP ---------------------------------------------------------------

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """GET with retries; turns API quota and auth errors into clear messages."""
        params = {**params, "key": self.api_key}
        url = f"{self.base_url}/{path}"
        last_error: Exception | None = None

        for attempt in range(1, self.settings.max_retries + 1):
            try:
                response = httpx.get(
                    url, params=params, timeout=self.settings.request_timeout
                )
                if response.status_code == 403:
                    raise ProviderError(
                        "YouTube API returned 403 — the key is invalid, the "
                        "Data API is not enabled for the project, or the daily "
                        "quota is exhausted."
                    )
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"retryable status {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response.json()
            except ProviderError:
                raise
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_error = exc
                if attempt == self.settings.max_retries:
                    break
                backoff = 2 ** (attempt - 1)
                logger.warning(
                    "YouTube API %s failed (attempt %d/%d): %s -- retrying in %ss",
                    path, attempt, self.settings.max_retries, exc, backoff,
                )
                time.sleep(backoff)

        raise ProviderError(
            f"YouTube API request to {path} failed after "
            f"{self.settings.max_retries} attempts: {last_error}"
        ) from last_error

    # -- channel ------------------------------------------------------------

    def resolve_channel(self, identifier: str) -> Channel:
        kind, value = parse_channel_identifier(identifier)

        params: Dict[str, Any] = {
            "part": "snippet,statistics,contentDetails",
        }
        if kind == "id":
            params["id"] = value
        elif kind == "username":
            params["forUsername"] = value
        else:
            params["forHandle"] = value

        payload = self._get("channels", params)
        items = payload.get("items") or []

        # forHandle is comparatively new; fall back to search for old vanity names.
        if not items and kind == "handle":
            resolved_id = self._search_channel_id(value)
            if resolved_id:
                payload = self._get(
                    "channels",
                    {"part": "snippet,statistics,contentDetails", "id": resolved_id},
                )
                items = payload.get("items") or []

        if not items:
            raise ProviderError(f"No channel found for '{identifier}'.")

        return self.parse_channel(items[0])

    def _search_channel_id(self, query: str) -> Optional[str]:
        payload = self._get(
            "search", {"part": "snippet", "type": "channel", "q": query, "maxResults": 1}
        )
        items = payload.get("items") or []
        if not items:
            return None
        return (items[0].get("snippet") or {}).get("channelId") or (
            items[0].get("id") or {}
        ).get("channelId")

    @staticmethod
    def parse_channel(item: Dict[str, Any]) -> Channel:
        snippet = item.get("snippet") or {}
        stats = item.get("statistics") or {}
        content = item.get("contentDetails") or {}
        thumbnails = snippet.get("thumbnails") or {}

        return Channel(
            id=item.get("id", ""),
            title=snippet.get("title", ""),
            description=snippet.get("description", ""),
            subscriber_count=int(stats.get("subscriberCount", 0) or 0),
            video_count=int(stats.get("videoCount", 0) or 0),
            view_count=int(stats.get("viewCount", 0) or 0),
            uploads_playlist_id=(
                (content.get("relatedPlaylists") or {}).get("uploads", "")
            ),
            published_at=snippet.get("publishedAt"),
            thumbnail_url=(thumbnails.get("high") or thumbnails.get("default") or {}).get(
                "url", ""
            ),
        )

    # -- videos -------------------------------------------------------------

    def fetch_videos(self, channel: Channel, limit: int) -> List[Video]:
        playlist_id = channel.uploads_playlist_id
        if not playlist_id:
            raise ProviderError(
                f"Channel '{channel.title}' exposes no uploads playlist."
            )

        video_ids = self._collect_video_ids(playlist_id, limit)
        if not video_ids:
            return []

        videos: List[Video] = []
        for start in range(0, len(video_ids), BATCH_SIZE):
            batch = video_ids[start : start + BATCH_SIZE]
            payload = self._get(
                "videos",
                {"part": "snippet,statistics,contentDetails", "id": ",".join(batch)},
            )
            videos.extend(
                self.parse_video(item) for item in (payload.get("items") or [])
            )
        return videos

    def _collect_video_ids(self, playlist_id: str, limit: int) -> List[str]:
        ids: List[str] = []
        page_token = ""

        while len(ids) < limit:
            params: Dict[str, Any] = {
                "part": "contentDetails",
                "playlistId": playlist_id,
                "maxResults": min(BATCH_SIZE, limit - len(ids)),
            }
            if page_token:
                params["pageToken"] = page_token

            payload = self._get("playlistItems", params)
            for item in payload.get("items") or []:
                video_id = (item.get("contentDetails") or {}).get("videoId")
                if video_id:
                    ids.append(video_id)

            page_token = payload.get("nextPageToken", "")
            if not page_token:
                break

        return ids[:limit]

    @staticmethod
    def parse_video(item: Dict[str, Any]) -> Video:
        snippet = item.get("snippet") or {}
        stats = item.get("statistics") or {}
        content = item.get("contentDetails") or {}
        thumbnails = snippet.get("thumbnails") or {}

        return Video(
            id=item.get("id", ""),
            title=snippet.get("title", ""),
            description=snippet.get("description", ""),
            published_at=snippet.get("publishedAt"),
            duration_seconds=parse_duration(content.get("duration", "")),
            # Statistics are absent when the uploader hides them.
            view_count=int(stats.get("viewCount", 0) or 0),
            like_count=int(stats.get("likeCount", 0) or 0),
            comment_count=int(stats.get("commentCount", 0) or 0),
            tags=snippet.get("tags") or [],
            thumbnail_url=(thumbnails.get("high") or thumbnails.get("default") or {}).get(
                "url", ""
            ),
        )

    # -- transcripts --------------------------------------------------------

    def fetch_transcript(self, video: Video) -> Transcript:
        """Fetch a transcript via ``youtube-transcript-api``.

        Missing transcripts are extremely common (disabled captions, music,
        age restriction), so failure is recorded on the transcript rather than
        raised — one silent video must not end the analysis.
        """
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            return Transcript(
                video_id=video.id,
                error=(
                    "youtube-transcript-api is not installed. "
                    "Run: pip install 'youtube-channel-analyzer[transcripts]'"
                ),
            )

        try:
            raw = YouTubeTranscriptApi.get_transcript(
                video.id, languages=self.settings.language_list
            )
        except Exception as exc:  # noqa: BLE001 - the library raises many types
            return Transcript(video_id=video.id, error=type(exc).__name__)

        return self.parse_transcript(video.id, raw)

    @staticmethod
    def parse_transcript(video_id: str, raw: List[Dict[str, Any]]) -> Transcript:
        segments = [
            TranscriptSegment(
                start=float(entry.get("start", 0.0) or 0.0),
                duration=float(entry.get("duration", 0.0) or 0.0),
                text=(entry.get("text") or "").strip(),
            )
            for entry in raw
            if (entry.get("text") or "").strip()
        ]
        return Transcript(video_id=video_id, segments=segments)


register_source("youtube", YouTubeDataSource)
