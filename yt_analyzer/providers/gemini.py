"""Gemini-backed content analyzer.

The model is given the channel's titles, tags and transcript openings, and asked
for qualitative structure only. Every number in the report comes from
``analytics.py`` instead, so the model is never in a position to get arithmetic
wrong.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from ..config import Settings
from ..models import Channel, ContentInsights, Transcript, Video
from .base import ContentAnalyzer, ProviderError, register_analyzer

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)

SYSTEM_PROMPT = """\
You are a YouTube content strategist. You read a channel's titles, tags and \
transcript openings, and identify the patterns that explain how the channel \
works.

Rules:
1. Base every observation on the supplied material. Never invent video titles, \
numbers or claims.
2. Do not restate view counts or statistics back — those are computed elsewhere.
3. Hook patterns must be the actual rhetorical openings used, described \
concretely (e.g. "opens with a numbered promise").
4. Recommended topics must be adjacent to what the channel already does, not \
generic advice.
5. Be specific and blunt. Vague praise is useless.
"""


class _InsightsSchema(BaseModel):
    """Permissive mirror of :class:`ContentInsights` for untrusted output."""

    themes: List[str] = Field(default_factory=list)
    hook_patterns: List[str] = Field(default_factory=list)
    content_format: str = Field(default="")
    target_audience: str = Field(default="")
    strengths: List[str] = Field(default_factory=list)
    opportunities: List[str] = Field(default_factory=list)
    recommended_topics: List[str] = Field(default_factory=list)


def loads_lenient(raw: str) -> Dict[str, Any]:
    """Parse JSON from a response that may carry fences or surrounding prose."""
    if not raw or not raw.strip():
        raise ProviderError("Model returned an empty response.")

    cleaned = _FENCE_RE.sub("", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Model returned malformed JSON: {exc}") from exc

    raise ProviderError("Model response contained no JSON object.")


def build_prompt(
    channel: Channel,
    videos: List[Video],
    transcripts: List[Transcript],
    transcript_chars: int,
) -> str:
    """Assemble the analysis prompt from titles, tags and transcript openings."""
    lines = [
        f"CHANNEL: {channel.title}",
        f"DESCRIPTION: {channel.description[:500]}",
        "",
        "RECENT VIDEO TITLES (newest first):",
    ]
    for video in videos[:40]:
        lines.append(f"- {video.title} [{video.duration_label}]")

    tags = sorted({tag.lower() for video in videos for tag in video.tags})
    if tags:
        lines += ["", f"TAGS USED: {', '.join(tags[:50])}"]

    usable = [t for t in transcripts if t.available]
    if usable:
        lines += ["", "TRANSCRIPT OPENINGS (the hooks):"]
        for transcript in usable:
            opening = transcript.opening(45)
            if opening:
                lines.append(f"- {opening[:400]}")

        lines += ["", "TRANSCRIPT EXCERPTS:"]
        budget = transcript_chars
        for transcript in usable:
            if budget <= 0:
                break
            excerpt = transcript.text[: min(budget, 1200)]
            budget -= len(excerpt)
            lines.append(f"---\n{excerpt}")

    lines += [
        "",
        "Identify: recurring themes, hook patterns, the content format, the "
        "target audience, what the channel does well, where the gaps are, and "
        "topics worth making next.",
    ]
    return "\n".join(lines)


class GeminiAnalyzer(ContentAnalyzer):
    """Qualitative content analysis through the Gemini SDK."""

    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.api_key = settings.require(
            settings.gemini_api_key, "GEMINI_API_KEY", "gemini"
        )
        self._client = None

    def _build_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise ProviderError(
                    "google-genai is not installed. "
                    "Run: pip install 'youtube-channel-analyzer[gemini]'"
                ) from exc
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def analyze(
        self,
        channel: Channel,
        videos: List[Video],
        transcripts: List[Transcript],
    ) -> Optional[ContentInsights]:
        if not videos:
            return None

        client = self._build_client()
        prompt = build_prompt(
            channel, videos, transcripts, self.settings.transcript_chars
        )

        try:
            from google.genai import types

            response = client.models.generate_content(
                model=self.settings.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=self.settings.temperature,
                    max_output_tokens=self.settings.max_tokens,
                    response_mime_type="application/json",
                    response_schema=_InsightsSchema,
                ),
            )
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ProviderError("google-genai is not installed.") from exc
        except Exception as exc:  # noqa: BLE001 - SDK raises many error types
            raise ProviderError(f"Gemini request failed: {exc}") from exc

        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, _InsightsSchema):
            return self.to_insights(parsed)

        payload = loads_lenient(getattr(response, "text", "") or "")
        return self.validate(payload)

    @staticmethod
    def validate(payload: Dict[str, Any]) -> Optional[ContentInsights]:
        try:
            schema = _InsightsSchema.model_validate(payload)
        except ValidationError as exc:
            logger.warning("Gemini output did not match the schema: %s", exc)
            return None
        return GeminiAnalyzer.to_insights(schema)

    @staticmethod
    def to_insights(schema: _InsightsSchema) -> ContentInsights:
        def clean(values: List[str]) -> List[str]:
            return [value.strip() for value in values if value and value.strip()]

        return ContentInsights(
            themes=clean(schema.themes),
            hook_patterns=clean(schema.hook_patterns),
            content_format=schema.content_format.strip(),
            target_audience=schema.target_audience.strip(),
            strengths=clean(schema.strengths),
            opportunities=clean(schema.opportunities),
            recommended_topics=clean(schema.recommended_topics),
        )


register_analyzer("gemini", GeminiAnalyzer)
