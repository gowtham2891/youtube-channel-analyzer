"""Live credential checks.

Each check makes the smallest authenticated request a provider supports and
classifies the response, so someone pasting a key into the UI finds out whether
it works *before* spending a full analysis run discovering that it doesn't.

Checks are deliberately single-attempt with a short timeout: this is a
"does this key work right now" probe, not a resilient production call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

#: Short on purpose -- a credential probe should fail fast, not hang a UI.
CHECK_TIMEOUT = 15.0

GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"


@dataclass
class CheckResult:
    """The verdict on one credential."""

    provider: str
    ok: bool
    message: str

    def __str__(self) -> str:
        return f"{'OK' if self.ok else 'FAILED'} {self.provider}: {self.message}"


def _missing(provider: str) -> CheckResult:
    return CheckResult(provider, False, "No key provided.")


def _network_error(provider: str, exc: Exception) -> CheckResult:
    return CheckResult(provider, False, f"Could not reach the API: {exc}")


def _api_error_message(response: httpx.Response) -> str:
    """Pull the human-readable reason out of a Google API error body."""
    try:
        payload: Dict[str, Any] = response.json()
    except ValueError:
        return response.text[:200].strip() or f"HTTP {response.status_code}"
    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or "").strip() or f"HTTP {response.status_code}"
    if isinstance(error, str):
        return error
    return f"HTTP {response.status_code}"


def check_youtube(
    api_key: str,
    base_url: str = "https://www.googleapis.com/youtube/v3",
    timeout: float = CHECK_TIMEOUT,
) -> CheckResult:
    """Verify a YouTube Data API key with a minimal channels.list call."""
    provider = "YouTube Data API"
    if not (api_key or "").strip():
        return _missing(provider)

    try:
        response = httpx.get(
            f"{base_url.rstrip('/')}/channels",
            params={"part": "id", "forHandle": "youtube", "key": api_key.strip()},
            timeout=timeout,
        )
    except httpx.TransportError as exc:
        return _network_error(provider, exc)

    if response.status_code == 200:
        return CheckResult(provider, True, "Key works — quota available.")

    reason = _api_error_message(response)
    lowered = reason.lower()

    if response.status_code == 400:
        return CheckResult(provider, False, f"Rejected: {reason}")
    if response.status_code == 403:
        if "quota" in lowered:
            return CheckResult(
                provider, False,
                "Key is valid but the daily quota is exhausted. It resets at "
                "midnight Pacific time.",
            )
        if "disabled" in lowered or "not been used" in lowered:
            return CheckResult(
                provider, False,
                "YouTube Data API v3 is not enabled for this key's project. "
                "Enable it in the Google Cloud console.",
            )
        return CheckResult(provider, False, f"Forbidden: {reason}")
    return CheckResult(provider, False, f"HTTP {response.status_code}: {reason}")


def check_gemini(
    api_key: str,
    model: str = "",
    timeout: float = CHECK_TIMEOUT,
) -> CheckResult:
    """Verify a Gemini key by listing models — no tokens are consumed."""
    provider = "Gemini"
    if not (api_key or "").strip():
        return _missing(provider)

    try:
        response = httpx.get(
            GEMINI_MODELS_URL, params={"key": api_key.strip()}, timeout=timeout
        )
    except httpx.TransportError as exc:
        return _network_error(provider, exc)

    if response.status_code == 200:
        names = _model_names(response)
        if model and not _model_available(model, names):
            return CheckResult(
                provider, True,
                f"Key works, but '{model}' was not in the {len(names)} available "
                f"models. It may still work, or pick another in the sidebar.",
            )
        return CheckResult(provider, True, f"Key works — {len(names)} models available.")

    reason = _api_error_message(response)
    if response.status_code in (400, 401, 403):
        return CheckResult(provider, False, f"Key rejected: {reason}")
    if response.status_code == 429:
        return CheckResult(
            provider, False, "Rate limited — the key is valid but throttled."
        )
    return CheckResult(provider, False, f"HTTP {response.status_code}: {reason}")


def _model_names(response: httpx.Response) -> List[str]:
    try:
        payload = response.json()
    except ValueError:
        return []
    return [
        str(entry.get("name", "")).split("/")[-1]
        for entry in payload.get("models", [])
        if isinstance(entry, dict)
    ]


def _model_available(model: str, names: List[str]) -> bool:
    wanted = model.split("/")[-1].strip().lower()
    return any(wanted == name.lower() for name in names)


def check_settings(settings, only: Optional[str] = None) -> List[CheckResult]:
    """Check every credential the current provider selection actually needs.

    Mock providers need nothing, so they report OK without any network call --
    which is what makes the offline demo honest rather than merely quiet.
    """
    results: List[CheckResult] = []

    if only in (None, "source"):
        if settings.source == "mock":
            results.append(
                CheckResult("Data source (mock)", True, "No credentials required.")
            )
        elif settings.source == "youtube":
            results.append(
                check_youtube(
                    settings.youtube_api_key,
                    settings.youtube_base_url,
                    timeout=min(settings.request_timeout, CHECK_TIMEOUT),
                )
            )

    if only in (None, "analyzer"):
        if settings.analyzer == "mock":
            results.append(
                CheckResult("Analyzer (mock)", True, "No credentials required.")
            )
        elif settings.analyzer == "gemini":
            results.append(
                check_gemini(
                    settings.gemini_api_key,
                    settings.gemini_model,
                    timeout=min(settings.request_timeout, CHECK_TIMEOUT),
                )
            )

    return results
