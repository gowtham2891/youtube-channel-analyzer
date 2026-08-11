"""Environment-driven configuration for the channel analyzer."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

MOCK_PROVIDERS = {"mock"}


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    return raw in {"1", "true", "yes", "on"} if raw else default


class ConfigError(RuntimeError):
    """Raised when a live provider is selected without its credentials."""


@dataclass
class Settings:
    """Resolved runtime settings for one analysis run."""

    # --- provider selection -------------------------------------------------
    source: str = field(default_factory=lambda: _env("DATA_SOURCE", "mock"))
    analyzer: str = field(default_factory=lambda: _env("ANALYZER", "mock"))

    # --- YouTube Data API ---------------------------------------------------
    youtube_api_key: str = field(default_factory=lambda: _env("YOUTUBE_API_KEY"))
    youtube_base_url: str = field(
        default_factory=lambda: _env(
            "YOUTUBE_BASE_URL", "https://www.googleapis.com/youtube/v3"
        )
    )

    # --- Gemini -------------------------------------------------------------
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY"))
    gemini_model: str = field(
        default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.0-flash")
    )
    temperature: float = field(default_factory=lambda: _env_float("TEMPERATURE", 0.4))
    max_tokens: int = field(default_factory=lambda: _env_int("MAX_TOKENS", 2000))

    # --- analysis behaviour -------------------------------------------------
    max_videos: int = field(default_factory=lambda: _env_int("MAX_VIDEOS", 50))
    max_transcripts: int = field(default_factory=lambda: _env_int("MAX_TRANSCRIPTS", 10))
    transcript_chars: int = field(
        default_factory=lambda: _env_int("TRANSCRIPT_CHARS", 3000)
    )
    fetch_transcripts: bool = field(
        default_factory=lambda: _env_bool("FETCH_TRANSCRIPTS", True)
    )
    transcript_languages: str = field(
        default_factory=lambda: _env("TRANSCRIPT_LANGUAGES", "en,te,hi")
    )

    # --- networking ---------------------------------------------------------
    request_timeout: float = field(
        default_factory=lambda: _env_float("REQUEST_TIMEOUT", 30.0)
    )
    max_retries: int = field(default_factory=lambda: _env_int("MAX_RETRIES", 3))
    output_dir: str = field(default_factory=lambda: _env("OUTPUT_DIR", "output"))

    @property
    def language_list(self) -> list:
        return [code.strip() for code in self.transcript_languages.split(",") if code.strip()]

    @property
    def is_mock(self) -> bool:
        return self.source in MOCK_PROVIDERS and self.analyzer in MOCK_PROVIDERS

    def require(self, value: str, name: str, provider: str) -> str:
        if not value:
            raise ConfigError(
                f"{name} is required for the '{provider}' provider. "
                f"Set it in your .env file or switch to the 'mock' provider."
            )
        return value

    def describe(self) -> str:
        mode = "mock (no credentials needed)" if self.is_mock else "live"
        return f"mode={mode} source={self.source} analyzer={self.analyzer}"


_settings: Optional[Settings] = None


def get_settings(refresh: bool = False) -> Settings:
    global _settings
    if _settings is None or refresh:
        _settings = Settings()
    return _settings
