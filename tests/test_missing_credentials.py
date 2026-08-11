"""Tests for the missing-credential detector. Never touches the network."""

from __future__ import annotations

import httpx
import pytest

from yt_analyzer.config import Settings
from yt_analyzer.health import (
    MissingCredential,
    credentials_ready,
    missing_credentials,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """The detector must be pure inspection: any request here is a bug."""

    def guard(*args, **kwargs):
        raise AssertionError("missing_credentials must not make network calls")

    monkeypatch.setattr(httpx, "get", guard)
    monkeypatch.setattr(httpx, "post", guard)


class TestMissingCredential:
    def test_str_includes_the_variable_name(self):
        item = MissingCredential("A Key", "A_KEY", "the thing")
        assert "A_KEY" in str(item)
        assert "the thing" in str(item)


class TestDetector:
    def test_mock_mode_needs_nothing(self):
        settings = Settings(source="mock", analyzer="mock")
        assert missing_credentials(settings) == []
        assert credentials_ready(settings)

    def test_youtube_without_a_key_is_reported(self):
        settings = Settings(source="youtube", analyzer="mock", youtube_api_key="")
        gaps = missing_credentials(settings)
        assert [item.env_var for item in gaps] == ["YOUTUBE_API_KEY"]

    def test_gemini_without_a_key_is_reported(self):
        settings = Settings(source="mock", analyzer="gemini", gemini_api_key="")
        assert [i.env_var for i in missing_credentials(settings)] == ["GEMINI_API_KEY"]

    def test_both_reported_together(self):
        settings = Settings(source="youtube", analyzer="gemini")
        assert {i.env_var for i in missing_credentials(settings)} == {
            "YOUTUBE_API_KEY", "GEMINI_API_KEY"}

    def test_supplied_keys_satisfy_the_check(self):
        settings = Settings(source="youtube", analyzer="gemini",
                            youtube_api_key="a", gemini_api_key="b")
        assert credentials_ready(settings)

    def test_whitespace_only_key_does_not_count(self):
        assert not credentials_ready(Settings(source="youtube", analyzer="mock",
                                              youtube_api_key="   "))

    def test_youtube_gap_points_at_the_console(self):
        settings = Settings(source="youtube", analyzer="mock")
        assert "console.cloud.google.com" in missing_credentials(settings)[0].get_it_at
