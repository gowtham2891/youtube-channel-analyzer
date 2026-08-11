"""Tests for the credential-check module. No network is ever touched."""

from __future__ import annotations

import httpx
import pytest

from yt_analyzer.config import Settings
from yt_analyzer.health import (
    CheckResult,
    check_gemini,
    check_settings,
    check_youtube,
)


def fake_response(status: int, payload=None, text: str = "") -> httpx.Response:
    request = httpx.Request("GET", "https://example.com")
    if payload is not None:
        return httpx.Response(status, json=payload, request=request)
    return httpx.Response(status, text=text, request=request)


@pytest.fixture
def no_network(monkeypatch):
    """Fail loudly if a test would make a real request."""

    def guard(*args, **kwargs):
        raise AssertionError("unexpected network call")

    monkeypatch.setattr(httpx, "get", guard)
    monkeypatch.setattr(httpx, "post", guard)


class TestCheckResult:
    def test_str_ok(self):
        assert str(CheckResult("X", True, "fine")) == "OK X: fine"

    def test_str_failed(self):
        assert str(CheckResult("X", False, "bad")) == "FAILED X: bad"


class TestYouTubeCheck:
    def test_blank_key_is_reported_without_a_request(self, no_network):
        result = check_youtube("   ")
        assert not result.ok
        assert "No key provided" in result.message

    def test_success(self, monkeypatch):
        monkeypatch.setattr(httpx, "get", lambda *a, **k: fake_response(200, {"items": []}))
        result = check_youtube("valid-key")
        assert result.ok
        assert "works" in result.message.lower()

    def test_sends_the_key_as_a_parameter(self, monkeypatch):
        seen = {}

        def capture(url, params=None, timeout=None):
            seen.update(params or {})
            return fake_response(200, {"items": []})

        monkeypatch.setattr(httpx, "get", capture)
        check_youtube("my-secret-key")
        assert seen["key"] == "my-secret-key"
        assert seen["part"] == "id"

    def test_key_is_stripped(self, monkeypatch):
        seen = {}

        def capture(url, params=None, timeout=None):
            seen.update(params or {})
            return fake_response(200, {"items": []})

        monkeypatch.setattr(httpx, "get", capture)
        check_youtube("  padded-key\n")
        assert seen["key"] == "padded-key"

    def test_quota_exhausted_is_explained(self, monkeypatch):
        payload = {"error": {"message": "The request cannot be completed because you have exceeded your quota."}}
        monkeypatch.setattr(httpx, "get", lambda *a, **k: fake_response(403, payload))
        result = check_youtube("valid-key")
        assert not result.ok
        assert "quota is exhausted" in result.message

    def test_api_not_enabled_is_explained(self, monkeypatch):
        payload = {"error": {"message": "YouTube Data API v3 has not been used in project 123 before or it is disabled."}}
        monkeypatch.setattr(httpx, "get", lambda *a, **k: fake_response(403, payload))
        result = check_youtube("valid-key")
        assert not result.ok
        assert "not enabled" in result.message

    def test_bad_key_reports_the_api_reason(self, monkeypatch):
        payload = {"error": {"message": "API key not valid. Please pass a valid API key."}}
        monkeypatch.setattr(httpx, "get", lambda *a, **k: fake_response(400, payload))
        result = check_youtube("bad-key")
        assert not result.ok
        assert "API key not valid" in result.message

    def test_network_failure_is_reported_not_raised(self, monkeypatch):
        def boom(*args, **kwargs):
            raise httpx.ConnectError("no route to host")

        monkeypatch.setattr(httpx, "get", boom)
        result = check_youtube("valid-key")
        assert not result.ok
        assert "Could not reach" in result.message

    def test_non_json_error_body_is_survivable(self, monkeypatch):
        monkeypatch.setattr(httpx, "get", lambda *a, **k: fake_response(500, text="<html>oops</html>"))
        result = check_youtube("valid-key")
        assert not result.ok
        assert "500" in result.message


class TestGeminiCheck:
    def test_blank_key(self, no_network):
        assert not check_gemini("").ok

    def test_success_counts_models(self, monkeypatch):
        payload = {"models": [{"name": "models/gemini-2.0-flash"}, {"name": "models/gemini-1.5-pro"}]}
        monkeypatch.setattr(httpx, "get", lambda *a, **k: fake_response(200, payload))
        result = check_gemini("valid-key")
        assert result.ok
        assert "2 models" in result.message

    def test_requested_model_present(self, monkeypatch):
        payload = {"models": [{"name": "models/gemini-2.0-flash"}]}
        monkeypatch.setattr(httpx, "get", lambda *a, **k: fake_response(200, payload))
        result = check_gemini("valid-key", model="gemini-2.0-flash")
        assert result.ok
        assert "not in" not in result.message

    def test_requested_model_missing_still_passes_with_a_warning(self, monkeypatch):
        payload = {"models": [{"name": "models/gemini-1.5-pro"}]}
        monkeypatch.setattr(httpx, "get", lambda *a, **k: fake_response(200, payload))
        result = check_gemini("valid-key", model="gemini-9.9-ultra")
        assert result.ok  # the key is fine; only the model name is suspect
        assert "gemini-9.9-ultra" in result.message

    def test_rejected_key(self, monkeypatch):
        payload = {"error": {"message": "API key not valid"}}
        monkeypatch.setattr(httpx, "get", lambda *a, **k: fake_response(400, payload))
        result = check_gemini("bad-key")
        assert not result.ok
        assert "rejected" in result.message.lower()

    def test_rate_limited(self, monkeypatch):
        monkeypatch.setattr(httpx, "get", lambda *a, **k: fake_response(429, {}))
        result = check_gemini("valid-key")
        assert not result.ok
        assert "throttled" in result.message

    def test_network_failure(self, monkeypatch):
        def boom(*args, **kwargs):
            raise httpx.ReadTimeout("timed out")

        monkeypatch.setattr(httpx, "get", boom)
        assert not check_gemini("valid-key").ok


class TestCheckSettings:
    def test_mock_mode_needs_no_network(self, no_network):
        results = check_settings(Settings(source="mock", analyzer="mock"))
        assert len(results) == 2
        assert all(result.ok for result in results)
        assert all("No credentials required" in r.message for r in results)

    def test_live_source_is_checked(self, monkeypatch):
        monkeypatch.setattr(httpx, "get", lambda *a, **k: fake_response(200, {"items": []}))
        results = check_settings(
            Settings(source="youtube", analyzer="mock", youtube_api_key="k")
        )
        providers = [result.provider for result in results]
        assert "YouTube Data API" in providers

    def test_live_analyzer_is_checked(self, monkeypatch):
        monkeypatch.setattr(
            httpx, "get", lambda *a, **k: fake_response(200, {"models": []})
        )
        results = check_settings(
            Settings(source="mock", analyzer="gemini", gemini_api_key="k")
        )
        assert any(result.provider == "Gemini" for result in results)

    def test_only_filter_limits_the_checks(self, no_network):
        results = check_settings(Settings(source="mock", analyzer="mock"), only="source")
        assert len(results) == 1

    def test_missing_key_on_a_live_provider_fails(self, no_network):
        results = check_settings(
            Settings(source="youtube", analyzer="mock", youtube_api_key="")
        )
        assert not results[0].ok

    def test_check_timeout_is_bounded_by_request_timeout(self, monkeypatch):
        seen = {}

        def capture(url, params=None, timeout=None):
            seen["timeout"] = timeout
            return fake_response(200, {"items": []})

        monkeypatch.setattr(httpx, "get", capture)
        check_settings(
            Settings(source="youtube", analyzer="mock", youtube_api_key="k",
                     request_timeout=3.0)
        )
        assert seen["timeout"] == 3.0
