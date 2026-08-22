"""Covers POST /api/vision/affect-capture (services/orion-hub/scripts/
api_routes.py) -- the route the Vision panel's "Affect check" button calls
(templates/index.html / static/js/app.js). Mirrors the direct-call testing
convention already used by test_autonomy_goal_archive_api.py: call the
handler function directly and mock requests.post, rather than standing up
the full app via TestClient.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from fastapi import HTTPException

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from scripts import api_routes  # noqa: E402


def test_returns_503_when_base_url_not_configured():
    with patch.object(api_routes.settings, "JUNIPER_AFFECTIVE_STATE_BASE_URL", ""):
        with pytest.raises(HTTPException) as exc_info:
            api_routes.api_vision_affect_capture()
    assert exc_info.value.status_code == 503


def test_proxies_successful_response_body():
    fake_resp = MagicMock()
    fake_resp.raise_for_status.return_value = None
    fake_resp.json.return_value = {
        "capture": {"ok": True, "video_sha256": "a" * 64},
        "result": {"ok": True, "raw_response": "sad, contemplative"},
        "event": {"ok": True},
    }
    with patch.object(
        api_routes.settings, "JUNIPER_AFFECTIVE_STATE_BASE_URL", "http://circe:32799"
    ), patch.object(api_routes.requests, "post", return_value=fake_resp) as mock_post:
        payload = api_routes.api_vision_affect_capture()

    mock_post.assert_called_once()
    called_url = mock_post.call_args.args[0]
    assert called_url == "http://circe:32799/v1/juniper/affect/capture_and_assess"
    assert payload["result"]["ok"] is True
    assert payload["result"]["raw_response"] == "sad, contemplative"


def test_proxies_internal_failure_body_without_raising():
    """The orchestrator's own endpoint replies 200 with ok=False fields
    inside the body even on internal failure (capture failed, GPU busy,
    etc.) -- this route must pass that straight through, not translate it
    into an HTTP error."""
    fake_resp = MagicMock()
    fake_resp.raise_for_status.return_value = None
    fake_resp.json.return_value = {
        "capture": {"ok": False, "error": "a capture is already in progress", "error_code": "busy"},
        "result": {"ok": False, "error": "capture failed: a capture is already in progress"},
        "event": {"ok": False},
    }
    with patch.object(
        api_routes.settings, "JUNIPER_AFFECTIVE_STATE_BASE_URL", "http://circe:32799"
    ), patch.object(api_routes.requests, "post", return_value=fake_resp):
        payload = api_routes.api_vision_affect_capture()

    assert payload["result"]["ok"] is False
    assert payload["capture"]["error_code"] == "busy"


def test_transport_failure_raises_502():
    with patch.object(
        api_routes.settings, "JUNIPER_AFFECTIVE_STATE_BASE_URL", "http://circe:32799"
    ), patch.object(
        api_routes.requests, "post", side_effect=requests.ConnectionError("refused")
    ):
        with pytest.raises(HTTPException) as exc_info:
            api_routes.api_vision_affect_capture()
    assert exc_info.value.status_code == 502
