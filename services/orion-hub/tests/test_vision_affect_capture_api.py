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
from scripts import vision_affect_ambient  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_ambient_state():
    """vision_affect_ambient.state is a module-level singleton -- reset it
    around every test in this file so one test's toggle doesn't leak into
    the next."""
    vision_affect_ambient.state.enabled = False
    vision_affect_ambient.state.tick_in_progress = False
    vision_affect_ambient.state.tick_count = 0
    vision_affect_ambient.state.last_attempt_at = None
    vision_affect_ambient.state.last_result_ok = None
    vision_affect_ambient.state.last_error = None
    yield
    vision_affect_ambient.state.enabled = False


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
    assert mock_post.call_args.kwargs["json"] == {"trigger": "manual"}
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


# --- Ambient (recurring) toggle -- design correction, 2026-08-22 ------------
# The route the "Affect check" button shipped as was mislabeled as fulfilling
# the toggle Juniper had actually asked for and approved before compaction
# ("a toggle that periodically grabs a clip... while on"). These cover the
# real toggle: services/orion-hub/scripts/vision_affect_ambient.py's
# module-level `state`, flipped via POST /api/vision/affect-ambient and read
# via GET /api/vision/affect-ambient/status.


def test_ambient_status_reflects_default_off_state():
    payload = api_routes.api_vision_affect_ambient_status()
    assert payload["enabled"] is False
    assert payload["tick_count"] == 0
    assert payload["last_attempt_at"] is None


def test_ambient_toggle_on_requires_base_url_and_env_enabled():
    req = api_routes.AffectAmbientToggleRequest(enabled=True)
    with patch.object(api_routes.settings, "JUNIPER_AFFECTIVE_STATE_BASE_URL", ""):
        with pytest.raises(HTTPException) as exc_info:
            api_routes.api_vision_affect_ambient_toggle(req)
    assert exc_info.value.status_code == 503
    assert vision_affect_ambient.state.enabled is False


def test_ambient_toggle_on_then_off():
    req_on = api_routes.AffectAmbientToggleRequest(enabled=True)
    with patch.object(
        api_routes.settings, "JUNIPER_AFFECTIVE_STATE_BASE_URL", "http://circe:32799"
    ), patch.object(api_routes.settings, "AFFECT_AMBIENT_ENABLED", True):
        payload = api_routes.api_vision_affect_ambient_toggle(req_on)
    assert payload["enabled"] is True
    assert vision_affect_ambient.state.enabled is True

    req_off = api_routes.AffectAmbientToggleRequest(enabled=False)
    payload = api_routes.api_vision_affect_ambient_toggle(req_off)
    assert payload["enabled"] is False
    assert vision_affect_ambient.state.enabled is False


def test_ambient_toggle_off_never_requires_base_url():
    """Turning OFF must always succeed -- an operator flipping this off
    (e.g. because JUNIPER_AFFECTIVE_STATE_BASE_URL was just cleared) must
    never be blocked by the same precondition that gates turning it on."""
    vision_affect_ambient.state.enabled = True
    req_off = api_routes.AffectAmbientToggleRequest(enabled=False)
    with patch.object(api_routes.settings, "JUNIPER_AFFECTIVE_STATE_BASE_URL", ""):
        payload = api_routes.api_vision_affect_ambient_toggle(req_off)
    assert payload["enabled"] is False
