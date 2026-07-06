from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from orion_aitown_mcp.client import send_input


def test_send_input_payload_shape(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("AITOWN_CONVEX_URL", "http://127.0.0.1:3210")
  monkeypatch.setenv("AITOWN_ADMIN_KEY", "admin-test")
  monkeypatch.setenv("AITOWN_WORLD_ID", "world-1")

  captured: dict = {}

  def fake_urlopen(req, timeout=15):
    captured["url"] = req.full_url
    captured["body"] = json.loads(req.data.decode("utf-8"))
    captured["auth"] = req.headers.get("Authorization")
    resp = MagicMock()
    resp.read.return_value = json.dumps({"value": {"ok": True}}).encode("utf-8")
    resp.__enter__ = lambda s: s
    resp.__exit__ = lambda *a: None
    return resp

  with patch("urllib.request.urlopen", fake_urlopen):
    result = send_input(name="moveTo", args={"playerId": "p1", "destination": {"x": 1, "y": 2}})

  assert result == {"ok": True}
  assert captured["url"] == "http://127.0.0.1:3210/api/mutation"
  assert captured["body"]["path"] == "aiTown/main:sendInput"
  assert captured["body"]["args"] == {
      "worldId": "world-1",
      "name": "moveTo",
      "args": {"playerId": "p1", "destination": {"x": 1, "y": 2}},
  }
  assert captured["auth"] == "Convex admin-test"
