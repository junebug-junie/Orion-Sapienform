"""HTTP tests for OTEL / Grafana observability routes (Phase 1)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs, unquote, urlparse

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)


def test_api_observability_grafana_tempo_trace_success() -> None:
    import scripts.main as hub_main

    tid = "a" * 32
    fake = SimpleNamespace(
        HUB_OTEL_GRAFANA_BASE_URL="http://127.0.0.1:3001",
        HUB_OTEL_GRAFANA_DATASOURCE_UID="tempo",
        HUB_OTEL_GRAFANA_ORG_ID=7,
    )
    with patch("scripts.api_routes.settings", fake):
        with TestClient(hub_main.app) as client:
            r = client.get(f"/api/observability/grafana-tempo-trace/{tid}")
    assert r.status_code == 200
    data = r.json()
    assert data["trace_id"] == tid
    url = data["grafana_explore_trace_url"]
    q = parse_qs(urlparse(url).query)
    assert q["orgId"] == ["7"]
    left = json.loads(unquote(q["left"][0]))
    assert left["queries"][0]["query"] == tid


def test_api_observability_grafana_tempo_trace_503_when_unconfigured() -> None:
    import scripts.main as hub_main

    tid = "a" * 32
    fake = SimpleNamespace(
        HUB_OTEL_GRAFANA_BASE_URL="",
        HUB_OTEL_GRAFANA_DATASOURCE_UID="tempo",
        HUB_OTEL_GRAFANA_ORG_ID=1,
    )
    with patch("scripts.api_routes.settings", fake):
        with TestClient(hub_main.app) as client:
            r = client.get(f"/api/observability/grafana-tempo-trace/{tid}")
    assert r.status_code == 503
    assert r.json().get("detail") == "grafana_base_url_not_configured"


def test_api_observability_grafana_tempo_trace_400_invalid_id() -> None:
    import scripts.main as hub_main

    fake = SimpleNamespace(
        HUB_OTEL_GRAFANA_BASE_URL="http://127.0.0.1:3001",
        HUB_OTEL_GRAFANA_DATASOURCE_UID="tempo",
        HUB_OTEL_GRAFANA_ORG_ID=1,
    )
    with patch("scripts.api_routes.settings", fake):
        with TestClient(hub_main.app) as client:
            r = client.get("/api/observability/grafana-tempo-trace/not-a-32-char-hex-trace-id")
    assert r.status_code == 400


class _FakeSignalsCache:
    enabled = True
    trace_enabled = True

    async def get_trace(self, trace_id: str) -> object:
        raise AssertionError("get_trace should not run when trace_id is invalid")


def test_api_signals_trace_invalid_id_returns_400() -> None:
    import scripts.main as hub_main

    with patch.object(hub_main, "signals_inspect_cache", _FakeSignalsCache()):
        with TestClient(hub_main.app) as client:
            r = client.get("/api/signals/trace/not-a-32-char-hex-trace-id")
    assert r.status_code == 400
    assert r.json().get("detail") == "invalid_otel_trace_id"
