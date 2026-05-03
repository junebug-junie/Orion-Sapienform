"""Grafana Tempo Explore URL builder (OTEL spec Phase 1)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse, parse_qs

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


def test_normalize_otel_trace_id_hex_and_prefix() -> None:
    from scripts.otel_trace_id import is_valid_otel_trace_id, normalize_otel_trace_id

    raw = "0x" + "a" * 32
    out = normalize_otel_trace_id(raw)
    assert out == "a" * 32
    assert is_valid_otel_trace_id(out)
    assert not is_valid_otel_trace_id("not-hex" * 4)


def test_build_grafana_tempo_trace_explore_url_shape() -> None:
    from scripts.grafana_tempo_link import (
        GRAFANA_EXPLORE_DEFAULT_ORG_ID,
        build_grafana_tempo_trace_explore_url,
    )

    tid = "a" * 32
    url = build_grafana_tempo_trace_explore_url(
        grafana_base_url="http://127.0.0.1:3001/",
        trace_id=tid,
        datasource_uid="tempo",
    )
    assert url
    assert url.startswith(f"http://127.0.0.1:3001/explore?orgId={GRAFANA_EXPLORE_DEFAULT_ORG_ID}&left=")
    q = parse_qs(urlparse(url).query)
    left = json.loads(unquote(q["left"][0]))
    assert left["datasource"] == "tempo"
    assert left["queries"][0]["query"] == tid
    assert left["queries"][0]["queryType"] == "traceId"
    assert left["range"]["from"] == "now-15m"
    assert left["range"]["to"] == "now"
    assert build_grafana_tempo_trace_explore_url(
        grafana_base_url="",
        trace_id=tid,
    ) is None


def test_build_grafana_tempo_trace_explore_url_org_override() -> None:
    from scripts.grafana_tempo_link import build_grafana_tempo_trace_explore_url

    tid = "b" * 32
    url = build_grafana_tempo_trace_explore_url(
        grafana_base_url="http://example.test/",
        trace_id=tid,
        datasource_uid="tempo",
        grafana_org_id=42,
    )
    assert url and "orgId=42&" in url
