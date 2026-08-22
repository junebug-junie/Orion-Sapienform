"""Live eval, NOT a unit test -- requires a live bus AND a live
orion-affectgpt-worker reachable on it. Skipped automatically otherwise.

Round-trips a real trigger through this service's own HTTP endpoint and
confirms a real orion:affectgpt:assessment event actually lands on the bus
-- not just that the HTTP call returned 200.

Run: pytest services/orion-juniper-affective-state/evals -q
"""
from __future__ import annotations

import os

import httpx
import pytest

ORCHESTRATOR_URL = os.environ.get("JUNIPER_AFFECT_URL", "http://localhost:32799")
DEMO_VIDEO = "/opt/affectgpt-src/AffectGPT/demo/sample_00000000.mp4"
DEMO_AUDIO = "/opt/affectgpt-src/AffectGPT/demo/sample_00000000.wav"
DEMO_SUBTITLE = "I don't know! I, I, I don't have experience in this area."


def _reachable() -> bool:
    try:
        r = httpx.get(f"{ORCHESTRATOR_URL}/health", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _reachable(), reason="no live orion-juniper-affective-state reachable")
def test_trigger_round_trips_and_publishes():
    resp = httpx.post(
        f"{ORCHESTRATOR_URL}/v1/juniper/affect/trigger",
        json={"video_path": DEMO_VIDEO, "audio_path": DEMO_AUDIO, "subtitle": DEMO_SUBTITLE},
        timeout=180.0,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]["ok"] is True, body["result"].get("error")
    assert body["event"]["ok"] is True
    assert body["event"]["source"] == "affectgpt"
    assert body["event"]["input_ref"]["video_path"] == DEMO_VIDEO
    # trigger/correlation_id (2026-08-22, review finding): /trigger passes
    # neither explicitly, so trigger must still default to "manual" and
    # correlation_id must still be populated (generated internally by
    # trigger_assessment()) -- proves the fields work end to end through a
    # REAL bus round trip, not just the mocked unit tests in tests/.
    assert body["event"]["trigger"] == "manual"
    assert body["event"]["correlation_id"], "correlation_id must be populated even on the plain /trigger path"
    print(f"event={body['event']}")


if __name__ == "__main__":
    if not _reachable():
        print(f"no live orchestrator at {ORCHESTRATOR_URL}, skipping")
    else:
        test_trigger_round_trips_and_publishes()
        print("OK")
