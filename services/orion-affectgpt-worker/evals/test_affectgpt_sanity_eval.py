"""Live eval, NOT a unit test -- requires a running orion-affectgpt-worker
with real model weights loaded (AFFECTGPT_WORKER_URL, default
http://localhost:32798). Skipped automatically if that worker is unreachable.

Reproduces the exact live check from 2026-08-22 that validated this
service's pipeline: upstream AffectGPT's own README documents that, given
the bundled demo clip + its real subtitle + the "infer emotional state"
prompt, correct output "should begin with 'In the text', otherwise your
inference code or downloaded model may contain errors." This eval is that
check, automated, against the demo clip already present inside the worker's
own container (cloned by the Dockerfile at build time -- no fixture needed
in this repo).

Run: pytest services/orion-affectgpt-worker/evals -q
"""
from __future__ import annotations

import os

import httpx
import pytest

WORKER_URL = os.environ.get("AFFECTGPT_WORKER_URL", "http://localhost:32798")
DEMO_VIDEO = "/opt/affectgpt-src/AffectGPT/demo/sample_00000000.mp4"
DEMO_AUDIO = "/opt/affectgpt-src/AffectGPT/demo/sample_00000000.wav"
# The real subtitle for this exact demo clip, per upstream's own README --
# see this service's README "Provenance" for why passing real subtitle text
# (vs. empty) materially changes output quality.
DEMO_SUBTITLE = "I don't know! I, I, I don't have experience in this area."


def _worker_reachable() -> bool:
    try:
        r = httpx.get(f"{WORKER_URL}/health", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _worker_reachable(), reason="no live orion-affectgpt-worker reachable")
def test_demo_clip_passes_upstream_sanity_check():
    resp = httpx.post(
        f"{WORKER_URL}/v1/affect/assess",
        json={
            "video_path": DEMO_VIDEO,
            "audio_path": DEMO_AUDIO,
            "subtitle": DEMO_SUBTITLE,
        },
        timeout=180.0,
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["ok"] is True, body.get("error")
    raw = body["raw_response"] or ""
    # Upstream's own documented sanity check.
    assert raw.strip().startswith("In the text"), (
        "output did not start with 'In the text' -- per upstream README this "
        f"means the inference pipeline or downloaded weights may be broken. "
        f"Got: {raw[:200]!r}"
    )

    face = body.get("face_detection") or {}
    assert face.get("frames_total", 0) > 0
    # Not asserting a specific detection_rate -- this is real hardware
    # variance, not a fixed contract. Recorded for visibility only.
    print(f"face_detection={face} timings={body.get('timings')}")


if __name__ == "__main__":
    if not _worker_reachable():
        print(f"no live worker at {WORKER_URL}, skipping")
    else:
        test_demo_clip_passes_upstream_sanity_check()
        print("OK")
