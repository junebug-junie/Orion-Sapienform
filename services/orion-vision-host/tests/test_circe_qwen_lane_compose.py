"""Regression guard for the circe Qwen2-VL lane's compose file.

Plain-text assertions, no Docker/YAML-parser dependency needed -- this
exists to catch a future edit accidentally pointing this lane back at the
shared channel (the exact race-condition class of incident PR #1859/#1860
fixed) or silently reverting the model back to BLIP.
"""
from __future__ import annotations

from pathlib import Path

COMPOSE_PATH = (
    Path(__file__).resolve().parent.parent / "docker-compose.circe-qwen.yml"
)


def test_circe_qwen_lane_uses_isolated_channel_not_shared():
    text = COMPOSE_PATH.read_text()
    assert "orion:exec:request:VisionHostService:circe-vl" in text, (
        "this lane must publish/subscribe on its OWN isolated channel -- "
        "PR #1859/#1860 is the live incident that made the shared bare "
        "channel a single-consumer contract"
    )
    # The shared bare channel name is a strict prefix of the isolated one
    # above, so assert its own CHANNEL_VISIONHOST_INTAKE line is exactly the
    # isolated value, not the bare shared name.
    intake_lines = [
        line for line in text.splitlines() if "CHANNEL_VISIONHOST_INTAKE=" in line
    ]
    assert len(intake_lines) == 1
    assert intake_lines[0].strip().endswith(
        "CHANNEL_VISIONHOST_INTAKE=orion:exec:request:VisionHostService:circe-vl"
    )


def test_circe_qwen_lane_runs_qwen_not_blip():
    text = COMPOSE_PATH.read_text()
    assert "VISION_VLM_MODEL_ID=Qwen/Qwen2-VL-2B-Instruct" in text, (
        "the entire point of this lane is a real VLM -- if this drifts back "
        "to a BLIP-family model_id, athena's shared instance already does "
        "that and this second GPU-resident container becomes pointless"
    )


def test_circe_qwen_lane_disables_artifact_broadcast():
    """No isolated PUB channel exists for this lane on purpose -- registering
    one with zero real consumers is exactly the orphan class
    scripts/check_metric_lineage.py's ratchet gate (CLAUDE.md section 0A)
    catches. The caller (orion-thought) gets its full result via the RPC
    reply; this must stay disabled outright, not silently re-enabled onto
    the shared orion:vision:artifacts channel (which would pollute
    orion-vision-window's real-camera projections -- see the compose
    file's own comment)."""
    text = COMPOSE_PATH.read_text()
    assert "VISION_ARTIFACT_BROADCAST_ENABLED=false" in text
    assert "CHANNEL_VISIONHOST_PUB=" not in text


def test_circe_qwen_lane_has_no_camera_frame_dependencies():
    """circe shares no filesystem with athena (see README/.env_example) --
    this lane must never enable a profile or path that assumes local frame
    access. Exact-line match (not a bare substring, review finding) -- a
    substring check still passes if a future edit appends another profile
    (e.g. "vlm_caption,retina_detect_open_vocab"), silently reintroducing a
    frame-dependent profile on a host with nothing to read a frame from."""
    text = COMPOSE_PATH.read_text()
    profile_lines = [
        line for line in text.splitlines() if "VISION_ENABLED_PROFILES=" in line
    ]
    assert len(profile_lines) == 1
    assert profile_lines[0].strip().endswith("VISION_ENABLED_PROFILES=vlm_caption")
    assert "/mnt/telemetry" not in text
