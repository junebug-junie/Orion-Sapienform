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


def test_circe_qwen_lane_overrides_use_prefixed_keys_not_shared_names():
    """Real regression, 2026-08-26, two rounds: this file used to read
    several lane-specific tunables as bare `${VAR:-<circe-correct-default>}`
    hooks. The shared athena instance's own .env_example sets several of
    those same bare names to a DIFFERENT value for its own (correct, for
    athena) purposes. The moment services/orion-vision-host/.env exists at
    all (this compose file's own bring-up instructions require passing it
    as a --env-file), the athena-only values silently won over this lane's
    real defaults with no error. Round 1 (caught live): VISION_PERCEPT_
    STORE_URL/TOKEN/TIMEOUT_SEC, the three VISION_VRAM_* floors,
    VISION_TIMEOUT_S, ORION_BUS_ENFORCE_CATALOG -- confirmed via the reverie
    visual chain's tick log going image_not_found on the athena-only
    docker-internal percept-store hostname. Round 2 (review finding, same
    fix not yet applied to every bare hook): LOG_LEVEL, ORION_BUS_ENABLED,
    HEARTBEAT_INTERVAL_SEC, TORCH_CUDA_ALLOC_CONF -- silently harmless only
    because the values happened to coincide, same collision mechanism.
    CIRCE_QWEN_-prefixing every lane-specific override, no exceptions, is
    what actually closes the collision class. Assert directly against the
    bare names never appearing as a live override hook for every key in
    `bare_names` below, not just that the CIRCE_QWEN_ prefixed one exists
    (a future edit could add the prefixed hook back as a second, redundant
    fallback while leaving the bare one in place, keeping the bug) --
    intentionally NOT a generic "any collision" scan, since that would
    require parsing this file's YAML/shell-substitution syntax properly to
    avoid false positives on genuinely-shared keys like ORION_BUS_URL and
    PROJECT; extend this literal list instead when a new lane-specific
    tunable is added."""
    text = COMPOSE_PATH.read_text()
    bare_names = [
        "VISION_PERCEPT_STORE_URL",
        "VISION_PERCEPT_STORE_TOKEN",
        "VISION_PERCEPT_TIMEOUT_SEC",
        "VISION_VRAM_RESERVE_MB",
        "VISION_VRAM_SOFT_FLOOR_MB",
        "VISION_VRAM_HARD_FLOOR_MB",
        "VISION_TIMEOUT_S",
        "ORION_BUS_ENFORCE_CATALOG",
        "LOG_LEVEL",
        "ORION_BUS_ENABLED",
        "HEARTBEAT_INTERVAL_SEC",
        "TORCH_CUDA_ALLOC_CONF",
    ]
    for name in bare_names:
        assert f"${{{name}:" not in text, (
            f"{name} must be read via its CIRCE_QWEN_{name} override hook, "
            "not the bare shared name -- see the collision this regression "
            "guard is named for"
        )
        assert f"CIRCE_QWEN_{name}" in text


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
