"""Gate: spark-introspector compose mounts telemetry for Plan 2 corpus + encoder weights."""
from __future__ import annotations

from pathlib import Path

COMPOSE = Path(__file__).resolve().parents[1] / "docker-compose.yml"


def test_compose_mounts_telemetry_root_for_plan2() -> None:
    text = COMPOSE.read_text(encoding="utf-8")
    assert "${TELEMETRY_ROOT:-/mnt/telemetry}:/mnt/telemetry" in text


def test_compose_defaults_inner_features_to_seed_v2() -> None:
    text = COMPOSE.read_text(encoding="utf-8")
    assert "INNER_FEATURES_VERSION: ${INNER_FEATURES_VERSION:-seed-v2}" in text
    assert "INNER_FEATURES_CORPUS_PATH: ${INNER_FEATURES_CORPUS_PATH:-/mnt/telemetry/phi/corpus/inner_state.jsonl}" in text
    assert "ORION_PHI_ENCODER_ENABLED: ${ORION_PHI_ENCODER_ENABLED:-false}" in text
    assert "ORION_PHI_ENCODER_WEIGHTS: ${ORION_PHI_ENCODER_WEIGHTS:-/mnt/telemetry/models/phi/encoders/active}" in text
    assert "INNER_FEATURES_SCALER_WINDOW_SEC: ${INNER_FEATURES_SCALER_WINDOW_SEC:-900}" in text
    assert "PHI_DEGENERATE_STREAK: ${PHI_DEGENERATE_STREAK:-20}" in text
    assert "SUBSTRATE_RUNTIME_URL: ${SUBSTRATE_RUNTIME_URL:-http://orion-athena-substrate-runtime:8115}" in text


def test_sync_script_includes_plan2_phi_keys() -> None:
    from scripts.sync_local_env_from_example import SYNC_EXACT, SYNC_PREFIXES

    prefixes = set(SYNC_PREFIXES)
    assert "INNER_FEATURES_" in prefixes
    assert "ORION_PHI_ENCODER_" in prefixes
    assert "PHI_DEGENERATE_" in prefixes
    assert "SUBSTRATE_READ_" in prefixes
    assert "EXEC_TRAJECTORY_" in prefixes
    exact = set(SYNC_EXACT)
    assert "SUBSTRATE_RUNTIME_URL" in exact
    assert "CHANNEL_INNER_FEATURES" in exact
    assert "CHANNEL_PHI_REWARD" in exact
