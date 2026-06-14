"""Runtime snapshot for substrate grammar reduction observe mode."""

from __future__ import annotations

from typing import Any

from app.cursor_gaps import has_cold_start_tail_seed, has_recent_tail_seed, tail_seed_snapshot
from app.cursor_reset import cursor_reset_snapshot, last_reset_skipped_history
from app.settings import get_settings
from app.store import BiometricsSubstrateStore, GRAMMAR_CURSOR_REGISTRY


def build_substrate_grammar_truth(store: BiometricsSubstrateStore) -> dict[str, Any]:
    settings = get_settings()
    degraded_reasons: list[str] = []

    if not settings.orion_bus_enabled:
        degraded_reasons.append("orion_bus_disabled")
    if has_recent_tail_seed():
        degraded_reasons.append("recent_cursor_tail_seed")
    if has_cold_start_tail_seed():
        degraded_reasons.append("cold_start_tail_seed_occurred")
    if last_reset_skipped_history():
        degraded_reasons.append("operator_cursor_reset_skipped_history")

    cursors = store.cursor_positions()
    lag_by_reducer: dict[str, float | None] = {}
    max_lag_sec = settings.substrate_cursor_lag_resync_hours * 3600
    for cursor in cursors:
        lag = cursor.get("lag_sec")
        lag_by_reducer[cursor["cursor_name"]] = lag
        if lag is not None and lag > max_lag_sec:
            degraded_reasons.append(f"cursor_lag:{cursor['cursor_name']}")

    tail = tail_seed_snapshot()
    reset_snap = cursor_reset_snapshot()
    last_gap = tail.get("latest")

    return {
        "ok": not degraded_reasons,
        "degraded": bool(degraded_reasons),
        "degraded_reasons": degraded_reasons,
        "known_risks": {
            "cold_start_tail_seed_skips_history": (
                "Cold-start tail-seed can skip unreduced history; loud in logs and recoverable "
                "via POST /grammar/cursor/reset?mode=earliest with operator token."
            ),
            "accepted_pressure_not_canonical_ingress": (
                "orion:grammar:accepted-pressure is reducer output, not canonical grammar ingress."
            ),
        },
        "enabled_reducers": {
            "biometrics_node": settings.enable_biometrics_node_reducer,
            "biometrics_pressure_organ": settings.enable_biometrics_pressure_organ,
            "node_pressure": settings.enable_node_pressure_reducer,
            "execution_trajectory": settings.enable_execution_trajectory_reducer,
            "transport_bus": settings.enable_transport_bus_reducer,
        },
        "grammar_poll_interval_sec": settings.grammar_poll_interval_sec,
        "cursor_settings": {
            "substrate_cursor_tail_seed_on_lag": settings.substrate_cursor_tail_seed_on_lag,
            "substrate_cursor_lag_resync_hours": settings.substrate_cursor_lag_resync_hours,
            "known_cursors": sorted(GRAMMAR_CURSOR_REGISTRY.keys()),
            "cursor_reset_auth_configured": bool(
                str(settings.substrate_cursor_reset_operator_token or "").strip()
            ),
        },
        "cursor_positions": cursors,
        "cursor_lag_by_reducer": lag_by_reducer,
        "last_data_gap": last_gap,
        "tail_seed": tail,
        "operator_cursor_reset": reset_snap,
        "cold_start_tail_seed_occurred": has_cold_start_tail_seed(),
        "lag_tail_seed_enabled": settings.substrate_cursor_tail_seed_on_lag,
        "accepted_pressure_output_channel": settings.accepted_pressure_grammar_channel,
        "canonical_grammar_input_channel": settings.grammar_event_channel,
        "effective_flags": {
            "orion_bus_enabled": settings.orion_bus_enabled,
            "publish_accepted_pressure_grammar": settings.publish_accepted_pressure_grammar,
            "accepted_pressure_grammar_channel": settings.accepted_pressure_grammar_channel,
            "grammar_event_channel": settings.grammar_event_channel,
        },
        "cursor_recovery": {
            "endpoint": "POST /grammar/cursor/reset",
            "auth_header": "X-Orion-Operator-Token",
            "internal_only": True,
            "modes": ["earliest", "tail", "timestamp"],
            "example_earliest": (
                "curl -X POST -H 'X-Orion-Operator-Token: $SUBSTRATE_CURSOR_RESET_OPERATOR_TOKEN' "
                "'http://127.0.0.1:8115/grammar/cursor/reset?"
                "cursor_name=biometrics_grammar_consumer&mode=earliest'"
            ),
        },
    }
