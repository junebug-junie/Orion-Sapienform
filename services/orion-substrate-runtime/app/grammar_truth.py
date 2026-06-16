"""Runtime snapshot for substrate grammar reduction observe mode."""

from __future__ import annotations

from typing import Any

from app.cursor_gaps import has_cold_start_tail_seed, has_recent_tail_seed, tail_seed_snapshot
from app.cursor_reset import cursor_reset_snapshot, last_reset_skipped_history
from app.reducer_health import health_snapshots, update_backlog_metrics
from app.settings import get_settings
from app.store import BiometricsSubstrateStore, GRAMMAR_CURSOR_REGISTRY

REDUCER_KEY_BY_CURSOR: dict[str, str] = {
    "biometrics_grammar_consumer": "biometrics",
    "execution_grammar_reducer": "execution_trajectory",
    "transport_grammar_reducer": "transport_bus",
}

ENABLED_BY_REDUCER_KEY: dict[str, Any] = {
    "biometrics": lambda s: True,
    "execution_trajectory": lambda s: s.enable_execution_trajectory_reducer,
    "transport_bus": lambda s: s.enable_transport_bus_reducer,
}


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
    cursor_by_name = {row["cursor_name"]: row for row in cursors}
    lag_by_reducer: dict[str, float | None] = {}
    stream_lag_by_reducer: dict[str, float | None] = {}
    backlog_by_reducer: dict[str, int] = {}
    max_lag_sec = settings.substrate_cursor_lag_resync_hours * 3600
    heartbeat_stale_sec = settings.reducer_heartbeat_stale_sec
    reducer_health: dict[str, dict[str, Any]] = {}

    for cursor_name in sorted(GRAMMAR_CURSOR_REGISTRY):
        cursor = cursor_by_name.get(cursor_name, {})
        lag = cursor.get("lag_sec")
        lag_by_reducer[cursor_name] = lag
        if lag is not None and lag > max_lag_sec:
            degraded_reasons.append(f"cursor_lag:{cursor_name}")

        metrics = store.grammar_cursor_metrics(cursor_name)
        stream_lag = metrics.get("stream_lag_sec")
        pending = int(metrics.get("pending_backlog") or 0)
        stream_lag_by_reducer[cursor_name] = stream_lag
        backlog_by_reducer[cursor_name] = pending

        reducer_key = REDUCER_KEY_BY_CURSOR.get(cursor_name, cursor_name)
        enabled_fn = ENABLED_BY_REDUCER_KEY.get(reducer_key, lambda _s: True)
        enabled = bool(enabled_fn(settings))
        update_backlog_metrics(
            reducer_key,
            cursor_name=cursor_name,
            enabled=enabled,
            pending_backlog=pending,
            stream_lag_sec=stream_lag,
            cursor_wall_lag_sec=metrics.get("cursor_wall_lag_sec"),
        )

    snapshots = health_snapshots()
    for cursor_name in sorted(GRAMMAR_CURSOR_REGISTRY):
        reducer_key = REDUCER_KEY_BY_CURSOR.get(cursor_name, cursor_name)
        enabled_fn = ENABLED_BY_REDUCER_KEY.get(reducer_key, lambda _s: True)
        enabled = bool(enabled_fn(settings))
        snap = snapshots.get(reducer_key)
        if snap is None:
            from app.reducer_health import ReducerHealthSnapshot

            snap = ReducerHealthSnapshot(
                reducer_key=reducer_key,
                cursor_name=cursor_name,
                enabled=enabled,
                pending_backlog=backlog_by_reducer.get(cursor_name),
                stream_lag_sec=stream_lag_by_reducer.get(cursor_name),
                cursor_wall_lag_sec=lag_by_reducer.get(cursor_name),
            )
        health_dict = snap.to_dict(
            heartbeat_stale_sec=heartbeat_stale_sec,
            stream_lag_degraded_sec=max_lag_sec,
        )
        reducer_health[reducer_key] = health_dict

        if not enabled:
            continue

        classification = health_dict["classification"]
        if classification == "dead_no_heartbeat":
            degraded_reasons.append(f"reducer_heartbeat_stale:{cursor_name}")
        elif classification == "blocked_on_event":
            degraded_reasons.append(f"reducer_blocked:{cursor_name}")
        elif classification == "cursor_commit_failing":
            degraded_reasons.append(f"reducer_cursor_commit_failing:{cursor_name}")
        elif pending > 0 and stream_lag is not None and stream_lag > max_lag_sec:
            degraded_reasons.append(f"reducer_stream_lag:{cursor_name}")

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
            "reducer_heartbeat_stale_sec": settings.reducer_heartbeat_stale_sec,
            "biometrics_grammar_batch_limit": settings.biometrics_grammar_batch_limit,
            "execution_grammar_batch_limit": settings.execution_grammar_batch_limit,
            "transport_grammar_batch_limit": settings.transport_grammar_batch_limit,
        },
        "cursor_positions": cursors,
        "cursor_lag_by_reducer": lag_by_reducer,
        "stream_lag_by_reducer": stream_lag_by_reducer,
        "pending_backlog_by_reducer": backlog_by_reducer,
        "reducer_health_by_name": reducer_health,
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
