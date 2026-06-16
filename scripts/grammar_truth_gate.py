"""Validate /grammar/truth payloads for the production smoke gate."""

from __future__ import annotations

from typing import Any


SQL_WRITER_REQUIRED = {
    "ok",
    "degraded",
    "degraded_reasons",
    "grammar_channel_enabled",
    "subscribed_channels",
    "grammar_worker_count",
    "grammar_queue",
    "grammar_fallbacks",
    "latest_by_source_service",
    "grammar_index",
    "grammar_retention",
}

SUBSTRATE_REQUIRED = {
    "ok",
    "degraded",
    "degraded_reasons",
    "enabled_reducers",
    "grammar_poll_interval_sec",
    "cursor_settings",
    "cursor_positions",
    "cursor_lag_by_reducer",
    "stream_lag_by_reducer",
    "pending_backlog_by_reducer",
    "reducer_health_by_name",
    "quarantine_by_reducer",
    "unacknowledged_quarantine_count_by_reducer",
    "tail_seed",
    "operator_cursor_reset",
    "accepted_pressure_output_channel",
    "canonical_grammar_input_channel",
}


def _missing_fields(payload: dict[str, Any], required: set[str]) -> list[str]:
    return sorted(k for k in required if k not in payload)


def validate_truth_payload(name: str, payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return [f"{name}: payload is not a JSON object"]

    required = SQL_WRITER_REQUIRED if name == "sql-writer" else SUBSTRATE_REQUIRED
    missing = _missing_fields(payload, required)
    if missing:
        errors.append(f"{name}: missing fields: {', '.join(missing)}")

    if "degraded" not in payload:
        return errors
    if not isinstance(payload.get("degraded_reasons"), list):
        errors.append(f"{name}: degraded_reasons must be a list")

    if payload.get("degraded") is True and not payload.get("degraded_reasons"):
        errors.append(f"{name}: degraded=true but degraded_reasons empty")

    return errors


def _classify_degraded_reason(reason: str) -> str:
    if reason.startswith("cursor_lag:"):
        return "stale_reducer_cursor"
    if reason.startswith("reducer_heartbeat_stale:"):
        return "missing_reducer_heartbeat"
    if reason.startswith("reducer_blocked:"):
        return "reducer_blocked_on_event"
    if reason.startswith("reducer_cursor_commit_failing:"):
        return "cursor_commit_failure"
    if reason.startswith("reducer_stream_lag:"):
        return "redis_stream_lag"
    if reason.startswith("reducer_quarantine_present:"):
        return "reducer_quarantine"
    if "tail_seed" in reason:
        return "stale_producer_cursor"
    return "other"


def format_degraded_reason_groups(reasons: list[str]) -> str:
    if not reasons:
        return "none"
    groups: dict[str, list[str]] = {}
    for reason in reasons:
        bucket = _classify_degraded_reason(reason)
        groups.setdefault(bucket, []).append(reason)
    parts = []
    for bucket in sorted(groups):
        parts.append(f"{bucket}=[{', '.join(groups[bucket])}]")
    return "; ".join(parts)


def format_reducer_health_summary(substrate: dict[str, Any] | None) -> str:
    if not substrate:
        return "substrate reducer health: (none)"
    health = substrate.get("reducer_health_by_name") or {}
    backlog = substrate.get("pending_backlog_by_reducer") or {}
    if not health:
        return "substrate reducer health: (empty)"
    lines = ["=== reducer health ==="]
    for name in sorted(health):
        row = health[name]
        lines.append(
            f"  {name}: class={row.get('classification')} "
            f"backlog={backlog.get(row.get('cursor_name'))} "
            f"stream_lag_sec={row.get('stream_lag_sec')} "
            f"heartbeat={row.get('last_tick_at')} "
            f"blocked={row.get('blocked_event_id')} "
            f"unack_quarantine={row.get('unacknowledged_quarantine_count', 0)}"
        )
    return "\n".join(lines)


def format_mode_summary(sql_writer: dict[str, Any] | None, substrate: dict[str, Any] | None) -> str:
    lines: list[str] = ["=== effective grammar mode ==="]
    if sql_writer:
        lines.append(
            f"sql-writer: grammar_enabled={sql_writer.get('grammar_channel_enabled')} "
            f"workers={sql_writer.get('grammar_worker_count')} "
            f"subscribed={sql_writer.get('subscribed_channels')}"
        )
        retention = sql_writer.get("grammar_retention") or {}
        lines.append(
            f"sql-writer retention: enabled={retention.get('enabled')} "
            f"remaining_debt={retention.get('remaining_debt')} "
            f"failure={retention.get('failure_reason')}"
        )
        latest = sql_writer.get("latest_by_source_service") or []
        if latest:
            brief = ", ".join(
                f"{row.get('source_service')}@{row.get('latest_created_at')}" for row in latest[:5]
            )
            lines.append(f"sql-writer latest_by_source: {brief}")
        else:
            lines.append("sql-writer latest_by_source: (none)")
    if substrate:
        lines.append(
            f"substrate: poll_interval={substrate.get('grammar_poll_interval_sec')} "
            f"reducers={substrate.get('enabled_reducers')}"
        )
        lines.append(
            f"substrate cursors: wall_lag={substrate.get('cursor_lag_by_reducer')} "
            f"stream_lag={substrate.get('stream_lag_by_reducer')} "
            f"backlog={substrate.get('pending_backlog_by_reducer')} "
            f"tail_seed_count={(substrate.get('tail_seed') or {}).get('count')}"
        )
        if substrate.get("degraded_reasons"):
            lines.append(
                "substrate degraded groups: "
                f"{format_degraded_reason_groups(substrate.get('degraded_reasons') or [])}"
            )
        lines.append(format_reducer_health_summary(substrate))
        lines.append(
            f"substrate accepted_pressure_channel={substrate.get('accepted_pressure_output_channel')}"
        )
    return "\n".join(lines)
