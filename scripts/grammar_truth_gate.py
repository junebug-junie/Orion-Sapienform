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
            f"substrate cursors: lag={substrate.get('cursor_lag_by_reducer')} "
            f"tail_seed_count={(substrate.get('tail_seed') or {}).get('count')}"
        )
        lines.append(
            f"substrate accepted_pressure_channel={substrate.get('accepted_pressure_output_channel')}"
        )
    return "\n".join(lines)
