"""Per-phase LLM telemetry for Mind run inspectability."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class MindPhaseTelemetry:
    phase_name: str
    route: str
    model: str | None = None
    provider: str | None = None
    started_at: str = ""
    elapsed_ms: float = 0.0
    ok: bool = False
    parse_ok: bool | None = None
    validation_ok: bool | None = None
    status: str | None = None
    error: str | None = None
    token_usage: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str | None = None
    raw_claim_count: int | None = None
    normalized_claim_count: int | None = None
    schema_valid_claim_count: int | None = None
    retained_claim_count: int | None = None
    filtered_claim_count: int | None = None
    filter_reasons_by_count: dict[str, int] | None = None
    sample_filter_reasons: list[dict[str, str]] | None = None
    authorization_reason: str | None = None
    authorized_for_stance_use: bool | None = None
    authorized_for_stance_skip: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        return {k: v for k, v in out.items() if v is not None}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def phase_telemetry_machine_keys(records: list[MindPhaseTelemetry]) -> dict[str, Any]:
    if not records:
        return {}
    return {
        "mind.phase_telemetry": [r.to_dict() for r in records],
        "mind.phase_routes": {r.phase_name: r.route for r in records if r.route},
    }
