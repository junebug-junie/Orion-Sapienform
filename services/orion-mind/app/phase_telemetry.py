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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def phase_telemetry_machine_keys(records: list[MindPhaseTelemetry]) -> dict[str, Any]:
    if not records:
        return {}
    return {
        "mind.phase_telemetry": [r.to_dict() for r in records],
        "mind.phase_routes": {r.phase_name: r.route for r in records if r.route},
    }
