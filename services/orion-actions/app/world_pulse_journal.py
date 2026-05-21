from __future__ import annotations

from orion.journaler import JournalTriggerV1, build_world_pulse_reflective_trigger
from orion.schemas.world_pulse import WorldPulseRunResultV1


def world_pulse_journal_skip_reason(
    result: WorldPulseRunResultV1,
    *,
    enabled: bool,
) -> str | None:
    """Return skip reason, or None when journal dispatch should proceed.

    Note: actions scheduler currently triggers world-pulse with dry_run=True; reflective
    journals require a non-dry-run run result on the bus (e.g. Hub/manual persistence mode).
    """
    if not enabled:
        return "world_pulse_journal_disabled"
    run = result.run
    if run.dry_run:
        return "world_pulse_dry_run"
    if run.status not in {"completed", "partial"}:
        return f"world_pulse_run_status_{run.status}"
    if result.digest is None:
        return "world_pulse_missing_digest"
    return None


def build_world_pulse_journal_trigger(result: WorldPulseRunResultV1) -> JournalTriggerV1:
    return build_world_pulse_reflective_trigger(result)
