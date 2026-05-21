from __future__ import annotations

from collections.abc import Awaitable, Callable

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.journaler import JournalTriggerV1, build_world_pulse_reflective_trigger, cooldown_key_for_trigger
from orion.schemas.world_pulse import WorldPulseRunResultV1

from .settings import Settings

DispatchJournalFn = Callable[..., Awaitable[bool]]
AuditFn = Callable[..., Awaitable[None]]


def world_pulse_journal_skip_reason(
    result: WorldPulseRunResultV1,
    *,
    enabled: bool,
    allow_dry_run: bool = False,
) -> str | None:
    """Return skip reason, or None when journal dispatch should proceed."""
    if not enabled:
        return "world_pulse_journal_disabled"
    run = result.run
    if run.dry_run and not allow_dry_run:
        return "world_pulse_dry_run"
    if run.status not in {"completed", "partial"}:
        return f"world_pulse_run_status_{run.status}"
    if result.digest is None:
        return "world_pulse_missing_digest"
    return None


def build_world_pulse_journal_trigger(result: WorldPulseRunResultV1) -> JournalTriggerV1:
    return build_world_pulse_reflective_trigger(result)


async def handle_world_pulse_run_result_journal(
    env: BaseEnvelope,
    *,
    settings: Settings,
    dispatch_journal: DispatchJournalFn,
    audit: AuditFn,
) -> bool:
    try:
        result = WorldPulseRunResultV1.model_validate(env.payload)
    except Exception:
        await audit(
            env,
            status="skipped",
            event_id=str(env.correlation_id),
            action_name="journal.world_pulse_digest",
            reason="invalid_run_result_payload",
        )
        return True

    skip = world_pulse_journal_skip_reason(
        result,
        enabled=settings.actions_world_pulse_journal_enabled,
        allow_dry_run=settings.actions_world_pulse_journal_allow_dry_run,
    )
    if skip == "world_pulse_journal_disabled":
        return True
    if skip is not None:
        await audit(
            env,
            status="skipped",
            event_id=result.run.run_id,
            action_name="journal.world_pulse_digest",
            reason=skip,
        )
        return True

    trigger = build_world_pulse_journal_trigger(result)
    await dispatch_journal(
        env,
        trigger=trigger,
        audit_action="journal.world_pulse_digest",
        dedupe_key=cooldown_key_for_trigger(trigger),
    )
    return True
