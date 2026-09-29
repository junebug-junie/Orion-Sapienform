from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from uuid import UUID

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.journaler import JournalTriggerV1, build_world_pulse_reflective_trigger, cooldown_key_for_trigger
from orion.schemas.world_pulse import WorldPulseRunResultV1

from .pending_journal_store import PendingJournalStore
from .settings import Settings

logger = logging.getLogger("orion-actions.world_pulse_journal")

DispatchJournalFn = Callable[..., Awaitable[bool]]
AuditFn = Callable[..., Awaitable[None]]
AUDIT_ACTION = "journal.world_pulse_digest"

# Substrings of a compose failure that mean "try again later", not "this can never
# work". gpu_pool_unavailable:* is the live 2026-09-25..29 failure (fast GPU lane
# congested at 06:00 local). Timeouts and empty/unparseable LLM output are also
# transient. Anything else (bad trigger, disabled journaling, schema errors) is not
# retried -- retrying a deterministic failure just burns GPU time.
_RETRYABLE_ERROR_MARKERS: tuple[str, ...] = (
    "gpu_pool_unavailable",
    "timeout",
    "timed out",
    "cortex_orch_decode_failed",
    "cortex_orch_missing_final_text",
    "journal_draft_parse_failed",
    "journal_draft_missing_required_key",
    "journal_draft_invalid_type",
)


def is_retryable_journal_error(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _RETRYABLE_ERROR_MARKERS)


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
    retry_store: PendingJournalStore | None = None,
    now_fn: Callable[[], datetime] | None = None,
) -> bool:
    try:
        result = WorldPulseRunResultV1.model_validate(env.payload)
    except Exception:
        await audit(
            env,
            status="skipped",
            event_id=str(env.correlation_id),
            action_name=AUDIT_ACTION,
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
            action_name=AUDIT_ACTION,
            reason=skip,
        )
        return True

    retry_on = retry_store is not None and settings.actions_world_pulse_journal_retry_enabled
    run_id = result.run.run_id
    if retry_on and retry_store.is_completed(run_id):
        # A retry (or an earlier delivery) already wrote this run's journal; a
        # redelivered run result must not produce a second entry.
        await audit(
            env,
            status="skipped",
            event_id=run_id,
            action_name=AUDIT_ACTION,
            reason="world_pulse_journal_already_written",
        )
        return True

    now = now_fn or (lambda: datetime.now(timezone.utc))
    failure: list[BaseException] = []

    async def _on_failure(exc: BaseException) -> None:
        failure.append(exc)

    trigger = build_world_pulse_journal_trigger(result)
    kwargs = {"on_failure": _on_failure} if retry_on else {}
    ok = await dispatch_journal(
        env,
        trigger=trigger,
        audit_action=AUDIT_ACTION,
        dedupe_key=cooldown_key_for_trigger(trigger),
        world_pulse_result=result,
        **kwargs,
    )
    if not retry_on:
        return True
    if ok:
        retry_store.mark_completed(run_id, now=now())
        return True
    if failure and is_retryable_journal_error(failure[0]):
        entry = retry_store.record_failure(
            run_id=run_id,
            payload=result.model_dump(mode="json"),
            correlation_id=str(env.correlation_id),
            error=str(failure[0]),
            now=now(),
        )
        if entry is not None:
            logger.warning(
                "world_pulse_journal_retry_enqueued run_id=%s attempts=%s next_at=%s error=%s",
                run_id,
                entry.attempts,
                entry.next_at,
                entry.last_error,
            )
            await audit(
                env,
                status="retry_scheduled",
                event_id=run_id,
                action_name=AUDIT_ACTION,
                reason=entry.last_error,
                extra={"attempts": entry.attempts, "next_at": entry.next_at},
            )
    return True


async def drain_pending_world_pulse_journals(
    *,
    store: PendingJournalStore,
    settings: Settings,
    dispatch_journal: DispatchJournalFn,
    audit: AuditFn,
    source: ServiceRef,
    now: datetime,
) -> list[tuple[str, str]]:
    """Retry due pending world_pulse_digest composes. Returns [(run_id, outcome)].

    Outcomes: completed, rescheduled, gave_up, dropped_already_written,
    dropped_not_dispatched (journaling disabled / cooldown -- nothing to retry),
    dropped_invalid_payload, gave_up_non_retryable.
    """
    outcomes: list[tuple[str, str]] = []
    if not settings.actions_world_pulse_journal_retry_enabled:
        return outcomes
    max_age = timedelta(hours=float(settings.actions_world_pulse_journal_retry_max_age_hours))
    for entry in store.due(now):
        run_id = entry.run_id
        # Reuse the original correlation_id so retries join the first attempt's trace.
        env_kwargs: dict = {}
        try:
            env_kwargs["correlation_id"] = UUID(entry.correlation_id)
        except (TypeError, ValueError):
            pass
        env = BaseEnvelope(
            kind="world.pulse.run.result.v1",
            source=source,
            payload=entry.payload,
            **env_kwargs,
        )
        if store.is_completed(run_id):
            store.remove(run_id)
            outcomes.append((run_id, "dropped_already_written"))
            continue
        age = now - entry.first_failed_at_dt
        if age > max_age:
            store.remove(run_id)
            logger.warning(
                "world_pulse_journal_gave_up run_id=%s attempts=%s age_hours=%.2f last_error=%s",
                run_id,
                entry.attempts,
                age.total_seconds() / 3600.0,
                entry.last_error,
            )
            await audit(
                env,
                status="failed",
                event_id=run_id,
                action_name="world_pulse_journal_gave_up",
                reason=entry.last_error or "max_age_exceeded",
                extra={"attempts": entry.attempts, "first_failed_at": entry.first_failed_at},
            )
            outcomes.append((run_id, "gave_up"))
            continue
        try:
            result = WorldPulseRunResultV1.model_validate(entry.payload)
        except Exception:
            store.remove(run_id)
            logger.warning("world_pulse_journal_retry_dropped_invalid_payload run_id=%s", run_id)
            outcomes.append((run_id, "dropped_invalid_payload"))
            continue

        failure: list[BaseException] = []

        async def _on_failure(exc: BaseException, _sink: list[BaseException] = failure) -> None:
            _sink.append(exc)

        trigger = build_world_pulse_journal_trigger(result)
        ok = await dispatch_journal(
            env,
            trigger=trigger,
            audit_action=AUDIT_ACTION,
            dedupe_key=cooldown_key_for_trigger(trigger),
            world_pulse_result=result,
            on_failure=_on_failure,
        )
        if ok:
            store.mark_completed(run_id, now=now)
            logger.info(
                "world_pulse_journal_retry_succeeded run_id=%s attempts=%s",
                run_id,
                entry.attempts,
            )
            outcomes.append((run_id, "completed"))
        elif not failure:
            # Not dispatched at all: journaling disabled, or the key is in cooldown /
            # in flight (another path already has it). Nothing left for us to retry.
            store.remove(run_id)
            logger.info("world_pulse_journal_retry_dropped_not_dispatched run_id=%s", run_id)
            outcomes.append((run_id, "dropped_not_dispatched"))
        elif is_retryable_journal_error(failure[0]):
            bumped = store.record_failure(
                run_id=run_id,
                payload=entry.payload,
                correlation_id=entry.correlation_id,
                error=str(failure[0]),
                now=now,
            )
            logger.warning(
                "world_pulse_journal_retry_rescheduled run_id=%s attempts=%s next_at=%s error=%s",
                run_id,
                bumped.attempts if bumped else entry.attempts,
                bumped.next_at if bumped else "",
                str(failure[0])[:200],
            )
            outcomes.append((run_id, "rescheduled"))
        else:
            store.remove(run_id)
            logger.warning(
                "world_pulse_journal_gave_up run_id=%s attempts=%s reason=non_retryable error=%s",
                run_id,
                entry.attempts + 1,
                str(failure[0])[:200],
            )
            await audit(
                env,
                status="failed",
                event_id=run_id,
                action_name="world_pulse_journal_gave_up",
                reason=str(failure[0])[:500],
                extra={"attempts": entry.attempts + 1, "non_retryable": True},
            )
            outcomes.append((run_id, "gave_up_non_retryable"))
    return outcomes
