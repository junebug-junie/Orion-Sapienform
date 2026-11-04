"""Hub's side of the curiosity replica: ask for a plan at kickoff, check it, use it or fall back.

Design: docs/superpowers/specs/2026-11-04-curiosity-replica-design.md, sections 1, 4, 5 and 7. Called from
``CuriosityInvestigationLoop._investigate`` only on a replica-arm run, right after the arm draw and the
worldview read and BEFORE the offer row is written, so the offer row records the order the run is really
shown.

1. Build the candidate list from the read Hub just made (orion/curiosity/replica/candidates.py).
2. Publish ``CuriosityOfferPlanRequestV1`` on ``orion:curiosity:replica:plan:request`` and wait at most
   ``deadline_sec`` (90 s) for ``CuriosityOfferPlanResultV1`` on ``...:plan:result:<correlation_id>``.
3. Check the plan with ``validate_plan`` (whole plan or nothing, never repaired).
4. A good plan re-selects the SAME read with the picks on top (``with_head``). Anything else -- no answer
   in time, a bus error, a malformed reply, abstain, unfunded, model failed, unparseable, a rejected plan,
   too few candidates -- keeps the uncertainty order and records why.
5. One ``curiosity_offer_plans`` row either way, and one log line:
   ``curiosity_replica_plan_used`` or ``curiosity_replica_fallback run=<id> reason=<reason>``.

The replica's reasons (``why``, ``reading``, stance reasons) are written to the plan row only. Nothing here
returns them to the caller, so they cannot reach the kickoff prompt. Stances are recorded on the row and have
no effect (answering Prime is patch 4 of the design).

Never raises into the kickoff: every failure is a fallback.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional, Sequence
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.curiosity.focus import is_prime_focus, read_pending
from orion.curiosity.replica.candidates import build_candidates, last_moves
from orion.curiosity.replica.plan import (
    REASON_MALFORMED_REPLY,
    REASON_NO_CANDIDATES,
    REASON_REQUEST_FAILED,
    REASON_TIMEOUT,
    PlanRowV1,
    invalid_status,
    validate_plan,
)
from orion.curiosity.value import PriorTestRecord, YieldModel, valid_confidence
from orion.curiosity.worldview import WorldviewSnapshot, build_prior, collapse_duplicate_priors, fresh_priors, with_head
from orion.physiology import prime_charge as pc
from orion.schemas.curiosity_replica import (
    PICKS_MIN,
    PLAN_REQUEST_CHANNEL,
    PLAN_REQUEST_KIND,
    PLAN_RESULT_CHANNEL_PREFIX,
    PLAN_RESULT_KIND,
    PLAN_STATUS_TIMEOUT,
    CuriosityOfferPlanRequestV1,
    CuriosityOfferPlanResultV1,
    CuriosityPendingFocusV1,
)

from .curiosity_offer_decisions import record_offer_plan

logger = logging.getLogger("orion-hub.curiosity_replica_plan")

# The family whose regulate signal the replica may answer (design section 9).
CURIOSITY_FAMILY = "durable.curiosity_investigate"


@dataclass(frozen=True)
class PlanOutcome:
    """What the kickoff uses: the (possibly re-ordered) view and the plan row that was recorded."""

    view: WorldviewSnapshot
    row: PlanRowV1


def _stale_ids(view: WorldviewSnapshot, *, stale_after: int) -> list[str]:
    priors = [p for p in (build_prior(r) for r in view.prior_rows) if p is not None]
    priors, _ = collapse_duplicate_priors(priors)
    return sorted(p.prior_id for p in priors if stale_after > 0 and p.times_tested >= stale_after)


async def _pending_focus(pool: Any, *, enabled: bool) -> Optional[CuriosityPendingFocusV1]:
    if not enabled or pool is None:
        return None
    try:
        row = await read_pending(pool)
    except Exception as exc:  # noqa: BLE001 -- a focus read never blocks the plan
        logger.warning("curiosity_replica_pending_focus_unread err=%s", str(exc)[:200])
        return None
    if row is None or row.origin not in ("self", "juniper") or not str(row.question or "").strip():
        return None
    return CuriosityPendingFocusV1(focus_id=str(row.focus_id), origin=row.origin, question=row.question,
                                   from_prime=row.origin == "self" and is_prime_focus(row))


async def live_signal_id(bus: Any, now: datetime) -> Optional[str]:
    """The regulate signal on curiosity in force now (either holdback arm), or None (none, or unread)."""
    redis = getattr(bus, "redis", None)
    if redis is None:
        return None
    try:
        fields = await redis.hgetall(pc.key(CURIOSITY_FAMILY)) or {}
    except Exception:  # noqa: BLE001
        return None
    raw = fields.get(pc.SIGNAL_FIELD, fields.get(pc.SIGNAL_FIELD.encode()))
    sig = pc.parse_signal(raw)
    return sig.signal_id if sig is not None and pc.active(sig, now) else None


def _fallback(run_id: str, status: str, reason: str, *, requested_at: datetime, **extra: Any) -> PlanRowV1:
    return PlanRowV1(run_id=run_id, status=status, fallback_reason=reason[:500], requested_at=requested_at, **extra)


async def plan_offer(
    *,
    bus: Any,
    pool: Any,
    run_id: str,
    view: WorldviewSnapshot,
    yield_model: YieldModel,
    history: Sequence[PriorTestRecord],
    sample: int,
    stale_after: int,
    coverage_slots: int,
    deadline_sec: float,
    source: ServiceRef,
    focus_enabled: bool,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    record: Optional[Callable[[Any, PlanRowV1], Any]] = None,
) -> PlanOutcome:
    """Never raises. ``view``: the uncertainty-order read (the fallback, and the pool the plan is checked
    against). ``record``: the plan-row writer (default ``record_offer_plan``)."""
    if record is None:
        record = record_offer_plan
    requested_at = clock()
    started = time.monotonic()
    try:
        outcome = await _plan(bus=bus, pool=pool, run_id=run_id, view=view, yield_model=yield_model, history=history,
                              sample=sample, stale_after=stale_after, coverage_slots=coverage_slots,
                              deadline_sec=deadline_sec, source=source, focus_enabled=focus_enabled, clock=clock,
                              requested_at=requested_at, started=started)
    except Exception as exc:  # noqa: BLE001 -- the kickoff never fails on the planner
        outcome = PlanOutcome(view, _fallback(run_id, PLAN_STATUS_TIMEOUT,
                                              f"{REASON_REQUEST_FAILED}: {type(exc).__name__}: {str(exc)[:200]}",
                                              requested_at=requested_at))
    row = outcome.row
    if row.used:
        logger.info("curiosity_replica_plan_used run=%s plan=%s picks=%s latency_ms=%s act=%s", run_id, row.plan_id,
                    ",".join(row.head_ids), row.latency_ms, row.act_id)
    else:
        logger.info("curiosity_replica_fallback run=%s reason=%s status=%s latency_ms=%s", run_id, row.fallback_reason,
                    row.status, row.latency_ms)
    try:
        await record(pool, row)
    except Exception as exc:  # noqa: BLE001
        logger.warning("curiosity_replica_plan_row_failed run=%s err=%s", run_id, str(exc)[:200])
    return outcome


async def _plan(*, bus, pool, run_id, view, yield_model, history, sample, stale_after, coverage_slots, deadline_sec,
                source, focus_enabled, clock, requested_at, started) -> PlanOutcome:
    fresh = [] if view.is_unavailable else fresh_priors(view, stale_after=stale_after)
    if len(fresh) < PICKS_MIN:
        detail = view.unavailable_reason or f"{len(fresh)} fresh live priors"
        return PlanOutcome(view, _fallback(run_id, invalid_status(REASON_NO_CANDIDATES), f"{REASON_NO_CANDIDATES}: {detail}",
                                           requested_at=requested_at))
    candidates = build_candidates(
        fresh, now=requested_at, rotate_seed=run_id,
        expected_nats_for=lambda p: yield_model.expected_nats(p.prior_id, valid_confidence(p.confidence)),
        moves=last_moves(history),
    )
    correlation = uuid4()
    request = CuriosityOfferPlanRequestV1(
        run_id=run_id, correlation_id=str(correlation), deadline_at=requested_at + timedelta(seconds=deadline_sec),
        candidates=candidates, stale_ids=_stale_ids(view, stale_after=stale_after),
        pending_focus=await _pending_focus(pool, enabled=focus_enabled),
    )
    reply = f"{PLAN_RESULT_CHANNEL_PREFIX}{correlation}"
    envelope = BaseEnvelope(kind=PLAN_REQUEST_KIND, correlation_id=correlation, reply_to=reply, source=source,
                            payload=request.model_dump(mode="json"))

    def ms() -> int:
        return int((time.monotonic() - started) * 1000)

    try:
        raw = await bus.rpc_request(PLAN_REQUEST_CHANNEL, envelope, reply_channel=reply, timeout_sec=float(deadline_sec))
    except (TimeoutError, OSError) as exc:
        reason = REASON_TIMEOUT if isinstance(exc, TimeoutError) else f"{REASON_REQUEST_FAILED}: {type(exc).__name__}"
        return PlanOutcome(view, _fallback(run_id, PLAN_STATUS_TIMEOUT, reason, requested_at=requested_at, latency_ms=ms()))
    answered_at = clock()
    try:
        decoded = bus.codec.decode(raw.get("data"))
        if not decoded.ok or decoded.envelope.kind != PLAN_RESULT_KIND:
            raise ValueError(f"kind {getattr(getattr(decoded, 'envelope', None), 'kind', None)!r}")
        result = CuriosityOfferPlanResultV1.model_validate(decoded.envelope.payload)
        if result.run_id != run_id:
            raise ValueError(f"answer for run {result.run_id}")
    except Exception as exc:  # noqa: BLE001
        return PlanOutcome(view, _fallback(run_id, invalid_status(REASON_MALFORMED_REPLY),
                                           f"{REASON_MALFORMED_REPLY}: {str(exc)[:200]}", requested_at=requested_at,
                                           answered_at=answered_at, latency_ms=ms()))
    common = dict(
        requested_at=requested_at, plan_id=result.plan_id, answered_at=answered_at,
        picks=[p.model_dump(mode="json") for p in result.picks],
        focus_stance=result.focus_stance.model_dump(mode="json") if result.focus_stance else None,
        signal_stance=result.signal_stance.model_dump(mode="json") if result.signal_stance else None,
        act_id=result.act_id, latency_ms=ms(), frame_tokens=result.frame_tokens,
    )
    if result.status != "ok":
        return PlanOutcome(view, PlanRowV1(run_id=run_id, status=result.status, fallback_reason=result.status, **common))
    reason = validate_plan(result.picks, candidate_ids=[c.prior_id for c in candidates],
                           fresh_ids=[p.prior_id for p in fresh], signal_stance=result.signal_stance,
                           live_signal_id=await live_signal_id(bus, answered_at),
                           max_picks=max(0, sample - coverage_slots))
    if reason is not None:
        return PlanOutcome(view, PlanRowV1(run_id=run_id, status=invalid_status(reason), fallback_reason=reason, **common))
    row = PlanRowV1(run_id=run_id, status="ok", fallback_reason=None, **common)
    return PlanOutcome(with_head(view, row.head_ids, sample=sample, stale_after=stale_after, rotate_seed=run_id,
                                 coverage_slots=coverage_slots), row)


__all__ = ["CURIOSITY_FAMILY", "PlanOutcome", "live_signal_id", "plan_offer"]
