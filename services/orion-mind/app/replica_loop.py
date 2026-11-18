"""The curiosity replica: instance #2 of the loop library (orion/mindloop/), next to Prime in orion-mind.

Design: docs/superpowers/specs/2026-11-04-curiosity-replica-design.md (sections 1-4, 7, 8). Pure parts are in
orion/curiosity/replica/; this module does the replica's I/O.

Identity: actor and family ``curiosity.replica`` (config/physiology/families.yaml, tier 2, lane chat), Redis
key ``orion:curiosity:replica:state``, settings prefix ``CURIOSITY_REPLICA_``, logs ``curiosity_replica_*``.
It shares nothing with Prime but the process, the bus server and the Juniper-turn tracker; it reads Prime
only through Prime's published concerns view (``orion:prime:concerns``) and its regulate signal on curiosity.

What one plan looks like:

1. Hub publishes a ``CuriosityOfferPlanRequestV1`` on ``orion:curiosity:replica:plan:request`` (only on a
   replica-arm run). The listener validates it, queues it and wakes the loop (no 15 s wait).
2. The tick: if the request can still be answered in time, the loop bids ``curiosity.replica`` (tier 2,
   lane chat). Not funded, a Juniper turn, a gate refusal or a try that never reached the model: it re-bids
   next tick, until there is no longer time for one model call before the deadline; then it answers
   ``unfunded`` or ``model_failed`` (so Hub's fallback reason is the real one, not ``timeout``).
3. Funded, or a ``capacity_reserved`` deferral with a positive net (it queues at the GPU pool; the seat wait
   below still bounds it): one model call on route CURIOSITY_REPLICA_LLM_ROUTE (``prime``, the warm chat lane). The seat wait
   is cut so seat wait + generation end before the deadline. The frame: orion/curiosity/replica/frame.py.
4. The answer is parsed, checked with the same ``validate_plan`` Hub runs, and sent back on
   ``orion:curiosity:replica:plan:result:<correlation_id>``. A stance on a PRIME focus or signal is delivered
   first (below), so a refused focus has left the slot before Hub takes it.
5. The plan act (verb ``infer``) opened when the model was reached. A plan that cannot be used (abstain,
   unparseable, failed its own check, answered late) closes 0 at once. A plan Hub can use stays open; every
   CURIOSITY_REPLICA_SETTLE_INTERVAL_SEC the tick reads Hub's plan row and the run's spend-log score
   (Postgres, read-only) and closes it on that score (orion/curiosity/replica/track_record.py).

Answers to Prime (patch 4, design section 9; pure rules in orion/curiosity/replica/answers.py). The replica may
answer each of Prime's acts on curiosity once -- a focus question Prime queued, a regulate signal on
``durable.curiosity_investigate`` -- with support, challenge or refuse and a one-sentence reason:

* inside a plan (the answer's ``focus_stance`` / ``signal_stance``), or
* between runs: the tick watches for a NEW Prime act (the signal hash every tick; Hub's pending focus row every
  WATCH_INTERVAL_SEC, read-only Postgres). Each one wakes ONE reply thought (bid, one short model call, the
  REPLY frame), even with no plan request.

Every answer is an act (actor curiosity.replica.answer -- its own actor, so answers stay out of the
curiosity.replica family's history in the body economy; nothing bids as it -- verb ask, parent = Prime's act) that closes at once with the
stance and reason as its note. At most CURIOSITY_REPLICA_ANSWERS_PER_DAY in 24 h, CURIOSITY_REPLICA_ANSWER_REFRACTORY_SEC
apart, never to an answer. A deterministic share is held back (CURIOSITY_REPLICA_REFUSAL_HOLDBACK_FRACTION,
CURIOSITY_REPLICA_CHALLENGE_HOLDBACK_FRACTION): recorded, not applied, never shown to Prime. Applied:

* refuse a focus: the RPC ``orion:curiosity:replica:focus_refusal:request`` (Hub runs orion/curiosity/focus.py
  ``refuse_focus``; Juniper's focus can never be refused there);
* refuse a signal: ``prime_charge.PrimeChargeRefusalV1`` in field ``refusal`` of the signal's own hash; the market
  stops applying the charge from the next bid;
* challenge or refuse (delivered): in the replies view, Redis ``orion:curiosity:replica:replies``, written on every
  save; Prime reads it (ORGAN REPLIES in its frame; its settle notes name a refusal).

Everything here is best-effort: a failed read shows as a gap in the frame; a failed publish is logged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Optional
from uuid import UUID, uuid4

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.curiosity.replica.answers import (
    ANSWER_COMPONENT,
    AnswerRecordV1,
    Block,
    TargetV1,
    answer_act,
    answer_blocker,
    answer_note,
    answer_produced,
    draw_answer_arm,
    focus_target,
    keep_answers,
    question_act_id,
    replies_view,
    reply_act,
    signal_target,
)
from orion.curiosity.replica.frame import (
    ReplicaFrameInputs,
    ReplyFrameInputs,
    build_frame,
    build_reply_frame,
    frame_tokens,
    parse_answer,
    parse_reply,
    reply_system_prompt,
    system_prompt,
)
from orion.curiosity.replica.plan import validate_plan
from orion.curiosity.replica.state import (
    PLAN_ACT_HORIZON_SEC,
    PLAN_COMPONENT,
    PLAN_REF_PREFIX,
    SOURCE_SERVICE,
    STATE_KEY,
    CuriosityReplicaStateV1,
    OpenPlanV1,
    add_note,
    drop_target,
    hold_plan,
    note_woken,
    opened_before,
    plan_act,
)
from orion.curiosity.replica.track_record import settle_plan
from orion.mindloop.gate import gate_why
from orion.mindloop.instance import LoopIdentity
from orion.mindloop.runtime import MindLoopRuntime, ModelReply, utcnow
from orion.mindloop.turns import JUNIPER_TURNS, JuniperTurns
from orion.physiology import prime_charge as pc
from orion.prime.state import PRIME_CONCERNS_KEY, PrimeConcernsViewV1
from orion.schemas.curiosity_replica import (
    FOCUS_REFUSAL_REQUEST_CHANNEL,
    FOCUS_REFUSAL_REQUEST_KIND,
    FOCUS_REFUSAL_RESULT_CHANNEL_PREFIX,
    FOCUS_REFUSAL_RESULT_KIND,
    PLAN_REQUEST_CHANNEL,
    PLAN_REQUEST_KIND,
    PLAN_RESULT_CHANNEL_PREFIX,
    PLAN_RESULT_KIND,
    REPLICA_ACTOR,
    REPLICA_FAMILY,
    REPLIES_KEY,
    REPLIES_WINDOW_SEC,
    CuriosityFocusRefusalRequestV1,
    CuriosityFocusRefusalResultV1,
    CuriosityOfferPlanRequestV1,
    CuriosityOfferPlanResultV1,
)
from orion.schemas.grammar import ActV1

logger = logging.getLogger("orion-mind.curiosity_replica")

PRODUCER = "orion-mind.curiosity_replica"
CURIOSITY_FAMILY = "durable.curiosity_investigate"
HTTP_TIMEOUT_SEC = 4.0
RUNS_DAYS = 3
# The answer must be on the bus this long before Hub stops waiting.
ANSWER_MARGIN_SEC = 5.0
# The smallest seat wait a try asks for (never 0: see _seat_wait_sec).
MIN_SEAT_WAIT_SEC = 0.5
# A failed turn (turn_ok false) is final only this long after its outcome row: a durable retry (up to 3
# attempts, within hours) overwrites the row (curiosity_offer_decisions.py UPSERT_OUTCOME_SQL).
FAILED_TURN_FINAL_SEC = 6 * 3600.0
# A plan with no Hub plan row after this long is let go ungraded (unknown is never a 0).
PLAN_ROW_GRACE_SEC = 3600.0
RECORD_PLANS = 10
# How often the tick reads Hub's pending focus row for a new Prime question (read-only Postgres). The signal
# hash is read every tick (one Redis read).
WATCH_INTERVAL_SEC = 60.0
# The focus refusal RPC's own wait; inside a plan it is cut to the time left before Hub's deadline.
FOCUS_REFUSAL_RPC_TIMEOUT_SEC = 10.0
# A reply thought waits at most this long for a seat, so a plan request that arrives meanwhile still has time.
REPLY_SEAT_WAIT_MAX_SEC = 20.0
# A reply thought whose bid was not funded, or whose thought the act gate refused, is tried again only after
# this long (a reply has no deadline; without it the loop would bid every tick for as long as the act is live).
REPLY_RETRY_SEC = 300.0
_PG_CONNECT_TIMEOUT_SEC = 3
_PG_STATEMENT_TIMEOUT_MS = 5000

REPLICA_IDENTITY = LoopIdentity(
    name="curiosity_replica", family=REPLICA_FAMILY, actor=REPLICA_ACTOR, producer=PRODUCER, state_key=STATE_KEY,
    settings_prefix="CURIOSITY_REPLICA_", source_service=SOURCE_SERVICE, thought_component=PLAN_COMPONENT,
    thought_ref=PLAN_REF_PREFIX, tier=2, logger_name=logger.name, verb="infer", temperature=0.3, thinking=False,
    cost_unit="plan",
)

# Hub's plan row joined with the run's spend-log score (no grant to orion_readonly on the plan table: this
# reads it as the DSN's owner, read-only, and never shows the picks' reasons to Orion).
PLAN_ROWS_SQL = """
SELECT p.run_id, p.requested_at, p.status, p.fallback_reason, p.picks, p.act_id,
       o.turn_ok, o.realized_nats, o.unknown_reason, o.per_prior, o.completed_at
FROM curiosity_offer_plans p LEFT JOIN curiosity_run_outcomes o ON o.run_id = p.run_id
WHERE p.run_id = ANY(%(ids)s)
"""
RECENT_PLANS_SQL = """
SELECT p.run_id, p.requested_at, p.status, p.fallback_reason, p.picks, p.act_id,
       o.turn_ok, o.realized_nats, o.unknown_reason, o.per_prior
FROM curiosity_offer_plans p LEFT JOIN curiosity_run_outcomes o ON o.run_id = p.run_id
ORDER BY p.requested_at DESC LIMIT %(n)s
"""
FALLBACK_COUNTS_SQL = """
SELECT status, count(*) AS n FROM curiosity_offer_plans
WHERE requested_at > now() - interval '30 days' GROUP BY status
"""
OUTCOMES_SQL = """
SELECT run_id, turn_ok, realized_nats, unknown_reason, per_prior
FROM curiosity_run_outcomes WHERE run_id = ANY(%(ids)s)
"""
# The focus waiting in Hub's slot (orion/curiosity/focus.py PENDING_SQL's rule: pending, unexpired; a refused
# row is stored 'cleared'). Read-only; only a Prime focus becomes a target (answers.focus_target).
PENDING_FOCUS_SQL = """
SELECT focus_id::text AS focus_id, origin, question, chat_correlation_id, created_at, expires_at
FROM curiosity_focus_request
WHERE status = 'pending' AND expires_at > now()
ORDER BY created_at DESC LIMIT 1
"""
# After a focus refusal RPC with no usable reply: did Hub apply it anyway? (read-only; the row names the act)
FOCUS_REFUSED_SQL = """
SELECT refused_act_id FROM curiosity_focus_request WHERE focus_id::text = %(id)s
"""
# The newest chat-history compactor digest (orion/cognition/chat_history_compactor; services/orion-cortex-orch
# chat_history_compactor_memory.py writes the card, workflow_runtime.py the journal entry): the card's summary
# and the journal entry's title, joined on the compactor index. Live 2026-11-04: one a day around 13:00 UTC.
DIGEST_SQL = """
SELECT mc.summary, mc.created_at, mc.subschema->>'window_end' AS window_end,
       mc.subschema->>'compactor_index' AS index,
       (SELECT je.title FROM journal_entries je
         WHERE je.source_ref = 'chat_history_compactor_pass:' || (mc.subschema->>'compactor_index')
         ORDER BY je.created_at DESC LIMIT 1) AS title
FROM memory_cards mc
WHERE mc.provenance = 'chat_compactor' AND mc.status = 'active'
ORDER BY mc.created_at DESC LIMIT 1
"""


def _get_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_SEC) as resp:  # noqa: S310 -- internal service URL
        return json.loads(resp.read())


def _pg_rows(dsn: str, sql: str, params: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    """Read-only rows; raises on any failure (the caller shows a gap)."""
    if not dsn:
        raise RuntimeError("CURIOSITY_REPLICA_PG_DSN is empty")
    import psycopg2  # inside: a missing driver degrades to an unreadable read, not a dead loop
    import psycopg2.extras

    conn = psycopg2.connect(dsn, connect_timeout=_PG_CONNECT_TIMEOUT_SEC,
                            options=f"-c statement_timeout={_PG_STATEMENT_TIMEOUT_MS}")
    try:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params or {})
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:160]}"


@dataclass
class PendingRequest:
    request: CuriosityOfferPlanRequestV1
    correlation_id: UUID
    reply_to: str
    received_at: datetime
    answered: bool = False
    # Why no plan has been made yet: the status to answer with if time runs out.
    miss_status: str = "unfunded"
    miss_why: str = "no try started before the deadline"


class ReplicaRuntime(MindLoopRuntime):
    identity = REPLICA_IDENTITY
    state_model = CuriosityReplicaStateV1

    def __init__(
        self,
        *,
        settings: Any,
        bus: Any,
        llm: Any,
        get_json: Callable[[str], Any] = _get_json,
        pg_rows: Optional[Callable[[str, Optional[dict[str, Any]]], list[dict[str, Any]]]] = None,
        bid: Optional[Callable[[Any], Awaitable[Any]]] = None,
        release: Optional[Callable[[str], Awaitable[Any]]] = None,
        clock: Callable[[], datetime] = utcnow,
        turns: JuniperTurns = JUNIPER_TURNS,
    ) -> None:
        super().__init__(settings=settings, bus=bus, llm=llm, bid=bid, release=release, clock=clock, turns=turns)
        self.state: CuriosityReplicaStateV1
        self.get_json = get_json
        dsn = str(getattr(settings, "CURIOSITY_REPLICA_PG_DSN", "") or "").strip()
        self.pg_rows = pg_rows or (lambda sql, params=None: _pg_rows(dsn, sql, params))
        self.requests: list[PendingRequest] = []
        self._current: Optional[PendingRequest] = None
        # The Prime act a reply thought (no plan request) is answering.
        self._current_reply: Optional[TargetV1] = None
        self._reached_model = False
        self._last_settle: Optional[datetime] = None
        self._last_watch: Optional[datetime] = None
        # The Prime focus last seen pending (None: none, or not read yet); a target only while it is pending.
        self._pending_prime_focus: Optional[TargetV1] = None
        # No reply thought before this (in memory: a restart costs at most one early bid).
        self._reply_retry_at: Optional[datetime] = None
        # A reply thought is between its bid and its end (a plan request that arrives now has to wait).
        self._replying = False
        self._listener: Optional[asyncio.Task] = None

    # --- answers to Prime: settings ----------------------------------------------------------
    def _answers_on(self) -> bool:
        return bool(self.cfg("ANSWERS_ENABLED", False))

    def _caps(self) -> tuple[int, float]:
        return int(self.cfg("ANSWERS_PER_DAY", 3)), float(self.cfg("ANSWER_REFRACTORY_SEC", 1800.0))

    def _prompt_caps(self) -> dict[str, int]:
        per_day, refractory = self._caps()
        return {"per_day": per_day, "refractory_min": int(refractory // 60)}

    def _blocker(self, target: TargetV1, now: datetime, *, thought_id: Optional[str] = None) -> Optional[Block]:
        per_day, refractory = self._caps()
        return answer_blocker(self.state.answers, target, now, per_day=per_day, refractory_sec=refractory,
                              thought_id=thought_id)

    # --- persistence: the replica's first published call -------------------------------------
    async def save(self) -> None:
        """Save the state, then publish the replies view (Redis ``orion:curiosity:replica:replies``): the
        delivered challenges and refusals of the last 48 h, the only thing Prime reads of the replica. Written
        on every save while answering is on, so its ``as_of`` doubles as the replica's liveness."""
        await super().save()
        if not self._answers_on():
            return
        try:
            # Expires with its window: a replica switched off leaves nothing behind for Prime to read past 48 h.
            await self.bus.redis.set(REPLIES_KEY, replies_view(self.state.answers, self.clock()).model_dump_json(),
                                     ex=int(REPLIES_WINDOW_SEC))
        except Exception as exc:  # noqa: BLE001
            logger.warning("curiosity_replica_replies_publish_failed err=%s", type(exc).__name__)

    # --- requests ----------------------------------------------------------------------------
    def accept(self, envelope: BaseEnvelope) -> bool:
        """Queue one plan request and wake the loop. False (logged) when it is not one."""
        if envelope.kind != PLAN_REQUEST_KIND:
            return False
        try:
            req = CuriosityOfferPlanRequestV1.model_validate(envelope.payload)
        except Exception as exc:  # noqa: BLE001 -- Hub then times out and falls back
            logger.warning("curiosity_replica_request_invalid err=%s", str(exc)[:300])
            return False
        reply = envelope.reply_to or f"{PLAN_RESULT_CHANNEL_PREFIX}{req.correlation_id}"
        if self._current is not None and self._current.request.run_id == req.run_id:
            logger.info("curiosity_replica_request_duplicate run=%s (already planning it)", req.run_id)
            return False
        self.requests = [r for r in self.requests if r.request.run_id != req.run_id]
        pending = PendingRequest(request=req, correlation_id=envelope.correlation_id, reply_to=reply,
                                 received_at=self.clock())
        if self._replying:
            # The loop is busy answering Prime: if this request runs out of time, that is why (the fallback
            # reasons are how a replica losing by starvation is seen, design section 8).
            pending.miss_status, pending.miss_why = "unfunded", "a reply thought to Prime was running when it arrived"
        self.requests.append(pending)
        logger.info("curiosity_replica_request run=%s candidates=%d stale=%d focus=%s deadline=%s",
                    req.run_id, len(req.candidates), len(req.stale_ids),
                    req.pending_focus.origin if req.pending_focus else "-", req.deadline_at.isoformat())
        self.wake()
        return True

    async def _listen(self) -> None:
        while not self._stop.is_set():
            try:
                async with self.bus.subscribe(PLAN_REQUEST_CHANNEL) as pubsub:
                    logger.info("curiosity_replica_listening channel=%s", PLAN_REQUEST_CHANNEL)
                    async for raw in self.bus.iter_messages(pubsub):
                        try:
                            decoded = self.bus.codec.decode(raw.get("data"))
                            if decoded.ok:
                                self.accept(decoded.envelope)
                        except Exception:  # noqa: BLE001
                            logger.warning("curiosity_replica_message_failed", exc_info=True)
                logger.warning("curiosity_replica_listener_ended (resubscribing)")
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.warning("curiosity_replica_listener_reconnecting", exc_info=True)
            await asyncio.sleep(1)  # never a hot loop, whether the subscription failed or just ended

    async def stop(self) -> None:
        if self._listener is not None:
            self._listener.cancel()
            try:
                await self._listener
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._listener = None
        await super().stop()

    def _time_left(self, pending: PendingRequest, now: datetime) -> float:
        """Seconds left for seat wait + generation before the answer must be on the bus."""
        return (pending.request.deadline_at - now).total_seconds() - ANSWER_MARGIN_SEC

    async def _answer(self, pending: PendingRequest, result: CuriosityOfferPlanResultV1) -> None:
        pending.answered = True
        try:
            await self.bus.publish(pending.reply_to, BaseEnvelope(
                kind=PLAN_RESULT_KIND, correlation_id=pending.correlation_id, source=self.source,
                payload=result.model_dump(mode="json")))
        except Exception as exc:  # noqa: BLE001 -- Hub then times out and falls back
            logger.warning("curiosity_replica_answer_failed run=%s err=%s", pending.request.run_id, type(exc).__name__)

    def _latency_ms(self, pending: PendingRequest) -> int:
        return max(0, int((self.clock() - pending.received_at).total_seconds() * 1000))

    async def _give_up(self, pending: PendingRequest, now: datetime) -> None:
        """No time left for a model call: tell Hub why (no act: no thought happened)."""
        await self._answer(pending, CuriosityOfferPlanResultV1(
            run_id=pending.request.run_id, plan_id=f"none-{pending.request.correlation_id}",
            status=pending.miss_status, latency_ms=self._latency_ms(pending)))
        logger.info("curiosity_replica_gave_up run=%s status=%s why=%s", pending.request.run_id, pending.miss_status,
                    pending.miss_why)
        self.state = add_note(self.state, f"at {now:%H:%M} UTC I could not plan run {pending.request.run_id} in "
                                          f"time ({pending.miss_status}: {pending.miss_why}); it had the rule order")

    # --- loop hooks (orion/mindloop/runtime.py) ----------------------------------------------
    def _log_started(self) -> None:
        logger.info("curiosity_replica_started tick_sec=%s route=%s open_plans=%d", self.cfg("TICK_SEC"),
                    self.cfg("LLM_ROUTE"), len(self.state.open_plans))
        if self._listener is None:
            self._listener = asyncio.create_task(self._listen(), name="orion-curiosity-replica-listener")

    async def _upkeep(self) -> tuple[Any, datetime]:
        now = self.clock()
        budget = float(self.cfg("LLM_TIMEOUT_SEC"))
        expired = [r for r in self.requests if not r.answered and self._time_left(r, now) < budget]
        # Rebuilt before any await: a request the listener adds while a give-up publishes is kept.
        self.requests = [r for r in self.requests if not r.answered and r not in expired]
        for pending in expired:
            await self._give_up(pending, now)
        await self._maybe_settle(now)
        if self._answers_on():
            await self._watch(now)
        return None, now

    # --- answers to Prime: what is there to answer -------------------------------------------
    async def _live_signal(self) -> Optional[pc.PrimeRegulationV1]:
        """Prime's signal in force on curiosity (either arm: the replica cannot tell them apart), or None."""
        fields = await self.bus.redis.hgetall(pc.key(CURIOSITY_FAMILY)) or {}
        sig = pc.parse_signal(fields.get(pc.SIGNAL_FIELD, fields.get(pc.SIGNAL_FIELD.encode())))
        return sig if sig is not None and pc.active(sig, self.clock()) else None

    async def _watch(self, now: datetime) -> None:
        """A NEW Prime act on curiosity wakes one reply thought (design section 9, "between runs"). The signal
        hash is read every tick; Hub's pending focus every WATCH_INTERVAL_SEC. A target that is gone (the focus
        left the slot, the signal ended) leaves the queue."""
        targets: list[TargetV1] = []
        try:
            sig = await self._live_signal()
            if sig is not None:
                targets.append(signal_target(sig))
            live_signal = sig.signal_id if sig is not None else None
        except Exception as exc:  # noqa: BLE001 -- unread: nothing new, nothing dropped
            logger.info("curiosity_replica_watch_signal_unread err=%s", _err(exc))
            live_signal = "unread"
        if self._last_watch is None or (now - self._last_watch).total_seconds() >= WATCH_INTERVAL_SEC:
            self._last_watch = now
            try:
                rows = await asyncio.to_thread(self.pg_rows, PENDING_FOCUS_SQL, None)
                row = rows[0] if rows else None
                self._pending_prime_focus = (focus_target(focus_id=str(row["focus_id"]), question=row.get("question") or "",
                                                          chat_correlation_id=row.get("chat_correlation_id"))
                                             if row is not None else None)
            except Exception as exc:  # noqa: BLE001 -- unread: keep what was last seen
                logger.info("curiosity_replica_watch_focus_unread err=%s", _err(exc))
        if self._pending_prime_focus is not None:
            targets.append(self._pending_prime_focus)
        live = {t.prime_act_id for t in targets}
        keep = []
        for t in self.state.pending_targets:
            gone = (t.kind == "signal" and live_signal != "unread" and t.target_id != live_signal) or (
                t.kind == "focus" and t.prime_act_id not in live)
            if gone:
                logger.info("curiosity_replica_target_gone kind=%s target=%s prime_act=%s", t.kind, t.target_id,
                            t.prime_act_id)
            else:
                keep.append(t)
        self.state = self.state.model_copy(update={"pending_targets": keep})
        for t in targets:
            block = self._blocker(t, now)
            if t.prime_act_id in self.state.woken_for or (block is not None and block.code == "answered"):
                continue
            self.state = note_woken(self.state, t)
            logger.info("curiosity_replica_woken kind=%s target=%s prime_act=%s", t.kind, t.target_id, t.prime_act_id)
            self.wake()

    def _next_reply_target(self, now: datetime) -> Optional[TargetV1]:
        """The first queued Prime act the replica may still answer now. One that can never be answered (already
        answered, an answer itself) leaves the queue; one waiting on the daily cap or the refractory stays."""
        for t in list(self.state.pending_targets):
            block = self._blocker(t, now)
            if block is None:
                return t
            if block.final:
                self.state = drop_target(self.state, t.prime_act_id)
        return None

    # --- answers to Prime: delivering one ----------------------------------------------------
    async def _deliver(self, target: TargetV1, stance: str, reason: str, *, source: str, now: datetime,
                       thought_id: Optional[str] = None,
                       deadline_at: Optional[datetime] = None) -> Optional[AnswerRecordV1]:
        """One answer to one Prime act: checked, drawn, applied (treated refusal) and recorded as a closed act.
        None when it may not be made (the reason is a note in the next frame). Never raises. ``deadline_at``:
        the plan's deadline (a focus refusal RPC must end ANSWER_MARGIN_SEC before it)."""
        if not self._answers_on():
            return None
        block = self._blocker(target, now, thought_id=thought_id)
        if block is not None:
            logger.info("curiosity_replica_answer_skipped kind=%s target=%s stance=%s why=%s", target.kind,
                        target.target_id, stance, block.code)
            self.state = add_note(self.state, f"I did not answer Prime's {target.kind} {target.target_id}: {block.why}")
            return None
        arm = draw_answer_arm(target.prime_act_id, stance,
                              refusal_fraction=float(self.cfg("REFUSAL_HOLDBACK_FRACTION", 0.2)),
                              challenge_fraction=float(self.cfg("CHALLENGE_HOLDBACK_FRACTION", 0.2)))
        act = answer_act(target, arm=arm)
        # The self-monitor gate on the answer, asked before anything is applied (both arms alike).
        # Under the answer's own actor (curiosity.replica.answer): its track record, not the plans'.
        d = await self._gate_decide("ask", act.object_ref or "", actor=act.actor)
        if not self._admit(d):
            logger.info("curiosity_replica_act_refused verb=ask gate=%s", d.label() if d else "-")
            self.state = add_note(self.state, f"my answer to Prime's {target.kind} {target.target_id} was not "
                                              f"made: {gate_why(d, self.gate.probe_interval_sec)}")
            return None
        await self._open(act, ANSWER_COMPONENT, at=now)
        delivered, note = False, ""
        if arm == "treated" and stance == "refuse":
            if target.kind == "focus":
                delivered, note = await self._refuse_focus(target, reason, act.act_id, deadline_at=deadline_at)
            else:
                delivered, note = await self._refuse_signal(target, reason, act.act_id, now)
        elif arm == "treated":
            delivered = True   # a challenge reaches Prime through the replies view (saved next); a support changes nothing
        record = AnswerRecordV1(answer_act_id=act.act_id, prime_act_id=target.prime_act_id, kind=target.kind,
                                target_id=target.target_id, stance=stance, reason=reason, arm=arm, delivered=delivered,
                                delivery_note=note[:300], about=target.about[:200], source=source,
                                thought_id=thought_id, at=now)
        self.state = drop_target(self.state.model_copy(update={"answers": keep_answers(self.state.answers, record)}),
                                 target.prime_act_id)
        await self._close(act, ANSWER_COMPONENT, produced=answer_produced(record), note=answer_note(record), at=now)
        logger.info("curiosity_replica_answer prime_act=%s act=%s kind=%s target=%s stance=%s arm=%s delivered=%s "
                    "source=%s note=%s", target.prime_act_id, act.act_id, target.kind, target.target_id, stance, arm,
                    delivered, source, note[:160] or "-")
        return record

    async def _refuse_focus(self, target: TargetV1, reason: str, act_id: str, *,
                            deadline_at: Optional[datetime]) -> tuple[bool, str]:
        """Hub applies the refusal (orion/curiosity/focus.py refuse_focus). (delivered, why not). The request
        carries ``not_after`` (Hub writes nothing past it), so after a lost reply the focus row is the truth."""
        timeout = FOCUS_REFUSAL_RPC_TIMEOUT_SEC
        if deadline_at is not None:
            # Measured right before the call: the gate and the act open already spent some of it.
            timeout = min(timeout, (deadline_at - self.clock()).total_seconds() - ANSWER_MARGIN_SEC)
        if timeout < 1.0:
            return False, "no time left before Hub's plan deadline to ask Hub"
        corr = uuid4()
        reply = f"{FOCUS_REFUSAL_RESULT_CHANNEL_PREFIX}{corr}"
        req = CuriosityFocusRefusalRequestV1(focus_id=target.target_id, reason=reason, act_id=act_id,
                                             prime_act_id=target.prime_act_id,
                                             not_after=self.clock() + timedelta(seconds=timeout))
        try:
            raw = await self.bus.rpc_request(
                FOCUS_REFUSAL_REQUEST_CHANNEL,
                BaseEnvelope(kind=FOCUS_REFUSAL_REQUEST_KIND, correlation_id=corr, reply_to=reply, source=self.source,
                             payload=req.model_dump(mode="json")),
                reply_channel=reply, timeout_sec=timeout, require_listener=True)
            decoded = self.bus.codec.decode(raw.get("data"))
            if not decoded.ok or decoded.envelope.kind != FOCUS_REFUSAL_RESULT_KIND:
                return await self._refusal_on_row(target, act_id, "Hub's reply was malformed")
            result = CuriosityFocusRefusalResultV1.model_validate(decoded.envelope.payload)
        except Exception as exc:  # noqa: BLE001
            return await self._refusal_on_row(target, act_id, f"no answer from Hub ({type(exc).__name__})")
        if result.focus_id != target.target_id:
            return False, f"Hub answered for another focus ({result.focus_id})"
        return (True, "") if result.ok else (False, f"Hub refused: {result.error or 'no reason given'}")

    async def _refusal_on_row(self, target: TargetV1, act_id: str, what: str) -> tuple[bool, str]:
        """No usable reply: read the focus row (read-only). Hub writes nothing past the request's ``not_after``,
        so the row now says whether THIS refusal (its act id) was applied."""
        try:
            rows = await asyncio.to_thread(self.pg_rows, FOCUS_REFUSED_SQL, {"id": target.target_id})
        except Exception as exc:  # noqa: BLE001
            return False, f"{what}; the focus row could not be read ({_err(exc)}): answer unknown"
        if rows and rows[0].get("refused_act_id") == act_id:
            return True, ""
        return False, f"{what}; the focus row shows no refusal of mine"

    async def _refuse_signal(self, target: TargetV1, reason: str, act_id: str, now: datetime) -> tuple[bool, str]:
        """The refusal goes next to the signal in its own hash; the market reads it from the next bid."""
        key = pc.key(CURIOSITY_FAMILY)
        redis = self.bus.redis
        try:
            sig = await self._live_signal()
            if sig is None or sig.signal_id != target.target_id:
                return False, "the signal is no longer in force (ended or replaced)"
            refusal = pc.PrimeChargeRefusalV1(signal_id=sig.signal_id, act_id=act_id, by=REPLICA_ACTOR,
                                              reason=reason[:pc.REFUSAL_REASON_MAX], at=now)
            # HSET keeps the key's expiry (Prime set it to the signal's end + grace). If the key expired in between,
            # HSET made a new key with no expiry: give it the grace so nothing is left forever.
            await redis.hset(key, pc.REFUSAL_FIELD, pc.dumps(refusal))
            if await redis.ttl(key) == -1:
                await redis.expire(key, int(pc.GRACE_SEC))
        except Exception as exc:  # noqa: BLE001
            return False, f"the refusal could not be written ({_err(exc)})"
        return True, ""

    def _wants_to_think(self, now: datetime) -> bool:
        """A request that can still be answered in time. No refractory and no no-model backoff: each request
        has its own deadline, and there are about two a day. Picks the request this thought is for (the
        runtime builds the act before it asks). With no request: a queued Prime act to answer (a reply
        thought), after the no-model backoff (a reply has no deadline to bound its retries)."""
        self._current = self.requests[0] if self.requests else None
        self._current_reply = None
        if self._current is not None:
            return True
        if not self._answers_on() or not self.state.pending_targets:
            return False
        if self.state.retry_not_before is not None and now < self.state.retry_not_before:
            return False
        if self._reply_retry_at is not None and now < self._reply_retry_at:
            return False
        self._current_reply = self._next_reply_target(now)
        return self._current_reply is not None

    def _pending_count(self) -> int:
        return len(self.requests) + len(self.state.pending_targets)

    def _tell(self, text: str) -> None:
        self.state = add_note(self.state, text)

    def _thought_act(self, thought_id: str, horizon_sec: float) -> ActV1:
        if self._current is None and self._current_reply is not None:
            return reply_act(thought_id, horizon_sec=horizon_sec)
        return plan_act(thought_id, self._current.request.run_id if self._current else "unknown")

    def _woke_note(self) -> str:
        if self._current is None and self._current_reply is not None:
            t = self._current_reply
            return f"woke: Prime's new {t.kind} {t.target_id} (act {t.prime_act_id})"
        return f"woke: plan request for run {self._current.request.run_id}" if self._current else "woke: no request"

    def _want_extras(self) -> dict[str, Any]:
        if self._current is None and self._current_reply is not None:
            return {"object_ref": f"prime_act:{self._current_reply.prime_act_id}"}
        return {"object_ref": f"curiosity_run:{self._current.request.run_id}"} if self._current else {}

    def _seat_wait_sec(self) -> float:
        """Never wait for a seat past the point where generation could still end before the deadline."""
        cap = float(self.cfg("SEAT_WAIT_SEC", 0.0) or 0.0)
        if self._current is None:
            # A reply thought has no deadline, but a plan request may arrive meanwhile: keep its wait short.
            return min(cap, REPLY_SEAT_WAIT_MAX_SEC) if self._current_reply is not None else cap
        left = self._time_left(self._current, self.clock()) - float(self.cfg("LLM_TIMEOUT_SEC"))
        # Never 0: the runtime sends no seat wait for 0, and the gateway then waits its own default
        # (LLM_GATEWAY_POOL_WAIT_SEC, 300 s) for a seat -- a call that could run long after Hub gave up.
        return max(MIN_SEAT_WAIT_SEC, min(cap, left))

    def _may_retry_5xx(self) -> bool:
        """A second full call only if it can still end before the deadline."""
        if self._current is None:
            return True
        return self._time_left(self._current, self.clock()) - float(self.cfg("LLM_TIMEOUT_SEC")) >= MIN_SEAT_WAIT_SEC

    async def _thought_gate_admits(self, now: datetime) -> bool:
        ok = await super()._thought_gate_admits(now)
        if not ok and self._current is not None:
            self._current.miss_status, self._current.miss_why = "unfunded", "my own act gate refused the thought"
        elif not ok and self._current_reply is not None:
            self._reply_retry_at = now + timedelta(seconds=REPLY_RETRY_SEC)
        return ok

    async def tick(self) -> None:
        if self.turns.active() and self.requests:
            self.requests[0].miss_status, self.requests[0].miss_why = "unfunded", "a Juniper turn was running"
        await super().tick()

    async def _on_not_funded(self, decision: Any) -> None:
        if self._current is None and self._current_reply is not None:
            self._reply_retry_at = self.clock() + timedelta(seconds=REPLY_RETRY_SEC)
            return
        if self._current is not None:
            self._current.miss_status = "unfunded"
            self._current.miss_why = (f"bid {getattr(decision, 'outcome', None) or 'unanswered'}"
                                      f" ({getattr(decision, 'reason', None) or '-'})")

    async def _on_no_model(self, reply: ModelReply) -> None:
        if self._current is not None:
            self._current.miss_status = "unfunded" if reply.kind == "no_seat" else "model_failed"
            reason = (reply.gateway_error or {}).get("reason")
            self._current.miss_why = f"the try never reached the model ({reply.kind}{': ' + str(reason) if reason else ''})"

    async def _open_thought(self, act: ActV1, started_at: datetime) -> None:
        self._reached_model = True
        await super()._open_thought(act, started_at)

    async def _after_think(self) -> None:
        """A thought happened but sent no answer (it crashed before one): tell Hub at once. A reply thought that
        reached the model used its one wake, answered or not."""
        cur, self._current = self._current, None
        target, self._current_reply = self._current_reply, None
        self._replying = False
        reached, self._reached_model = self._reached_model, False
        if cur is None:
            if target is not None and reached:
                self.state = drop_target(self.state, target.prime_act_id)
            return
        if reached and not cur.answered:
            await self._answer(cur, CuriosityOfferPlanResultV1(
                run_id=cur.request.run_id, plan_id=f"none-{cur.request.correlation_id}", status="model_failed",
                latency_ms=self._latency_ms(cur)))
        if cur.answered:
            self.requests = [r for r in self.requests if r is not cur]

    def _parse_answer(self, parsed: Optional[dict[str, Any]]) -> tuple[Any, Optional[str]]:
        if self._current is None and self._current_reply is not None:
            return parse_reply(parsed)
        return parse_answer(parsed)

    async def _ask(self, inputs: Any, now: datetime) -> ModelReply:
        if self._current is None and self._current_reply is not None:
            self._replying = True
            reply_inputs = ReplyFrameInputs(now=now, target=self._current_reply, notes=list(self.state.notes),
                                            answers=list(self.state.answers), answers_on=self._answers_on())
            await self._read_world(reply_inputs, now)
            system, frame = reply_system_prompt(**self._prompt_caps()), build_reply_frame(reply_inputs)
            reply = await self._ask_model(system, frame)
            reply.context.update(frame_tokens=frame_tokens(system, frame))
            return reply
        pending = self._current
        assert pending is not None  # _wants_to_think chose it
        frame_inputs = await self._read_frame_inputs(pending.request, now)
        system, frame = system_prompt(**self._prompt_caps()), build_frame(frame_inputs)
        if self._time_left(pending, self.clock()) < float(self.cfg("LLM_TIMEOUT_SEC")):
            # The reads ate the time a model call needs: not a thought (no act); given up next tick.
            return ModelReply(route=self._route(), parsed=None, err="no time left after the frame reads",
                              gateway_error={"reason": "no time left after the frame reads"}, kind="no_seat")
        reply = await self._ask_model(system, frame)
        reply.context.update(frame_tokens=frame_tokens(system, frame),
                             signal_id=(frame_inputs.signal or {}).get("signal_id"), signal=frame_inputs.signal)
        return reply

    async def _settle_reply(self, act: ActV1, thought_id: str, inputs: Any, now: datetime,
                            reply: ModelReply) -> tuple[Optional[bool], str]:
        preempted = self.turns.started_since(reply.started_mono)
        produced, note = await super()._settle_reply(act, thought_id, inputs, now, reply)
        cur = self._current
        if cur is not None and not cur.answered:
            status = "model_failed" if (preempted or reply.err) else "unparseable"
            await self._answer(cur, CuriosityOfferPlanResultV1(
                run_id=cur.request.run_id, plan_id=thought_id, status=status, act_id=act.act_id,
                latency_ms=self._latency_ms(cur), frame_tokens=reply.context.get("frame_tokens")))
            logger.info("curiosity_replica_plan run=%s status=%s note=%s", cur.request.run_id, status, note[:200])
        return produced, note

    async def _conclude(self, act: ActV1, thought_id: str, inputs: Any, now: datetime, reply: ModelReply,
                        answer: Any) -> tuple[Optional[bool], str]:
        if self._current is None and self._current_reply is not None:
            target = self._current_reply
            record = await self._deliver(target, answer.stance, answer.reason, source="reply", now=self.clock(),
                                         thought_id=thought_id)
            what = (f"{record.stance} ({record.arm}{', delivered' if record.delivered else ''})" if record is not None
                    else "no answer made (see notes)")
            logger.info("curiosity_replica_reply prime_act=%s kind=%s %s route=%s", target.prime_act_id, target.kind,
                        what, reply.route)
            return True, f"answered Prime's {target.kind} {target.target_id}: {what}"
        pending = self._current
        assert pending is not None
        req = pending.request
        # Stances on Prime's acts are delivered BEFORE the plan result: a refused focus must have left the slot
        # before Hub takes it (Hub takes the focus after this answer).
        await self._deliver_plan_stances(answer, req, reply, thought_id)
        if answer.abstain:
            status = "abstain"
        elif not answer.picks:
            status = "unparseable"
        else:
            status = "ok"
        ids = [c.prior_id for c in req.candidates]
        verdict = (validate_plan(answer.picks, candidate_ids=ids, fresh_ids=ids, signal_stance=answer.signal_stance,
                                 live_signal_id=reply.context.get("signal_id")) if status == "ok" else None)
        late = self.clock() > req.deadline_at
        await self._answer(pending, CuriosityOfferPlanResultV1(
            run_id=req.run_id, plan_id=thought_id, status=status, picks=answer.picks if status == "ok" else [],
            focus_stance=answer.focus_stance, signal_stance=answer.signal_stance, act_id=act.act_id,
            latency_ms=self._latency_ms(pending), frame_tokens=reply.context.get("frame_tokens")))
        stances = ",".join(f"{k}={v.stance}" for k, v in (("focus", answer.focus_stance), ("signal", answer.signal_stance))
                           if v is not None) or "-"
        # No reading or reasons in the log: they stay on Hub's plan row only (design section 5).
        logger.info("curiosity_replica_plan run=%s status=%s picks=%s check=%s late=%s stances=%s route=%s",
                    req.run_id, status, ",".join(p.prior_id for p in answer.picks) or "-", verdict or "ok", late,
                    stances, reply.route)
        if status != "ok":
            return False, f"no plan: {status}"
        # Every plan sent as ok is graded from Hub's own row, even one this check or clock expects Hub to
        # refuse: Hub's verdict (its timer starts after the request is built; the live signal can change) is
        # the one the experiment scores, so the act must not guess it.
        self.state = hold_plan(self.state, OpenPlanV1(act_id=act.act_id, thought_id=thought_id, run_id=req.run_id,
                                                      opened_at=now, picks=[p.prior_id for p in answer.picks]))
        # The act stays open: it closes on Hub's plan row and the run's score (_maybe_settle).
        return None, "plan sent; graded on its run"

    async def _deliver_plan_stances(self, answer: Any, req: CuriosityOfferPlanRequestV1, reply: ModelReply,
                                    thought_id: str) -> None:
        """A plan's stance on a PRIME focus or signal is an answer to Prime (one per Prime act). A stance on
        Juniper's focus, or on one Orion chose in a conversation, is not: it stays on Hub's plan row only."""
        if not self._answers_on():
            return
        f = req.pending_focus
        if answer.focus_stance is not None and f is not None and f.from_prime:
            target = TargetV1(kind="focus", target_id=f.focus_id, prime_act_id=question_act_id(f.focus_id),
                              about=" ".join(f.question.split())[:600])
            await self._deliver(target, answer.focus_stance.stance, answer.focus_stance.reason, source="plan",
                                now=self.clock(), thought_id=thought_id, deadline_at=req.deadline_at)
        sig = reply.context.get("signal") or {}
        if (answer.signal_stance is not None and sig.get("act_id")
                and answer.signal_stance.signal_id == sig.get("signal_id")):
            target = TargetV1(kind="signal", target_id=str(sig["signal_id"]), prime_act_id=str(sig["act_id"]),
                              about=f"{sig.get('direction')} curiosity: {' '.join(str(sig.get('why') or '').split())}"[:600],
                              direction=sig.get("direction"), expires_at=sig.get("expires_at"))
            await self._deliver(target, answer.signal_stance.stance, answer.signal_stance.reason, source="plan",
                                now=self.clock(), thought_id=thought_id)

    # --- reads -------------------------------------------------------------------------------
    async def _read_frame_inputs(self, req: CuriosityOfferPlanRequestV1, now: datetime) -> ReplicaFrameInputs:
        inp = ReplicaFrameInputs(now=now, request=req, notes=list(self.state.notes), answers=list(self.state.answers),
                                 answers_on=self._answers_on())
        await self._read_world(inp, now, plan=True)
        return inp

    async def _read_world(self, inp: Any, now: datetime, *, plan: bool = False) -> None:
        """The reads both frames share (runs, scores, Prime's view and signal, the digest); a plan's frame also
        reads its own track record. Each failed read is that section's gap."""
        base = str(self.cfg("HUB_URL")).rstrip("/")
        try:
            data = await asyncio.to_thread(self.get_json, f"{base}/curiosity/api/runs?days={RUNS_DAYS}&line=investigate")
            runs = data.get("runs") if isinstance(data, dict) else None
            if isinstance(runs, list):
                inp.runs = [r for r in runs if isinstance(r, dict)]
            else:
                inp.runs_error = "no runs list in Hub's answer"
        except Exception as exc:  # noqa: BLE001
            inp.runs_error = _err(exc)
        run_ids = [str(r.get("run_id")) for r in (inp.runs or []) if r.get("run_id")]

        reads = [("outcomes", OUTCOMES_SQL, {"ids": run_ids})]
        if plan:
            reads += [("record", RECENT_PLANS_SQL, {"n": RECORD_PLANS}), ("counts", FALLBACK_COUNTS_SQL, None)]
        reads.append(("digest", DIGEST_SQL, None))

        def pg() -> dict[str, Any]:
            out: dict[str, Any] = {}
            for name, sql, params in reads:
                try:
                    out[name] = self.pg_rows(sql, params)
                except Exception as exc:  # noqa: BLE001
                    out[name + "_error"] = _err(exc)
            return out

        rows = await asyncio.to_thread(pg)
        if "outcomes" in rows:
            inp.outcomes = {str(r["run_id"]): r for r in rows["outcomes"]}
        else:
            inp.outcomes_error = rows.get("outcomes_error")
        if plan and "record" in rows:
            inp.record_rows = rows["record"]
        elif plan:
            inp.record_error = rows.get("record_error")
        if plan and "counts" in rows:
            inp.fallback_counts = {str(r["status"]): int(r["n"]) for r in rows["counts"]}
            inp.plans_total = sum(inp.fallback_counts.values())
        if "digest" in rows:
            inp.digest = rows["digest"][0] if rows["digest"] else None
        else:
            inp.digest_error = rows.get("digest_error")
        try:
            raw = await self.bus.redis.get(PRIME_CONCERNS_KEY)
            if raw:
                inp.concerns = PrimeConcernsViewV1.model_validate_json(raw)
            else:
                inp.concerns_error = "Prime has published no concerns view"
        except Exception as exc:  # noqa: BLE001
            inp.concerns_error = _err(exc)
        try:
            fields = await self.bus.redis.hgetall(pc.key(CURIOSITY_FAMILY)) or {}
            sig = pc.parse_signal(fields.get(pc.SIGNAL_FIELD, fields.get(pc.SIGNAL_FIELD.encode())))
            if sig is not None and pc.active(sig, now):
                # No arm: the replica must not tell Prime's treated signals from its controls. The act id is the
                # Prime act an answer to it is parented to (never shown in the frame).
                inp.signal = {"signal_id": sig.signal_id, "direction": sig.direction, "why": sig.why,
                              "expires_at": sig.expires_at, "act_id": sig.act_id}
        except Exception as exc:  # noqa: BLE001
            inp.signal_error = _err(exc)

    # --- settling plans ----------------------------------------------------------------------
    async def _maybe_settle(self, now: datetime) -> None:
        if not self.state.open_plans:
            return
        due = self._last_settle is None or (now - self._last_settle).total_seconds() >= float(self.cfg("SETTLE_INTERVAL_SEC"))
        if not due:
            return
        self._last_settle = now
        ids = [p.run_id for p in self.state.open_plans]
        try:
            rows = await asyncio.to_thread(self.pg_rows, PLAN_ROWS_SQL, {"ids": ids})
        except Exception as exc:  # noqa: BLE001
            logger.info("curiosity_replica_settle_skipped err=%s", _err(exc))
            return
        by_run = {str(r["run_id"]): r for r in rows}
        keep: list[OpenPlanV1] = []
        for plan in self.state.open_plans:
            act = plan_act(plan.thought_id, plan.run_id)
            row = by_run.get(plan.run_id)
            if row is None:
                if opened_before(plan, now, PLAN_ROW_GRACE_SEC):
                    # Unknown, never a 0: Hub may have restarted mid-wait (no run), or used the plan and then
                    # failed to write its row. The ledger's horizon expires the act.
                    logger.info("curiosity_replica_plan_unrecorded run=%s act=%s (left to the act's horizon)",
                                plan.run_id, plan.act_id)
                    self.state = add_note(self.state, f"Hub has no record of my plan for run {plan.run_id}")
                else:
                    keep.append(plan)
                continue
            else:
                s = settle_plan(row, now=now, failed_final_sec=FAILED_TURN_FINAL_SEC)
                if s.kind == "pending":
                    if opened_before(plan, now, PLAN_ACT_HORIZON_SEC):
                        self.state = add_note(self.state, f"my plan for run {plan.run_id} got no score before its horizon")
                    else:
                        keep.append(plan)
                    continue
                if s.kind == "ungradable":
                    logger.info("curiosity_replica_plan_ungradable run=%s act=%s note=%s", plan.run_id, plan.act_id, s.note)
                    continue
                produced, note = s.produced, s.note
            if not await self._close(act, PLAN_COMPONENT, produced=produced, note=f"plan for run {plan.run_id} | {note}"):
                keep.append(plan)
                continue
            logger.info("curiosity_replica_plan_settled run=%s act=%s produced=%s note=%s", plan.run_id, plan.act_id,
                        produced, note)
        self.state = self.state.model_copy(update={"open_plans": keep})


__all__ = ["REPLICA_IDENTITY", "PendingRequest", "ReplicaRuntime", "STATE_KEY"]
