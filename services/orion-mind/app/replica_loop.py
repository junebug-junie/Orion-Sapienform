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
3. Funded: one model call on route CURIOSITY_REPLICA_LLM_ROUTE (``prime``, the warm chat lane). The seat wait
   is cut so seat wait + generation end before the deadline. The frame: orion/curiosity/replica/frame.py.
4. The answer is parsed, checked with the same ``validate_plan`` Hub runs, and sent back on
   ``orion:curiosity:replica:plan:result:<correlation_id>``. Stances (support / challenge / refuse) are sent
   and recorded only: answering Prime is patch 4.
5. The plan act (verb ``infer``) opened when the model was reached. A plan that cannot be used (abstain,
   unparseable, failed its own check, answered late) closes 0 at once. A plan Hub can use stays open; every
   CURIOSITY_REPLICA_SETTLE_INTERVAL_SEC the tick reads Hub's plan row and the run's spend-log score
   (Postgres, read-only) and closes it on that score (orion/curiosity/replica/track_record.py).

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
from uuid import UUID

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.curiosity.replica.frame import ReplicaFrameInputs, build_frame, frame_tokens, parse_answer, system_prompt
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
    hold_plan,
    opened_before,
    plan_act,
)
from orion.curiosity.replica.track_record import settle_plan
from orion.mindloop.instance import LoopIdentity
from orion.mindloop.runtime import MindLoopRuntime, ModelReply, utcnow
from orion.mindloop.turns import JUNIPER_TURNS, JuniperTurns
from orion.physiology import prime_charge as pc
from orion.prime.state import PRIME_CONCERNS_KEY, PrimeConcernsViewV1
from orion.schemas.curiosity_replica import (
    PLAN_REQUEST_CHANNEL,
    PLAN_REQUEST_KIND,
    PLAN_RESULT_CHANNEL_PREFIX,
    PLAN_RESULT_KIND,
    REPLICA_ACTOR,
    REPLICA_FAMILY,
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
# A used plan with no Hub plan row after this long: Hub never recorded it (its write failed, or it never
# got the answer). Closed 0: the run had the rule order as far as anyone can show.
PLAN_ROW_GRACE_SEC = 3600.0
RECORD_PLANS = 10
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
       o.turn_ok, o.realized_nats, o.unknown_reason, o.per_prior
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
        self._reached_model = False
        self._last_settle: Optional[datetime] = None
        self._listener: Optional[asyncio.Task] = None

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
        self.requests = [r for r in self.requests if r.request.run_id != req.run_id]
        self.requests.append(PendingRequest(request=req, correlation_id=envelope.correlation_id, reply_to=reply,
                                            received_at=self.clock()))
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
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.warning("curiosity_replica_listener_reconnecting", exc_info=True)
                await asyncio.sleep(1)

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
        keep = []
        for pending in self.requests:
            if pending.answered:
                continue
            if self._time_left(pending, now) < float(self.cfg("LLM_TIMEOUT_SEC")):
                await self._give_up(pending, now)
                continue
            keep.append(pending)
        self.requests = keep
        await self._maybe_settle(now)
        return None, now

    def _wants_to_think(self, now: datetime) -> bool:
        """A request that can still be answered in time. No refractory and no no-model backoff: each request
        has its own deadline, and there are about two a day. Picks the request this thought is for (the
        runtime builds the act before it asks)."""
        self._current = self.requests[0] if self.requests else None
        return self._current is not None

    def _pending_count(self) -> int:
        return len(self.requests)

    def _tell(self, text: str) -> None:
        self.state = add_note(self.state, text)

    def _thought_act(self, thought_id: str, horizon_sec: float) -> ActV1:
        return plan_act(thought_id, self._current.request.run_id if self._current else "unknown")

    def _woke_note(self) -> str:
        return f"woke: plan request for run {self._current.request.run_id}" if self._current else "woke: no request"

    def _want_extras(self) -> dict[str, Any]:
        return {"object_ref": f"curiosity_run:{self._current.request.run_id}"} if self._current else {}

    def _seat_wait_sec(self) -> float:
        """Never wait for a seat past the point where generation could still end before the deadline."""
        cap = float(self.cfg("SEAT_WAIT_SEC", 0.0) or 0.0)
        if self._current is None:
            return cap
        left = self._time_left(self._current, self.clock()) - float(self.cfg("LLM_TIMEOUT_SEC"))
        return max(0.0, min(cap, left))

    async def _thought_gate_admits(self, now: datetime) -> bool:
        ok = await super()._thought_gate_admits(now)
        if not ok and self._current is not None:
            self._current.miss_status, self._current.miss_why = "unfunded", "my own act gate refused the thought"
        return ok

    async def tick(self) -> None:
        if self.turns.active() and self.requests:
            self.requests[0].miss_status, self.requests[0].miss_why = "unfunded", "a Juniper turn was running"
        await super().tick()

    async def _on_not_funded(self, decision: Any) -> None:
        if self._current is not None:
            self._current.miss_status = "unfunded"
            self._current.miss_why = (f"bid {getattr(decision, 'outcome', None) or 'unanswered'}"
                                      f" ({getattr(decision, 'reason', None) or '-'})")

    async def _on_no_model(self, reply: ModelReply) -> None:
        if self._current is not None:
            self._current.miss_status = "unfunded" if reply.kind == "no_seat" else "model_failed"
            self._current.miss_why = f"the try never reached the model ({reply.kind})"

    async def _open_thought(self, act: ActV1, started_at: datetime) -> None:
        self._reached_model = True
        await super()._open_thought(act, started_at)

    async def _after_think(self) -> None:
        """A thought happened but sent no answer (it crashed before one): tell Hub at once."""
        cur, self._current = self._current, None
        reached, self._reached_model = self._reached_model, False
        if cur is None:
            return
        if reached and not cur.answered:
            await self._answer(cur, CuriosityOfferPlanResultV1(
                run_id=cur.request.run_id, plan_id=f"none-{cur.request.correlation_id}", status="model_failed",
                latency_ms=self._latency_ms(cur)))
        if cur.answered:
            self.requests = [r for r in self.requests if r is not cur]

    def _parse_answer(self, parsed: Optional[dict[str, Any]]) -> tuple[Any, Optional[str]]:
        return parse_answer(parsed)

    async def _ask(self, inputs: Any, now: datetime) -> ModelReply:
        pending = self._current
        assert pending is not None  # _wants_to_think chose it
        frame_inputs = await self._read_frame_inputs(pending.request, now)
        system, frame = system_prompt(), build_frame(frame_inputs)
        reply = await self._ask_model(system, frame)
        reply.context.update(frame_tokens=frame_tokens(system, frame),
                             signal_id=(frame_inputs.signal or {}).get("signal_id"))
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
        pending = self._current
        assert pending is not None
        req = pending.request
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
        logger.info("curiosity_replica_plan run=%s status=%s picks=%s check=%s late=%s stances=%s route=%s reading=%r",
                    req.run_id, status, ",".join(p.prior_id for p in answer.picks) or "-", verdict or "ok", late,
                    stances, reply.route, answer.reading[:200])
        if status != "ok":
            return False, f"no plan: {status}"
        if verdict is not None:
            return False, f"plan failed its own check ({verdict}); the run had the rule order"
        if late:
            return False, "answered after Hub's deadline; the run had the rule order"
        self.state = hold_plan(self.state, OpenPlanV1(act_id=act.act_id, thought_id=thought_id, run_id=req.run_id,
                                                      opened_at=now, picks=[p.prior_id for p in answer.picks]))
        # The act stays open: it closes on the run's score (_maybe_settle).
        return None, "plan sent; graded on its run"

    # --- reads -------------------------------------------------------------------------------
    async def _read_frame_inputs(self, req: CuriosityOfferPlanRequestV1, now: datetime) -> ReplicaFrameInputs:
        inp = ReplicaFrameInputs(now=now, request=req, notes=list(self.state.notes))
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

        def pg() -> dict[str, Any]:
            out: dict[str, Any] = {}
            for name, sql, params in (("outcomes", OUTCOMES_SQL, {"ids": run_ids}),
                                      ("record", RECENT_PLANS_SQL, {"n": RECORD_PLANS}),
                                      ("counts", FALLBACK_COUNTS_SQL, None),
                                      ("digest", DIGEST_SQL, None)):
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
        if "record" in rows:
            inp.record_rows = rows["record"]
        else:
            inp.record_error = rows.get("record_error")
        if "counts" in rows:
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
                inp.signal = {"signal_id": sig.signal_id, "direction": sig.direction, "why": sig.why,
                              "expires_at": sig.expires_at}
        except Exception as exc:  # noqa: BLE001
            inp.signal_error = _err(exc)
        return inp

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
                if not opened_before(plan, now, PLAN_ROW_GRACE_SEC):
                    keep.append(plan)
                    continue
                produced, note = False, "Hub recorded no plan row for this run; the run had the rule order"
            else:
                s = settle_plan(row)
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
