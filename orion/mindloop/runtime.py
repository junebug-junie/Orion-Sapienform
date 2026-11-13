"""The resident-loop runtime every instance shares (moved from services/orion-mind/app/prime_loop.py, 2026-11-04).

Design: docs/superpowers/specs/2026-11-04-curiosity-replica-design.md section 10; the behavior itself is Orion
Prime's (docs/superpowers/specs/2026-10-29-orion-prime-design.md sections 7.2 and 8). The move changed none of
it: a recorded tick replay (services/orion-mind/tests/test_prime_replay.py) gives the same acts, bids, model
calls, saved state and log lines as before.

One tick (every ``<prefix>TICK_SEC``):

1. ``_upkeep()`` (instance): the reads and CPU-only bookkeeping. Then the state is saved.
2. Juniper first: while a Juniper turn runs, no bid (``<name>_yield``).
3. ``_wants_to_think(now)`` (instance), then the self-monitor gate on the thought (``infer``).
4. The bid: a ``WantV1`` of the instance's family, tier and lane, priced at its cost ceiling (the
   generation budget alone). ``funded`` thinks. A ``capacity_reserved`` deferral with a positive net (the
   lane is full, but the thought is worth its price) also thinks: the GPU pool's own queue holds the call,
   bounded by the seat wait (the same rule as ``bind_verdict`` in orion/physiology/binding.py and
   ``goes_ahead`` in orion/physiology/prime_charge.py). Anything else waits, and no answer is "not funded":
   the loop never fails open.
5. ``think``: one model call (``_ask``, instance-built frame, via ``_ask_model``), seat wait and generation
   budget kept apart. A try that never reached the model is not a thought (``orion/mindloop/backoff.py``):
   no act, no refractory, a bounded backoff, and after a no-seat try the starved route. Otherwise the
   thought act opens stamped with the try's start, a call that overlapped a Juniper turn is ``preempted``,
   the answer is parsed (``_parse_answer``) and applied (``_conclude``), and the act closes at once.
6. Release the bid, ``_after_think()``, save.

What an instance supplies (subclass hooks; the ones marked * have a no-op default, the rest raise
NotImplementedError here):

* ``identity`` (``LoopIdentity``) and ``state_model``: a pydantic model with ``last_thought_at``,
  ``no_model_streak``, ``no_model_tries_total``, ``last_try_kind`` and ``retry_not_before``;
* ``_upkeep() -> (inputs, now)``; ``_wants_to_think(now)``; ``_pending_count()``; ``_tell(text)`` (a line in
  the next frame that is NOT a reason to think); ``_thought_act(thought_id, horizon_sec)``; ``_woke_note()``;
  ``_ask(inputs, now) -> ModelReply``; ``_parse_answer(parsed) -> (answer, error)``;
  ``_conclude(act, thought_id, inputs, now, reply, answer) -> (produced, note)`` (``produced`` None: the
  instance keeps the thought act open and closes it itself later); ``_after_think()``*;
  ``_log_started()``*; ``_new_thought_id()``*; ``_want_extras()``* (more ``WantV1`` fields, e.g. a request's
  ``object_ref``); ``_may_retry_5xx()``* (default True); ``_on_not_funded(decision)``* and ``_on_no_model(reply)``* (an instance that answers a
  requester can tell it why no thought happened). ``wake()`` ends the current tick wait early (a request
  arrived); Prime never calls it.

Everything here is best-effort: a failed read, publish or save is logged and the loop goes on. Every
blocking call runs in a worker thread.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, ClassVar, Optional
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.grammar.acts import act_closed_event, act_opened_event
from orion.grammar.publish import GRAMMAR_EVENT_CHANNEL
from orion.mindloop.backoff import TryKind, classify_try, note_model_reached, note_no_model, wants_starved_route
from orion.mindloop.gate import GATE_RECHECK_SEC, build_act_gate, gate_why
from orion.mindloop.instance import LoopIdentity
from orion.mindloop.turns import JUNIPER_TURNS, JuniperTurns
from orion.physiology.binding import consent_block
from orion.physiology.prime_charge import goes_ahead
from orion.schemas.grammar import ActV1
from orion.self_monitor.adapters import act_faculty
from orion.self_monitor.gate import FAILING_STATUSES, GateDecision
from orion.self_monitor.view import TRACK_RECORD_FAILING_REASON

# The no-model backoff's base never drops below this, even with <prefix>REFRACTORY_SEC=0 (no tick-rate loop).
NO_MODEL_BACKOFF_FLOOR_SEC = 60.0

_MISSING = object()
# Every instance's state model carries these (the runtime reads and writes them; orion/mindloop/backoff.py).
LOOP_STATE_FIELDS = ("last_thought_at", "no_model_streak", "no_model_tries_total", "last_try_kind", "retry_not_before")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ModelReply:
    """One model call's result, and whether it reached the model (orion/mindloop/backoff.py classify_try).
    ``context``: whatever the instance read for this frame and wants back in ``_conclude``."""

    route: str
    parsed: Optional[dict[str, Any]]
    err: Optional[str]
    gateway_error: Optional[dict[str, Any]]
    kind: TryKind
    pool_wait_sec: Optional[float] = None
    started_mono: float = 0.0
    context: dict[str, Any] = field(default_factory=dict)


class MindLoopRuntime:
    identity: ClassVar[LoopIdentity]
    state_model: ClassVar[type]

    def __init__(
        self,
        *,
        settings: Any,
        bus: Any,
        llm: Any,
        bid: Optional[Callable[[Any], Awaitable[Any]]] = None,
        release: Optional[Callable[[str], Awaitable[Any]]] = None,
        clock: Callable[[], datetime] = utcnow,
        turns: JuniperTurns = JUNIPER_TURNS,
    ) -> None:
        self.s = settings
        self.bus = bus
        self.llm = llm
        if bid is None or release is None:
            from orion.physiology.client import abid, arelease

            bid, release = bid or abid, release or arelease
        self.bid, self.release = bid, release
        self.clock = clock
        self.turns = turns
        missing = [f for f in LOOP_STATE_FIELDS if f not in getattr(self.state_model, "model_fields", {})]
        if missing:
            raise TypeError(f"{self.state_model.__name__} lacks the loop state fields {missing}")
        self.state = self.state_model()
        self.gate = build_act_gate(settings)
        self.log = logging.getLogger(self.identity.logger_name)
        # The offered infer probe, spent only when the model is reached (_open_thought).
        self._infer_probe: Optional[str] = None
        self._infer_refusal_told = False
        self._infer_refused_at: Optional[datetime] = None
        self._stop = asyncio.Event()
        self._wake = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self.source = ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION, node=settings.NODE_NAME)

    # --- settings --------------------------------------------------------------------------
    def cfg(self, name: str, default: Any = _MISSING) -> Any:
        """``<prefix><name>`` from settings, read at use time (so a changed setting applies at once)."""
        key = f"{self.identity.settings_prefix}{name}"
        return getattr(self.s, key) if default is _MISSING else getattr(self.s, key, default)

    # --- lifecycle -------------------------------------------------------------------------
    async def start(self, *, connect: bool = False) -> None:
        """Start the loop task. ``connect``: the task connects the bus first, retrying with backoff,
        so a bus blip at boot delays the loop instead of disabling it for the process lifetime."""
        self._task = asyncio.create_task(self._run(connect=connect), name=f"orion-{self.identity.name}")

    async def _connect(self) -> bool:
        delay = 1.0
        while not self._stop.is_set():
            try:
                await self.bus.connect()
                return True
            except Exception as exc:  # noqa: BLE001
                self.log.warning(f"{self.identity.name}_bus_connect_failed err=%s retry_in=%.0fs",
                                 type(exc).__name__, delay)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            delay = min(delay * 2, 60.0)
        return False

    def wake(self) -> None:
        """End the current wait between ticks now (call from the loop's own event loop)."""
        self._wake.set()

    async def _wait_tick(self) -> None:
        """Sleep TICK_SEC, or less if stopped or woken."""
        stop = asyncio.ensure_future(self._stop.wait())
        wake = asyncio.ensure_future(self._wake.wait())
        try:
            await asyncio.wait({stop, wake}, timeout=float(self.cfg("TICK_SEC")), return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in (stop, wake):
                t.cancel()
            self._wake.clear()

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    async def _run(self, *, connect: bool = False) -> None:
        if connect and not await self._connect():
            return
        await self.load()
        self._log_started()
        while not self._stop.is_set():
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                self.log.exception(f"{self.identity.name}_tick_failed")
            await self._wait_tick()

    # --- persistence -----------------------------------------------------------------------
    async def load(self) -> None:
        try:
            raw = await self.bus.redis.get(self.identity.state_key)
        except Exception as exc:  # noqa: BLE001
            self.log.warning(f"{self.identity.name}_state_load_failed err=%s (starting empty)", type(exc).__name__)
            return
        if not raw:
            return
        try:
            self.state = self.state_model.model_validate_json(raw)
        except Exception as exc:  # noqa: BLE001
            self.log.warning(f"{self.identity.name}_state_unreadable err=%s (starting empty)", str(exc)[:200])

    async def save(self) -> None:
        try:
            await self.bus.redis.set(self.identity.state_key, self.state.model_dump_json())
        except Exception as exc:  # noqa: BLE001
            self.log.warning(f"{self.identity.name}_state_save_failed err=%s", type(exc).__name__)

    # --- acts ------------------------------------------------------------------------------
    async def _publish(self, event: Any) -> bool:
        try:
            await self.bus.publish(GRAMMAR_EVENT_CHANNEL, BaseEnvelope(
                kind="grammar.event.v1", source=self.source, payload=event.model_dump(mode="json")))
            return True
        except Exception as exc:  # noqa: BLE001
            self.log.warning(f"{self.identity.name}_act_publish_failed act=%s err=%s",
                             event.act.act_id if event.act else "-", type(exc).__name__)
            return False

    async def _open(self, act: ActV1, component: str, *, at: Optional[datetime] = None) -> bool:
        return await self._publish(act_opened_event(act, source_service=self.identity.source_service,
                                                    source_component=component, at=at))

    async def _close(self, act: ActV1, component: str, *, produced: bool, note: str,
                     at: Optional[datetime] = None) -> bool:
        return await self._publish(act_closed_event(act, produced=produced, source_service=self.identity.source_service,
                                                    source_component=component, note=note, at=at))

    # --- self-monitor act gate (E1) ---------------------------------------------------------
    async def _gate_decide(self, verb: str, object_ref: str, *, job_key: Optional[str] = None,
                           recheck_soon: bool = False, actor: Optional[str] = None) -> Optional[GateDecision]:
        """The gate's answer, or None (gate off, or a gate fault: one direction only, nothing changes).
        Runs in a worker thread: the view read is Redis, the ledger read Postgres. ``recheck_soon``: this
        act was refused under GATE_RECHECK_SEC ago; while the fresh view still holds the faculty failing
        it stays refused without a ledger read. The view is always read, so an unknown, stale or healed
        view admits at once, exactly like the gate being off. ``actor``: the act's own actor when it is not
        the instance's (the curiosity replica's answers are actor ``curiosity.replica.answer``), so the gate
        reads that actor's track record. Default: the instance's actor (Prime never passes one)."""
        if not self.gate.enabled:
            return None
        now = self.clock()
        actor = actor or self.identity.actor

        def run() -> GateDecision:
            view = self.gate.begin(now)
            faculty = act_faculty(actor, verb, object_ref)
            if recheck_soon and view.known and view.status(faculty) in FAILING_STATUSES:
                return GateDecision(faculty, False, False, view.status(faculty), TRACK_RECORD_FAILING_REASON)
            return self.gate.decide(actor, verb, object_ref, now=now, job_key=job_key)

        try:
            return await asyncio.to_thread(run)
        except Exception:  # noqa: BLE001
            self.log.warning(f"{self.identity.name}_self_monitor_gate_failed verb=%s", verb, exc_info=True)
            return None

    def _admit(self, d: Optional[GateDecision], *, job_key: Optional[str] = None) -> bool:
        """Spend an offered probe now (the act is about to open). One loop per instance, so a claim only
        fails if something else already started this window's probe: then the act is refused."""
        if d is None or not d.admit:
            return d is None
        if d.probe:
            return self.gate.claim_probe(d.faculty, now=self.clock(), job_key=job_key)
        self.gate.note_job_started(d.faculty, job_key, now=self.clock())
        return True

    async def _thought_gate_admits(self, now: datetime) -> bool:
        """Before the bid. A probe is only offered here; it is spent when the model is reached. A refusal
        is re-checked against the ledger at most every GATE_RECHECK_SEC (the view every tick)."""
        self._infer_probe = None
        last = self._infer_refused_at
        d = await self._gate_decide(self.identity.verb, self.identity.thought_ref, recheck_soon=(
            last is not None and (now - last).total_seconds() < GATE_RECHECK_SEC))
        if d is None or (d.admit and not d.probe):
            self._infer_refusal_told = False
            self._infer_refused_at = None
            return True
        if d.probe:
            self._infer_probe = d.faculty
            self._infer_refused_at = None
            return True
        if last is None or (now - last).total_seconds() >= GATE_RECHECK_SEC:
            self._infer_refused_at = now
        self.log.info(f"{self.identity.name}_act_refused verb={self.identity.verb} gate=%s pending=%d",
                      d.label(), self._pending_count())
        if not self._infer_refusal_told:
            self._infer_refusal_told = True
            self._tell(f"at {now:%H:%M} UTC I did not start a thought: {gate_why(d, self.gate.probe_interval_sec)}")
        return False

    # --- the tick --------------------------------------------------------------------------
    async def tick(self) -> None:
        inputs, now = await self._upkeep()
        await self.save()
        name = self.identity.name
        if self.turns.active():
            self.log.info(f"{name}_yield reason=juniper_turn pending=%d", self._pending_count())
            return
        if not self._wants_to_think(now):
            return
        if not await self._thought_gate_admits(now):
            await self.save()
            return
        from orion.schemas.physiology_want import WantV1

        want = WantV1(want_id=f"{name}-think-{uuid4().hex[:12]}", family=self.identity.family,
                      producer=self.identity.producer, verb=self.identity.verb, tier=self.identity.tier,
                      lanes=[self.cfg("BID_LANE")], cost=self._cost_ceiling(), **self._want_extras())
        decision = await self.bid(want)
        outcome = getattr(decision, "outcome", None)
        if not goes_ahead(decision) or (decision is not None and consent_block(decision) is not None):
            # Never fail open: no answer is "not funded" (orion/physiology/client.py). A capacity deferral
            # worth its price is NOT a refusal: it goes ahead and the GPU pool's queue holds it (bind_verdict,
            # orion/physiology/binding.py). Live 10-30..11-03: 910 Prime thought bids were deferred
            # capacity_reserved on lane:chat with a positive net and treated as refusals here. A consent block
            # never goes ahead (bind_verdict's first rule), whatever the outcome says.
            self.log.info(f"{name}_bid_not_funded outcome=%s reason=%s binding=%s pending=%d",
                          outcome or "none", getattr(decision, "reason", None) or "-",
                          getattr(decision, "binding_constraint", None) or "-", self._pending_count())
            await self._on_not_funded(decision)
            return
        if outcome != "funded":
            # Queued at the pool, not funded: say so, with the market's own reason and net. The market made
            # no reservation for a deferral, so the release below is a no-op for it (kept on one path).
            self.log.info(f"{name}_bid_queued_at_pool outcome=%s reason=%s binding=%s net=%s pending=%d",
                          outcome, getattr(decision, "reason", None) or "-",
                          getattr(decision, "binding_constraint", None) or "-", getattr(decision, "net", None),
                          self._pending_count())
        try:
            await self.think(inputs)
        finally:
            await self.release(want.want_id)
            await self._after_think()
            await self.save()

    def _cost_ceiling(self) -> Any:
        """One thought's lane cost at its enforced maximum: the model call's timeout, on the loop's lane.
        (The history below is Prime's, the first instance; the rule holds for every instance.)

        Without it a family with no measured slot time is priced at a whole lane window
        (orion/physiology/choice.py "priced_at_one_window"). Live 10-30, every Prime bid was then
        deferred capacity_reserved on lane:chat (24/24), so Prime could never think and never build the
        history that would price it. The rule: charge the doer's enforced maximum while nothing is on
        record (the retired AI Town dispatch route's cold_cost_ceiling_sec did the same until 2026-11-05)."""
        from orion.schemas.grammar import ActCostV2

        ceiling = float(self.cfg("LLM_TIMEOUT_SEC"))
        key = f"{self.identity.settings_prefix}LLM_TIMEOUT_SEC"
        return ActCostV2(gpu_slot_s_by_lane={self.cfg("BID_LANE"): ceiling},
                         basis={"gpu_slot_s_by_lane": f"ceiling: {key} {ceiling:g} s per {self.identity.cost_unit}"})

    # --- the thought -----------------------------------------------------------------------
    def _route(self) -> str:
        if wants_starved_route(self.state):
            return str(self.cfg("STARVED_LLM_ROUTE", "") or self.cfg("LLM_ROUTE"))
        return self.cfg("LLM_ROUTE")

    def _seat_wait_sec(self) -> float:
        return float(self.cfg("SEAT_WAIT_SEC", 0.0) or 0.0)

    async def think(self, inputs: Any) -> None:
        now = self.clock()
        thought_id = self._new_thought_id()
        # The act's horizon covers the whole try: the seat wait plus the generation budget.
        act = self._thought_act(thought_id, self._seat_wait_sec() + float(self.cfg("LLM_TIMEOUT_SEC")))
        # Captured before anything can clear pending changes: every close of this act says why it woke.
        woke = self._woke_note()
        component = self.identity.thought_component
        try:
            reply = await self._ask(inputs, now)
        except Exception as exc:  # noqa: BLE001 -- a crash is a failed thought (refractory holds: no hot loop)
            self.log.exception(f"{self.identity.name}_thought_crashed thought=%s", thought_id)
            await self._open_thought(act, now)
            await self._close(act, component, produced=False,
                              note=f"{woke} | failed: {type(exc).__name__}: {str(exc)[:160]}")
            return
        if reply.kind != "model":
            self._note_no_model(reply, self.clock())   # the backoff counts from the end of the try
            await self._on_no_model(reply)
            return
        await self._open_thought(act, now)
        try:
            produced, note = await self._settle_reply(act, thought_id, inputs, now, reply)
        except Exception as exc:  # noqa: BLE001 -- the act always closes
            self.log.exception(f"{self.identity.name}_thought_crashed thought=%s", thought_id)
            produced, note = False, f"failed: {type(exc).__name__}: {str(exc)[:160]}"
        if produced is None:
            # The instance holds the act open and closes it itself later (the curiosity replica grades a
            # plan on the run it ordered). Prime's _conclude never returns None.
            return
        await self._close(act, component, produced=produced, note=f"{woke} | {note}")

    async def _settle_reply(self, act: ActV1, thought_id: str, inputs: Any, now: datetime,
                            reply: ModelReply) -> tuple[bool, str]:
        """Juniper first, then the instance's parse and apply."""
        name = self.identity.name
        if self.turns.started_since(reply.started_mono):
            self.log.info(f"{name}_thought_preempted thought=%s", thought_id)
            return False, "preempted: a Juniper turn started"
        answer, perr = (None, reply.err) if reply.err else self._parse_answer(reply.parsed)
        if answer is None:
            self.log.info(f"{name}_thought_failed thought=%s err=%s", thought_id, perr)
            return False, f"failed: {perr}"
        return await self._conclude(act, thought_id, inputs, now, reply, answer)

    async def _open_thought(self, act: ActV1, started_at: datetime) -> None:
        """The model was reached: the thought happened. Opened stamped with the try's start; counts for
        the refractory interval from then (a failed or preempted thought must not retry every tick)."""
        if self._infer_probe is not None:
            # The model was reached: the offered probe is a real attempt now. One loop per instance, so
            # the claim cannot lose a race; the thought already happened either way.
            self.gate.claim_probe(self._infer_probe, now=self.clock())
            self._infer_probe = None
        await self._open(act, self.identity.thought_component, at=started_at)
        self.state = note_model_reached(self.state).model_copy(update={"last_thought_at": started_at})

    def _note_no_model(self, reply: ModelReply, now: datetime) -> None:
        gw = reply.gateway_error or {}
        first = self.state.no_model_streak == 0
        self.state = note_no_model(self.state, reply.kind, now,
                                   base=timedelta(seconds=max(float(self.cfg("REFRACTORY_SEC")),
                                                              NO_MODEL_BACKOFF_FLOOR_SEC)),
                                   retry_after_sec=gw.get("retry_after_sec"))
        if first:
            # Once per streak, so the next real thought knows its last try was starved (not a change:
            # it must not wake the loop by itself).
            self._tell(f"a try to think at {now:%H:%M} UTC never reached the model ({reply.kind}"
                       f"{': ' + str(gw.get('reason')) if gw.get('reason') else ''}); I tried again")
        self.log.info(f"{self.identity.name}_try_no_model kind=%s route=%s reason=%s err=%s streak=%d "
                      "retry_not_before=%s pending=%d",
                      reply.kind, reply.route, gw.get("reason") or "-", (reply.err or "-")[:160],
                      self.state.no_model_streak,
                      self.state.retry_not_before.isoformat() if self.state.retry_not_before else "-",
                      self._pending_count())

    async def _ask_model(self, system: str, frame: str) -> ModelReply:
        """Call the model on the current route; one immediate retry on an upstream 5xx."""
        route = self._route()
        started = time.monotonic()
        reply = await self._call(route, system, frame)
        if reply.kind == "upstream_5xx" and self._may_retry_5xx():
            self.log.info(f"{self.identity.name}_upstream_5xx_retry route=%s err=%s", route, (reply.err or "-")[:160])
            started = time.monotonic()
            reply = await self._call(route, system, frame)
        # Juniper-first preemption counts from the GRANT: a Juniper turn during the seat wait went ahead
        # of the loop in the pool and is over or running beside it, not overlapped by generation. When the
        # gateway did not report its wait, count from the start (the conservative old rule).
        reply.started_mono = started + (reply.pool_wait_sec or 0.0)
        return reply

    async def _call(self, route: str, system: str, frame: str) -> ModelReply:
        seat = self._seat_wait_sec()
        try:
            parsed, err, meta = await asyncio.to_thread(
                self.llm.request_json,
                system_prompt=system, user_prompt=frame, route=route,
                max_tokens=int(self.cfg("MAX_TOKENS")), temperature=self.identity.temperature,
                thinking=self.identity.thinking,
                timeout_sec=float(self.cfg("LLM_TIMEOUT_SEC")), pool_wait_sec=seat if seat > 0 else None,
            )
        except Exception as exc:  # noqa: BLE001
            parsed, err, meta = None, f"{type(exc).__name__}: {str(exc)[:160]}", {}
        gw = (meta or {}).get("gateway_error") if isinstance(meta, dict) else None
        gw = gw if isinstance(gw, dict) else None
        kind = "model" if parsed is not None and not err else classify_try(err, gw)
        wait_ms = (meta or {}).get("pool_wait_ms") if isinstance(meta, dict) else None
        return ModelReply(route=route, parsed=parsed, err=err, gateway_error=gw, kind=kind,
                          pool_wait_sec=float(wait_ms) / 1000.0 if isinstance(wait_ms, (int, float)) else None)

    # --- instance hooks --------------------------------------------------------------------
    def _new_thought_id(self) -> str:
        return uuid4().hex[:16]

    async def _upkeep(self) -> tuple[Any, datetime]:
        raise NotImplementedError

    def _wants_to_think(self, now: datetime) -> bool:
        raise NotImplementedError

    def _pending_count(self) -> int:
        raise NotImplementedError

    def _tell(self, text: str) -> None:
        raise NotImplementedError

    def _thought_act(self, thought_id: str, horizon_sec: float) -> ActV1:
        raise NotImplementedError

    def _woke_note(self) -> str:
        raise NotImplementedError

    async def _ask(self, inputs: Any, now: datetime) -> ModelReply:
        raise NotImplementedError

    def _parse_answer(self, parsed: Optional[dict[str, Any]]) -> tuple[Any, Optional[str]]:
        raise NotImplementedError

    async def _conclude(self, act: ActV1, thought_id: str, inputs: Any, now: datetime, reply: ModelReply,
                        answer: Any) -> tuple[Optional[bool], str]:
        raise NotImplementedError

    async def _after_think(self) -> None:
        return None

    def _want_extras(self) -> dict[str, Any]:
        return {}

    def _may_retry_5xx(self) -> bool:
        """Whether the one immediate retry after an upstream 5xx may run (an instance with a deadline says no
        when a second full call would end past it). Prime: always."""
        return True

    async def _on_not_funded(self, decision: Any) -> None:
        return None

    async def _on_no_model(self, reply: ModelReply) -> None:
        return None

    def _log_started(self) -> None:
        self.log.info(f"{self.identity.name}_started tick_sec=%s route=%s", self.cfg("TICK_SEC"), self.cfg("LLM_ROUTE"))


__all__ = ["LOOP_STATE_FIELDS", "NO_MODEL_BACKOFF_FLOOR_SEC", "MindLoopRuntime", "ModelReply", "utcnow"]
