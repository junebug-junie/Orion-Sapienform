"""Orion Prime's resident loop (rollout step 2A): orion-mind keeps thinking between conversations.

Design: docs/superpowers/specs/2026-10-29-orion-prime-design.md, sections 7.2 and 8. Pure logic is in
orion/prime/; this module does Prime's I/O. Since 2026-11-04 (curiosity replica design, section 10) Prime is
instance #1 of the loop library ``orion/mindloop/``: the tick, the Juniper-first yield, the bid, the model call
with its seat wait, no-model backoff and starved route, the thought act, act publishing and the state in Redis
are ``orion/mindloop/runtime.py``; this module keeps what only Prime does (its reads, concerns, frame, answer,
moves and how they settle). Log line names (``prime_thought``, ``prime_try_no_model``, ``prime_started`` ...),
the actor, act ids, family, Redis key and env keys did not change.

Two clocks:

* **tick** (every PRIME_TICK_SEC, CPU only): read the body (durable-runs GET /physiology/state) and
  attention (GET /temporal-self/frame; with PRIME_CONCERNS_FROM_EVENT_WANTS, the funded event wants
  instead, GET /physiology/event-wants, Prime's only concern source then), record what changed
  (orion.prime.state.observe), settle any
  open question, save the state to Redis. Most ticks stop here.
* **thought** (only when something changed, the refractory interval has passed, no Juniper turn is
  running, and the body economy FUNDS the bid): one model call on route PRIME_LLM_ROUTE (circe chat
  lane 0 at pool priority system), then at most one move.

Seat wait and generation are two numbers (2026-11-03). The call may wait PRIME_SEAT_WAIT_SEC for a pool
seat and then generate for PRIME_LLM_TIMEOUT_SEC (the gateway's ``gateway_pool_wait_sec`` option); the
bid's cost ceiling is the generation alone. A try that never reached the model (no seat, no gateway
reply, or an upstream 5xx after one immediate retry) is not a thought: no act is opened, the refractory
interval does not start, and the next tick re-bids (orion/prime/state.py ``note_no_model``, with a
bounded backoff). After a no-seat try the next one goes on PRIME_STARVED_LLM_ROUTE (pool priority
``starved``: ahead of other system work, still behind Juniper's turn and shed by heat).

The thought act is opened only once the model answered (or failed in a way that is the model's own:
empty, unparseable, out of generation time), stamped with the time the try started, and closed at once.
That keeps the act ledger honest: a ``prime`` infer act is a thought that happened, so queue failures
neither count as failed thoughts nor reset the market's hunger for the ``prime`` family.

Prime obeys its own bid: ``funded`` thinks, and so does a ``capacity_reserved`` deferral with a positive net
(the lane is full but the thought is worth its price: it queues at the GPU pool, orion/mindloop/runtime.py).
Anything else (another deferral, no answer, flag off) waits. It is
the first producer whose decision binds (every other bid is still shadow); it never fails open.

The move (2A): one question for curiosity's next regular investigate run, queued through Hub's
existing focus slot (orion/curiosity/focus.py) with binding.invocation_context "prime". A share of
questions (PRIME_HOLDBACK_FRACTION) is the narrator arm: opened as an act, never sent.

The second move (step 3, 2026-10-30): a regulate signal ("slow" / "quicken") to one want family that
declares the receptor in config/physiology/families.yaml (today curiosity investigate and the dream cycle). Written to
Redis ``orion:prime:regulate:<family>`` (orion/physiology/prime_charge.py), where the body-economy market
(orion-durable-runs) applies it as a bounded, temporary charge on that family's bids and records what it
did to each bid. Each signal is a ``regulate`` act; after the signal ends the runtime reads the market's
record and closes the act (1 = the charge changed at least one decision). A share is held back
(PRIME_HOLDBACK_FRACTION): written as ``control``, never applied, graded the same way. PRIME_REGULATE_ENABLED
false: the move is not offered and any regulate answer is refused (signals already sent are still graded).

Organs answer (curiosity replica patch 4, docs/superpowers/specs/2026-11-04-curiosity-replica-design.md section 9).
Curiosity's planner (app/replica_loop.py, same process, separate state) may challenge or refuse a question or
signal of Prime's. Prime reads only the replica's published replies view (Redis ``orion:curiosity:replica:replies``):
in its frame (ORGAN REPLIES, while answering is on) and when it settles -- a treated question the organ refused
closes 0 at once with ``refused by curiosity.replica: <reason>`` (still the treated arm: intention to treat); a
regulate act's note names a refusal of its signal (the market's record already shows the charge did nothing).

Juniper first: a Juniper turn through run_mind (engine.carries_live_turn) is tracked by JUNIPER_TURNS.
Today ``mind_run`` calls the synchronous run_mind on the event loop, so no tick runs during a turn at
all; ``JUNIPER_TURNS.active()`` keeps that true if run_mind ever moves off the loop. What does the
work today: a thought whose model call (in a worker thread) overlapped a turn is recorded ``preempted``
and discarded, and the pool ranks Juniper's turn (interactive) ahead of both Prime routes (system,
starved), so a Prime call does not START while Juniper's call waits for the slot. A call already
generating is NOT cancelled (the bus RPC has no cancel); the gateway generates nothing for a call
granted after its seat wait (llm-gateway app/main.py).

Self-monitor act gate (E1, 2026-11-03; orion/self_monitor/gate.py): Prime opens its acts itself, so
the gate sits here, on the same faculty names the organ grades (``act.prime.<verb>[.<name>]``) and the
same rules: refuse-only, an unknown/stale/absent state or a gate fault changes nothing, one re-test
probe per faculty per SELF_MONITOR_PROBE_INTERVAL_SEC, spent only when the act really opens. Where
each verb is gated, and what a refusal means:

* ``predict`` (a concern bet): before the concern's act opens. A refusal places no bet; the concern
  itself stays in Prime's state, keeps its TTL, can be kept, dropped or settled by a question, and
  shows in the track record as before. Prime is told once per concern (a frame line, not a change).
  When a concern whose act never opened ends, a late open+close is placed only if the gate admits it
  as an ordinary act at that moment; while the faculty is failing it is not (not even as the probe:
  its outcome is known, so it would test nothing) and the record is marked sent with no act. The
  probe is spent only after the open was published.
* ``ask`` / ``regulate`` (a move): before the holdback arm is drawn, so a refusal removes the move
  from both arms alike and the treated/control comparison stays fair. Shown as "my move was not sent".
* ``infer`` (the thought): before the bid. A refusal is no bid and no model call; the probe is spent
  only when the model was reached (the thought act opens), matching the act's own rule.

Every turn ends back in Prime (living Prime design Part 2, PRIME_TURN_HANDBACK_ENABLED; orion/prime/handback.py):
a listener on the bus channel orion:cockpit:hop (the hops Hub already publishes for every unified turn) cuts each
ingress and finalize hop to a small signal and queues it in memory; the tick drains the queue and folds it into
Prime's state deterministically. A turn that made a promise opens a concern (event_ref ``turn:<correlation_id>``)
within the loop-guard bounds; a turn whose prompt cites an open ``prime_concern:<id>`` and replied settles it.
Only those two wake a thought; every finished turn shows in the next frame. The hand-back record lives in its own
Redis key (orion:prime:turns). Pub/sub is not durable: a turn that finishes while orion-mind is down is not handed
back. Flag off: no listener, no fold, no frame section.

The day letter is Prime's end-of-day turn (living Prime design Part 3, PRIME_DAY_LETTER_ENABLED;
orion/prime/day_letter.py): on the tick, when the temporal chronicle has closed yesterday and Hub has no letter for
it, Prime opens a ``make`` act at once. Tier 0 (Juniper 2026-11-12, "It needs to happen each day"): no bid, no
price, no self-monitor gate. Prime orders the run from Hub (one durable unified turn). A run that wrote nothing is
ordered again as the next attempt (PRIME_DAY_LETTER_MAX_ATTEMPTS) while the first order's deadline holds; all
attempts failed -> no letter that day, and the next order names the day. The run's terminal state
(GET /runs/<run_id> on durable-runs) closes the act, and the letter's threads become Prime concerns (at most 2,
never past MAX_CONCERNS), so curiosity hears them only through Prime's own graded ``ask``. Its record lives in its
own Redis key (orion:prime:day_letter). No thought is spent on it: the tick decides.

Everything here is best-effort: a failed read, publish or save is logged and the loop goes on.
``run_mind`` never waits on this loop: every blocking call runs in a worker thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Optional
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.introspect.transport import (
    QUEUE_CURIOSITY_FOCUS_REQUEST_CHANNEL,
    REQUEST_KIND,
    RESULT_KIND,
    RESULT_PREFIX,
)
from orion.mindloop.gate import GATE_RECHECK_SEC, build_act_gate, gate_why
from orion.mindloop.instance import LoopIdentity
from orion.mindloop.runtime import MindLoopRuntime, ModelReply, utcnow
from orion.mindloop.turns import JUNIPER_TURNS, JuniperTurns
from orion.physiology import prime_charge as pc
from orion.prime.acts import (
    ACTOR,
    organ_refusal,
    refusal_words,
    CONCERN_COMPONENT,
    REGULATE_COMPONENT,
    regulate_act,
    regulation_hours,
    settle_regulation,
    GONE_READS_TO_SETTLE,
    NOMINATION_COMPONENT,
    NOMINATION_HORIZON_SEC,
    SOURCE_SERVICE,
    THOUGHT_COMPONENT,
    chat_correlation_id,
    concern_act,
    draw_arm,
    lapse_concerns,
    nomination_act,
    prime_focus_id,
    Settlement,
    settle_concern,
    settle_nomination,
    thought_act,
    treated_focus_gone,
)
from orion.prime import day_letter as dl
from orion.prime.handback import (
    HANDBACK_CHANNEL,
    TURNS_KEY,
    PrimeTurnsStateV1,
    TurnSignalV1,
    fold_turn_signals,
    link_started_turn,
    signal_from_hop,
    turn_lines,
)
from orion.prime.state import (
    MAX_RECENT_OUTCOMES,
    PRIME_CONCERNS_KEY,
    PrimeNominationV1,
    PrimeRegulationOpenV1,
    PrimeSnapshotV1,
    PrimeStateV1,
    new_id,
    note_outcome,
    observe,
    concerns_view,
    trim_record,
    wants_to_think,
)
from orion.prime.thought import (
    FrameInputs,
    RegulateMoveV1,
    apply_answer,
    build_frame,
    event_ids,
    parse_answer,
    shows_organ_replies,
    summary_note,
    woke_note,
    system_prompt,
)
from orion.refs import RefV1, resolve as resolve_ref
from orion.schemas.cockpit_sighting import CockpitHopV1
from orion.schemas.curiosity_replica import REPLIES_KEY, CuriosityReplicaRepliesViewV1
from orion.schemas.grammar import ActV1
from orion.schemas.physiology_want import FundedEventWantsV1
from orion.schemas.introspect import (
    IntrospectRequestV1,
    IntrospectResultV1,
    IntrospectToolBindingV1,
    QueueCuriosityFocusArguments,
)
from orion.schemas.orion_day import (
    ORION_DAY_ORDER_PATH,
    ORION_DAY_STATUS_PATH,
    OrionDayLetterOrderResultV1,
    OrionDayLetterOrderV1,
    OrionDayLetterStatusV1,
)
from orion.self_monitor.gate import GateDecision

logger = logging.getLogger("orion-mind.prime")

STATE_KEY = "orion:prime:state"
FAMILY = "prime"
PRODUCER = "orion-mind.prime"
HTTP_TIMEOUT_SEC = 4.0
# Hub gathers the day before it answers an order (read-only SQL, seconds); a timed-out order is re-sent and Hub
# answers ``exists`` for a run it already submitted.
ORDER_HTTP_TIMEOUT_SEC = 45.0
FOCUS_RPC_TIMEOUT_SEC = 10.0
CURIOSITY_RUNS_DAYS = 4  # covers a nomination's whole horizon (48 h focus TTL + 24 h run)
# A regulate act is graded this long after its signal ends: the market's last bid under it has landed.
REGULATE_SETTLE_LAG_SEC = 120.0
# Hand-back signals queued between ticks (a tick drains them). Hops of sessions that are not handed back never
# reach it (signal_from_hop); world-pulse finalize hops do, ~300 a day, so a 15 s tick never comes near this.
TURN_INBOX_MAX = 500
LISTEN_BACKOFF_CAP_SEC = 10.0

# Prime on the bus, in the body economy, the ledger, Redis and the logs (orion/mindloop/instance.py).
PRIME_IDENTITY = LoopIdentity(
    name="prime", family=FAMILY, actor=ACTOR, producer=PRODUCER, state_key=STATE_KEY, settings_prefix="PRIME_",
    source_service=SOURCE_SERVICE, thought_component=THOUGHT_COMPONENT, thought_ref="prime_thought:", tier=1,
    logger_name=logger.name, verb="infer", temperature=0.4, thinking=False, cost_unit="thought",
)

__all__ = ["JUNIPER_TURNS", "JuniperTurns", "PRIME_IDENTITY", "PrimeRuntime", "STATE_KEY", "build_act_gate"]


def _get_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_SEC) as resp:  # noqa: S310 -- internal service URL
        return json.loads(resp.read())


def _post_json(url: str, body: dict[str, Any]) -> Any:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=ORDER_HTTP_TIMEOUT_SEC) as resp:  # noqa: S310 -- internal service URL
        return json.loads(resp.read())


def _with_outcome_line(state: PrimeStateV1, text: str) -> PrimeStateV1:
    """Show a line in the next frame WITHOUT making it a change (a refusal Prime caused itself must
    not wake it again, or a refused move would loop)."""
    return state.model_copy(update={"recent_outcomes": [*state.recent_outcomes, text][-MAX_RECENT_OUTCOMES:]})


class PrimeRuntime(MindLoopRuntime):
    identity = PRIME_IDENTITY
    state_model = PrimeStateV1

    def __init__(
        self,
        *,
        settings: Any,
        bus: Any,
        llm: Any,
        get_json: Callable[[str], Any] = _get_json,
        post_json: Callable[[str, dict[str, Any]], Any] = _post_json,
        bid: Optional[Callable[[Any], Awaitable[Any]]] = None,
        release: Optional[Callable[[str], Awaitable[Any]]] = None,
        clock: Callable[[], datetime] = utcnow,
        turns: JuniperTurns = JUNIPER_TURNS,
    ) -> None:
        super().__init__(settings=settings, bus=bus, llm=llm, bid=bid, release=release, clock=clock, turns=turns)
        self.state: PrimeStateV1
        self.get_json = get_json
        self.post_json = post_json
        # The day letter (Part 3): its own record, saved under dl.DAY_LETTER_KEY.
        self.day_letter = dl.PrimeDayLetterStateV1()
        self._day_letter_checked_at: Optional[datetime] = None
        # concern_id -> when its bet was last refused (re-check throttle; also: never a late bet for it).
        self._bet_refused_at: dict[str, datetime] = {}
        self._last_settle: Optional[datetime] = None
        # Turn hand-back (Part 2): the listener appends, the tick drains (one event loop, no lock needed).
        self.turn_record = PrimeTurnsStateV1()
        self._turn_inbox: deque[TurnSignalV1] = deque(maxlen=TURN_INBOX_MAX)
        self._turn_hops_dropped = 0
        self._listener: Optional[asyncio.Task] = None

    # --- turn hand-back (living Prime design, Part 2) ----------------------------------------
    def _handback_on(self) -> bool:
        return bool(getattr(self.s, "PRIME_TURN_HANDBACK_ENABLED", False))

    async def start(self, *, connect: bool = False) -> None:
        await super().start(connect=connect)
        if self._handback_on():
            self._listener = asyncio.create_task(self._listen_turns(), name="orion-prime-turn-handback")

    async def stop(self) -> None:
        listening = self._listener is not None
        if self._listener is not None:
            self._listener.cancel()
            try:
                await self._listener
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._listener = None
        await super().stop()
        if listening and self._turn_inbox:
            # Hops heard since the last tick would otherwise be lost on every deploy: fold and save them now
            # (the loop task is cancelled above, so nothing else touches the state).
            try:
                self._fold_turns(self.clock())
                await self.save()
            except Exception as exc:  # noqa: BLE001
                logger.warning("prime_handback_final_fold_failed err=%s", type(exc).__name__)

    async def _listen_turns(self) -> None:
        """Subscribe to the cockpit hops; resubscribe with backoff on any failure (a silent socket included:
        iter_messages raises on it)."""
        delay = 1.0
        while not self._stop.is_set():
            try:
                async with self.bus.subscribe(HANDBACK_CHANNEL) as pubsub:
                    logger.info("prime_handback_subscribed channel=%s", HANDBACK_CHANNEL)
                    delay = 1.0
                    async for msg in self.bus.iter_messages(pubsub):
                        self.accept_hop(msg.get("data"))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning("prime_handback_listen_failed err=%s retry_in=%.0fs", type(exc).__name__, delay)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            # Capped low: a hop is only heard while subscribed, so a long backoff is lost turns.
            delay = min(delay * 2, LISTEN_BACKOFF_CAP_SEC)

    def accept_hop(self, data: Any) -> None:
        """One bus message: decode, validate as CockpitHopV1 (the channel's registered schema), cut to a signal."""
        try:
            decoded = self.bus.codec.decode(data)
            if not decoded.ok:
                raise ValueError("undecodable")
            hop = CockpitHopV1.model_validate(decoded.envelope.payload)
            signal = signal_from_hop(hop.model_dump(mode="json"))
        except Exception as exc:  # noqa: BLE001
            self._turn_hops_dropped += 1
            if self._turn_hops_dropped in (1, 10, 100) or self._turn_hops_dropped % 1000 == 0:
                logger.warning("prime_handback_hop_dropped total=%d err=%s", self._turn_hops_dropped,
                               str(exc)[:160])
            return
        if signal is not None:
            if len(self._turn_inbox) == self._turn_inbox.maxlen:
                # The oldest queued hop is pushed out (counted, never silent).
                self._turn_hops_dropped += 1
                logger.warning("prime_handback_inbox_full dropped_total=%d", self._turn_hops_dropped)
            self._turn_inbox.append(signal)

    def _fold_turns(self, now: datetime) -> None:
        if not self._turn_inbox:
            return
        signals = list(self._turn_inbox)
        self._turn_inbox.clear()
        self.state, self.turn_record, folded = fold_turn_signals(self.state, self.turn_record, signals, now)
        for t in folded:
            logger.info("prime_turn_handback corr=%s who=%s verdict=%s promises=%d opened=%s refused=%s settled=%s",
                        t.correlation_id, t.who, t.verdict, len(t.promise_refs), t.opened_concern_id or "-",
                        t.thread_refused or "-", ",".join(t.settled_concern_ids) or "-")

    async def load(self) -> None:
        await super().load()
        if self._day_letter_on():
            try:
                raw = await self.bus.redis.get(dl.DAY_LETTER_KEY)
                if raw:
                    self.day_letter = dl.PrimeDayLetterStateV1.model_validate_json(raw)
            except Exception as exc:  # noqa: BLE001
                logger.warning("prime_day_letter_load_failed err=%s (starting empty)", str(exc)[:200])
        if not self._handback_on():
            return
        try:
            raw = await self.bus.redis.get(TURNS_KEY)
            if raw:
                self.turn_record = PrimeTurnsStateV1.model_validate_json(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("prime_turns_load_failed err=%s (starting empty)", str(exc)[:200])

    # --- persistence: Prime's first published call ------------------------------------------
    async def save(self) -> None:
        """Save the state, then publish the read-only concerns view (curiosity replica design section 2):
        ids, words, age and arc of the open concerns, for another loop instance to read without touching
        Prime's state. Written on every save, so its ``as_of`` doubles as Prime's liveness: a reader that
        finds it old knows Prime stopped."""
        await super().save()
        if self._handback_on():
            try:
                await self.bus.redis.set(TURNS_KEY, self.turn_record.model_dump_json())
            except Exception as exc:  # noqa: BLE001
                logger.warning("prime_turns_save_failed err=%s", type(exc).__name__)
        if self._day_letter_on():
            try:
                await self.bus.redis.set(dl.DAY_LETTER_KEY, self.day_letter.model_dump_json())
            except Exception as exc:  # noqa: BLE001
                logger.warning("prime_day_letter_save_failed err=%s", type(exc).__name__)
        try:
            await self.bus.redis.set(PRIME_CONCERNS_KEY, concerns_view(self.state, now=self.clock()).model_dump_json())
        except Exception as exc:  # noqa: BLE001
            logger.warning("prime_concerns_view_publish_failed err=%s", type(exc).__name__)

    # --- loop hooks (orion/mindloop/runtime.py) ----------------------------------------------
    def _log_started(self) -> None:
        logger.info("prime_started tick_sec=%s route=%s holdback=%s concerns=%d open_questions=%d",
                    self.s.PRIME_TICK_SEC, self.s.PRIME_LLM_ROUTE, self.s.PRIME_HOLDBACK_FRACTION,
                    len(self.state.concerns), len(self.state.open_nominations))

    async def _upkeep(self) -> tuple[FrameInputs, datetime]:
        """Read the body and attention, record what changed, settle what is due, publish concern acts."""
        snap, inputs = await self._read_body_and_attention()
        now = self.clock()
        self.state = lapse_concerns(observe(self.state, snap, now), now)
        if self._handback_on():
            self._fold_turns(now)
            inputs.turns = turn_lines(self.turn_record, now)
        if self._day_letter_on():
            await self._day_letter_upkeep(now)
        await self._maybe_settle(now)
        await self._settle_regulations(now)
        await self._flush_concern_acts()
        return inputs, now

    def _wants_to_think(self, now: datetime) -> bool:
        return wants_to_think(self.state, now, refractory=timedelta(seconds=float(self.s.PRIME_REFRACTORY_SEC)))

    def _pending_count(self) -> int:
        return len(self.state.pending_changes)

    def _tell(self, text: str) -> None:
        self.state = _with_outcome_line(self.state, text)

    def _thought_act(self, thought_id: str, horizon_sec: float) -> ActV1:
        return thought_act(thought_id, llm_timeout_sec=horizon_sec)

    def _woke_note(self) -> str:
        return woke_note(list(self.state.pending_changes))

    def _new_thought_id(self) -> str:
        return new_id()

    def _parse_answer(self, parsed: Optional[dict[str, Any]]) -> tuple[Any, Optional[str]]:
        return parse_answer(parsed)

    async def _after_think(self) -> None:
        await self._flush_concern_acts()

    # --- the day letter: Prime's end-of-day turn (living Prime design, Part 3) -----------------
    def _day_letter_on(self) -> bool:
        return bool(getattr(self.s, "PRIME_DAY_LETTER_ENABLED", False))

    def _letter_cfg(self, name: str, default: float) -> float:
        return float(getattr(self.s, f"PRIME_DAY_LETTER_{name}", default))

    async def _post(self, url: str, body: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        try:
            data = await asyncio.to_thread(self.post_json, url, body)
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {str(exc)[:160]}"
        if not isinstance(data, dict):
            return None, "not a JSON object"
        return data, None

    def _letter_check_due(self, now: datetime) -> bool:
        last = self._day_letter_checked_at
        if last is not None and (now - last).total_seconds() < dl.CHECK_INTERVAL_SEC:
            return False
        self._day_letter_checked_at = now
        return True

    async def _day_letter_upkeep(self, now: datetime) -> None:
        """One step of the end-of-day move per tick at most. Never raises into the tick."""
        try:
            await self._day_letter_step(now)
        except Exception:  # noqa: BLE001
            logger.warning("prime_day_letter_step_failed", exc_info=True)

    async def _day_letter_step(self, now: datetime) -> None:
        op = self.day_letter.open
        if op is not None and op.act_id is not None:
            # Writing: the order was not answered yet (send it again), or the run is out (read it).
            if not self._letter_check_due(now):
                return
            if op.order_status in (None, "unavailable"):
                if op.deadline_at is not None and now >= op.deadline_at:
                    # The first order may have landed with its answer lost: a run that exists is followed.
                    run, _err = await self._read(f"{self.s.PRIME_DURABLE_RUNS_URL.rstrip('/')}/runs/{op.run_id}")
                    if run is not None:
                        self.day_letter = self.day_letter.model_copy(update={"open": op.model_copy(update={
                            "order_status": "exists"})})
                        await self._settle_letter(now)
                        return
                    await self._close_letter(now, produced=False, outcome="not_written",
                                             note=f"Hub never took the order before the run's deadline "
                                                  f"({op.order_detail or 'no answer'})")
                    return
                await self._order_letter(now)
                return
            await self._settle_letter(now)
            return
        if op is not None:
            # The act's open failed last time: try again (no bid; tier 0).
            if self._letter_check_due(now):
                await self._open_letter(now)
            return
        if not self._letter_check_due(now):
            return
        day = dl.yesterday(now)
        if dl.handled(self.day_letter, day):
            return
        closed, _err = await self._read(f"{self.s.PRIME_DURABLE_RUNS_URL.rstrip('/')}/temporal-self/day/{day.isoformat()}")
        if closed is None:
            # Not drained by the chronicle yet (404), or unreadable: look again later. A day the chronicle never
            # closes gets no letter; Hub names it in the next letter's order (the day before with no row).
            logger.info("prime_day_letter_day_not_closed day=%s err=%s", day, (_err or "-")[:160])
            return
        status_raw, err = await self._read(
            f"{self.s.PRIME_HUB_URL.rstrip('/')}{ORION_DAY_STATUS_PATH.format(letter_date=day.isoformat())}")
        if status_raw is None:
            logger.info("prime_day_letter_status_unread day=%s err=%s", day, err)
            return
        try:
            status = OrionDayLetterStatusV1.model_validate(status_raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("prime_day_letter_status_invalid day=%s err=%s", day, str(exc)[:160])
            return
        if status.letter_exists:
            self.day_letter = dl.finish(self.day_letter, dl.PrimeDayLetterDoneV1(
                letter_date=day, outcome="already_written", reason=f"run {status.run_id} wrote it", closed_at=now,
                run_id=status.run_id))
            logger.info("prime_day_letter_already_written day=%s run=%s", day, status.run_id)
            return
        self.day_letter = dl.start(self.day_letter, day, now)
        self._tell(f"yesterday ({day}) is closed and has no letter yet: I am writing it")
        logger.info("prime_day_letter_due day=%s", day)
        await self._open_letter(now)

    async def _open_letter(self, now: datetime) -> None:
        """Tier 0 (Juniper 2026-11-12: "It needs to happen each day"): no bid, no price, no self-monitor gate.
        Open the make act and order the first run. Live 11-11: a tier-2 bid was deferred for 10 h."""
        op = self.day_letter.open
        day = op.letter_date
        deadline_h = self._letter_cfg("RUN_DEADLINE_HOURS", 12.0)
        act = dl.make_act(day, horizon_sec=deadline_h * 3600.0 + dl.ACT_HORIZON_GRACE_SEC)
        if not await self._open(act, dl.DAY_LETTER_COMPONENT, at=now):
            logger.info("prime_day_letter_act_open_failed day=%s (tried again next check)", day)
            return
        self.day_letter = self.day_letter.model_copy(update={"open": op.model_copy(update={
            "act_id": act.act_id, "act_opened_at": now, "attempt": 1,
            "run_id": dl.letter_run_id(day, 1), "correlation_id": dl.letter_correlation_id(day, 1),
            "deadline_at": now + timedelta(hours=deadline_h)})})
        logger.info("prime_day_letter_opened day=%s act=%s", day, act.act_id)
        self._day_letter_checked_at = now
        await self._order_letter(now)

    async def _order_letter(self, now: datetime) -> None:
        op = self.day_letter.open
        missed = dl.missed_for(self.day_letter, op.letter_date)
        order = OrionDayLetterOrderV1(
            letter_date=op.letter_date, run_id=op.run_id, correlation_id=op.correlation_id, prime_act_id=op.act_id,
            missed_letters=missed, deadline_at=op.deadline_at, timeout_sec=self._letter_cfg("TURN_TIMEOUT_SEC", 1800.0))
        data, err = await self._post(f"{self.s.PRIME_HUB_URL.rstrip('/')}{ORION_DAY_ORDER_PATH}",
                                     order.model_dump(mode="json"))
        result: Optional[OrionDayLetterOrderResultV1] = None
        if data is not None:
            try:
                result = OrionDayLetterOrderResultV1.model_validate(data)
            except Exception as exc:  # noqa: BLE001
                err = f"unreadable answer: {str(exc)[:120]}"
        if result is None or result.status == "unavailable":
            detail = err if result is None else (result.detail or "unavailable")
            self.day_letter = self.day_letter.model_copy(update={"open": op.model_copy(update={
                "order_status": "unavailable", "order_detail": (detail or "")[:500]})})
            logger.info("prime_day_letter_order_unanswered day=%s run=%s err=%s", op.letter_date, op.run_id, detail)
            return
        logger.info("prime_day_letter_ordered day=%s run=%s status=%s missed=%d detail=%s", op.letter_date, op.run_id,
                    result.status, len(missed), result.detail or "-")
        if result.status in ("submitted", "exists"):
            # "Told" only once the letter that carried them is written (_close_letter): a failed run tells nobody.
            self.day_letter = self.day_letter.model_copy(update={
                "open": op.model_copy(update={"order_status": result.status, "order_detail": result.detail,
                                              "missed_sent": [m.letter_date for m in missed]})})
            return
        if result.status == "already_written" and dl.own_run(op, result.run_id):
            # An earlier attempt of ours wrote the row after all (a later step failed that run). It is this act's
            # letter: close it written, with that run's threads when its detail is readable.
            run, _err = await self._read(f"{self.s.PRIME_DURABLE_RUNS_URL.rstrip('/')}/runs/{result.run_id}")
            detail = run.get("orion_day") if isinstance(run, dict) and isinstance(run.get("orion_day"), dict) else {}
            threads = [str(t) for t in (detail.get("threads") or []) if isinstance(t, str)]
            turn = detail.get("turn_correlation_id")
            logger.info("prime_day_letter_own_attempt_wrote_it day=%s run=%s", op.letter_date, result.run_id)
            await self._close_letter(now, produced=True, outcome="written",
                                     note=f"an earlier attempt's run ({result.run_id}) wrote the day's letter",
                                     run_id=result.run_id, turn_correlation_id=str(turn) if turn else None,
                                     threads=threads)
            return
        outcome = {"already_written": "already_written", "empty_day": "empty"}.get(result.status, "not_written")
        note = {"already_written": f"a letter for this day already existed (run {result.run_id})",
                "empty_day": "the day held nothing to write about"}.get(
            result.status, f"Hub refused the order: {result.detail or result.status}")
        await self._close_letter(now, produced=False, outcome=outcome, note=note, run_id=result.run_id)

    async def _settle_letter(self, now: datetime) -> None:
        op = self.day_letter.open
        run, err = await self._read(f"{self.s.PRIME_DURABLE_RUNS_URL.rstrip('/')}/runs/{op.run_id}")
        if run is None:
            past = op.deadline_at is not None and now >= op.deadline_at + timedelta(seconds=dl.ACT_HORIZON_GRACE_SEC)
            if past:
                await self._close_letter(now, produced=False, outcome="not_written",
                                         note=f"the letter run {op.run_id} could not be read past its deadline ({err})")
            return
        produced, note, detail = dl.run_outcome(run)
        if produced is None:
            # A paused or stuck run must not hold the day open forever (it would block every later letter).
            if op.deadline_at is not None and now >= op.deadline_at + timedelta(seconds=dl.ACT_HORIZON_GRACE_SEC):
                await self._close_letter(now, produced=False, outcome="not_written",
                                         note=f"the letter run {op.run_id} was still {note} past its deadline")
            return
        threads = [str(t) for t in (detail.get("threads") or []) if isinstance(t, str)]
        turn = detail.get("turn_correlation_id")
        outcome = dl.done_outcome(produced, detail)
        max_attempts = int(self._letter_cfg("MAX_ATTEMPTS", float(dl.DEFAULT_MAX_ATTEMPTS)))
        # A cancel is someone's decision: not retried. Failed, or completed with no row: the next attempt.
        retryable = outcome == "not_written" and str(run.get("status") or "") != "cancelled"
        if retryable and dl.can_retry(op, now, max_attempts=max_attempts):
            # Live 11-11: one failed run (E2BIG, fixed in #3248) ended the day. Order the next attempt.
            self.day_letter = self.day_letter.model_copy(update={"open": dl.next_attempt(op, note)})
            nxt = self.day_letter.open
            logger.info("prime_day_letter_retry day=%s attempt=%d/%d run=%s why=%r", op.letter_date, nxt.attempt,
                        max_attempts, nxt.run_id, note)
            await self._order_letter(now)
            return
        await self._close_letter(now, produced=produced, outcome=outcome, note=note,
                                 run_id=op.run_id, turn_correlation_id=str(turn) if turn else None, threads=threads)

    async def _close_letter(self, now: datetime, *, produced: bool, outcome: dl.DoneOutcome, note: str,
                            run_id: Optional[str] = None, turn_correlation_id: Optional[str] = None,
                            threads: Optional[list[str]] = None) -> None:
        """Close the make act (kept open and retried next check if the publish fails), turn the threads into
        concerns, record the day."""
        op = self.day_letter.open
        day = op.letter_date
        if not produced:
            note = dl.attempts_note(op, note, max_attempts=int(self._letter_cfg(
                "MAX_ATTEMPTS", float(dl.DEFAULT_MAX_ATTEMPTS))))
        act = dl.make_act(day, horizon_sec=self._letter_cfg("RUN_DEADLINE_HOURS", 12.0) * 3600.0
                          + dl.ACT_HORIZON_GRACE_SEC)
        act_note = f"{note}; turn {turn_correlation_id}" if turn_correlation_id else note
        if not await self._close(act, dl.DAY_LETTER_COMPONENT, produced=produced, note=act_note[:500]):
            return
        threads = threads or []
        concern_ids: list[str] = []
        no_room = 0
        if produced and threads:
            self.state, concern_ids, no_room = dl.thread_concerns(self.state, day, threads, now)
        if produced and op.missed_sent:
            self.day_letter = dl.mark_told(self.day_letter, list(op.missed_sent))
        if turn_correlation_id and self._handback_on():
            self.turn_record = link_started_turn(self.turn_record, turn_correlation_id, op.act_id)
        self.day_letter = dl.finish(self.day_letter, dl.PrimeDayLetterDoneV1(
            letter_date=day, outcome=outcome, reason="" if produced else note[:dl.REASON_CAP], closed_at=now,
            act_id=op.act_id, run_id=run_id or op.run_id, turn_correlation_id=turn_correlation_id,
            threads=len(threads), concern_ids=concern_ids, threads_without_room=no_room))
        logger.info("prime_day_letter_closed day=%s act=%s produced=%s outcome=%s run=%s turn=%s threads=%d "
                    "concerns=%s no_room=%d note=%r", day, op.act_id, produced, outcome, run_id or op.run_id,
                    turn_correlation_id or "-", len(threads), ",".join(concern_ids) or "-", no_room, note)
        tail = (f"; {len(threads)} thread(s), {len(concern_ids)} now held as concerns"
                + (f", {no_room} without room" if no_room else "")) if produced else ""
        self.state = note_outcome(self.state, f"my letter for {day} (act {op.act_id}): {note}{tail}")


    # --- organ replies (curiosity replica patch 4) --------------------------------------------
    def _organ_replies_on(self) -> bool:
        """The replica answers Prime in this process (its own two switches)."""
        return bool(getattr(self.s, "CURIOSITY_REPLICA_ENABLED", False)) and bool(
            getattr(self.s, "CURIOSITY_REPLICA_ANSWERS_ENABLED", False))

    async def _read_organ_replies(self) -> tuple[Optional[CuriosityReplicaRepliesViewV1], Optional[str]]:
        """The replica's published replies view; (None, None) when none was ever published."""
        try:
            raw = await self.bus.redis.get(REPLIES_KEY)
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {str(exc)[:120]}"
        if not raw:
            return None, None
        try:
            return CuriosityReplicaRepliesViewV1.model_validate_json(raw), None
        except Exception as exc:  # noqa: BLE001
            return None, f"the replies view is unreadable ({str(exc)[:120]})"

    async def _ask(self, inputs: FrameInputs, now: datetime) -> ModelReply:
        """Build the frame and call the model (one immediate retry on an upstream 5xx: _ask_model)."""
        curiosity, curiosity_err = await self._read_curiosity()
        inputs.curiosity, inputs.curiosity_error, inputs.now = curiosity, curiosity_err, now
        # Once per thought, never per tick: the report takes ~1.5 s to build.
        inputs.energy, inputs.energy_error = await self._read(
            f"{self.s.PRIME_DURABLE_RUNS_URL.rstrip('/')}/physiology/report")
        inputs.organ_replies_on = self._organ_replies_on()
        if inputs.organ_replies_on:
            inputs.organ_replies, inputs.organ_replies_error = await self._read_organ_replies()
        receptors, receptors_err = self._receptors()
        inputs.receptors, inputs.receptors_error = receptors, receptors_err
        if receptors is not None or receptors_err:
            for reg in self.state.open_regulations:
                rec = await self._read_regulation(reg.family, reg.signal_id)
                inputs.regulation_effects[reg.signal_id] = (
                    None if rec is None or rec[0] is None or rec[0].signal_id != reg.signal_id else rec[1])
        frame = build_frame(self.state, inputs)
        reply = await self._ask_model(system_prompt(receptors, events=inputs.event_mode,
                                                    organ_replies=shows_organ_replies(inputs)), frame)
        reply.context.update(curiosity=curiosity, receptors=receptors)
        return reply

    # --- reads -----------------------------------------------------------------------------
    async def _read(self, url: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        try:
            data = await asyncio.to_thread(self.get_json, url)
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {str(exc)[:120]}"
        if not isinstance(data, dict):
            return None, "not a JSON object"
        return data, None

    async def _read_body_and_attention(self) -> tuple[PrimeSnapshotV1, FrameInputs]:
        base = self.s.PRIME_DURABLE_RUNS_URL.rstrip("/")
        body, body_err = await self._read(f"{base}/physiology/state")
        if body is not None and not body.get("ok", False):
            body, body_err = None, "body state not ok"
        # Attention is read for the frame (background), never as a change that wakes Prime. Event mode
        # (PRIME_CONCERNS_FROM_EVENT_WANTS): the funded event wants replace the Temporal Self arcs.
        if self._event_mode():
            events, events_err = await self._read(
                f"{base}/physiology/event-wants?hours={float(self.s.PRIME_EVENT_WANTS_HOURS):g}"
                f"&limit={int(self.s.PRIME_EVENT_WANTS_LIMIT)}")
            if events is not None:
                # Only the documented contract's fields reach the frame (FundedEventWantsV1).
                try:
                    events = FundedEventWantsV1.model_validate(events).model_dump(mode="json")
                except Exception as exc:  # noqa: BLE001
                    events, events_err = None, f"not a funded event wants payload ({type(exc).__name__})"
            snap = PrimeSnapshotV1(binding=self._binding(body), body_ok=body is not None)
            inputs = FrameInputs(now=self.clock(), body=body, body_error=body_err, events=events,
                                 events_error=events_err, event_mode=True, bid_lane=self.s.PRIME_BID_LANE)
            return snap, inputs
        temporal, temporal_err = await self._read(f"{base}/temporal-self/frame")
        active = (temporal or {}).get("active_arc") or {}
        snap = PrimeSnapshotV1(binding=self._binding(body), body_ok=body is not None)
        arc_ids = [a.get("arc_id") for a in (temporal or {}).get("arcs_today") or [] if a.get("arc_id")]
        if active.get("arc_id"):
            arc_ids.append(active["arc_id"])
        inputs = FrameInputs(now=self.clock(), body=body, body_error=body_err, temporal=temporal,
                             temporal_error=temporal_err, arc_ids=arc_ids, bid_lane=self.s.PRIME_BID_LANE)
        return snap, inputs

    def _event_mode(self) -> bool:
        return bool(getattr(self.s, "PRIME_CONCERNS_FROM_EVENT_WANTS", False))

    @staticmethod
    def _binding(body: Optional[dict[str, Any]]) -> Optional[list[str]]:
        prices = (body or {}).get("prices")
        return (sorted(n for n, p in prices.items() if isinstance(p, dict) and p.get("bind_state") == "binding")
                if isinstance(prices, dict) else None)

    async def _read_curiosity(self) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        base = self.s.PRIME_HUB_URL.rstrip("/")
        return await self._read(f"{base}/curiosity/api/runs?days={CURIOSITY_RUNS_DAYS}&line=investigate")

    # --- self-monitor act gate (E1): Prime's own acts -------------------------------------
    async def _concern_bet(self, concern_id: str, text: str, now: datetime) -> tuple[bool, Optional[GateDecision]]:
        """(may open, the decision). The probe is spent by the caller only after the open was published,
        so a failed publish spends nothing (and leaves no in-process memo a restart could lose)."""
        refused_at = self._bet_refused_at.get(concern_id)
        ref = f"prime_concern:{concern_id}"
        d = await self._gate_decide("predict", ref, job_key=ref, recheck_soon=(
            refused_at is not None and (now - refused_at).total_seconds() < GATE_RECHECK_SEC))
        if d is None or d.admit:
            self._bet_refused_at.pop(concern_id, None)
            return True, d
        if refused_at is None:
            # Once per concern: a frame line, not a change (a refusal must not wake Prime by itself).
            self.state = _with_outcome_line(self.state, f"my bet that the concern {text[:120]!r} leads somewhere "
                                                        f"was not placed: {gate_why(d, self.gate.probe_interval_sec)}")
            logger.info("prime_act_refused verb=predict concern=%s gate=%s", concern_id, d.label())
            self._bet_refused_at[concern_id] = now
        elif (now - refused_at).total_seconds() >= GATE_RECHECK_SEC:
            self._bet_refused_at[concern_id] = now
        return False, d

    def _bet_opened(self, d: Optional[GateDecision], concern_id: str) -> None:
        if d is None:
            return
        ref = f"prime_concern:{concern_id}"
        if d.probe:
            self.gate.claim_probe(d.faculty, now=self.clock(), job_key=ref)
        else:
            self.gate.note_job_started(d.faculty, ref, now=self.clock())

    async def _late_concern_bet_allowed(self, concern_id: str) -> bool:
        """A concern whose act never opened has ended: open+close it now only if the gate admits it as an
        ordinary act NOW (gate off, unknown/stale/absent view, healthy faculty, or a same-job retry). A
        failing faculty -> no late bet, not even as its probe: the outcome is known, so a late open would
        test nothing."""
        ref = f"prime_concern:{concern_id}"
        d = await self._gate_decide("predict", ref, job_key=ref)
        return d is None or (d.admit and not d.probe)

    async def _move_refusal(self, move: Any, *, thought_id: str) -> Optional[str]:
        """None: the move may go (probe spent if it was one). Else why it was refused. Asked before the
        holdback arm is drawn, so both arms are refused alike."""
        if isinstance(move, RegulateMoveV1):
            verb, ref = "regulate", f"physiology_family:{move.family}"
        else:
            verb = "ask"
            ref = f"curiosity_focus:{prime_focus_id(question=move.question, why=move.why, thought_id=thought_id)}"
        d = await self._gate_decide(verb, ref)
        if self._admit(d):
            return None
        logger.info("prime_act_refused verb=%s gate=%s", verb, d.label() if d else "-")
        what = "signals" if verb == "regulate" else "questions"
        return gate_why(d, self.gate.probe_interval_sec).replace("these", f"my {what}", 1)

    async def _flush_concern_acts(self) -> None:
        """Open the act of every concern that has none yet (new, or held from before concern acts), then
        publish the close of every unpublished concern record. Marked done only after a publish
        succeeds, so a failed or interrupted send is retried next tick; the ledger keeps the first open
        and the first close of an act, so a retry never counts twice. The self-monitor gate decides
        each open (see the module docstring)."""
        now = self.clock()
        no_bet = set(self.turn_record.no_bet_concern_ids)
        concerns = []
        for c in self.state.concerns:
            if c.act_opened_at is None and c.concern_id not in no_bet:
                allowed, decision = await self._concern_bet(c.concern_id, c.text, now)
                if allowed and await self._open(concern_act(c.concern_id), CONCERN_COMPONENT, at=now):
                    self._bet_opened(decision, c.concern_id)
                    c = c.model_copy(update={"act_opened_at": now})
            concerns.append(c)
        records = []
        for r in self.state.concern_record:
            if not r.published and r.act_opened_at is None and r.concern_id in no_bet:
                # A concern a turn opened: it never placed a bet (orion/prime/handback.py), so nothing to close.
                logger.info("prime_concern_ended_without_bet concern=%s end=%s reason=opened_by_turn",
                            r.concern_id, r.end)
                records.append(r.model_copy(update={"published": True}))
                continue
            if not r.published:
                act = concern_act(r.concern_id)
                if r.act_opened_at is None and not await self._late_concern_bet_allowed(r.concern_id):
                    # No bet was placed for this concern, so there is no act to close.
                    logger.info("prime_concern_ended_without_bet concern=%s end=%s", r.concern_id, r.end)
                    records.append(r.model_copy(update={"published": True}))
                    continue
                opened = r.act_opened_at is not None or await self._open(act, CONCERN_COMPONENT, at=r.opened_at)
                if opened and await self._close(act, CONCERN_COMPONENT, produced=r.produced, note=r.note,
                                                at=r.closed_at):
                    logger.info("prime_concern_closed concern=%s end=%s", r.concern_id, r.end)
                    r = r.model_copy(update={"published": True})
            records.append(r)
        self.state = self.state.model_copy(update={"concerns": concerns, "concern_record": trim_record(records)})
        live = {c.concern_id for c in self.state.concerns} | {r.concern_id for r in self.state.concern_record
                                                             if not r.published}
        for cid in [k for k in self._bet_refused_at if k not in live]:
            del self._bet_refused_at[cid]

    async def _conclude(self, act: ActV1, thought_id: str, inputs: FrameInputs, now: datetime,
                        reply: ModelReply, answer: Any) -> tuple[bool, str]:
        curiosity, receptors = reply.context.get("curiosity"), reply.context.get("receptors")
        schedule = (curiosity or {}).get("schedule") or {}
        pending_focus = schedule.get("pending_focus") or {}
        focus_pending = bool(pending_focus.get("focus")) if pending_focus.get("available") else None
        applied = apply_answer(self.state, answer, now=now, arc_ids=inputs.arc_ids, focus_pending=focus_pending,
                               receptors=receptors, event_ids=event_ids(inputs) if inputs.event_mode else None)
        if inputs.event_mode:
            new = set(applied.added)
            for c in applied.state.concerns:
                if c.concern_id in new:
                    logger.info("prime_concern_added concern=%s event_ref=%s", c.concern_id, c.event_ref)
        self.state = applied.state.model_copy(update={"last_thought_id": thought_id})
        if applied.move is not None and applied.refs:
            unopened = await self._unresolved(applied.refs)
            if unopened:
                applied.move, applied.refs = None, []
                applied.move_refusal = f"it cites {'; '.join(unopened)}, which a curiosity run could not open"
        if applied.move is not None:
            gate_refusal = await self._move_refusal(applied.move, thought_id=thought_id)
            if gate_refusal:
                applied.move, applied.refs = None, []
                applied.move_refusal = gate_refusal
        if applied.move_refusal:
            self.state = _with_outcome_line(self.state, f"my move was not sent: {applied.move_refusal}")
        if isinstance(applied.move, RegulateMoveV1):
            await self._regulate(applied.move, thought_id=thought_id, thought_act_id=act.act_id, now=now)
        elif applied.move is not None:
            await self._nominate(applied.move, thought_id=thought_id, thought_act_id=act.act_id, now=now,
                                 refs=applied.refs)
        note = summary_note(applied)
        logger.info("prime_thought thought=%s route=%s %s reading=%r", thought_id, reply.route, note,
                    answer.reading[:200])
        return True, note

    async def _unresolved(self, refs: list[RefV1]) -> list[str]:
        """Each ref opened with the same resolver a curiosity run's ``orion.dig <kind> <id>`` uses."""
        base = self.s.PRIME_DURABLE_RUNS_URL
        out = []
        for ref in refs:
            res = await asyncio.to_thread(resolve_ref, ref, base_url=base, get_json=self.get_json)
            if not res.ok:
                out.append(res.error or f"{ref.kind} {ref.id}")
        return out

    async def _nominate(self, move: Any, *, thought_id: str, thought_act_id: str, now: datetime,
                        refs: Optional[list[RefV1]] = None) -> None:
        focus_id = prime_focus_id(question=move.question, why=move.why, thought_id=thought_id)
        arm = draw_arm(focus_id, float(self.s.PRIME_HOLDBACK_FRACTION))
        act = nomination_act(focus_id=focus_id, arm=arm, thought_act_id=thought_act_id, concern_id=move.concern_id)
        await self._open(act, NOMINATION_COMPONENT)
        nom = PrimeNominationV1(act_id=act.act_id, thought_id=thought_id, focus_id=focus_id, question=move.question,
                                why=move.why, arm=arm, opened_at=now, concern_id=move.concern_id)
        if arm == "treated":
            queued, error = await self._queue_focus(move.question, move.why, thought_id, refs=refs or [])
            nom = nom.model_copy(update={"queued": queued, "queue_error": error})
        logger.info("prime_nomination focus=%s arm=%s queued=%s err=%s question=%r",
                    focus_id, arm, nom.queued, nom.queue_error, move.question[:160])
        self.state = self.state.model_copy(update={"open_nominations": [*self.state.open_nominations, nom]})

    async def _queue_focus(self, question: str, why: str, thought_id: str, *,
                           refs: Optional[list[RefV1]] = None) -> tuple[Optional[bool], Optional[str]]:
        """(True, None) queued; (False, why) Hub refused before writing; (None, why) answer unknown."""
        corr = uuid4()
        reply = f"{RESULT_PREFIX}{corr}"
        args = QueueCuriosityFocusArguments(question=question, why=why, origin="self",
                                            refs=refs or []).model_dump(mode="json")
        if not args["refs"]:
            args.pop("refs")  # a Hub from before refs (extra="forbid") still accepts a question that cites nothing
        request = IntrospectRequestV1(
            operation="queue_curiosity_focus",
            binding=IntrospectToolBindingV1(invocation_context="prime", parent_run_id=thought_id,
                                            parent_trace_id=chat_correlation_id(thought_id), memory_allowed=False),
            args=args,
        )
        try:
            raw = await self.bus.rpc_request(
                QUEUE_CURIOSITY_FOCUS_REQUEST_CHANNEL,
                BaseEnvelope(kind=REQUEST_KIND, correlation_id=corr, reply_to=reply, source=self.source,
                             payload=request.model_dump(mode="json")),
                reply_channel=reply, timeout_sec=FOCUS_RPC_TIMEOUT_SEC,
            )
            decoded = self.bus.codec.decode(raw.get("data"))
            if not decoded.ok or decoded.envelope.kind != RESULT_KIND:
                return None, "malformed reply"
            result = IntrospectResultV1.model_validate(decoded.envelope.payload)
        except Exception as exc:  # noqa: BLE001
            return None, f"no answer: {type(exc).__name__}"
        if result.ok:
            return True, None
        # Hub answers ok=false only before it writes, except its own "answer unknown".
        if "answer unknown" in (result.error or ""):
            return None, result.error
        return False, (result.error or "refused")[:300]

    # --- regulate signals (step 3) ------------------------------------------------------------
    def _receptors(self) -> tuple[Optional[dict[str, tuple[str, ...]]], Optional[str]]:
        """(family -> directions it receives, None) when regulation is on; (None, None) when off;
        (None, reason) when the families config could not be read (shown to Prime as a gap)."""
        if not bool(getattr(self.s, "PRIME_REGULATE_ENABLED", False)):
            return None, None
        from orion.physiology.families import load_families

        fams, err = load_families(getattr(self.s, "PRIME_FAMILIES_PATH", "/repo/config/physiology/families.yaml"))
        if fams is None:
            return None, f"families config unreadable ({err})"
        return {f: c.receptors for f, c in fams.families.items() if c.receptors}, None

    async def _read_regulation(
        self, family: str, signal_id: str,
    ) -> Optional[tuple[Optional[pc.PrimeRegulationV1], list[pc.PrimeChargeEffectV1]]]:
        """(the signal in the hash, the market's per-want record for ``signal_id``); None when Redis could
        not be read."""
        try:
            fields = await self.bus.redis.hgetall(pc.key(family))
        except Exception as exc:  # noqa: BLE001
            logger.warning("prime_regulation_read_failed family=%s err=%s", family, type(exc).__name__)
            return None
        fields = fields or {}
        raw = fields.get(pc.SIGNAL_FIELD, fields.get(pc.SIGNAL_FIELD.encode()))
        return pc.parse_signal(raw), pc.parse_effects(fields, signal_id=signal_id)

    async def _market_regulation_on(self) -> Optional[bool]:
        """The market's own switch, from orion-durable-runs /health (None: unread or the market is off)."""
        data, _err = await self._read(f"{self.s.PRIME_DURABLE_RUNS_URL.rstrip('/')}/health")
        market = (data or {}).get("physiology_market")
        if not isinstance(market, dict) or "prime_regulation" not in market:
            return None
        return bool(market["prime_regulation"])

    async def _regulate(self, move: RegulateMoveV1, *, thought_id: str, thought_act_id: str, now: datetime) -> None:
        signal_id = new_id()
        arm = draw_arm(f"regulate:{signal_id}", float(self.s.PRIME_HOLDBACK_FRACTION))
        hours = float(move.hours or pc.DEFAULT_HOURS)
        act = regulate_act(signal_id, family=move.family, arm=arm, hours=hours, thought_act_id=thought_act_id)
        await self._open(act, REGULATE_COMPONENT, at=now)
        expires = now + timedelta(hours=hours)
        reg = PrimeRegulationOpenV1(act_id=act.act_id, signal_id=signal_id, thought_id=thought_id, family=move.family,
                                    direction=move.direction, why=move.why, arm=arm, magnitude=pc.DEFAULT_MAGNITUDE,
                                    opened_at=now, expires_at=expires)
        sig = pc.PrimeRegulationV1(signal_id=signal_id, act_id=act.act_id, family=move.family,
                                   direction=move.direction, arm=arm, magnitude=pc.DEFAULT_MAGNITUDE,
                                   why=move.why[:600], opened_at=now, expires_at=expires)
        key = pc.key(move.family)
        try:
            # Both arms are written: the market applies a treated signal and only records what a control
            # one would have done. The key outlives the signal by GRACE_SEC so the final record can be read.
            # One MULTI: the signal is in force with its expiry, or not at all (never "not delivered" while
            # the market applies it, never a key without a TTL).
            async with self.bus.redis.pipeline(transaction=True) as pipe:
                pipe.delete(key)
                pipe.hset(key, pc.SIGNAL_FIELD, pc.dumps(sig))
                pipe.expire(key, int(hours * 3600.0 + pc.GRACE_SEC))
                await pipe.execute()
            reg = reg.model_copy(update={"written": True})
        except Exception as exc:  # noqa: BLE001
            reg = reg.model_copy(update={"written": False, "write_error": f"{type(exc).__name__}: {str(exc)[:120]}"})
        logger.info("prime_regulate signal=%s act=%s family=%s direction=%s arm=%s hours=%g written=%s err=%s why=%r",
                    signal_id, act.act_id, move.family, move.direction, arm, hours, reg.written, reg.write_error,
                    move.why[:160])
        self.state = self.state.model_copy(update={"open_regulations": [*self.state.open_regulations, reg]})

    async def _settle_regulations(self, now: datetime) -> None:
        """Grade each signal once it has ended (+ a short lag): its act closes with what the MARKET recorded.
        A record that is gone (Redis lost it, or Prime was down past the grace) is not graded here: the act
        is left to the ledger's horizon expiry, which is not counted as Prime's outcome."""
        keep: list[PrimeRegulationOpenV1] = []
        rest = dict(self.state.regulation_rest_until)
        for fam in [f for f, until in rest.items() if until <= now]:
            del rest[fam]
        for reg in self.state.open_regulations:
            if reg.expires_at <= now:
                # The rest (no new signal to this organ) runs as long as the signal did, from its end.
                rest[reg.family] = max(rest.get(reg.family, now), reg.expires_at + (reg.expires_at - reg.opened_at))
            act = regulate_act(reg.signal_id, family=reg.family, arm=reg.arm, hours=regulation_hours(reg),
                               thought_act_id=thought_act(reg.thought_id, llm_timeout_sec=0).act_id)
            if reg.written is False:
                produced, note = settle_regulation(reg, [])
                if not await self._close(act, REGULATE_COMPONENT, produced=produced, note=note):
                    keep.append(reg)
                    continue
                self.state = _with_outcome_line(self.state, f"my {reg.direction} signal to {reg.family} was not "
                                                            f"delivered: {reg.write_error or 'write failed'}")
                continue
            if (now - reg.expires_at).total_seconds() < REGULATE_SETTLE_LAG_SEC:
                keep.append(reg)
                continue
            rec = await self._read_regulation(reg.family, reg.signal_id)
            gone = rec is not None and (rec[0] is None or rec[0].signal_id != reg.signal_id)
            past_grace = (now - reg.expires_at).total_seconds() > pc.GRACE_SEC
            if rec is None and not past_grace:
                keep.append(reg)          # Redis unread: try again next tick
                continue
            if rec is None or gone:
                self.state = note_outcome(self.state, f"my {reg.direction} signal to {reg.family} (act {reg.act_id}) "
                                                      "could not be graded: the market's record of it is gone")
                continue
            market_on = None if rec[1] else await self._market_regulation_on()
            produced, note = settle_regulation(reg, rec[1], market_on=market_on)
            replies, _ = await self._read_organ_replies()
            refused = organ_refusal(reg.act_id, replies)
            if refused is not None:
                # The market already graded what happened (unchanged from the refusal on); name who and why.
                note = f"{note}; {refusal_words(refused)} (at {refused.at.strftime('%H:%MZ')})"
            if not await self._close(act, REGULATE_COMPONENT, produced=produced, note=note):
                keep.append(reg)
                continue
            logger.info("prime_regulation_settled signal=%s act=%s produced=%s note=%s",
                        reg.signal_id, reg.act_id, produced, note)
            self.state = note_outcome(self.state, f"my {reg.direction} signal to {reg.family} (act {reg.act_id}): {note}")
        self.state = self.state.model_copy(update={"open_regulations": keep, "regulation_rest_until": rest})

    # --- settling questions ----------------------------------------------------------------
    @staticmethod
    def _run_turn_correlation(runs: list[Any], run_id: Optional[str]) -> Optional[str]:
        """The unified turn a curiosity run ran as (Hub's runs API ``harness.turn_correlation_id``), or None."""
        if not run_id:
            return None
        run = next((r for r in runs if isinstance(r, dict) and str(r.get("run_id") or "") == run_id), None)
        corr = ((run or {}).get("harness") or {}).get("turn_correlation_id")
        return str(corr)[:200] if corr else None

    async def _maybe_settle(self, now: datetime) -> None:
        if not self.state.open_nominations:
            return
        due = self._last_settle is None or (now - self._last_settle).total_seconds() >= float(self.s.PRIME_SETTLE_INTERVAL_SEC)
        if not due:
            return
        self._last_settle = now
        curiosity, err = await self._read_curiosity()
        runs = (curiosity or {}).get("runs") if curiosity else None
        if not isinstance(runs, list):
            logger.info("prime_settle_skipped err=%s", err or "no runs list")
            return
        replies = None
        if any(n.arm == "treated" for n in self.state.open_nominations):
            replies, _ = await self._read_organ_replies()
        keep: list[PrimeNominationV1] = []
        for nom in self.state.open_nominations:
            settled = settle_nomination(arm=nom.arm, focus_id=nom.focus_id, opened_at=nom.opened_at, queued=nom.queued,
                                        queue_error=nom.queue_error, runs=runs, now=now)
            refused = organ_refusal(nom.act_id, replies) if settled is None and nom.arm == "treated" else None
            if refused is not None:
                # The organ refused it: no run was shown it. Still the treated arm (intention to treat).
                settled = Settlement(False, refusal_words(refused))
            if settled is None and nom.arm == "treated":
                gone = treated_focus_gone(nom.focus_id, curiosity)
                nom = nom.model_copy(update={"gone_reads": nom.gone_reads + 1 if gone else 0})
                if nom.gone_reads >= GONE_READS_TO_SETTLE:
                    settled = Settlement(False, "treated focus left Hub's slot with no run taking it "
                                                "(replaced, cleared, expired, or never written)")
            if settled is None and (now - nom.opened_at).total_seconds() <= NOMINATION_HORIZON_SEC:
                keep.append(nom)
                continue
            act = nomination_act(focus_id=nom.focus_id, arm=nom.arm,
                                 thought_act_id=thought_act(nom.thought_id, llm_timeout_sec=0).act_id,
                                 concern_id=nom.concern_id)
            if settled is None:
                # Past the horizon with no outcome: the ledger expires the act at 0; Prime lets go.
                self.state = note_outcome(self.state, f"my question {nom.question[:120]!r} got no outcome before its horizon")
                continue
            turn_corr = self._run_turn_correlation(runs, settled.run_id)
            if turn_corr and self._handback_on():
                self.turn_record = link_started_turn(self.turn_record, turn_corr, act.act_id)
            await self._close(act, NOMINATION_COMPONENT, produced=settled.produced,
                              note=f"{settled.note} turn {turn_corr}" if turn_corr else settled.note)
            logger.info("prime_nomination_settled focus=%s arm=%s produced=%s note=%s",
                        nom.focus_id, nom.arm, settled.produced, settled.note)
            seen = settled.note if nom.arm == "treated" else "held back as a control (it was never sent)"
            self.state, concern_line = settle_concern(self.state, nom.concern_id, arm=nom.arm,
                                                      produced=settled.produced, now=now,
                                                      note=f"question act {act.act_id}: {settled.note}")
            tail = f"; {concern_line}" if concern_line else ""
            self.state = note_outcome(self.state, f"my question {nom.question[:120]!r} (act {act.act_id}): {seen}{tail}")
        self.state = self.state.model_copy(update={"open_nominations": keep})
