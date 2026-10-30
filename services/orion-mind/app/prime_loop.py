"""Orion Prime's resident loop (rollout step 2A): orion-mind keeps thinking between conversations.

Design: docs/superpowers/specs/2026-10-29-orion-prime-design.md, sections 7.2 and 8. Pure logic is in
orion/prime/; this module does the I/O.

Two clocks:

* **tick** (every PRIME_TICK_SEC, CPU only): read the body (durable-runs GET /physiology/state) and
  attention (GET /temporal-self/frame), record what changed (orion.prime.state.observe), settle any
  open question, save the state to Redis. Most ticks stop here.
* **thought** (only when something changed, the refractory interval has passed, no Juniper turn is
  running, and the body economy FUNDS the bid): one model call on route PRIME_LLM_ROUTE (circe chat
  lane 0 at background priority), then at most one move.

Prime obeys its own bid: ``funded`` thinks, anything else (deferred, no answer, flag off) waits. It is
the first producer whose decision binds (every other bid is still shadow); it never fails open.

The move (2A): one question for curiosity's next regular investigate run, queued through Hub's
existing focus slot (orion/curiosity/focus.py) with binding.invocation_context "prime". A share of
questions (PRIME_HOLDBACK_FRACTION) is the narrator arm: opened as an act, never sent.

Juniper first: a Juniper turn through run_mind (engine.carries_live_turn) is tracked by JUNIPER_TURNS.
Today ``mind_run`` calls the synchronous run_mind on the event loop, so no tick runs during a turn at
all; ``JUNIPER_TURNS.active()`` keeps that true if run_mind ever moves off the loop. What does the
work today: a thought whose model call (in a worker thread) overlapped a turn is recorded ``preempted``
and discarded, and the ``prime`` route's background priority keeps a Prime call from STARTING while
Juniper's call holds the slot. A call already generating is NOT cancelled (the bus RPC has no cancel);
the gateway generates nothing for a call granted after its 60 s budget (llm-gateway app/main.py).

Everything here is best-effort: a failed read, publish or save is logged and the loop goes on.
``run_mind`` never waits on this loop: every blocking call runs in a worker thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Iterator, Optional
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.grammar.acts import act_closed_event, act_opened_event
from orion.grammar.publish import GRAMMAR_EVENT_CHANNEL
from orion.introspect.transport import (
    QUEUE_CURIOSITY_FOCUS_REQUEST_CHANNEL,
    REQUEST_KIND,
    RESULT_KIND,
    RESULT_PREFIX,
)
from orion.prime.acts import (
    GONE_READS_TO_SETTLE,
    NOMINATION_COMPONENT,
    NOMINATION_HORIZON_SEC,
    SOURCE_SERVICE,
    THOUGHT_COMPONENT,
    chat_correlation_id,
    draw_arm,
    nomination_act,
    prime_focus_id,
    Settlement,
    settle_nomination,
    thought_act,
    treated_focus_gone,
)
from orion.prime.state import (
    MAX_RECENT_OUTCOMES,
    PrimeNominationV1,
    PrimeSnapshotV1,
    PrimeStateV1,
    new_id,
    note_outcome,
    observe,
    wants_to_think,
)
from orion.prime.thought import SYSTEM_PROMPT, FrameInputs, apply_answer, build_frame, parse_answer, summary_note
from orion.schemas.grammar import ActV1
from orion.schemas.introspect import (
    IntrospectRequestV1,
    IntrospectResultV1,
    IntrospectToolBindingV1,
    QueueCuriosityFocusArguments,
)

logger = logging.getLogger("orion-mind.prime")

STATE_KEY = "orion:prime:state"
FAMILY = "prime"
PRODUCER = "orion-mind.prime"
HTTP_TIMEOUT_SEC = 4.0
FOCUS_RPC_TIMEOUT_SEC = 10.0
CURIOSITY_RUNS_DAYS = 4  # covers a nomination's whole horizon (48 h focus TTL + 24 h run)


class JuniperTurns:
    """Juniper's live turns through run_mind. Thread-safe: run_mind runs on the request path."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self._last_started = 0.0

    @contextmanager
    def turn(self) -> Iterator[None]:
        with self._lock:
            self._active += 1
            self._last_started = time.monotonic()
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1

    def active(self) -> bool:
        with self._lock:
            return self._active > 0

    def started_since(self, mono: float) -> bool:
        with self._lock:
            return self._active > 0 or self._last_started >= mono


JUNIPER_TURNS = JuniperTurns()


def _get_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_SEC) as resp:  # noqa: S310 -- internal service URL
        return json.loads(resp.read())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _with_outcome_line(state: PrimeStateV1, text: str) -> PrimeStateV1:
    """Show a line in the next frame WITHOUT making it a change (a refusal Prime caused itself must
    not wake it again, or a refused move would loop)."""
    return state.model_copy(update={"recent_outcomes": [*state.recent_outcomes, text][-MAX_RECENT_OUTCOMES:]})


class PrimeRuntime:
    def __init__(
        self,
        *,
        settings: Any,
        bus: Any,
        llm: Any,
        get_json: Callable[[str], Any] = _get_json,
        bid: Optional[Callable[[Any], Awaitable[Any]]] = None,
        release: Optional[Callable[[str], Awaitable[Any]]] = None,
        clock: Callable[[], datetime] = _utcnow,
        turns: JuniperTurns = JUNIPER_TURNS,
    ) -> None:
        self.s = settings
        self.bus = bus
        self.llm = llm
        self.get_json = get_json
        if bid is None or release is None:
            from orion.physiology.client import abid, arelease

            bid, release = bid or abid, release or arelease
        self.bid, self.release = bid, release
        self.clock = clock
        self.turns = turns
        self.state = PrimeStateV1()
        self._last_settle: Optional[datetime] = None
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self.source = ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION, node=settings.NODE_NAME)

    # --- lifecycle -------------------------------------------------------------------------
    async def start(self, *, connect: bool = False) -> None:
        """Start the loop task. ``connect``: the task connects the bus first, retrying with backoff,
        so a bus blip at boot delays Prime instead of disabling it for the process lifetime."""
        self._task = asyncio.create_task(self._run(connect=connect), name="orion-prime")

    async def _connect(self) -> bool:
        delay = 1.0
        while not self._stop.is_set():
            try:
                await self.bus.connect()
                return True
            except Exception as exc:  # noqa: BLE001
                logger.warning("prime_bus_connect_failed err=%s retry_in=%.0fs", type(exc).__name__, delay)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            delay = min(delay * 2, 60.0)
        return False

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
        logger.info("prime_started tick_sec=%s route=%s holdback=%s concerns=%d open_questions=%d",
                    self.s.PRIME_TICK_SEC, self.s.PRIME_LLM_ROUTE, self.s.PRIME_HOLDBACK_FRACTION,
                    len(self.state.concerns), len(self.state.open_nominations))
        while not self._stop.is_set():
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("prime_tick_failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=float(self.s.PRIME_TICK_SEC))
            except asyncio.TimeoutError:
                pass

    # --- persistence -----------------------------------------------------------------------
    async def load(self) -> None:
        try:
            raw = await self.bus.redis.get(STATE_KEY)
        except Exception as exc:  # noqa: BLE001
            logger.warning("prime_state_load_failed err=%s (starting empty)", type(exc).__name__)
            return
        if not raw:
            return
        try:
            self.state = PrimeStateV1.model_validate_json(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("prime_state_unreadable err=%s (starting empty)", str(exc)[:200])

    async def save(self) -> None:
        try:
            await self.bus.redis.set(STATE_KEY, self.state.model_dump_json())
        except Exception as exc:  # noqa: BLE001
            logger.warning("prime_state_save_failed err=%s", type(exc).__name__)

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
        temporal, temporal_err = await self._read(f"{base}/temporal-self/frame")
        active = (temporal or {}).get("active_arc") or {}
        prices = (body or {}).get("prices")
        binding = (sorted(n for n, p in prices.items() if isinstance(p, dict) and p.get("bind_state") == "binding")
                   if isinstance(prices, dict) else None)
        snap = PrimeSnapshotV1(
            active_arc_id=active.get("arc_id"),
            active_arc_label=active.get("subject_label") or active.get("subject_ref"),
            binding=binding,
            temporal_self_ok=temporal is not None,
            body_ok=body is not None,
        )
        arc_ids = [a.get("arc_id") for a in (temporal or {}).get("arcs_today") or [] if a.get("arc_id")]
        if active.get("arc_id"):
            arc_ids.append(active["arc_id"])
        inputs = FrameInputs(now=self.clock(), body=body, body_error=body_err, temporal=temporal,
                             temporal_error=temporal_err, arc_ids=arc_ids, bid_lane=self.s.PRIME_BID_LANE)
        return snap, inputs

    async def _read_curiosity(self) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        base = self.s.PRIME_HUB_URL.rstrip("/")
        return await self._read(f"{base}/curiosity/api/runs?days={CURIOSITY_RUNS_DAYS}&line=investigate")

    # --- acts ------------------------------------------------------------------------------
    async def _publish(self, event: Any) -> None:
        try:
            await self.bus.publish(GRAMMAR_EVENT_CHANNEL, BaseEnvelope(
                kind="grammar.event.v1", source=self.source, payload=event.model_dump(mode="json")))
        except Exception as exc:  # noqa: BLE001
            logger.warning("prime_act_publish_failed act=%s err=%s", event.act.act_id if event.act else "-",
                           type(exc).__name__)

    async def _open(self, act: ActV1, component: str) -> None:
        await self._publish(act_opened_event(act, source_service=SOURCE_SERVICE, source_component=component))

    async def _close(self, act: ActV1, component: str, *, produced: bool, note: str) -> None:
        await self._publish(act_closed_event(act, produced=produced, source_service=SOURCE_SERVICE,
                                             source_component=component, note=note))

    # --- the tick --------------------------------------------------------------------------
    async def tick(self) -> None:
        snap, inputs = await self._read_body_and_attention()
        now = self.clock()
        self.state = observe(self.state, snap, now)
        await self._maybe_settle(now)
        await self.save()
        if self.turns.active():
            logger.info("prime_yield reason=juniper_turn pending=%d", len(self.state.pending_changes))
            return
        if not wants_to_think(self.state, now, refractory=timedelta(seconds=float(self.s.PRIME_REFRACTORY_SEC))):
            return
        from orion.schemas.physiology_want import WantV1

        want = WantV1(want_id=f"prime-think-{uuid4().hex[:12]}", family=FAMILY, producer=PRODUCER, verb="infer",
                      tier=1, lanes=[self.s.PRIME_BID_LANE], cost=self._cost_ceiling())
        decision = await self.bid(want)
        outcome = getattr(decision, "outcome", None)
        if outcome != "funded":
            # Never fail open: no answer is "not funded" (orion/physiology/client.py).
            logger.info("prime_bid_not_funded outcome=%s reason=%s binding=%s pending=%d",
                        outcome or "none", getattr(decision, "reason", None) or "-",
                        getattr(decision, "binding_constraint", None) or "-", len(self.state.pending_changes))
            return
        try:
            await self.think(inputs)
        finally:
            await self.release(want.want_id)
            await self.save()

    def _cost_ceiling(self) -> Any:
        """One thought's lane cost at its enforced maximum: the model call's timeout, on Prime's lane.

        Without it a family with no measured slot time is priced at a whole lane window
        (orion/physiology/choice.py "priced_at_one_window"). Live 10-30, every Prime bid was then
        deferred capacity_reserved on lane:chat (24/24), so Prime could never think and never build the
        history that would price it. Same rule as a dispatch route's cold_cost_ceiling_sec: charge the
        doer's enforced maximum while nothing is on record."""
        from orion.schemas.grammar import ActCostV2

        ceiling = float(self.s.PRIME_LLM_TIMEOUT_SEC)
        return ActCostV2(gpu_slot_s_by_lane={self.s.PRIME_BID_LANE: ceiling},
                         basis={"gpu_slot_s_by_lane": f"ceiling: PRIME_LLM_TIMEOUT_SEC {ceiling:g} s per thought"})

    # --- the thought -----------------------------------------------------------------------
    async def think(self, inputs: FrameInputs) -> None:
        now = self.clock()
        thought_id = new_id()
        act = thought_act(thought_id, llm_timeout_sec=float(self.s.PRIME_LLM_TIMEOUT_SEC))
        await self._open(act, THOUGHT_COMPONENT)
        # Counts for the refractory interval from the moment it opens, whatever happens next (a failed,
        # preempted or crashed thought must not retry every tick); its pending changes stay unless applied.
        self.state = self.state.model_copy(update={"last_thought_at": now})
        try:
            produced, note = await self._think(act, thought_id, inputs, now)
        except Exception as exc:  # noqa: BLE001 -- the act always closes
            logger.exception("prime_thought_crashed thought=%s", thought_id)
            produced, note = False, f"failed: {type(exc).__name__}: {str(exc)[:160]}"
        await self._close(act, THOUGHT_COMPONENT, produced=produced, note=note)

    async def _think(self, act: ActV1, thought_id: str, inputs: FrameInputs, now: datetime) -> tuple[bool, str]:
        curiosity, curiosity_err = await self._read_curiosity()
        inputs.curiosity, inputs.curiosity_error, inputs.now = curiosity, curiosity_err, now
        frame = build_frame(self.state, inputs)
        started = time.monotonic()
        try:
            parsed, err, _meta = await asyncio.to_thread(
                self.llm.request_json,
                system_prompt=SYSTEM_PROMPT, user_prompt=frame, route=self.s.PRIME_LLM_ROUTE,
                max_tokens=int(self.s.PRIME_MAX_TOKENS), temperature=0.4, thinking=False,
                timeout_sec=float(self.s.PRIME_LLM_TIMEOUT_SEC),
            )
        except Exception as exc:  # noqa: BLE001
            parsed, err = None, f"{type(exc).__name__}: {str(exc)[:160]}"
        if self.turns.started_since(started):
            logger.info("prime_thought_preempted thought=%s", thought_id)
            return False, "preempted: a Juniper turn started"
        answer, perr = (None, err) if err else parse_answer(parsed)
        if answer is None:
            logger.info("prime_thought_failed thought=%s err=%s", thought_id, perr)
            return False, f"failed: {perr}"
        schedule = (curiosity or {}).get("schedule") or {}
        pending_focus = schedule.get("pending_focus") or {}
        focus_pending = bool(pending_focus.get("focus")) if pending_focus.get("available") else None
        applied = apply_answer(self.state, answer, now=now, arc_ids=inputs.arc_ids, focus_pending=focus_pending)
        self.state = applied.state
        if applied.move_refusal:
            self.state = _with_outcome_line(self.state, f"my move was not sent: {applied.move_refusal}")
        if applied.move is not None:
            await self._nominate(applied.move, thought_id=thought_id, thought_act_id=act.act_id, now=now)
        note = summary_note(applied)
        logger.info("prime_thought thought=%s %s reading=%r", thought_id, note, answer.reading[:200])
        return True, note

    async def _nominate(self, move: Any, *, thought_id: str, thought_act_id: str, now: datetime) -> None:
        focus_id = prime_focus_id(question=move.question, why=move.why, thought_id=thought_id)
        arm = draw_arm(focus_id, float(self.s.PRIME_HOLDBACK_FRACTION))
        act = nomination_act(focus_id=focus_id, arm=arm, thought_act_id=thought_act_id, concern_id=move.concern_id)
        await self._open(act, NOMINATION_COMPONENT)
        nom = PrimeNominationV1(act_id=act.act_id, thought_id=thought_id, focus_id=focus_id, question=move.question,
                                why=move.why, arm=arm, opened_at=now, concern_id=move.concern_id)
        if arm == "treated":
            queued, error = await self._queue_focus(move.question, move.why, thought_id)
            nom = nom.model_copy(update={"queued": queued, "queue_error": error})
        logger.info("prime_nomination focus=%s arm=%s queued=%s err=%s question=%r",
                    focus_id, arm, nom.queued, nom.queue_error, move.question[:160])
        self.state = self.state.model_copy(update={"open_nominations": [*self.state.open_nominations, nom]})

    async def _queue_focus(self, question: str, why: str, thought_id: str) -> tuple[Optional[bool], Optional[str]]:
        """(True, None) queued; (False, why) Hub refused before writing; (None, why) answer unknown."""
        corr = uuid4()
        reply = f"{RESULT_PREFIX}{corr}"
        request = IntrospectRequestV1(
            operation="queue_curiosity_focus",
            binding=IntrospectToolBindingV1(invocation_context="prime", parent_run_id=thought_id,
                                            parent_trace_id=chat_correlation_id(thought_id), memory_allowed=False),
            args=QueueCuriosityFocusArguments(question=question, why=why, origin="self").model_dump(mode="json"),
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

    # --- settling questions ----------------------------------------------------------------
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
        keep: list[PrimeNominationV1] = []
        for nom in self.state.open_nominations:
            settled = settle_nomination(arm=nom.arm, focus_id=nom.focus_id, opened_at=nom.opened_at, queued=nom.queued,
                                        queue_error=nom.queue_error, runs=runs, now=now)
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
            await self._close(act, NOMINATION_COMPONENT, produced=settled.produced, note=settled.note)
            logger.info("prime_nomination_settled focus=%s arm=%s produced=%s note=%s",
                        nom.focus_id, nom.arm, settled.produced, settled.note)
            seen = settled.note if nom.arm == "treated" else "held back as a control (it was never sent)"
            self.state = note_outcome(self.state, f"my question {nom.question[:120]!r}: {seen}")
        self.state = self.state.model_copy(update={"open_nominations": keep})
