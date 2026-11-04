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
  attention (GET /temporal-self/frame), record what changed (orion.prime.state.observe), settle any
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

Prime obeys its own bid: ``funded`` thinks, anything else (deferred, no answer, flag off) waits. It is
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

Everything here is best-effort: a failed read, publish or save is logged and the loop goes on.
``run_mind`` never waits on this loop: every blocking call runs in a worker thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
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
from orion.prime.state import (
    MAX_RECENT_OUTCOMES,
    PrimeNominationV1,
    PrimeRegulationOpenV1,
    PrimeSnapshotV1,
    PrimeStateV1,
    new_id,
    note_outcome,
    observe,
    trim_record,
    wants_to_think,
)
from orion.prime.thought import (
    FrameInputs,
    RegulateMoveV1,
    apply_answer,
    build_frame,
    parse_answer,
    summary_note,
    woke_note,
    system_prompt,
)
from orion.refs import RefV1, resolve as resolve_ref
from orion.schemas.grammar import ActV1
from orion.schemas.introspect import (
    IntrospectRequestV1,
    IntrospectResultV1,
    IntrospectToolBindingV1,
    QueueCuriosityFocusArguments,
)
from orion.self_monitor.gate import GateDecision

logger = logging.getLogger("orion-mind.prime")

STATE_KEY = "orion:prime:state"
FAMILY = "prime"
PRODUCER = "orion-mind.prime"
HTTP_TIMEOUT_SEC = 4.0
FOCUS_RPC_TIMEOUT_SEC = 10.0
CURIOSITY_RUNS_DAYS = 4  # covers a nomination's whole horizon (48 h focus TTL + 24 h run)
# A regulate act is graded this long after its signal ends: the market's last bid under it has landed.
REGULATE_SETTLE_LAG_SEC = 120.0

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
        bid: Optional[Callable[[Any], Awaitable[Any]]] = None,
        release: Optional[Callable[[str], Awaitable[Any]]] = None,
        clock: Callable[[], datetime] = utcnow,
        turns: JuniperTurns = JUNIPER_TURNS,
    ) -> None:
        super().__init__(settings=settings, bus=bus, llm=llm, bid=bid, release=release, clock=clock, turns=turns)
        self.state: PrimeStateV1
        self.get_json = get_json
        # concern_id -> when its bet was last refused (re-check throttle; also: never a late bet for it).
        self._bet_refused_at: dict[str, datetime] = {}
        self._last_settle: Optional[datetime] = None

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

    async def _ask(self, inputs: FrameInputs, now: datetime) -> ModelReply:
        """Build the frame and call the model (one immediate retry on an upstream 5xx: _ask_model)."""
        curiosity, curiosity_err = await self._read_curiosity()
        inputs.curiosity, inputs.curiosity_error, inputs.now = curiosity, curiosity_err, now
        receptors, receptors_err = self._receptors()
        inputs.receptors, inputs.receptors_error = receptors, receptors_err
        if receptors is not None or receptors_err:
            for reg in self.state.open_regulations:
                rec = await self._read_regulation(reg.family, reg.signal_id)
                inputs.regulation_effects[reg.signal_id] = (
                    None if rec is None or rec[0] is None or rec[0].signal_id != reg.signal_id else rec[1])
        frame = build_frame(self.state, inputs)
        reply = await self._ask_model(system_prompt(receptors), frame)
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
        # Attention is read for the frame (background), never as a change that wakes Prime.
        temporal, temporal_err = await self._read(f"{base}/temporal-self/frame")
        active = (temporal or {}).get("active_arc") or {}
        prices = (body or {}).get("prices")
        binding = (sorted(n for n, p in prices.items() if isinstance(p, dict) and p.get("bind_state") == "binding")
                   if isinstance(prices, dict) else None)
        snap = PrimeSnapshotV1(binding=binding, body_ok=body is not None)
        arc_ids = [a.get("arc_id") for a in (temporal or {}).get("arcs_today") or [] if a.get("arc_id")]
        if active.get("arc_id"):
            arc_ids.append(active["arc_id"])
        inputs = FrameInputs(now=self.clock(), body=body, body_error=body_err, temporal=temporal,
                             temporal_error=temporal_err, arc_ids=arc_ids, bid_lane=self.s.PRIME_BID_LANE)
        return snap, inputs

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
        concerns = []
        for c in self.state.concerns:
            if c.act_opened_at is None:
                allowed, decision = await self._concern_bet(c.concern_id, c.text, now)
                if allowed and await self._open(concern_act(c.concern_id), CONCERN_COMPONENT, at=now):
                    self._bet_opened(decision, c.concern_id)
                    c = c.model_copy(update={"act_opened_at": now})
            concerns.append(c)
        records = []
        for r in self.state.concern_record:
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
                               receptors=receptors)
        self.state = applied.state
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
            if not await self._close(act, REGULATE_COMPONENT, produced=produced, note=note):
                keep.append(reg)
                continue
            logger.info("prime_regulation_settled signal=%s act=%s produced=%s note=%s",
                        reg.signal_id, reg.act_id, produced, note)
            self.state = note_outcome(self.state, f"my {reg.direction} signal to {reg.family} (act {reg.act_id}): {note}")
        self.state = self.state.model_copy(update={"open_regulations": keep, "regulation_rest_until": rest})

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
            self.state, concern_line = settle_concern(self.state, nom.concern_id, arm=nom.arm,
                                                      produced=settled.produced, now=now,
                                                      note=f"question act {act.act_id}: {settled.note}")
            tail = f"; {concern_line}" if concern_line else ""
            self.state = note_outcome(self.state, f"my question {nom.question[:120]!r} (act {act.act_id}): {seen}{tail}")
        self.state = self.state.model_copy(update={"open_nominations": keep})
