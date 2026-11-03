"""Orion Prime's resident loop (rollout step 2A): orion-mind keeps thinking between conversations.

Design: docs/superpowers/specs/2026-10-29-orion-prime-design.md, sections 7.2 and 8. Pure logic is in
orion/prime/; this module does the I/O.

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
import threading
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
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
    TryKind,
    classify_try,
    new_id,
    note_model_reached,
    note_no_model,
    note_outcome,
    observe,
    trim_record,
    wants_starved_route,
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
from orion.self_monitor.adapters import act_faculty
from orion.self_monitor.gate import FAILING_STATUSES, ActGate, GateDecision, last_attempt_sql, read_view_sync
from orion.self_monitor.view import TRACK_RECORD_FAILING_REASON

logger = logging.getLogger("orion-mind.prime")

STATE_KEY = "orion:prime:state"
FAMILY = "prime"
PRODUCER = "orion-mind.prime"
HTTP_TIMEOUT_SEC = 4.0
FOCUS_RPC_TIMEOUT_SEC = 10.0
CURIOSITY_RUNS_DAYS = 4  # covers a nomination's whole horizon (48 h focus TTL + 24 h run)
# A regulate act is graded this long after its signal ends: the market's last bid under it has landed.
REGULATE_SETTLE_LAG_SEC = 120.0
# The no-model backoff's base never drops below this, even with PRIME_REFRACTORY_SEC=0 (no tick-rate loop).
NO_MODEL_BACKOFF_FLOOR_SEC = 60.0


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


# A refused concern bet is asked again at most this often (the tick is 15 s; the answer changes at most
# hourly, when the self-monitor publishes).
GATE_RECHECK_SEC = 60.0
_LEDGER_CONNECT_TIMEOUT_SEC = 3
_LEDGER_STATEMENT_TIMEOUT_MS = 5000


def _ledger_rows(dsn: str, actor: str, verb: str, secs: float) -> list[dict[str, Any]]:
    """grammar_acts rows for the gate's probe and same-job bookkeeping. Raises on failure; the gate
    then lets its in-process memo decide (never "no attempt")."""
    import psycopg2  # inside: a missing driver degrades to an unreadable ledger
    import psycopg2.extras

    conn = psycopg2.connect(dsn, connect_timeout=_LEDGER_CONNECT_TIMEOUT_SEC,
                            options=f"-c statement_timeout={_LEDGER_STATEMENT_TIMEOUT_MS}")
    try:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(last_attempt_sql("pyformat"), {"actor": actor, "verb": verb, "secs": float(secs)})
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def build_act_gate(settings: Any) -> ActGate:
    """Prime's E1 gate. No DSN: the ledger is unreadable and the memo alone decides. A concern act's job
    is its concern (``object_ref prime_concern:<id>``): a retry of a started open is the same job."""
    dsn = str(getattr(settings, "SELF_MONITOR_LEDGER_PG_DSN", "") or "").strip()
    return ActGate(
        enabled=bool(getattr(settings, "SELF_MONITOR_ACT_GATE_ENABLED", False)),
        probe_interval_sec=float(getattr(settings, "SELF_MONITOR_PROBE_INTERVAL_SEC", 21600.0)),
        read_view=lambda now: read_view_sync(str(getattr(settings, "ORION_BUS_URL", "") or ""), now=now),
        ledger_rows=lambda actor, verb, secs: _ledger_rows(dsn, actor, verb, secs) if dsn else None,
        job_key_of=lambda row: row.get("object_ref"),
    )


def _gate_why(d: GateDecision, interval_sec: float) -> str:
    return (f"my own record says these have been {d.status} lately (self-monitor: track_record_failing); "
            f"one is re-tested every {interval_sec / 3600:g} h")


def _with_outcome_line(state: PrimeStateV1, text: str) -> PrimeStateV1:
    """Show a line in the next frame WITHOUT making it a change (a refusal Prime caused itself must
    not wake it again, or a refused move would loop)."""
    return state.model_copy(update={"recent_outcomes": [*state.recent_outcomes, text][-MAX_RECENT_OUTCOMES:]})


@dataclass
class _Reply:
    """One model call's result, and whether it reached the model (orion/prime/state.py classify_try)."""

    route: str
    parsed: Optional[dict[str, Any]]
    err: Optional[str]
    gateway_error: Optional[dict[str, Any]]
    kind: TryKind
    pool_wait_sec: Optional[float] = None
    started_mono: float = 0.0
    curiosity: Optional[dict[str, Any]] = None
    receptors: Optional[dict[str, tuple[str, ...]]] = None


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
        self.gate = build_act_gate(settings)
        # concern_id -> when its bet was last refused (re-check throttle; also: never a late bet for it).
        self._bet_refused_at: dict[str, datetime] = {}
        # The offered infer probe, spent only when the model is reached (_open_thought).
        self._infer_probe: Optional[str] = None
        self._infer_refusal_told = False
        self._infer_refused_at: Optional[datetime] = None
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

    # --- acts ------------------------------------------------------------------------------
    async def _publish(self, event: Any) -> bool:
        try:
            await self.bus.publish(GRAMMAR_EVENT_CHANNEL, BaseEnvelope(
                kind="grammar.event.v1", source=self.source, payload=event.model_dump(mode="json")))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("prime_act_publish_failed act=%s err=%s", event.act.act_id if event.act else "-",
                           type(exc).__name__)
            return False

    async def _open(self, act: ActV1, component: str, *, at: Optional[datetime] = None) -> bool:
        return await self._publish(act_opened_event(act, source_service=SOURCE_SERVICE, source_component=component,
                                                    at=at))

    async def _close(self, act: ActV1, component: str, *, produced: bool, note: str,
                     at: Optional[datetime] = None) -> bool:
        return await self._publish(act_closed_event(act, produced=produced, source_service=SOURCE_SERVICE,
                                                    source_component=component, note=note, at=at))

    # --- self-monitor act gate (E1) ---------------------------------------------------------
    async def _gate_decide(self, verb: str, object_ref: str, *, job_key: Optional[str] = None,
                           recheck_soon: bool = False) -> Optional[GateDecision]:
        """The gate's answer, or None (gate off, or a gate fault: one direction only, nothing changes).
        Runs in a worker thread: the view read is Redis, the ledger read Postgres. ``recheck_soon``: this
        act was refused under GATE_RECHECK_SEC ago; while the fresh view still holds the faculty failing
        it stays refused without a ledger read. The view is always read, so an unknown, stale or healed
        view admits at once, exactly like the gate being off."""
        if not self.gate.enabled:
            return None
        now = self.clock()

        def run() -> GateDecision:
            view = self.gate.begin(now)
            faculty = act_faculty(ACTOR, verb, object_ref)
            if recheck_soon and view.known and view.status(faculty) in FAILING_STATUSES:
                return GateDecision(faculty, False, False, view.status(faculty), TRACK_RECORD_FAILING_REASON)
            return self.gate.decide(ACTOR, verb, object_ref, now=now, job_key=job_key)

        try:
            return await asyncio.to_thread(run)
        except Exception:  # noqa: BLE001
            logger.warning("prime_self_monitor_gate_failed verb=%s", verb, exc_info=True)
            return None

    def _admit(self, d: Optional[GateDecision], *, job_key: Optional[str] = None) -> bool:
        """Spend an offered probe now (the act is about to open). Prime is one loop, so a claim only
        fails if something else already started this window's probe: then the act is refused."""
        if d is None or not d.admit:
            return d is None
        if d.probe:
            return self.gate.claim_probe(d.faculty, now=self.clock(), job_key=job_key)
        self.gate.note_job_started(d.faculty, job_key, now=self.clock())
        return True

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
                                                        f"was not placed: {_gate_why(d, self.gate.probe_interval_sec)}")
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
        return _gate_why(d, self.gate.probe_interval_sec).replace("these", f"my {what}", 1)

    async def _thought_gate_admits(self, now: datetime) -> bool:
        """Before the bid. A probe is only offered here; it is spent when the model is reached. A refusal
        is re-checked against the ledger at most every GATE_RECHECK_SEC (the view every tick)."""
        self._infer_probe = None
        last = self._infer_refused_at
        d = await self._gate_decide("infer", "prime_thought:", recheck_soon=(
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
        logger.info("prime_act_refused verb=infer gate=%s pending=%d", d.label(), len(self.state.pending_changes))
        if not self._infer_refusal_told:
            self._infer_refusal_told = True
            self.state = _with_outcome_line(self.state, f"at {now:%H:%M} UTC I did not start a thought: "
                                                        f"{_gate_why(d, self.gate.probe_interval_sec)}")
        return False

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

    # --- the tick --------------------------------------------------------------------------
    async def tick(self) -> None:
        snap, inputs = await self._read_body_and_attention()
        now = self.clock()
        self.state = lapse_concerns(observe(self.state, snap, now), now)
        await self._maybe_settle(now)
        await self._settle_regulations(now)
        await self._flush_concern_acts()
        await self.save()
        if self.turns.active():
            logger.info("prime_yield reason=juniper_turn pending=%d", len(self.state.pending_changes))
            return
        if not wants_to_think(self.state, now, refractory=timedelta(seconds=float(self.s.PRIME_REFRACTORY_SEC))):
            return
        if not await self._thought_gate_admits(now):
            await self.save()
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
            await self._flush_concern_acts()
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
    def _route(self) -> str:
        if wants_starved_route(self.state):
            return str(getattr(self.s, "PRIME_STARVED_LLM_ROUTE", "") or self.s.PRIME_LLM_ROUTE)
        return self.s.PRIME_LLM_ROUTE

    def _seat_wait_sec(self) -> float:
        return float(getattr(self.s, "PRIME_SEAT_WAIT_SEC", 0.0) or 0.0)

    async def think(self, inputs: FrameInputs) -> None:
        now = self.clock()
        thought_id = new_id()
        # The act's horizon covers the whole try: the seat wait plus the generation budget.
        act = thought_act(thought_id, llm_timeout_sec=self._seat_wait_sec() + float(self.s.PRIME_LLM_TIMEOUT_SEC))
        # Captured before anything can clear pending changes: every close of this act says why it woke.
        woke = woke_note(list(self.state.pending_changes))
        try:
            reply = await self._ask(inputs, now)
        except Exception as exc:  # noqa: BLE001 -- a crash is a failed thought (refractory holds: no hot loop)
            logger.exception("prime_thought_crashed thought=%s", thought_id)
            await self._open_thought(act, now)
            await self._close(act, THOUGHT_COMPONENT, produced=False,
                              note=f"{woke} | failed: {type(exc).__name__}: {str(exc)[:160]}")
            return
        if reply.kind != "model":
            self._note_no_model(reply, self.clock())   # the backoff counts from the end of the try
            return
        await self._open_thought(act, now)
        try:
            produced, note = await self._conclude(act, thought_id, inputs, now, reply)
        except Exception as exc:  # noqa: BLE001 -- the act always closes
            logger.exception("prime_thought_crashed thought=%s", thought_id)
            produced, note = False, f"failed: {type(exc).__name__}: {str(exc)[:160]}"
        await self._close(act, THOUGHT_COMPONENT, produced=produced, note=f"{woke} | {note}")

    async def _open_thought(self, act: ActV1, started_at: datetime) -> None:
        """The model was reached: the thought happened. Opened stamped with the try's start; counts for
        the refractory interval from then (a failed or preempted thought must not retry every tick)."""
        if self._infer_probe is not None:
            # The model was reached: the offered probe is a real attempt now. Prime is one loop, so the
            # claim cannot lose a race; the thought already happened either way.
            self.gate.claim_probe(self._infer_probe, now=self.clock())
            self._infer_probe = None
        await self._open(act, THOUGHT_COMPONENT, at=started_at)
        self.state = note_model_reached(self.state).model_copy(update={"last_thought_at": started_at})

    def _note_no_model(self, reply: "_Reply", now: datetime) -> None:
        gw = reply.gateway_error or {}
        first = self.state.no_model_streak == 0
        self.state = note_no_model(self.state, reply.kind, now,
                                   base=timedelta(seconds=max(float(self.s.PRIME_REFRACTORY_SEC),
                                                              NO_MODEL_BACKOFF_FLOOR_SEC)),
                                   retry_after_sec=gw.get("retry_after_sec"))
        if first:
            # Once per streak, so the next real thought knows its last try was starved (not a change:
            # it must not wake Prime by itself).
            self.state = _with_outcome_line(
                self.state, f"a try to think at {now:%H:%M} UTC never reached the model ({reply.kind}"
                            f"{': ' + str(gw.get('reason')) if gw.get('reason') else ''}); I tried again")
        logger.info("prime_try_no_model kind=%s route=%s reason=%s err=%s streak=%d retry_not_before=%s pending=%d",
                    reply.kind, reply.route, gw.get("reason") or "-", (reply.err or "-")[:160],
                    self.state.no_model_streak,
                    self.state.retry_not_before.isoformat() if self.state.retry_not_before else "-",
                    len(self.state.pending_changes))

    async def _ask(self, inputs: FrameInputs, now: datetime) -> "_Reply":
        """Build the frame and call the model; one immediate retry on an upstream 5xx."""
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
        route = self._route()
        started = time.monotonic()
        reply = await self._call(route, system_prompt(receptors), frame)
        if reply.kind == "upstream_5xx":
            logger.info("prime_upstream_5xx_retry route=%s err=%s", route, (reply.err or "-")[:160])
            started = time.monotonic()
            reply = await self._call(route, system_prompt(receptors), frame)
        # Juniper-first preemption counts from the GRANT: a Juniper turn during the seat wait went ahead
        # of Prime in the pool and is over or running beside it, not overlapped by generation. When the
        # gateway did not report its wait, count from the start (the conservative old rule).
        reply.started_mono = started + (reply.pool_wait_sec or 0.0)
        reply.curiosity, reply.receptors = curiosity, receptors
        return reply

    async def _call(self, route: str, system: str, frame: str) -> "_Reply":
        seat = self._seat_wait_sec()
        try:
            parsed, err, meta = await asyncio.to_thread(
                self.llm.request_json,
                system_prompt=system, user_prompt=frame, route=route,
                max_tokens=int(self.s.PRIME_MAX_TOKENS), temperature=0.4, thinking=False,
                timeout_sec=float(self.s.PRIME_LLM_TIMEOUT_SEC), pool_wait_sec=seat if seat > 0 else None,
            )
        except Exception as exc:  # noqa: BLE001
            parsed, err, meta = None, f"{type(exc).__name__}: {str(exc)[:160]}", {}
        gw = (meta or {}).get("gateway_error") if isinstance(meta, dict) else None
        gw = gw if isinstance(gw, dict) else None
        kind = "model" if parsed is not None and not err else classify_try(err, gw)
        wait_ms = (meta or {}).get("pool_wait_ms") if isinstance(meta, dict) else None
        return _Reply(route=route, parsed=parsed, err=err, gateway_error=gw, kind=kind,
                      pool_wait_sec=float(wait_ms) / 1000.0 if isinstance(wait_ms, (int, float)) else None)

    async def _conclude(self, act: ActV1, thought_id: str, inputs: FrameInputs, now: datetime,
                        reply: "_Reply") -> tuple[bool, str]:
        if self.turns.started_since(reply.started_mono):
            logger.info("prime_thought_preempted thought=%s", thought_id)
            return False, "preempted: a Juniper turn started"
        answer, perr = (None, reply.err) if reply.err else parse_answer(reply.parsed)
        if answer is None:
            logger.info("prime_thought_failed thought=%s err=%s", thought_id, perr)
            return False, f"failed: {perr}"
        curiosity, receptors = reply.curiosity, reply.receptors
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
