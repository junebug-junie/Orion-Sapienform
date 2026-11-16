"""Family C Experiment 1: randomly hold Orion's AI Town session starts for 40 minutes, and measure the cost.

Design: docs/superpowers/specs/2026-11-16-family-c-first-experiment-economy-native.md (Experiment 1).
Juniper's yes: #3482 comment, 2026-11-16 (randomised 40-min holds, never more than 2 in a row, ~16 nights;
a new tier-2 family ``prediction.experiment`` whose bids bind; tier-1 slowdown on ``fast`` as an outcome).

What it does, in plain words. When a town session is about to be allowed to start, the runner takes the
next arm from a random list drawn for the night. ON: nothing happens, the town starts as it would today.
OFF: the runner tells the town controller "no new session for 40 minutes" (Redis key
``orion:physiology:run_cap:aitown.session``, ``RunCapOverrideV1``, TTL 90 s, refreshed every 30 s,
``set_at`` fixed at the block's start). Running sessions are never stopped; the daily cap is unchanged.

Never starts itself. A run starts only by ``POST /experiments/town-hold/run`` (operator action). The run's
schedule (first night, number of nights, seed) is kept in Redis (``SCHEDULE_KEY``) so a rebuild of this
service costs one night, not the run: at boot ``recover()`` cancels the half-done night and
``TownHoldRunner.resume()`` continues the SAME explicitly started run from the next night. ``POST
/experiments/town-hold/abort`` ends the night and deletes the schedule.

Per night (00:00-08:00 America/Denver), one experiment row (``design.kind = event_switchback``):
  * the seeded arm list: balanced, re-drawn until it has no 3 OFF in a row (``causal.draw_arm_sequence``);
  * eligibility comes from the town itself (embodiment ``/health`` ``start_eligible_at``), never from env;
  * 60 s before eligibility a *pending* hold is set, then /health is read AGAIN: a start that read the key
    before it was set shows as ``start_in_flight`` or an open session, and the decision is void (the race);
  * at eligibility, the stop rules (tier-0/1 work queued on ``fast`` > 5 s, cabinet hot or missing, strain
    latched or unknown) and the experiment's own BINDING bid (``prediction.experiment``): anything but a
    funded bid refuses the decision (no arm taken, the key is deleted, the town starts normally);
  * the arm is logged (``prediction_intervention``) BEFORE the key changes. ON deletes the key; OFF keeps
    it as the block's hold for 40 min (``set_at`` = the block start);
  * a session that starts inside an OFF block (seen in /health or in the act ledger) aborts the night as
    contaminated; a manual (Juniper) session in a block's window is flagged and left out of the primary
    analysis (a ``block_check`` row once the window is over).
The nights are analysed pooled (``family_c.event_run_rows``): night = stratum, restricted permutation.

Reads only sensors, the pool and the town's own gate. It never reads Orion's experience (``GET
/temporal-self/now``): an arm that depended on Orion's state would no longer be randomised
(``config/integration/organs.yaml``, the prediction organ's ``exempt_read``).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional
from zoneinfo import ZoneInfo

import numpy as np

from app.causal import CausalError, draw_arm_sequence, max_off_run
from app.experiment_runner import ACTIVE_KEY, LOCK_KEY, LUA_REFRESH, LUA_RELEASE, RUNNER_SOURCE, TOWN_HOLD_SCHEDULE_KEY
from app.family_c import EVENT_KIND, LANE_TARGETS

logger = logging.getLogger("orion-prediction.town_hold")

ACTION = "run_cap:aitown.session"
FAMILY = "aitown.session"
KNOB = "hold_starts"
EXP_FAMILY = "prediction.experiment"
PRODUCER = "orion-prediction.experiment.binding"
SCHEDULE_KEY = TOWN_HOLD_SCHEDULE_KEY
PRIMARY_TARGET = "circe.gpu3_watts"
LANE_PRIMARY = "lane.fast.tier1_hold_s"
CABINET_SERIES = "athena.cabinet_temp_c"
TIER01_PRIORITIES = ("urgent", "interactive", "perception", "system")
ESTIMATOR_RUN = "family_c_pooled_run"
READING_MAX_AGE = timedelta(minutes=10)


def run_cap_key() -> str:
    from orion.schemas.physiology_run_cap import run_cap_key as key

    return key(FAMILY)


@dataclass(frozen=True)
class TownHoldConfig:
    hold_min: int = 40
    window_min: tuple[int, int] = (2, 35)      # outcome window after the block start
    night_start_hour: int = 0
    night_end_hour: int = 8                     # decisions only in [00:00, 08:00) local
    tz: str = "America/Denver"
    pending_lead_sec: float = 60.0
    arms_per_night: int = 8                     # ~7 decisions fit a night; the list is never exhausted early
    max_off_run: int = 2                        # the play floor
    queued_wait_sec: float = 5.0
    quiet_sec: float = 600.0                    # after a decision or refusal, let the town act before re-deciding
    key_ttl_sec: int = 90
    tick_sec: float = 30.0
    regulation_max_age_sec: float = 600.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["window_min"] = list(self.window_min)
        return d


class _Abort(Exception):
    pass


class TownHoldRefused(Exception):
    def __init__(self, status: int, detail: Any):
        super().__init__(str(detail))
        self.status, self.detail = status, detail


# --- pure helpers ------------------------------------------------------------------------------

def night_window(day: date, cfg: TownHoldConfig) -> tuple[datetime, datetime]:
    tz = ZoneInfo(cfg.tz)
    w0 = datetime.combine(day, time(cfg.night_start_hour), tzinfo=tz)
    w1 = datetime.combine(day, time(cfg.night_end_hour), tzinfo=tz)
    return w0.astimezone(timezone.utc), w1.astimezone(timezone.utc)


def night_seed(run_seed: int, day: date) -> int:
    return int.from_bytes(hashlib.sha256(f"{run_seed}:{day.isoformat()}".encode()).digest()[:4], "big") % (2 ** 31)


def plan_night(run_seed: int, day: date, cfg: TownHoldConfig) -> dict:
    seed = night_seed(run_seed, day)
    arms = draw_arm_sequence(cfg.arms_per_night, cfg.max_off_run, np.random.default_rng(seed))
    w0, w1 = night_window(day, cfg)
    return {"night": day.isoformat(), "seed": seed, "arms": arms, "window_start": w0.isoformat(),
            "window_end": w1.isoformat()}


def first_night_after(now: datetime, cfg: TownHoldConfig) -> date:
    """The next local night whose window has not started yet (a run never begins mid-night by default)."""
    local = now.astimezone(ZoneInfo(cfg.tz)).date()
    for d in (local, local + timedelta(days=1)):
        if night_window(d, cfg)[0] > now:
            return d
    return local + timedelta(days=2)


def _parse_iso(v: Any) -> Optional[datetime]:
    if not v:
        return None
    try:
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _as_dt(v: Any) -> datetime:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    dt = _parse_iso(v)
    if dt is None:
        raise ValueError(f"not a timestamp: {v!r}")
    return dt


@dataclass
class TownView:
    """The town's own answer, from embodiment /health ``town_self_start``."""
    eligible_at: Optional[datetime]
    session_open: bool
    manual_session_open: bool
    start_in_flight: bool
    override_configured: bool
    running: bool

    @classmethod
    def parse(cls, block: dict) -> "TownView":
        rc = block.get("run_cap_override") or {}
        return cls(eligible_at=_parse_iso(block.get("start_eligible_at")),
                   session_open=bool(block.get("session_open")),
                   manual_session_open=bool(block.get("manual_session_open")),
                   start_in_flight=bool(block.get("start_in_flight")),
                   override_configured=bool(rc.get("configured")),
                   running=bool(block.get("running")))


class EmbodimentProbe:
    """GET <embodiment>/health -> ``TownView``. None on any failure (the caller decides; never fatal)."""

    def __init__(self, url: str, timeout: float = 3.0):
        self.url, self.timeout = url.rstrip("/"), timeout

    def view(self) -> Optional[TownView]:
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=self.timeout) as r:   # noqa: S310 internal
                body = json.loads(r.read())
            return TownView.parse(body["town_self_start"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("town_hold_embodiment_probe_failed %s: %s", type(exc).__name__, exc)
            return None


def default_bid(want: Any) -> Any:
    from orion.physiology.client import bid

    return bid(want)


def default_release(want_id: str) -> Any:
    from orion.physiology.client import release

    return release(want_id)


@dataclass
class NightState:
    experiment_id: str
    night: str
    w0: datetime
    w1: datetime
    arms: list[str]
    seed: int
    k: int = 0
    taken: list[str] = field(default_factory=list)
    pending: Optional[dict] = None              # {set_at, idx, verified}
    off: Optional[dict] = None                  # {idx, start, end}
    quiet_until: Optional[datetime] = None
    quiet_marker: Optional[datetime] = None
    last_decision_at: Optional[datetime] = None
    checks: list[dict] = field(default_factory=list)   # {due, idx, arm, start, want_id}
    refusals: dict = field(default_factory=dict)
    voided: int = 0
    status: str = "running"
    abort_reason: Optional[str] = None

    def public(self) -> dict:
        d = asdict(self)
        for k in ("w0", "w1", "quiet_until", "quiet_marker", "last_decision_at"):
            d[k] = d[k].isoformat() if d[k] else None
        d["checks"] = [{**c, "due": c["due"].isoformat(), "start": c["start"].isoformat()} for c in self.checks]
        if self.pending:
            d["pending"] = {**self.pending, "set_at": self.pending["set_at"].isoformat()}
        if self.off:
            d["off"] = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in self.off.items()}
        return d


class TownHoldRunner:
    def __init__(self, effects: Any, redis: Any, probe: Any, *, cfg: TownHoldConfig | None = None,
                 bid: Callable[[Any], Any] = default_bid, release: Callable[[str], Any] = default_release,
                 on_run_complete: Callable[[str], Any] | None = None, vision_runner: Any = None,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 to_thread: Callable[..., Awaitable[Any]] | None = None, lock_ttl_sec: float = 180.0):
        self.cfg = cfg or TownHoldConfig()
        if self.cfg.key_ttl_sec < 2 * self.cfg.tick_sec:
            raise ValueError("key TTL must be at least two ticks, or the hold lapses mid-block")
        self.effects, self.redis, self.probe = effects, redis, probe
        self._bid, self._release_want = bid, release
        self.on_run_complete = on_run_complete
        self.vision_runner = vision_runner
        self._sleep = sleep
        self._to_thread = to_thread or asyncio.to_thread
        self.lock_ttl = int(max(lock_ttl_sec, 2 * self.cfg.tick_sec))
        self._task: asyncio.Task | None = None
        self._cancel_reason: str | None = None
        self.schedule: dict | None = None
        self.night: NightState | None = None
        self.nights_done: list[dict] = []
        self.result: dict | None = None

    @property
    def store(self):
        return self.effects.store

    def clock(self) -> datetime:
        return self.effects.engine.clock()

    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def status(self) -> dict:
        return {"running": self.running(), "schedule": self.schedule,
                "night": self.night.public() if self.night else None,
                "nights_done": self.nights_done[-20:], "result": self.result, "config": self.cfg.as_dict()}

    # --- start / resume / stop -----------------------------------------------------------------
    async def start(self, nights: int, first_night: date | None = None, seed: int | None = None) -> dict:
        """The operator's explicit action. Refuses while any run (this or a vision.focus one) is active."""
        if self.running():
            raise TownHoldRefused(409, "a town-hold run is already scheduled")
        if self.vision_runner is not None and self.vision_runner.status().get("running"):
            raise TownHoldRefused(409, "a vision.focus experiment is running")
        if await self.redis.get(SCHEDULE_KEY):
            raise TownHoldRefused(409, f"{SCHEDULE_KEY} names a run; restart the service to resume it or abort it")
        if await self.redis.get(ACTIVE_KEY) or await self.redis.get(LOCK_KEY):
            raise TownHoldRefused(409, "another experiment holds the runner lock")
        view = await self._to_thread(self.probe.view)
        if view is None:
            raise TownHoldRefused(503, "embodiment /health is unreadable; cannot see the town's start gate")
        if not view.override_configured:
            raise TownHoldRefused(422, "embodiment does not obey the run-cap override for aitown.session "
                                       "(config/physiology/run_cap_overrides.yaml missing in its image?)")
        if not view.running:
            raise TownHoldRefused(422, "embodiment's town self-start timer is not running; nothing to hold")
        now = self.clock()
        first = first_night or first_night_after(now, self.cfg)
        if night_window(first, self.cfg)[1] <= now:
            raise TownHoldRefused(422, f"night {first} is already over")
        run_seed = int(seed if seed is not None else uuid.uuid4().int % (2 ** 31))
        sched = {"run_id": str(uuid.uuid4()), "action": ACTION, "first_night": first.isoformat(), "nights": int(nights),
                 "seed": run_seed, "created_at": now.isoformat(), "config": self.cfg.as_dict()}
        await self.redis.set(SCHEDULE_KEY, json.dumps(sched))   # no TTL: survives a restart, deleted at the end
        logger.warning("town_hold_scheduled run_id=%s first_night=%s nights=%d seed=%d", sched["run_id"], first,
                       nights, run_seed)
        self._launch(sched)
        return {"schedule": sched, "first_night_plan": plan_night(run_seed, first, self.cfg)}

    async def resume(self) -> bool:
        """At boot, after ``recover()``: continue an explicitly started run from its next night."""
        raw = await self._safe(self.redis.get(SCHEDULE_KEY))
        if not raw or self.running():
            return False
        try:
            sched = json.loads(raw)
            sched["run_id"], sched["first_night"], sched["nights"], sched["seed"]   # noqa: B018 -- shape check
        except Exception:  # noqa: BLE001
            logger.warning("town_hold_schedule_unreadable; deleting it")
            await self._safe(self.redis.delete(SCHEDULE_KEY))
            return False
        logger.warning("town_hold_resumed run_id=%s", sched["run_id"])
        self._launch(sched)
        return True

    def _launch(self, sched: dict) -> None:
        self.schedule, self._cancel_reason, self.result = sched, None, None
        self._task = asyncio.create_task(self._run_schedule(sched))

    async def shutdown(self, reason: str, *, clear_schedule: bool) -> bool:
        """Operator abort (``clear_schedule=True``) or service stop (False: the run resumes at boot)."""
        task = self._task
        if clear_schedule:
            await self._safe(self.redis.delete(SCHEDULE_KEY))
        if task is None or task.done():
            return False
        self._cancel_reason = reason
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        if clear_schedule:
            self.schedule = None
        return True

    # --- the schedule ----------------------------------------------------------------------------
    def nights_of(self, sched: dict) -> list[date]:
        first = date.fromisoformat(sched["first_night"])
        return [first + timedelta(days=i) for i in range(int(sched["nights"]))]

    def _existing_nights(self, run_id: str) -> set[str]:
        return {str(e["design"].get("night")) for e in self.store.experiments(limit=500)
                if (e.get("design") or {}).get("kind") == EVENT_KIND and e["design"].get("run_id") == run_id}

    async def _run_schedule(self, sched: dict) -> None:
        try:
            for day in self.nights_of(sched):
                done = await self._to_thread(self._existing_nights, sched["run_id"])
                w0, w1 = night_window(day, self.cfg)
                if day.isoformat() in done or self.clock() >= w1:
                    continue   # already run (or cancelled by a restart), or over: one night lost, not the run
                await self._wait_until(w0)
                await self._run_night(sched, day)
            await self._safe(self.redis.delete(SCHEDULE_KEY))
            if self.on_run_complete is not None:
                try:
                    self.result = await self._to_thread(self.on_run_complete, sched["run_id"])
                except Exception as exc:  # noqa: BLE001 -- the nights are stored; POST .../result retries
                    self.result = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            logger.warning("town_hold_run_complete run_id=%s", sched["run_id"])
        except asyncio.CancelledError:
            raise

    async def _wait_until(self, until: datetime) -> None:
        while True:
            left = (until - self.clock()).total_seconds()
            if left <= 0:
                return
            await self._sleep(min(left, 300.0))

    # --- one night -------------------------------------------------------------------------------
    async def _run_night(self, sched: dict, day: date) -> None:
        plan = plan_night(int(sched["seed"]), day, self.cfg)
        now = self.clock()
        design = {"kind": EVENT_KIND, "run_id": sched["run_id"], "night": plan["night"], "seed": plan["seed"],
                  "run_seed": int(sched["seed"]), "arms": plan["arms"], "n_blocks": len(plan["arms"]),
                  "max_off_run": self.cfg.max_off_run, "hold_min": self.cfg.hold_min,
                  "window_min": list(self.cfg.window_min), "block_min": self.cfg.window_min[1],
                  "window_start": plan["window_start"], "window_end": plan["window_end"],
                  "outcomes": [PRIMARY_TARGET, LANE_PRIMARY, *LANE_TARGETS]}
        row = {"experiment_id": str(uuid.uuid4()), "action": ACTION, "series_id": PRIMARY_TARGET, "created_at": now,
               "burn_in_min": self.cfg.window_min[0], "design": design,
               "guard": {"stop_rules": ["tier0_1_queued_on_fast", "cabinet_hot_or_missing", "strain", "bid"]}}
        await self._to_thread(self.store.insert_experiment, row)
        eid = row["experiment_id"]
        w0, w1 = night_window(day, self.cfg)
        st = NightState(eid, plan["night"], w0, w1, list(plan["arms"]), plan["seed"])
        self.night = st
        logger.warning("town_hold_night_start id=%s night=%s seed=%d arms=%s", eid, st.night, st.seed, st.arms)
        if not await self.redis.set(LOCK_KEY, eid, nx=True, ex=self.lock_ttl):
            await self._abort(st, "another experiment holds the runner lock")
            return
        await self.redis.set(ACTIVE_KEY, eid)
        try:
            while True:
                state = await self.step(st)
                if state == "done":
                    break
                await self._sleep(self.cfg.tick_sec)
            await self._finish(st)
        except asyncio.CancelledError:
            await self._abort(st, self._cancel_reason or "runner cancelled")
            await self._release(st)
            raise
        except _Abort as exc:
            await self._abort(st, str(exc))
        except Exception as exc:  # noqa: BLE001 -- any failure ends the night safely, never leaves a hold
            logger.exception("town_hold_night_failed")
            await self._abort(st, f"runner error: {type(exc).__name__}: {exc}"[:300])
        await self._release(st)
        self.nights_done.append({"experiment_id": eid, "night": st.night, "status": st.status,
                                 "blocks": st.k, "abort_reason": st.abort_reason})

    async def _release(self, st: NightState) -> None:
        await self._safe(self.redis.delete(run_cap_key()))
        await self._safe(self.redis.eval(LUA_RELEASE, 1, LOCK_KEY, st.experiment_id))
        await self._safe(self.redis.delete(ACTIVE_KEY))

    async def _abort(self, st: NightState, reason: str) -> None:
        st.status, st.abort_reason = "cancelled", reason
        logger.warning("town_hold_night_aborted id=%s reason=%s", st.experiment_id, reason)
        await self._safe(self.redis.delete(run_cap_key()))
        now = self.clock()
        try:
            await self._to_thread(self.effects.log_intervention, ACTION, now - timedelta(seconds=1), now,
                                  RUNNER_SOURCE, json.dumps({"aborted": reason, "block_index": st.k,
                                                             "contaminated": reason.startswith("contaminated")}),
                                  None, st.experiment_id)
            await self._to_thread(self.store.set_experiment_status, st.experiment_id, "cancelled")
        except Exception as exc:  # noqa: BLE001 -- the key is cleared; the DB note is best effort
            logger.warning("town_hold_abort_not_recorded %s: %s", type(exc).__name__, exc)

    async def _finish(self, st: NightState) -> None:
        await self._safe(self.redis.delete(run_cap_key()))
        if st.k == 0:
            await self._abort(st, "no decision tonight (the town was never eligible, or every decision was refused)")
            return
        await self._to_thread(self.store.set_experiment_status, st.experiment_id, "analyzed")
        st.status = "analyzed"
        logger.warning("town_hold_night_done id=%s blocks=%d arms=%s refusals=%s", st.experiment_id, st.k,
                       st.taken, st.refusals)

    # --- the key -----------------------------------------------------------------------------------
    async def _write_key(self, st: NightState, idx: int, phase: str, set_at: datetime) -> None:
        from orion.schemas.physiology_run_cap import RunCapOverrideV1

        now = self.clock()
        ov = RunCapOverrideV1(family=FAMILY, knob=KNOB, value=True, experiment_id=st.experiment_id, block_index=idx,
                              phase=phase, set_at=set_at, expires_at=now + timedelta(seconds=self.cfg.key_ttl_sec),
                              writer=RUNNER_SOURCE)
        await self.redis.set(run_cap_key(), ov.model_dump_json(), ex=self.cfg.key_ttl_sec)

    async def _drop_pending(self, st: NightState, why: str) -> None:
        if st.pending is not None:
            await self.redis.delete(run_cap_key())
            logger.info("town_hold_pending_dropped id=%s idx=%s why=%s", st.experiment_id, st.pending["idx"], why)
            st.pending = None

    # --- one tick ------------------------------------------------------------------------------------
    async def step(self, st: NightState) -> str:
        """One tick. Returns ``done`` when the night is over and every block check has run."""
        if not await self.redis.eval(LUA_REFRESH, 1, LOCK_KEY, st.experiment_id, self.lock_ttl):
            raise _Abort("lost the runner lock (it expired, or another run took it)")
        view = await self._to_thread(self.probe.view)
        now = self.clock()
        for chk in [c for c in st.checks if c["due"] <= now]:
            await self._block_check(st, chk)
            st.checks.remove(chk)

        if st.off is not None:
            await self._write_key(st, st.off["idx"], "block", st.off["start"])
            if view is not None and view.session_open:
                raise _Abort(f"contaminated: an Orion session is open inside OFF block {st.off['idx']}")
            if now < st.off["end"]:
                return "holding"
            acts = await self._to_thread(self.store.town_acts, st.off["start"], st.off["end"])
            if any(st.off["start"] <= _as_dt(a["opened_at"]) < st.off["end"] for a in acts):
                raise _Abort(f"contaminated: an aitown.session act opened inside OFF block {st.off['idx']}")
            logger.info("town_hold_off_block_end id=%s idx=%d", st.experiment_id, st.off["idx"])
            # The key stays as the next decision's pending hold (no gap a start could slip through); it is
            # dropped below if no decision is due now.
            st.pending = {"set_at": now, "idx": st.k, "verified": True}
            await self._write_key(st, st.k, "pending", now)
            st.off = None

        if now >= st.w1:
            await self._drop_pending(st, "night_over")
            return "done" if not st.checks else "draining"
        if view is None:
            await self._drop_pending(st, "embodiment_unreadable")
            return "blind"
        if st.quiet_until is not None:
            moved = view.eligible_at is not None and st.quiet_marker is not None and view.eligible_at > st.quiet_marker
            if view.session_open or moved or now >= st.quiet_until:
                st.quiet_until = st.quiet_marker = None
            else:
                await self._drop_pending(st, "quiet")
                return "quiet"
        elig = view.eligible_at
        if st.k >= len(st.arms):
            await self._drop_pending(st, "arms_exhausted")
            return "idle"
        if elig is None or view.session_open or view.manual_session_open:
            await self._drop_pending(st, "not_eligible")
            return "idle"
        if elig >= st.w1:
            await self._drop_pending(st, "eligible_after_window")
            return "idle"
        if now < elig - timedelta(seconds=self.cfg.pending_lead_sec):
            await self._drop_pending(st, "not_yet")
            return "waiting"

        if st.pending is None:
            st.pending = {"set_at": now, "idx": st.k, "verified": False}
            await self._write_key(st, st.k, "pending", now)
            # The race check: read the town AFTER the key is set. A start that read the key before it was set is
            # in flight (or already open); this decision is void and the town keeps its session.
            after = await self._to_thread(self.probe.view)
            if after is None or after.session_open or after.start_in_flight or after.manual_session_open:
                st.voided += 1
                await self._drop_pending(st, "race_void")
                st.quiet_until, st.quiet_marker = now + timedelta(seconds=self.cfg.quiet_sec), now
                return "void"
            st.pending["verified"] = True
        else:
            await self._write_key(st, st.pending["idx"], "pending", st.pending["set_at"])
        if now < elig or not st.pending["verified"]:
            return "pending"
        return await self._decide(st, now)

    async def _decide(self, st: NightState, now: datetime) -> str:
        reasons = await self.stop_reasons(now)
        if reasons:
            return await self._refuse(st, now, reasons)
        want_id = f"{st.experiment_id}:{st.k}"
        decision = await self._to_thread(self._bid, self._want(st, want_id, now))
        outcome = getattr(decision, "outcome", None) if decision is not None else None
        if outcome != "funded":
            why = "no_answer" if decision is None else f"{outcome}:{getattr(decision, 'reason', None)}"
            return await self._refuse(st, now, [f"bid:{why}"], decision)
        idx, arm = st.k, st.arms[st.k]
        if arm == "off" and max_off_run(st.taken + ["off"]) > self.cfg.max_off_run:
            raise _Abort("play floor: the arm list would hold the town more than "
                         f"{self.cfg.max_off_run} times in a row")   # impossible by construction; never silent
        end = now + timedelta(minutes=self.cfg.hold_min)
        notes = {"block_index": idx, "seed": st.seed, "want_id": want_id,
                 "decision_id": getattr(decision, "decision_id", None), "phase": "decision"}
        # The arm is on the record BEFORE the key changes.
        await self._to_thread(self.effects.log_intervention, ACTION, now, end, RUNNER_SOURCE, json.dumps(notes), arm,
                              st.experiment_id)
        st.taken.append(arm)
        st.k += 1
        st.last_decision_at = now
        due = now + timedelta(minutes=max(self.cfg.hold_min, self.cfg.window_min[1]))
        st.checks.append({"due": due, "idx": idx, "arm": arm, "start": now, "want_id": want_id})
        st.pending = None
        if arm == "on":
            await self.redis.delete(run_cap_key())
            # No new decision inside this block's outcome window (no overlapping blocks), unless the town moved
            # on first (a session opened, or its gate moved past this decision).
            quiet = max(self.cfg.quiet_sec, 60.0 * self.cfg.window_min[1])
            st.quiet_until, st.quiet_marker = now + timedelta(seconds=quiet), now
        else:
            await self._write_key(st, idx, "block", now)
            st.off = {"idx": idx, "start": now, "end": end}
        logger.warning("town_hold_block id=%s idx=%d arm=%s", st.experiment_id, idx, arm)
        return f"decided:{arm}"

    async def _refuse(self, st: NightState, now: datetime, reasons: list[str], decision: Any = None) -> str:
        await self._drop_pending(st, "refused")
        for r in reasons:
            key = r if r.startswith("bid:") else r.split(":", 1)[0]
            st.refusals[key] = st.refusals.get(key, 0) + 1
        await self._to_thread(self.effects.log_intervention, ACTION, now, now + timedelta(seconds=1), RUNNER_SOURCE,
                              json.dumps({"refused": reasons, "decision_index": st.k,
                                          "decision_id": getattr(decision, "decision_id", None)}),
                              None, st.experiment_id)
        st.quiet_until, st.quiet_marker = now + timedelta(seconds=self.cfg.quiet_sec), now
        logger.warning("town_hold_decision_refused id=%s idx=%d reasons=%s", st.experiment_id, st.k, reasons)
        return "refused"

    def _want(self, st: NightState, want_id: str, now: datetime) -> Any:
        from orion.schemas.physiology_want import WantV1, WantWindowV1

        window = timedelta(minutes=self.cfg.hold_min)
        since = st.last_decision_at or st.w0
        opened = max(since, now - window)   # hunger since the last decision, at most one block old
        return WantV1(want_id=want_id, family=EXP_FAMILY, producer=PRODUCER, verb="experiment", tier=2,
                      object_ref=f"{ACTION}#{st.k}", opened_at=opened, correlation_id=st.experiment_id,
                      window=WantWindowV1(latest=now + window))

    async def _block_check(self, st: NightState, chk: dict) -> None:
        """Once a block's window is over: flag a manual session in it (session ledger), log a check row, and
        release the experiment's want."""
        from orion.embodiment.town_sessions import load_rows, row_end

        w0 = chk["start"] + timedelta(minutes=self.cfg.window_min[0])
        w1 = chk["start"] + timedelta(minutes=self.cfg.window_min[1])
        manual = None
        try:
            rows = await load_rows(self.redis)
            a, b = w0.timestamp(), w1.timestamp()
            now_s = self.clock().timestamp()
            manual = any(r.kind == "manual" and r.started_at < b and row_end(r, now_s) > a for r in rows)
        except Exception as exc:  # noqa: BLE001 -- unknown is recorded, and treated as manual (excluded)
            logger.warning("town_hold_ledger_unreadable %s: %s", type(exc).__name__, exc)
        notes = {"block_check": chk["idx"], "arm": chk["arm"], "manual_session": True if manual is None else manual,
                 "ledger_read": manual is not None}
        now = self.clock()
        await self._to_thread(self.effects.log_intervention, ACTION, now, now + timedelta(seconds=1), RUNNER_SOURCE,
                              json.dumps(notes), None, st.experiment_id)
        try:
            await self._to_thread(self._release_want, chk["want_id"])
        except Exception:  # noqa: BLE001 -- best effort
            pass

    async def stop_reasons(self, now: datetime) -> list[str]:
        """The stop rules, read at decision time. A failed read refuses (fail closed for data quality)."""
        out: list[str] = []
        try:
            n = await self._to_thread(self.store.queued_fast_leases, self.cfg.queued_wait_sec, TIER01_PRIORITIES)
            if n:
                out.append(f"tier0_1_queued_on_fast:{n}")
        except Exception as exc:  # noqa: BLE001
            out.append(f"lease_read_failed:{type(exc).__name__}")
        try:
            spec = self.effects._spec(CABINET_SERIES, "target")
            t = spec.threshold("experiment_guard")
            pts = await self._to_thread(self.store.points, spec, now - READING_MAX_AGE)
            vals = [v for ts, v in pts if v is not None and ts <= now]
            if t is None or not vals:
                out.append("cabinet_missing")
            elif float(vals[-1]) >= float(t.value):
                out.append(f"cabinet_hot:{float(vals[-1]):.2f}")
        except Exception as exc:  # noqa: BLE001
            out.append(f"cabinet_read_failed:{type(exc).__name__}")
        try:
            from orion.schemas.regulation import REGULATION_STATE_REDIS_KEY

            raw = await self.redis.get(REGULATION_STATE_REDIS_KEY)
            doc = json.loads(raw) if raw else None
            gen = _parse_iso((doc or {}).get("generated_at"))
            if doc is None or gen is None:
                out.append("strain_unknown")
            elif (now - gen).total_seconds() > self.cfg.regulation_max_age_sec:
                out.append("strain_stale")
            elif bool(((doc.get("arousal") or {}).get("strain_latched"))):
                out.append("strain_latched")
        except Exception as exc:  # noqa: BLE001
            out.append(f"strain_read_failed:{type(exc).__name__}")
        return out

    @staticmethod
    async def _safe(aw):
        try:
            return await aw
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001 -- cleanup must finish
            logger.warning("town_hold_cleanup_error %s: %s", type(exc).__name__, exc)
            return None

    # --- dry run ------------------------------------------------------------------------------------
    async def preview(self, day: date | None = None, seed: int | None = None) -> dict:
        """What a night would do, read-only: the arm list for ``seed`` and every stop rule read NOW, plus the
        town's own gate. Writes nothing, places no bid."""
        now = self.clock()
        d = day or first_night_after(now, self.cfg)
        s = seed if seed is not None else (int(self.schedule["seed"]) if self.schedule else 0)
        view = await self._to_thread(self.probe.view)
        return {"night": plan_night(s, d, self.cfg), "run_seed": s,
                "town": asdict(view) if view else None,
                "stop_rules_now": await self.stop_reasons(now), "now": now.isoformat()}


# --- the run's result (experience: TemporalSourceKind experiment_result) --------------------------

def run_result(effect_table: Any, store: Any, run_id: str) -> dict:
    """Pool the run's analysed nights, and store ONE effect estimate for the run (estimator
    ``family_c_pooled_run``): the primary GPU3 watts row, with the lane rows and the manipulation check in
    ``details``. orion-durable-runs turns it into Orion's experience (``experiment_result``)."""
    nights = [e for e in store.experiments(status="analyzed", limit=500)
              if (e.get("design") or {}).get("kind") == EVENT_KIND and e["design"].get("run_id") == run_id]
    if not nights:
        raise CausalError(f"run {run_id} has no analysed nights")
    rows, extra = effect_table.rows_for_run(nights)
    by = {r.target: r for r in rows}
    main = by.get(PRIMARY_TARGET)
    lane = by.get(LANE_PRIMARY)
    targets = {t: by[t].as_dict() for t in (PRIMARY_TARGET, "circe.gpu3_temp_c", "circe.chassis_watts",
                                            "athena.cabinet_temp_c", *LANE_TARGETS) if t in by}
    z = None
    if main is not None and main.estimate is not None and main.eff_se:
        z = main.estimate / main.eff_se
    last = max(nights, key=lambda e: str(e["design"].get("night")))

    def txt(r, unit, fmt):
        if r is None or r.estimate is None:
            return "unmeasured"
        ci = f" (95% {r.ci_lo:{fmt}}..{r.ci_hi:{fmt}})" if r.ci_lo is not None else ""
        return f"{r.estimate:{fmt}} {unit}{ci}"

    summary = (f"I held my AI Town sessions at random for {len(nights)} night(s) to see what they cost: "
               f"GPU3 {txt(main, 'W', '+.1f')}, tier-1 calls on fast {txt(lane, 's', '+.1f')} slower; "
               f"manipulation check {extra['manipulation']['verdict']}")
    details = {"estimator": ESTIMATOR_RUN, "run_id": run_id, "nights": sorted(str(e["design"]["night"]) for e in nights),
               "experiment_ids": sorted(str(e["experiment_id"]) for e in nights), "effect_z": z,
               "targets": targets, "manipulation": extra["manipulation"], "summary": summary}
    return store.insert_effect({"action": ACTION, "series_id": PRIMARY_TARGET, "claim_kind": "experimental_effect",
                                "experiment_id": str(last["experiment_id"]),
                                "effect": main.estimate if main else None, "ci_lo": main.ci_lo if main else None,
                                "ci_hi": main.ci_hi if main else None, "p_value": main.p_value if main else None,
                                "n": (main.n_on + main.n_off) if main else 0, "details": details})


__all__ = ["ACTION", "EXP_FAMILY", "PRODUCER", "SCHEDULE_KEY", "EmbodimentProbe", "TownHoldConfig",
           "TownHoldRefused", "TownHoldRunner", "TownView", "first_night_after", "night_window", "plan_night",
           "run_cap_key", "run_result"]
