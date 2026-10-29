"""Switchback experiment runner: Orion performs an action on a randomised schedule and measures it.

First (and only) actuator: ``vision.focus:<stream_id>``. During an ON block the runner holds Redis
key ``orion:vision:experiment:<stream_id> = on`` (with a short TTL it refreshes every tick); the
vision frame router reads it and forces that stream's triggered tier (more frames to the GPU,
captions on detections). During an OFF block the key is absent: normal operation. So the
experiment only ever ADDS focus; it never removes perception.

Lifecycle (POST /experiments/run):
  guard must pass -> take the runner lock -> design via Effects.design_experiment -> for each block:
  wait for its start, (ON) re-check the guard and set the key, keep the key + lock fresh, watch the
  cabinet reading and the guard, (ON) clear the key, log the block via Effects.log_intervention with
  arm + experiment_id and the router's dispatch counts -> analyze_experiment + manipulation check.

Safety:
  * Heat: any raw cabinet reading >= the experiment_guard line (thermal_gate.hot), a missing
    reading, or a failed guard check aborts: key cleared, abort row logged (arm NULL, so the
    analysis ignores it), experiment status -> cancelled.
  * Dead runner: the key's TTL (PREDICTION_EXPERIMENT_OVERRIDE_TTL_SEC) expires on its own; the
    router also ignores an "on" key that has no TTL.
  * Double run: one task per process plus a Redis lock (SET NX EX, refreshed every tick).
  * Restart: ``recover()`` at boot marks any experiment this runner left half-done as cancelled
    (with an abort row) and deletes its key. A run is never resumed or auto-started.

Spec: docs/superpowers/specs/2026-10-15-orion-prediction-service-design.md (switchback experiments).
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

import pandas as pd

from app.causal import CausalError, analyze_switchback
from app.frame import to_grain
from app.series import SeriesSpec, Source

logger = logging.getLogger("orion-prediction.experiment_runner")

RUNNER_SOURCE = "orion-prediction.experiment-runner"
LOCK_KEY = "orion:prediction:experiment:runner"
OVERRIDE_PREFIX = "orion:vision:experiment:"   # read by orion-vision-frame-router app/expectation.py
ACTION_RE = re.compile(r"^vision\.focus:([a-z0-9_-]{1,40})$")
READING_MAX_AGE = timedelta(minutes=10)
ALIGN_MIN = 5   # blocks start on a 5-min grain boundary so buckets never straddle two arms

# Manipulation check: did an ON block actually load the machines? GPU watts is not in the series
# registry (no forecast consumer), so it is read ad hoc from the same biometrics table.
GPU_SPEC = SeriesSpec(series_id="athena.gpu_watts_total", role="covariate",
                      source=Source(kind="biometrics", node="athena", key="gpu_watts_total"), unit="W",
                      grain_min=5, agg="mean", max_fill_min=15, consumers=("experiment-manipulation-check",))
REGISTRY_CHECKS = ("athena.chassis_watts", "circe.chassis_watts")   # captions run on circe's VL lane


class RunnerRefused(Exception):
    def __init__(self, status: int, detail: Any):
        super().__init__(str(detail))
        self.status, self.detail = status, detail


class _Abort(Exception):
    pass


def override_key(stream_id: str) -> str:
    return f"{OVERRIDE_PREFIX}{stream_id}"


def stream_of(action: str) -> str:
    m = ACTION_RE.match(action or "")
    if not m:
        raise CausalError(f"no actuator for {action!r}; the runner only performs vision.focus:<stream_id>")
    return m.group(1)


def _ceil_to(ts: datetime, minutes: int) -> datetime:
    step = minutes * 60
    return datetime.fromtimestamp(math.ceil(ts.timestamp() / step) * step, tz=timezone.utc)


def _ts(v: Any) -> datetime:
    return pd.Timestamp(v).to_pydatetime()


class RouterProbe:
    """GET <router>/healthz. Returns None on any failure: a missing probe is recorded, never fatal."""

    def __init__(self, url: str, timeout: float = 3.0):
        self.url, self.timeout = url.rstrip("/"), timeout

    def snapshot(self, stream_id: str) -> dict | None:
        try:
            with urllib.request.urlopen(f"{self.url}/healthz", timeout=self.timeout) as r:
                body = json.loads(r.read())
        except Exception as exc:  # noqa: BLE001
            logger.warning("router_probe_failed %s: %s", type(exc).__name__, exc)
            return None
        return {"tier_counts": dict((body.get("dispatch_tier_counts") or {}).get(stream_id) or {}),
                "override_active": stream_id in (body.get("experiment_override_streams") or []),
                "override_enabled": body.get("experiment_override_enabled")}


def _delta(a: dict | None, b: dict | None) -> dict | None:
    if a is None or b is None:
        return None
    ca, cb = a["tier_counts"], b["tier_counts"]
    return {k: cb.get(k, 0) - ca.get(k, 0) for k in set(ca) | set(cb) if cb.get(k, 0) - ca.get(k, 0)}


@dataclass
class RunState:
    experiment_id: str
    action: str
    stream_id: str
    series_id: str
    status: str = "running"        # running | completed | aborted
    block_index: int | None = None
    arm: str | None = None
    started_at: str | None = None
    abort_reason: str | None = None
    last_reading: float | None = None
    last_guard: dict | None = None
    blocks_logged: int = 0
    n_blocks: int = 0
    result: dict | None = None
    notes: list[str] = field(default_factory=list)


class ExperimentRunner:
    def __init__(self, effects: Any, redis: Any, probe: Any, *, override_ttl_sec: float = 90,
                 tick_sec: float = 30, guard_every_sec: float = 300, lock_ttl_sec: float = 180,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 to_thread: Callable[..., Awaitable[Any]] | None = None,
                 guard: Callable[[SeriesSpec], dict] | None = None):
        if override_ttl_sec < 2 * tick_sec:
            raise ValueError("override TTL must be at least two ticks, or the key lapses mid-block")
        self.effects, self.redis, self.probe = effects, redis, probe
        self.ttl, self.tick, self.guard_every = int(override_ttl_sec), float(tick_sec), float(guard_every_sec)
        self.lock_ttl = int(max(lock_ttl_sec, 2 * tick_sec))
        self._sleep = sleep
        self._to_thread = to_thread or asyncio.to_thread
        self._guard = guard or effects.guard
        self._task: asyncio.Task | None = None
        self._cancel_reason: str | None = None
        self.state: RunState | None = None

    @property
    def clock(self):
        return self.effects.engine.clock

    def status(self) -> dict:
        running = self._task is not None and not self._task.done()
        return {"running": running, **(asdict(self.state) if self.state else {})}

    # --- start ---------------------------------------------------------------------------------
    async def start(self, action: str, series_id: str, block_min: int, n_blocks: int, burn_in_min: int,
                    seed: int | None = None, start_delay_min: int = 2) -> dict:
        if self._task is not None and not self._task.done():
            raise RunnerRefused(409, f"experiment {self.state.experiment_id} is already running")
        stream = stream_of(action)
        spec = self.effects._spec(series_id, "target")
        t = spec.threshold("experiment_guard")
        if t is None:
            raise RunnerRefused(422, f"{series_id} declares no experiment_guard threshold")
        guard = await self._to_thread(self._guard, spec)
        if not guard.get("safe"):
            raise RunnerRefused(409, {"error": "guard refused", "guard": guard})
        if not await self.redis.set(LOCK_KEY, "starting", nx=True, ex=self.lock_ttl):
            holder = await self.redis.get(LOCK_KEY)
            raise RunnerRefused(409, f"another runner holds {LOCK_KEY} ({holder!r})")
        try:
            start_at = _ceil_to(self.clock() + timedelta(minutes=start_delay_min), ALIGN_MIN)
            row = await self._to_thread(self.effects.design_experiment, action, series_id, start_at, block_min,
                                        n_blocks, burn_in_min, seed)
            await self.redis.set(LOCK_KEY, row["experiment_id"], xx=True, ex=self.lock_ttl)
        except BaseException:
            await self.redis.delete(LOCK_KEY)
            raise
        self.state = RunState(row["experiment_id"], action, stream, series_id, n_blocks=n_blocks,
                              started_at=self.clock().isoformat(), last_guard=guard)
        self._cancel_reason = None
        self._task = asyncio.create_task(self._run(row, spec, t.value))
        return {"experiment": row, "run": self.status()}

    async def shutdown(self, reason: str) -> bool:
        """Cancel the active run (operator abort or service stop). True if one was running."""
        task = self._task
        if task is None or task.done():
            return False
        self._cancel_reason = reason
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        st = self.state
        if st is not None and st.status == "running":
            # Cancelled before its first step: _run's own cleanup never ran.
            key = override_key(st.stream_id)
            await self._abort(key, self.clock(), reason)
            await self._safe(self.redis.delete(key))
            if await self._safe(self.redis.get(LOCK_KEY)) in (st.experiment_id, st.experiment_id.encode()):
                await self._safe(self.redis.delete(LOCK_KEY))
        return True

    # --- the loop ------------------------------------------------------------------------------
    async def _run(self, exp: dict, spec: SeriesSpec, hot: float) -> None:
        st = self.state
        key = override_key(st.stream_id)
        segment_start = self.clock()
        logged: list[dict] = []
        try:
            for b in exp["design"]["blocks"]:
                b0, b1 = _ts(b["start"]), _ts(b["end"])
                st.block_index, st.arm = None, None
                await self._hold(b0, spec, hot, None)          # waiting: watch heat, keep the lock
                st.block_index, st.arm = b["index"], b["arm"]
                segment_start = self.clock()
                if b["arm"] == "on":
                    await self._check_guard(spec)              # never start an ON block on a stale "safe"
                snap0 = await self._to_thread(self.probe.snapshot, st.stream_id)
                if b["arm"] == "on":
                    await self.redis.set(key, "on", ex=self.ttl)
                actual_start = self.clock()
                await self._hold(b1, spec, hot, key if b["arm"] == "on" else None, guard_now=b["arm"] != "on")
                snap1 = await self._to_thread(self.probe.snapshot, st.stream_id)   # before clearing
                if b["arm"] == "on":
                    await self.redis.delete(key)
                notes = {"stream_id": st.stream_id, "actual_start": actual_start.isoformat(),
                         "router_dispatch_delta": _delta(snap0, snap1),
                         "router_override_active_at_end": snap1["override_active"] if snap1 else None}
                await self._to_thread(self.effects.log_intervention, st.action, b0, b1, RUNNER_SOURCE,
                                      json.dumps(notes), b["arm"], st.experiment_id)
                logged.append({**b, "notes": notes})
                st.blocks_logged += 1
                segment_start = self.clock()
            st.block_index, st.arm = None, None
            st.result = await self._finish(exp, logged)
            st.status = "completed"
            logger.info("experiment_completed id=%s", st.experiment_id)
        except asyncio.CancelledError:
            await self._abort(key, segment_start, self._cancel_reason or "runner cancelled")
            raise
        except _Abort as exc:
            await self._abort(key, segment_start, str(exc))
        except Exception as exc:  # noqa: BLE001 -- any failure ends the run safely, never leaves focus on
            logger.exception("experiment_runner_failed")
            await self._abort(key, segment_start, f"runner error: {type(exc).__name__}: {exc}"[:300])
        finally:
            await self._safe(self.redis.delete(key))
            if await self._safe(self.redis.get(LOCK_KEY)) in (st.experiment_id, st.experiment_id.encode()):
                await self._safe(self.redis.delete(LOCK_KEY))

    async def _hold(self, until: datetime, spec: SeriesSpec, hot: float, key: str | None,
                    guard_now: bool = True) -> None:
        """Sleep until ``until`` in ticks; each tick refreshes the key (ON) and lock, checks the heat.
        The guard (a forecast build) runs every ``guard_every`` seconds."""
        next_guard = self.clock() + (timedelta(0) if guard_now else timedelta(seconds=self.guard_every))
        while True:
            if key is not None:
                await self.redis.set(key, "on", ex=self.ttl)
            if not await self.redis.set(LOCK_KEY, self.state.experiment_id, xx=True, ex=self.lock_ttl):
                raise _Abort("lost the runner lock (it expired or someone deleted it)")
            await self._check_heat(spec, hot)
            now = self.clock()
            if now >= next_guard:
                await self._check_guard(spec)
                next_guard = now + timedelta(seconds=self.guard_every)
            left = (until - self.clock()).total_seconds()
            if left <= 0:
                return
            await self._sleep(min(self.tick, left))

    async def _check_heat(self, spec: SeriesSpec, hot: float) -> None:
        now = self.clock()
        pts = await self._to_thread(self.effects.store.points, spec, now - READING_MAX_AGE)
        vals = [(t, v) for t, v in pts if v is not None and t <= now]
        if not vals:
            raise _Abort(f"no {spec.series_id} reading in the last {READING_MAX_AGE}; cannot watch the heat")
        self.state.last_reading = float(vals[-1][1])
        if self.state.last_reading >= hot:
            raise _Abort(f"{spec.series_id} read {self.state.last_reading:.2f} >= hot line {hot:g}")

    async def _check_guard(self, spec: SeriesSpec) -> None:
        g = await self._to_thread(self._guard, spec)
        self.state.last_guard = g
        if not g.get("safe"):
            raise _Abort(f"guard breached: {g.get('reason')} (p={g.get('probability')})")

    async def _abort(self, key: str, segment_start: datetime, reason: str) -> None:
        st = self.state
        st.status, st.abort_reason = "aborted", reason
        logger.warning("experiment_aborted id=%s reason=%s", st.experiment_id, reason)
        await self._safe(self.redis.delete(key))
        end = max(self.clock(), segment_start + timedelta(seconds=1))
        try:
            await self._to_thread(self.effects.log_intervention, st.action, segment_start, end, RUNNER_SOURCE,
                                  json.dumps({"aborted": reason, "block_index": st.block_index, "arm": st.arm}),
                                  None, st.experiment_id)
            await self._to_thread(self.effects.store.set_experiment_status, st.experiment_id, "cancelled")
        except Exception as exc:  # noqa: BLE001 -- the key is already cleared; the DB note is best effort
            st.notes.append(f"could not record the abort: {type(exc).__name__}: {exc}"[:300])

    @staticmethod
    async def _safe(aw):
        try:
            return await aw
        except Exception as exc:  # noqa: BLE001
            logger.warning("experiment_runner_redis_error %s: %s", type(exc).__name__, exc)
            return None

    # --- end of run ----------------------------------------------------------------------------
    async def _finish(self, exp: dict, logged: list[dict]) -> dict:
        try:
            analysis = await self._to_thread(self.effects.analyze_experiment, exp["experiment_id"])
        except CausalError as exc:
            analysis = {"error": str(exc)}
        manip = await self._to_thread(self.manipulation_check, exp, logged)
        return {"analysis": analysis, "manipulation": manip, "summary": summarize(exp, analysis, manip)}

    def manipulation_check(self, exp: dict, logged: list[dict]) -> dict:
        """Did ON blocks actually raise the tier and load the machines? Router dispatch counts per arm
        plus the switchback estimate (ON minus OFF, no burn-in: power responds in seconds) on watts."""
        router: dict[str, dict] = {}
        for arm in ("on", "off"):
            deltas = [b["notes"]["router_dispatch_delta"] for b in logged if b["arm"] == arm]
            seen = [d for d in deltas if d is not None]
            total = sum(sum(d.values()) for d in seen)
            trig = sum(v for d in seen for k, v in d.items() if k.startswith("triggered:"))
            router[arm] = {"blocks_probed": len(seen), "blocks": len(deltas), "dispatches": total,
                           "triggered_fraction": (trig / total) if total else None,
                           "experiment_dispatches": sum(d.get("triggered:experiment", 0) for d in seen)}
        blocks = [{k: b[k] for k in ("index", "arm", "start", "end")} for b in logged]
        watts = {}
        if blocks:
            first = min(_ts(b["start"]) for b in blocks)
            last = max(_ts(b["end"]) for b in blocks)
            specs = [GPU_SPEC] + [s for s in (self.effects.engine.registry.get(i) for i in REGISTRY_CHECKS) if s]
            for s in specs:
                try:
                    y = to_grain(self.effects.store.points(s, first, last), s.grain_min, s.agg, s.max_fill_min,
                                 start=first, end=last)
                    est = analyze_switchback(y, blocks, 0, s.grain_min, None, n_perm=2000)
                    watts[s.series_id] = {"unit": s.unit, "effect": est.effect, "ci_lo": est.ci_lo,
                                          "ci_hi": est.ci_hi, "p_value": est.p_value,
                                          "reason": est.details.get("reason")}
                except Exception as exc:  # noqa: BLE001 -- a missing watts series is reported, not fatal
                    watts[s.series_id] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        on, off = router["on"], router["off"]
        verdict = "unknown"
        if on["blocks_probed"] and off["blocks_probed"]:
            if on["experiment_dispatches"] == 0:
                verdict = "override_never_took_effect"
            elif (off["triggered_fraction"] or 0) >= 0.9:
                verdict = "no_lever_off_arm_already_triggered"
            else:
                verdict = "tier_raised"
        return {"verdict": verdict, "router": router, "watts": watts}


def summarize(exp: dict, analysis: dict, manip: dict) -> str:
    gpu = manip["watts"].get(GPU_SPEC.series_id) or {}
    gtxt = (f"GPU {gpu['effect']:+.1f} W (95% {gpu['ci_lo']:+.1f}..{gpu['ci_hi']:+.1f})"
            if gpu.get("effect") is not None else "GPU watts unmeasured")
    if analysis.get("effect") is None:
        etxt = f"no temperature estimate ({analysis.get('error') or (analysis.get('details') or {}).get('reason')})"
    else:
        etxt = (f"{exp['series_id']} {analysis['effect']:+.2f} (95% {analysis['ci_lo']:+.2f}..{analysis['ci_hi']:+.2f},"
                f" p={analysis['p_value']:.3f})")
    return f"{exp['action']}: {etxt}; manipulation {manip['verdict']}, {gtxt}"


async def recover(effects: Any, redis: Any, to_thread=asyncio.to_thread) -> list[str]:
    """Boot-time cleanup: an experiment this runner left half-done (process died mid-run) is marked
    cancelled with an abort row, and its override key deleted. Nothing is resumed."""
    found: set[str] = set()
    held = await ExperimentRunner._safe(redis.get(LOCK_KEY))
    if isinstance(held, bytes):
        held = held.decode()
    if held and held != "starting":
        found.add(held)
    rows = await to_thread(effects.store.interventions, None, None, 500)
    found |= {str(r["experiment_id"]) for r in rows if r.get("source") == RUNNER_SOURCE and r.get("experiment_id")}
    cancelled = []
    for eid in sorted(found):
        exp = await to_thread(effects.store.experiment, eid)
        if exp is None or exp["status"] != "designed":
            continue
        mine = await to_thread(effects.store.interventions, None, eid, 1000)
        if sum(1 for r in mine if r.get("arm")) >= exp["design"]["n_blocks"]:
            continue   # every block ran; only the analysis is missing (POST /experiments/{id}/analyze)
        try:
            await ExperimentRunner._safe(redis.delete(override_key(stream_of(exp["action"]))))
        except CausalError:
            pass
        now = effects.engine.clock()
        await to_thread(effects.log_intervention, exp["action"], now - timedelta(seconds=1), now, RUNNER_SOURCE,
                        json.dumps({"aborted": "interrupted: the service restarted before the run finished"}),
                        None, eid)
        await to_thread(effects.store.set_experiment_status, eid, "cancelled")
        cancelled.append(eid)
        logger.warning("experiment_interrupted_marked_cancelled id=%s", eid)
    if held and held in cancelled:
        await ExperimentRunner._safe(redis.delete(LOCK_KEY))
    return cancelled
