"""Hub starts Orion's solo Zelda leisure on its own timer (ladder retirement PR 1, 2026-11-05).

Spec: docs/superpowers/specs/2026-11-04-retire-proposal-policy-dispatch-rungs.md ("PR 1. Retro play").

Before this module, only the proposal -> policy -> execution-dispatch ladder could start a leisure
session: a ``retro_leisure_session`` candidate on every warranted tick, the allocator's ``play`` act,
then a bus RPC to Hub. In practice that was a 5-minute poller of the runner's own gate (about 2,314
sends for 4 sessions in 14 days). This timer replaces it, at the same cadence, inside Hub.

One tick (every ``RETRO_LEISURE_SELF_START_TICK_SEC``, default 300 s):

1. A session this Hub started is still running: do nothing (``session_running``).
2. The leisure switch is off (``RETRO_LEISURE_ENABLED=false``): do nothing (``disabled``), no bid.
3. No runner module in this image: do nothing (``runner_unavailable``), no bid, never a fake session.
3b. The self-monitor act gate (E1, the same ``ActGate`` dispatch held for this act until 2026-11-05): a
   faculty held broken or unreliable is refused (``refused:track_record_failing``), no bid, except one
   re-test probe per 6 h, spent only when a session really starts.
4. Mint the act id (``retro_self_start:<uuid>``). Place the SHADOW ``hub.retro_leisure`` bid with it,
   BEFORE the gate, so the market sees the want and not the aftermath. The bid is logged, never obeyed.
5. Ask the runner's own gate, ``should_start(config)``. A named refusal ends the tick there.
6. Start ``run_leisure_session(config, act_id=)`` and wait up to START_GRACE_SEC for its own re-check.
7. Still running (or finished): open the ``play`` act, then follow the session. At the end
   ``scripts.retro_memory_export.export_leisure_outcome`` writes the memory and closes the act.

Clock and freshness: the tick reads its own clock FIRST and only uses it as a label
(``ticked_at``). Every age the gate checks is computed by the reader at its own read time
(``retro_leisure.read_regulation`` -> ``arousal_view(now=datetime.now())`` after the GET;
``PresenceStore.seconds_since_human`` after its GET). Never compare a later read against
``ticked_at``: #3134 stopped every painting that way.

A failure is harmless and retried: an exception or an unreadable dependency (Redis, Retro, the
regulation key mid-restart during a post-pull rebuild) is a refused tick, recorded on ``/health``, and
the next tick comes ERROR_RETRY_SEC later instead of the full interval.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import time
from collections import Counter
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

logger = logging.getLogger("orion-hub.retro_leisure_self_start")

# The act id prefix of a self-started session (the play act's key and correlation id).
ACT_ID_PREFIX = "retro_self_start:"
PLAY_ACTOR = "hub.retro_leisure"
PLAY_VERB = "play"
# How long to wait for the runner's own start re-check. Its refusals (gate, lock, save read, Retro
# start) all return before the first planner call.
START_GRACE_SEC = 5.0
# The first tick waits for Hub's own startup (the bus, the presence store) to settle.
FIRST_TICK_DELAY_SEC = 60.0
# After an error, retry sooner than the full interval (a dependency was probably mid-restart).
ERROR_RETRY_SEC = 60.0
ERROR_RESULTS = frozenset({"gate_failed", "start_failed", "tick_failed"})


def import_runner() -> Any:
    """Lane B's module, or None when it is not deployed in this Hub."""
    try:
        return importlib.import_module("scripts.retro_leisure")
    except ImportError:
        logger.warning("retro_leisure_runner_missing", exc_info=True)
        return None


def gate_verdict(raw: Any) -> tuple[bool, str | None]:
    """``should_start``'s answer: None (or True) = may start; a string (or False) = refused."""
    if raw is None or raw is True:
        return True, None
    if raw is False or not str(raw).strip():
        return False, "unnamed_refusal"
    return False, str(raw)[:200]


def new_act_id() -> str:
    return f"{ACT_ID_PREFIX}{uuid4()}"


def _iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat()


class RetroLeisureSelfStart:
    def __init__(
        self,
        *,
        config_provider: Callable[[], Any],
        pool_provider: Callable[[], Any],
        runner_provider: Callable[[], Any] = import_runner,
        tick_sec: float = 300.0,
        enabled: bool = True,
        clock: Callable[[], float] = time.time,
        first_delay_sec: float = FIRST_TICK_DELAY_SEC,
        gate: Any = None,
    ) -> None:
        self.config_provider = config_provider
        self.pool_provider = pool_provider
        self.runner_provider = runner_provider
        self.tick_sec = float(tick_sec)
        self.enabled = bool(enabled)
        self.clock = clock
        self.first_delay_sec = float(first_delay_sec)
        self.gate = gate   # orion.self_monitor.gate.ActGate, or None (no E1 gate)
        self.bus: Any = None
        self.task: asyncio.Task | None = None
        self.session_task: asyncio.Task | None = None
        self._followers: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()
        self.ticks = 0
        self.results: Counter = Counter()
        self.refusals: Counter = Counter()
        self.last: dict[str, Any] | None = None
        self.last_started: dict[str, Any] | None = None

    # --- one tick ---------------------------------------------------------------------------

    def _record(self, ticked_at: float, result: str, **extra: Any) -> dict[str, Any]:
        out = {"result": result, "ticked_at": _iso(ticked_at), **extra}
        self.ticks += 1
        self.results[result] += 1
        if extra.get("refusal"):
            self.refusals[str(extra["refusal"]).split(":", 1)[0]] += 1
        self.last = out
        if result == "started":
            self.last_started = out
        return out

    async def tick(self) -> dict[str, Any]:
        """One attempt. Never raises (except cancellation)."""
        async with self._lock:
            ticked_at = self.clock()   # a label only; never an age reference (module docstring)
            try:
                return await self._tick(ticked_at)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning("retro_leisure_self_start_tick_failed", exc_info=True)
                return self._record(ticked_at, "tick_failed", error=type(exc).__name__)

    async def _tick(self, ticked_at: float) -> dict[str, Any]:
        if self.session_task is not None and not self.session_task.done():
            return self._record(ticked_at, "session_running")
        config = self.config_provider()
        if not bool(getattr(config, "RETRO_LEISURE_ENABLED", False)):
            return self._record(ticked_at, "disabled")
        runner = self.runner_provider()
        if runner is None or not hasattr(runner, "should_start") or not hasattr(runner, "run_leisure_session"):
            return self._record(ticked_at, "runner_unavailable")

        decision = None
        if self.gate is not None:
            # A Redis GET, and a ledger read when the faculty is failing: off the loop. The view's age
            # is checked against this time with a 300 s future-skew allowance, so a view published
            # after the tick's clock still reads fresh.
            now = datetime.fromtimestamp(ticked_at, tz=timezone.utc)
            await asyncio.to_thread(self.gate.begin, now)
            decision = await asyncio.to_thread(self.gate.decide, PLAY_ACTOR, PLAY_VERB, "", now=now)
            if not decision.admit:
                return self._record(ticked_at, "refused", refusal=f"self_monitor:{decision.reason}")

        act_id = new_act_id()
        # Body economy Patch C3b: SHADOW bid, never obeyed. Placed before the gate: the want exists
        # whether or not the body lets it run now. want_id = the play act's correlation id.
        from scripts.act_emit import play_object_ref
        from scripts.physiology_shadow import ShadowBid

        shadow = ShadowBid(family=PLAY_ACTOR, verb=PLAY_VERB, tier=2, want_id=act_id,
                           object_ref=play_object_ref(act_id), component="retro_leisure")
        shadow.place()
        try:
            result, extra = await self._attempt(runner, config, act_id)
        except BaseException:
            shadow.settle("blocked:error")
            raise
        shadow.settle("ran" if result == "started" else f"blocked:{extra.get('refusal') or result}")
        if result == "started" and decision is not None and decision.probe:
            # Spent only once the session really started: a refused start keeps the window's probe.
            self.gate.claim_probe(decision.faculty, now=datetime.fromtimestamp(ticked_at, tz=timezone.utc))
            extra["probe"] = True
        return self._record(ticked_at, result, act_id=act_id, **extra)

    async def _attempt(self, runner: Any, config: Any, act_id: str) -> tuple[str, dict[str, Any]]:
        try:
            ok, reason = gate_verdict(await runner.should_start(config))
        except Exception as exc:  # noqa: BLE001
            logger.warning("retro_leisure_gate_failed act_id=%s", act_id, exc_info=True)
            return "gate_failed", {"refusal": "gate_failed", "error": type(exc).__name__}
        if not ok:
            logger.info("retro_leisure_refused act_id=%s refusal=%s", act_id, reason)
            return "refused", {"refusal": reason or "unnamed_refusal"}

        session = asyncio.create_task(runner.run_leisure_session(config, act_id=act_id))
        done, _ = await asyncio.wait({session}, timeout=START_GRACE_SEC)
        if session in done:
            try:
                outcome = session.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("retro_leisure_run_failed act_id=%s", act_id, exc_info=True)
                return "start_failed", {"refusal": "start_failed", "error": type(exc).__name__}
            if not getattr(outcome, "started", False):
                reason = str(getattr(outcome, "refusal", None) or "unnamed_refusal")[:200]
                logger.info("retro_leisure_refused act_id=%s refusal=%s at=runner", act_id, reason)
                return "refused", {"refusal": reason}
        self.session_task = session
        ended = getattr(session.result(), "ended", None) if session in done else None
        session_id = getattr(ended, "session_id", None) or None
        from scripts.act_emit import publish_play_opened

        await publish_play_opened(self.bus, dispatch_id=act_id, session_id=session_id)
        follower = asyncio.create_task(self._follow(act_id, session))
        self._followers.add(follower)
        follower.add_done_callback(self._followers.discard)
        logger.info("retro_leisure_started act_id=%s session_id=%s", act_id, session_id)
        return "started", {"session_id": session_id}

    async def _follow(self, act_id: str, session: asyncio.Task) -> None:
        """Wait for the session to end, then export it (which closes the play act)."""
        from scripts.act_emit import publish_play_closed
        from scripts.retro_memory_export import export_leisure_outcome

        try:
            outcome = await session
        except asyncio.CancelledError:
            # Hub is stopping (often a deliberate rebuild mid-game). Not graded here: the act is left
            # open and the writer expires it at its horizon.
            raise
        except Exception:  # noqa: BLE001
            logger.warning("retro_leisure_run_failed act_id=%s", act_id, exc_info=True)
            await publish_play_closed(self.bus, dispatch_id=act_id, session_id=None, produced=False,
                                      note="runner_failed")
            return
        ended = getattr(outcome, "ended", None)
        if ended is None:
            await publish_play_closed(self.bus, dispatch_id=act_id, session_id=None, produced=False,
                                      note=f"no_session:{getattr(outcome, 'refusal', None)}"[:200])
            return
        result = await export_leisure_outcome(pool=self.pool_provider(), bus=self.bus, ended=ended,
                                              dispatch_id=act_id)
        logger.info("retro_leisure_exported act_id=%s session_id=%s state=%s card_id=%s",
                    act_id, result.session_id, result.state, result.card_id)

    # --- the loop ---------------------------------------------------------------------------

    def next_delay(self, result: dict[str, Any] | None) -> float:
        if result is not None and result.get("result") in ERROR_RESULTS:
            return min(ERROR_RETRY_SEC, self.tick_sec)
        return self.tick_sec

    async def _run(self) -> None:
        await asyncio.sleep(self.first_delay_sec)
        while True:
            result = await self.tick()
            await asyncio.sleep(self.next_delay(result))

    async def begin(self, bus: Any) -> None:
        self.bus = bus
        if not self.enabled:
            logger.info("retro_leisure_self_start_disabled")
            return
        self.task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        tasks = [t for t in (self.task, self.session_task, *self._followers) if t is not None]
        for t in tasks:
            t.cancel()
        for t in tasks:
            with suppress(asyncio.CancelledError, Exception):
                await t
        self.task = self.session_task = None

    def snapshot(self) -> dict[str, Any]:
        """For Hub's /health: a silent hold is visible here, not only in the logs."""
        return {
            "enabled": self.enabled,
            "running": self.task is not None and not self.task.done(),
            "tick_sec": self.tick_sec,
            "ticks": self.ticks,
            "results": dict(self.results),
            "refusals": dict(self.refusals),
            "session_running": self.session_task is not None and not self.session_task.done(),
            "last": self.last,
            "last_started": self.last_started,
        }


# --- live wiring --------------------------------------------------------------------------------

def build_act_gate(settings: Any, *, pool_provider: Callable[[], Any], loop: asyncio.AbstractEventLoop) -> Any:
    """The E1 gate for the play act. The view is one Redis GET (``read_view_sync``). The ledger read
    runs on Hub's asyncpg pool: the gate calls it from a worker thread, which hands the query to the
    event loop and waits (the pool belongs to the loop). A missing pool raises, and the gate treats
    that as an unreadable ledger, never as a refusal."""
    from orion.self_monitor.gate import ActGate, last_attempt_sql, read_view_sync

    sql = last_attempt_sql("asyncpg")

    async def _rows(actor: str, verb: str, secs: float) -> list[dict]:
        pool = pool_provider()
        if pool is None:
            raise RuntimeError("memory_pg_pool_unavailable")
        async with pool.acquire() as conn:
            return [dict(r) for r in await conn.fetch(sql, actor, verb, float(secs))]

    def ledger_rows(actor: str, verb: str, secs: float) -> list[dict]:
        return asyncio.run_coroutine_threadsafe(_rows(actor, verb, secs), loop).result(timeout=10.0)

    return ActGate(
        enabled=bool(getattr(settings, "SELF_MONITOR_ACT_GATE_ENABLED", True)),
        probe_interval_sec=float(getattr(settings, "SELF_MONITOR_PROBE_INTERVAL_SEC", 21600.0)),
        read_view=lambda now: read_view_sync(settings.ORION_BUS_URL, now=now),
        ledger_rows=ledger_rows,
    )


__all__ = [
    "build_act_gate",
    "ACT_ID_PREFIX",
    "ERROR_RETRY_SEC",
    "FIRST_TICK_DELAY_SEC",
    "RetroLeisureSelfStart",
    "START_GRACE_SEC",
    "gate_verdict",
    "import_runner",
    "new_act_id",
]
