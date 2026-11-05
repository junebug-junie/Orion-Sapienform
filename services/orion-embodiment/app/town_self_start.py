"""orion-embodiment starts Orion's AI Town sessions on its own timer (ladder retirement PR 3, 2026-11-05).

Spec: docs/superpowers/specs/2026-11-04-retire-proposal-policy-dispatch-rungs.md ("PR 3. AI Town").

Before this module, only the proposal -> policy -> execution-dispatch ladder could start an Orion
session: an ``aitown_session`` candidate on every warranted tick, the allocator's ``play`` act, then a
bus RPC (``orion:aitown_session:start:request``) to the session controller here. This timer replaces
it, next to the controller it calls. The controller still runs every cap; this only decides WHEN to ask.

One tick (every ``EMBODIMENT_TOWN_SELF_START_TICK_SEC``, default 300 s; the first 60 s after startup):

1. An Orion session is open: do nothing (``session_running``), no bid. (Dispatch held the route while
   the play act was in flight.)
2. The session switch is off (``EMBODIMENT_TOWN_SESSIONS_ENABLED=false``): do nothing (``disabled``),
   no bid.
3. The controller's last refusal still holds, for its own ``retry_after_sec`` (the next local midnight
   after ``daily_cap``, the rest of the gap after ``cooldown``): do nothing (``held``), no bid. This is
   dispatch's refusal hold, carried over: dispatch sent no RPC while it held, so no bid was placed.
4. The self-monitor act gate (E1, the same ``ActGate`` dispatch held for this act until 2026-11-05): a
   faculty held broken or unreliable is refused (``self_monitor:track_record_failing``), no bid, except
   one re-test probe per 6 h, spent only when a session really starts.
5. Mint the act id (``aitown_self_start:<uuid>``). Place the SHADOW ``aitown.session`` bid with the
   session id (``session_id_for(act_id)``, the play act's correlation id) as its want id, BEFORE the
   gate, so the market sees the want and not the aftermath. The bid is logged, never obeyed.
6. Ask the controller: ``start_session(act_id)``. Its refusal ends the tick with its own name; a pass
   unpauses the town and opens the play act. The controller closes the act when it pauses the town.

Clock and freshness (#3134): the tick reads its clock FIRST and uses it only as a label. The controller
takes its own clock AFTER its ledger read, the self-monitor view is aged after its GET
(``read_view_sync(now=None)``), and the E1 gate's "now" is taken after that read. The hold compares two
fresh clock reads, never a read value against an earlier clock.

A failure is harmless and retried: an exception, or a dependency mid-restart (Redis, Convex, the bus
during a post-pull rebuild), is a recorded tick (``tick_failed``, or the controller's
``ledger_unknown`` / ``town_unreachable`` / ``resume_failed``), never a hold, and the next tick comes
ERROR_RETRY_SEC later instead of the full interval.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from uuid import uuid4

from orion.schemas.aitown_session import AITOWN_SESSION_FAMILY, aitown_session_object_ref

from app.town_sessions import ACT_VERB, StartOutcome, TownSessionController, session_id_for

logger = logging.getLogger("orion.embodiment.town_self_start")

#: The act id prefix of a self-started session (the play act's key; test_before keeps the
#: ``aitown_session_dispatch:`` prefix, so every aitown.session act has one shape).
ACT_ID_PREFIX = "aitown_self_start:"
#: The first tick waits for the controller's own boot (its recover tick runs every 10 s).
FIRST_TICK_DELAY_SEC = 60.0
#: After an error, retry sooner than the full interval (a dependency was probably mid-restart).
ERROR_RETRY_SEC = 60.0
#: Controller refusals that mean "could not read or reach", not "the gate said no": retried soon, never held.
ERROR_REFUSALS = frozenset({"ledger_unknown", "town_unreachable", "resume_failed"})
ERROR_RESULTS = frozenset({"tick_failed"})


def new_act_id() -> str:
    return f"{ACT_ID_PREFIX}{uuid4()}"


def _iso(epoch: Optional[float]) -> Optional[str]:
    if epoch is None:
        return None
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat()


class TownSelfStart:
    def __init__(
        self,
        controller: TownSessionController,
        *,
        tick_sec: float = 300.0,
        enabled: bool = True,
        gate: Any = None,
        shadow_bid_enabled: bool = True,
        clock: Callable[[], float] = time.time,
        first_delay_sec: float = FIRST_TICK_DELAY_SEC,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        self.controller = controller
        self.tick_sec = float(tick_sec)
        self.enabled = bool(enabled)
        self.gate = gate   # orion.self_monitor.gate.ActGate, or None (no E1 gate)
        self.shadow_bid_enabled = bool(shadow_bid_enabled)
        self.clock = clock
        self.first_delay_sec = float(first_delay_sec)
        self._sleep = sleep
        self.task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        # (until epoch, the controller's refusal): dispatch's refusal hold, carried over.
        self._hold: Optional[tuple[float, str]] = None
        self.ticks = 0
        self.bids = 0
        self.results: Counter = Counter()
        self.refusals: Counter = Counter()
        self.last: Optional[dict[str, Any]] = None
        self.last_started: Optional[dict[str, Any]] = None
        # Rebuilt by the loop after each tick; /health only reads this copy.
        self._published: dict[str, Any] = self._build_snapshot()

    # --- one tick ---------------------------------------------------------------------------

    def _record(self, ticked_at: float, result: str, **extra: Any) -> dict[str, Any]:
        out = {"result": result, "ticked_at": _iso(ticked_at), **{k: v for k, v in extra.items() if v is not None}}
        self.ticks += 1
        self.results[result] += 1
        if extra.get("refusal"):
            self.refusals[str(extra["refusal"]).split(":", 1)[0]] += 1
        self.last = out
        if result == "started":
            self.last_started = out
        self._published = self._build_snapshot()
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
                logger.warning("town_self_start_tick_failed", exc_info=True)
                return self._record(ticked_at, "tick_failed", error=type(exc).__name__)

    async def _tick(self, ticked_at: float) -> dict[str, Any]:
        c = self.controller
        if c.orion is not None:
            return self._record(ticked_at, "session_running", session_id=c.orion.session_id)
        if not c.cfg.enabled:
            return self._record(ticked_at, "disabled")
        if self._hold is not None:
            until, why = self._hold
            if self.clock() < until:
                return self._record(ticked_at, "held", refusal=f"held:{why}", until=_iso(until))
            self._hold = None

        decision = None
        if self.gate is not None:
            # The view is one Redis GET, off the loop; it is aged at its own read time. The gate's
            # "now" (for the age of past attempts) is taken after that read.
            await asyncio.to_thread(self.gate.begin, datetime.now(timezone.utc))
            gate_now = datetime.fromtimestamp(self.clock(), tz=timezone.utc)
            decision = await asyncio.to_thread(self.gate.decide, AITOWN_SESSION_FAMILY, ACT_VERB, "", now=gate_now)
            if not decision.admit:
                return self._record(ticked_at, "refused", refusal=f"self_monitor:{decision.reason}")

        act_id = new_act_id()
        # Body economy: SHADOW bid, never obeyed. Placed before the gate: the want exists whether or
        # not the caps let it run now. want_id = the session id = the play act's correlation id.
        from app.physiology_shadow import ShadowBid

        shadow = ShadowBid(family=AITOWN_SESSION_FAMILY, verb=ACT_VERB, tier=2, want_id=session_id_for(act_id),
                           object_ref=aitown_session_object_ref(act_id), enabled=self.shadow_bid_enabled)
        shadow.place()
        if self.shadow_bid_enabled:
            self.bids += 1
        try:
            out: StartOutcome = await c.start_session(act_id)
        except BaseException:
            shadow.settle("blocked:error")
            raise
        shadow.settle("ran" if out.state == "started" else f"blocked:{out.refusal}")
        if out.state == "started":
            extra: dict[str, Any] = {"session_id": out.session_id}
            if decision is not None and decision.probe:
                # Spent only once the session really started: a refused start keeps the window's probe.
                if not self.gate.claim_probe(decision.faculty, now=datetime.fromtimestamp(self.clock(), tz=timezone.utc)):
                    logger.warning("town_self_start_probe_lost_race act_id=%s", act_id)
                extra["probe"] = True
            logger.info("town_self_start_started act_id=%s session=%s", act_id, out.session_id)
            return self._record(ticked_at, "started", act_id=act_id, **extra)
        refusal = str(out.refusal or "unnamed_refusal")
        if refusal not in ERROR_REFUSALS and out.retry_after_sec:
            self._hold = (self.clock() + float(out.retry_after_sec), refusal)
        return self._record(ticked_at, "refused", act_id=act_id, refusal=refusal,
                            retry_after_sec=out.retry_after_sec)

    # --- the loop ---------------------------------------------------------------------------

    def next_delay(self, result: Optional[dict[str, Any]]) -> float:
        if result is not None and (result.get("result") in ERROR_RESULTS
                                   or result.get("refusal") in ERROR_REFUSALS):
            return min(ERROR_RETRY_SEC, self.tick_sec)
        return self.tick_sec

    async def _run(self) -> None:
        await self._sleep(self.first_delay_sec)
        while True:
            result = await self.tick()
            await self._sleep(self.next_delay(result))

    def begin(self) -> None:
        if not self.enabled:
            logger.info("town_self_start_disabled")
            return
        self.task = asyncio.create_task(self._run(), name="embodiment-town-self-start")
        self._published = self._build_snapshot()

    async def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self.task
        self.task = None

    def _build_snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "tick_sec": self.tick_sec,
            "ticks": self.ticks,
            "bids": self.bids,
            "results": dict(self.results),
            "refusals": dict(self.refusals),
            "held_until": _iso(self._hold[0]) if self._hold else None,
            "held_for": self._hold[1] if self._hold else None,
            "last": self.last,
            "last_started": self.last_started,
        }

    def snapshot(self) -> dict[str, Any]:
        """For /health: a silent hold is visible here, not only in the logs."""
        return {**self._published, "running": self.task is not None and not self.task.done(),
                "session_open": self.controller.orion is not None}


# --- live wiring --------------------------------------------------------------------------------


def build_act_gate(settings: Any, controller: TownSessionController) -> Any:
    """The E1 gate for the play act. The view is one Redis GET (``read_view_sync``, aged at read time).

    The ledger read (only when the faculty is failing) comes from the controller's own session ledger,
    not Postgres: every Orion session row is a play act this controller opened, and this controller is
    the ledger's only writer. Before the controller has loaded the ledger it raises, which the gate
    treats as an unreadable ledger (its probe memo decides), never as a refusal."""
    from orion.self_monitor.gate import ActGate, read_view_sync

    return ActGate(
        enabled=bool(getattr(settings, "self_monitor_act_gate_enabled", True)),
        probe_interval_sec=float(getattr(settings, "self_monitor_probe_interval_sec", 21600.0)),
        # now=None: the view's age is measured after the GET, never against an earlier clock.
        read_view=lambda _now: read_view_sync(settings.bus_url),
        ledger_rows=lambda actor, verb, secs: controller.act_attempt_rows(secs),
    )


__all__ = [
    "ACT_ID_PREFIX",
    "ERROR_REFUSALS",
    "ERROR_RETRY_SEC",
    "FIRST_TICK_DELAY_SEC",
    "TownSelfStart",
    "build_act_gate",
    "new_act_id",
]
