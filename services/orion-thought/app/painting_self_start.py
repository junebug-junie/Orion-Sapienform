"""Orion starts its own waking paintings (ladder retirement PR 2, 2026-11-05).

Spec: docs/superpowers/specs/2026-11-04-retire-proposal-policy-dispatch-rungs.md, "PR 2. Painting"
and "Decided" item 2 (Juniper: painting keeps today's rate, about 9 a day -- the baseline schedule
plus a hunger bid).

Before this module, the only thing that could start a painting was the proposal -> policy ->
execution-dispatch ladder: the `render_scene` proposal template (picked by the motor allocator),
plus the visual-baseline schedule that orion-proposal-runtime evaluated on every proposal tick.
Both are removed in the same PR, so a painting has exactly one trigger: this loop.

One tick (every ``ORION_PAINTING_SELF_START_TICK_SEC``, default 60 s):

1. The legacy run-once worker (``ORION_VISUAL_CHAIN_ENABLED``) is on: issue nothing. A durable
   baseline run would only wait it out (visual_steps.prepare_step), and two painters is the one
   failure this switch-over must never cause.
2. Read the painting activity (last acknowledged production, any open attempt) and run the
   baseline schedule (``orion.reverie.baseline.schedule``) on its durable checkpoint row
   (``visual_baseline_checkpoint``, the same row proposal-runtime used, under ``FOR UPDATE``).
   Due = last painting + ``interval_sec`` (5,400 s). An open attempt (a dream hop too) waits. A
   failed attempt is retried after ``retry_sec`` (600 s). Arousal ``strained`` at the moment of
   issue defers by ``retry_sec`` (R3 reader 4). Not due / deferred: nothing else happens.
3. The self-monitor act gate (E1, the same ``ActGate`` dispatch used for this act): a faculty held
   broken or unreliable is refused ``track_record_failing``, except one re-test probe per 6 h.
4. The hunger bid: a SHADOW clock want on ``thought.painting`` (no event: hunger alone) through the
   body-economy market (``orion.physiology.client`` via app/physiology_shadow.py). Logged beside
   the real outcome, never obeyed: painting is fourth in Juniper's binding order, so its bids bind
   only in attention patch 3. ``want_id`` is the run's correlation id; ``object_ref`` is
   ``durable_run:<run_id>``, which is also the act's ``object_ref``, so want_ledger joins the act.
5. Submit one ``reverie.visual`` durable run to orion-durable-runs (``POST /runs``). From there
   nothing changed: the run waits for its GPU-pool ``diffusion`` hold (the diffusion lock), the
   generate step checks the ambient thermal gate, and the steps in app/visual_steps.py open and
   close the ``thought.painting`` ``make`` act exactly as they did for a dispatched run.

The act ledger: a painting's act is opened by its first real generation and closed by its produced
caption (or a failed terminal / abandon), all in app/visual_steps.py, keyed on the run id. The act's
``test_before`` is ``visual_dispatch:<dispatch_id>``; ``dispatch_id`` is now
``painting_self_start:<need_id>:<issued_at>`` (not an execution-dispatch id). sql-writer's
act_downstream_use joins it to the chain row by that same id, so nothing downstream changes.

The rate is the schedule's own cadence (5,400 s after each painting, plus queue, gates and retries).
evals/painting_self_start_replay.py replays it on 14 days of real history; see the PR report.
"""
from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional
from uuid import NAMESPACE_URL, uuid5

from orion.regulation.arousal_read import read_arousal_sync
from orion.reverie.baseline import VisualBaselinePolicy, load_baseline_policy
from orion.schemas.durable_run import DurableRunRequestV1
from orion.schemas.resource_admission import ResourceRequirementV1
from orion.schemas.reverie_visual import VisualActivityV1, VisualBaselineEligibilityV1, VisualRunRequestV1
from orion.schemas.reverie_visual_run import (
    REVERIE_VISUAL_HOLD_LANE,
    REVERIE_VISUAL_MAX_RETRY_WINDOW_SEC,
    REVERIE_VISUAL_WORKFLOW,
    ReverieVisualRunBriefV1,
    reverie_visual_run_id,
)

logger = logging.getLogger("orion-thought.painting_self_start")

PAINT_ACTOR = "thought.painting"
PAINT_VERB = "make"
PAINT_FAMILY = "thought.painting"
PAINT_TIER = 2
COMPONENT = "painting_self_start"
DISPATCH_PREFIX = "painting_self_start:"
SUBMIT_TIMEOUT_SEC = 10.0


def dispatch_id_for(eligibility: VisualBaselineEligibilityV1, issued_at: datetime) -> str:
    """One id per issue of one need: a retry of the same need (after a failed attempt) is a new run."""
    return f"{DISPATCH_PREFIX}{eligibility.need_id}:{issued_at.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}"


def build_run_request(eligibility: VisualBaselineEligibilityV1, issued_at: datetime,
                      policy: VisualBaselinePolicy) -> DurableRunRequestV1:
    """The ``reverie.visual`` run for one issued need. Same shape cortex-exec's render_scene
    kickoff submitted for a dispatched painting (diffusion hold, deadline = one interval, capped by
    the run contract's own maximum), minus the proposal/decision ids, which no longer exist."""
    dispatch_id = dispatch_id_for(eligibility, issued_at)
    run_id = reverie_visual_run_id(dispatch_id)
    correlation_id = str(uuid5(NAMESPACE_URL, f"orion:painting_self_start:{dispatch_id}"))
    window = min(float(policy.interval_sec), REVERIE_VISUAL_MAX_RETRY_WINDOW_SEC)
    visual = VisualRunRequestV1(dispatch_id=dispatch_id, correlation_id=correlation_id,
                                visual_baseline=eligibility)
    return DurableRunRequestV1(
        run_id=run_id,
        workflow=REVERIE_VISUAL_WORKFLOW,
        correlation_id=correlation_id,
        requested_at=issued_at,
        brief=ReverieVisualRunBriefV1(visual_request=visual),
        admission=ResourceRequirementV1(
            resource=f"service.route.{REVERIE_VISUAL_HOLD_LANE}",
            preferred_lane=REVERIE_VISUAL_HOLD_LANE,
            deadline_at=issued_at + timedelta(seconds=window),
        ),
    )


def post_run(base_url: str, request: DurableRunRequestV1, *, timeout_sec: float = SUBMIT_TIMEOUT_SEC) -> Optional[str]:
    """POST the run to orion-durable-runs. None when it was registered; otherwise why not. Never raises."""
    url = base_url.rstrip("/") + "/runs"
    body = request.model_dump_json(exclude_none=True).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # noqa: S310 -- internal service URL
            if resp.status not in (200, 202):
                return f"http_{resp.status}"
            data = json.loads(resp.read() or b"{}")
    except Exception as exc:  # noqa: BLE001 -- any failure is "not submitted"
        return f"unreachable:{type(exc).__name__}"
    if not isinstance(data, dict) or data.get("run_id") != request.run_id:
        return "receipt_mismatch"
    return None


@dataclass
class TickResult:
    result: str
    at: str
    detail: dict = field(default_factory=dict)


class PaintingSelfStart:
    """The trigger. Every dependency is injected, so the replay eval runs this same class."""

    def __init__(
        self,
        *,
        load_activity: Callable[[], VisualActivityV1],
        baseline_eligibility: Callable[..., tuple[Optional[VisualBaselineEligibilityV1], str]],
        arousal_level: Callable[[], Optional[str]],
        submit: Callable[[DurableRunRequestV1], Awaitable[Optional[str]]],
        gate: Any = None,
        shadow_bid: Optional[Callable[[DurableRunRequestV1], Any]] = None,
        legacy_worker_enabled: Callable[[], bool] = lambda: False,
        policy_loader: Callable[[], VisualBaselinePolicy] = load_baseline_policy,
    ) -> None:
        self._load_activity = load_activity
        self._baseline_eligibility = baseline_eligibility
        self._arousal_level = arousal_level
        self._submit = submit
        self._gate = gate
        self._shadow_bid = shadow_bid
        self._legacy_worker_enabled = legacy_worker_enabled
        self._policy_loader = policy_loader
        self.last: Optional[TickResult] = None
        self.submitted: list[DurableRunRequestV1] = []

    def _done(self, now: datetime, result: str, **detail: Any) -> str:
        self.last = TickResult(result=result, at=now.isoformat(), detail=detail)
        return result

    async def tick(self, now: datetime) -> str:
        policy = self._policy_loader()
        if not policy.enabled:
            return self._done(now, "visual_baseline_disabled")
        if self._legacy_worker_enabled():
            return self._done(now, "legacy_visual_worker_enabled")
        activity = await asyncio.to_thread(self._load_activity)
        eligibility, reason = await asyncio.to_thread(
            self._baseline_eligibility, activity, now=now, policy=policy, arousal_level=self._arousal_level)
        if eligibility is None:
            return self._done(now, reason or "not_eligible")

        request = build_run_request(eligibility, now, policy)
        object_ref = f"durable_run:{request.run_id}"
        decision = None
        if self._gate is not None:
            self._gate.begin(now)
            decision = self._gate.decide(PAINT_ACTOR, PAINT_VERB, object_ref, now=now)
            if not decision.admit:
                # The schedule already set its retry clock: the same need is asked again in retry_sec.
                return self._done(now, f"refused:{decision.reason}", need_id=eligibility.need_id)

        shadow = None
        if self._shadow_bid is not None:
            try:
                shadow = self._shadow_bid(request)
            except Exception:  # noqa: BLE001 -- shadow only
                shadow = None
        if decision is not None and decision.probe and not self._gate.claim_probe(decision.faculty, now=now):
            if shadow is not None:
                shadow.settle("blocked:track_record_failing")
            return self._done(now, "refused:track_record_failing", need_id=eligibility.need_id)

        error = await self._submit(request)
        if shadow is not None:
            shadow.settle("submitted" if error is None else f"blocked:submit_failed:{error}")
        if error is not None:
            logger.warning("painting_self_start_submit_failed run=%s need=%s error=%s",
                           request.run_id, eligibility.need_id, error)
            return self._done(now, f"submit_failed:{error}", run_id=request.run_id, need_id=eligibility.need_id)
        self.submitted.append(request)
        logger.info("painting_self_start_submitted run=%s need=%s due_at=%s dispatch=%s probe=%s",
                    request.run_id, eligibility.need_id, eligibility.due_at.isoformat(),
                    request.brief.visual_request.dispatch_id, bool(decision and decision.probe))
        return self._done(now, "submitted", run_id=request.run_id, need_id=eligibility.need_id,
                          due_at=eligibility.due_at.isoformat())


# --- live wiring --------------------------------------------------------------------------------

def _shadow_bid(request: DurableRunRequestV1):
    from .physiology_shadow import ShadowBid

    shadow = ShadowBid(family=PAINT_FAMILY, verb=PAINT_VERB, tier=PAINT_TIER, want_id=request.correlation_id,
                       object_ref=f"durable_run:{request.run_id}", component=COMPONENT)
    shadow.place()
    return shadow


def arousal_level() -> Optional[str]:
    """R3 reader 4 (moved from orion-proposal-runtime): Orion's arousal at the moment a baseline
    painting would be issued. ``schedule`` calls it only after every other check passed, so this is
    one Redis GET per due painting, not per tick. None (unknown, flag off, Redis down) changes nothing."""
    from .settings import settings

    view = read_arousal_sync(
        settings.orion_bus_url,
        enabled=bool(settings.visual_baseline_arousal_enabled),
        reader="visual_baseline",
        max_age_sec=float(settings.regulation_reader_max_age_sec),
    )
    if view.level == "strained":
        logger.info("visual_baseline_deferred reason=arousal_strained since=%s reasons=%s",
                    view.since, ",".join(view.reasons))
    return view.level if view.known else None


def build_live() -> PaintingSelfStart:
    from orion.self_monitor.gate import ActGate, read_view_sync

    from . import store
    from .settings import settings

    async def submit(request: DurableRunRequestV1) -> Optional[str]:
        return await asyncio.to_thread(post_run, settings.painting_self_start_durable_runs_url, request)

    gate = ActGate(
        enabled=bool(settings.self_monitor_act_gate_enabled),
        probe_interval_sec=float(settings.self_monitor_probe_interval_sec),
        read_view=lambda now: read_view_sync(settings.orion_bus_url, now=now),
        ledger_rows=store.painting_act_attempt_rows,
    )
    return PaintingSelfStart(
        load_activity=store.load_visual_activity,
        baseline_eligibility=store.visual_baseline_eligibility,
        arousal_level=arousal_level,
        submit=submit,
        gate=gate,
        shadow_bid=_shadow_bid if settings.thought_physiology_shadow_bid_enabled else None,
        legacy_worker_enabled=lambda: bool(settings.visual_chain_enabled),
    )


_live: Optional[PaintingSelfStart] = None


def health() -> dict:
    """``/health`` -> ``painting_self_start``: the loop's last tick, or why it is not running."""
    from .settings import settings

    if not settings.painting_self_start_enabled:
        return {"enabled": False}
    last = _live.last if _live is not None else None
    return {"enabled": True, "tick_sec": float(settings.painting_self_start_tick_sec),
            "last": None if last is None else {"result": last.result, "at": last.at, **last.detail},
            "submitted_since_boot": 0 if _live is None else len(_live.submitted)}


async def run_painting_self_start(stop_event: asyncio.Event) -> None:
    """The loop. No-op unless ORION_PAINTING_SELF_START_ENABLED (shipped on)."""
    global _live
    from .settings import settings

    if not settings.painting_self_start_enabled:
        logger.info("painting_self_start disabled")
        return
    _live = build_live()
    tick_sec = max(5.0, float(settings.painting_self_start_tick_sec))
    logger.info("painting_self_start started tick_sec=%.0f url=%s", tick_sec,
                settings.painting_self_start_durable_runs_url)
    while not stop_event.is_set():
        try:
            await _live.tick(datetime.now(timezone.utc))
        except Exception:  # noqa: BLE001 -- a bad tick never stops the loop
            logger.exception("painting_self_start_tick_failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=tick_sec)
        except asyncio.TimeoutError:
            continue


__all__ = ["PaintingSelfStart", "build_run_request", "dispatch_id_for", "health", "post_run",
           "run_painting_self_start"]
