"""Ladder retirement PR 2: Orion starts its own waking paintings (app/painting_self_start.py).

Pins: the trigger runs the REAL baseline schedule (due 5400 s after the last painting, 600 s retry,
arousal deferral, an open attempt waits); the self-monitor gate can refuse; the hunger bid is a
shadow clock want placed before the run exists and joins the act by object_ref; the submitted run is
the same ``reverie.visual`` shape the dispatch path submitted; and -- the hidden coupling -- that run,
driven through the real thought steps, still opens and closes the ``thought.painting`` ``make`` act.
"""
from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from orion.reverie.baseline import VisualBaselinePolicy, schedule
from orion.schemas.reverie_visual import VisualActivityV1
from orion.schemas.reverie_visual_run import (
    REVERIE_VISUAL_WORKFLOW,
    ReverieVisualStepRequestV1,
    reverie_visual_run_id,
)
from orion.self_monitor.gate import GateDecision

from app import painting_self_start as pss
from test_visual_steps import env  # noqa: F401 -- fixture

NOW = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)
POLICY = VisualBaselinePolicy(enabled=True)


class Clock:
    """An in-memory checkpoint row running the real reducer, and the activity it reads."""

    def __init__(self, *, last_success_at=NOW - timedelta(seconds=5400 + 60), active=None):
        self.checkpoint: dict = {}
        self.last_success_at = last_success_at
        self.active = active
        self.arousal = None
        self.arousal_calls = 0
        self.now = NOW

    def activity(self) -> VisualActivityV1:
        return VisualActivityV1(history_status="ok", observed_at=self.now, last_success_at=self.last_success_at,
                                last_success_chain_id="chain-1" if self.last_success_at else None,
                                last_success_sha256="a" * 64 if self.last_success_at else None,
                                active_attempt_id=self.active)

    def eligibility(self, activity, *, now, policy, arousal_level=None):
        need, self.checkpoint, reason = schedule(activity, copy.deepcopy(self.checkpoint), now=now, policy=policy,
                                                 arousal_level=arousal_level)
        return need, reason

    def arousal_level(self):
        self.arousal_calls += 1
        return self.arousal


class Shadow:
    def __init__(self, request):
        self.request = request
        self.real = None

    def settle(self, real, **_):
        self.real = real


class Gate:
    def __init__(self, decision):
        self.decision = decision
        self.claimed = []

    def begin(self, now):
        return None

    def decide(self, actor, verb, object_ref, *, now, job_key=None):
        self.asked = (actor, verb, object_ref)
        return self.decision

    def claim_probe(self, faculty, *, now, job_key=None):
        self.claimed.append(faculty)
        return True


def _trigger(clock: Clock, *, submit_error=None, gate=None, legacy=False):
    sent, shadows = [], []

    async def submit(request):
        sent.append(request)
        return submit_error

    def shadow_bid(request):
        s = Shadow(request)
        shadows.append(s)
        return s

    t = pss.PaintingSelfStart(load_activity=clock.activity, baseline_eligibility=clock.eligibility,
                              arousal_level=clock.arousal_level, submit=submit, gate=gate, shadow_bid=shadow_bid,
                              legacy_worker_enabled=lambda: legacy, policy_loader=lambda: POLICY)
    t.clock = clock
    return t, sent, shadows


def _tick(t, now):
    t.clock.now = now   # the activity is observed at the tick's own time, as live
    return asyncio.run(t.tick(now))


def test_not_due_issues_nothing_and_bids_nothing():
    clock = Clock(last_success_at=NOW - timedelta(seconds=600))
    t, sent, shadows = _trigger(clock)
    assert _tick(t, NOW) == "visual_baseline_not_due"
    assert sent == [] and shadows == [] and clock.arousal_calls == 0


def test_due_submits_one_reverie_visual_run_with_the_dispatch_shape():
    clock = Clock()
    t, sent, shadows = _trigger(clock)
    assert _tick(t, NOW) == "submitted"
    (req,) = sent
    visual = req.brief.visual_request
    assert req.workflow == REVERIE_VISUAL_WORKFLOW
    assert visual.dispatch_id.startswith("painting_self_start:visual-need:")
    assert req.run_id == reverie_visual_run_id(visual.dispatch_id)
    assert req.correlation_id == visual.correlation_id
    assert visual.visual_baseline is not None and visual.visual_baseline.due_at <= NOW
    assert visual.proposal_id is None and visual.decision_id is None
    # The diffusion lock: the run waits for the pool's diffusion hold, as the dispatched run did.
    assert (req.admission.resource, req.admission.preferred_lane) == ("service.route.diffusion", "diffusion")
    assert req.admission.deadline_at == NOW + timedelta(seconds=5400)
    # The hunger bid: placed before the run exists, named so want_ledger joins the act.
    (shadow,) = shadows
    assert shadow.request is req and shadow.real == "submitted"


def test_one_need_issues_once_then_waits_for_its_retry_clock():
    clock = Clock()
    t, sent, _ = _trigger(clock)
    assert _tick(t, NOW) == "submitted"
    clock.active = "attempt-1"   # the run claimed its attempt
    assert _tick(t, NOW + timedelta(seconds=60)) == "visual_baseline_active"
    clock.active = None          # the attempt failed (no new painting)
    assert _tick(t, NOW + timedelta(seconds=120)) == "visual_baseline_cooldown"
    assert _tick(t, NOW + timedelta(seconds=601)) == "submitted"
    assert len(sent) == 2 and sent[0].run_id != sent[1].run_id
    assert sent[0].brief.visual_request.visual_baseline.need_id == sent[1].brief.visual_request.visual_baseline.need_id


def test_a_new_painting_resets_the_clock():
    clock = Clock()
    t, sent, _ = _trigger(clock)
    assert _tick(t, NOW) == "submitted"
    clock.last_success_at = NOW + timedelta(seconds=300)
    assert _tick(t, NOW + timedelta(seconds=360)) == "visual_baseline_not_due"
    assert _tick(t, NOW + timedelta(seconds=300 + 5400)) == "submitted"
    assert len(sent) == 2


def test_arousal_strained_defers_and_releases():
    clock = Clock()
    clock.arousal = "strained"
    t, sent, shadows = _trigger(clock)
    assert _tick(t, NOW) == "arousal_strained"
    assert sent == [] and shadows == []
    assert clock.checkpoint["deferral"]["reason"] == "arousal_strained"
    clock.arousal = "idle"
    assert _tick(t, NOW + timedelta(seconds=300)) == "visual_baseline_cooldown"
    assert _tick(t, NOW + timedelta(seconds=601)) == "submitted"


def test_legacy_worker_on_never_issues():
    clock = Clock()
    t, sent, _ = _trigger(clock, legacy=True)
    assert _tick(t, NOW) == "legacy_visual_worker_enabled"
    assert sent == [] and clock.checkpoint == {}


def test_policy_disabled_never_issues():
    clock = Clock()
    t, sent, _ = _trigger(clock)
    t._policy_loader = lambda: VisualBaselinePolicy(enabled=False)
    assert _tick(t, NOW) == "visual_baseline_disabled" and sent == []


def test_self_monitor_refusal_issues_no_run_and_no_bid():
    clock = Clock()
    gate = Gate(GateDecision("act.thought.painting.make", False, False, "broken", "track_record_failing", None))
    t, sent, shadows = _trigger(clock, gate=gate)
    assert _tick(t, NOW) == "refused:track_record_failing"
    assert sent == [] and shadows == []
    assert gate.asked[:2] == ("thought.painting", "make") and gate.asked[2].startswith("durable_run:")


def test_self_monitor_probe_is_claimed_when_the_run_is_submitted():
    clock = Clock()
    gate = Gate(GateDecision("act.thought.painting.make", True, True, "broken", "probe", None))
    t, sent, _ = _trigger(clock, gate=gate)
    assert _tick(t, NOW) == "submitted" and len(sent) == 1
    assert gate.claimed == ["act.thought.painting.make"]


def test_submit_failure_settles_blocked_and_retries_on_the_schedule():
    clock = Clock()
    t, sent, shadows = _trigger(clock, submit_error="unreachable:URLError")
    assert _tick(t, NOW) == "submit_failed:unreachable:URLError"
    assert shadows[0].real == "blocked:submit_failed:unreachable:URLError"
    assert _tick(t, NOW + timedelta(seconds=60)) == "visual_baseline_cooldown"
    assert t.submitted == []


def test_a_broken_shadow_never_blocks_the_painting():
    clock = Clock()
    t, sent, _ = _trigger(clock)

    def boom(_request):
        raise RuntimeError("economy client broke")

    t._shadow_bid = boom
    assert _tick(t, NOW) == "submitted" and len(sent) == 1


def test_live_shadow_bid_is_a_tier2_clock_want_joined_by_object_ref(monkeypatch):
    from app import physiology_shadow as ps

    wants = []

    async def fake_abid(want):
        wants.append(want)
        return None

    monkeypatch.setattr(ps, "abid", fake_abid)
    monkeypatch.setattr(ps, "_enabled", lambda: True)
    clock = Clock()
    need, _ = clock.eligibility(clock.activity(), now=NOW, policy=POLICY)
    req = pss.build_run_request(need, NOW, POLICY)

    async def go():
        shadow = pss._shadow_bid(req)
        await asyncio.sleep(0.01)
        return shadow

    asyncio.run(go())
    (want,) = wants
    assert (want.family, want.verb, want.tier) == ("thought.painting", "make", 2)
    assert want.want_id == req.correlation_id
    assert want.object_ref == f"durable_run:{req.run_id}"
    assert want.producer == "orion-thought.painting_self_start"
    assert want.event_ref is None and want.salience is None   # hunger alone, no event


def test_post_run_refuses_a_receipt_for_another_run(monkeypatch):
    clock = Clock()
    need, _ = clock.eligibility(clock.activity(), now=NOW, policy=POLICY)
    req = pss.build_run_request(need, NOW, POLICY)

    class Resp:
        status = 202

        def __init__(self, body):
            self.body = body

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import json

    monkeypatch.setattr(pss.urllib.request, "urlopen",
                        lambda r, timeout: Resp(json.dumps({"run_id": req.run_id, "status": "waiting_resource"}).encode()))
    assert pss.post_run("http://durable-runs:8121", req) is None
    monkeypatch.setattr(pss.urllib.request, "urlopen", lambda r, timeout: Resp(b'{"run_id": "other"}'))
    assert pss.post_run("http://durable-runs:8121", req) == "receipt_mismatch"

    def down(r, timeout):
        raise OSError("refused")

    monkeypatch.setattr(pss.urllib.request, "urlopen", down)
    assert pss.post_run("http://durable-runs:8121", req) == "unreachable:OSError"


def _arousal(monkeypatch, *, enabled, view):
    from app.settings import settings

    calls = []

    def fake_read(url, *, enabled, reader, max_age_sec, **_kw):
        from orion.regulation.arousal_view import UNKNOWN_DISABLED

        calls.append({"enabled": enabled, "reader": reader, "max_age_sec": max_age_sec})
        return view if enabled else UNKNOWN_DISABLED

    monkeypatch.setattr(pss, "read_arousal_sync", fake_read)
    monkeypatch.setattr(settings, "visual_baseline_arousal_enabled", enabled)
    return calls


def test_live_arousal_reader_returns_strained_to_the_schedule(monkeypatch):
    """Moved with the schedule from orion-proposal-runtime's worker (R3 reader 4)."""
    from orion.regulation.arousal_view import ArousalView

    calls = _arousal(monkeypatch, enabled=True, view=ArousalView(level="strained", reason="strained"))
    assert pss.arousal_level() == "strained"
    assert calls == [{"enabled": True, "reader": "visual_baseline", "max_age_sec": 360.0}]


def test_live_arousal_flag_off_and_unknown_are_none(monkeypatch):
    from orion.regulation.arousal_view import ArousalView

    _arousal(monkeypatch, enabled=False, view=ArousalView(level="strained", reason="strained"))
    assert pss.arousal_level() is None
    _arousal(monkeypatch, enabled=True, view=ArousalView(level="unknown", reason="stale"))
    assert pss.arousal_level() is None


def test_flags_ship_on():
    from app.settings import ThoughtSettings

    f = ThoughtSettings.model_fields
    assert f["painting_self_start_enabled"].default is True
    assert f["visual_baseline_arousal_enabled"].default is True
    assert f["self_monitor_act_gate_enabled"].default is True
    assert f["painting_self_start_tick_sec"].default == 60.0


# --- the hidden coupling: the act ledger still grades a self-started painting -------------------

@pytest.mark.asyncio
async def test_a_self_started_run_opens_and_closes_the_painting_act(env, monkeypatch):
    """Drive the run this trigger submits through the real thought steps. Fails if the trigger's
    request could not pass prepare (baseline validation), or if the act lost its link to the run."""
    from test_visual_steps import _capture_production, _caption, _GrammarBus

    from orion.gpu_pool.client import durable_run_holder
    from orion.schemas.gpu_pool import GpuLeaseRefV1

    monkeypatch.setattr(env.steps, "load_baseline_policy", lambda: POLICY)
    monkeypatch.setattr(env.steps.settings, "thought_act_ledger_enabled", True, raising=False)
    clock = Clock()
    need, _ = clock.eligibility(clock.activity(), now=NOW, policy=POLICY)
    run = pss.build_run_request(need, NOW, POLICY)
    visual = run.brief.visual_request
    bus = _GrammarBus()
    _capture_production(env, monkeypatch)
    _caption(env, monkeypatch, "a lantern over still water")

    def step(name, attempt_id=None, corr=None):
        lease = None
        if name == "generate":
            lease = GpuLeaseRefV1(lease_id="lease-1", generation=1, role="diffusion",
                                  holder=durable_run_holder(run.run_id))
        return ReverieVisualStepRequestV1(run_id=run.run_id, correlation_id=corr or f"{name}-corr", step=name,
                                          visual_request=visual, attempt_id=attempt_id, gpu_lease=lease)

    now = lambda: NOW + timedelta(seconds=30)  # noqa: E731
    prepared = await env.steps.run_visual_step(bus, step("prepare"), now_fn=now)
    assert prepared.status == "done", prepared
    generated = await env.steps.run_visual_step(bus, step("generate", prepared.attempt_id), now_fn=now)
    assert generated.status == "done", generated
    captioned = await env.steps.run_visual_step(bus, step("caption", prepared.attempt_id), now_fn=now)
    assert (captioned.status, captioned.outcome) == ("done", "produced")
    events = [e.payload for ch, e in bus.sent if ch == "orion:grammar:event"]
    assert [e["event_kind"] for e in events] == ["act_opened", "act_closed"]
    opened, closed = events
    for ev in events:
        assert ev["act"]["actor"] == "thought.painting" and ev["act"]["verb"] == "make"
        assert ev["act"]["object_ref"] == f"durable_run:{run.run_id}"
        assert ev["act"]["test_before"] == f"visual_dispatch:{visual.dispatch_id}"
    assert opened["act"]["act_id"] == closed["act"]["act_id"]
    assert closed["act"]["outcome"]["value"] == 1.0
