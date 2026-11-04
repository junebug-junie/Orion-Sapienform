"""H5: a swap seat drained only for background owner work lets the running hold finish its node.

Live 2026-10-28..11-04: 319 background diffusion jobs (~1 min each, reverie-visual and orion-thought)
drained agent-gpu2 81 times; each recalled reading/curiosity hold died at the 600 s hold grace and
its turn (often 10-60 min) restarted from scratch (`resource.lease_expired:recall_grace_exceeded`).
"""
from __future__ import annotations

from datetime import timedelta

from orion.gpu_pool.scheduler import Abort, BACKGROUND_DRAIN, BACKGROUND_DRAIN_CUT, Expire, Recall
from orion.gpu_pool.tests.test_scheduler import CFG, T0, lease, of
from orion.gpu_pool.tests.test_warm_standby import SEAT, gpu2, sched

D = CFG.defaults
LONG = max(D.hold_clawback_grace_sec, D.background_drain_grace_sec)


def diffusion(priority="background", **kw):
    return lease("diffusion", "queued", None, priority=priority, kind="hold", retryable=True, **kw)


def on_seat(**kw):
    return lease("agent", kw.pop("status", "granted"), SEAT, priority="background", kind="hold",
                 retryable=True, lease_id="s", **kw)


def recalls(d):
    return [(r.lease_id, r.reason, (r.recall_by - T0).total_seconds()) for r in of(Recall, d)]


def test_live_config_gives_background_drains_the_long_grace():
    assert D.background_drain_grace_sec > D.hold_clawback_grace_sec


def test_background_image_waiting_gives_the_running_hold_the_long_grace():
    d = sched([on_seat(), diffusion(lease_id="img")], crds=gpu2(loaded=True))
    assert recalls(d) == [("s", BACKGROUND_DRAIN, LONG)]


def test_non_background_owner_waiting_keeps_the_normal_grace():
    d = sched([on_seat(), diffusion(lease_id="img", priority="interactive")], crds=gpu2(loaded=True))
    assert recalls(d) == [("s", "draining", D.hold_clawback_grace_sec)]


def test_mixed_demand_is_not_background_only():
    d = sched([on_seat(), diffusion(lease_id="a"), diffusion(lease_id="b", priority="system")],
              crds=gpu2(loaded=True))
    assert recalls(d) == [("s", "draining", D.hold_clawback_grace_sec)]


def test_a_request_lease_keeps_its_short_clawback():
    req = lease("agent", "granted", SEAT, priority="background", lease_id="r")
    d = sched([req, diffusion(lease_id="img")], crds=gpu2(loaded=True))
    assert recalls(d) == [("r", "draining", D.clawback_grace_sec)]


def _recalling(seconds_ago: float):
    return on_seat(status="recalling", reason=BACKGROUND_DRAIN,
                   recall_by=T0 - timedelta(seconds=seconds_ago) + timedelta(seconds=LONG))


def test_a_later_interactive_owner_cuts_the_long_grace_back_to_the_normal_one():
    urgent_img = diffusion(lease_id="img", priority="interactive")
    late = sched([_recalling(D.hold_clawback_grace_sec + 1), urgent_img], crds=gpu2(loaded=True))
    assert [(a.lease_id, a.reason) for a in of(Abort, late)] == [("s", BACKGROUND_DRAIN_CUT)]
    early = sched([_recalling(D.hold_clawback_grace_sec - 60), urgent_img], crds=gpu2(loaded=True))
    assert of(Abort, early) == []


def test_still_background_only_the_long_grace_runs_out_normally():
    bg = diffusion(lease_id="img")
    assert of(Abort, sched([_recalling(D.hold_clawback_grace_sec + 1), bg], crds=gpu2(loaded=True))) == []
    assert [a.lease_id for a in of(Abort, sched([_recalling(LONG + 1), bg], crds=gpu2(loaded=True)))] == ["s"]


def test_a_hold_whose_heartbeat_was_lost_gets_one_decision_not_two():
    lost = on_seat(status="recalling", reason=BACKGROUND_DRAIN, expires_at=T0 - timedelta(seconds=1),
                   recall_by=T0 - timedelta(seconds=D.hold_clawback_grace_sec + 1) + timedelta(seconds=LONG))
    d = sched([lost, diffusion(lease_id="img", priority="interactive")], crds=gpu2(loaded=True))
    assert [e.lease_id for e in of(Expire, d)] == ["s"] and of(Abort, d) == []
