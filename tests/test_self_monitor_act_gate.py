"""Self-monitor E1 in the motor allocator (orion-execution-dispatch-runtime), and the must-not-loosen
rule: the gate may only ADD a refusal.

* With an unknown view (absent, unparseable, stale, future-stamped, unreadable), a healthy status,
  an untracked faculty or the gate off, every allocation is identical to the gate not existing.
* With the act's faculty held failing, the gate only removes admissions, never adds one.
* A held-failing painting is refused ``track_record_failing`` (recorded on the frame), except one
  probe per 6 h; an in-flight candidate never spends the probe.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parents[1]
SVC = REPO / "services" / "orion-execution-dispatch-runtime"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SVC))

from app.worker import ExecutionDispatchRuntimeWorker  # noqa: E402
from orion.autonomy.prediction import EffectPosterior  # noqa: E402
from orion.schemas.action_prediction import ExpectedEffectV1  # noqa: E402
from orion.schemas.execution_dispatch_frame import (  # noqa: E402
    ExecutionDispatchCandidateV1,
    ExecutionDispatchFrameV1,
)
from orion.self_monitor.gate import ActGate  # noqa: E402
from tests._dispatch_minted_act_route import with_dispatch_minted_act_route  # noqa: E402
from orion.self_monitor.view import SelfMonitorView, self_monitor_view  # noqa: E402

NOW = datetime.now(timezone.utc)
PAINT_CELL = ("act:make", "thought.painting", "artifact_produced", 0)
LIVE_PAINT = EffectPosterior(mean=0.017, variance=0.0083, n=29)
RENDER_VERB = "skills.imagination.render_scene.v1"
PAINT = "act.thought.painting.make"
MOTOR = SimpleNamespace(remaining_sec=1e6)


def _state(status: str, *, faculty: str = PAINT, computed_at: datetime | None = None) -> str:
    at = computed_at or NOW
    return json.dumps({"kind": "self_monitor.state.v1", "computed_at": at.isoformat(), "window_days": 7.0,
                       "faculties": [{
                           "faculty": faculty, "gradeability": "graded", "window_start": (at - timedelta(days=7)).isoformat(),
                           "window_end": at.isoformat(), "n_bets": 28, "n_graded": 6, "n_failed_to_execute": 22,
                           "n_expired": 0, "weight_graded": 2.9, "candidate_status": status, "status": status,
                           "status_since": (at - timedelta(days=1)).isoformat()}]})


def _worker(monkeypatch, *, posteriors=None, in_flight=False, view_raw=None, gate_enabled=True,
            ledger_rows=None, view_fn=None) -> ExecutionDispatchRuntimeWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = ExecutionDispatchRuntimeWorker()
    monkeypatch.setattr(worker._settings, "orion_dispatch_act_candidates_enabled", True, raising=False)
    store = MagicMock()
    store.load_action_cost_estimates.side_effect = lambda lookback_hours=24: (
        {("express", "host:circe_gpu"): 57.1} if lookback_hours > 24 else {})
    store.load_effect_posteriors.return_value = {PAINT_CELL: LIVE_PAINT} if posteriors is None else posteriors
    store.act_in_flight.return_value = in_flight
    store.act_episode_in_flight.return_value = False
    store.act_control_stats.return_value = (0, 0)
    store.act_arm_stats.return_value = (0, 0)
    store.act_attempt_rows.return_value = list(ledger_rows or [])
    worker._store = store
    with_dispatch_minted_act_route(worker)
    worker._self_monitor_gate = None if gate_enabled is None else ActGate(
        enabled=gate_enabled, probe_interval_sec=6 * 3600.0,
        read_view=view_fn or (lambda now: self_monitor_view(view_raw, now=now)),
        ledger_rows=lambda actor, verb, secs: store.act_attempt_rows(actor, verb, secs))
    return worker


def _render(dispatch_id: str = "dispatch:proposal:render_scene:t1") -> ExecutionDispatchCandidateV1:
    return ExecutionDispatchCandidateV1(
        dispatch_id=dispatch_id, source_decision_id=f"pd:{dispatch_id}", source_proposal_id=f"p:{dispatch_id}",
        dispatch_status="prepared_for_dispatch", dispatch_mode="dispatch_read_only", dispatch_kind="express",
        target_id="host:circe_gpu", target_kind="system", cortex_verb=RENDER_VERB, cortex_mode="brain",
        risk_score=0.05, confidence_score=0.9)


def _field(dispatch_id: str = "dispatch:proposal:prune:t1") -> ExecutionDispatchCandidateV1:
    return ExecutionDispatchCandidateV1(
        dispatch_id=dispatch_id, source_decision_id=f"pd:{dispatch_id}", source_proposal_id=f"p:{dispatch_id}",
        dispatch_status="prepared_for_dispatch", dispatch_mode="dispatch_read_only", dispatch_kind="maintain",
        target_id="host:docker_images", target_kind="host", cortex_verb="skills.runtime.builder_prune.v1",
        cortex_mode="brain", risk_score=0.05, confidence_score=0.9,
        expected_effect=ExpectedEffectV1(signal_id="resource_pressure", direction="decrease", predicted_delta=0.0,
                                         predictor_variance=0.25, predictor_n=0, cold_start=True))


def _baseline(monkeypatch) -> ExecutionDispatchCandidateV1:
    from orion.schemas.reverie_visual import VisualBaselineEligibilityV1

    monkeypatch.setattr("orion.reverie.baseline.validate_eligibility", lambda *a, **k: None)
    b = VisualBaselineEligibilityV1(need_id="visual_baseline", observed_at=NOW, due_at=NOW,
                                    policy_id="visual_baseline.v1")
    return ExecutionDispatchCandidateV1.model_validate(
        {**_render("dispatch:baseline:t1").model_dump(), "visual_baseline": b.model_dump()})


def _frame(*cands) -> ExecutionDispatchFrameV1:
    return ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:test", source_policy_frame_id="policy.frame:test",
        source_proposal_frame_id="proposal.frame:test", generated_at=NOW,
        dispatch_mode="dispatch_read_only", candidates=list(cands))


def _decisions(allocation):
    return (sorted(c.dispatch_id for c in allocation.admitted),
            sorted((c.dispatch_id, why) for c, why in allocation.refused))


SCENARIOS = {
    "measured": dict(posteriors=None, in_flight=False),
    "cold": dict(posteriors={}, in_flight=False),
    "in_flight": dict(posteriors={}, in_flight=True),
}


def _scenario_frames(monkeypatch):
    return {"render": lambda: _frame(_render()), "render+field": lambda: _frame(_render(), _field()),
            "baseline": lambda: _frame(_baseline(monkeypatch)), "two_renders": lambda: _frame(_render("d:1"), _render("d:2"))}


def _boom(now):
    raise RuntimeError("redis down")


UNKNOWN_OR_HEALTHY = {
    "absent": dict(view_raw=None),
    "unparseable": dict(view_raw="{nope"),
    "stale": dict(view_raw=_state("broken", computed_at=NOW - timedelta(hours=3, minutes=1))),
    "future": dict(view_raw=_state("broken", computed_at=NOW + timedelta(hours=1))),
    "read_failed": dict(view_fn=_boom),
    "reliable": dict(view_raw=_state("reliable")),
    "miscalibrated": dict(view_raw=_state("miscalibrated")),
    "untracked_faculty": dict(view_raw=_state("broken", faculty="act.someone.else.make")),
    "gate_off_but_broken": dict(view_raw=_state("broken"), gate_enabled=False),
}


@pytest.mark.parametrize("case", sorted(UNKNOWN_OR_HEALTHY))
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_must_not_loosen_unknown_or_healthy_is_identical_to_no_gate(monkeypatch, case, scenario) -> None:
    for name, make in _scenario_frames(monkeypatch).items():
        off = _worker(monkeypatch, gate_enabled=None, **SCENARIOS[scenario])._log_allocator_preview(make(), MOTOR)
        worker = _worker(monkeypatch, ledger_rows=[], **SCENARIOS[scenario], **UNKNOWN_OR_HEALTHY[case])
        on = worker._log_allocator_preview(make(), MOTOR)
        assert _decisions(on) == _decisions(off), (case, scenario, name)
        assert worker._last_gate_labels == {}
        worker._store.act_attempt_rows.assert_not_called()


@pytest.mark.parametrize("ledger", [[], [{"object_ref": "durable_run:reverie-visual-0f1e", "opened_at": NOW}]])
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_must_not_loosen_failing_only_removes_admissions(monkeypatch, scenario, ledger) -> None:
    for status in ("broken", "unreliable"):
        for name, make in _scenario_frames(monkeypatch).items():
            off = _worker(monkeypatch, gate_enabled=None, **SCENARIOS[scenario])._log_allocator_preview(make(), MOTOR)
            on = _worker(monkeypatch, view_raw=_state(status), ledger_rows=ledger,
                         **SCENARIOS[scenario])._log_allocator_preview(make(), MOTOR)
            assert set(_decisions(on)[0]) <= set(_decisions(off)[0]), (status, scenario, name)
            if ledger and scenario != "in_flight":
                # A recent attempt really spends the window: every act-backed candidate is refused.
                assert any(why == "track_record_failing" for _, why in _decisions(on)[1]), (status, scenario, name)


def test_failing_painting_is_refused_track_record_failing(monkeypatch) -> None:
    recent = [{"object_ref": "durable_run:reverie-visual-0f1e", "opened_at": NOW - timedelta(hours=2)}]
    worker = _worker(monkeypatch, posteriors={}, view_raw=_state("broken"), ledger_rows=recent)
    out = worker._log_allocator_preview(_frame(_render()), MOTOR)
    assert out.admitted == () and out.refusals_by_reason() == {"track_record_failing": 1}
    assert worker._last_gate_labels["dispatch:proposal:render_scene:t1"].startswith(
        f"self_monitor:track_record_failing:{PAINT}:broken")


def test_probe_after_six_hours_is_admitted_and_marked(monkeypatch) -> None:
    old = [{"object_ref": "durable_run:reverie-visual-0f1e", "opened_at": NOW - timedelta(hours=7)}]
    worker = _worker(monkeypatch, posteriors={}, view_raw=_state("broken"), ledger_rows=old)
    out = worker._log_allocator_preview(_frame(_render()), MOTOR)
    assert [c.dispatch_id for c in out.admitted] == ["dispatch:proposal:render_scene:t1"]
    assert worker._last_gate_labels["dispatch:proposal:render_scene:t1"].startswith(f"self_monitor:probe:{PAINT}")
    # Not sent yet: an offered probe spends nothing, so the next tick offers it again.
    retry = worker._log_allocator_preview(_frame(_render("dispatch:proposal:render_scene:t2")), MOTOR)
    assert [c.dispatch_id for c in retry.admitted] == ["dispatch:proposal:render_scene:t2"]
    # Sent: the probe is spent. Next tick (act row not landed yet) is refused.
    assert [c.dispatch_id for c in worker._claim_self_monitor_probes([_render("dispatch:proposal:render_scene:t2")])] \
        == ["dispatch:proposal:render_scene:t2"]
    again = worker._log_allocator_preview(_frame(_render("dispatch:proposal:render_scene:t3")), MOTOR)
    assert again.refusals_by_reason() == {"track_record_failing": 1}


def test_one_probe_per_faculty_per_tick_across_both_lanes(monkeypatch) -> None:
    worker = _worker(monkeypatch, posteriors={}, view_raw=_state("broken"))
    out = worker._log_allocator_preview(_frame(_baseline(monkeypatch), _render()), MOTOR)
    assert len(out.admitted) == 1 and out.refusals_by_reason() == {"track_record_failing": 1}


def test_in_flight_candidate_never_spends_the_probe(monkeypatch) -> None:
    worker = _worker(monkeypatch, posteriors={}, in_flight=True, view_raw=_state("broken"))
    assert worker._log_allocator_preview(_frame(_render()), MOTOR).refusals_by_reason() == {"act_in_flight": 1}
    worker._store.act_attempt_rows.assert_not_called()
    worker._store.act_in_flight.return_value = False
    assert [c.dispatch_id for c in worker._log_allocator_preview(_frame(_render()), MOTOR).admitted] == [
        "dispatch:proposal:render_scene:t1"]


def test_baseline_lane_is_gated_on_the_same_act(monkeypatch) -> None:
    recent = [{"object_ref": "durable_run:reverie-visual-0f1e", "opened_at": NOW - timedelta(hours=1)}]
    worker = _worker(monkeypatch, view_raw=_state("unreliable"), ledger_rows=recent)
    out = worker._log_allocator_preview(_frame(_baseline(monkeypatch)), MOTOR)
    assert out.admitted == () and out.refusals_by_reason() == {"track_record_failing": 1}


def test_healed_painting_is_admitted_normally(monkeypatch) -> None:
    worker = _worker(monkeypatch, posteriors={}, view_raw=_state("reliable"))
    assert [c.dispatch_id for c in worker._log_allocator_preview(_frame(_render()), MOTOR).admitted] == [
        "dispatch:proposal:render_scene:t1"]
    assert worker._last_gate_labels == {}


def test_settings_ship_the_gate_on_with_a_six_hour_probe(monkeypatch) -> None:
    monkeypatch.delenv("SELF_MONITOR_ACT_GATE_ENABLED", raising=False)
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    from app.settings import Settings

    s = Settings()
    assert s.self_monitor_act_gate_enabled is True and s.self_monitor_probe_interval_sec == 21600.0


def test_unknown_view_type_is_frozen() -> None:
    assert SelfMonitorView(known=False, reason="x").status(PAINT) == "unknown"
