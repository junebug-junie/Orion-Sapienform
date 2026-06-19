from __future__ import annotations

from app.state_machine import ServicePhase, ServiceState, TransitionInput, transition


def _inp(**kwargs) -> TransitionInput:
    base = dict(
        equilibrium_bad=False,
        probe_status="probe_ok",
        auto_remediate=True,
        now=1000.0,
        cooldown_sec=300,
        max_attempts_per_hour=3,
        consecutive_probe_fails_threshold=2,
        post_grace_sec=60,
    )
    base.update(kwargs)
    return TransitionInput(**base)


def test_healthy_to_suspect_on_equilibrium_bad() -> None:
    out = transition(ServiceState(), _inp(equilibrium_bad=True), service_id="landing-pad")
    assert out.new_state.phase == ServicePhase.suspect


def test_healthy_to_suspect_on_probe_bad() -> None:
    out = transition(ServiceState(), _inp(probe_status="probe_bad"), service_id="landing-pad")
    assert out.new_state.phase == ServicePhase.suspect


def test_suspect_confirmed_on_equilibrium_and_probe_bad() -> None:
    state = ServiceState(phase=ServicePhase.suspect, consecutive_probe_fails=1)
    out = transition(state, _inp(equilibrium_bad=True, probe_status="probe_bad"), service_id="landing-pad")
    assert out.new_state.phase == ServicePhase.unhealthy_confirmed
    assert out.attention_events


def test_suspect_confirmed_after_two_probe_fails() -> None:
    state = ServiceState(phase=ServicePhase.suspect, consecutive_probe_fails=1)
    out = transition(state, _inp(probe_status="probe_bad"), service_id="landing-pad")
    assert out.new_state.phase == ServicePhase.unhealthy_confirmed


def test_unhealthy_confirmed_auto_remediate_tier1() -> None:
    state = ServiceState(phase=ServicePhase.unhealthy_confirmed)
    out = transition(state, _inp(probe_status="probe_bad"), service_id="landing-pad")
    assert out.should_remediate_tier1 is True
    assert out.new_state.phase == ServicePhase.remediating_tier1


def test_unhealthy_confirmed_observe_only_without_auto_remediate() -> None:
    state = ServiceState(phase=ServicePhase.unhealthy_confirmed)
    out = transition(state, _inp(probe_status="probe_bad", auto_remediate=False), service_id="landing-pad")
    assert out.should_remediate_tier1 is False
    assert out.attention_events


def test_remediating_tier1_enters_post_grace() -> None:
    state = ServiceState(phase=ServicePhase.remediating_tier1)
    out = transition(state, _inp(now=2000.0), service_id="landing-pad")
    assert out.new_state.phase == ServicePhase.post_check_grace
    assert out.new_state.post_grace_until_ts == 2060.0


def test_post_grace_recovery_to_healthy() -> None:
    state = ServiceState(phase=ServicePhase.post_check_grace, post_grace_until_ts=1000.0)
    out = transition(state, _inp(now=1100.0, probe_status="probe_ok"), service_id="landing-pad")
    assert out.new_state.phase == ServicePhase.healthy
    assert any(e.get("context", {}).get("event") == "recovery" for e in out.attention_events)


def test_post_grace_still_bad_schedules_tier2() -> None:
    state = ServiceState(phase=ServicePhase.post_check_grace, post_grace_until_ts=1000.0)
    out = transition(state, _inp(now=1100.0, probe_status="probe_bad"), service_id="landing-pad")
    assert out.new_state.pending_tier2 is True


def test_max_attempts_moves_to_attention_only() -> None:
    state = ServiceState(phase=ServicePhase.unhealthy_confirmed, attempts_this_hour=3)
    out = transition(state, _inp(probe_status="probe_bad"), service_id="landing-pad")
    assert out.new_state.phase == ServicePhase.attention_only
