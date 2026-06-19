from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Literal
from uuid import uuid4


class ServicePhase(str, Enum):
    healthy = "healthy"
    suspect = "suspect"
    unhealthy_confirmed = "unhealthy_confirmed"
    remediating_tier1 = "remediating_tier1"
    remediating_tier2 = "remediating_tier2"
    post_check_grace = "post_check_grace"
    attention_only = "attention_only"


@dataclass
class ServiceState:
    phase: ServicePhase = ServicePhase.healthy
    consecutive_probe_fails: int = 0
    last_remediate_ts: float | None = None
    attempts_this_hour: int = 0
    hour_window_start_ts: float = field(default_factory=time)
    post_grace_until_ts: float | None = None
    correlation_id: str | None = None
    pending_tier2: bool = False


@dataclass(frozen=True)
class TransitionInput:
    equilibrium_bad: bool
    probe_status: Literal["probe_ok", "probe_bad"]
    auto_remediate: bool
    now: float
    cooldown_sec: int
    max_attempts_per_hour: int
    consecutive_probe_fails_threshold: int
    post_grace_sec: int


@dataclass(frozen=True)
class TransitionOutput:
    new_state: ServiceState
    attention_events: list[dict]
    should_remediate_tier1: bool = False
    should_remediate_tier2: bool = False


def _reset_hour_window(state: ServiceState, now: float) -> None:
    if now - state.hour_window_start_ts >= 3600:
        state.attempts_this_hour = 0
        state.hour_window_start_ts = now


def _attention_event(*, severity: str, message: str, service_id: str, correlation_id: str | None, **ctx: object) -> dict:
    return {
        "severity": severity,
        "message": message,
        "correlation_id": correlation_id or str(uuid4()),
        "context": {"service_id": service_id, **ctx},
    }


def transition(state: ServiceState, inp: TransitionInput, *, service_id: str = "unknown") -> TransitionOutput:
    new_state = ServiceState(
        phase=state.phase,
        consecutive_probe_fails=state.consecutive_probe_fails,
        last_remediate_ts=state.last_remediate_ts,
        attempts_this_hour=state.attempts_this_hour,
        hour_window_start_ts=state.hour_window_start_ts,
        post_grace_until_ts=state.post_grace_until_ts,
        correlation_id=state.correlation_id or str(uuid4()),
        pending_tier2=state.pending_tier2,
    )
    _reset_hour_window(new_state, inp.now)
    attention_events: list[dict] = []
    should_remediate_tier1 = False
    should_remediate_tier2 = False
    corr = new_state.correlation_id

    if new_state.phase == ServicePhase.attention_only:
        return TransitionOutput(new_state=new_state, attention_events=attention_events)

    if new_state.phase == ServicePhase.post_check_grace:
        if new_state.post_grace_until_ts and inp.now < new_state.post_grace_until_ts:
            return TransitionOutput(new_state=new_state, attention_events=attention_events)
        if inp.probe_status == "probe_ok" and not inp.equilibrium_bad:
            new_state.phase = ServicePhase.healthy
            new_state.consecutive_probe_fails = 0
            new_state.pending_tier2 = False
            attention_events.append(
                _attention_event(
                    severity="info",
                    message=f"mesh health: {service_id} recovered after remediation",
                    service_id=service_id,
                    correlation_id=corr,
                    event="recovery",
                )
            )
            return TransitionOutput(new_state=new_state, attention_events=attention_events)
        if new_state.pending_tier2:
            if inp.auto_remediate and new_state.attempts_this_hour < inp.max_attempts_per_hour:
                new_state.phase = ServicePhase.remediating_tier2
                should_remediate_tier2 = True
            else:
                new_state.phase = ServicePhase.attention_only
                attention_events.append(
                    _attention_event(
                        severity="error",
                        message=f"mesh health: {service_id} persistent failure after tier-1",
                        service_id=service_id,
                        correlation_id=corr,
                        event="attention_only",
                    )
                )
            return TransitionOutput(
                new_state=new_state,
                attention_events=attention_events,
                should_remediate_tier2=should_remediate_tier2,
            )
        new_state.pending_tier2 = True
        new_state.post_grace_until_ts = inp.now + inp.post_grace_sec
        return TransitionOutput(new_state=new_state, attention_events=attention_events)

    if new_state.phase == ServicePhase.remediating_tier1:
        new_state.phase = ServicePhase.post_check_grace
        new_state.post_grace_until_ts = inp.now + inp.post_grace_sec
        new_state.last_remediate_ts = inp.now
        new_state.attempts_this_hour += 1
        return TransitionOutput(new_state=new_state, attention_events=attention_events)

    if new_state.phase == ServicePhase.remediating_tier2:
        new_state.phase = ServicePhase.post_check_grace
        new_state.post_grace_until_ts = inp.now + inp.post_grace_sec
        new_state.last_remediate_ts = inp.now
        new_state.attempts_this_hour += 1
        return TransitionOutput(new_state=new_state, attention_events=attention_events)

    bad_signal = inp.equilibrium_bad or inp.probe_status == "probe_bad"
    if new_state.phase == ServicePhase.healthy:
        if bad_signal:
            new_state.phase = ServicePhase.suspect
            new_state.consecutive_probe_fails = 1 if inp.probe_status == "probe_bad" else 0
        return TransitionOutput(new_state=new_state, attention_events=attention_events)

    if new_state.phase == ServicePhase.suspect:
        if inp.probe_status == "probe_bad":
            new_state.consecutive_probe_fails += 1
        elif inp.probe_status == "probe_ok" and not inp.equilibrium_bad:
            new_state.phase = ServicePhase.healthy
            new_state.consecutive_probe_fails = 0
            return TransitionOutput(new_state=new_state, attention_events=attention_events)

        confirmed = (inp.equilibrium_bad and inp.probe_status == "probe_bad") or (
            new_state.consecutive_probe_fails >= inp.consecutive_probe_fails_threshold
        )
        if not confirmed:
            return TransitionOutput(new_state=new_state, attention_events=attention_events)

        if not inp.auto_remediate:
            new_state.phase = ServicePhase.attention_only
            attention_events.append(
                _attention_event(
                    severity="error",
                    message=(
                        f"mesh health: {service_id} unhealthy confirmed "
                        f"(observe-only, auto_remediate disabled)"
                    ),
                    service_id=service_id,
                    correlation_id=corr,
                    event="observe_only",
                )
            )
            return TransitionOutput(new_state=new_state, attention_events=attention_events)

        new_state.phase = ServicePhase.unhealthy_confirmed
        attention_events.append(
            _attention_event(
                severity="error",
                message=f"mesh health: {service_id} unhealthy confirmed",
                service_id=service_id,
                correlation_id=corr,
                event="unhealthy_confirmed",
            )
        )
        return TransitionOutput(new_state=new_state, attention_events=attention_events)

    if new_state.phase == ServicePhase.unhealthy_confirmed:
        if not inp.auto_remediate:
            new_state.phase = ServicePhase.attention_only
            attention_events.append(
                _attention_event(
                    severity="error",
                    message=f"mesh health: {service_id} observe-only (auto_remediate disabled)",
                    service_id=service_id,
                    correlation_id=corr,
                    event="observe_only",
                )
            )
            return TransitionOutput(new_state=new_state, attention_events=attention_events)

        if new_state.attempts_this_hour >= inp.max_attempts_per_hour:
            new_state.phase = ServicePhase.attention_only
            attention_events.append(
                _attention_event(
                    severity="error",
                    message=f"mesh health: {service_id} max remediation attempts reached",
                    service_id=service_id,
                    correlation_id=corr,
                    event="max_attempts",
                )
            )
            return TransitionOutput(new_state=new_state, attention_events=attention_events)

        cooldown_clear = (
            new_state.last_remediate_ts is None or inp.now - new_state.last_remediate_ts >= inp.cooldown_sec
        )
        if cooldown_clear:
            new_state.phase = ServicePhase.remediating_tier1
            should_remediate_tier1 = True
        return TransitionOutput(
            new_state=new_state,
            attention_events=attention_events,
            should_remediate_tier1=should_remediate_tier1,
        )

    return TransitionOutput(new_state=new_state, attention_events=attention_events)
