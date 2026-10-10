"""GPU actuation watch: catch a GPU lane controller that silently refuses every swap.

On 2026-10-09/10 circe's orion-gpu-lane-controller ran an image older than a new
field in config/gpu_pool.yaml (``roles.agent-gpu2.max_holds``). Its own config no
longer parsed, so it refused every actuation ``config_unloadable:ValidationError``
for 28 h (155 refusals, role agent-gpu2). The diffusion card never swapped, no
painting was made, and nothing alerted: the controller's /health stays ok and the
pool only logs and emits ``actuate_refused`` on ``orion:gpu_pool:event``. In the
16 days before, there were zero ``actuate_refused`` events.

Two views, both feeding the guardian's AlertGate (one card per key per window):

- ACTIVE (``ActiveProbeTracker``): every MESH_GUARDIAN_GPU_PROBE_INTERVAL_SEC, ask
  each actuated role's controller ``status`` then ``digest``
  (orion/gpu_pool/actuator_probe.py; read-only). Catches the breakage even when
  the pool asks for nothing.
- PASSIVE (``RefusalWatch``): the pool's own ``actuate_refused`` events, so a real
  refused swap raises a card within seconds, between probe cycles.

Keys are per role + kind and shared between the views, so the incident's 155
refusals plus its probe verdicts produce one card per window, not 155.

Pure (no I/O) so the incident can be replayed; collection lives in service.py.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from orion.gpu_pool.actuator_probe import Verdict

from .stability import StabilityAlert

NO_ANSWER_STREAK = 2              # consecutive cycles before "not answering" (one miss is a blip)
REFUSAL_BURST = 3                 # refusals for one role ...
REFUSAL_WINDOW_SEC = 60 * 60      # ... within this window
CONTROLLER_SERVICE = "orion-gpu-lane-controller"


@dataclass(frozen=True)
class RoleTarget:
    role: str
    actuator: str
    host: str


def role_targets(cfg) -> dict[str, RoleTarget]:
    """Every role with a ``launch`` block -> its actuator and that actuator's host."""
    out: dict[str, RoleTarget] = {}
    for name, spec in cfg.roles.items():
        if spec.launch is None:
            continue
        actuator = spec.launch.actuator
        act = cfg.actuators.get(actuator)
        out[name] = RoleTarget(role=name, actuator=actuator, host=act.host if act else actuator)
    return out


def _unloadable_alert(target: RoleTarget, reason: str, context: dict[str, Any]) -> StabilityAlert:
    return StabilityAlert(
        key=f"gpu_config_unloadable:{target.role}",
        subject=f"gpu-lane:{target.role}",
        kind="gpu_controller_config_unloadable",
        severity="critical",
        message=(
            f"The GPU lane controller on {target.host} cannot read its own config/gpu_pool.yaml "
            f"({reason}), so it refuses every request to move role {target.role} onto its card. "
            f"Nothing on that card will swap until it is fixed. Fix: rebuild {CONTROLLER_SERVICE} "
            f"on {target.host} from main; its image predates config/gpu_pool.yaml."
        ),
        context={"role": target.role, "actuator": target.actuator, "host": target.host,
                 "reason": reason, **context},
    )


def _mismatch_alert(target: RoleTarget, severity: str, context: dict[str, Any]) -> StabilityAlert:
    return StabilityAlert(
        key=f"gpu_digest_mismatch:{target.role}",
        subject=f"gpu-lane:{target.role}",
        kind="gpu_controller_digest_mismatch",
        severity=severity,
        message=(
            f"The GPU lane controller on {target.host} and this host disagree about how to launch "
            f"role {target.role} (launch_digest_mismatch): its checkout or image is on a different "
            f"commit. The pool's swaps for {target.role} are refused while they differ. Fix: pull "
            f"main on {target.host} and rebuild {CONTROLLER_SERVICE} there (or redeploy the pool if "
            f"it is the one behind)."
        ),
        context={"role": target.role, "actuator": target.actuator, "host": target.host,
                 "reason": "launch_digest_mismatch", **context},
    )


class ActiveProbeTracker:
    """Turns probe verdicts into alerts. A no-answer streak is per role + check and resets on any
    answer; config_unloadable / digest_mismatch alert on the first verdict."""

    def __init__(self) -> None:
        self._no_answer: dict[tuple[str, str], int] = {}

    def observe(self, target: RoleTarget, verdict: Verdict) -> list[StabilityAlert]:
        streak_key = (target.role, verdict.check)
        if verdict.kind != "no_answer":
            self._no_answer.pop(streak_key, None)
        ctx = {"check": verdict.check, "verdict": verdict.kind, "source": "active_probe"}
        if verdict.kind == "config_unloadable":
            return [_unloadable_alert(target, verdict.reason, ctx)]
        if verdict.kind == "digest_mismatch":
            return [_mismatch_alert(target, "error", ctx)]
        if verdict.kind == "no_answer":
            streak = self._no_answer.get(streak_key, 0) + 1
            self._no_answer[streak_key] = streak
            if streak < NO_ANSWER_STREAK:
                return []
            return [StabilityAlert(
                key=f"gpu_no_answer:{target.role}",
                subject=f"gpu-lane:{target.role}",
                kind="gpu_controller_no_answer",
                severity="error",
                message=(
                    f"The GPU lane controller on {target.host} is not answering (actuator "
                    f"{target.actuator}, role {target.role}, {streak} probe cycles in a row). The pool "
                    f"cannot swap anything on that card. Check that {CONTROLLER_SERVICE} is running on "
                    f"{target.host} and connected to the bus."
                ),
                context={"role": target.role, "actuator": target.actuator, "host": target.host,
                         "no_answer_streak": streak, **ctx},
            )]
        return []  # ok, or a transient refusal (busy, deadline_passed, ...): not ours to page on


class RefusalWatch:
    """Sliding per-role window over the pool's ``actuate_refused`` events (GpuPoolEventV1 dicts).
    In memory: a guardian restart starts the window empty, which only delays a burst card."""

    def __init__(self) -> None:
        self._seen: dict[str, deque[float]] = {}

    def observe(self, event: dict[str, Any], now: float, targets: dict[str, RoleTarget]) -> list[StabilityAlert]:
        if event.get("event") != "actuate_refused":
            return []
        role = str(event.get("role") or "unknown")
        reason = str(event.get("reason") or "")
        seen = self._seen.setdefault(role, deque())
        seen.append(now)
        while seen and seen[0] < now - REFUSAL_WINDOW_SEC:
            seen.popleft()
        count = len(seen)
        target = targets.get(role) or RoleTarget(role=role, actuator="unknown", host="the actuator host")
        ctx = {"source": "pool_event", "latest_reason": reason, "refusals_in_window": count,
               "window_min": REFUSAL_WINDOW_SEC // 60}
        # These two mean the controller cannot act at all until someone rebuilds/redeploys it, so
        # they page on the first refusal; busy / deadline_passed / stale_generation need a burst.
        if reason.startswith("config_unloadable:"):
            return [_unloadable_alert(target, reason, ctx)]
        if reason.startswith("launch_digest_mismatch"):
            return [_mismatch_alert(target, "critical", ctx)]
        if count < REFUSAL_BURST:
            return []
        return [StabilityAlert(
            key=f"gpu_refusal_burst:{role}",
            subject=f"gpu-lane:{role}",
            kind="gpu_actuate_refusal_burst",
            severity="error",
            message=(
                f"The GPU lane controller on {target.host} refused {count} requests to move role {role} "
                f"in the last {REFUSAL_WINDOW_SEC // 60} min (latest reason: {reason or 'none given'}). "
                f"Healthy baseline is zero. The card is not swapping; check {CONTROLLER_SERVICE} logs "
                f"on {target.host} for gpu_actuate_refused."
            ),
            context={"role": role, "actuator": target.actuator, "host": target.host, **ctx},
        )]
