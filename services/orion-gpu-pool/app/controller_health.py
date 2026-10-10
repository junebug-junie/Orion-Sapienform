"""Is the seat's actuator (the GPU lane controller) able to act on what the pool asks?

Incident 2026-10-09 03:56 -> 2026-10-10 06:01: the lane controller on circe ran code built
2026-10-08 against a checkout that had since been pulled to stage 7.3 config. Its
``pool_fence.load_config`` raised ValidationError, so it refused every request with
``config_unloadable:ValidationError``. The pool backed off 10 min and asked again -- 136 times --
while the agent seat granted nothing and agent work waited 3-12 h. ``/health`` said ok the whole
time and nobody was told.

A refusal like that is not something a retry fixes: it means the controller cannot read, or does
not agree with, the config the pool acts on. This tracker counts those refusals per seat, marks the
seat degraded after ``DEGRADE_AFTER`` in a row, and clears it the moment the controller answers
anything that proves it read its config (a succeeded or failed action, a succeeded status).
Refusals that a retry does fix (``busy``, ``deadline_passed``, ``upstream_not_idle:*``, ...)
neither trip nor clear it. Pure: no clock, no I/O; the runtime calls it and does the alerting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Two in a row, not one: a controller reading the YAML in the middle of a `git pull` can fail
# once and be fine on the next read. With the pool's swap_cooldown_sec (600) between attempts,
# the second refusal -- and the alert -- lands ~10 min after the first.
DEGRADE_AFTER = 2

# Refusal reason -> kind. Prefix match on the part before ':' (the controller appends the
# exception type or role name). Explicit on purpose: an unknown reason is neutral, never a page.
# Reasons from services/orion-gpu-lane-controller/app/actuator_bus.py and app/pool_fence.py.
_KINDS: dict[str, str] = {
    "config_unloadable": "config_unreadable",
    "fence_state_unreadable": "config_unreadable",
    "launch_digest_mismatch": "config_mismatch",
    "profile_not_allowed": "config_mismatch",
    "unknown_role": "config_mismatch",
    "role_not_on_this_actuator": "config_mismatch",
    "cards_mismatch": "config_mismatch",
    "no_launch_block": "config_mismatch",
    "not_a_swap_seat": "config_mismatch",
    # The controller's GpuActuateV1 rejected the pool's request: its schema is older than the pool's.
    "invalid_request": "request_rejected",
}


def classify(reason: str | None) -> str | None:
    """The kind of controller breakage a refusal reason means, or None for one a retry can fix."""
    if not reason:
        return None
    return _KINDS.get(reason.split(":", 1)[0].strip())


def advice(kind: str, *, seat: str, host: str, reason: str) -> str:
    """One plain sentence: what is broken and what to do about it."""
    if kind == "config_unreadable":
        return (f"The GPU lane controller on {host} can't read its config ({reason}): its code is older "
                f"than the checkout it reads. Rebuild the controller on {host}. Until then the pool "
                f"cannot load or unload {seat}.")
    if kind == "request_rejected":
        return (f"The GPU lane controller on {host} rejects the pool's requests ({reason}): its code is "
                f"older than the pool's. Rebuild the controller on {host}. Until then the pool cannot "
                f"load or unload {seat}.")
    return (f"The GPU lane controller on {host} and the pool disagree about {seat}'s config ({reason}). "
            f"Pull the same commit on both hosts, then rebuild the controller on {host}. Until then "
            f"the pool cannot load or unload {seat}.")


@dataclass
class SeatTrouble:
    seat: str
    host: str
    kind: str
    reason: str
    count: int
    first_seen: datetime
    last_seen: datetime
    degraded_since: datetime | None = None
    alerted: bool = False

    def view(self) -> dict[str, Any]:
        return {"degraded": self.degraded_since is not None, "kind": self.kind, "reason": self.reason,
                "refusals": self.count, "first_seen": self.first_seen.isoformat(),
                "last_seen": self.last_seen.isoformat(),
                "degraded_since": self.degraded_since.isoformat() if self.degraded_since else None,
                "host": self.host,
                "advice": advice(self.kind, seat=self.seat, host=self.host, reason=self.reason)}


@dataclass
class ControllerHealth:
    threshold: int = DEGRADE_AFTER
    seats: dict[str, SeatTrouble] = field(default_factory=dict)

    def on_refused(self, seat: str, reason: str | None, *, host: str, now: datetime) -> SeatTrouble | None:
        """Count one refusal. Returns the record when THIS refusal made the seat degraded (the
        caller alerts exactly then), else None."""
        kind = classify(reason)
        if kind is None:
            return None
        t = self.seats.get(seat)
        if t is None:
            t = self.seats[seat] = SeatTrouble(seat=seat, host=host, kind=kind, reason=reason or "",
                                               count=0, first_seen=now, last_seen=now)
        t.count += 1
        t.last_seen, t.kind, t.reason, t.host = now, kind, reason or "", host
        if t.degraded_since is None and t.count >= self.threshold:
            t.degraded_since = now
            return t
        return None

    def on_answered(self, seat: str) -> SeatTrouble | None:
        """The controller read its config and acted (or reported). Returns the record it clears when
        the seat WAS degraded (the caller logs the recovery), else None."""
        t = self.seats.pop(seat, None)
        return t if t is not None and t.degraded_since is not None else None

    def degraded(self) -> dict[str, SeatTrouble]:
        return {s: t for s, t in self.seats.items() if t.degraded_since is not None}

    def view(self) -> dict[str, dict[str, Any]]:
        return {s: t.view() for s, t in sorted(self.seats.items())}
