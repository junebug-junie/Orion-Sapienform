"""The GPU pool's one shedding lever: named reasons that stop NEW grants to low priorities.

Plan: docs/superpowers/plans/2026-09-29-urgent-curiosity-plan-4-5-hardware-watch-and-shedding.md
("The shared shedding lever"). Scheduler rule U4 (orion/gpu_pool/scheduler.py) consumes only the
result: ``priority -> reason name``. Everything about who may shed, how strongly, and how the
reasons combine lives here.

- A reason is named, has a precedence (lower number wins) and a fixed set of priorities it blocks.
  Only ``background`` and ``system`` can ever be shed; a reason naming anything else is refused at
  import time. ``interactive`` (chat) and ``urgent`` are never shed.
- A signal is one producer's claim that a reason holds: ``(reason, source_id, detail, valid_until)``.
  The board keeps one per ``(reason, source_id)``; a signal past ``valid_until`` counts as absent,
  so a producer that dies stops shedding within its own validity window (fail-open).
- The effective shed is the UNION of the active reasons' blocks. Each blocked priority is
  attributed to the highest-precedence active reason that blocks it. A lower-precedence reason can
  add blocks but never lift one: there is no "unshed" signal, only a reason's own clear.
- ``enabled=False`` (GPU_POOL_SHED_ENABLED) turns the whole lever off: nothing is blocked, the
  signals are still shown so an operator can see what would have been shed.

Orion's learned "shed background GPU" action (attend-to-act loop A1, amended 2026-09-29) is the
one lower-precedence reason, ``orion_self_shed``: precedence 1, background only, set and cleared
by ``orion/gpu_pool/orion_shed.py`` through the pool's shed RPC -- never by an incident, and the
RPC can only name ``orion_self_shed``.

Thermal controller v2 (docs/superpowers/specs/2026-10-06-thermal-controller-redesign-design.md, D2):
the reflex reasons are ``cabinet_hot`` (cabinet >= 34 C, or unreadable with the AC low: background
+ system) and ``cabinet_unknown`` (cabinet unreadable past grace: background only), both precedence
0, asserted per tick by hardware-watch on ``orion:hardware:watch:reflex_shed``. ``cooling_incident``
is the v1 reflex: it stays ONLY while ``HARDWARE_WATCH_HEAT_CONTROLLER=v1`` is the one-week rollback
path, then it is deleted (spec "Rollback"); a v2 watcher never asserts it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

SHEDDABLE: tuple[str, ...] = ("background", "system")


@dataclass(frozen=True)
class ShedReasonSpec:
    name: str
    precedence: int                 # lower wins; 0 is reserved for physical-safety reflexes
    blocks: tuple[str, ...]         # priorities this reason stops new grants for
    description: str


SHED_REASONS: dict[str, ShedReasonSpec] = {
    "cabinet_hot": ShedReasonSpec(
        "cabinet_hot", 0, ("background", "system"),
        "orion-hardware-watch reflex (v2): the cabinet is at/above 34 C, or unreadable while the AC reads low"),
    "cabinet_unknown": ShedReasonSpec(
        "cabinet_unknown", 0, ("background",),
        "orion-hardware-watch reflex (v2): no cabinet reading past the grace window (counts as elevated)"),
    # v1 rollback path only (HARDWARE_WATCH_HEAT_CONTROLLER=v1); delete after v2 runs clean a week.
    "cooling_incident": ShedReasonSpec(
        "cooling_incident", 0, ("background", "system"),
        "orion-hardware-watch (v1 rollback): the cabinet AC incident is open and the cabinet is warming"),
    # Background only, never system: system work is Orion's own cognition/execution; shedding it on a
    # routine warm afternoon is the self-DOS failure mode (design, "Background only, not system").
    "orion_self_shed": ShedReasonSpec(
        "orion_self_shed", 1, ("background",),
        "Orion's learned action: the cabinet is elevated and rising with the AC healthy; no new "
        "background grants for a bounded TTL (caps enforced by the pool)"),
}


# Reasons the hardware-watch reflex asserts (precedence 0). Orion's learned shed is refused while any is active.
REFLEX_REASONS: frozenset[str] = frozenset({"cabinet_hot", "cabinet_unknown", "cooling_incident"})


def _validate(reasons: dict[str, ShedReasonSpec]) -> None:
    for key, spec in reasons.items():
        if key != spec.name:
            raise ValueError(f"shed reason key {key!r} != spec name {spec.name!r}")
        bad = [p for p in spec.blocks if p not in SHEDDABLE]
        if bad or not spec.blocks:
            raise ValueError(f"shed reason {key!r} may only block {SHEDDABLE}, got {spec.blocks}")


_validate(SHED_REASONS)


@dataclass(frozen=True)
class ShedSignal:
    reason: str
    source_id: str                  # e.g. the hardware-watch incident_id
    set_at: datetime
    valid_until: datetime
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ShedView:
    enabled: bool
    blocked: dict[str, str]         # priority -> reason name (what the scheduler sees)
    active_reason: str | None       # the highest-precedence active reason
    reasons: list[dict[str, Any]]   # every known reason, active or not, for status/health

    def as_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "active_reason": self.active_reason,
                "blocked": dict(self.blocked), "reasons": list(self.reasons)}


class ShedBoard:
    def __init__(self, reasons: dict[str, ShedReasonSpec] | None = None):
        self.reasons = dict(reasons or SHED_REASONS)
        _validate(self.reasons)
        self._signals: dict[tuple[str, str], ShedSignal] = {}

    def set(self, signal: ShedSignal) -> bool:
        """Record (or refresh) a signal. False for an unknown reason (logged by the caller)."""
        if signal.reason not in self.reasons:
            return False
        self._signals[(signal.reason, signal.source_id)] = signal
        return True

    def clear(self, reason: str, source_id: str) -> bool:
        return self._signals.pop((reason, source_id), None) is not None

    def prune(self, now: datetime) -> list[ShedSignal]:
        """Drop expired signals; return them so the caller can log the lapse."""
        gone = [s for s in self._signals.values() if s.valid_until <= now]
        for s in gone:
            self._signals.pop((s.reason, s.source_id), None)
        return gone

    def view(self, now: datetime, enabled: bool) -> ShedView:
        live = [s for s in self._signals.values() if s.valid_until > now]
        by_reason: dict[str, list[ShedSignal]] = {}
        for s in live:
            by_reason.setdefault(s.reason, []).append(s)
        ordered = sorted(self.reasons.values(), key=lambda r: (r.precedence, r.name))
        blocked: dict[str, str] = {}
        active: str | None = None
        rows: list[dict[str, Any]] = []
        for spec in ordered:
            sigs = sorted(by_reason.get(spec.name, []), key=lambda s: s.set_at)
            is_active = bool(sigs)
            if is_active and enabled:
                active = active or spec.name
                for p in spec.blocks:
                    blocked.setdefault(p, spec.name)
            rows.append({
                "name": spec.name, "precedence": spec.precedence, "blocks": list(spec.blocks),
                "active": is_active, "effective": is_active and enabled,
                "sources": [{"source_id": s.source_id, "set_at": s.set_at.isoformat(),
                             "valid_until": s.valid_until.isoformat(), "detail": s.detail} for s in sigs],
            })
        return ShedView(enabled, blocked, active, rows)
