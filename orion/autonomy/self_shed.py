"""When may Orion's learned shed action be proposed? Pure rules, no I/O, no clock.

Spec: docs/superpowers/specs/2026-09-29-attend-to-act-loop-design.md -- D1 (binding to the
workspace winner) and "Amendment 2026-09-29" ("Trigger", "Mutual exclusion"). The amendment is
authoritative where it differs from the original.

Two questions, answered separately so the frame can say which one failed:

1. ``bind_workspace_winner`` -- is the workspace broadcast winner bindable for this template?
   An action was selected (not ``none``), the projection is <= 90 s old (three broadcast ticks),
   the coalition held >= 2 ticks, and an attended node is in the template's ``binds_to_nodes``.
2. ``evaluate_shed_eligibility`` -- is the world in the state the action is for? Thermal controller
   v2, D8 (APPROVED 2026-10-06, docs/superpowers/specs/2026-10-06-thermal-controller-redesign-design.md):
   Orion's learned shed owns the 29.5-34 C band. The cabinet is ``elevated``, ``hot`` below the
   reflex's 34 C critical line, or ``unknown``; no rise is required (in the band the decision is the
   attend-to-act loop's, not a fixed rise threshold); hardware-watch is reachable and ticking, and
   the reflex's OWN shed signal (``cabinet_hot``/``cabinet_unknown``, or a v1 cooling incident that
   requests shedding) is not active -- any other open incident (a gpu_heat outlier) no longer
   disables it; at least one background lease is granted or queued; and no earlier episode of this
   action is still in flight.

The returned snapshot is persisted on the proposal, the dispatch candidate and the episode row, so
treated and control rows can be shown to come from the same population.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

from orion.autonomy.cabinet_heat import CabinetHeatReading
from orion.schemas.proposal_frame import AttentionWinnerRefV1

WINNER_MAX_AGE_SEC = 90.0
WINNER_MIN_DWELL_TICKS = 2
# hardware-watch ticks every ~30 s; three missed ticks means we cannot tell the reflex is idle.
HARDWARE_WATCH_MAX_TICK_AGE_SEC = 180.0
# The one action this module rules on.
SHED_TEMPLATE = "shed_background_gpu"
# D8: the band Orion's learned shed owns (below the reflex's critical line).
SHED_ELIGIBLE_STATES = frozenset({"elevated", "hot", "unknown"})


def _parse_ts(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def bind_workspace_winner(
    projection: Mapping[str, Any] | None,
    *,
    broadcast_log_id: str | None,
    binds_to_nodes: Sequence[str],
    now: datetime,
) -> tuple[AttentionWinnerRefV1 | None, str | None]:
    """(winner, None) when bindable, else (None, ``winner_unbindable:<reason>``)."""
    if not projection:
        return None, "winner_unbindable:no_projection"
    action = str(projection.get("selected_action_type") or "none")
    if action == "none":
        return None, "winner_unbindable:no_selected_action"
    loop_id = projection.get("selected_open_loop_id")
    if not loop_id:
        return None, "winner_unbindable:no_open_loop"
    generated = _parse_ts(projection.get("generated_at"))
    if generated is None:
        return None, "winner_unbindable:no_timestamp"
    age = (now - generated).total_seconds()
    if age < 0 or age > WINNER_MAX_AGE_SEC:
        return None, f"winner_unbindable:stale_{max(age, 0):.0f}s"
    dwell = int(projection.get("dwell_ticks") or 0)
    if dwell < WINNER_MIN_DWELL_TICKS:
        return None, f"winner_unbindable:dwell_{dwell}"
    # attended_node_ids IS the selected loop's source_refs ([node_id] + contributing ids): its first
    # node ref is the node the selected loop is about. Binding to any other member would answer a
    # different loop than the one the action's verdict is written on.
    attended = [str(n) for n in projection.get("attended_node_ids") or []]
    node = next((n for n in attended if n.startswith("node:")), None)
    if node is None or node not in set(binds_to_nodes):
        return None, "winner_unbindable:node_not_bound"
    if not broadcast_log_id:
        return None, "winner_unbindable:no_broadcast_log_row"
    return AttentionWinnerRefV1(
        broadcast_log_id=broadcast_log_id,
        open_loop_id=str(loop_id),
        node_id=node,
        generated_at=generated,
        dwell_ticks=dwell,
        selected_action_type=action,
        age_sec=round(max(age, 0.0), 1),
    ), None


@dataclass(frozen=True)
class HardwareWatchView:
    """What the builder could read from hardware-watch's /health. ``reachable=False`` -> unknown."""

    reachable: bool
    enabled: bool = False
    last_tick_ok: bool = False
    last_tick_age_sec: float | None = None
    open_incidents: tuple[dict[str, Any], ...] = ()
    error: str | None = None
    # The reflex shed reason hardware-watch is asserting right now (v2 /health ``reflex_shed``), or a
    # v1 cooling incident's shed reason while it requests shedding. None = the reflex is not shedding.
    reflex_reason: str | None = None

    @classmethod
    def from_health(cls, health: Mapping[str, Any] | None, *, now: datetime, error: str | None = None) -> "HardwareWatchView":
        if not health:
            return cls(reachable=False, error=error or "unreachable")
        tick = _parse_ts(health.get("last_tick_at"))
        incidents = tuple(dict(i) for i in health.get("open_incidents") or [])
        reflex = health.get("reflex_shed") or {}
        reason = reflex.get("reason") if reflex.get("active") else None
        if reason is None:   # a v1 watcher: the reflex is a cooling incident that requests shedding
            reason = next((f"cooling_incident:{i.get('shed_reason')}" for i in incidents
                           if i.get("rule") == "cooling" and i.get("shed_requested")), None)
        return cls(
            reachable=True,
            enabled=bool(health.get("enabled")),
            last_tick_ok=bool(health.get("last_tick_ok")),
            last_tick_age_sec=None if tick is None else (now - tick).total_seconds(),
            open_incidents=incidents,
            reflex_reason=None if reason is None else str(reason)[:64],
        )

    def idle_refusal(self) -> str | None:
        """None when the reflex is known idle; otherwise why it is not (or cannot be known)."""
        if not self.reachable:
            return "hardware_watch_unknown:unreachable"
        if not self.enabled:
            return "hardware_watch_unknown:disabled"
        if not self.last_tick_ok:
            return "hardware_watch_unknown:last_tick_failed"
        if self.last_tick_age_sec is None or self.last_tick_age_sec > HARDWARE_WATCH_MAX_TICK_AGE_SEC:
            return "hardware_watch_unknown:stale"
        if self.reflex_reason:
            return f"reflex_active:{self.reflex_reason}"
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "reachable": self.reachable, "enabled": self.enabled, "last_tick_ok": self.last_tick_ok,
            "last_tick_age_sec": None if self.last_tick_age_sec is None else round(self.last_tick_age_sec, 1),
            "open_incident_ids": sorted(str(i.get("incident_id")) for i in self.open_incidents),
            "open_incident_rules": sorted({str(i.get("rule")) for i in self.open_incidents}),
            "reflex_reason": self.reflex_reason,
            "error": self.error,
        }


def evaluate_shed_eligibility(
    *,
    cabinet: CabinetHeatReading,
    hardware_watch: HardwareWatchView,
    background_granted: int | None,
    background_queued: int | None,
    in_flight_episode_ids: Sequence[str],
    holdback_fraction: float,
    now: datetime,
) -> dict[str, Any]:
    """The eligibility snapshot. ``eligible`` is True only when ``refusals`` is empty."""
    refusals: list[str] = []
    if cabinet.thermal_state not in SHED_ELIGIBLE_STATES:
        refusals.append(f"thermal_not_elevated:{cabinet.thermal_state}")
    elif cabinet.critical:
        # >= 34 C: the reflex already blocks background (and system); Orion's signal adds nothing.
        refusals.append("thermal_critical:reflex_covers")
    idle = hardware_watch.idle_refusal()
    if idle:
        refusals.append(idle)
    if background_granted is None or background_queued is None:
        refusals.append("pool_occupancy_unknown")
    elif background_granted + background_queued < 1:
        refusals.append("no_background_work")
    if in_flight_episode_ids:
        refusals.append("winner_loop_in_flight")
    return {
        "template": SHED_TEMPLATE,
        "eligible": not refusals,
        "refusals": refusals,
        "evaluated_at": now.isoformat(),
        "cabinet": cabinet.as_dict(),
        "hardware_watch": hardware_watch.as_dict(),
        "pool": {"background_granted": background_granted, "background_queued": background_queued},
        "in_flight_episode_ids": list(in_flight_episode_ids)[:10],
        "holdback_fraction": holdback_fraction,
    }


__all__ = [
    "HARDWARE_WATCH_MAX_TICK_AGE_SEC",
    "HardwareWatchView",
    "SHED_ELIGIBLE_STATES",
    "SHED_TEMPLATE",
    "WINNER_MAX_AGE_SEC",
    "WINNER_MIN_DWELL_TICKS",
    "bind_workspace_winner",
    "evaluate_shed_eligibility",
]
