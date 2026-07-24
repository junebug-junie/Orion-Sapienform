from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from orion.schemas.telemetry.system_health import (
    EquilibriumServiceState,
    EquilibriumServiceTransitionV1,
)


def _service_key(service: str, node: Optional[str]) -> str:
    return f"{service}@{node or 'unknown'}"


@dataclass
class DowntimeTransitionTracker:
    """Detects real `status` transitions across successive equilibrium
    snapshot ticks and builds the corresponding `EquilibriumServiceTransitionV1`
    events.

    Deliberately holds no bus/db dependency -- pure in-memory state plus a
    detect() call, so it can be exercised with plain unit tests. The caller
    (EquilibriumService._publish_snapshot) owns publishing.

    Ownership/cadence note: this must be driven from exactly one call site
    per publish tick (the `_publish_snapshot` loop, on
    `EQUILIBRIUM_PUBLISH_INTERVAL_SEC`). `EquilibriumService._calculate_metrics()`
    is called from several other places (metacognition tick, baseline metacog
    trigger, per-message metacog handling) on independent cadences; feeding
    those same `states` lists through `detect()` too would fire duplicate/
    spurious transitions every time any of those paths happens to run in the
    same status window. `detect()` is intentionally not called from those
    other paths.
    """

    _last_status: Dict[str, str] = field(default_factory=dict)
    _down_since: Dict[str, datetime] = field(default_factory=dict)

    def detect(
        self,
        states: List[EquilibriumServiceState],
        *,
        now: datetime,
        source_service: str,
        source_node: Optional[str],
        producer_boot_id: str,
    ) -> List[EquilibriumServiceTransitionV1]:
        """Compare each state's current status against the last-observed
        status for that (service, node) pair and return one transition event
        per real change. A service seen for the first time seeds tracking
        state but never fires a transition (there is no real "from" status
        to report, and treating first-sight as a transition would spuriously
        fire on every service the moment this tracker boots). A repeated
        identical status is a no-op -- this is the case most likely to have
        an off-by-one/duplicate-firing bug, so it is exercised directly by
        tests, not just implied by the other cases.
        """
        transitions: List[EquilibriumServiceTransitionV1] = []

        for state in states:
            key = _service_key(state.service, state.node)
            prev_status = self._last_status.get(key)

            if prev_status is None:
                # First observation of this service -- seed tracking only.
                self._last_status[key] = state.status
                if state.status == "down":
                    self._down_since.setdefault(key, now)
                continue

            if prev_status == state.status:
                # No change -- must not fire, this is the duplicate-firing
                # trap the design doc called out explicitly.
                continue

            down_since_ts: Optional[datetime] = None
            down_duration_ms: Optional[int] = None
            if prev_status == "down":
                down_since_ts = self._down_since.pop(key, None)
                if down_since_ts is not None:
                    down_duration_ms = max(
                        0, int((now - down_since_ts).total_seconds() * 1000.0)
                    )

            if state.status == "down":
                self._down_since[key] = now

            transitions.append(
                EquilibriumServiceTransitionV1(
                    service=state.service,
                    node=state.node,
                    boot_id=state.boot_id,
                    version=state.version,
                    instance=state.instance,
                    from_status=prev_status,
                    to_status=state.status,
                    transitioned_at=now,
                    down_since_ts=down_since_ts,
                    down_duration_ms=down_duration_ms,
                    source_service=source_service,
                    source_node=source_node,
                    producer_boot_id=producer_boot_id,
                )
            )
            self._last_status[key] = state.status

        return transitions
