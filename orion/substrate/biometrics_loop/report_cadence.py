"""Which nodes have stopped reporting, judged against each node's own report rhythm.

Why this exists: hecate is an on-demand GPU server. When it is not needed it is powered
down to standby -- the OS is off, only its BMC and PDU outlets are live. Its
orion-biometrics agent stops, so its entry in `substrate_node_biometrics_projection`
freezes at the last report (live: last_seen_at 2026-10-15T06:31:03Z, 9 pressure_hints).
`biometrics_prediction_error()` still diffed those 9 frozen keys every tick and averaged
in 9 zero deltas next to the live nodes' ~26 keys, so the body read about a quarter
calmer than it was. The standby wattage (95-97 W, polled by athena's PDU proxy) lands
only in `orion_biometrics_cluster.measurements_by_node`, never in this projection, so
there is no BMC key here to keep or drop.

"Stopped reporting" is relative, not an absolute number of seconds: a node is silent
when its last report is more than `multiple` times its own typical report interval
behind the newest report any node has sent. The interval is learned from the projection's own `last_seen_at` advances
(median of the last 16, needing at least 3, so one long gap -- a node waking from a 5-day standby -- does
not move it). A node with no interval learned yet in this process (hecate after a
restart while it is down) is judged against the fleet's median interval: every node
runs the same orion-biometrics agent. With nothing learned at all (the first tick after
a restart), nothing is excluded, which is the pre-2026-10-20 behaviour.

The tracker observes the projection once per runtime tick. When one tick's batch
holds several reports from a node (catch-up after a stall; batch limit 500, about 61
events per report), it records one long interval instead of several short ones. The
learned interval can only drift up from that, which delays exclusion -- the safe
direction. All events of one report share one `observed_at`, so a report split
across two ticks does not create a false short interval.

Live cadence, 3 days of grammar_events to 2026-10-20: athena median 31.24 s, circe
31.32 s; the longest gap either had was 94 s (3x). `DEFAULT_MULTIPLE = 6` is twice that
worst observed gap and lines up with the existing 180 s `biometrics_node_stale_after_sec`.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from statistics import median

from orion.schemas.biometrics_projection import NodeBiometricsProjectionV1

DEFAULT_MULTIPLE = 6.0
DEFAULT_WINDOW = 16
# A node's own interval is trusted only once this many have been seen. The first
# interval after a node wakes from standby is the whole standby (days); with fewer
# samples the median would be that gap and the node could never read as silent again.
MIN_OWN_INTERVALS = 3


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


class NodeReportCadence:
    """Per-node report interval, learned from successive `last_seen_at` values."""

    def __init__(self, window: int = DEFAULT_WINDOW) -> None:
        self._window = max(1, int(window))
        self._last_seen: dict[str, datetime] = {}
        self._intervals: dict[str, deque[float]] = {}

    def observe(self, projection: NodeBiometricsProjectionV1) -> None:
        for node_id, state in (projection.nodes or {}).items():
            if state.last_seen_at is None:
                continue
            seen = _aware(state.last_seen_at)
            prior = self._last_seen.get(node_id)
            if prior is not None and seen > prior:
                self._intervals.setdefault(node_id, deque(maxlen=self._window)).append(
                    (seen - prior).total_seconds()
                )
            if prior is None or seen > prior:
                self._last_seen[node_id] = seen

    def cadence_sec(self, node_id: str) -> float | None:
        intervals = self._intervals.get(node_id)
        if not intervals or len(intervals) < MIN_OWN_INTERVALS:
            return None
        return float(median(intervals))

    def fleet_cadence_sec(self) -> float | None:
        own = [c for c in (self.cadence_sec(n) for n in self._intervals) if c]
        return float(median(own)) if own else None


def silent_nodes(
    projection: NodeBiometricsProjectionV1,
    *,
    cadence: NodeReportCadence,
    multiple: float = DEFAULT_MULTIPLE,
) -> frozenset[str]:
    """Nodes whose last report is more than `multiple` x their own report interval
    behind the newest report in the projection.

    Their `pressure_hints` are the last values they sent, not readings of now. A node
    with no `last_seen_at` is skipped (it has never reported, so it has no hints).

    Age is measured against the newest `last_seen_at` in the projection, not the wall
    clock. After a worker restart or a stall the runtime replays a backlog, and every
    node's report is minutes or hours old by the wall clock; against the clock, every
    node would read silent and the error would be a false 0.0. Against the newest
    report, a node is silent only relative to the nodes that are still reporting. For
    the same reason the newest reporter itself can never be silent, so at least one
    node always counts.
    """
    seen = [
        _aware(s.last_seen_at) for s in (projection.nodes or {}).values() if s.last_seen_at
    ]
    if not seen:
        return frozenset()
    newest = max(seen)
    fleet = cadence.fleet_cadence_sec()
    out: set[str] = set()
    for node_id, state in (projection.nodes or {}).items():
        if state.last_seen_at is None:
            continue
        interval = cadence.cadence_sec(node_id) or fleet
        if not interval or interval <= 0:
            continue
        age = (newest - _aware(state.last_seen_at)).total_seconds()
        if age > multiple * interval:
            out.add(node_id)
    return frozenset(out)
