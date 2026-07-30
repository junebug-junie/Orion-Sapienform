"""graph_delta producer: polls ``graphify-out/graph.json``'s own
``built_at_commit`` tag on an interval, publishes a
``CodebaseDeltaV1(domain="graph")`` event only when graphify has actually
re-run since the last check -- event-triggered off a real content change,
not a fixed-cadence diff like pr_lifecycle.

State (the last-seen ``GraphSnapshotStats``) is caller-owned, kept in-process
only -- **deliberately not appended to ``snapshot_history.py``'s JSONL log
here.** That log lives inside this service's read-only repo mount
(``COCREATION_SIGNALS_REPO_PATH``); a live producer cannot write into a
read-only mount, and mounting it read-write just for one small log file would
widen this service's write surface for no real benefit. The JSONL log
utility (``append_snapshot``/``read_snapshots``) remains for the offline
backfill/replay scripts, which run against a real writable checkout, not for
this live producer's own tick-to-tick bookkeeping.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.codebase_delta import CodebaseDeltaV1, GraphDeltaPayloadV1
from orion.structural_mass.graph_delta import graph_structural_delta
from orion.structural_mass.snapshot_history import (
    GraphSnapshotStats,
    graph_snapshot_stats_from_text,
)

logger = logging.getLogger("orion.cocreation_signals.graph_delta")

_GRAPH_JSON_RELPATH = "graphify-out/graph.json"
_GRAPH_REPORT_RELPATH = "graphify-out/GRAPH_REPORT.md"
_BUILT_AT_COMMIT_RE = re.compile(r'"built_at_commit"\s*:\s*"([^"]*)"')


def _read_built_at_commit(graph_json_path: Path) -> str | None:
    """Cheap check: regex over the raw file text for the built_at_commit tag,
    not a full `json.loads` of a 28k+-node file on every poll tick. Full
    parsing only happens once a real change is actually detected."""
    text = graph_json_path.read_text(encoding="utf-8")
    match = _BUILT_AT_COMMIT_RE.search(text)
    return match.group(1) if match else None


def _load_snapshot(repo_path: str, commit_sha: str | None) -> GraphSnapshotStats:
    graph_json_text = (Path(repo_path) / _GRAPH_JSON_RELPATH).read_text(encoding="utf-8")
    report_path = Path(repo_path) / _GRAPH_REPORT_RELPATH
    graph_report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    return graph_snapshot_stats_from_text(
        graph_json_text, graph_report_text, commit_sha=commit_sha, backfilled=False
    )


async def _publish(bus: OrionBusAsync, channel: str, source: ServiceRef, delta) -> bool:
    """Returns True if communicated (real publish or bus intentionally
    disabled), False only on a real transient publish failure -- see
    ``git_delta._publish()``'s docstring for why the caller must not advance
    ``last_snapshot`` on False."""
    if not getattr(bus, "enabled", False):
        logger.info("cocreation_graph_delta_publish_skipped_bus_disabled")
        return True
    event = CodebaseDeltaV1(
        domain="graph",
        observed_at=datetime.now(timezone.utc),
        graph=GraphDeltaPayloadV1(
            node_count_delta=delta.node_count_delta,
            edge_count_delta=delta.edge_count_delta,
            community_count_delta=delta.community_count_delta,
            god_node_jaccard_similarity=delta.god_node_jaccard_similarity,
            god_nodes_entered=list(delta.god_nodes_entered) if delta.god_nodes_entered is not None else None,
            god_nodes_exited=list(delta.god_nodes_exited) if delta.god_nodes_exited is not None else None,
        ),
    )
    envelope = BaseEnvelope(
        kind="substrate.codebase_delta.v1", source=source, payload=event.model_dump(mode="json")
    )
    try:
        await bus.publish(channel, envelope)
        logger.info(
            "cocreation_graph_delta_published node_delta=%+d edge_delta=%+d god_node_jaccard=%s",
            delta.node_count_delta, delta.edge_count_delta, delta.god_node_jaccard_similarity,
        )
        return True
    except Exception:
        logger.exception("cocreation_graph_delta_publish_failed")
        return False


async def graph_delta_loop(
    *,
    bus: OrionBusAsync,
    channel: str,
    source: ServiceRef,
    repo_path: str,
    poll_interval_sec: float,
    stop: asyncio.Event,
) -> None:
    last_snapshot: GraphSnapshotStats | None = None
    graph_json_path = Path(repo_path) / _GRAPH_JSON_RELPATH
    while not stop.is_set():
        try:
            if not graph_json_path.exists():
                logger.warning("cocreation_graph_delta_graph_json_missing path=%s", graph_json_path)
            else:
                commit_sha = await asyncio.to_thread(_read_built_at_commit, graph_json_path)
                if last_snapshot is None:
                    last_snapshot = await asyncio.to_thread(_load_snapshot, repo_path, commit_sha)
                    logger.info("cocreation_graph_delta_cold_start commit_sha=%s", commit_sha)
                elif commit_sha != last_snapshot.commit_sha:
                    curr_snapshot = await asyncio.to_thread(_load_snapshot, repo_path, commit_sha)
                    delta = graph_structural_delta(last_snapshot, curr_snapshot)
                    published = await _publish(bus, channel, source, delta)
                    if published:
                        last_snapshot = curr_snapshot
                    # else: leave last_snapshot unchanged, same reasoning as
                    # git_delta_loop's identical guard.
        except Exception:
            logger.exception("cocreation_graph_delta_tick_failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=poll_interval_sec)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break
