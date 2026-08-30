"""Live bus-synaptic-graph anomaly awareness for chat reasoning.

Idea 4 of docs/superpowers/specs/2026-07-24-bus-vitality-field-signal-brainstorm.md's
Phase 3+ arc. Design doc: docs/superpowers/specs/2026-07-24-bus-synaptic-graph-
reasoning-consumer-design.md (proposal-mode pass, per this repo's CLAUDE.md
section 0A -- first idea in that arc touching a reasoning pipeline, not just
infrastructure telemetry.

Derived from the anomaly-detection Cypher live-verified in
services/orion-hub/scripts/bus_synaptic_graph_routes.py::anomalies(), with an
extra recency floor on ``last_seen_epoch`` so long-dead edges (e.g.
``orion:dream:log`` frozen at z=36 for weeks) do not reach unified-turn recall.
Fixed, parameterized queries only -- never free-form Cypher.

Unlike falkor_neighborhood_adapter.py (this module's closest sibling, which
keyword-matches the user's query_text), this fetch is deliberately
unconditional -- self-awareness of transport-layer stress isn't naturally
"about" what the user said. Called once per recall invocation (effectively
every chat turn), returns [] on the common case where nothing is
anomalous -- an empty list is the correct, honest output, not a gap to fill
with "nothing found" filler content.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from orion.graph.falkor_client import RedisGraphQueryClient

logger = logging.getLogger(__name__)

_CLIENT: Optional[RedisGraphQueryClient] = None

_DEFAULT_MAX_EDGE_AGE_SEC = 86400.0  # 24h -- stale frozen z-scores age out

_PUBLISH_ANOMALY_QUERY = """
MATCH (o:Organ)-[e:PUBLISHES]->(c:Channel)
WHERE e.gap_zscore IS NOT NULL
  AND abs(e.gap_zscore) > $threshold
  AND e.count > $min_count
  AND e.last_seen_epoch IS NOT NULL
  AND e.last_seen_epoch >= $min_last_seen_epoch
RETURN o.organ_id AS organ_id, c.channel AS channel,
       e.gap_zscore AS zscore, e.count AS count, e.last_seen_epoch AS last_seen_epoch
ORDER BY abs(e.gap_zscore) DESC
LIMIT $limit
"""

_CAUSAL_ANOMALY_QUERY = """
MATCH (a:Organ)-[e:CAUSALLY_FOLLOWED_BY]->(b:Organ)
WHERE e.latency_zscore IS NOT NULL
  AND abs(e.latency_zscore) > $threshold
  AND e.count > $min_count
  AND e.last_seen_epoch IS NOT NULL
  AND e.last_seen_epoch >= $min_last_seen_epoch
RETURN a.organ_id AS source_organ, b.organ_id AS target_organ,
       e.latency_zscore AS zscore, e.count AS count, e.last_seen_epoch AS last_seen_epoch
ORDER BY abs(e.latency_zscore) DESC
LIMIT $limit
"""


def _max_edge_age_sec() -> float:
    raw = os.getenv("RECALL_BUS_SYNAPTIC_ANOMALY_MAX_AGE_SEC", str(_DEFAULT_MAX_EDGE_AGE_SEC)).strip()
    try:
        return max(60.0, float(raw))
    except ValueError:
        return _DEFAULT_MAX_EDGE_AGE_SEC


def get_bus_synaptic_falkor_client() -> Optional[RedisGraphQueryClient]:
    """Return (or lazily initialise) the process-level ``orion_bus_synapse``
    Falkor client -- same lazy-singleton, never-raises, self-healing-on-retry
    shape as recall_falkor_store.py::get_recall_falkor_client(), reading env
    directly (this service's established convention, not a pydantic Settings
    field) rather than via app.settings.
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    uri = os.getenv("FALKORDB_URI", "").strip()
    graph_name = os.getenv("FALKORDB_BUS_GRAPH", "orion_bus_synapse").strip()
    if not uri:
        logger.debug("bus_synaptic_falkor_init_skipped reason=no_falkordb_uri")
        return None
    try:
        _CLIENT = RedisGraphQueryClient(uri=uri, graph_name=graph_name)
    except Exception as exc:
        logger.debug("bus_synaptic_falkor_init_failed error=%s", exc)
        return None
    return _CLIENT


def _format_age_phrase(age_sec: float) -> str:
    if age_sec < 120:
        return "just now"
    if age_sec < 3600:
        return f"{int(age_sec // 60)} min ago"
    if age_sec < 86400:
        return f"{int(age_sec // 3600)} h ago"
    return f"{int(age_sec // 86400)} d ago"


def _iso_from_epoch(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    try:
        return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _format_publish_anomaly_text(row: Dict[str, Any], *, now_epoch: float) -> str:
    zscore = float(row.get("zscore") or 0.0)
    count = int(row.get("count") or 0)
    last_seen = row.get("last_seen_epoch")
    age_phrase = "unknown age"
    if last_seen is not None:
        try:
            age_phrase = _format_age_phrase(max(0.0, now_epoch - float(last_seen)))
        except (TypeError, ValueError):
            pass
    return (
        f"Bus synaptic snapshot (not live traffic): channel \"{row.get('channel')}\" from "
        f"{row.get('organ_id')} had an unusual inter-arrival gap at its last publish "
        f"(|z|={abs(zscore):.1f} vs EWMA baseline, {count} samples, last seen {age_phrase}). "
        f"High z here means gap surprise at that publish, not current publish rate."
    )


def _format_causal_anomaly_text(row: Dict[str, Any], *, now_epoch: float) -> str:
    zscore = float(row.get("zscore") or 0.0)
    count = int(row.get("count") or 0)
    last_seen = row.get("last_seen_epoch")
    age_phrase = "unknown age"
    if last_seen is not None:
        try:
            age_phrase = _format_age_phrase(max(0.0, now_epoch - float(last_seen)))
        except (TypeError, ValueError):
            pass
    return (
        f"Bus synaptic snapshot (not live traffic): {row.get('source_organ')} -> "
        f"{row.get('target_organ')} hop latency was unusual at last observation "
        f"(|z|={abs(zscore):.1f} vs EWMA baseline, {count} samples, last seen {age_phrase})."
    )


async def fetch_bus_synaptic_anomaly_fragments(
    *,
    max_items: int = 5,
    zscore_threshold: float = 3.0,
    min_count: int = 5,
    max_edge_age_sec: float | None = None,
) -> List[Dict[str, Any]]:
    """Real edges from the live bus synaptic graph whose latest observation is
    a genuine statistical outlier against their own history -- not a static
    threshold, not simulated.

    Only edges with ``last_seen_epoch`` within ``max_edge_age_sec`` are
    returned so frozen z-scores on long-dead channels cannot masquerade as
    current bus activity in unified-turn recall.

    Same fragment shape as every other recall source (id/source/source_ref/
    uri/text/ts/tags/score/meta) so fusion.py treats it identically, no
    fusion.py changes needed. Never raises: any Falkor failure degrades to
    [], same fail-open contract as every other adapter in this arc.
    """
    client = get_bus_synaptic_falkor_client()
    if client is None:
        return []

    age_sec = float(max_edge_age_sec if max_edge_age_sec is not None else _max_edge_age_sec())
    now_epoch = time.time()
    min_last_seen_epoch = now_epoch - age_sec
    per_query_limit = max(1, max_items)
    query_params = {
        "threshold": zscore_threshold,
        "min_count": min_count,
        "min_last_seen_epoch": min_last_seen_epoch,
        "limit": per_query_limit,
    }
    try:
        publish_rows, causal_rows = await asyncio.gather(
            asyncio.to_thread(
                client.graph_query,
                _PUBLISH_ANOMALY_QUERY,
                query_params,
            ),
            asyncio.to_thread(
                client.graph_query,
                _CAUSAL_ANOMALY_QUERY,
                query_params,
            ),
        )
    except Exception as exc:
        logger.debug("bus_synaptic_anomaly_fetch_skipped error=%s", exc)
        return []

    out: List[Dict[str, Any]] = []
    for row in publish_rows or []:
        if row.get("zscore") is None:
            continue
        last_seen_epoch = row.get("last_seen_epoch")
        frag_id = f"bus_synaptic_publish:{row.get('organ_id')}:{row.get('channel')}"
        out.append(
            {
                "id": frag_id,
                "source": "bus_synaptic_anomaly",
                "source_ref": "falkordb",
                "uri": frag_id,
                "text": _format_publish_anomaly_text(row, now_epoch=now_epoch),
                "ts": float(last_seen_epoch) if last_seen_epoch is not None else None,
                "tags": ["bus_synaptic", "anomaly", "publish_gap", "stale_telemetry_snapshot"],
                "score": 0.5,
                "meta": {
                    "organ_id": row.get("organ_id"),
                    "channel": row.get("channel"),
                    "zscore": row.get("zscore"),
                    "count": row.get("count"),
                    "last_seen_epoch": last_seen_epoch,
                    "last_seen_at": _iso_from_epoch(last_seen_epoch),
                    "max_edge_age_sec": age_sec,
                    "signal_kind": "publish_gap_zscore",
                },
            }
        )
    for row in causal_rows or []:
        if row.get("zscore") is None:
            continue
        last_seen_epoch = row.get("last_seen_epoch")
        frag_id = f"bus_synaptic_causal:{row.get('source_organ')}:{row.get('target_organ')}"
        out.append(
            {
                "id": frag_id,
                "source": "bus_synaptic_anomaly",
                "source_ref": "falkordb",
                "uri": frag_id,
                "text": _format_causal_anomaly_text(row, now_epoch=now_epoch),
                "ts": float(last_seen_epoch) if last_seen_epoch is not None else None,
                "tags": ["bus_synaptic", "anomaly", "causal_latency", "stale_telemetry_snapshot"],
                "score": 0.5,
                "meta": {
                    "source_organ": row.get("source_organ"),
                    "target_organ": row.get("target_organ"),
                    "zscore": row.get("zscore"),
                    "count": row.get("count"),
                    "last_seen_epoch": last_seen_epoch,
                    "last_seen_at": _iso_from_epoch(last_seen_epoch),
                    "max_edge_age_sec": age_sec,
                    "signal_kind": "causal_latency_zscore",
                },
            }
        )
    return out[:max_items]


__all__ = ["fetch_bus_synaptic_anomaly_fragments", "get_bus_synaptic_falkor_client"]
