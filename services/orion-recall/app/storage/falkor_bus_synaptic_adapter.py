"""Live bus-synaptic-graph anomaly awareness for chat reasoning.

Idea 4 of docs/superpowers/specs/2026-07-24-bus-vitality-field-signal-brainstorm.md's
Phase 3+ arc. Design doc: docs/superpowers/specs/2026-07-24-bus-synaptic-graph-
reasoning-consumer-design.md (proposal-mode pass, per this repo's CLAUDE.md
section 0A -- first idea in that arc touching a reasoning pipeline, not just
infrastructure telemetry).

Reuses the exact anomaly-detection Cypher already live-verified in
services/orion-hub/scripts/bus_synaptic_graph_routes.py::anomalies() --
edges whose most recent observation deviated sharply from that edge's own
rolling EWMA baseline, guarded by a min_count floor against the documented
cold-start z-score instability (see services/orion-bus-mirror/README.md).
Fixed, parameterized queries only -- never free-form Cypher, matching the
hard constraint the design doc names as non-negotiable.

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
from typing import Any, Dict, List, Optional

from orion.graph.falkor_client import RedisGraphQueryClient

logger = logging.getLogger(__name__)

_CLIENT: Optional[RedisGraphQueryClient] = None

_PUBLISH_ANOMALY_QUERY = """
MATCH (o:Organ)-[e:PUBLISHES]->(c:Channel)
WHERE e.gap_zscore IS NOT NULL AND abs(e.gap_zscore) > $threshold AND e.count > $min_count
RETURN o.organ_id AS organ_id, c.channel AS channel,
       e.gap_zscore AS zscore, e.count AS count
ORDER BY abs(e.gap_zscore) DESC
LIMIT $limit
"""

_CAUSAL_ANOMALY_QUERY = """
MATCH (a:Organ)-[e:CAUSALLY_FOLLOWED_BY]->(b:Organ)
WHERE e.latency_zscore IS NOT NULL AND abs(e.latency_zscore) > $threshold AND e.count > $min_count
RETURN a.organ_id AS source_organ, b.organ_id AS target_organ,
       e.latency_zscore AS zscore, e.count AS count
ORDER BY abs(e.latency_zscore) DESC
LIMIT $limit
"""


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


def _format_publish_anomaly_text(row: Dict[str, Any]) -> str:
    return (
        f"Bus channel \"{row.get('channel')}\" from {row.get('organ_id')} is publishing at an "
        f"unusual rate (z-score {row.get('zscore'):.1f} against its own rolling baseline, "
        f"{row.get('count')} samples)."
    )


def _format_causal_anomaly_text(row: Dict[str, Any]) -> str:
    return (
        f"{row.get('source_organ')} -> {row.get('target_organ')} hop latency is unusual "
        f"(z-score {row.get('zscore'):.1f} against its own rolling baseline, "
        f"{row.get('count')} samples)."
    )


async def fetch_bus_synaptic_anomaly_fragments(
    *,
    max_items: int = 5,
    zscore_threshold: float = 3.0,
    min_count: int = 5,
) -> List[Dict[str, Any]]:
    """Real edges from the live bus synaptic graph whose latest observation is
    a genuine statistical outlier against their own history -- not a static
    threshold, not simulated.

    Same fragment shape as every other recall source (id/source/source_ref/
    uri/text/ts/tags/score/meta) so fusion.py treats it identically, no
    fusion.py changes needed. Never raises: any Falkor failure degrades to
    [], same fail-open contract as every other adapter in this arc.
    """
    client = get_bus_synaptic_falkor_client()
    if client is None:
        return []

    per_query_limit = max(1, max_items)
    try:
        publish_rows, causal_rows = await asyncio.gather(
            asyncio.to_thread(
                client.graph_query,
                _PUBLISH_ANOMALY_QUERY,
                {"threshold": zscore_threshold, "min_count": min_count, "limit": per_query_limit},
            ),
            asyncio.to_thread(
                client.graph_query,
                _CAUSAL_ANOMALY_QUERY,
                {"threshold": zscore_threshold, "min_count": min_count, "limit": per_query_limit},
            ),
        )
    except Exception as exc:
        logger.debug("bus_synaptic_anomaly_fetch_skipped error=%s", exc)
        return []

    out: List[Dict[str, Any]] = []
    for row in publish_rows or []:
        if row.get("zscore") is None:
            continue
        frag_id = f"bus_synaptic_publish:{row.get('organ_id')}:{row.get('channel')}"
        out.append(
            {
                "id": frag_id,
                "source": "bus_synaptic_anomaly",
                "source_ref": "falkordb",
                # Per-fragment-unique, matching every sibling adapter's convention
                # (falkor_neighborhood_adapter.py/falkor_chat_adapter.py use
                # turn_id) -- NOT a shared constant. Caught in review:
                # fusion.py::_key_for() dedupes on uri (which wins over id when
                # both are set), so a shared constant here silently collapsed
                # every anomaly from a single fetch down to just one surviving
                # fragment -- live-reproduced against real data (5 real
                # anomalies in, 1 survived fuse_candidates()).
                "uri": frag_id,
                "text": _format_publish_anomaly_text(row),
                "ts": None,
                "tags": ["bus_synaptic", "anomaly", "publish_gap"],
                "score": 0.5,
                "meta": {
                    "organ_id": row.get("organ_id"),
                    "channel": row.get("channel"),
                    "zscore": row.get("zscore"),
                    "count": row.get("count"),
                },
            }
        )
    for row in causal_rows or []:
        if row.get("zscore") is None:
            continue
        frag_id = f"bus_synaptic_causal:{row.get('source_organ')}:{row.get('target_organ')}"
        out.append(
            {
                "id": frag_id,
                "source": "bus_synaptic_anomaly",
                "source_ref": "falkordb",
                "uri": frag_id,  # see publish-anomaly branch above for why this must be per-fragment-unique
                "text": _format_causal_anomaly_text(row),
                "ts": None,
                "tags": ["bus_synaptic", "anomaly", "causal_latency"],
                "score": 0.5,
                "meta": {
                    "source_organ": row.get("source_organ"),
                    "target_organ": row.get("target_organ"),
                    "zscore": row.get("zscore"),
                    "count": row.get("count"),
                },
            }
        )
    return out[:max_items]


__all__ = ["fetch_bus_synaptic_anomaly_fragments", "get_bus_synaptic_falkor_client"]
