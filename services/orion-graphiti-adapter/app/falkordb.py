from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def sync_to_falkordb(
    *,
    uri: str,
    graph_name: str,
    crystallization_id: str,
    kind: str,
    subject: str,
    summary: str,
) -> dict[str, Any]:
    """Best-effort FalkorDB projection via Redis GRAPH commands."""
    if not uri.strip():
        return {"synced": False, "reason": "falkordb_uri_unset"}

    try:
        import redis

        client = redis.from_url(uri)
        entity_id = f"gent_{crystallization_id}"
        episode_id = f"gep_{crystallization_id}"
        cypher = (
            f"MERGE (e:Entity {{id: '{entity_id}', crystallization_id: '{crystallization_id}', name: '{subject.replace(chr(39), '')}'}}) "
            f"MERGE (ep:Episode {{id: '{episode_id}', kind: '{kind}', summary: '{summary[:200].replace(chr(39), '')}'}}) "
            f"MERGE (e)-[:HAS_EPISODE]->(ep)"
        )
        result = client.execute_command("GRAPH.QUERY", graph_name, cypher)
        return {"synced": True, "graph": graph_name, "result_type": type(result).__name__}
    except Exception as exc:
        logger.warning("falkordb_sync_failed id=%s error=%s", crystallization_id, exc)
        return {"synced": False, "reason": str(exc)}
