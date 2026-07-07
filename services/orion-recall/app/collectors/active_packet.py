from __future__ import annotations

from typing import Any

from orion.memory.crystallization.active_packet import KIND_TO_BUCKET
from orion.memory.crystallization.projection_graphiti import GraphitiAdapter
from orion.memory.crystallization.repository import list_crystallizations
from orion.memory.crystallization.retriever import retrieve_active_packet
from orion.schemas.recall_pcr import RetrievalIntentV1

_BUCKET_BY_INTENT: dict[str, frozenset[str]] = {
    "relational": frozenset({"stance"}),
    "semantic": frozenset({"project_state", "stance", "attractors", "warnings"}),
    "procedural": frozenset({"procedures"}),
    "open_loop": frozenset({"open_loops"}),
    "contradiction": frozenset({"contradictions"}),
}

_PACKET_BUCKETS: tuple[tuple[str, str], ...] = (
    ("stance", "stance"),
    ("project_state", "project_state"),
    ("procedures", "procedures"),
    ("open_loops", "open_loops"),
    ("contradictions", "contradictions"),
    ("warnings", "warnings"),
    ("attractors", "attractors"),
)


def _enabled(settings: Any) -> bool:
    return bool(getattr(settings, "RECALL_ACTIVE_PACKET_ENABLED", True))


def _intent(query: Any) -> str:
    return str(getattr(query, "retrieval_intent", None) or "semantic")


def _allowed_buckets(intent: str) -> frozenset[str] | None:
    return _BUCKET_BY_INTENT.get(intent)


def _entry_to_fragment(entry: dict[str, Any], *, bucket: str) -> dict[str, Any]:
    kind = str(entry.get("kind") or "semantic")
    summary = str(entry.get("summary") or "").strip()
    crystallization_id = str(entry.get("crystallization_id") or "")
    try:
        salience = float(entry.get("salience") or 0.5)
    except Exception:
        salience = 0.5
    return {
        "id": f"ap:{crystallization_id}",
        "source": "active_packet",
        "snippet": summary,
        "text": summary,
        "score": max(0.0, min(1.0, salience)),
        "tags": [f"kind:{kind}", f"bucket:{bucket}"],
    }


def _packet_items(packet: Any) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    if hasattr(packet, "items"):
        for item in getattr(packet, "items") or []:
            if isinstance(item, dict):
                kind = str(item.get("kind") or "semantic")
                bucket = str(item.get("bucket") or KIND_TO_BUCKET.get(kind, "project_state"))
                if bucket == "semantic":
                    bucket = KIND_TO_BUCKET.get(kind, "project_state")
                rows.append((bucket, item))
        return rows
    for attr, bucket in _PACKET_BUCKETS:
        for entry in getattr(packet, attr, None) or []:
            if isinstance(entry, dict):
                rows.append((bucket, entry))
    return rows


def _graphiti_adapter(settings: Any, *, intent: str) -> GraphitiAdapter | None:
    graphiti_enabled = bool(getattr(settings, "RECALL_GRAPHITI_IN_CHAT", False))
    if not graphiti_enabled or intent != "contradiction":
        return None
    url = str(getattr(settings, "RECALL_GRAPHITI_ADAPTER_URL", "") or "").strip()
    if not url:
        return None
    return GraphitiAdapter(
        enabled=True,
        url=url,
        falkordb_uri=str(getattr(settings, "RECALL_GRAPHITI_FALKORDB_URI", "") or "").strip() or None,
        timeout_sec=float(getattr(settings, "RECALL_GRAPHITI_TIMEOUT_SEC", 10.0) or 10.0),
    )


def _pick_seed_crystallization(active_items: list[Any], fragment: str, seed_id: str) -> str:
    if seed_id:
        return seed_id
    fragment_lower = str(fragment or "").strip().lower()
    best_id = ""
    best_score = -1.0
    for item in active_items:
        cid = str(getattr(item, "crystallization_id", "") or "")
        if not cid:
            continue
        summary = str(getattr(item, "summary", "") or getattr(item, "subject", "") or "").lower()
        try:
            salience = float(getattr(item, "salience", None) or 0.5)
        except Exception:
            salience = 0.5
        score = salience
        if fragment_lower and fragment_lower in summary:
            score += 0.5
        if score > best_score:
            best_score = score
            best_id = cid
    return best_id


async def fetch_active_packet_fragments(
    query: Any,
    *,
    pool,
    settings,
) -> list[dict[str, Any]]:
    if not _enabled(settings):
        return []
    if pool is None:
        return []

    fragment = str(getattr(query, "fragment", None) or "").strip()
    if not fragment:
        return []

    intent = _intent(query)
    allowed = _allowed_buckets(intent)
    task_hints = getattr(query, "task_hints", None) or {}
    task_type = task_hints.get("task_mode") if isinstance(task_hints, dict) else None

    try:
        active_items = await list_crystallizations(pool, status="active", limit=100)
    except Exception:
        return []

    seed_id = str(getattr(query, "seed_crystallization_id", None) or "").strip()
    if not seed_id and active_items:
        seed_id = _pick_seed_crystallization(active_items, fragment, seed_id)

    graphiti = _graphiti_adapter(settings, intent=intent)
    packet = await retrieve_active_packet(
        query=fragment,
        crystallizations=active_items,
        task_type=str(task_type) if task_type else None,
        session_id=getattr(query, "session_id", None),
        chroma_host=str(getattr(settings, "VECTOR_DB_HOST", "") or ""),
        chroma_port=int(getattr(settings, "VECTOR_DB_PORT", 8000) or 8000),
        chroma_collection=str(
            getattr(settings, "RECALL_CRYSTALLIZATION_VECTOR_COLLECTION", "orion_memory_crystallizations")
            or "orion_memory_crystallizations"
        ),
        embed_host_url=str(getattr(settings, "RECALL_CARDS_EMBEDDING_URL", "") or ""),
        graphiti_adapter=graphiti,
        seed_crystallization_id=seed_id or None,
    )

    fragments: list[dict[str, Any]] = []
    for bucket, entry in _packet_items(packet):
        if allowed is not None and bucket not in allowed:
            continue
        frag = _entry_to_fragment(entry, bucket=bucket)
        if frag["snippet"]:
            fragments.append(frag)
    return fragments


__all__ = ["fetch_active_packet_fragments"]
