from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import requests

from orion.core.contracts.recall import MemoryBundleStatsV1, MemoryBundleV1, MemoryItemV1, RecallQueryV1

from .render import render_items
from .settings import settings
from .sql_timeline import fetch_exact_fragments, fetch_recent_fragments
from .storage.rdf_adapter import fetch_rdf_chatturn_exact_matches, fetch_rdf_fragments
from .storage.vector_adapter import fetch_vector_exact_matches, fetch_vector_fragments


@dataclass(frozen=True)
class RecallV2Plan:
    query_text: str
    entity_anchors: tuple[str, ...]
    project_anchors: tuple[str, ...]
    temporal_anchor: str | None
    time_window_days: int
    exact_anchor_tokens: tuple[str, ...]


def _extract_entities(query: str) -> List[str]:
    return [item.strip() for item in re.findall(r"[A-Z][A-Za-z0-9_]+(?:\s+[A-Z][A-Za-z0-9_]+)?", query or "") if item.strip()]


def _extract_project_anchors(query: str) -> List[str]:
    tokens = re.findall(r"(?:orion-[a-z0-9\-]+|services/[a-z0-9\-_/.]+|orion/[a-z0-9\-_/.]+)", query or "", flags=re.I)
    deduped: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token not in seen:
            deduped.append(token)
            seen.add(token)
    return deduped[:8]


def _extract_exact_anchor_tokens(query: str) -> List[str]:
    tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\d+\b|\b[A-Fa-f0-9]{7,40}\b", query or "")
    deduped: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token not in seen:
            deduped.append(token)
            seen.add(token)
    return deduped[:8]


def _temporal_hint(query: str) -> tuple[str | None, int]:
    q = (query or "").lower()
    if "today" in q or "latest" in q:
        return "today", 1
    if "yesterday" in q:
        return "yesterday", 2
    if "this week" in q:
        return "this_week", 7
    if "last week" in q:
        return "last_week", 14
    if "this month" in q:
        return "this_month", 31
    return None, settings.RECALL_DEFAULT_TIME_WINDOW_DAYS


def build_recall_v2_plan(query: RecallQueryV1) -> RecallV2Plan:
    temporal_anchor, window_days = _temporal_hint(query.fragment)
    return RecallV2Plan(
        query_text=str(query.fragment or ""),
        entity_anchors=tuple(_extract_entities(query.fragment)),
        project_anchors=tuple(_extract_project_anchors(query.fragment)),
        temporal_anchor=temporal_anchor,
        time_window_days=window_days,
        exact_anchor_tokens=tuple(_extract_exact_anchor_tokens(query.fragment)),
    )


def _contains_any(text: str, values: tuple[str, ...]) -> bool:
    if not values:
        return True
    lowered = (text or "").lower()
    return any(item.lower() in lowered for item in values)


def _pageindex_candidates(plan: RecallV2Plan, *, top_k: int = 8) -> List[Dict[str, Any]]:
    base = str(getattr(settings, "RECALL_V2_PAGEINDEX_URL", "http://orion-athena-pageindex:8384")).rstrip("/")
    try:
        resp = requests.post(
            f"{base}/corpora/journals/query",
            json={"query": plan.query_text, "allow_fallback": True, "top_k": top_k},
            timeout=3.0,
        )
        payload = resp.json() if resp.status_code < 500 else {}
    except Exception:
        payload = {}
    rows = payload.get("results") if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return []
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows[:top_k]):
        excerpt = str(row.get("excerpt") or "")
        out.append(
            {
                "id": str(row.get("node_id") or f"pageindex:{idx}"),
                "source": "pageindex_lexical",
                "source_ref": "pageindex:journals",
                "text": excerpt,
                "ts": None,
                "score": max(0.1, 1.0 - (idx * 0.08)),
                "meta": {
                    "entry_id": row.get("entry_id"),
                    "created_at": row.get("created_at"),
                    "heading": row.get("heading"),
                    "provenance": row.get("provenance") or {},
                },
                "tags": ["pageindex", "lexical", "section_index"],
            }
        )
    return out


async def run_recall_v2_shadow(query: RecallQueryV1) -> tuple[MemoryBundleV1, Dict[str, Any]]:
    started = time.time()
    plan = build_recall_v2_plan(query)
    candidates: List[Dict[str, Any]] = []
    backend_counts: Dict[str, int] = {}
    filter_debug: Dict[str, Any] = {
        "entity_anchors": list(plan.entity_anchors),
        "project_anchors": list(plan.project_anchors),
        "temporal_anchor": plan.temporal_anchor,
        "time_window_days": plan.time_window_days,
        "exact_anchor_tokens": list(plan.exact_anchor_tokens),
    }

    if plan.exact_anchor_tokens:
        sql_exact = await fetch_exact_fragments(
            tokens=list(plan.exact_anchor_tokens),
            session_id=query.session_id,
            node_id=query.node_id,
            limit=10,
        )
        backend_counts["sql_exact_anchor"] = len(sql_exact)
        for row in sql_exact:
            candidates.append(
                {
                    "id": row.id,
                    "source": "sql_timeline",
                    "source_ref": row.source_ref,
                    "text": row.text,
                    "ts": row.ts,
                    "score": 1.0,
                    "tags": list(row.tags or []) + ["exact_anchor"],
                    "explain": {"exact_anchor": True, "backend": "sql_timeline"},
                }
            )
        vector_exact = fetch_vector_exact_matches(
            tokens=list(plan.exact_anchor_tokens),
            max_items=8,
            session_id=query.session_id,
            profile_name="recall.v2.shadow",
            node_id=query.node_id,
        )
        backend_counts["vector_exact_anchor"] = len(vector_exact)
        for row in vector_exact:
            row = dict(row)
            row["source"] = "vector"
            row["score"] = max(0.7, float(row.get("score") or 0.0))
            row["tags"] = list(row.get("tags") or []) + ["exact_anchor"]
            row["explain"] = {"exact_anchor": True, "backend": "vector"}
            candidates.append(row)
        rdf_exact = fetch_rdf_chatturn_exact_matches(tokens=list(plan.exact_anchor_tokens), session_id=query.session_id, max_items=8)
        backend_counts["rdf_exact_anchor"] = len(rdf_exact)
        for row in rdf_exact:
            row = dict(row)
            row["source"] = "rdf_chat"
            row["score"] = max(0.65, float(row.get("score") or 0.0))
            row["tags"] = list(row.get("tags") or []) + ["exact_anchor"]
            row["explain"] = {"exact_anchor": True, "backend": "rdf_chat"}
            candidates.append(row)

    pageindex = _pageindex_candidates(plan, top_k=8)
    backend_counts["pageindex_lexical"] = len(pageindex)
    candidates.extend(pageindex)

    vector = fetch_vector_fragments(
        query_text=plan.query_text,
        time_window_days=plan.time_window_days,
        max_items=10,
        session_id=query.session_id,
        profile_name="recall.v2.shadow",
        node_id=query.node_id,
    )
    backend_counts["vector"] = len(vector)
    for row in vector:
        item = dict(row)
        item["source"] = "vector"
        item["explain"] = {"backend": "vector", "semantic_signal": float(item.get("score") or 0.0)}
        candidates.append(item)

    rdf = fetch_rdf_fragments(query_text=plan.query_text, max_items=8)
    backend_counts["rdf"] = len(rdf)
    for row in rdf:
        item = dict(row)
        item["source"] = "rdf"
        item["explain"] = {"backend": "rdf", "graph_expansion": True}
        candidates.append(item)

    # Deterministic timeline fallback constrained by temporal window.
    since_minutes = max(60, int(plan.time_window_days * 24 * 60))
    recent = await fetch_recent_fragments(query.session_id, query.node_id, since_minutes, 10)
    backend_counts["sql_timeline_recent"] = len(recent)
    for row in recent:
        candidates.append(
            {
                "id": row.id,
                "source": "sql_timeline",
                "source_ref": row.source_ref,
                "text": row.text,
                "ts": row.ts,
                "score": 0.5,
                "tags": list(row.tags or []) + ["timeline_recent"],
                "explain": {"backend": "sql_timeline", "temporal_filter_days": plan.time_window_days},
            }
        )

    ranked_cards: List[Dict[str, Any]] = []
    for row in candidates:
        text = str(row.get("text") or row.get("snippet") or "")
        entity_match = _contains_any(text, plan.entity_anchors)
        project_match = _contains_any(text, plan.project_anchors)
        exact_anchor = "exact_anchor" in list(row.get("tags") or []) or bool((row.get("explain") or {}).get("exact_anchor"))
        base = float(row.get("score") or 0.0)
        score = base
        if exact_anchor:
            score += 0.6
        if entity_match:
            score += 0.25
        if project_match:
            score += 0.25
        ranked_cards.append(
            {
                "id": str(row.get("id") or ""),
                "source": str(row.get("source") or "unknown"),
                "source_ref": row.get("source_ref"),
                "score": score,
                "snippet": text[:800],
                "ts": row.get("ts"),
                "tags": list(row.get("tags") or []),
                "why_selected": {
                    "exact_anchor": exact_anchor,
                    "entity_match": entity_match,
                    "project_match": project_match,
                    "source_score": base,
                    "source_explain": row.get("explain") or {},
                },
            }
        )

    ranked_cards.sort(key=lambda item: (-(float(item.get("score") or 0.0)), str(item.get("source") or ""), str(item.get("id") or "")))
    top_cards = ranked_cards[:12]
    items = [
        MemoryItemV1(
            id=card["id"],
            source=card["source"],
            source_ref=card.get("source_ref"),
            score=max(0.0, min(1.0, float(card.get("score") or 0.0))),
            ts=card.get("ts"),
            snippet=str(card.get("snippet") or ""),
            tags=list(card.get("tags") or []),
        )
        for card in top_cards
    ]
    latency_ms = int((time.time() - started) * 1000)
    bundle = MemoryBundleV1(
        rendered=render_items(items, 320, profile_name="recall.v2.shadow"),
        items=items,
        stats=MemoryBundleStatsV1(
            backend_counts=backend_counts,
            latency_ms=latency_ms,
            profile="recall.v2.shadow",
            diagnostic={
                "anchors": filter_debug,
                "ranked_cards": top_cards,
            },
        ),
    )
    debug = {
        "plan": {
            "query_text": plan.query_text,
            "entity_anchors": list(plan.entity_anchors),
            "project_anchors": list(plan.project_anchors),
            "temporal_anchor": plan.temporal_anchor,
            "time_window_days": plan.time_window_days,
            "exact_anchor_tokens": list(plan.exact_anchor_tokens),
        },
        "filters": filter_debug,
        "backend_counts": backend_counts,
        "ranked_cards": top_cards,
        "latency_ms": latency_ms,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return bundle, debug
