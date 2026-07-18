"""Cypher-native chat-turn recall fragments, standing in for
storage/rdf_adapter.py::fetch_rdf_chatturn_fragments when
settings.RECALL_FALKOR_IN_CHAT is true.

Phase 4 of the recall RDF->Falkor cutover
(docs/superpowers/specs/2026-07-17-recall-rdf-writer-falkor-cutover-phase2-spec.md).
The Falkor ``:ChatTurn`` node (written by orion-meta-tags' Phase 2 writer,
services/orion-meta-tags/app/falkor_recall_writer.py) is deliberately thin --
no prompt/response text, Postgres owns that -- so this is not a pure
single-store Cypher read: turn discovery/ordering comes from Falkor, the
actual quoted text comes from a Postgres join by turn_id
(app/sql_chat.py::fetch_chat_turns_by_id).

Deliberate behavior divergence from fetch_rdf_chatturn_fragments, named not
silent: the RDF version's own docstring says "NO keyword filtering at SPARQL
layer (sustainable)... ranking happens later... in fusion", but its actual
SPARQL does inject a keyword CONTAINS filter when the query yields keywords
-- code and docstring disagree there (a pre-existing drift in the RDF path,
not something this module reproduces). This function follows the stated
intent instead: pull the most recent turns, no keyword filtering, let
fusion.py's own keyword-overlap scoring rank them. ``session_id`` is
accepted (same signature as the RDF version, so it drops in cleanly at the
worker.py call site) but not used for filtering -- matching the RDF
version's actual behavior (also unused there despite its docstring) and this
service's own documented system-wide behavior (README.md: "recall ignores
[session_id] for retrieval and ranking").
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from ..recall_falkor_store import get_recall_falkor_client
from ..sql_chat import _to_epoch, fetch_chat_turns_by_id

logger = logging.getLogger(__name__)


async def fetch_falkor_chatturn_fragments(
    *,
    query_text: str,
    session_id: str | None,
    max_items: int = 20,
) -> List[Dict[str, Any]]:
    """Pull recent ChatTurns from FalkorDB's orion_recall graph, joined to
    Postgres for prompt/response text. Same return-fragment shape as
    fetch_rdf_chatturn_fragments (id/source/source_ref/uri/text/ts/tags/score/meta)
    so fusion.py treats it identically -- there is no per-backend adapter
    layer to paper over shape drift.

    Never raises: any Falkor or Postgres failure degrades to [], same
    fail-open contract as the RDF version.
    """
    if not query_text:
        return []
    client = get_recall_falkor_client()
    if client is None:
        return []

    try:
        rows = await asyncio.to_thread(
            client.graph_query,
            # source_kind filter is load-bearing, not cosmetic: :ChatTurn is
            # shared with social.turn.stored.v1 (falkor_recall_writer.py's
            # own docstring warns about turn_id collisions between the two
            # kinds sharing this label). Only chat.history turns exist in
            # Postgres chat_history_log -- social turns land in a separate
            # table (SocialRoomTurnSQL) -- so without this filter, social
            # turns would consume slots in the ORDER BY ts DESC LIMIT
            # window, get silently dropped after a failed Postgres join
            # below, and crowd out real chat.history turns during a burst
            # of social-room activity.
            "MATCH (t:ChatTurn {source_kind: 'chat.history'}) "
            "RETURN t.turn_id AS turn_id, t.ts AS ts, t.correlation_id AS correlation_id "
            "ORDER BY t.ts DESC "
            "LIMIT $max_items",
            {"max_items": int(max(1, min(max_items, 100)))},
        )
    except Exception as exc:
        logger.debug("falkor chatturn fetch skipped: %s", exc)
        return []
    if not rows:
        return []

    turn_ids = [str(r.get("turn_id") or "").strip() for r in rows]
    turn_ids = [t for t in turn_ids if t]
    if not turn_ids:
        return []

    try:
        text_map = await fetch_chat_turns_by_id(turn_ids)
    except Exception as exc:
        logger.debug("falkor chatturn postgres text join skipped: %s", exc)
        return []

    out: List[Dict[str, Any]] = []
    for row in rows:
        turn_id = str(row.get("turn_id") or "").strip()
        if not turn_id or turn_id not in text_map:
            # No matching Postgres row for this turn_id -- nothing to quote,
            # so this fragment carries no value fusion.py could rank on.
            continue
        prompt, response = text_map[turn_id]
        text = f'ExactUserText: "{prompt}"\nOrionResponse: "{response}"'.strip()

        out.append(
            {
                "id": turn_id,
                "source": "falkor_chat",
                "source_ref": "falkordb",
                "uri": turn_id,
                "text": text[:1800],
                "ts": _to_epoch(row.get("ts")),
                "tags": ["falkor", "chat", "chatturn"],
                "score": 0.50,
                "meta": {},
            }
        )
    return out


__all__ = ["fetch_falkor_chatturn_fragments"]
