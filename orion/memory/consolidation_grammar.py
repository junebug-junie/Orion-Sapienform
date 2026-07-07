from __future__ import annotations

import json
from typing import Any


def _parse_event_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _atom_semantic_role(event_json: dict[str, Any]) -> str:
    atom = event_json.get("atom")
    if not isinstance(atom, dict):
        return ""
    return str(atom.get("semantic_role") or "")


async def fetch_grammar_evidence_for_window(
    pool,
    *,
    turns: list[dict[str, Any]],
    node_id: str,
    enabled: bool,
) -> tuple[bool, list[str]]:
    if not enabled or pool is None:
        return False, []
    event_ids: list[str] = []
    repair = False
    for turn in turns:
        corr = str(turn.get("correlation_id") or "").strip()
        if not corr:
            continue
        trace_id = f"hub.chat:{node_id}:{corr}"
        rows = await pool.fetch(
            """
            SELECT event_id, event_json
            FROM grammar_events
            WHERE trace_id = $1
            ORDER BY created_at ASC
            """,
            trace_id,
        )
        for row in rows:
            eid = str(row["event_id"])
            event_ids.append(eid)
            event_json = _parse_event_json(row["event_json"])
            if _atom_semantic_role(event_json) == "repair_signal":
                repair = True
    return repair, event_ids
