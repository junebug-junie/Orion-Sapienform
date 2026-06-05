"""Normalize and repair SuggestDraftV1-shaped dicts before RDF / approval."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

_UUID_TAIL = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.I,
)


def is_blank_ref(ref: Any) -> bool:
    if ref is None:
        return True
    s = str(ref).strip()
    return not s or s.lower() in ("null", "none")


def is_resolvable_entity_ref(ref: Any, *, entity_base: str = "https://orion.example/ns/mem/entity/") -> bool:
    if is_blank_ref(ref):
        return False
    s = str(ref).strip()
    if s.startswith("http://") or s.startswith("https://"):
        return True
    if s.startswith("urn:uuid:"):
        return bool(_UUID_TAIL.match(s.removeprefix("urn:uuid:")))
    return False


def _known_node_ids(data: Dict[str, Any]) -> Tuple[Set[str], Set[str], Set[str]]:
    entity_ids: Set[str] = set()
    situation_ids: Set[str] = set()
    utterance_ids: Set[str] = set()
    for ent in data.get("entities") or []:
        if isinstance(ent, dict):
            eid = str(ent.get("id") or "").strip()
            if eid:
                entity_ids.add(eid)
    for sit in data.get("situations") or []:
        if isinstance(sit, dict):
            sid = str(sit.get("id") or "").strip()
            if sid:
                situation_ids.add(sid)
    for uid in data.get("utterance_ids") or []:
        u = str(uid or "").strip()
        if u:
            utterance_ids.add(u)
    return entity_ids, situation_ids, utterance_ids


def _structural_edges(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Edges implied by situation fields (preview + RDF spine when model edges are wrong)."""
    entity_ids, situation_ids, _ = _known_node_ids(data)
    graph_nodes = entity_ids | situation_ids
    out: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()

    def add(s: str, p: str, o: str, confidence: float = 0.85) -> None:
        if not is_resolvable_entity_ref(s) or not is_resolvable_entity_ref(o):
            return
        if s not in graph_nodes or o not in graph_nodes:
            return
        key = (s, p, o)
        if key in seen:
            return
        seen.add(key)
        out.append({"s": s, "p": p, "o": o, "confidence": confidence})

    for sit in data.get("situations") or []:
        if not isinstance(sit, dict):
            continue
        sid = str(sit.get("id") or "").strip()
        if not sid or sid not in situation_ids:
            continue
        stim = sit.get("stimulus_entity_id")
        if not is_blank_ref(stim):
            add(sid, "orionmem:stimulusEntity", str(stim).strip())
        for ae in sit.get("about_entity_ids") or []:
            if not is_blank_ref(ae):
                add(sid, "schema:about", str(ae).strip())
        for ta in sit.get("target_entity_ids") or []:
            if not is_blank_ref(ta):
                add(sid, "orionmem:targetOfNegativeAffect", str(ta).strip())
        for part in sit.get("participants") or []:
            if not isinstance(part, dict):
                continue
            eid = str(part.get("entity_id") or part.get("entityId") or "").strip()
            if eid and eid in entity_ids:
                add(eid, "orionmem:inSituation", sid)

    return out


def sanitize_suggest_draft_dict(
    data: Dict[str, Any],
    *,
    rebuild_edges_from_structure: bool = False,
) -> Dict[str, Any]:
    """Strip null refs, drop invalid edges, optionally rebuild edges from situations."""
    if not isinstance(data, dict):
        return data

    out = dict(data)
    entity_ids, situation_ids, utterance_ids = _known_node_ids(out)
    graph_nodes = entity_ids | situation_ids

    entities = []
    for ent in out.get("entities") or []:
        if not isinstance(ent, dict):
            continue
        row = dict(ent)
        if is_blank_ref(row.get("generalizes_to")):
            row["generalizes_to"] = None
        elif not is_resolvable_entity_ref(row.get("generalizes_to")):
            row["generalizes_to"] = None
        entities.append(row)
    out["entities"] = entities

    situations = []
    for sit in out.get("situations") or []:
        if not isinstance(sit, dict):
            continue
        row = dict(sit)
        for key in ("stimulus_entity_id",):
            if is_blank_ref(row.get(key)) or (
                not is_blank_ref(row.get(key))
                and str(row.get(key)).strip() not in entity_ids
                and str(row.get(key)).strip() in utterance_ids
            ):
                row[key] = None
        row["about_entity_ids"] = [
            str(x).strip()
            for x in (row.get("about_entity_ids") or [])
            if not is_blank_ref(x)
            and str(x).strip() in entity_ids
        ]
        row["target_entity_ids"] = [
            str(x).strip()
            for x in (row.get("target_entity_ids") or [])
            if not is_blank_ref(x) and str(x).strip() in entity_ids
        ]
        parts = []
        for part in row.get("participants") or []:
            if not isinstance(part, dict):
                continue
            eid = str(part.get("entity_id") or part.get("entityId") or "").strip()
            if eid and eid in entity_ids:
                parts.append(
                    {
                        "entity_id": eid,
                        "role": str(part.get("role") or "participant").strip() or "participant",
                    }
                )
        row["participants"] = parts
        situations.append(row)
    out["situations"] = situations

    dispositions = []
    for disp in out.get("dispositions") or []:
        if not isinstance(disp, dict):
            continue
        row = dict(disp)
        if is_blank_ref(row.get("id")):
            row["id"] = None
        if is_blank_ref(row.get("holder_id")):
            continue
        if is_blank_ref(row.get("target_id")):
            continue
        dispositions.append(row)
    out["dispositions"] = dispositions

    cleaned_edges: List[Dict[str, Any]] = []
    for edge in out.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        s = str(edge.get("s") or "").strip()
        o = str(edge.get("o") or "").strip()
        p = str(edge.get("p") or "").strip()
        if is_blank_ref(s) or is_blank_ref(o) or not p:
            continue
        if s in utterance_ids or o in utterance_ids:
            continue
        if s not in graph_nodes or o not in graph_nodes:
            continue
        if not is_resolvable_entity_ref(s) or not is_resolvable_entity_ref(o):
            continue
        conf = edge.get("confidence")
        try:
            c = float(conf) if conf is not None else 0.8
        except (TypeError, ValueError):
            c = 0.8
        cleaned_edges.append({"s": s, "p": p, "o": o, "confidence": c})

    if rebuild_edges_from_structure or not cleaned_edges:
        structural = _structural_edges(out)
        merged_keys = {(e["s"], e["p"], e["o"]) for e in cleaned_edges}
        for e in structural:
            key = (e["s"], e["p"], e["o"])
            if key not in merged_keys:
                cleaned_edges.append(e)
                merged_keys.add(key)

    out["edges"] = cleaned_edges
    return out
