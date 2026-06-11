"""Normalize and repair SuggestDraftV1-shaped dicts before RDF / approval."""

from __future__ import annotations

import re
import uuid
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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


def is_malformed_urn_uuid(ref: Any) -> bool:
    """True when ref looks like urn:uuid:… but fails RFC-4122 hex tail validation."""
    if is_blank_ref(ref):
        return False
    s = str(ref).strip()
    if not s.startswith("urn:uuid:"):
        return False
    return not bool(_UUID_TAIL.match(s.removeprefix("urn:uuid:")))


_REPAIR_URN_UUID_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c0")


def _stable_repaired_urn_uuid(ref: str) -> str:
    return f"urn:uuid:{uuid.uuid5(_REPAIR_URN_UUID_NAMESPACE, str(ref).strip())}"


def _collect_refs(value: Any, out: Set[str]) -> None:
    if is_blank_ref(value):
        return
    if isinstance(value, str):
        out.add(str(value).strip())
        return
    if isinstance(value, list):
        for item in value:
            _collect_refs(item, out)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_refs(item, out)


def _build_malformed_urn_uuid_remap(data: Dict[str, Any]) -> Dict[str, str]:
    """Map invalid urn:uuid refs (e.g. LLM sequential …90ag suffixes) to stable valid ids."""
    refs: Set[str] = set()
    for ent in data.get("entities") or []:
        if isinstance(ent, dict):
            _collect_refs(ent.get("id"), refs)
            _collect_refs(ent.get("generalizes_to"), refs)
    for sit in data.get("situations") or []:
        if isinstance(sit, dict):
            _collect_refs(sit.get("id"), refs)
            _collect_refs(sit.get("stimulus_entity_id"), refs)
            _collect_refs(sit.get("about_entity_ids"), refs)
            _collect_refs(sit.get("target_entity_ids"), refs)
            _collect_refs(sit.get("participants"), refs)
    for disp in data.get("dispositions") or []:
        if isinstance(disp, dict):
            _collect_refs(disp.get("id"), refs)
            _collect_refs(disp.get("holder_id"), refs)
            _collect_refs(disp.get("target_id"), refs)
    for edge in data.get("edges") or []:
        if isinstance(edge, dict):
            _collect_refs(edge.get("s"), refs)
            _collect_refs(edge.get("o"), refs)

    remap: Dict[str, str] = {}
    for ref in sorted(refs):
        if is_malformed_urn_uuid(ref):
            remap[ref] = _stable_repaired_urn_uuid(ref)
    return remap


def _apply_urn_uuid_remap(value: Any, remap: Dict[str, str], apply_fn: Callable[[Any], Any]) -> Any:
    if is_blank_ref(value):
        return value
    if isinstance(value, str):
        s = str(value).strip()
        return remap.get(s, s)
    if isinstance(value, list):
        return [apply_fn(item) for item in value]
    if isinstance(value, dict):
        out = dict(value)
        for key, item in out.items():
            if key in ("entity_id", "entityId"):
                out[key] = apply_fn(item)
            elif key in (
                "id",
                "generalizes_to",
                "stimulus_entity_id",
                "holder_id",
                "target_id",
                "s",
                "o",
            ):
                out[key] = apply_fn(item)
            elif key in ("about_entity_ids", "target_entity_ids"):
                out[key] = apply_fn(item)
            elif key == "participants":
                out[key] = apply_fn(item)
        return out
    return value


def repair_malformed_urn_uuid_refs(data: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite invalid urn:uuid ids/refs so RDF conversion and preview do not fail."""
    if not isinstance(data, dict):
        return data
    remap = _build_malformed_urn_uuid_remap(data)
    if not remap:
        return data

    def apply_ref(value: Any) -> Any:
        return _apply_urn_uuid_remap(value, remap, apply_ref)

    out = dict(data)
    out["entities"] = [
        apply_ref(ent) for ent in (out.get("entities") or []) if isinstance(ent, dict)
    ]
    out["situations"] = [
        apply_ref(sit) for sit in (out.get("situations") or []) if isinstance(sit, dict)
    ]
    out["dispositions"] = [
        apply_ref(disp) for disp in (out.get("dispositions") or []) if isinstance(disp, dict)
    ]
    out["edges"] = [
        apply_ref(edge) for edge in (out.get("edges") or []) if isinstance(edge, dict)
    ]
    return out


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


def _normalize_user_entity_label(label: Any) -> str:
    s = str(label or "").strip()
    if s.lower() in ("user", "the user", "operator"):
        return "Juniper"
    return s


def _local_today_iso() -> str:
    return date.today().isoformat()


def _occurred_at_needs_today(value: Any) -> bool:
    s = str(value or "").strip()
    if not s or s.lower() in ("null", "none"):
        return True
    m = re.match(r"^(\d{4})-\d{2}-\d{2}$", s)
    if not m:
        return True
    try:
        year = int(m.group(1))
    except ValueError:
        return True
    return year < date.today().year - 1


def normalize_role_grounded_draft_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Operator-facing fixes: user speaker label and stale occurredAt dates."""
    if not isinstance(data, dict):
        return data
    out = dict(data)
    entities = []
    for ent in out.get("entities") or []:
        if not isinstance(ent, dict):
            continue
        row = dict(ent)
        row["label"] = _normalize_user_entity_label(row.get("label"))
        entities.append(row)
    out["entities"] = entities
    today = _local_today_iso()
    situations = []
    for sit in out.get("situations") or []:
        if not isinstance(sit, dict):
            continue
        row = dict(sit)
        if _occurred_at_needs_today(row.get("occurredAt")):
            row["occurredAt"] = today
            tq = str(row.get("timeQualitative") or "").strip().lower()
            if not tq or tq == "unknown":
                row["timeQualitative"] = "today"
        situations.append(row)
    out["situations"] = situations
    return out


def sanitize_suggest_draft_dict(
    data: Dict[str, Any],
    *,
    rebuild_edges_from_structure: bool = False,
) -> Dict[str, Any]:
    """Strip null refs, drop invalid edges, optionally rebuild edges from situations."""
    if not isinstance(data, dict):
        return data

    out = normalize_role_grounded_draft_dict(dict(data))
    out = repair_malformed_urn_uuid_refs(out)
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
