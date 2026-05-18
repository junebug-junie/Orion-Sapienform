"""Local validation for memory-graph suggest drafts (escalation vs operator warnings)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

ONTOLOGY_VERSION = "orionmem-2026-05"

ENTITY_KINDS = frozenset(
    {"person", "animal", "pet", "breed", "taxon", "collective", "abstract"}
)

ALLOWED_PREDICATES = frozenset(
    {
        "orionmem:inSituation",
        "orionmem:stimulusEntity",
        "orionmem:aboutEntity",
        "orionmem:targetOfNegativeAffect",
        "orionmem:generalizationOf",
        "orionmem:contradictsSituation",
        "prov:wasDerivedFrom",
        "schema:about",
    }
)

_REQUIRED_TOP_KEYS = frozenset(
    {
        "ontology_version",
        "utterance_ids",
        "entities",
        "situations",
        "edges",
        "dispositions",
    }
)

_NAMED_SUBJECT_RE = re.compile(
    r"\b(?:[A-Z][a-z]{2,}|[A-Z]{2,})\b(?!\s*(?:said|says|asked|told))",
)
_EVENT_HINT_RE = re.compile(
    r"\b(?:today|yesterday|last\s+week|recently|when|after|before|because|"
    r"angered|annoyed|pissed|scared|trusted|feared|happened|did|was|were)\b",
    re.I,
)


def utterance_likely_has_named_subjects(text: str) -> bool:
    body = (text or "").strip()
    if len(body) < 8:
        return False
    return bool(_NAMED_SUBJECT_RE.search(body))


def utterance_likely_describes_event(text: str) -> bool:
    body = (text or "").strip()
    if len(body) < 12:
        return False
    return bool(_EVENT_HINT_RE.search(body))


def validate_for_escalation(
    data: Dict[str, Any],
    *,
    utterance_text: str = "",
) -> Tuple[bool, List[str]]:
    """Return (should_escalate, validation_errors). True escalates to Brain."""
    errors: List[str] = []

    if not isinstance(data, dict):
        return True, ["invalid_json:not_an_object"]

    missing = sorted(_REQUIRED_TOP_KEYS - set(data.keys()))
    if missing:
        errors.append(f"missing_keys:{','.join(missing)}")

    if str(data.get("ontology_version") or "") != ONTOLOGY_VERSION:
        errors.append("ontology_version_mismatch")

    utterance_ids = data.get("utterance_ids")
    if not isinstance(utterance_ids, list) or len(utterance_ids) == 0:
        errors.append("no_utterance_ids")

    entities = data.get("entities")
    if not isinstance(entities, list):
        errors.append("entities_not_array")
        entities = []

    situations = data.get("situations")
    if not isinstance(situations, list):
        errors.append("situations_not_array")
        situations = []

    edges = data.get("edges")
    if not isinstance(edges, list):
        errors.append("edges_not_array")
        edges = []

    dispositions = data.get("dispositions")
    if dispositions is not None and not isinstance(dispositions, list):
        errors.append("dispositions_not_array")

    for ent in entities:
        if not isinstance(ent, dict):
            errors.append("entity_not_object")
            continue
        kind = str(ent.get("entityKind") or "").strip()
        if not kind:
            errors.append("entity_missing_entityKind")
        elif kind not in ENTITY_KINDS:
            errors.append(f"unknown_entityKind:{kind}")
        for key in ("id", "label", "entityKind", "surfaceForms"):
            if key not in ent:
                errors.append(f"entity_missing_{key}")
        sf = ent.get("surfaceForms")
        if sf is not None and not isinstance(sf, list):
            errors.append("entity_surfaceForms_not_array")

    for sit in situations:
        if not isinstance(sit, dict):
            errors.append("situation_not_object")
            continue
        uids = sit.get("utterance_ids")
        if not isinstance(uids, list) or len(uids) == 0:
            errors.append("situation_missing_utterance_ids")

    for edge in edges:
        if not isinstance(edge, dict):
            errors.append("edge_not_object")
            continue
        pred = str(edge.get("p") or "").strip()
        if not pred:
            errors.append("edge_missing_predicate")
        elif pred not in ALLOWED_PREDICATES:
            errors.append(f"unknown_predicate:{pred}")
        conf = edge.get("confidence")
        if conf is not None:
            try:
                c = float(conf)
                if c < 0.0 or c > 1.0:
                    errors.append("invalid_confidence")
            except (TypeError, ValueError):
                errors.append("invalid_confidence")

    if utterance_likely_has_named_subjects(utterance_text) and len(entities) == 0:
        errors.append("no_entities_when_subjects_expected")

    if utterance_likely_describes_event(utterance_text) and len(situations) == 0:
        errors.append("no_situations_when_event_expected")

    return (len(errors) > 0, errors)


def parse_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    from orion.memory_graph.json_extract import extract_first_json_object_text

    import json

    raw = (text or "").strip()
    if not raw:
        return None, "empty_output"
    blob = extract_first_json_object_text(raw)
    if not blob:
        return None, "no_json_object"
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(data, dict):
        return None, "invalid_json"
    return data, None
