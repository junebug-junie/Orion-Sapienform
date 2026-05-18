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
_ROLE_TURN_RE = re.compile(r"\brole=(user|assistant)\b", re.I)
_TURN_BODY_RE = re.compile(
    r"---\s*turn\s+\d+\s+id=\S+\s+role=(?:user|assistant)\s*---\s*\n\s*(\S.+)",
    re.I | re.M,
)
_RELATION_CUE_RE = re.compile(
    r"\b(?:"
    r"off\s+to|shower|be\s+back|back\s+soon|i['\u2019]?ll\s+be\s+here|i\s+will\s+be\s+here|"
    r"going\s+to|leaving|returning|waiting|available|thanks|sorry|love|want|need|"
    r"like|dislike|feel|family|work|bike|bikes|kids|ride|riding"
    r")\b",
    re.I,
)
_USER_ENTITY_HINTS = frozenset({"user", "juniper", "entity:user"})
_ASSISTANT_ENTITY_HINTS = frozenset({"orion", "assistant", "entity:orion"})


def extract_selected_role_evidence(utterance_text: str) -> Dict[str, bool]:
    """Parse bridge-style transcript evidence for selected user/assistant turns."""
    text = (utterance_text or "").strip()
    roles = [m.group(1).lower() for m in _ROLE_TURN_RE.finditer(text)]
    has_user_turn = "user" in roles
    has_assistant_turn = "assistant" in roles
    has_nonempty_text = bool(_TURN_BODY_RE.search(text))
    has_extractable_relation = utterance_likely_contains_extractable_relation(text)
    return {
        "has_user_turn": has_user_turn,
        "has_assistant_turn": has_assistant_turn,
        "has_nonempty_text": has_nonempty_text,
        "has_extractable_relation": has_extractable_relation,
    }


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


def utterance_likely_has_role_grounded_subjects(text: str) -> bool:
    ev = extract_selected_role_evidence(text)
    return bool(ev.get("has_user_turn") or ev.get("has_assistant_turn"))


def utterance_likely_contains_extractable_relation(text: str) -> bool:
    body = (text or "").strip()
    if len(body) < 6:
        return False
    return bool(_RELATION_CUE_RE.search(body))


def role_grounded_extraction_expected(utterance_text: str) -> bool:
    """True when selected turns imply minimal semantic projection (not salience)."""
    ev = extract_selected_role_evidence(utterance_text)
    if not utterance_likely_has_role_grounded_subjects(utterance_text):
        return False
    if not ev.get("has_nonempty_text"):
        return False
    return bool(ev.get("has_extractable_relation"))


def _entity_tokens(entities: List[Any]) -> Tuple[set[str], set[str]]:
    ids: set[str] = set()
    labels: set[str] = set()
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        eid = str(ent.get("id") or "").strip().lower()
        if eid:
            ids.add(eid)
        label = str(ent.get("label") or "").strip().lower()
        if label:
            labels.add(label)
        sf = ent.get("surfaceForms")
        if isinstance(sf, list):
            for item in sf:
                s = str(item or "").strip().lower()
                if s:
                    labels.add(s)
    return ids, labels


def _has_role_entity(entities: List[Any], hints: frozenset[str]) -> bool:
    ids, labels = _entity_tokens(entities)
    combined = ids | labels
    if combined & hints:
        return True
    for token in combined:
        for hint in hints:
            if hint in token:
                return True
    return False


def _has_user_role_entity(entities: List[Any]) -> bool:
    return _has_role_entity(entities, _USER_ENTITY_HINTS)


def _has_assistant_role_entity(entities: List[Any]) -> bool:
    return _has_role_entity(entities, _ASSISTANT_ENTITY_HINTS)


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

    role_expected = role_grounded_extraction_expected(utterance_text)
    role_ev = extract_selected_role_evidence(utterance_text)

    if role_expected:
        if len(entities) == 0:
            errors.append("no_entities_when_role_grounded_subjects_expected")
        if len(situations) == 0:
            errors.append("no_situations_when_role_grounded_context_expected")
        if role_ev.get("has_user_turn") and role_ev.get("has_assistant_turn"):
            if not _has_user_role_entity(entities):
                errors.append("missing_user_role_entity")
            if not _has_assistant_role_entity(entities):
                errors.append("missing_assistant_role_entity")
    elif utterance_likely_has_named_subjects(utterance_text) and len(entities) == 0:
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
