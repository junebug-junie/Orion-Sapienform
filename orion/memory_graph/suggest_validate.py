"""Local validation for memory-graph suggest drafts (escalation vs operator warnings)."""

from __future__ import annotations

import re
import uuid
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
_ASSISTANT_ENTITY_HINTS = frozenset(
    {"orion", "assistant", "entity:orion", "sapienform", "orion-sapienform"}
)

_ROLE_ENTITY_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c0")

_ROLE_ENTITY_LABELS = frozenset({"user", "orion", "juniper", "assistant", "here"})

_TOPICAL_CUE_RE = re.compile(
    r"\b(?:take|stance|approach|rollout|training|design|usable|decorative|pragmatic|pov|plan)\b",
    re.I,
)
_QUOTED_PHRASE_RE = re.compile(r'["\']([^"\']{3,80})["\']')
_TOPICAL_BIGRAM_RE = re.compile(
    r"\b(pragmatic\s+take|point\s+of\s+view|model\s+training|cert\s+rollout)\b",
    re.I,
)


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


def _draft_utterance_corpus(data: Dict[str, Any], utterance_text: str = "") -> str:
    parts: List[str] = []
    if utterance_text and str(utterance_text).strip():
        parts.append(str(utterance_text))
    text_map = data.get("utterance_text_by_id")
    if isinstance(text_map, dict):
        for val in text_map.values():
            s = str(val or "").strip()
            if s:
                parts.append(s)
    return " ".join(parts)


def _entity_surface_forms_lower(entities: List[Any]) -> set[str]:
    out: set[str] = set()
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        label = str(ent.get("label") or "").strip().lower()
        if label:
            out.add(label)
        sf = ent.get("surfaceForms")
        if isinstance(sf, list):
            for item in sf:
                s = str(item or "").strip().lower()
                if s:
                    out.add(s)
    return out


def _has_topical_entity(entities: List[Any]) -> bool:
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        kind = str(ent.get("entityKind") or "").strip().lower()
        label = str(ent.get("label") or "").strip().lower()
        if kind == "abstract" and label not in _ROLE_ENTITY_LABELS:
            return True
        if kind not in ("person", "abstract"):
            return True
        if kind == "person" and label not in _ROLE_ENTITY_LABELS:
            return True
    return False


def _salient_phrases_in_corpus(corpus: str) -> List[str]:
    phrases: List[str] = []
    for match in _QUOTED_PHRASE_RE.finditer(corpus):
        phrases.append(match.group(1).strip())
    for match in _TOPICAL_BIGRAM_RE.finditer(corpus):
        phrases.append(match.group(0).strip())
    seen: set[str] = set()
    out: List[str] = []
    for phrase in phrases:
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(phrase)
    return out


def _phrase_covered_by_surfaces(phrase: str, surfaces: set[str]) -> bool:
    norm = phrase.strip().lower()
    if not norm:
        return True
    for surface in surfaces:
        if norm in surface or surface in norm:
            return True
    return False


def collect_topical_spine_warnings(
    data: Dict[str, Any],
    *,
    utterance_text: str = "",
) -> List[str]:
    """Non-blocking hints when relational spine exists but topical subjects are thin."""
    if not isinstance(data, dict):
        return []
    corpus = _draft_utterance_corpus(data, utterance_text)
    if not corpus.strip() or not _TOPICAL_CUE_RE.search(corpus):
        return []

    entities = data.get("entities")
    if not isinstance(entities, list):
        entities = []

    warnings: List[str] = []
    surfaces = _entity_surface_forms_lower(entities)

    if role_grounded_extraction_expected(utterance_text or corpus) and entities and not _has_topical_entity(
        entities
    ):
        warnings.append("topical_spine_missing:consider_abstract_topic_entities")

    for phrase in _salient_phrases_in_corpus(corpus):
        if not _phrase_covered_by_surfaces(phrase, surfaces):
            warnings.append(f"topical_phrase_not_extracted:{phrase[:72]}")

    return warnings[:10]


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


def _turn_ids_by_role(utterance_text: str) -> Dict[str, List[str]]:
    """Map user/assistant turn ids from bridge-style transcript evidence."""
    out: Dict[str, List[str]] = {"user": [], "assistant": []}
    for match in re.finditer(
        r"---\s*turn\s+\d+\s+id=(\S+)\s+role=(user|assistant)\s*---",
        utterance_text or "",
        re.I,
    ):
        out[str(match.group(2)).lower()].append(str(match.group(1)).strip())
    return out


def _stable_role_entity_id(role: str, utterance_ids: List[str]) -> str:
    key = f"{role}:{','.join(sorted(str(u).strip() for u in utterance_ids if str(u).strip()))}"
    return f"urn:uuid:{uuid.uuid5(_ROLE_ENTITY_NAMESPACE, key)}"


def _has_user_role_entity(
    entities: List[Any],
    *,
    situations: Optional[List[Any]] = None,
    utterance_text: str = "",
) -> bool:
    if _has_role_entity(entities, _USER_ENTITY_HINTS):
        return True
    user_turns = set(_turn_ids_by_role(utterance_text).get("user") or [])
    if not user_turns or not situations:
        return False
    ent_ids = {
        str(ent.get("id") or "").strip()
        for ent in entities
        if isinstance(ent, dict) and str(ent.get("id") or "").strip()
    }
    for sit in situations:
        if not isinstance(sit, dict):
            continue
        uids = {str(uid).strip() for uid in (sit.get("utterance_ids") or [])}
        if not uids & user_turns:
            continue
        stim = str(sit.get("stimulus_entity_id") or "").strip()
        if stim and stim in ent_ids:
            return True
        for part in sit.get("participants") or []:
            if isinstance(part, dict):
                eid = str(part.get("entity_id") or "").strip()
                if eid and eid in ent_ids:
                    return True
    return False


def _has_assistant_role_entity(
    entities: List[Any],
    *,
    situations: Optional[List[Any]] = None,
    utterance_text: str = "",
) -> bool:
    if _has_role_entity(entities, _ASSISTANT_ENTITY_HINTS):
        return True
    asst_turns = set(_turn_ids_by_role(utterance_text).get("assistant") or [])
    if not asst_turns or not situations:
        return False
    ent_ids = {
        str(ent.get("id") or "").strip()
        for ent in entities
        if isinstance(ent, dict) and str(ent.get("id") or "").strip()
    }
    for sit in situations:
        if not isinstance(sit, dict):
            continue
        uids = {str(uid).strip() for uid in (sit.get("utterance_ids") or [])}
        if not uids & asst_turns:
            continue
        stim = str(sit.get("stimulus_entity_id") or "").strip()
        if stim and stim in ent_ids:
            return True
        for part in sit.get("participants") or []:
            if isinstance(part, dict):
                eid = str(part.get("entity_id") or "").strip()
                if eid and eid in ent_ids:
                    return True
    return False


def repair_role_grounded_suggest_draft(
    data: Dict[str, Any],
    *,
    utterance_text: str = "",
) -> Dict[str, Any]:
    """Inject canonical User/Orion entities when both roles are present but the model omitted one."""
    if not isinstance(data, dict):
        return data
    role_ev = extract_selected_role_evidence(utterance_text)
    if not (
        role_grounded_extraction_expected(utterance_text)
        and role_ev.get("has_user_turn")
        and role_ev.get("has_assistant_turn")
    ):
        return data

    out = dict(data)
    entities = list(out.get("entities") or [])
    if not isinstance(entities, list):
        entities = []
    situations = list(out.get("situations") or [])
    if not isinstance(situations, list):
        situations = []

    utterance_ids = [
        str(uid).strip() for uid in (out.get("utterance_ids") or []) if str(uid).strip()
    ]
    if not utterance_ids:
        utterance_ids = [
            *(_turn_ids_by_role(utterance_text).get("user") or []),
            *(_turn_ids_by_role(utterance_text).get("assistant") or []),
        ]

    if not _has_user_role_entity(entities, situations=situations, utterance_text=utterance_text):
        entities.append(
            {
                "id": _stable_role_entity_id("user", utterance_ids),
                "label": "User",
                "entityKind": "person",
                "surfaceForms": ["I"],
                "generalizes_to": None,
            }
        )
    if not _has_assistant_role_entity(entities, situations=situations, utterance_text=utterance_text):
        entities.append(
            {
                "id": _stable_role_entity_id("assistant", utterance_ids),
                "label": "Orion",
                "entityKind": "abstract",
                "surfaceForms": ["I"],
                "generalizes_to": None,
            }
        )

    out["entities"] = entities
    return out


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
            if not _has_user_role_entity(entities, situations=situations, utterance_text=utterance_text):
                errors.append("missing_user_role_entity")
            if not _has_assistant_role_entity(
                entities, situations=situations, utterance_text=utterance_text
            ):
                errors.append("missing_assistant_role_entity")
    elif utterance_likely_has_named_subjects(utterance_text) and len(entities) == 0:
        errors.append("no_entities_when_subjects_expected")

    if utterance_likely_describes_event(utterance_text) and len(situations) == 0:
        errors.append("no_situations_when_event_expected")

    return (len(errors) > 0, errors)


def parse_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    from orion.memory_graph.json_extract import (
        extract_first_json_object_text,
        salvage_truncated_json_object_text,
    )

    import json

    raw = (text or "").strip()
    if not raw:
        return None, "empty_output"
    blob = extract_first_json_object_text(raw)
    if not blob:
        blob = salvage_truncated_json_object_text(raw)
    if not blob:
        return None, "no_json_object"
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(data, dict):
        return None, "invalid_json"
    return data, None
