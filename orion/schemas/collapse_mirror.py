# orion/schemas/collapse_mirror.py
from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


V1_REQUIRED_KEYS = {
    "observer",
    "trigger",
    "observer_state",
    "field_resonance",
    "type",
    "emergent_entity",
    "summary",
    "mantra",
    "causal_echo",
    "timestamp",
    "environment",
}

V2_MIN_REQUIRED_KEYS = {
    "observer",
    "trigger",
    "type",
    "emergent_entity",
    "summary",
    "mantra",
}

# Entry type → change_type (different ontology; do NOT default change_type=type)
DEFAULT_CHANGE_TYPE_BY_ENTRY_TYPE: dict[str, str] = {
    "idle": "deadband",
    "flow": "stabilizing",
    "turbulence": "escalating",
    "glitch": "anomaly_detected",
    "epiphany": "reorientation",
}


def _normalize_token(token: str) -> str:
    cleaned = " ".join(token.strip().split())
    cleaned = cleaned.replace("_ ", "_").replace(" _", "_")
    return cleaned


def _parse_string_set_like(value: str) -> List[str]:
    if not value:
        return []

    raw = value.strip()

    if raw.startswith("["):
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, list):
                return [_normalize_token(str(item)) for item in loaded if str(item).strip()]
        except Exception:
            pass

    if raw.startswith("{") and raw.endswith("}"):
        raw = raw[1:-1]

    if "," in raw or "\n" in raw:
        parts = re.split(r"[,\n]+", raw)
    else:
        parts = [raw]

    normalized: List[str] = []
    for part in parts:
        cleaned = part.strip().strip('"').strip("'")
        if cleaned:
            normalized.append(_normalize_token(cleaned))

    return normalized


def _normalize_string_list(value: Any) -> List[str]:
    if value is None:
        return []

    items: List[str] = []

    if isinstance(value, str):
        items = _parse_string_set_like(value)
    elif isinstance(value, Sequence):
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                items.extend(_parse_string_set_like(item))
            else:
                items.append(_normalize_token(str(item)))
    else:
        items = [_normalize_token(str(value))]

    deduped: List[str] = []
    seen: set[str] = set()
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)

    return deduped


def _normalize_telemetry_key(key: Any) -> str:
    explicit = {
        "gpu_ mem": "gpu_mem",
        "gpu_ util": "gpu_util",
        "phi_ hint": "phi_hint",
    }
    raw = str(key or "").strip()
    if raw in explicit:
        raw = explicit[raw]
    raw = raw.replace(" ", "_")
    raw = re.sub(r"_+", "_", raw)
    if raw in explicit:
        raw = explicit[raw]
    return raw


def _normalize_telemetry_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    normalized: Dict[str, Any] = {}
    for key, value in data.items():
        normalized[_normalize_telemetry_key(key)] = value
    return normalized


def _canonical_phi_hint(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None

    if value.get("schema_version") == "v1":
        bands = value.get("bands") if isinstance(value.get("bands"), dict) else None
        numeric = value.get("numeric") if isinstance(value.get("numeric"), dict) else None
        out: Dict[str, Any] = {"schema_version": "v1"}
        if bands:
            out["bands"] = bands
        if numeric:
            out["numeric"] = numeric
        return out

    band_keys = {"valence_band", "valence_dir", "energy_band", "coherence_band", "novelty_band"}
    numeric_keys = {
        "valence",
        "energy",
        "coherence",
        "novelty",
        "arousal",
        "dominance",
        "clarity",
        "overload",
        "risk_score",
    }

    bands = {k: value[k] for k in band_keys if k in value}
    numeric = {k: value[k] for k in numeric_keys if k in value}
    if not bands and not numeric:
        return None
    out = {"schema_version": "v1"}
    if bands:
        out["bands"] = bands
    if numeric:
        out["numeric"] = numeric
    return out


def _normalize_entry_telemetry(entry: "CollapseMirrorEntryV2") -> None:
    telemetry = entry.state_snapshot.telemetry
    if not isinstance(telemetry, dict):
        entry.state_snapshot.telemetry = {}
        return
    normalized = _normalize_telemetry_keys(telemetry)
    if "phi_hint" in normalized:
        canonical = _canonical_phi_hint(normalized.get("phi_hint"))
        if canonical:
            normalized["phi_hint"] = canonical
    entry.state_snapshot.telemetry = normalized


def _is_v2_shape(data: Dict[str, Any]) -> bool:
    return V2_MIN_REQUIRED_KEYS.issubset(data.keys())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip_diacritics(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def _observer_is_orion(observer: Any) -> bool:
    if observer is None:
        return False
    try:
        s = str(observer).strip().lower()
    except Exception:
        return False
    s = _strip_diacritics(s)
    return s == "orion"


def _observer_is_juniper(observer: Any) -> bool:
    if observer is None:
        return False
    try:
        s = str(observer).strip().lower()
    except Exception:
        return False
    s = _strip_diacritics(s)
    return s == "juniper"


def _origin_service(payload_or_entry: Any) -> Optional[str]:
    if isinstance(payload_or_entry, dict):
        origin = payload_or_entry.get("origin")
        source_service = payload_or_entry.get("source_service")
    else:
        origin = getattr(payload_or_entry, "origin", None)
        source_service = getattr(payload_or_entry, "source_service", None)

    if source_service:
        return str(source_service)
    if isinstance(origin, dict):
        for key in ("source_service", "service", "name"):
            val = origin.get(key)
            if val:
                return str(val)
        return None
    if origin:
        return str(origin)
    return None


def mirror_kind(payload_or_entry: Any) -> str:
    observer = None
    if isinstance(payload_or_entry, dict):
        observer = payload_or_entry.get("observer")
    else:
        observer = getattr(payload_or_entry, "observer", None)

    if _observer_is_juniper(observer):
        return "strict"

    origin_service = _origin_service(payload_or_entry)
    if origin_service:
        normalized = _strip_diacritics(str(origin_service).strip().lower())
        if normalized == "collapse_mirror_service":
            return "strict"

    if _observer_is_orion(observer):
        return "metacog"

    if origin_service:
        normalized = _strip_diacritics(str(origin_service).strip().lower())
        if normalized == "metacog":
            return "metacog"

    return "unknown"


def should_route_to_triage(payload_or_entry: Any) -> bool:
    return mirror_kind(payload_or_entry) == "strict"


def _coerce_change_type_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    change_type_payload = data.get("change_type")
    if not isinstance(change_type_payload, dict):
        return data

    reserved_keys = {"change_type", "label", "value", "name", "type", "change_type_scores", "telemetry"}
    chosen_key: Optional[str] = None
    ct_str: Optional[str] = None

    for key in ("change_type", "label", "value", "name", "type"):
        val = change_type_payload.get(key)
        if isinstance(val, str) and val.strip():
            chosen_key = key
            ct_str = val.strip()
            break

    if ct_str is None:
        for key, val in change_type_payload.items():
            if isinstance(val, str) and val.strip():
                chosen_key = key
                ct_str = val.strip()
                break

    def _is_numeric(val: Any) -> bool:
        return isinstance(val, (int, float)) and not isinstance(val, bool)

    numeric_scores: Dict[str, float] = {}
    meta_payload: Dict[str, Any] = {}

    for key, val in change_type_payload.items():
        if key == chosen_key or key in reserved_keys:
            continue
        if _is_numeric(val):
            numeric_scores[str(key)] = float(val)
        else:
            meta_payload[str(key)] = val

    existing_scores = data.get("change_type_scores")
    if not isinstance(existing_scores, dict):
        existing_scores = {}
    data["change_type_scores"] = {**numeric_scores, **existing_scores}

    if ct_str:
        data["change_type"] = ct_str
    elif numeric_scores:
        data["change_type"] = max(numeric_scores.items(), key=lambda item: item[1])[0]
    else:
        data["change_type"] = None

    if meta_payload:
        meta_json = json.dumps(meta_payload, ensure_ascii=False, default=str)
        state_snapshot = data.get("state_snapshot")
        if not isinstance(state_snapshot, dict):
            state_snapshot = {}
        telemetry = state_snapshot.get("telemetry")
        if not isinstance(telemetry, dict):
            telemetry = {}
        if len(meta_json) <= 2048:
            telemetry["change_type_meta"] = meta_payload
        else:
            telemetry["change_type_meta_keys"] = list(meta_payload.keys())
        state_snapshot["telemetry"] = telemetry
        data["state_snapshot"] = state_snapshot

    return data


class CollapseMirrorEntryV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    observer: str
    trigger: str
    observer_state: Union[str, List[str]]
    field_resonance: str
    type: str
    emergent_entity: str
    summary: str
    mantra: str
    causal_echo: Optional[str] = None
    id: Optional[str] = None
    timestamp: Optional[str] = Field(default=None, description="ISO timestamp")
    environment: Optional[str] = Field(default=None, description="Environment (dev, prod, etc.)")

    @field_validator("observer_state", mode="before")
    @classmethod
    def _normalize_observer_state(cls, v: Any) -> List[str]:
        return _normalize_string_list(v)


class CollapseMirrorConstraints(BaseModel):
    model_config = ConfigDict(extra="ignore")

    severity_score: Optional[float] = None
    notes: Optional[str] = None


class CollapseMirrorNumericSisters(BaseModel):
    model_config = ConfigDict(extra="ignore")

    valence: Optional[float] = None
    arousal: Optional[float] = None
    clarity: Optional[float] = None
    overload: Optional[float] = None
    risk_score: Optional[float] = None
    constraints: CollapseMirrorConstraints = Field(default_factory=CollapseMirrorConstraints)


class CollapseMirrorCausalDensity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str = "ambient"
    score: float = 0.0
    rationale: Optional[str] = None

    @field_validator("label", mode="before")
    @classmethod
    def _normalize_label(cls, v: Any) -> str:
        if v is None:
            return "ambient"
        return str(v)

    @field_validator("score", mode="before")
    @classmethod
    def _normalize_score(cls, v: Any) -> float:
        if v is None:
            return 0.0
        return float(v)


class CollapseMirrorWhatChanged(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summary: Optional[str] = None
    previous_state: Optional[str] = None
    new_state: Optional[str] = None
    evidence: List[str] = Field(default_factory=list)

    @field_validator("evidence", mode="before")
    @classmethod
    def _normalize_evidence(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(x) for x in v if str(x)]
        return [str(v)]


class CollapseMirrorStateSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore")

    observer_state: List[str] = Field(default_factory=list)
    field_resonance: Optional[str] = None
    telemetry: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None

    @field_validator("observer_state", mode="before")
    @classmethod
    def _normalize_observer_state(cls, v: Any) -> List[str]:
        return _normalize_string_list(v)

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, v: Any) -> List[str]:
        return _normalize_string_list(v)


class CollapseMirrorEntryV2(BaseModel):
    model_config = ConfigDict(extra="ignore")

    event_id: str = Field(default_factory=lambda: f"collapse_{uuid4().hex}")
    id: Optional[str] = None

    # Allow LLM to emit null; we fill system-side in _fill_defaults().
    timestamp: Optional[str] = Field(default_factory=_utc_now_iso)

    environment: Optional[str] = None

    observer: str
    trigger: str
    observer_state: List[str] = Field(default_factory=list)
    field_resonance: Optional[str] = None
    type: str
    emergent_entity: str
    summary: str
    mantra: str
    causal_echo: Optional[str] = None

    snapshot_kind: str = "baseline"
    what_changed_summary: Optional[str] = None
    what_changed: Optional[CollapseMirrorWhatChanged] = None
    state_snapshot: CollapseMirrorStateSnapshot = Field(default_factory=CollapseMirrorStateSnapshot)

    pattern_candidate: Optional[str] = None
    resonance_signature: Optional[str] = None
    change_type: Optional[str] = None
    change_type_scores: Dict[str, float] = Field(default_factory=dict)
    tag_scores: Dict[str, float] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)

    numeric_sisters: CollapseMirrorNumericSisters = Field(default_factory=CollapseMirrorNumericSisters)
    causal_density: CollapseMirrorCausalDensity = Field(default_factory=CollapseMirrorCausalDensity)
    is_causally_dense: bool = False

    epistemic_status: str = "observed"
    visibility: str = "internal"
    redaction_level: str = "low"

    source_service: Optional[str] = None
    source_node: Optional[str] = None

    @field_validator("observer_state", mode="before")
    @classmethod
    def _normalize_observer_state(cls, v: Any) -> List[str]:
        return _normalize_string_list(v)

    @field_validator("state_snapshot", mode="before")
    @classmethod
    def _normalize_state_snapshot(cls, v: Any) -> Any:
        if v is None:
            return {}
        if isinstance(v, CollapseMirrorStateSnapshot):
            return v
        if isinstance(v, dict):
            return v
        return {}

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, v: Any) -> List[str]:
        return _normalize_string_list(v)

    @field_validator("what_changed", mode="before")
    @classmethod
    def _normalize_what_changed(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, CollapseMirrorWhatChanged):
            return v
        if isinstance(v, dict):
            return v
        if isinstance(v, str) and v.strip():
            return {"summary": v}
        return None

    @field_validator("numeric_sisters", mode="before")
    @classmethod
    def _normalize_numeric_sisters(cls, v: Any) -> Any:
        if v is None:
            return {}
        if isinstance(v, CollapseMirrorNumericSisters):
            return v
        if isinstance(v, dict):
            return v
        return {}

    @field_validator("causal_density", mode="before")
    @classmethod
    def _normalize_causal_density(cls, v: Any) -> Any:
        if v is None:
            return {}
        if isinstance(v, CollapseMirrorCausalDensity):
            return v
        if isinstance(v, dict):
            return v
        return {}

    @field_validator("change_type_scores", mode="before")
    @classmethod
    def _normalize_change_type_scores(cls, v: Any) -> Dict[str, float]:
        if v is None or isinstance(v, list):
            return {}
        if isinstance(v, dict):
            return {str(k): float(val) for k, val in v.items() if val is not None}
        return {}

    @field_validator("tag_scores", mode="before")
    @classmethod
    def _normalize_tag_scores(cls, v: Any) -> Dict[str, float]:
        if v is None or isinstance(v, list):
            return {}
        if isinstance(v, dict):
            return {str(k): float(val) for k, val in v.items() if val is not None}
        return {}

    @model_validator(mode="before")
    @classmethod
    def _coerce_v1(cls, data: Any) -> Any:
        if isinstance(data, CollapseMirrorEntryV2):
            return data
        if isinstance(data, CollapseMirrorEntryV1):
            return v1_to_v2(data).model_dump(mode="json")
        # IMPORTANT: avoid infinite recursion.
        #
        # When constructing V2 from a dict, pydantic runs this validator *before* model
        # creation. Our v1_to_v2() helper constructs a CollapseMirrorEntryV2(...), which
        # also runs this validator. If we treat *any* dict that contains the V1 keys as
        # "V1", then a V2-shaped dict will re-enter this conversion path and recurse.
        if isinstance(data, dict) and V1_REQUIRED_KEYS.issubset(data.keys()):
            # If the dict already looks like V2, keep it as-is.
            v2_markers = {"event_id", "snapshot_kind", "state_snapshot", "numeric_sisters", "causal_density"}
            if _is_v2_shape(data) or any(k in data for k in v2_markers):
                return _coerce_change_type_payload(data)
            return v1_to_v2(CollapseMirrorEntryV1.model_validate(data)).model_dump(mode="json")
        if isinstance(data, dict):
            return _coerce_change_type_payload(data)
        return data

    @model_validator(mode="after")
    def _fill_defaults(self) -> "CollapseMirrorEntryV2":
        # timestamp: allow null/empty, always fill system-side
        if not self.timestamp:
            self.timestamp = _utc_now_iso()

        if not self.environment:
            self.environment = os.getenv("CHRONICLE_ENVIRONMENT", "dev")

        if not self.snapshot_kind:
            self.snapshot_kind = "baseline"

        if not self.what_changed_summary:
            self.what_changed_summary = self.summary

        # change_type: map entry type → change_type if missing
        if not self.change_type and self.type:
            self.change_type = DEFAULT_CHANGE_TYPE_BY_ENTRY_TYPE.get(self.type, "context_shift")

        # tags: allow idle to remain empty if model explicitly emitted []
        if not self.tags:
            if self.type and self.type != "idle":
                self.tags = [self.type]
            else:
                self.tags = []

        if not self.state_snapshot.observer_state:
            self.state_snapshot.observer_state = list(self.observer_state)

        if not self.state_snapshot.field_resonance:
            self.state_snapshot.field_resonance = self.field_resonance

        # don't force tags into state_snapshot for idle (tags empty)
        if not self.state_snapshot.tags and self.tags:
            self.state_snapshot.tags = list(self.tags)

        if not self.event_id:
            self.event_id = f"collapse_{uuid4().hex}"

        if self.event_id and not self.id:
            self.id = self.event_id
        elif self.id and not self.event_id:
            self.event_id = self.id

        # If both are present but different, keep event_id authoritative.
        if self.event_id and self.id != self.event_id:
            self.id = self.event_id

        return self

    def with_defaults(self) -> "CollapseMirrorEntryV2":
        return self._fill_defaults()

    @field_validator("pattern_candidate", mode="before")
    @classmethod
    def _normalize_pattern_candidate(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        # If we receive a rich object (dict/list), serialize to JSON so SQL VARCHAR won't crash.
        if isinstance(v, (dict, list)):
            return json.dumps(v)
        return str(v)

    @field_validator("resonance_signature", mode="before")
    @classmethod
    def _normalize_resonance_signature(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, (dict, list)):
            return json.dumps(v)
        return str(v)


def v1_to_v2(v1: CollapseMirrorEntryV1) -> CollapseMirrorEntryV2:
    observer_state = v1.observer_state
    if isinstance(observer_state, str):
        observer_state_list = [observer_state]
    else:
        observer_state_list = [str(x) for x in observer_state if str(x)]

    return CollapseMirrorEntryV2(
        event_id=v1.id or f"collapse_{uuid4().hex}",
        id=v1.id,  # Pass explicit ID if present
        observer=v1.observer,
        trigger=v1.trigger,
        observer_state=observer_state_list,
        field_resonance=v1.field_resonance,
        type=v1.type,
        emergent_entity=v1.emergent_entity,
        summary=v1.summary,
        mantra=v1.mantra,
        causal_echo=v1.causal_echo,
        timestamp=v1.timestamp or _utc_now_iso(),
        environment=v1.environment,
        snapshot_kind="baseline",
        what_changed_summary=v1.summary,
        what_changed=CollapseMirrorWhatChanged(summary=v1.summary),
        state_snapshot=CollapseMirrorStateSnapshot(
            observer_state=observer_state_list,
            field_resonance=v1.field_resonance,
        ),
        change_type=DEFAULT_CHANGE_TYPE_BY_ENTRY_TYPE.get(v1.type, "context_shift") if v1.type else "context_shift",
        tags=[v1.type] if v1.type and v1.type != "idle" else [],
        causal_density=CollapseMirrorCausalDensity(label="ambient", score=0.0),
        is_causally_dense=False,
        epistemic_status="observed",
        visibility="internal",
        redaction_level="low",
    )


def normalize_collapse_entry(payload: Any) -> CollapseMirrorEntryV2:
    if isinstance(payload, CollapseMirrorEntryV2):
        entry = payload.with_defaults()
    elif isinstance(payload, CollapseMirrorEntryV1):
        entry = v1_to_v2(payload).with_defaults()
    elif isinstance(payload, dict) and V1_REQUIRED_KEYS.issubset(payload.keys()) and not _is_v2_shape(payload):
        entry = v1_to_v2(CollapseMirrorEntryV1.model_validate(payload)).with_defaults()
    else:
        entry = CollapseMirrorEntryV2.model_validate(payload).with_defaults()
    _normalize_entry_telemetry(entry)
    return entry


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text or "{" not in text:
        return None

    starts = [i for i, ch in enumerate(text) if ch == "{"]

    for s in starts:
        depth = 0
        in_str = False
        esc = False

        for e in range(s, len(text)):
            ch = text[e]

            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[s : e + 1].strip()
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        pass
                    break

    return None


def _find_any_string(obj: Any):
    if obj is None:
        return
    if isinstance(obj, str):
        yield obj
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _find_any_string(v)
        return
    if isinstance(obj, list):
        for v in obj:
            yield from _find_any_string(v)
        return


def _is_collapse_shape(obj: Dict[str, Any]) -> bool:
    return _is_v2_shape(obj) or V1_REQUIRED_KEYS.issubset(obj.keys())


def find_collapse_entry(prior_step_results: Any) -> Optional[Dict[str, Any]]:
    """
    Best-effort extraction of a CollapseMirror entry from nested step results.

    Preference: if multiple candidates exist, prefer observer=="orion" (diacritics-insensitive).
    """
    if prior_step_results is None:
        return None

    best: Optional[Dict[str, Any]] = None

    def consider(obj: Dict[str, Any]) -> None:
        nonlocal best
        if not _is_collapse_shape(obj):
            return
        # If we already have an Orion-authored candidate, keep it.
        if best is not None and _observer_is_orion(best.get("observer")):
            return
        # Prefer Orion-authored.
        if _observer_is_orion(obj.get("observer")):
            best = obj
            return
        # Otherwise keep as fallback if none yet.
        if best is None:
            best = obj

    if isinstance(prior_step_results, dict) and _is_collapse_shape(prior_step_results):
        consider(prior_step_results)
        return best

    for s in _find_any_string(prior_step_results):
        obj = _extract_first_json_object(s)
        if isinstance(obj, dict):
            consider(obj)
            if best is not None and _observer_is_orion(best.get("observer")):
                return best

    if isinstance(prior_step_results, dict):
        for v in prior_step_results.values():
            found = find_collapse_entry(v)
            if isinstance(found, dict):
                consider(found)
                if best is not None and _observer_is_orion(best.get("observer")):
                    return best
    elif isinstance(prior_step_results, list):
        for v in prior_step_results:
            found = find_collapse_entry(v)
            if isinstance(found, dict):
                consider(found)
                if best is not None and _observer_is_orion(best.get("observer")):
                    return best

    return best


CollapseMirrorEntry = CollapseMirrorEntryV2
