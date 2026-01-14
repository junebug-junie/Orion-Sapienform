from __future__ import annotations

import json
from datetime import datetime, timezone
import os
import re
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
        cleaned = part.strip().strip("\"").strip("'")
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


def _is_v2_shape(data: Dict[str, Any]) -> bool:
    return V2_MIN_REQUIRED_KEYS.issubset(data.keys())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    timestamp: str = Field(default_factory=_utc_now_iso)
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
                return data
            return v1_to_v2(CollapseMirrorEntryV1.model_validate(data)).model_dump(mode="json")
        return data

    @model_validator(mode="after")
    def _fill_defaults(self) -> "CollapseMirrorEntryV2":
        if not self.timestamp:
            self.timestamp = _utc_now_iso()
        if not self.environment:
            self.environment = os.getenv("CHRONICLE_ENVIRONMENT", "dev")
        if not self.snapshot_kind:
            self.snapshot_kind = "baseline"
        if not self.what_changed_summary:
            self.what_changed_summary = self.summary
        if not self.change_type:
            self.change_type = self.type
        if not self.tags:
            if self.type:
                self.tags = [self.type]
        if not self.state_snapshot.observer_state:
            self.state_snapshot.observer_state = list(self.observer_state)
        if not self.state_snapshot.field_resonance:
            self.state_snapshot.field_resonance = self.field_resonance
        if not self.state_snapshot.tags and self.tags:
            self.state_snapshot.tags = list(self.tags)
        if not self.event_id:
            self.event_id = f"collapse_{uuid4().hex}"
        if self.event_id and not self.id:
            self.id = self.event_id
        elif self.id and not self.event_id:
            self.event_id = self.id
        # If both are present but different, current logic keeps them as is,
        # but typically event_id is the authoritative one.
        # However, to be safe, we can enforce id = event_id if event_id is present.
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

        # If we receive a rich object (dict), serialize it to JSON
        # so we can store it in the SQL VARCHAR column without crashing.
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
        id=v1.id, # Pass explicit ID if present
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
        change_type=v1.type,
        tags=[v1.type] if v1.type else [],
        causal_density=CollapseMirrorCausalDensity(label="ambient", score=0.0),
        is_causally_dense=False,
        epistemic_status="observed",
        visibility="internal",
        redaction_level="low",
    )


def normalize_collapse_entry(payload: Any) -> CollapseMirrorEntryV2:
    if isinstance(payload, CollapseMirrorEntryV2):
        return payload.with_defaults()
    if isinstance(payload, CollapseMirrorEntryV1):
        return v1_to_v2(payload).with_defaults()
    if isinstance(payload, dict) and V1_REQUIRED_KEYS.issubset(payload.keys()) and not _is_v2_shape(payload):
        return v1_to_v2(CollapseMirrorEntryV1.model_validate(payload)).with_defaults()
    return CollapseMirrorEntryV2.model_validate(payload).with_defaults()


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


def find_collapse_entry(prior_step_results: Any) -> Optional[Dict[str, Any]]:
    if prior_step_results is None:
        return None

    if isinstance(prior_step_results, dict) and (
        _is_v2_shape(prior_step_results) or V1_REQUIRED_KEYS.issubset(prior_step_results.keys())
    ):
        return prior_step_results

    for s in _find_any_string(prior_step_results):
        obj = _extract_first_json_object(s)
        if isinstance(obj, dict) and (_is_v2_shape(obj) or V1_REQUIRED_KEYS.issubset(obj.keys())):
            return obj

    if isinstance(prior_step_results, dict):
        for v in prior_step_results.values():
            found = find_collapse_entry(v)
            if found:
                return found
    elif isinstance(prior_step_results, list):
        for v in prior_step_results:
            found = find_collapse_entry(v)
            if found:
                return found

    return None


CollapseMirrorEntry = CollapseMirrorEntryV2
