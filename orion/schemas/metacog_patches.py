from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class MetacogDraftWhatChangedV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: Optional[str] = None
    evidence: Optional[List[str]] = None
    new_state: Optional[str] = None
    previous_state: Optional[str] = None


class MetacogDraftTextPatchV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Optional[str] = None
    trigger: Optional[str] = None
    mantra: Optional[str] = None
    summary: Optional[str] = None
    causal_echo: Optional[str] = None
    field_resonance: Optional[str] = None
    emergent_entity: Optional[str] = None
    resonance_signature: Optional[str] = None
    what_changed: Optional[MetacogDraftWhatChangedV1] = None
    tags_suggested: Optional[List[str]] = None


class MetacogConstraintsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    notes: Optional[str] = None
    severity_score: Optional[float] = None


class MetacogNumericSistersV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arousal: Optional[float] = None
    clarity: Optional[float] = None
    valence: Optional[float] = None
    overload: Optional[float] = None
    risk_score: Optional[float] = None
    constraints: Optional[MetacogConstraintsV1] = None


class MetacogCausalDensityV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Optional[str] = None
    score: Optional[float] = None
    rationale: Optional[str] = None


class MetacogEnrichScorePatchV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tag_scores: Optional[Dict[str, float]] = None
    change_type_scores: Optional[Dict[str, float]] = None
    numeric_sisters: Optional[MetacogNumericSistersV1] = None
    causal_density: Optional[MetacogCausalDensityV1] = None
    epistemic_status: Optional[str] = None
    is_causally_dense: Optional[bool] = None
    snapshot_kind: Optional[str] = None
