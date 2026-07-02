from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Literal

from pydantic import BaseModel, ConfigDict, Field


class VisionObject(BaseModel):
    label: str
    score: float
    box_xyxy: List[float]
    class_id: Optional[int] = None


# Alias for explicit requirement
VisionDetection = VisionObject


class VisionCaption(BaseModel):
    text: str
    confidence: Optional[float] = None


class VisionEmbedding(BaseModel):
    ref: str
    path: str
    dim: int


class VisionArtifactOutputs(BaseModel):
    model_config = ConfigDict(extra="allow")
    objects: Optional[List[VisionObject]] = None
    caption: Optional[VisionCaption] = None
    embedding: Optional[VisionEmbedding] = None


class VisionArtifactPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_id: str
    correlation_id: str
    task_type: str
    device: str
    inputs: Dict[str, Any]
    outputs: VisionArtifactOutputs
    timing: Dict[str, float]
    model_fingerprints: Dict[str, str]
    # Debug refs (overlays, etc)
    debug_refs: Optional[Dict[str, str]] = None


# Specific Edge Artifact Schema (matches VisionArtifactPayload but with stricter intent)
class VisionEdgeArtifact(VisionArtifactPayload):
    model_fingerprints: dict
    model_config = ConfigDict(protected_namespaces=())

class VisionTaskRequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_type: str = Field(..., description="embed_image|detect_open_vocab|caption_frame|retina_fast")
    request: Dict[str, Any]
    meta: Optional[Dict[str, Any]] = None


class VisionTaskResultPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    task_type: str
    device: Optional[str] = None
    error: Optional[str] = None
    error_code: Optional[str] = Field(
        default=None,
        description="Stable machine-readable failure reason when ok=false (mirrors runner/service meta).",
    )
    # result: Optional[Dict[str, Any]] = None # Deprecating in favor of typed artifact
    artifact: Optional[VisionArtifactPayload] = None
    timings: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional extras (e.g. warnings); omit empty.",
    )


class VisionFramePointerPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_path: Optional[str] = None
    frame_paths: Optional[List[str]] = None
    video_path: Optional[str] = None
    camera_id: Optional[str] = None
    stream_id: Optional[str] = None
    frame_ts: Optional[float] = None
    clip_id: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None


# Alias for explicit requirement
VisionFramePointer = VisionFramePointerPayload


class VisionWindowPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    window_id: str
    start_ts: float
    end_ts: float
    summary: Dict[str, Any]
    artifact_ids: List[str]
    # For one-shot flow, it's helpful to carry the full artifacts if needed,
    # but the schema usually just has IDs. We'll stick to IDs + summary for payload.
    # --- Optional projection envelope (vision_window_snapshot.v1); omitted on legacy payloads ---
    schema_version: Optional[str] = Field(
        default=None,
        description="When set, e.g. vision_window_snapshot.v1 — orion-vision-window projection contract.",
    )
    stream_id: Optional[str] = None
    source_node: Optional[str] = None
    camera_id: Optional[str] = None
    cursor: Optional[str] = None
    upstream_event_ids: List[str] = Field(default_factory=list)
    artifact_uris: List[str] = Field(default_factory=list)
    freshness: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None


class VisionEventBundleItem(BaseModel):
    event_id: str
    event_type: str
    narrative: str
    entities: List[str]
    tags: List[str]
    confidence: float
    salience: float
    evidence_refs: List[str]


class VisionEventPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    events: List[VisionEventBundleItem]


class VisionSceneEntityV1(BaseModel):
    entity_id: str | None = None
    label: str
    entity_type: str | None = None
    confidence: float = 0.5
    evidence_refs: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class VisionSceneRelationV1(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float = 0.5
    evidence_refs: list[str] = Field(default_factory=list)


class VisionSalientObservationV1(BaseModel):
    observation: str
    salience: float = 0.5
    confidence: float = 0.5
    evidence_refs: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class VisionUncertaintyV1(BaseModel):
    uncertainty: str
    reason: str | None = None
    confidence: float = 0.5
    evidence_refs: list[str] = Field(default_factory=list)


class VisionTaskRelevanceV1(BaseModel):
    task: str
    relevance: float = 0.5
    reason: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)


class VisionMemoryDeltaCandidateV1(BaseModel):
    claim: str
    claim_kind: str = "observation"
    confidence: float = 0.5
    salience: float = 0.5
    evidence_refs: list[str] = Field(default_factory=list)
    should_persist: bool = False
    reason: str | None = None


class VisionEventCandidateV1(BaseModel):
    event_type: str
    narrative: str
    entities: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    salience: float = 0.5
    evidence_refs: list[str] = Field(default_factory=list)


class VisionGrammarProjectionCandidateV1(BaseModel):
    atoms: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class VisionSceneInterpretationV1(BaseModel):
    schema_version: str = "1.0"
    window_id: str
    stream_id: str | None = None
    camera_id: str | None = None
    scene_summary: str
    scene_state: dict[str, Any] = Field(default_factory=dict)
    entities: list[VisionSceneEntityV1] = Field(default_factory=list)
    relations: list[VisionSceneRelationV1] = Field(default_factory=list)
    salient_observations: list[VisionSalientObservationV1] = Field(default_factory=list)
    uncertainties: list[VisionUncertaintyV1] = Field(default_factory=list)
    task_relevance: list[VisionTaskRelevanceV1] = Field(default_factory=list)
    event_candidates: list[VisionEventCandidateV1] = Field(default_factory=list)
    memory_delta_candidates: list[VisionMemoryDeltaCandidateV1] = Field(default_factory=list)
    grammar_projection: VisionGrammarProjectionCandidateV1 | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    raw_model_output: dict[str, Any] | None = None


class VisionScribeAckPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    message: Optional[str] = None
    error: Optional[str] = None


class VisionGuardSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    camera_id: str
    window_start: float
    window_end: float
    decision: Literal["presence", "unknown", "absent", "alert"]
    confidence: float
    summary: Dict[str, Any]
    evidence_refs: List[str]  # List of artifact_ids
    salience: float = 0.0


class VisionGuardAlert(BaseModel):
    model_config = ConfigDict(extra="forbid")
    camera_id: str
    ts: float
    alert_type: str
    severity: Literal["low", "medium", "high"]
    summary: str
    evidence_refs: List[str]
    snapshot_path: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class VisionEdgeHealth(BaseModel):
    model_config = ConfigDict(extra="forbid")
    camera_id: str
    ts: float
    ok: bool
    fps: float
    mean_brightness: Optional[float] = None
    resolution: Optional[str] = None
    dropped_frames: Optional[int] = None


class VisionEdgeError(BaseModel):
    model_config = ConfigDict(extra="forbid")
    camera_id: str
    ts: float
    error_type: str
    message: str
    meta: Optional[Dict[str, Any]] = None


# --- Cortex / RPC Schemas ---

class VisionWindowRequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # For one-shot, we pass the artifact directly
    artifact: VisionArtifactPayload

class VisionWindowResultPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    window: VisionWindowPayload

class VisionCouncilRequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    window: VisionWindowPayload

class VisionCouncilResultPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    events: VisionEventPayload

class VisionScribeRequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    events: VisionEventPayload

class VisionScribeResultPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ack: VisionScribeAckPayload
