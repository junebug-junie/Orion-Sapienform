from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    # Additive (2026-08-19, P2, docs/superpowers/specs/2026-08-12-perception-
    # frontier-design.md). Previously the embedding vector itself lived only
    # in the on-disk .npy at `path`, inside orion-vision-host's own model
    # cache volume -- not a documented cross-service seam another service may
    # reach into (CLAUDE.md section 5). Inlining the vector on the wire is
    # what lets a bus consumer (e.g. orion-substrate-runtime's perceptual
    # prediction-error tick) score it without touching vision-host's
    # filesystem. `ref`/`path`/`dim` are unchanged and still written for the
    # existing on-disk/reference consumers -- this is a new field, not a
    # replacement.
    vector: Optional[List[float]] = None


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
    """Where a captured frame is, so a consumer can go and read it.

    Carries a POINTER, never bytes -- a frame is ~100KB and this crosses Redis
    on every capture.

    Two addressing modes, and which one a node uses is a property of the node,
    not a preference:

    * ``image_path`` -- a path on a filesystem the consumer shares. Cheapest,
      no copy, and correct for capture running on the same host as the vision
      host. athena uses this and is unchanged.
    * ``sha256`` -- a content address in orion-percept-store. The ONLY option
      for a node with no shared filesystem, which is every node except athena.
      Until this field existed, a second machine physically could not feed this
      pipeline no matter what else was configured.

    At least one must be set; a pointer that points nowhere is not a pointer.
    A frame MAY carry both (uploaded and also on local disk), and consumers
    should prefer whichever they can actually reach rather than assuming.
    """

    model_config = ConfigDict(extra="forbid")
    image_path: Optional[str] = None
    # 64-char lowercase hex. Validated here rather than at every consumer,
    # because it becomes part of a fetch URL downstream and the gateway's
    # rebuild-from-trusted-base assumes it is exactly this shape.
    sha256: Optional[str] = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="Content address in orion-percept-store, for nodes with no shared filesystem.",
    )
    frame_paths: Optional[List[str]] = None
    video_path: Optional[str] = None
    camera_id: Optional[str] = None
    stream_id: Optional[str] = None
    frame_ts: Optional[float] = None
    clip_id: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None

    @model_validator(mode="after")
    def _require_an_address(self) -> "VisionFramePointerPayload":
        """A pointer with neither address is silently undeliverable.

        Without this the failure surfaces far downstream as "task produced no
        artifact", which is indistinguishable from a detector finding nothing.
        """
        if not (self.image_path or self.frame_paths or self.video_path or self.sha256):
            raise ValueError(
                "frame pointer needs image_path, frame_paths, video_path or sha256"
            )
        return self


# Alias for explicit requirement
VisionFramePointer = VisionFramePointerPayload


class RetinaClipCaptureRequestPayload(BaseModel):
    """Bus RPC request: 'record an on-demand video+audio clip right now.'

    Mirrors ``POST /capture/clip`` on orion-vision-retina (2026-08-22,
    services/orion-vision-retina/app/clip_capture.py) -- this is the
    bus-reachable twin of that HTTP route, for callers with no network path
    to the capturing node (e.g. carbon accepts no inbound HTTP per
    docs/operations/carbon-webcam.md; the bus is its only reachable surface).

    Deliberately empty of tunable fields for v1: duration/framerate/device
    are the capturing node's own configured defaults
    (RETINA_CLIP_DURATION_SEC etc.), not caller-overridable -- a remote
    caller dictating recording parameters to a physical webcam it cannot see
    is a bigger surface than this capability needs yet.
    """

    model_config = ConfigDict(extra="forbid")


class RetinaClipCaptureResultPayload(BaseModel):
    """Reply to RetinaClipCaptureRequestPayload. Field-for-field mirror of
    the JSON body POST /capture/clip already returns -- see that route's
    docstring for the "refs are not yet consumable end-to-end" caveat this
    inherits (services/orion-affectgpt-worker fetch-by-hash side)."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    video_sha256: Optional[str] = None
    audio_sha256: Optional[str] = None
    duration_sec: Optional[float] = None
    video_bytes: Optional[int] = None
    audio_bytes: Optional[int] = None
    error: Optional[str] = None
    error_code: Optional[str] = None


class VisionEdgeActivityPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stream_id: str
    camera_id: Optional[str] = None
    labels: List[str] = Field(default_factory=list)
    max_score: float = 1.0
    frame_ts: Optional[float] = None
    image_path: Optional[str] = None
    artifact_id: Optional[str] = None


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


class VisionSceneInventoryV1(BaseModel):
    """One window's observed inventory of the scene, persisted per window.

    **Why this exists as its own record rather than riding on
    ``vision_events``.** Object permanence needs a continuous record of what
    was present, and the event stream cannot supply one: the council's
    evidence-transition gate only re-interprets on a **label-set** change and
    logs ``reason=stable_scene`` otherwise, so a pure *count* change (two boxes
    become one box) produces no event at all. A departure is also a non-event
    by nature -- nothing fires when a thing stops being there. Both facts mean
    the inventory has to be written on every window, unconditionally, and read
    later by a timer-driven reducer.

    ``counts`` is the per-frame **max** from
    ``orion-vision-window.projection.summarize_items`` -- an estimate of how
    many are in the room. It is deliberately not the detection tally, which
    scales with the number of frames in the window and which
    ``detections`` carries separately under an honest name. The two differed by
    exactly the frame count until 2026-08-21; see that function's docstring.

    **Privacy.** Labels and counts only. No frame path, no bounding boxes, no
    caption text, no embedding, and nothing identity-bearing. This is a
    furniture census, and it should stay one: an inventory table is a poor
    place to discover that it has quietly started recording people.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["vision.scene.inventory.v1"] = "vision.scene.inventory.v1"
    window_id: str
    stream_id: Optional[str] = None
    camera_id: Optional[str] = None
    observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    window_start_ts: Optional[float] = None
    window_end_ts: Optional[float] = None
    # Frames that actually contributed. A count derived from 0 frames is
    # absence-of-evidence, not evidence-of-absence, and the reducer must be
    # able to tell those apart.
    frame_count: int = Field(default=0, ge=0)
    # label -> how many are believed to be present (per-frame max).
    counts: Dict[str, int] = Field(default_factory=dict)
    # label -> raw detections fired across the window. Frame-rate dependent.
    detections: Dict[str, int] = Field(default_factory=dict)
    # Habituated belief set from the window service's SceneBeliefTracker.
    believed_labels: List[str] = Field(default_factory=list)


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
