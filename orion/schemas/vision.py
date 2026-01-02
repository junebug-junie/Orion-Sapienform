from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class VisionObject(BaseModel):
    label: str
    score: float
    box_xyxy: List[float]


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
    result: Optional[Dict[str, Any]] = None
    artifact_id: Optional[str] = None
    timings: Optional[Dict[str, float]] = None


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


class VisionFramePointerPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_path: Optional[str] = None
    frame_paths: Optional[List[str]] = None
    video_path: Optional[str] = None
    camera_id: Optional[str] = None
    stream_id: Optional[str] = None
    frame_ts: Optional[float] = None
    clip_id: Optional[str] = None


class VisionWindowPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    window_id: str
    start_ts: float
    end_ts: float
    summary: Dict[str, Any]
    artifact_ids: List[str]


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


class VisionScribeAckPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    message: Optional[str] = None
    error: Optional[str] = None
