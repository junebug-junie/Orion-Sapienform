# app/models.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class Detection(BaseModel):
    kind: str
    bbox: Optional[Tuple[int, int, int, int]] = None
    score: float = 1.0
    label: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="ignore")


class VisionEvent(BaseModel):
    ts: datetime
    stream_id: str
    frame_index: Optional[int] = None
    detections: List[Detection] = Field(default_factory=list)
    meta: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="ignore")


class SecurityState(BaseModel):
    enabled: bool
    armed: bool
    mode: str
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None


class VisitSummary(BaseModel):
    ts: datetime
    service: str
    version: str

    visit_id: str
    camera_id: str
    started_at: datetime
    ended_at: datetime
    duration_sec: float

    frames_seen: int
    motion_frames: int
    human_frames: int

    humans_present: bool
    num_tracks: int = 1
    num_unknown_tracks: int
    num_known_tracks: int

    best_identity: str
    best_identity_conf: float
    identity_votes: Dict[str, float]

    security_decision: Dict[str, Any]


class AlertSnapshot(BaseModel):
    kind: str  # "snapshot" or "prebuffer" etc.
    captured_at: datetime
    url: Optional[str] = None
    filename: Optional[str] = None
    note: Optional[str] = None


class AlertPayload(BaseModel):
    ts: datetime
    service: str
    version: str

    alert_id: str
    visit_id: str
    camera_id: str

    armed: bool
    mode: str

    humans_present: bool
    best_identity: str
    best_identity_conf: float
    identity_votes: Dict[str, float]

    reason: str
    severity: str

    snapshots: List[AlertSnapshot] = Field(default_factory=list)

    rate_limit: Dict[str, Any]
