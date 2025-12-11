# app/models.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Literal

from pydantic import BaseModel

BBox = Tuple[int, int, int, int]


class Detection(BaseModel):
    """
    Detection coming from the vision edge service.

    We align this with what edge publishes on orion:vision:edge:raw:
    - kind: "face", "motion", "yolo", "presence", etc.
    - bbox: [x, y, w, h]
    - score: confidence
    - label: e.g. "person" for YOLO; face has None
    - track_id: optional tracking id (future)
    - meta: free-form extra info
    """
    kind: str
    bbox: BBox
    score: float = 1.0
    label: Optional[str] = None
    track_id: Optional[str | int] = None
    meta: Dict[str, Any] = {}


class VisionEvent(BaseModel):
    """
    Raw event from vision edge, as seen on orion:vision:edge:raw.
    """

    ts: datetime
    stream_id: str
    frame_index: Optional[int] = None

    service: Optional[str] = None
    service_version: Optional[str] = None

    detections: List[Detection]
    note: Optional[str] = None
    meta: Dict[str, Any] = {}


class SecurityState(BaseModel):
    """
    Simple persistent security state.
    """

    armed: bool = False         # whether system is currently armed
    mode: str = "off"           # "off", "vacation_strict", etc.
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None


class VisitSummary(BaseModel):
    """
    Summary of a logical 'visit' (a contiguous episode of humans present).
    v1 is simple: we mostly care that someone was there while armed.
    """

    visit_id: str
    first_ts: datetime
    last_ts: datetime
    stream_id: str
    humans_present: bool = True
    events: int = 1


class AlertSnapshot(BaseModel):
    """
    A single snapshot image captured around the time of an alert.
    """

    ts: datetime
    path: str


class AlertPayload(BaseModel):
    """
    What we publish as an alert + optionally email.

    identity_* is future-proof; for now best_identity is always "unknown".
    """

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
    severity: Literal["low", "medium", "high"] = "high"

    snapshots: List[AlertSnapshot] = []
    rate_limit: Dict[str, Any]
