from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

BBox = Tuple[int, int, int, int]  # x, y, w, h


class Detection(BaseModel):
    kind: str                        # "motion", "face", "yolo", "presence", ...
    bbox: BBox                       # 0,0,0,0 for non-spatial events like presence
    score: float = 1.0
    label: Optional[str] = None      # e.g. "person", "car", "Juniper?"

    # Future-proof hooks (you don't have to use them yet)
    track_id: Optional[str] = None   # per-stream temporary ID for a blob/person
    meta: Dict[str, Any] = Field(default_factory=dict)


class Event(BaseModel):
    """
    Raw vision detection event from the edge service.

    This is intentionally low-level: a single timestamp/frame + multiple detections.
    Higher-level services (identity, episodes, etc.) subscribe and build on this.
    """
    ts: datetime
    stream_id: str                   # "cam0" / "living_room" / etc.
    frame_index: Optional[int] = None

    service: str = "vision"
    service_version: str = "0.2.0"

    detections: List[Detection] = Field(default_factory=list)
    note: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
