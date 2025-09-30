
from pydantic import BaseModel
from typing import List, Tuple, Literal, Optional
from datetime import datetime

BBox = Tuple[int, int, int, int]  # x, y, w, h

class Detection(BaseModel):
    kind: str
    bbox: BBox
    score: float = 1.0
    label: Optional[str] = None

class Event(BaseModel):
    ts: datetime
    stream_id: str
    detections: List[Detection] = []
    note: Optional[str] = None
