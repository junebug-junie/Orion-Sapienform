# app/state.py
from dataclasses import dataclass, field
from typing import Optional

from .schemas import Event


@dataclass
class VisionState:
    frame_counter: int = 0
    last_event: Optional[Event] = None


state = VisionState()
