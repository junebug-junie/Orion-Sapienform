from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.db import Base


class VisionSceneInventorySQL(Base):
    """One row per vision window: what was in the room and how many.

    Written unconditionally, unlike `vision_events`. The council only produces
    an event when the observed LABEL SET changes (`reason=stable_scene`
    otherwise), so the event stream has gaps exactly where object permanence
    needs continuity -- a count going from two boxes to one emits nothing, and
    a departure is a non-event by nature.

    `counts` is the per-frame max (how many are believed present).
    `detections` is the raw tally across the window and scales with
    `frame_count`; the two are kept apart on purpose, because conflating them
    is the bug this table's producer was fixed for on 2026-08-21.
    """

    __tablename__ = "vision_scene_inventory"

    window_id = Column(String, primary_key=True)
    stream_id = Column(String, nullable=True, index=True)
    camera_id = Column(String, nullable=True)
    observed_at = Column(DateTime(timezone=True), nullable=False, index=True)
    window_start_ts = Column(Float, nullable=True)
    window_end_ts = Column(Float, nullable=True)
    frame_count = Column(Integer, nullable=False, default=0)
    counts = Column(JSONB, nullable=False, default=dict)
    detections = Column(JSONB, nullable=False, default=dict)
    believed_labels = Column(JSONB, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
