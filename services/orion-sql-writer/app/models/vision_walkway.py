"""ORM for the walkway-camera rows the bus worker writes.

Deliberately on its OWN declarative base, not ``app.db.Base``: ``main.py``
runs ``Base.metadata.create_all`` at boot, which would create these tables
from the ORM shape and silently skip the CHECK constraints (the patio rule)
and partial indexes that ``services/orion-sql-db/manual_migration_walkway_
camera_v1.sql`` defines. The migration is the single source of truth for
these tables; if it is not applied, writes fail loudly instead of landing
in a weaker table.

Only ``vision_unresolved`` goes through the generic ``MODEL_MAP`` path.
``vision_crop_observation`` fans out per crop (``app/vision_crop_persist.py``)
and the reducer tables are written with explicit SQL by their loops.
"""

from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

WalkwayBase = declarative_base()


class VisionUnresolvedSQL(WalkwayBase):
    __tablename__ = "vision_unresolved"

    unresolved_id = Column(String, primary_key=True)
    stream_id = Column(String, nullable=True)
    camera_id = Column(String, nullable=True)
    window_id = Column(String, nullable=True)
    observed_at = Column(DateTime(timezone=True), nullable=False)
    reason = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    what_was_tried = Column(JSONB, nullable=False, default=list)
    evidence_refs = Column(JSONB, nullable=False, default=list)
    image_ref = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
