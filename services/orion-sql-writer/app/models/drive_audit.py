from sqlalchemy import Column, DateTime, Integer, JSON, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.db import Base

# Generic JSON on SQLite (tests), JSONB on Postgres (prod). Declared once so
# Base.metadata.create_all and the boot DDL in app/main.py agree on the type.
_JSONB = JSON().with_variant(JSONB(), "postgresql")


class DriveAuditSQL(Base):
    """Slim measurement record of a drive audit tick.

    Produced by orion-spark-concept-induction via `memory.drives.audit.v1`
    (schema `DriveAuditV1`, `orion/core/schemas/drives.py`) on channel
    `orion:memory:drives:audit`. Read by the autonomy measurement gate, which
    does windowed range scans over `COALESCE(observed_at, created_at)` and
    reads `active_count`.
    `artifact_id` is the primary key so re-delivered events upsert idempotently.

    This is intentionally NOT an archive of the full artifact:
    evidence_items/source_event_refs/summary/tick_attribution are dropped by
    the generic `_write_row` column filter. `active_count` is derived in the
    sql-writer worker as `len(active_drives)` (0 when malformed/absent) — it is
    not on the wire payload. `active_drives`/`drive_pressures` are already
    bounded upstream to the 6 fixed DRIVE_KEYS, so no capping is needed here.

    Secondary indexes are intentionally NOT declared here via `index=True`.
    They are owned by the boot DDL in `app/main.py` (and the standalone
    migration `services/orion-sql-db/manual_migration_drive_audits_v1.sql`) so
    there is a single source of truth: an expression index on
    `COALESCE(observed_at, created_at) DESC` matching the gate's windowed
    range scans. Declaring them here too would make
    `Base.metadata.create_all` emit duplicate `ix_*` indexes alongside the
    DDL's `idx_*` indexes.
    """

    __tablename__ = "drive_audits"

    artifact_id = Column(String, primary_key=True)
    subject = Column(String, nullable=False)
    active_count = Column(Integer, nullable=False)
    active_drives = Column(_JSONB, nullable=True)
    dominant_drive = Column(String, nullable=True)
    drive_pressures = Column(_JSONB, nullable=True)
    correlation_id = Column(String, nullable=True)
    observed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
