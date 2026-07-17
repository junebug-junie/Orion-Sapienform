import json

from sqlalchemy import Boolean, Column, DateTime, JSON, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import TypeDecorator
from app.db import Base

# Generic JSON on SQLite (tests), JSONB on Postgres (prod) -- same pattern as
# app/models/drive_audit.py's _JSONB, declared once so Base.metadata.create_all
# and the boot DDL in app/main.py agree on the type.
_JSONB = JSON().with_variant(JSONB(), "postgresql")


class _JSONSafeList(TypeDecorator):
    """`edges`/`divergence` are lists of `CausalGeometryEdgeV1`/
    `CausalGeometryDivergenceEntryV1` sub-models, each carrying their own
    `datetime` fields (window_start/window_end). The generic `_write_row()`
    path builds this column's value via plain `obj.model_dump()` (not
    `mode="json"`), so those nested datetimes survive as native `datetime`
    objects -- neither SQLite's JSON type nor psycopg2's default JSONB adapter
    can serialize a raw `datetime` inside a nested dict. Round-tripping through
    `json.dumps(..., default=str)` / `json.loads(...)` here converts anything
    non-JSON-native (datetimes, etc.) to its string form before it ever reaches
    the DB driver, regardless of which mode upstream code dumped the pydantic
    model with.
    """

    impl = _JSONB
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return json.loads(json.dumps(value, default=str))


class CausalGeometrySnapshotSQL(Base):
    """Causal Geometry v1 Phase A measurement persistence.

    Column names deliberately match `orion.schemas.causal_geometry.CausalGeometrySnapshotV1`'s
    own pydantic field names exactly (edges/divergence/notes, not edges_json/etc.) so the
    generic `_write_row()` path in app/worker.py needs no special-casing for this kind --
    the schema-validated payload's `model_dump()` keys map directly onto these columns.

    Produced by orion-field-digester (`orion/substrate/causal_geometry_producer.py`) on
    channel `orion:causal_geometry:snapshot`, kind `causal.geometry.snapshot.v1`. Read by
    orion-hub's `/api/causal-geometry/{snapshot,history}` endpoints. `snapshot_id` is the
    primary key; rows are immutable (fresh id per cycle), so this model is in the worker's
    `INSERT_ONLY_MODELS` fast path -- a re-delivered event hits the duplicate-key catch and
    is skipped, with no per-write merge SELECT.
    """

    __tablename__ = "causal_geometry_snapshots"

    snapshot_id = Column(String, primary_key=True)
    generated_at = Column(DateTime(timezone=True), nullable=False)
    window_start = Column(DateTime(timezone=True), nullable=False)
    window_end = Column(DateTime(timezone=True), nullable=False)
    designed_topology_version = Column(String, nullable=True)
    insufficient_data = Column(Boolean, nullable=False)
    edges = Column(_JSONSafeList, nullable=False)
    divergence = Column(_JSONSafeList, nullable=False)
    notes = Column(_JSONB, nullable=False)
