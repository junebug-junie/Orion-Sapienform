from sqlalchemy import BigInteger, Column, DateTime, Index, String

from app.db import Base


class EquilibriumServiceTransitionSQL(Base):
    """Historical downtime persistence for orion-equilibrium-service.

    Column names deliberately match `orion.schemas.telemetry.system_health.
    EquilibriumServiceTransitionV1`'s own pydantic field names exactly
    (same convention as `CausalGeometrySnapshotSQL`) so the generic
    `_write_row()` path in app/worker.py needs no special-casing for this
    kind.

    Produced by orion-equilibrium-service (`services/orion-equilibrium-
    service/app/downtime_transition_tracker.py`) on channel
    `orion:equilibrium:transition`, kind `equilibrium.service.transition.v1`,
    the instant it detects a tracked service's `status` actually changed --
    NOT on every periodic snapshot tick. See
    docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.

    `transition_id` is the primary key; rows are immutable (fresh id per
    transition), so this model is in the worker's `INSERT_ONLY_MODELS` fast
    path -- a re-delivered event hits the duplicate-key catch and is
    skipped (idempotent), with no per-event merge SELECT.

    A downtime episode's duration is answerable from a single row: the
    recovery transition (`from_status == "down"`) carries `down_since_ts`
    and `down_duration_ms`, so "how long was X down last week" does not
    need to join a paired down/up event.

    `down_duration_ms` is `BigInteger`, not `Integer` -- review caught that
    Postgres `INTEGER` (4-byte signed, max ~24.85 days in milliseconds) is a
    real overflow risk here: `EquilibriumService._calculate_metrics()`
    re-adds any `expected_services` entry that never heartbeats as
    `status="down"` on every tick indefinitely (no cap), so a service can sit
    tracked-as-down for longer than 24.85 days before it ever reports back
    in and fires a recovery transition with an out-of-range duration.
    """

    __tablename__ = "equilibrium_service_transitions"
    __table_args__ = (
        Index("ix_equilibrium_service_transitions_service_transitioned_at", "service", "transitioned_at"),
    )

    transition_id = Column(String, primary_key=True)
    service = Column(String, nullable=False, index=True)
    node = Column(String, nullable=True)
    boot_id = Column(String, nullable=True)
    version = Column(String, nullable=True)
    instance = Column(String, nullable=True)

    from_status = Column(String, nullable=False)
    to_status = Column(String, nullable=False)
    transitioned_at = Column(DateTime(timezone=True), nullable=False, index=True)

    down_since_ts = Column(DateTime(timezone=True), nullable=True)
    down_duration_ms = Column(BigInteger, nullable=True)

    source_service = Column(String, nullable=False)
    source_node = Column(String, nullable=True)
    producer_boot_id = Column(String, nullable=False)
