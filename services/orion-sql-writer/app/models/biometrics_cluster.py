from sqlalchemy import Column, DateTime, Float, Index, String, func
from sqlalchemy.dialects.postgresql import JSONB
from app.db import Base
from uuid import uuid4


class BiometricsClusterSQL(Base):
    """Persisted fleet aggregate from ``orion:biometrics:cluster``.

    ``orion/bus/channels.yaml`` has listed ``orion-sql-writer`` as a consumer of this
    channel for as long as the channel has existed, but no model or handler was ever
    written -- so the contract said "consumed" while the payload was dropped. This is
    that handler.

    WHY IT MATTERS BEYOND CLOSING THE GAP: this is the only place the fleet total
    exists. Per-node rows in ``orion_biometrics_summary`` cannot substitute, because
    circe has no LAN path to the PDU and no BMC -- its watts reach the fleet only as a
    PROXIED reading taken by athena on its behalf, which
    ``biometrics/app/main.py`` deliberately refuses to write into circe's own row (a
    proxied value must never pass as a self-report). So a query over per-node rows sees
    athena and silently misses roughly two thirds of the fleet's draw.

    TWO CLOCKS, DELIBERATELY BOTH KEPT.
    ``observed_at`` is the payload's own timestamp -- when the aggregate described the
    world. ``created_at`` is when this row was written. They are not the same clock and
    must not be conflated: a bus backlog replay writes old observations at a new wall
    time. Settlement windows (does a declared power intent match what the meter saw?)
    MUST range over ``observed_at``; retention ages rows by ``created_at``, matching
    every other retention-managed table in this service.

    WATTS ARE HOISTED INTO REAL COLUMNS, not left only in ``measurements``. Settlement
    is a time-range aggregate over watts, and JSONB extraction in a range predicate
    cannot use a btree the way a float column can. ``measurements`` is still stored
    whole, because it carries keys these columns do not.

    READING ``chassis_watts`` OR ``pdu_watts`` WITHOUT ``measurements_missing`` AND
    ``nodes_absent`` IS READING A PARTIAL SUM AS A COMPLETE ONE. Both are persisted for
    exactly that reason -- see BiometricsClusterV1's own field comments, and the live
    2026-08-14 case where a two-machine sum presented as the whole fleet.
    ``measurements_proxied`` names which keys on which node were measured by somebody
    else; it is the provenance that keeps a proxied reading from later being mistaken
    for a node's own recovery.
    """

    __tablename__ = "orion_biometrics_cluster"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))

    # Occurrence time (payload). Settlement ranges over THIS.
    observed_at = Column(DateTime(timezone=True), nullable=False)
    # Write time. Retention ages rows by THIS.
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # Hoisted fleet physical units, in watts. Nullable because a key absent from
    # `measurements` means UNMEASURED, never 0.0 -- storing a zero here would invent a
    # reading that no instrument produced.
    pdu_watts = Column(Float, nullable=True)
    chassis_watts = Column(Float, nullable=True)
    gpu_watts_total = Column(Float, nullable=True)
    gpu_count = Column(Float, nullable=True)

    # Named to mirror the payload and BiometricsSummarySQL's own column.
    constraint = Column(String, nullable=True)
    peak_pressure = Column(Float, nullable=True)
    peak_pressure_channel = Column(String, nullable=True)
    peak_pressure_node = Column(String, nullable=True)

    # none_as_null=True for the same reason BiometricsSummarySQL.measurements needs it:
    # without it SQLAlchemy stores Python None as JSONB 'null', creating a third state
    # that `IS NULL` silently misses. Confirmed live on that table 2026-08-14.
    # The normalised 0-1 fleet signals. Persisted rather than dropped: the PER-NODE
    # versions live in orion_biometrics_summary, but the FLEET-level ones exist nowhere
    # else, and _write_row discards an unmapped key silently rather than raising -- so
    # "we only needed watts" would have become permanent invisible data loss. JSONB is
    # cheap; a missing column is not recoverable after the fact.
    role_weights = Column(JSONB(none_as_null=True), nullable=True)
    pressures = Column(JSONB(none_as_null=True), nullable=True)
    headroom = Column(JSONB(none_as_null=True), nullable=True)
    composites = Column(JSONB(none_as_null=True), nullable=True)

    sources = Column(JSONB(none_as_null=True), nullable=True)
    measurements = Column(JSONB(none_as_null=True), nullable=True)
    measurements_missing = Column(JSONB(none_as_null=True), nullable=True)
    measurements_proxied = Column(JSONB(none_as_null=True), nullable=True)
    nodes_absent = Column(JSONB(none_as_null=True), nullable=True)

    __table_args__ = (
        # Settlement's access pattern: watts over a time window, newest first.
        Index("idx_orion_biometrics_cluster_observed_at", observed_at.desc()),
        # Retention's access pattern: rows older than a cutoff.
        Index("idx_orion_biometrics_cluster_created_at", created_at.desc()),
    )
