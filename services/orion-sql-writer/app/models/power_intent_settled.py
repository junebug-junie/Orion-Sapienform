from sqlalchemy import Column, DateTime, Float, Index, Integer, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from app.db import Base
from uuid import uuid4


class PowerIntentSettledSQL(Base):
    """Settled power intents from ``orion:power:intent:settled``.

    A workload declared what it was about to draw; the meter said what actually
    happened. This is where that pair accumulates, and it is the only place the
    residual distribution can be read from -- which is the whole point of stage 2. See
    docs/superpowers/specs/2026-08-28-consequential-action-space-and-power-budget-design.md.

    NULLABILITY IS THE CONTRACT HERE, NOT LAZINESS. ``expected_watts`` is null when the
    workload had no measured expectation yet, ``residual_watts`` is null whenever either
    side is unknown, and every ``actual_*`` is null unless ``outcome == 'settled'``. A
    zero in any of those columns is a MEASUREMENT, and writing one where we mean "not
    known" is how an unmeasured workload comes to look perfectly predicted. Filter on
    ``outcome``, never on ``actual_peak_watts IS NOT NULL`` alone.

    ``sample_count`` and ``achieved_sample_hz`` are persisted so a reader can tell
    measurement from arithmetic. The standing GPU telemetry samples every ~31s and
    caught 4 of 332 real diffusion jobs (measured 2026-08-28), which is why the intent
    triggers its own fast window -- and why the rate that was actually achieved has to
    travel with the row rather than being assumed from config.
    """

    __tablename__ = "power_intent_settled"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    correlation_id = Column(String, nullable=True)

    # Occurrence vs write time, same split and same reason as orion_biometrics_cluster.
    settled_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    intent_id = Column(String, nullable=False)
    workload_kind = Column(String, nullable=False)
    node = Column(String, nullable=False)
    gpu_index = Column(Integer, nullable=True)

    outcome = Column(String, nullable=False)

    window_start = Column(DateTime(timezone=True), nullable=True)
    window_end = Column(DateTime(timezone=True), nullable=True)
    sample_count = Column(Integer, nullable=True)
    achieved_sample_hz = Column(Float, nullable=True)

    actual_peak_watts = Column(Float, nullable=True)
    actual_mean_watts = Column(Float, nullable=True)
    energy_joules = Column(Float, nullable=True)
    baseline_watts = Column(Float, nullable=True)

    expected_watts = Column(Float, nullable=True)
    residual_watts = Column(Float, nullable=True)

    extra = Column(JSONB(none_as_null=True), nullable=True)

    __table_args__ = (
        Index("idx_power_intent_settled_settled_at", settled_at.desc()),
        Index("idx_power_intent_settled_created_at", created_at.desc()),
        Index("idx_power_intent_settled_workload", workload_kind, settled_at.desc()),
        # One settlement per intent. A redelivered envelope must not add a second row --
        # this table is read as a distribution, and a duplicated settlement silently
        # reweights it. Also activates _write_row's existing 23505 duplicate handling.
        UniqueConstraint("intent_id", name="uq_power_intent_settled_intent_id"),
    )
