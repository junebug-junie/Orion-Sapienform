from sqlalchemy import Column, DateTime, Float, Index, Integer, String, func

from app.db import Base


class CabinetAmbientSpikeSQL(Base):
    """Sustained cabinet ambient audio activity spikes from ``orion:cabinet:ambient:spike``.

    One row per ``spike_id``. Redelivered envelopes must not duplicate rows --
    this table is read as an occurrence log for later correlation / STT hooks.
    """

    __tablename__ = "cabinet_ambient_spike"

    spike_id = Column(String, primary_key=True)
    correlation_id = Column(String, nullable=True)

    node = Column(String, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    activity = Column(Float, nullable=False)
    rms = Column(Float, nullable=False)
    peak = Column(Float, nullable=True)
    activity_threshold = Column(Float, nullable=False)
    consecutive_ticks = Column(Integer, nullable=False)

    source_service = Column(String, nullable=False)
    source_node = Column(String, nullable=True)

    __table_args__ = (
        Index("idx_cabinet_ambient_spike_timestamp", timestamp.desc()),
        Index("idx_cabinet_ambient_spike_node_ts", node, timestamp.desc()),
    )
