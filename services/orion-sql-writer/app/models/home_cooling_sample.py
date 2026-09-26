from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, Integer, JSON, String

from app.db import Base


class HomeCoolingSampleSQL(Base):
    """Read-only cabinet cooling samples from ``orion:home:cooling:sample``.

    Measurements are flattened for history queries; the full validated payload
    is preserved in ``payload_json``.
    """

    __tablename__ = "home_cooling_sample"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ts = Column(DateTime(timezone=True), nullable=False, index=True)
    node = Column(String, nullable=False)
    role = Column(String, nullable=False)
    cooling_watts = Column(Float, nullable=True)
    cooling_volts = Column(Float, nullable=True)
    cooling_amps = Column(Float, nullable=True)
    switch_on = Column(Boolean, nullable=True)
    zwave_node_id = Column(Integer, nullable=False)
    controller_ready = Column(Boolean, nullable=False)
    device_online = Column(Boolean, nullable=False)
    payload_json = Column(JSON, nullable=True)
