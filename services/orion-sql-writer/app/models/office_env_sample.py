from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, Integer, JSON, String

from app.db import Base


class OfficeEnvSampleSQL(Base):
    """Office temperature/humidity from ``orion:office:env:sample`` (pan-1 SHT40).

    Measurements are flattened for history queries; a stale sample has NULL
    readings (never 0.0). The full validated payload is kept in ``payload_json``.
    """

    __tablename__ = "office_env_sample"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ts = Column(DateTime(timezone=True), nullable=False, index=True)
    node = Column(String, nullable=False)
    role = Column(String, nullable=False)
    sensor_id = Column(String, nullable=False)
    sensor_online = Column(Boolean, nullable=False)
    office_temp_c = Column(Float, nullable=True)
    office_humidity_pct = Column(Float, nullable=True)
    stale = Column(Boolean, nullable=False)
    sample_age_sec = Column(Float, nullable=True)
    frame_seq = Column(Integer, nullable=True)
    payload_json = Column(JSON, nullable=True)
