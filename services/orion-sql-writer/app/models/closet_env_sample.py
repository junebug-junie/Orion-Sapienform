from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, Integer, JSON, String

from app.db import Base


class ClosetEnvSampleSQL(Base):
    """Server-closet CO2/temp/humidity from ``orion:closet:env:sample`` (pan-1 SCD40).

    Measurements are flattened for history queries; a stale sample has NULL
    readings (never 0). The full validated payload is kept in ``payload_json``.
    """

    __tablename__ = "closet_env_sample"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ts = Column(DateTime(timezone=True), nullable=False, index=True)
    node = Column(String, nullable=False)
    role = Column(String, nullable=False)
    sensor_id = Column(String, nullable=False)
    sensor_online = Column(Boolean, nullable=False)
    closet_co2_ppm = Column(Integer, nullable=True)
    closet_temp_c = Column(Float, nullable=True)
    closet_humidity_pct = Column(Float, nullable=True)
    stale = Column(Boolean, nullable=False)
    sample_age_sec = Column(Float, nullable=True)
    frame_seq = Column(Integer, nullable=True)
    payload_json = Column(JSON, nullable=True)
