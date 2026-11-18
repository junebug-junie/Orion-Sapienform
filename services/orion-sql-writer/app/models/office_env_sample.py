from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, Integer, JSON, String

from app.db import Base


class OfficeEnvSampleSQL(Base):
    """Office env from ``orion:office:env:sample`` (pan-1 SHT40 + CCS811 + PMSA003I + TSL2591).

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
    # CCS811. NULL while warming (first 20 min after a sensor start) or when not reading.
    office_eco2_ppm = Column(Integer, nullable=True)
    office_tvoc_ppb = Column(Integer, nullable=True)
    air_warming = Column(Boolean, nullable=True)
    # PMSA003I CF=1 mass concentrations (ug/m3). NULL when not reading.
    office_pm1_ug_m3 = Column(Integer, nullable=True)
    office_pm25_ug_m3 = Column(Integer, nullable=True)
    office_pm10_ug_m3 = Column(Integer, nullable=True)
    # TSL2591 at the windowsill. NULL when not reading OR saturated (light_saturated=true).
    office_lux = Column(Float, nullable=True)
    light_saturated = Column(Boolean, nullable=True)
    stale = Column(Boolean, nullable=False)
    sample_age_sec = Column(Float, nullable=True)
    frame_seq = Column(Integer, nullable=True)
    payload_json = Column(JSON, nullable=True)
