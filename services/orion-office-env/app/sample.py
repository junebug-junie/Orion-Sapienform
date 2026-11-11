from __future__ import annotations

"""Frame -> bus sample. Pure: no I/O, no clock reads, so freshness is testable."""

from datetime import datetime
from typing import Optional

from orion.schemas.telemetry.closet_env import (
    ClosetEnvMeasurementsV1,
    ClosetEnvProvenanceV1,
    ClosetEnvSampleV1,
    ClosetEnvSensorV1,
    ClosetEnvStateV1,
)
from orion.schemas.telemetry.office_env import (
    OfficeEnvFrameAirV1,
    OfficeEnvFrameLightV1,
    OfficeEnvFrameParticulateV1,
    OfficeEnvFrameSoundV1,
    OfficeEnvFrameV1,
    OfficeEnvMeasurementsV1,
    OfficeEnvProvenanceV1,
    OfficeEnvSampleV1,
    OfficeEnvSensorV1,
    OfficeEnvStateV1,
)

# CCS811 datasheet: readings are not valid until 20 minutes after each sensor start.
CCS811_CONDITIONING_MS = 20 * 60 * 1000
# CCS811 datasheet output ranges; anything outside is not a reading.
ECO2_RANGE_PPM = (400, 29206)
TVOC_RANGE_PPB = (0, 32768)
# PMSA003I datasheet maximum range.
PM_RANGE_UG_M3 = (0, 1000)
# TSL2591 datasheet range (upper end; the lower end is 188 micro-lux).
LUX_MAX = 88000.0
# DFRobot SEN0232: output 0.6..2.6 V maps to 30..130 dBA, dBA = volts x 50 (+-1.5 dB).
SOUND_VOLTS_TO_DBA = 50.0
SOUND_RANGE_MV = (600, 2600)


def _in_range(value: Optional[int], bounds: tuple[int, int]) -> Optional[int]:
    if value is None or not bounds[0] <= value <= bounds[1]:
        return None
    return value


def _air_readings(air: Optional[OfficeEnvFrameAirV1]) -> tuple[Optional[int], Optional[int], bool]:
    """(eco2, tvoc, warming). Warming readings are withheld, not passed on as data.

    A frame without ``since_init_ms`` cannot prove the sensor is past conditioning,
    so it counts as warming too.
    """
    if air is None:
        return None, None, False
    if air.since_init_ms is None or air.since_init_ms < CCS811_CONDITIONING_MS:
        return None, None, True
    return _in_range(air.eco2_ppm, ECO2_RANGE_PPM), _in_range(air.tvoc_ppb, TVOC_RANGE_PPB), False


def _pm_readings(part: Optional[OfficeEnvFrameParticulateV1]) -> tuple[Optional[int], ...]:
    if part is None:
        return None, None, None
    return tuple(_in_range(v, PM_RANGE_UG_M3) for v in (part.pm1_ug_m3, part.pm25_ug_m3, part.pm10_ug_m3))


def _light_reading(light: Optional[OfficeEnvFrameLightV1]) -> tuple[Optional[float], bool]:
    """(lux, saturated). Saturated means too bright to measure: lux absent, not zero."""
    if light is None:
        return None, False
    if light.saturated:
        return None, True
    lux = light.lux
    if lux is None or not 0.0 <= lux <= LUX_MAX:
        return None, False
    return lux, False


def _sound_reading(sound: Optional[OfficeEnvFrameSoundV1]) -> Optional[float]:
    """dBA from the meter's mean pin voltage. Outside its output range it is not a reading."""
    mv = sound.level_mv if sound is not None else None
    if mv is None or not SOUND_RANGE_MV[0] <= mv <= SOUND_RANGE_MV[1]:
        return None
    return mv / 1000.0 * SOUND_VOLTS_TO_DBA


def _age(received_at: Optional[datetime], now: datetime) -> Optional[float]:
    return max(0.0, (now - received_at).total_seconds()) if received_at is not None else None


def build_sample(
    *,
    frame: Optional[OfficeEnvFrameV1],
    received_at: Optional[datetime],
    now: datetime,
    stale_after_sec: float,
    node: str,
    sensor_id: str,
) -> OfficeEnvSampleV1:
    """A stale or empty frame yields NO measurements -- never a repeated or zero value.

    "Stale" covers: no frame ever, last frame older than ``stale_after_sec``, and a
    fresh frame carrying no usable reading from any office sensor (SHT40, CCS811,
    PMSA003I, TSL2591, sound meter). Each sensor's readings are otherwise independent: one failing
    does not drop the others.
    """
    age = _age(received_at, now)
    env = frame.environment if frame is not None else None
    temp = env.temp_c if env is not None else None
    humidity = env.humidity_pct if env is not None else None
    air = frame.air if frame is not None else None
    eco2, tvoc, warming = _air_readings(air)
    pm1, pm25, pm10 = _pm_readings(frame.particulate if frame is not None else None)
    lux, saturated = _light_reading(frame.light if frame is not None else None)
    sound_dba = _sound_reading(frame.sound if frame is not None else None)
    stale = (
        frame is None
        or age is None
        or age > stale_after_sec
        or all(v is None for v in (temp, humidity, eco2, tvoc, pm1, pm25, pm10, lux, sound_dba))
    )
    if stale:
        measurements = OfficeEnvMeasurementsV1()
    else:
        measurements = OfficeEnvMeasurementsV1(
            office_temp_c=temp,
            office_humidity_pct=humidity,
            office_eco2_ppm=eco2,
            office_tvoc_ppb=tvoc,
            office_pm1_ug_m3=pm1,
            office_pm25_ug_m3=pm25,
            office_pm10_ug_m3=pm10,
            office_lux=lux,
            office_sound_dba=sound_dba,
        )
    return OfficeEnvSampleV1(
        ts=now,
        node=node,
        sensor=OfficeEnvSensorV1(id=sensor_id, online=not stale),
        measurements=measurements,
        # A stale frame says nothing current about the sensor, warming or not.
        state=OfficeEnvStateV1(
            stale=stale,
            air_warming=warming and not stale,
            light_saturated=True if saturated and not stale else None,
        ),
        provenance=OfficeEnvProvenanceV1(
            frame_seq=frame.seq if frame is not None else None,
            firmware_uptime_ms=frame.uptime_ms if frame is not None else None,
            sample_age_sec=age,
            air_since_init_ms=air.since_init_ms if air is not None else None,
        ),
    )


def build_closet_sample(
    *,
    frame: Optional[OfficeEnvFrameV1],
    received_at: Optional[datetime],
    now: datetime,
    stale_after_sec: float,
    node: str,
    sensor_id: str,
) -> ClosetEnvSampleV1:
    """Closet Nano frame -> closet sample. Only the SCD40 ``co2`` object is used.

    Stale (no measurements) when: no frame ever, last frame older than
    ``stale_after_sec``, or a fresh frame with no SCD40 reading.
    """
    age = _age(received_at, now)
    co2 = frame.co2 if frame is not None else None
    values = (co2.co2_ppm, co2.temp_c, co2.humidity_pct) if co2 is not None else (None, None, None)
    stale = frame is None or age is None or age > stale_after_sec or all(v is None for v in values)
    measurements = (
        ClosetEnvMeasurementsV1()
        if stale
        else ClosetEnvMeasurementsV1(
            closet_co2_ppm=values[0], closet_temp_c=values[1], closet_humidity_pct=values[2]
        )
    )
    return ClosetEnvSampleV1(
        ts=now,
        node=node,
        sensor=ClosetEnvSensorV1(id=sensor_id, online=not stale),
        measurements=measurements,
        state=ClosetEnvStateV1(stale=stale),
        provenance=ClosetEnvProvenanceV1(
            frame_seq=frame.seq if frame is not None else None,
            firmware_uptime_ms=frame.uptime_ms if frame is not None else None,
            sample_age_sec=age,
        ),
    )
