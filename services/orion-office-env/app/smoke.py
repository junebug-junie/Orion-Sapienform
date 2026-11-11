from __future__ import annotations

"""Serial smoke: read the Nano for a few seconds and report whether real frames arrive.

    python -m app.smoke            (inside the container; exit 0 only if >=1 valid frame)
"""

import sys
import time
from typing import Any, Callable, Optional

from .serial_reader import parse_line
from .settings import get_settings


def run_smoke(
    *,
    port: str,
    baud: int,
    seconds: float,
    room: str = "office",
    open_port: Optional[Callable[..., Any]] = None,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Exit codes: 0 PASS, 1 no valid frame, 2 port, then one per missing sensor:
    office 3 SHT40, 4 CCS811, 5 PMSA003I, 6 TSL2591; closet 7 SCD40.

    A sensor counts as reading if ANY frame in the window carried it. The PMSA003I is
    sent only 30 s after it starts, so run the smoke at least 40 s after a Nano reset.
    """
    if open_port is None:
        import serial

        open_port = serial.Serial
    try:
        ser = open_port(port, baudrate=baud, timeout=1.0)
    except Exception as exc:
        print(f"FAIL cannot open {port}: {type(exc).__name__}: {exc}")
        return 2
    ok = rejected = 0
    last = None
    seen: dict[str, Any] = {}
    deadline = clock() + seconds
    try:
        while clock() < deadline:
            raw = ser.readline()
            if not raw:
                continue
            frame = parse_line(raw)
            if frame is None:
                rejected += 1
                continue
            ok += 1
            last = frame
            env = frame.environment
            if env is not None and (env.temp_c is not None or env.humidity_pct is not None):
                seen["sht40"] = env
            if frame.air is not None and frame.air.eco2_ppm is not None:
                seen["ccs811"] = frame.air
            if frame.particulate is not None and frame.particulate.pm25_ug_m3 is not None:
                seen["pmsa003i"] = frame.particulate
            if frame.light is not None and (frame.light.lux is not None or frame.light.saturated):
                seen["tsl2591"] = frame.light
            if frame.co2 is not None and frame.co2.co2_ppm is not None:
                seen["scd40"] = frame.co2
    finally:
        ser.close()
    print(f"frames_ok={ok} lines_rejected={rejected} room={room}")
    if last is None:
        print("FAIL no valid orion.office_env_frame.v1 frame seen")
        return 1
    expected = (("scd40", 7),) if room == "closet" else (
        ("sht40", 3), ("ccs811", 4), ("pmsa003i", 5), ("tsl2591", 6)
    )
    for name, code in expected:
        if name not in seen:
            print(f"WARN frames arrive (seq={last.seq}) but no {name.upper()} reading")
            return code
    print("PASS seq=%s %s" % (last.seq, " ".join(
        f"{k}={v.model_dump(exclude_none=True)}" for k, v in seen.items()
    )))
    return 0


def main() -> None:
    s = get_settings()
    sys.exit(
        run_smoke(port=s.OFFICE_SERIAL_PORT, baud=s.OFFICE_SERIAL_BAUD, seconds=10.0, room=s.ENV_ROOM)
    )


if __name__ == "__main__":
    main()
