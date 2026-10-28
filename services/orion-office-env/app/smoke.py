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
    open_port: Optional[Callable[..., Any]] = None,
    clock: Callable[[], float] = time.monotonic,
) -> int:
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
    deadline = clock() + seconds
    try:
        while clock() < deadline:
            raw = ser.readline()
            if not raw:
                continue
            frame = parse_line(raw)
            if frame is None:
                rejected += 1
            else:
                ok += 1
                last = frame
    finally:
        ser.close()
    print(f"frames_ok={ok} lines_rejected={rejected}")
    if last is None:
        print("FAIL no valid orion.office_env_frame.v1 frame seen")
        return 1
    env = last.environment
    if env is None or (env.temp_c is None and env.humidity_pct is None):
        print(f"WARN frames arrive (seq={last.seq}) but carry no environment: SHT40 not reading")
        return 3
    print(f"PASS seq={last.seq} temp_c={env.temp_c} humidity_pct={env.humidity_pct}")
    return 0


def main() -> None:
    s = get_settings()
    sys.exit(run_smoke(port=s.OFFICE_SERIAL_PORT, baud=s.OFFICE_SERIAL_BAUD, seconds=10.0))


if __name__ == "__main__":
    main()
