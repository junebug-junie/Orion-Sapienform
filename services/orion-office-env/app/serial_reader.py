from __future__ import annotations

"""USB-serial NDJSON reader for the office Nano. Blocking; run it in a thread.

A malformed line never replaces the last good frame. A dead port is retried
forever; while it is down the publisher sees an ageing frame and reports stale.
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from pydantic import ValidationError

from orion.schemas.telemetry.office_env import OfficeEnvFrameV1

logger = logging.getLogger("orion-office-env.serial")


def parse_line(line: bytes | str) -> Optional[OfficeEnvFrameV1]:
    """One NDJSON line -> frame, or None for anything that is not a valid v1 frame."""
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="replace")
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        return OfficeEnvFrameV1.model_validate(json.loads(line))
    except (ValueError, ValidationError):
        return None


class FrameStore:
    """Thread-safe latest-frame holder plus counters for the heartbeat."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[OfficeEnvFrameV1] = None
        self._received_at: Optional[datetime] = None
        self.serial_open = False
        self.frames_ok = 0
        self.lines_rejected = 0

    def put(self, frame: OfficeEnvFrameV1, at: datetime) -> None:
        with self._lock:
            self._frame, self._received_at = frame, at
            self.frames_ok += 1

    def reject(self) -> None:
        with self._lock:
            self.lines_rejected += 1

    def latest(self) -> tuple[Optional[OfficeEnvFrameV1], Optional[datetime]]:
        with self._lock:
            return self._frame, self._received_at


def run_reader(
    store: FrameStore,
    *,
    port: str,
    baud: int,
    stop: threading.Event,
    open_port: Optional[Callable[..., Any]] = None,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    reconnect_sec: float = 5.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    if open_port is None:
        import serial  # pyserial; imported lazily so tests need no hardware

        open_port = serial.Serial
    while not stop.is_set():
        try:
            ser = open_port(port, baudrate=baud, timeout=1.0)
        except Exception as exc:
            store.serial_open = False
            logger.warning("serial_open_failed port=%s error=%s: %s", port, type(exc).__name__, exc)
            sleep_fn(reconnect_sec)
            continue
        store.serial_open = True
        logger.info("serial_open port=%s baud=%s", port, baud)
        try:
            while not stop.is_set():
                raw = ser.readline()
                if not raw:
                    continue  # read timeout; loop so `stop` is honoured
                frame = parse_line(raw)
                if frame is None:
                    store.reject()
                    continue
                store.put(frame, now_fn())
        except Exception as exc:
            logger.warning("serial_read_failed port=%s error=%s: %s", port, type(exc).__name__, exc)
        finally:
            store.serial_open = False
            try:
                ser.close()
            except Exception:
                pass
        sleep_fn(reconnect_sec)
