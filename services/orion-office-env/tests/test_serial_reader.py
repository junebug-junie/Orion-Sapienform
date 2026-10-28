from __future__ import annotations

import threading
from datetime import datetime, timezone

from app.serial_reader import FrameStore, parse_line, run_reader

GOOD = b'{"schema":"orion.office_env_frame.v1","seq":1,"uptime_ms":1000,"environment":{"temp_c":21.5,"humidity_pct":30.2}}\n'
NOW = datetime(2026, 10, 27, 20, 0, 0, tzinfo=timezone.utc)


def test_parse_line_accepts_valid_frame() -> None:
    f = parse_line(GOOD)
    assert f is not None
    assert f.seq == 1
    assert f.environment.temp_c == 21.5


def test_parse_line_rejects_junk() -> None:
    for bad in (b"", b"boot: SHT40 ok\n", b"{not json\n", b'{"schema":"other.v1","seq":1,"uptime_ms":1}\n'):
        assert parse_line(bad) is None


def test_parse_line_rejects_out_of_range_reading() -> None:
    line = b'{"schema":"orion.office_env_frame.v1","seq":1,"uptime_ms":1,"environment":{"temp_c":999,"humidity_pct":30}}\n'
    assert parse_line(line) is None


class FakeSerial:
    def __init__(self, lines: list[bytes], stop: threading.Event) -> None:
        self._lines = list(lines)
        self._stop = stop
        self.closed = False

    def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        self._stop.set()
        return b""

    def close(self) -> None:
        self.closed = True


def test_reader_keeps_last_good_frame_across_malformed_lines() -> None:
    stop = threading.Event()
    store = FrameStore()
    ser = FakeSerial([GOOD, b"garbage\n", b"{broken\n"], stop)
    run_reader(
        store, port="/dev/x", baud=115200, stop=stop,
        open_port=lambda *a, **k: ser, now_fn=lambda: NOW, sleep_fn=lambda s: None,
    )
    frame, received = store.latest()
    assert frame is not None and frame.seq == 1
    assert received == NOW
    assert store.frames_ok == 1
    assert store.lines_rejected == 2
    assert ser.closed is True
    assert store.serial_open is False


def test_reader_retries_when_port_missing_then_stops() -> None:
    stop = threading.Event()
    store = FrameStore()
    attempts: list[int] = []

    def opener(*a, **k):
        attempts.append(1)
        raise OSError("no such device")

    def sleeper(_s: float) -> None:
        if len(attempts) >= 3:
            stop.set()

    run_reader(store, port="/dev/x", baud=115200, stop=stop, open_port=opener, sleep_fn=sleeper)
    assert len(attempts) == 3
    assert store.latest() == (None, None)
    assert store.serial_open is False
