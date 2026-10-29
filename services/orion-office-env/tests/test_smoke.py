from __future__ import annotations

from app.smoke import run_smoke

OK = b'{"schema":"orion.office_env_frame.v1","seq":9,"uptime_ms":1,"environment":{"temp_c":20.0,"humidity_pct":40.0}}\n'
NO_ENV = b'{"schema":"orion.office_env_frame.v1","seq":9,"uptime_ms":1}\n'


class Ser:
    def __init__(self, lines):
        self.lines = list(lines)

    def readline(self):
        return self.lines.pop(0) if self.lines else b""

    def close(self):
        pass


def _clock():
    t = [0.0]

    def tick():
        t[0] += 1.0
        return t[0]

    return tick


def test_pass_with_real_reading() -> None:
    assert run_smoke(port="p", baud=1, seconds=5, open_port=lambda *a, **k: Ser([OK]), clock=_clock()) == 0


def test_warn_when_frames_have_no_environment() -> None:
    assert run_smoke(port="p", baud=1, seconds=5, open_port=lambda *a, **k: Ser([NO_ENV]), clock=_clock()) == 3


def test_fail_on_silence_and_on_junk() -> None:
    assert run_smoke(port="p", baud=1, seconds=5, open_port=lambda *a, **k: Ser([]), clock=_clock()) == 1
    assert run_smoke(port="p", baud=1, seconds=5, open_port=lambda *a, **k: Ser([b"junk\n"]), clock=_clock()) == 1


def test_fail_when_port_missing() -> None:
    def boom(*a, **k):
        raise OSError("nope")

    assert run_smoke(port="p", baud=1, seconds=5, open_port=boom, clock=_clock()) == 2
