"""scripts/wait_for_chat_turns_idle.py: the bounded wait before Hub/governor restarts."""

from __future__ import annotations

import importlib.util
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("wait_for_chat_turns_idle", ROOT / "scripts" / "wait_for_chat_turns_idle.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _snap(**lanes):
    return {"lanes": {name: {"running": v.get("running", []), "queued": v.get("queued", []),
                             "recent": v.get("recent", [])} for name, v in lanes.items()}}


def test_counts_chat_turns_on_any_lane_running_or_queued():
    snap = _snap(
        chat={"running": [{"source": "chat"}], "recent": [{"source": "chat"}]},
        agent={"queued": [{"source": "chat"}], "running": [{"source": "curiosity_self_inquiry"}]},
        unknown={"running": [{"source": "unknown"}]},
    )
    assert len(mod.chat_turns_in_flight(snap)) == 2


def test_durable_and_finished_turns_are_not_waited_for():
    snap = _snap(agent={"running": [{"source": "world_pulse_read_stage2"}, {"source": "consult_orion_turn"}]},
                 chat={"recent": [{"source": "chat"}]})
    assert mod.chat_turns_in_flight(snap) == []
    assert mod.chat_turns_in_flight({}) == []


class _Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_idle_at_first_look():
    c = _Clock()
    assert mod.wait_idle(lambda: _snap(), wait_s=600, clock=c.now, sleep=c.sleep) == "idle"
    assert c.t == 0.0


def test_waits_until_the_chat_turn_finishes():
    c = _Clock()
    busy = _snap(chat={"running": [{"source": "chat"}]})
    answers = iter([busy, busy, busy, _snap()])
    out = mod.wait_idle(lambda: next(answers), wait_s=600, poll_s=5, clock=c.now, sleep=c.sleep)
    assert out == "drained waited_s=15"


def test_deadline_lets_the_rebuild_proceed():
    c = _Clock()
    busy = _snap(chat={"running": [{"source": "chat"}], "queued": [{"source": "chat"}]})
    out = mod.wait_idle(lambda: busy, wait_s=12, poll_s=5, clock=c.now, sleep=c.sleep)
    assert out == "timeout in_flight=2 waited_s=12"


def test_hub_down_or_tracker_off_never_blocks():
    def refused():
        raise ConnectionRefusedError()

    def disabled():
        raise urllib.error.HTTPError("u", 503, "off", None, None)

    assert mod.wait_idle(refused, wait_s=600) == "unknown reason=ConnectionRefusedError"
    assert mod.wait_idle(disabled, wait_s=600) == "unknown reason=http_503"


def test_hub_url_prefers_override_then_hub_env_port(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=1\nHUB_PORT=8181\n")
    assert mod.hub_url({"ORION_HUB_URL": "http://h:9/"}, env_file) == "http://h:9"
    assert mod.hub_url({}, env_file) == "http://127.0.0.1:8181"
    assert mod.hub_url({}, tmp_path / "missing") == "http://127.0.0.1:8080"
