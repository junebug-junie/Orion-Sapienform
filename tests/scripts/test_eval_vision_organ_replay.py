"""scripts/eval_vision_organ_replay.py: a router log replays to calm AND non-calm.

Fixture shaped like the real router stdout: cam0 dispatching every 5 s, then a
70 s gap (a real outage shape), then recovery. carbon is configured and silent.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location("eval_vision_organ_replay", REPO / "scripts" / "eval_vision_organ_replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _line(t: datetime, text: str) -> str:
    return f"{t.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} | INFO     | app.x:y:1 - {text}\n"


def _log() -> list[str]:
    t0 = datetime(2026, 9, 24, 3, 49, 37, 538000)
    lines = [_line(t0, "[FRAME-ROUTER] Started → orion:vision:frames")]
    t = t0
    for _ in range(36):  # 3 healthy minutes
        t += timedelta(seconds=5)
        lines.append(_line(t, "[ROUTER] dispatch tier=baseline task_type=retina_fast want_caption=False camera_id=cam0 stream_id=cam0"))
    t += timedelta(seconds=70)  # outage
    for _ in range(36):
        t += timedelta(seconds=5)
        lines.append(_line(t, "[ROUTER] dispatch tier=baseline task_type=retina_fast want_caption=False camera_id=cam0 stream_id=cam0"))
    return lines


def test_router_log_replay_reads_calm_and_non_calm() -> None:
    mod = _load()
    replay = mod.replay_router_log(_log(), ["cam0", "carbon"])
    summary = mod.summarize(replay)
    assert summary["organ_staleness"]["calm"] >= 4
    assert summary["organ_staleness"]["non_calm"] >= 1
    assert summary["max_organ_staleness"] > 0.5
    assert summary["per_stream_staleness"]["carbon"] == {"non_calm:never_seen": summary["windows"]}
    assert summary["organ_failure"] == {"calm": summary["windows"]}


def test_configured_streams_come_from_the_policy_file() -> None:
    mod = _load()
    streams = mod.configured_streams(REPO / "config" / "vision_frame_router.yaml")
    assert {"cam0", "carbon"} <= set(streams)
