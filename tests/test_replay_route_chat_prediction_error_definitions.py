"""The before/after replay script reproduces the v1 route dilution on a live-shaped
projection and shows v2 reading the same decision flip at full size."""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "replay_pe_defs",
    REPO / "scripts" / "analysis" / "replay_route_chat_prediction_error_definitions.py",
)
replay = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(replay)

T0 = datetime(2026, 9, 25, tzinfo=timezone.utc)


def _run(i: int, lane: str = "background", reason: str = "verb_background") -> dict:
    return {
        "trace_id": f"r{i}", "correlation_id": f"r{i}", "node_id": "athena",
        "lane": lane, "lane_reason": reason, "output_mode": "direct_answer",
        "mind_requested": False, "evidence_event_ids": [],
        "last_updated_at": (T0 + timedelta(seconds=30 * i)).isoformat(),
    }


def test_route_replay_v1_diluted_v2_full_size() -> None:
    runs = {f"r{i}": _run(i) for i in range(400)}
    runs["r400"] = _run(400, lane="chat", reason="mode_chat")
    out = replay.replay_route({"runs": runs})
    assert out["v2"][-1] == 0.5
    assert out["v1"][-1] < 0.002  # 0.5 * 2 / 401
    assert max(out["v2"][:-1]) == 0.0  # identical background decisions stay calm


def test_chat_replay_runs_and_v2_drops_topic_coherence() -> None:
    turns = {}
    for i in range(6):
        turns[f"t{i}"] = {
            "trace_id": f"hub.chat:athena:t{i}", "turn_id": f"t{i}", "session_id": "s",
            "node_id": "athena", "observed_at": T0.isoformat(), "word_count": 20 * i,
            "repair_pressure_level": 0.3 if i % 2 else 0.0,
            "last_updated_at": (T0 + timedelta(minutes=i)).isoformat(),
        }
    out = replay.replay_chat({"turns": turns})
    assert len(out["v1"]) == len(out["v2"]) == 5
    # raw deltas differ because v1 counted repair twice
    assert out["v1:raw"] != out["v2:raw"]
