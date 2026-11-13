"""The before/after chat replay script (its route half was retired 2026-11)."""

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


def test_route_replay_is_retired() -> None:
    # route_prediction_error() was retired 2026-11 (PE suite rebuild step 5).
    assert not hasattr(replay, "replay_route")


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
    assert len(out["v1"]) == len(out["v2"]) == len(out["v3"]) == 5
    # raw deltas differ because v1 counted repair twice
    assert out["v1:raw"] != out["v2:raw"]
    # v2 divides the new turn's change by every stored turn; v3 does not.
    assert out["v3:raw"][-1] > out["v2:raw"][-1] * 4
