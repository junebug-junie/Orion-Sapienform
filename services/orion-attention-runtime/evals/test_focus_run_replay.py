from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_focus_runs import replay


def test_replay_mixed_day_with_returns_idle_restart_and_long_focus():
    sequence = ["A"] * 3 + [None] * 4 + ["A"] + ["B"] * 2000 + ["A"] * 2
    start = datetime(2026, 10, 9, tzinfo=timezone.utc)
    rows = [dict(observed_at=(start + timedelta(seconds=i * 2)).isoformat(), target_id=target,
                 streak_count=1, min_streak_at_tick=3, source_field_tick_id=f"tick-{i}",
                 source_attention_frame_id=f"frame-{i}") for i, target in enumerate(sequence)]
    result = replay(rows)
    assert result["completed_runs"] == 3
    assert result["max_ticks"] == 2000
    assert result["no_winner_ticks"] == 4
    assert result["open_tail_excluded"]
