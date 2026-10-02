"""The offline distill eval scores whatever the validator returns (review 2026-10-02: a stale
field reference crashed 10 of 12 live runs after the model had already answered)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_episode_distill_eval as ev  # noqa: E402
from orion.memory.episode.tests.test_validate import TURNS, _mem, _run  # noqa: E402


def test_score_runs_on_a_real_validation_result():
    result = _run(_mem(), _mem(purpose="about_juniper", statement="Juniper told me she will be busy with travel."))
    s = ev.score(result, TURNS, {"usage": {"prompt_tokens": 1, "completion_tokens": 2}, "latency_ms": 3}, 2)
    assert s["kept"] == len(result.memories) and s["about_juniper"] == 1
    assert ev.austin_checks(result, TURNS)["no_kept_memory_rests_only_on_command_turns"] is True
