"""Street attention eval on a simulated week (run_street_attention_eval.py):
calibrated p-values pass ~5%, the cap binds, a quiet hour rests at 0.0, a ring
wins world-first and a calm 0.0 does not -- also when the model floods."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_street_attention_eval import synthetic, verdict  # noqa: E402


def test_calibrated_week_passes_every_check() -> None:
    r = synthetic()
    assert verdict(r) == []
    assert 0.03 <= r["graded_pass_rate"] <= 0.07
    assert r["quiet_hour_max_value"] == 0.0


def test_a_flooding_model_is_held_by_the_cap_and_a_ring_still_wins() -> None:
    r = synthetic(p_scale=0.04)  # every graded event clears p < 0.05
    assert r["worst_trailing_hour_graded_passes"] <= r["max_per_hour"]
    assert r["nonzero_tick_share"] <= 0.11
    assert r["ring_wins"] and not r["calm_wins"]
