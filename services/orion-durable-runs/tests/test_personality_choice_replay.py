"""evals/personality_choice_replay.py: the pure summary and its checks (no DB, no embedder)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
sys.path[:0] = [str(SERVICE_ROOT.parents[1]), str(SERVICE_ROOT), str(TESTS)]

from test_personality_loop import T0, choice, utc, vec_for  # noqa: E402

spec = importlib.util.spec_from_file_location("replay", SERVICE_ROOT / "evals/personality_choice_replay.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def test_summary_counts_departures_and_passes_checks_on_varied_choices():
    picks = ["people", "heat", "camera", "reading", "walkway", "budget"]
    choices = [choice(f"prime:t{i}", utc(minutes=-i), chosen=p, declined=(f"pipeline {i}", f"other {i}"), rank=1 + i % 3)
               for i, p in enumerate(picks)]
    vecs = {t: vec_for(t) for c in choices for t in [c.chosen.text, *[o.text for o in c.alternatives]]}
    out = replay.summarize(choices, vecs, T0)
    assert out["choices"] == 6 and out["rank1"] == 2 and out["departed"] == 4
    assert out["checks"] == {"has_choices": True, "has_departures": True, "directions_not_degenerate": True,
                             "labels_computed": True}
    assert all(label.endswith(" of 1") for _, _, label in out["top"])


def test_summary_fails_closed_on_no_choices_or_only_rank_one():
    assert replay.summarize([], {}, T0)["checks"]["has_choices"] is False
    only_top = [choice("prime:a", T0, rank=1)]
    vecs = {"people": vec_for("people"), "pipeline": vec_for("pipeline")}
    assert replay.summarize(only_top, vecs, T0)["checks"]["has_departures"] is False
