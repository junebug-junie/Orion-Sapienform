"""Pure parts of run_clue_rates_eval.py."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_clue_rates_eval as ev  # noqa: E402


def test_tally_counts_a_crop_once_per_clue_and_named_objects_as_their_own_clue() -> None:
    rows = [("A", "c1", "dog", "Biscuit"), ("A", "c1", "dog", None), ("A", "c2", None, None),
            (ev.STREET, "s1", "stroller", None), (ev.STREET, "s2", None, None)]
    n, hits = ev.tally(rows)
    assert n == {"A": 2, ev.STREET: 2}
    assert hits == {"A": {"dog": 1, "with:Biscuit": 1}, ev.STREET: {"stroller": 1}}


def test_weight_of_evidence_is_negative_when_the_clue_is_rarer_for_the_person() -> None:
    assert ev.weight_of_evidence(0, 435, 41, 517) < -2.0          # a stroller is evidence it is not them
    assert math.isclose(ev.weight_of_evidence(5, 99, 5, 99), 0.0)
    assert ev.weight_of_evidence(30, 100, 5, 100) > 1.5


def test_report_skips_thin_people_and_prints_the_street_first() -> None:
    n, hits = ev.tally([("A", f"a{i}", None, None) for i in range(25)] + [("B", "b1", "dog", None),
                       (ev.STREET, "s1", "dog", None)])
    lines = ev.report(n, hits, min_crops=20)
    assert lines[0].startswith("street: 1 crops") and lines[1].startswith("A: dog 0/25")
    assert not any(line.startswith("B:") for line in lines)
