"""Deterministic tests for measure_candidate_b_novelty_alternation.py.

No DB. Old-formula frames are synthesized by hand; new-formula frames come
from the real, fixed `build_attention_frame()`, so the fingerprint test also
proves what the live builder now writes.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_MODULE_PATH = Path(__file__).resolve().parents[1] / "measure_candidate_b_novelty_alternation.py"
_spec = importlib.util.spec_from_file_location("measure_candidate_b_novelty_alternation", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["measure_candidate_b_novelty_alternation"] = mod
_spec.loader.exec_module(mod)

from orion.attention.field_attention.builder import build_attention_frame  # noqa: E402
from orion.attention.field_attention.policy import load_attention_policy  # noqa: E402
from orion.schemas.field_state import FieldStateV1  # noqa: E402

POLICY = load_attention_policy(REPO / "config" / "attention" / "field_attention_policy.v1.yaml")
NOW = datetime(2026, 9, 25, tzinfo=timezone.utc)


def _target(target_id: str, pressure: float, novelty: float, kind: str = "capability") -> dict:
    return {
        "target_id": target_id,
        "target_kind": kind,
        "salience_score": novelty,
        "pressure_score": pressure,
        "novelty_score": novelty,
    }


def _old_formula_frames(pressure: float, ticks: int) -> list[dict]:
    """What the pre-2026-09-25 code wrote for a steady input: novelty diffed
    against the previous frame's salience (= previous novelty)."""
    frames, prev_novelty = [], None
    for _ in range(ticks):
        novelty = 0.0 if prev_novelty is None else abs(pressure - prev_novelty)
        frames.append({"capability_targets": [_target("capability:vision", pressure, novelty)]})
        prev_novelty = novelty
    return frames


def test_old_formula_frames_show_half_steady_pairs_with_novelty_and_old_fingerprint() -> None:
    report = mod.analyze(_old_formula_frames(0.8, 9))
    assert report.steady_pairs == 8
    assert report.steady_pairs_with_novelty == 4
    assert report.steady_novelty_fraction == 0.5
    assert report.old_formula_pairs == 4
    # On the 0.0 ticks both formulas give 0, so those pairs cannot say which
    # formula wrote them.
    assert report.new_formula_pairs == 0
    assert report.ambiguous_pairs == 4
    assert report.per_target_steady_with_novelty == {"capability:vision": 4}


def test_fixed_builder_frames_show_no_steady_novelty() -> None:
    frames, prev = [], None
    for i in range(6):
        field = FieldStateV1(
            generated_at=NOW,
            tick_id=f"t{i}",
            capability_vectors={"capability:vision": {"execution_pressure": 0.8}},
        )
        prev = build_attention_frame(field=field, policy=POLICY, previous_frame=prev, now=NOW)
        frames.append(prev.model_dump(mode="json"))
    report = mod.analyze(frames)
    assert report.steady_pairs == 5
    assert report.steady_pairs_with_novelty == 0
    assert report.steady_novelty_fraction == 0.0
    assert report.old_formula_pairs == 0


def test_candidate_a_targets_are_ignored() -> None:
    a_target = _target("node:substrate.chat", 0.5, 0.5, kind="node")
    frames = [{"node_targets": [a_target]}, {"node_targets": [a_target]}]
    report = mod.analyze(frames)
    assert report.targets == 0
    assert report.consecutive_pairs == 0


def test_host_nodes_outside_candidate_a_count() -> None:
    frames = [
        {"node_targets": [_target("node:athena", 0.4, 0.0, kind="node")]},
        {"node_targets": [_target("node:athena", 0.4, 0.4, kind="node")]},
    ]
    report = mod.analyze(frames)
    assert report.steady_pairs == 1
    assert report.steady_pairs_with_novelty == 1


def test_one_frame_gap_is_the_over_cap_fingerprint() -> None:
    present = {"capability_targets": [_target("capability:c6", 0.9, 0.0)]}
    frames = [present, {"capability_targets": []}, present]
    report = mod.analyze(frames)
    assert report.one_frame_gaps == 1
    assert report.consecutive_pairs == 0


def test_first_bucket_wins_and_malformed_targets_are_skipped() -> None:
    frame = {
        "dominant_targets": [_target("capability:graph", 0.3, 0.1)],
        "suppressed_targets": [
            _target("capability:graph", 0.9, 0.9),
            {"target_id": "capability:bad", "target_kind": "capability", "pressure_score": "x"},
            "not-a-dict",
        ],
    }
    readings = mod.candidate_b_readings(frame)
    assert readings == {"capability:graph": mod.TargetReading(pressure=0.3, novelty=0.1)}


def test_no_steady_pairs_reports_none_not_zero() -> None:
    report = mod.analyze([])
    assert report.steady_novelty_fraction is None
    assert "no steady pairs" in mod.render(report)
