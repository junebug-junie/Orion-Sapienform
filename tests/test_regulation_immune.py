"""Runtime immune sweep (Temporal Self rev 4, R4): the pure detector in orion/regulation/immune.py.

Each known zombie is reconstructed as a fixture series from its own code/history and must be
caught with the right kind; healthy and calm-at-rest series must not be flagged. Fixture glossary
semantics carry ``rest_range`` explicitly (the real YAML gains it in a parallel patch).
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

from orion.field.channel_glossary import FieldChannelGlossaryEntry, _freeze_semantics
from orion.regulation.immune import (
    ImmuneConfig,
    ImmuneSample,
    inspect_channel,
    rest_range_of,
    semantic_channel_names,
    semantic_channels,
    sweep,
)

T0 = datetime(2026, 10, 12, 6, 0, tzinfo=timezone.utc)
DAY_MIN = 24 * 60


def sem(kind="level", sparsity="per_tick", rest_range=(0.0, 0.03)):
    raw = {"value_kind": kind, "sparsity": sparsity, "rest": "prose", "absent_means": "prose"}
    if rest_range is not None:
        raw["rest_range"] = list(rest_range)
    # The exact frozen shape FieldChannelGlossaryEntry.semantics holds (lists -> tuples).
    return _freeze_semantics(raw)


def minutes(n=DAY_MIN, start=T0):
    return [start + timedelta(minutes=i) for i in range(n)]


# --- the three zombies -------------------------------------------------------------------------

def test_repair_pressure_max_replayed_into_curiosity_is_replayed():
    """worker._repair_appraisal_from_chat (before the 10-09 fix) took the all-time max over the
    chat projection: one 0.913 turn from 10-06 02:02 re-emitted, freshly, on every tick."""
    src = "chat_turn:2026-10-06T02:02:00Z"
    samples = [ImmuneSample(t, 0.913, updated_at=t, source=src) for t in minutes()]
    v = inspect_channel("curiosity:endogenous", "repair_pressure", samples,
                        sem(sparsity="event_gated", rest_range=(0.0, 0.3)))
    assert v.finding is not None and v.finding.kind == "replayed"
    ev = v.finding.evidence
    assert ev["restamps"] == DAY_MIN - 1 and ev["value"] == 0.913 and ev["source"] == src
    assert ev["source_evidence"] is True


def test_ontology_sparse_region_constant_one_is_constant():
    """frontier_curiosity: fired whenever the slice held concepts but no ontology_branch node
    (nothing creates those), so it read 1.0 on every decision -- recomputed each time from a
    different read slice, so not a replay of one source row: a constant."""
    samples = [ImmuneSample(t, 1.0, updated_at=t, source=f"slice:{i}") for i, t in enumerate(minutes())]
    v = inspect_channel("frontier:signals", "ontology_sparse_region", samples, sem(rest_range=(0.0, 0.5)))
    assert v.finding is not None and v.finding.kind == "constant"
    assert v.checks["replayed"] == "clear"
    assert v.finding.evidence["distinct"] == 1 and v.finding.evidence["examples"] == [1.0]


def test_spark_rollup_zombie_writer_is_replayed():
    """orion-state-journaler's rollup (retired 10-10): its only input producer died 07-28, and
    from then it wrote pct_missing=1.0 every 30 s from the same last snapshot."""
    src = "spark_snapshot:2026-07-28T07:09:00Z"
    samples = [ImmuneSample(t, 1.0, updated_at=t, source=src) for t in minutes()]
    v = inspect_channel("spark:rollup", "pct_missing", samples, sem(rest_range=(0.0, 0.2)))
    assert v.finding is not None and v.finding.kind == "replayed"
    assert v.finding.evidence["source_evidence"] is True


def test_route_staleness_decay_is_decayed_to_zero():
    """node:substrate.route, 2026-07-26: decay.py multiplied prediction_error by 0.92 every tick
    (~28 ticks/min, irregular) with no real event. Sampled once a minute, so each step is
    0.92**k with a varying k; updated_at never moved."""
    rng = random.Random(7)
    v, out = 0.5, []
    frozen = T0 - timedelta(hours=3)
    for t in minutes(300):
        out.append(ImmuneSample(t, v, updated_at=frozen))
        v *= 0.92 ** rng.randint(25, 32)
    verdict = inspect_channel("node:substrate.route", "prediction_error", out,
                              sem(kind="count", sparsity="designed_sparse", rest_range=(0.0, 0.0)))
    assert verdict.finding is not None and verdict.finding.kind == "decayed_to_zero"
    ev = verdict.finding.evidence
    assert math.isclose(ev["ratio"], 0.92, rel_tol=1e-6)
    assert ev["changes"] >= 5 and ev["updated_at_moved"] is False
    assert len(ev["examples"]) <= 5 and len(ev["powers"]) <= 5


# --- must not flag ---------------------------------------------------------------------------

def test_healthy_noisy_series_is_not_flagged():
    rng = random.Random(1)
    samples = [ImmuneSample(t, round(max(0.0, rng.gauss(0.03, 0.04)), 4), updated_at=t) for t in minutes()]
    v = inspect_channel("node:substrate.biometrics", "prediction_error", samples, sem())
    assert v.finding is None, v.checks
    assert set(v.checks.values()) == {"clear"}


def test_calm_series_at_its_declared_rest_is_not_flagged():
    samples = [ImmuneSample(t, 0.0, updated_at=t) for t in minutes()]
    v = inspect_channel("node:substrate.vision_organ", "vision_frame_staleness", samples, sem(rest_range=(0.0, 0.0)))
    assert v.finding is None and v.checks["constant"] == "clear" and v.checks["replayed"] == "clear"


def test_missing_rest_range_cannot_judge_rest_dependent_checks():
    samples = [ImmuneSample(t, 1.0, updated_at=t, source=f"s{i}") for i, t in enumerate(minutes())]
    v = inspect_channel("frontier:signals", "ontology_sparse_region", samples, sem(rest_range=None))
    assert v.finding is None
    for check in ("constant", "floor_locked", "replayed"):
        assert v.checks[check] == "cannot_judge:no rest_range"
    assert v.checks["decayed_to_zero"] == "clear"


def test_malformed_rest_range_is_treated_as_missing():
    assert rest_range_of({"rest_range": (0.5, 0.1)}) is None
    assert rest_range_of({"rest_range": ("a", 1)}) is None
    assert rest_range_of({"rest_range": (0, 1, 2)}) is None
    assert rest_range_of({"rest_range": (0, 0.03)}) == (0.0, 0.03)


def test_value_only_carried_forward_is_not_constant():
    """absent_means 'carried forward': a stale last value with updated_at frozen is no reading."""
    stamp = T0 - timedelta(hours=1)
    samples = [ImmuneSample(t, 1.0, updated_at=stamp) for t in minutes()]
    v = inspect_channel("node:substrate.chat", "prediction_error", samples,
                        sem(sparsity="designed_sparse", rest_range=(0.0, 0.0)))
    assert v.finding is None
    assert v.checks["constant"].startswith("clear:carried forward")


def test_floor_locked_per_tick_level_never_reaching_rest():
    rng = random.Random(3)
    samples = [ImmuneSample(t, round(rng.uniform(0.05, 0.2), 4), updated_at=t) for t in minutes()]
    v = inspect_channel("node:substrate.bus_synaptic", "prediction_error", samples, sem())
    assert v.finding is not None and v.finding.kind == "floor_locked"
    assert v.finding.evidence["min"] > 0.03


def test_floor_locked_not_applicable_to_designed_sparse():
    rng = random.Random(3)
    samples = [ImmuneSample(t, round(rng.uniform(0.05, 0.2), 4), updated_at=t) for t in minutes()]
    v = inspect_channel("node:substrate.execution", "prediction_error", samples, sem(sparsity="designed_sparse"))
    assert v.finding is None and v.checks["floor_locked"].startswith("not_applicable")


def test_saturated_value_without_source_is_not_a_replay():
    """1.0 re-stamped for an hour with no source evidence: a camera really silent, not a replay."""
    ts = minutes(180)
    samples = [ImmuneSample(t, 1.0 if 60 <= i < 120 else 0.0, updated_at=t) for i, t in enumerate(ts)]
    v = inspect_channel("node:substrate.vision_organ", "vision_frame_staleness", samples, sem(rest_range=(0.0, 0.0)))
    assert v.checks["replayed"] == "clear"


def test_non_saturated_partial_replay_without_source_is_caught():
    ts = minutes(180)
    samples = [ImmuneSample(t, 0.4321 if 60 <= i < 100 else round(0.01 * (i % 3), 4), updated_at=t)
               for i, t in enumerate(ts)]
    v = inspect_channel("node:substrate.biometrics", "prediction_error", samples, sem())
    assert v.finding is not None and v.finding.kind == "replayed"
    assert v.finding.evidence["restamps"] == 39
    assert v.finding.window_start == ts[60] and v.finding.window_end == ts[99]


def test_short_window_cannot_judge_constant():
    samples = [ImmuneSample(t, 1.0, updated_at=t, source=f"s{i}") for i, t in enumerate(minutes(120))]
    v = inspect_channel("frontier:signals", "ontology_sparse_region", samples, sem(rest_range=(0.0, 0.5)))
    assert v.finding is None and v.checks["constant"].startswith("cannot_judge:samples")


def test_placeholder_and_no_semantics_are_skipped():
    samples = [ImmuneSample(t, 0.65, updated_at=t) for t in minutes()]
    assert inspect_channel("node:x", "c", samples, sem(kind="placeholder")).skipped.startswith("placeholder")
    assert inspect_channel("node:x", "c", samples, ()).skipped == "no semantics"


def test_garbage_input_never_raises():
    junk = [None, "x", 3, (None, 1.0), ("2026-10-12T00:00:00Z", float("nan")), {"t": "bad", "value": 1},
            ("2026-10-12T00:00:00Z", True), ("2026-10-12T00:01:00Z", "0.5", "not-a-date")]
    v = inspect_channel("node:x", "c", junk, sem())
    assert v.skipped == "fewer than 2 usable samples"
    assert inspect_channel("node:x", "c", object(), sem()).skipped is not None
    assert inspect_channel("node:x", "c", junk, 42).skipped == "no semantics"
    res = sweep({("node:x", "c"): junk, ("bad",): [1]}, {("node:x", "c"): sem(), ("bad",): sem()})
    assert res.findings == []
    assert sweep(None, None).findings == []


def test_samples_as_tuples_and_iso_strings_are_accepted():
    rows = [((T0 + timedelta(minutes=i)).isoformat(), 1.0, (T0 + timedelta(minutes=i)).isoformat())
            for i in range(DAY_MIN)]
    v = inspect_channel("frontier:signals", "ontology_sparse_region", rows, sem(rest_range=(0.0, 0.5)))
    assert v.finding is not None and v.finding.kind == "constant"


# --- sweep + glossary resolution ---------------------------------------------------------------

def entry(channel, node=None, semantics=None):
    return FieldChannelGlossaryEntry(channel=channel, level=(), category="c", meaning="m", node=node,
                                     semantics=_freeze_semantics(semantics) if semantics else ())


def test_semantic_channels_resolves_qualified_over_bare():
    entries = [
        entry("prediction_error"),  # bare, no semantics
        entry("prediction_error", node="node:substrate.route", semantics={"value_kind": "count"}),
        entry("prediction_error", node="node:substrate.cabinet", semantics={"value_kind": "trigger"}),
        entry("vision_frame_staleness", semantics={"value_kind": "level", "rest_range": [0, 0]}),
        entry("cpu_pressure"),
    ]
    observed = [("node:athena", "prediction_error"), ("node:substrate.route", "prediction_error"),
                ("node:substrate.vision_organ", "vision_frame_staleness")]
    got = semantic_channels(entries, observed)
    assert set(got) == {("node:substrate.route", "prediction_error"), ("node:substrate.cabinet", "prediction_error"),
                        ("node:substrate.vision_organ", "vision_frame_staleness")}
    assert got[("node:substrate.vision_organ", "vision_frame_staleness")]["rest_range"] == (0, 0)
    assert semantic_channel_names(entries) == ["prediction_error", "vision_frame_staleness"]


def test_sweep_reports_absent_channels_and_counts_findings():
    healthy = [ImmuneSample(t, 0.0, updated_at=t) for t in minutes()]
    zombie = [ImmuneSample(t, 1.0, updated_at=t, source=f"s{i}") for i, t in enumerate(minutes())]
    res = sweep({("n:a", "c"): healthy, ("n:b", "c"): zombie},
                {("n:a", "c"): sem(rest_range=(0, 0)), ("n:b", "c"): sem(rest_range=(0, 0.5)),
                 ("n:gone", "c"): sem()},
                config=ImmuneConfig())
    assert res.open_count == 1 and res.findings[0].metric_id == "n:b.c"
    by_id = {v.metric_id: v for v in res.verdicts}
    assert by_id["n:gone.c"].skipped == "absent in the window"
    assert res.findings[0].to_dict()["kind"] == "constant"


def test_one_early_rewrite_then_carried_all_day_is_not_constant():
    """Review finding: two stamps at the window start, then 23 h carried, used to pass as written."""
    ts = minutes()
    samples = [ImmuneSample(t, 0.5, updated_at=ts[min(i, 1)]) for i, t in enumerate(ts)]
    v = inspect_channel("node:substrate.chat", "prediction_error", samples, sem(rest_range=(0.0, 0.0)))
    assert v.finding is None and v.checks["constant"].startswith("clear:carried forward, re-written in only 1 h")


def test_series_without_stamps_cannot_judge_constant():
    samples = [ImmuneSample(t, 1.0) for t in minutes()]
    v = inspect_channel("frontier:signals", "ontology_sparse_region", samples, sem(rest_range=(0.0, 0.5)))
    assert v.finding is None and v.checks["constant"] == "cannot_judge:no updated_at stamps"


def test_non_mapping_arguments_do_not_raise():
    assert sweep({("n", "c"): []}, ["not", "a", "mapping"]).findings == []
    assert sweep(["x"], {("n", "c"): sem()}).findings == []
