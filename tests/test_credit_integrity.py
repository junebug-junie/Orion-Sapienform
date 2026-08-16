"""Tests for the feedback-credit integrity watch (rebuilt 2026-08-16).

The load-bearing ones are `test_real_fix_is_not_flagged_silent` and
`test_merge_winner_churn_is_not_flagged_silent`: they are the exact two live
failure modes post-merge review found in the prior (shape-based) version --
a genuine 0.9->0.0 fix reading as "only decay touched this", and a max()
merge across multiple always-writing producers reading as a silent outage
purely because the envelope fell. If either fails, this module has
regressed to the defect it was rebuilt to fix.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.field.credit_integrity import (  # noqa: E402
    SILENT,
    _classify_tick,
    analyse_credit_integrity,
    load_credited_dimensions,
)
from orion.field.pressure import collect_field_channel_pressures  # noqa: E402
from orion.schemas.field_state import FieldStateV1  # noqa: E402

T0 = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)
TICK = 2.0  # seconds; live median is 2.04


def _node_state(
    i: int,
    *,
    value: float,
    write_at: datetime | None,
    node_id: str = "node:athena",
    channel: str = "reliability_pressure",
) -> FieldStateV1:
    """One tick where `channel` is a genuine node_vectors entry.
    `write_at=None` reproduces "never stamped"; a real datetime reproduces
    node_vector_updated_at's actual shape (persists across ticks until the
    next real write, exactly like live data)."""
    at = T0 + timedelta(seconds=i * TICK)
    return FieldStateV1.model_validate(
        {
            "schema_version": "field.state.v1",
            "generated_at": at.isoformat(),
            "tick_id": f"t{i}",
            "node_vectors": {node_id: {channel: value}},
            "node_vector_updated_at": (
                {node_id: {channel: write_at.isoformat()}} if write_at else {}
            ),
            "capability_vectors": {},
            "capability_provenance": {},
            "edges": [],
            "recent_perturbations": [],
        }
    )


def _capability_state(
    i: int,
    *,
    value: float,
    contributing: bool,
    capability_id: str = "capability:llm_inference",
    channel: str = "pressure",
) -> FieldStateV1:
    """One tick where `channel` is a capability_vectors entry.
    `contributing=False` reproduces apply_diffusion's documented "clear
    stale provenance too... when nobody contributes" -- provenance for this
    tick is simply absent, exactly like the live producer."""
    at = T0 + timedelta(seconds=i * TICK)
    provenance = {capability_id: {channel: "node:athena"}} if contributing else {}
    return FieldStateV1.model_validate(
        {
            "schema_version": "field.state.v1",
            "generated_at": at.isoformat(),
            "tick_id": f"t{i}",
            "node_vectors": {"node:athena": {"availability": 1.0}},
            "capability_vectors": {capability_id: {channel: value}},
            "capability_provenance": provenance,
            "edges": [],
            "recent_perturbations": [],
        }
    )


DIMS_RELIABILITY = {"reliability_pressure": ["reliability_pressure"]}
DIMS_RESOURCE = {"resource_pressure": ["pressure"]}


def test_policy_dimensions_map_to_real_channels():
    """The bug the first version shipped with: the policy names DIMENSIONS
    and the merge returns CHANNELS. `resource_pressure` is fed by
    `pressure`."""
    dims, window = load_credited_dimensions()
    assert dims["resource_pressure"] == ["pressure"]
    assert window > 0
    unmapped = [d for d, chans in dims.items() if not chans]
    assert not unmapped, unmapped


def test_unmappable_dimension_is_a_finding_not_silence():
    states = [_node_state(i, value=0.5, write_at=T0) for i in range(40)]
    report = analyse_credit_integrity(states, {"nonexistent_dim": []}, 30.0)
    assert [f.kind for f in report.findings] == ["unmappable_dimension"]


def test_healthy_node_sourced_series_produces_no_findings():
    states = []
    for i in range(60):
        write_at = T0 + timedelta(seconds=i * TICK)
        states.append(_node_state(i, value=0.5 + (i % 3) * 0.01, write_at=write_at))
    report = analyse_credit_integrity(states, DIMS_RELIABILITY, 30.0)
    assert report.findings == []
    assert report.source_kind["reliability_pressure"]["node"] == 60


def test_healthy_capability_sourced_series_produces_no_findings():
    states = [
        _capability_state(i, value=0.5 + (i % 3) * 0.01, contributing=True)
        for i in range(60)
    ]
    report = analyse_credit_integrity(states, DIMS_RESOURCE, 30.0)
    assert report.findings == []
    assert report.source_kind["resource_pressure"]["capability"] == 60
    assert report.longest_unbacked_run["resource_pressure"] == 0


def test_real_fix_is_not_flagged_silent():
    """The exact live failure the prior version shipped: reliability_pressure
    genuinely dropping 0.9 -> 0.0 because a real problem was fixed -- the
    single most important true positive the feedback loop should credit.
    The prior shape classifier read this as `silent_producer` because it
    never increased. A real node write behind the drop must clear it."""
    states = []
    for i in range(60):
        value = 0.9 if i < 30 else 0.0
        write_at = T0 + timedelta(seconds=i * TICK)  # written EVERY tick
        states.append(_node_state(i, value=value, write_at=write_at))
    report = analyse_credit_integrity(states, DIMS_RELIABILITY, 30.0)
    assert report.findings == [], report.findings


def test_merge_winner_churn_is_not_flagged_silent():
    """The other live failure mode: resource_pressure's merge winner
    changes source mid-window while every producer keeps writing. A max()
    envelope over several producers can fall monotonically even though
    nothing went quiet -- checked here via TWO capability contributors
    alternating who wins, both writing every tick."""
    states = []
    for i in range(60):
        at = T0 + timedelta(seconds=i * TICK)
        # Two sources both contribute every tick; the envelope (max) falls
        # as the currently-higher one decreases and the winner alternates.
        a = max(0.0, 0.9 - i * 0.01)
        b = max(0.0, 0.85 - i * 0.009)
        states.append(
            FieldStateV1.model_validate(
                {
                    "schema_version": "field.state.v1",
                    "generated_at": at.isoformat(),
                    "tick_id": f"t{i}",
                    "node_vectors": {"node:athena": {"availability": 1.0}},
                    "capability_vectors": {
                        "capability:a": {"pressure": a},
                        "capability:b": {"pressure": b},
                    },
                    "capability_provenance": {
                        "capability:a": {"pressure": "node:athena"},
                        "capability:b": {"pressure": "node:athena"},
                    },
                    "edges": [],
                    "recent_perturbations": [],
                }
            )
        )
    report = analyse_credit_integrity(states, DIMS_RESOURCE, 30.0)
    assert report.findings == [], report.findings


def test_node_sourced_outage_is_a_finding():
    """Signal 1: a genuine gap in node_vector_updated_at -- nothing wrote
    reliability_pressure for 30+ seconds, verified against real live data
    (2026-08-14T01:12:51-01:13:20, node:rpc_timeout's last write frozen at
    01:12:49 for the whole stretch) before this fixture was written."""
    states = []
    write_at = T0
    for i in range(80):
        at = T0 + timedelta(seconds=i * TICK)
        if i < 20:
            write_at = at  # writes every tick for the first 40s
        # from i=20 onward the stamp freezes -- nobody writes again
        states.append(_node_state(i, value=0.5, write_at=write_at))
    report = analyse_credit_integrity(states, DIMS_RELIABILITY, 30.0)
    silent = [f for f in report.findings if f.kind == SILENT]
    assert silent, "a 30s+ gap with no new write must be found"
    assert silent[0].evidence == "timestamp"
    assert silent[0].span_seconds >= 30.0


def test_capability_sourced_outage_is_a_finding():
    """Signal 2: capability_provenance genuinely cleared (apply_diffusion's
    documented "nobody contributed" state) for 60 straight seconds."""
    states = []
    for i in range(120):
        contributing = not (30 <= i < 60)  # 60s with nothing contributing
        states.append(_capability_state(i, value=0.3, contributing=contributing))
    report = analyse_credit_integrity(states, DIMS_RESOURCE, 30.0)
    silent = [f for f in report.findings if f.kind == SILENT]
    assert silent, "a 60s uncontributed stretch must be found"
    assert silent[0].evidence == "contribution"
    assert silent[0].tick_count >= 30
    assert report.longest_unbacked_run["resource_pressure"] >= 30


def test_a_short_capability_gap_is_not_a_finding():
    """Isolated single-tick gaps below the window must not fire, or the
    watch cries wolf on ordinary jitter."""
    states = [
        _capability_state(i, value=0.5, contributing=(i % 40 != 0)) for i in range(120)
    ]
    report = analyse_credit_integrity(states, DIMS_RESOURCE, 30.0)
    assert [f for f in report.findings if f.kind == SILENT] == []
    assert report.longest_unbacked_run["resource_pressure"] == 1


def test_node_tautology_is_fixed_by_using_the_real_stamp():
    """The exact synthetic proof from the post-merge review: a node channel
    with a value that changes every tick but a write timestamp that never
    advances must NOT read as fresh just because the winner is a node."""
    states = []
    for i in range(40):
        value = 0.9 * (0.92 ** i)  # value moves every tick
        states.append(_node_state(i, value=value, write_at=T0))  # stamp frozen at T0
    report = analyse_credit_integrity(states, DIMS_RELIABILITY, 30.0)
    silent = [f for f in report.findings if f.kind == SILENT]
    assert silent, "a frozen stamp must be caught even though the value keeps changing"


def test_window_size_comes_from_the_policy_not_a_constant():
    states = []
    write_at = T0
    for i in range(60):
        at = T0 + timedelta(seconds=i * TICK)
        if i < 10:
            write_at = at
        states.append(_node_state(i, value=0.5, write_at=write_at))
    assert analyse_credit_integrity(states, DIMS_RELIABILITY, 20.0).findings
    assert not analyse_credit_integrity(states, DIMS_RELIABILITY, 10_000.0).findings


def test_findings_report_one_row_per_channel_not_per_overlapping_window():
    states = []
    for i in range(200):
        at = T0 + timedelta(seconds=i * TICK)
        states.append(_node_state(i, value=0.5, write_at=T0))  # never advances
    report = analyse_credit_integrity(states, DIMS_RELIABILITY, 30.0)
    assert len([f for f in report.findings if f.kind == SILENT]) == 1


def test_mixed_sourcing_picks_the_plurality_mechanism_and_reports_the_split():
    """A dimension whose winner type genuinely changes mid-series (a node
    starts also carrying `reliability_pressure` after living entirely under
    a capability) must not crash, and must report BOTH counts rather than
    hide the split -- even though it commits to whichever mechanism has
    more ticks. 25 capability-sourced ticks (with a genuine 30s+ gap) vs 15
    node-sourced -> capability signal wins and the gap must still surface."""
    states = []
    for i in range(25):
        contributing = not (5 <= i < 21)  # 16 ticks = 30s uncontributed stretch
        states.append(
            _capability_state(i, value=0.5, contributing=contributing, channel="reliability_pressure")
        )
    for i in range(25, 40):
        states.append(_node_state(i, value=0.5, write_at=T0 + timedelta(seconds=i * TICK)))
    report = analyse_credit_integrity(states, DIMS_RELIABILITY, 30.0)
    assert report.source_kind["reliability_pressure"] == {"node": 15, "capability": 25}
    silent = [f for f in report.findings if f.kind == SILENT]
    assert silent and silent[0].evidence == "contribution"


@pytest.mark.parametrize("n", [0, 1, 5])
def test_short_histories_do_not_raise(n):
    states = [_node_state(i, value=0.5, write_at=T0) for i in range(n)]
    report = analyse_credit_integrity(states, DIMS_RELIABILITY, 30.0)
    assert report.findings == [] or all(
        f.kind in ("unmappable_dimension", "mixed_or_absent_sourcing") for f in report.findings
    )


def test_absent_channel_tick_is_not_read_as_fresh():
    """A channel entirely missing from the merge classifies as "absent",
    not silently as fresh under either signal. Unreachable from
    `_samples_for`'s normal call path today (the winning channel is always a
    member of `merged` by construction -- `map_channels_to_dimensions_with_
    provenance` only ever picks a winner from `channel_pressures.items()`),
    but `_classify_tick` is defensive on purpose and this pins that."""
    state = _node_state(0, value=0.5, write_at=T0, channel="reliability_pressure")
    merged, provenance = collect_field_channel_pressures(state)
    kind, stamp, fresh = _classify_tick(state, merged, provenance, "execution_pressure")
    assert kind == "absent"
    assert stamp is None
    assert fresh is None
