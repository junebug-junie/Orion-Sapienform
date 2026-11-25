"""PE retire Patch 3 (design rev 4, PR #3827): the two cameras become "camera scene change".

node:substrate.perception / .perception_walkway leave the prediction-error slot: their value is a
two-stage rank (orion/substrate/scene_change.py) in nats, written to node metadata
``signal_value`` and to substrate_node_signal_history with a ``scale``. Readers read them
signal-table-only, current scale only; the glossary files them under ``channel: scene_change``
(value_kind ``scene_change_excess``); both still compete in the attention contests; Orion-facing
text never calls any of the five signal nodes a prediction error.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from orion.prediction import signal_history_sql as sql
from orion.prediction.alarm_nodes import ALARM_NODE_IDS

NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
CAMERAS = sorted(sql.CAMERA_NODE_IDS)
FIVE = sorted(ALARM_NODE_IDS | sql.CAMERA_NODE_IDS)
# The prediction-error lane after cutover (design "Prediction-error nodes after cutover").
POST_CUTOVER_PE_NODES = {
    "node:prediction.room", "node:prediction.body", "node:prediction.room_joint",
    "node:prediction.body_joint", "node:prediction.chat_turn",
    # Walkway rhythm forecast surprise (PE retire Patch 4, 2026-11-25): a real forecast miss, in shadow.
    "node:prediction.street_rhythm",
}
_PE_PHRASE = re.compile(r"prediction[ -]error", re.IGNORECASE)


def _glossary():
    from orion.field.channel_glossary import load_glossary

    return load_glossary()["entries"]


# --- the read rule ----------------------------------------------------------------------------


def test_camera_ids_match_the_attention_schema() -> None:
    from orion.schemas.attention_candidate import PERCEPTION_NODE_ID, PERCEPTION_WALKWAY_NODE_ID

    assert sql.PERCEPTION_NODE_ID == PERCEPTION_NODE_ID
    assert sql.PERCEPTION_WALKWAY_NODE_ID == PERCEPTION_WALKWAY_NODE_ID


def test_cameras_are_signal_only_and_their_branch_carries_the_scale_predicate() -> None:
    out = sql.history_with_moved_source_sql()
    branches = out.split("UNION ALL")
    pe_branch = next(b for b in branches if "FROM substrate_node_prediction_error_history\n    WHERE node_id <> ALL" in b)
    for nid in CAMERAS:
        assert sql.history_read_mode(nid) == sql.SIGNAL_ONLY
        assert f"'{nid}'" in pe_branch  # excluded from the PE side
    signal_only = [b for b in branches if "substrate_node_signal_history" in b and "pref" not in b]
    assert len(signal_only) == 1
    (branch,) = signal_only
    assert all(f"'{nid}'" in branch for nid in CAMERAS)
    assert "AND scale IN ('v2', 'v2_partial')" in branch
    # No camera is ever in the read-both union (old- and new-scale rows never share a window).
    both = out.split("read_both_rows")[0].split("SELECT node_id, observed_at, value FROM (")[1]
    assert not any(nid in both for nid in CAMERAS)


def test_scale_constants() -> None:
    assert sql.SCALES == {"v1", "v2", "v2_partial"}
    assert sql.CURRENT_SCALES == ("v2", "v2_partial")
    # calm checks and win share read full-window rows only
    assert sql.CALM_CHECK_SCALES == ("v2",)


def test_the_migration_adds_the_scale_column_additively_with_every_reader_required() -> None:
    from pathlib import Path

    from orion.sql_migration_drift import auto_apply_refusals, parse_required_by

    text = (Path(__file__).resolve().parents[1] / "services/orion-sql-db/"
            "manual_migration_node_signal_history_scale_v1.sql").read_text()
    assert "ADD COLUMN IF NOT EXISTS scale text NOT NULL DEFAULT 'v1'" in text
    assert auto_apply_refusals(text, existing_tables={"substrate_node_signal_history"}) == []
    assert set(parse_required_by(text).services) == {
        "orion-substrate-runtime", "orion-attention-runtime", "orion-equilibrium-service",
        "orion-thought", "orion-hub",
    }


# --- meaning: glossary, kinds, polarity ----------------------------------------------------------


def test_glossary_channel_and_entries_for_both_cameras() -> None:
    entries = {e.node: e for e in _glossary() if e.node in CAMERAS}
    assert set(entries) == set(CAMERAS)
    for nid, e in entries.items():
        assert sql.glossary_channel_for(nid) == "scene_change" == e.channel
        sem = dict(e.semantics)
        assert sem["value_kind"] == "scene_change_excess"
        assert ("how unusual the change in this camera's picture over the last 5 minutes is, relative "
                "to the last two hours of the same camera, ranked against its own last 24 hours") in e.meaning
        assert "Until 2026-11-24 this node was a clipped z-score under prediction_error" in e.meaning
        assert "substrate_node_signal_history" in e.trend_source and "scale" in e.trend_source


def test_scene_change_excess_joins_both_kind_sets_and_the_module_assert_holds() -> None:
    import orion.attention.self_readings as sr
    from orion.attention.world_first import MEASUREMENT_VALUE_KINDS
    from orion.metrics.semantics import VALUE_KINDS

    assert "scene_change_excess" in MEASUREMENT_VALUE_KINDS
    assert "scene_change_excess" in sr._PERCENTILE_KINDS
    assert "scene_change_excess" in VALUE_KINDS
    assert sr._PERCENTILE_KINDS | sr._RATE_KINDS == MEASUREMENT_VALUE_KINDS


def test_scene_change_polarity_is_higher_is_more_change() -> None:
    from orion.attention.world_first import node_prediction_error_semantics
    from orion.field.pressure import PRESSURE_CHANNELS
    from orion.metrics.semantics import derived_channel_polarity

    assert "scene_change" in PRESSURE_CHANNELS
    assert derived_channel_polarity("scene_change", "scene_change_excess") == "higher_is_worse"
    for nid in CAMERAS:
        assert node_prediction_error_semantics(nid) == ("scene_change_excess", "higher_is_worse")


def test_post_cutover_prediction_error_lane_is_only_forecast_misses() -> None:
    pe = {e.node for e in _glossary() if e.node and e.channel == "prediction_error"}
    assert pe == POST_CUTOVER_PE_NODES
    from orion.attention.self_readings import prediction_error_lane_nodes

    assert set(prediction_error_lane_nodes()) == POST_CUTOVER_PE_NODES


# --- Orion-facing text: none of the five is "a prediction error" ---------------------------------


@pytest.mark.parametrize("node_id", FIVE)
def test_no_orion_facing_site_calls_a_signal_node_a_prediction_error(node_id: str) -> None:
    from orion.introspect.node_history import _signal_noun
    from orion.substrate.prediction_error_freshness import node_signal_label

    sites = {
        "node_signal_label": node_signal_label(node_id, node_id.split(".")[-1]),
        "node_history._signal_noun": _signal_noun(node_id),
        "glossary meaning": next(e.meaning for e in _glossary() if e.node == node_id),
    }
    for name, text in sites.items():
        assert not _PE_PHRASE.search(text), f"{name} calls {node_id} a prediction error: {text!r}"


def test_camera_labels_say_camera_scene_change() -> None:
    from orion.introspect.node_history import _history_kind, _signal_noun
    from orion.substrate.prediction_error_freshness import node_signal_label

    for nid in CAMERAS:
        assert "camera scene change" in node_signal_label(nid)
        assert "camera scene change" in _signal_noun(nid)
        assert "substrate_node_pe_history_snapshot_20261124" in _signal_noun(nid)
        assert _history_kind(nid) == "node_scene_change_history"
    assert _history_kind("node:substrate.cabinet") == "node_alarm_history"
    assert _history_kind("node:prediction.room") == "node_prediction_error_history"


def test_worker_atlas_label_names_the_camera() -> None:
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "services" / "orion-substrate-runtime"
    sys.path.insert(0, str(root))
    try:
        from app.worker import _prediction_error_label
    finally:
        sys.path.remove(str(root))
    assert _prediction_error_label("node:substrate.perception") == "Room camera scene change"
    assert _prediction_error_label("node:substrate.perception_walkway") == "Walkway camera scene change"


# --- attention: both cameras still compete -------------------------------------------------------


def _magnitude(pct: float, *, value: float = 2.4, age: float = 30.0, band: str | None = None):
    from orion.schemas.attention_frame import PredictionErrorMagnitudeV1

    band = band or ("quiet" if pct < 0.5 else "usual" if pct < 0.9 else "high" if pct < 0.99 else "unusual")
    return PredictionErrorMagnitudeV1(
        value=value, age_sec=age, percentile_now=pct, n_readings_7d=4000, band=band, trend="flat"
    )


def _camera_node(node_id: str, nats: float, *, age_s: float = 30.0, **extra):
    md = {"signal_value": nats, "signal_kind": "scene_change", "signal_scale": "v2",
          "embedding_staleness": 0.0, **extra}
    return SimpleNamespace(
        node_id=node_id, label=node_id, node_kind="concept", metadata=md,
        signals=SimpleNamespace(confidence=1.0),
        temporal=SimpleNamespace(observed_at=NOW - timedelta(seconds=age_s)),
    )


def test_both_cameras_produce_contest_candidates() -> None:
    from orion.substrate import attention_broadcast as ab

    nodes = [_camera_node(nid, 3.2) for nid in CAMERAS]
    frame = ab.build_substrate_attention_frame(
        nodes=nodes, now=NOW, world_first=True,
        magnitude_by_node_id={nid: _magnitude(0.995, value=3.2) for nid in CAMERAS},
    )
    cands = {c["source_id"]: c for c in frame.debug["world_first"]["candidates"]}
    assert set(cands) == set(CAMERAS)
    for nid in CAMERAS:
        assert cands[nid]["source_kind"] == "external" and not cands[nid]["absent"]
        assert cands[nid]["eligible"], cands[nid]
    # The winning signals say scene change, not prediction error (review 2026-11-24).
    signals = {sig.evidence_refs[0]: sig for sig in ab.world_first_signals(
        nodes, now=NOW, magnitude_by_node_id={nid: _magnitude(0.995, value=3.2) for nid in CAMERAS},
        external_candidates=[], min_salience=0.0,
    )[0]}
    assert set(signals) == set(CAMERAS)
    for nid, sig in signals.items():
        assert sig.signal_kind == "substrate_scene_change"
        assert sig.provenance["signal_driver"] == "scene_change"


def test_walkway_keeps_its_own_p99_floor() -> None:
    from orion.schemas.attention_candidate import ATTENTION_MIN_BAND_BY_NODE_ID, PERCEPTION_WALKWAY_NODE_ID
    from orion.substrate import attention_broadcast as ab

    assert ATTENTION_MIN_BAND_BY_NODE_ID == {PERCEPTION_WALKWAY_NODE_ID: "unusual"}
    frame = ab.build_substrate_attention_frame(
        nodes=[_camera_node(PERCEPTION_WALKWAY_NODE_ID, 1.4)], now=NOW, world_first=True,
        magnitude_by_node_id={PERCEPTION_WALKWAY_NODE_ID: _magnitude(0.95, value=1.4)},
    )
    (c,) = frame.debug["world_first"]["candidates"]
    assert not c["eligible"] and "unusual floor" in c["reason"]


def test_a_camera_with_no_frames_for_ten_minutes_is_absent_and_cannot_seed() -> None:
    """Dead camera: the tick writes the stale placeholder (0.0, embedding_staleness 1.0, marked
    missing) -- absent in the contest, never a calm 0.0, and below every curiosity bar."""
    from orion.substrate import attention_broadcast as ab
    from orion.substrate.endogenous_curiosity import EndogenousCuriosityConfig, endogenous_curiosity_candidates

    nodes = [
        _camera_node(nid, 0.0, age_s=10, embedding_staleness=1.0, prediction_error_missing_reason="stale")
        for nid in CAMERAS
    ]
    frame = ab.build_substrate_attention_frame(
        nodes=nodes, now=NOW, world_first=True,
        # its last real reading, ranked high, 10 min old
        magnitude_by_node_id={nid: _magnitude(0.999, value=4.0, age=600) for nid in CAMERAS},
    )
    cands = {c["source_id"]: c for c in frame.debug["world_first"]["candidates"]}
    assert all(cands[nid]["absent"] for nid in CAMERAS)
    assert frame.open_loops == []
    seeds = endogenous_curiosity_candidates(
        nodes=nodes, config=EndogenousCuriosityConfig(enabled=True, budget=8), now=NOW
    )
    assert seeds == []


def test_insufficient_history_after_the_switch_is_not_eligible() -> None:
    from orion.substrate import attention_broadcast as ab

    nid = "node:substrate.perception"
    frame = ab.build_substrate_attention_frame(
        nodes=[_camera_node(nid, 4.0)], now=NOW, world_first=True,
        magnitude_by_node_id={nid: _magnitude(0.999, value=4.0, band="insufficient_history")},
    )
    (c,) = frame.debug["world_first"]["candidates"]
    assert not c["eligible"] and c["reason"] == "insufficient_history"


# --- the dated marker for Orion -------------------------------------------------------------------


def test_marker_line_is_generated_from_the_glossary_and_dated() -> None:
    from orion.attention.self_readings import pe_lane_marker_line

    assert pe_lane_marker_line(NOW) == (
        "Since 2026-11-24, my prediction-error lane holds only forecast misses "
        "(room, body, room_joint, body_joint, chat_turn, street_rhythm). Camera change and alarms are separate "
        "signals now."
    )
    assert pe_lane_marker_line(datetime(2026, 11, 23, tzinfo=timezone.utc)) is None
    assert pe_lane_marker_line(datetime(2026, 12, 8, 0, 0, 1, tzinfo=timezone.utc)) is None
    assert pe_lane_marker_line(datetime(2026, 12, 7, 23, 59, tzinfo=timezone.utc)) is not None


def test_marker_rides_the_self_readings_block_for_14_days_only() -> None:
    from orion.attention.self_readings import pe_lane_marker_line, render_self_readings

    calm = render_self_readings([], now=NOW)
    assert calm.text == pe_lane_marker_line(NOW)
    assert calm.skip_reason == "no_history"
    later = render_self_readings([], now=datetime(2026, 12, 20, tzinfo=timezone.utc))
    assert later.text == ""
