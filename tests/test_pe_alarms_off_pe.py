"""PE retire Patch 2 (design PR #3808): the three alarms leave the prediction-error slot.

node:substrate.cabinet / .strain / .street write node metadata ``signal_value`` and the
signal table, the glossary files them under ``channel: alarm``, and every reader still sees
them as alarms -- most importantly, all three still become attention contest candidates
(Orion's review question 1: a channel move that silently drops them from contests is the
dangerous failure).
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion.prediction.alarm_nodes import ALARM_NODE_IDS
from orion.substrate import signal_history as sh

REPO = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)
ALARMS = sorted(ALARM_NODE_IDS)
CAMERAS = sorted(sh.SCENE_CHANGE_NODE_IDS)


def _alarm_magnitude(value: float = 1.0):
    from orion.substrate.prediction_error_magnitude import compute_prediction_error_magnitude

    hist = [(NOW - timedelta(minutes=15 * i), 0.0) for i in range(1, 672)]
    hist.append((NOW - timedelta(seconds=5), value))
    return compute_prediction_error_magnitude(
        value=value, observed_at=NOW - timedelta(seconds=5), history=sorted(hist), now=NOW
    )


# --- the moved set --------------------------------------------------------------------------


def test_moved_set_is_exactly_the_alarms_and_cameras_stay_on_pe_until_patch_3() -> None:
    assert sh.MOVED_SIGNAL_NODE_IDS == ALARM_NODE_IDS
    for nid in ALARMS:
        assert sh.is_moved_signal_node(nid)
        assert sh.glossary_channel_for(nid) == "alarm"
        assert sh.node_value_key(nid) == "signal_value"
        assert sh.history_table_for(nid) == "substrate_node_signal_history"
    for nid in CAMERAS + ["node:prediction.body"]:
        assert not sh.is_moved_signal_node(nid)
        assert sh.glossary_channel_for(nid) == "prediction_error"
        assert sh.node_value_key(nid) == "prediction_error"
        assert sh.history_table_for(nid) == "substrate_node_prediction_error_history"


def test_alarm_markers_still_importable_for_every_importer() -> None:
    """ALARM_NODE_IDS / is_alarm_node stay (8 importers incl. orion/field/pressure.py:29)."""
    from orion.field.pressure import skips_merge  # noqa: F401  (imports is_alarm_node)
    from orion.prediction.alarm_nodes import is_alarm_node
    from orion.substrate.prediction_error_freshness import ALARM_NODE_IDS as reexport

    assert reexport == ALARM_NODE_IDS and all(is_alarm_node(n) for n in ALARMS)


def test_light_sql_module_does_not_import_orion_substrate() -> None:
    """node_history (introspect) and the thought cue import it without the heavy package."""
    code = (
        "import sys; import orion.prediction.signal_history_sql as m; "
        "assert 'orion.substrate' not in sys.modules, 'heavy import'; print(len(m.MOVED_SIGNAL_NODE_IDS))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "3"


# --- node values ----------------------------------------------------------------------------


@pytest.mark.parametrize("node_id", ALARMS)
def test_a_stale_prediction_error_key_on_an_alarm_is_never_read(node_id: str) -> None:
    assert sh.node_signal_value(node_id, {"prediction_error": 0.9}) is None
    assert sh.node_signal_value(node_id, {"prediction_error": 0.9, "signal_value": 0.0}) == 0.0
    assert sh.node_signal_value("node:prediction.body", {"prediction_error": 0.4, "signal_value": 1.0}) == 0.4


def test_history_samples_route_alarms_to_signal_table_only() -> None:
    t = NOW
    samples = [
        ("node:substrate.strain", t, 1.0),
        ("node:substrate.perception", t, 0.2),
        ("node:prediction.body", t, 0.1),
    ]
    pe_rows, sig_rows = sh.split_history_samples(iter(samples))
    assert [r[0] for r in pe_rows] == ["node:substrate.perception", "node:prediction.body"]
    assert sorted((r[0], r[3]) for r in sig_rows) == [
        ("node:substrate.perception", "scene_change"), ("node:substrate.strain", "alarm"),
    ]


def test_pressure_seed_reads_signal_value_for_an_alarm() -> None:
    from orion.substrate.pressure import PressureConfig, prediction_error_pressure

    cfg = PressureConfig()

    def node(nid, **md):
        return SimpleNamespace(node_id=nid, metadata=md, temporal=SimpleNamespace(observed_at=NOW))

    assert prediction_error_pressure(node("node:substrate.cabinet", signal_value=1.0), cfg, now=NOW) > 0
    # The old key alone (a stale write) seeds nothing: kill means kill.
    assert prediction_error_pressure(node("node:substrate.cabinet", prediction_error=1.0), cfg, now=NOW) == 0


# --- the attention contest (Orion Q1) ---------------------------------------------------------


@pytest.mark.parametrize("node_id", ALARMS)
def test_every_alarm_still_becomes_a_contest_candidate_judged_as_a_trigger(node_id: str) -> None:
    from orion.attention.world_first import (
        judge_candidate,
        node_candidate,
        node_prediction_error_semantics,
    )
    from orion.substrate.prediction_error_freshness import node_signal_label

    assert node_prediction_error_semantics(node_id) == ("trigger", None)
    cand = node_candidate(
        node_id=node_id,
        label=node_signal_label(node_id, node_id.removeprefix("node:substrate.")),
        magnitude=_alarm_magnitude(),
        observed_at=NOW - timedelta(seconds=5),
        now=NOW,
    )
    assert cand.value_kind == "trigger"
    verdict = judge_candidate(cand)
    assert "not a measurement" not in verdict.reason, verdict.reason
    assert "no declared polarity" not in verdict.reason, verdict.reason
    assert verdict.eligible, verdict.reason
    assert "prediction error" not in cand.label


def test_walkway_band_floor_stays_live() -> None:
    from orion.schemas.attention_candidate import ATTENTION_MIN_BAND_BY_NODE_ID

    assert "node:substrate.perception_walkway" in ATTENTION_MIN_BAND_BY_NODE_ID


@pytest.mark.parametrize("node_id", ALARMS)
def test_substrate_broadcast_keeps_an_alarm_that_carries_only_signal_value(node_id: str) -> None:
    import orion.substrate.attention_broadcast as ab

    node = SimpleNamespace(
        node_id=node_id, label="alarm", node_kind="concept",
        metadata={"signal_value": 1.0, "signal_kind": "alarm"},
        signals=SimpleNamespace(confidence=1.0),
        temporal=SimpleNamespace(observed_at=NOW - timedelta(seconds=5)),
    )
    signals, trace, _ = ab.world_first_signals(
        [node], magnitude_by_node_id={node_id: _alarm_magnitude()}, external_candidates=[], now=NOW,
    )
    assert any(node_id in s.evidence_refs and s.signal_kind == "substrate_alarm" for s in signals), trace


def test_self_readings_still_find_the_alarm_prompt_nodes() -> None:
    from orion.attention import self_readings as sr

    sr._NODES_CACHE = None
    try:
        nodes = sr.prompt_reaching_nodes()
    finally:
        sr._NODES_CACHE = None
    assert {"node:substrate.cabinet", "node:substrate.strain"} <= set(nodes)


# --- Orion-facing text --------------------------------------------------------------------------


def test_self_reading_for_an_alarm_never_says_prediction_error_and_cites_signal_table() -> None:
    from orion.attention import self_readings as sr
    from orion.attention.pe_history_cache import NodePeHistoryCache

    nid = "node:substrate.strain"
    rows = [(nid, NOW - timedelta(minutes=m), 0.0) for m in range(2, 7 * 24 * 60, 30)]
    for start in (60 * 24, 60 * 48, 60 * 72, 60 * 96, 60 * 120, 60 * 144):
        rows.append((nid, NOW - timedelta(minutes=start) + timedelta(seconds=1), 1.0))
    rows.append((nid, NOW - timedelta(seconds=20), 1.0))
    rows.sort(key=lambda r: r[1])
    cache = NodePeHistoryCache()
    cache.refresh(lambda since: [r for r in rows if r[1] >= since], now=NOW)
    (reading,) = sr.build_self_readings(cache, now=NOW, nodes=[nid])
    assert reading.ref.startswith("alarm:strain@")
    assert reading.source_ref.startswith("substrate_node_signal_history:")
    assert "prediction error" not in reading.label.lower()
    block = sr.render_self_readings([reading], now=NOW)
    assert "prediction error" not in block.text.lower()


def test_node_history_dig_output_calls_an_alarm_an_alarm() -> None:
    import asyncio

    from orion.introspect import node_history as nh
    from orion.introspect.tests.node_history_fake import FakeConn

    nid = "node:substrate.cabinet"
    rows = [(nid, NOW - timedelta(minutes=5 * i), 0.0) for i in range(1, 100)]
    rows.append((nid, NOW - timedelta(minutes=1), 0.7))
    item = asyncio.run(nh.read_node_history(FakeConn(rows), nid, NOW - timedelta(hours=24), NOW))
    assert item.kind == "node_alarm_history"
    assert "prediction error" not in item.text.lower()
    assert "an alarm" in item.text and "substrate_node_signal_history" in item.text
    assert "substrate_node_signal_history" in nh.SUMMARY_SQL


def test_metacog_episode_ref_for_an_alarm_names_the_signal_table() -> None:
    from orion.metacog.self_signal_episode import evidence_ref

    assert evidence_ref("node:substrate.strain", NOW.timestamp()).startswith("substrate_node_signal_history:")
    assert evidence_ref("node:prediction.body", NOW.timestamp()).startswith(
        "substrate_node_prediction_error_history:"
    )


# --- the shared history SQL ---------------------------------------------------------------------


def test_history_source_sql_keeps_other_nodes_and_unions_moved_ids() -> None:
    sql = sh.HISTORY_WITH_MOVED_SOURCE_SQL
    assert "node_id <> ALL(ARRAY[" in sql          # PE side: every node except the moved ids
    assert "DISTINCT ON (node_id, observed_at)" in sql
    assert sql.index("0 AS pref") < sql.index("1 AS pref")  # the signal table wins a tie
    assert ":" not in sql.replace("::text[]", "").replace("node:substrate", "")  # no bind params
    for nid in ALARMS:
        assert f"'{nid}'" in sql
    for nid in CAMERAS:
        assert nid not in sql


def test_history_source_sql_refuses_unsafe_literals_and_unknown_modes() -> None:
    with pytest.raises(ValueError):
        sh.history_with_moved_source_sql({"node:x'; drop table t; --": sh.READ_BOTH})
    with pytest.raises(ValueError):
        sh.history_with_moved_source_sql({"node:substrate.cabinet": "both_please"})


def test_read_rule_is_per_node_and_cameras_are_never_read_both() -> None:
    """Design rev 3 (PR #3827): Patch 3 flips the cameras to signal_only (new-scale rows only);
    their old- and new-scale rows must never share a window, so no camera is in a read-both union."""
    assert sh.HISTORY_READ_MODE_BY_NODE_ID == {nid: sh.READ_BOTH for nid in ALARMS}
    for nid in CAMERAS:
        assert sh.history_read_mode(nid) is None
    sql = sh.history_with_moved_source_sql(
        {"node:substrate.cabinet": sh.READ_BOTH, "node:substrate.perception": sh.SIGNAL_ONLY}
    )
    both_part, signal_only_part = sql.split("read_both_rows")
    assert "node:substrate.perception" not in both_part.split("WHERE node_id <> ALL")[1].split("\n", 1)[1]
    assert "pref" not in signal_only_part and "substrate_node_prediction_error_history" not in signal_only_part
    assert "node:substrate.perception" in signal_only_part
