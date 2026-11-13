"""PE suite rebuild, "Alarms move out of the PE group" (2026-11-13).

node:substrate.cabinet (warming alarm), node:substrate.strain (strain news) and
node:substrate.street (doorbell ring / graded handoff) fire on a fact, not on a
miss against a forecast. They keep their real alarm path (the attention
contests, the dynamics pressure seed, self-readings as fires per hour) and stop
being read as a surprise SIZE (curiosity's prediction-error seeds, the
self-model's PE domains, PE labels).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import yaml
from pathlib import Path

from orion.substrate.prediction_error_freshness import (
    ALARM_LABELS,
    ALARM_NODE_IDS,
    RETIRED_PREDICTION_ERROR_NODE_IDS,
    is_alarm_node,
    node_signal_label,
)

REPO = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)
ALARMS = sorted(ALARM_NODE_IDS)
LIVE = "node:substrate.execution"


def _glossary_trigger_nodes() -> set[str]:
    glossary = yaml.safe_load((REPO / "config/field/field_channel_glossary.v1.yaml").read_text())
    return {
        e["node"]
        for e in glossary.get("channels") or []
        if e.get("node")
        and e.get("channel") == "prediction_error"
        and (e.get("semantics") or {}).get("value_kind") == "trigger"
    }


def test_alarm_set_is_exactly_the_glossary_trigger_entries() -> None:
    """The glossary owns the meaning: a prediction_error entry whose value is "it
    fired" is an alarm. The id set must not drift from it."""
    assert ALARM_NODE_IDS == {
        "node:substrate.cabinet", "node:substrate.strain", "node:substrate.street",
    }
    assert ALARM_NODE_IDS == _glossary_trigger_nodes()
    assert not ALARM_NODE_IDS & RETIRED_PREDICTION_ERROR_NODE_IDS
    assert set(ALARM_LABELS) == ALARM_NODE_IDS


def test_glossary_never_calls_an_alarm_a_prediction_error_reading() -> None:
    glossary = yaml.safe_load((REPO / "config/field/field_channel_glossary.v1.yaml").read_text())
    for e in glossary.get("channels") or []:
        if e.get("node") in ALARM_NODE_IDS:
            assert "not a prediction error" in e["meaning"], e["node"]
            assert "orion.substrate.endogenous_curiosity:_prediction_error_candidates" not in (
                e.get("consumers") or []
            ), e["node"]


@pytest.mark.parametrize("node_id", ALARMS)
def test_labels_say_alarm_not_prediction_error(node_id: str) -> None:
    label = node_signal_label(node_id, node_id.removeprefix("node:substrate."))
    assert "prediction error" not in label
    assert is_alarm_node(node_id)
    assert node_signal_label(LIVE, "execution") == "execution prediction error"
    assert not is_alarm_node(LIVE) and not is_alarm_node(None)


def _node(node_id: str, value: float) -> SimpleNamespace:
    return SimpleNamespace(
        node_id=node_id,
        metadata={"prediction_error": value},
        temporal=SimpleNamespace(observed_at=NOW - timedelta(seconds=10)),
    )


def test_curiosity_prediction_error_seeds_skip_alarms() -> None:
    """Live 7 days to 2026-11-13: strain seeded 56 and cabinet 6 candidates as
    "sustained prediction error". An alarm reaches curiosity through the
    attention open loop instead."""
    from orion.substrate.endogenous_curiosity import _prediction_error_candidates

    nodes = [_node(nid, 1.0) for nid in ALARMS] + [_node(LIVE, 1.0)]
    out = _prediction_error_candidates(
        nodes, anchor_scope="orion", subject_ref=None, min_error=0.1, now=NOW
    )
    assert {ref for c in out for ref in c.focal_node_refs} == {LIVE}


def test_alarms_keep_their_attention_path() -> None:
    """Not fenced from the contests, still seed dynamics pressure (the cabinet's
    documented route into the workspace), still judged as triggers."""
    from orion.attention.world_first import node_prediction_error_semantics
    from orion.schemas.attention_candidate import world_first_fenced
    from orion.substrate.pressure import PressureConfig, prediction_error_pressure

    cfg = PressureConfig()
    for nid in ALARMS:
        assert not world_first_fenced(nid)
        assert prediction_error_pressure(_node(nid, 1.0), cfg, now=NOW) > 0.0
        assert node_prediction_error_semantics(nid)[0] == "trigger"


def test_alarms_are_not_self_model_or_native_goal_domains() -> None:
    from orion.attention.field_attention.selectors import PREDICTION_ERROR_NATIVE_TARGETS
    from orion.substrate.attention_self_model import ACTIVE_INFERENCE_DOMAINS

    assert not ALARM_NODE_IDS & set(PREDICTION_ERROR_NATIVE_TARGETS)
    assert not {n.removeprefix("node:substrate.") for n in ALARM_NODE_IDS} & set(
        ACTIVE_INFERENCE_DOMAINS
    )


def test_self_reading_cites_an_alarm_as_alarm() -> None:
    from orion.attention.pe_history_cache import NodePeHistoryCache
    from orion.attention import self_readings as sr

    rows = [("node:substrate.cabinet", NOW - timedelta(minutes=m), 0.0) for m in range(2, 7 * 24 * 60, 30)]
    for start in (60 * 24, 60 * 48, 60 * 72, 60 * 96, 60 * 120, 60 * 144):
        rows.append(("node:substrate.cabinet", NOW - timedelta(minutes=start) + timedelta(seconds=1), 0.7))
    rows.append(("node:substrate.cabinet", NOW - timedelta(seconds=20), 0.7))
    rows.sort(key=lambda r: r[1])
    cache = NodePeHistoryCache()
    cache.refresh(lambda since: [r for r in rows if r[1] >= since], now=NOW)
    (reading,) = sr.build_self_readings(cache, now=NOW, nodes=["node:substrate.cabinet"])
    assert reading.ref.startswith("alarm:cabinet@"), reading.ref


def test_broadcast_signal_for_an_alarm_says_alarm() -> None:
    import orion.substrate.attention_broadcast as ab
    from orion.substrate.prediction_error_magnitude import compute_prediction_error_magnitude

    nid = "node:substrate.strain"
    hist = [(NOW - timedelta(minutes=15 * i), 0.0) for i in range(1, 672)]
    hist.append((NOW - timedelta(seconds=5), 1.0))
    mag = compute_prediction_error_magnitude(
        value=1.0, observed_at=NOW - timedelta(seconds=5), history=sorted(hist), now=NOW
    )
    node = SimpleNamespace(
        node_id=nid, label="Strain news", node_kind="concept",
        metadata={"prediction_error": 1.0, "dynamic_pressure": 0.5},
        signals=SimpleNamespace(confidence=1.0),
        temporal=SimpleNamespace(observed_at=NOW - timedelta(seconds=5)),
    )
    signals, _trace, _scores = ab.world_first_signals(
        [node], magnitude_by_node_id={nid: mag}, external_candidates=[], now=NOW,
    )
    sig = next(s for s in signals if nid in s.evidence_refs)
    assert sig.signal_kind == "substrate_alarm"
    assert sig.provenance["signal_driver"] == "alarm"
