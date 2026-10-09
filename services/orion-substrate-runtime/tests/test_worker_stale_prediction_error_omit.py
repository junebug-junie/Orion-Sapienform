"""Stale prediction_error readings are OMITTED from the self-model aggregate
(not faded -- a faded value reads as calm). Spec breadcrumbs:
docs/superpowers/specs/2026-10-02-selfmodel-stale-reading-omit-breadcrumbs.md
"""

from __future__ import annotations

import sys
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, SUBSTRATE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from app.worker import BiometricsSubstrateWorker
from orion.substrate.attention_self_model import reduce_attention_self_model

NOW = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
DOMAINS = ["execution", "biometrics", "chat", "route", "bus_synaptic"]


def _node(domain, value, age_sec, *, with_temporal=True):
    ns = SimpleNamespace(node_id=f"node:substrate.{domain}", metadata={"prediction_error": value})
    if with_temporal:
        ns.temporal = SimpleNamespace(observed_at=NOW - timedelta(seconds=age_sec))
    return ns


def _worker(*, omit=True):
    w = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    w._settings = SimpleNamespace(attention_self_model_omit_stale_pe=omit)
    return w


def _nodes(chat_age=7200, **kw):
    vals = {"execution": 0.1, "biometrics": 0.2, "chat": 0.834, "route": 0.0, "bus_synaptic": 0.3}
    return [_node(d, v, chat_age if d == "chat" else 5, **kw) for d, v in vals.items()]


def _conf(pe, omitted=None):
    return reduce_attention_self_model(
        None, None, now=NOW, prediction_error_by_domain=pe or None,
        prediction_error_omitted_by_domain=omitted or None,
    )


def test_stale_chat_omitted_and_mean_is_four_domain_mean():
    pe, ev, omitted = _worker()._fresh_prediction_error_and_evidence_by_domain(_nodes(), now=NOW)
    assert "chat" not in pe and set(pe) == {"execution", "biometrics", "route", "bus_synaptic"}
    assert omitted == {"chat": pytest.approx(7200)}
    m = _conf(pe, omitted)
    assert m.prediction_error_confidence == pytest.approx(1 - (0.1 + 0.2 + 0.0 + 0.3) / 4, abs=1e-4)
    assert "from 4 of 5 domains" in m.prediction_error_confidence_basis
    assert "chat(7200s)" in m.prediction_error_confidence_basis
    assert "chat" not in m.prediction_error_by_domain


def test_fresh_domains_unchanged():
    pe, _, omitted = _worker()._fresh_prediction_error_and_evidence_by_domain(_nodes(chat_age=10), now=NOW)
    assert omitted == {} and pe["chat"] == 0.834 and len(pe) == 5
    m = _conf(pe)
    assert "from" not in m.prediction_error_confidence_basis


def test_age_at_horizon_is_fresh_just_past_is_stale():
    w = _worker()
    _, _, o1 = w._fresh_prediction_error_and_evidence_by_domain(_nodes(chat_age=1800), now=NOW)
    _, _, o2 = w._fresh_prediction_error_and_evidence_by_domain(_nodes(chat_age=1801), now=NOW)
    assert o1 == {} and "chat" in o2


def test_all_omitted_is_unknown_not_zero_or_one():
    nodes = [_node(d, 0.5, 9999) for d in DOMAINS]
    pe, _, omitted = _worker()._fresh_prediction_error_and_evidence_by_domain(nodes, now=NOW)
    assert pe == {} and len(omitted) == 5
    m = _conf(pe, omitted)
    assert m.prediction_error_confidence is None
    assert m.prediction_error_confidence_basis.startswith("no current reading")
    assert "omitted 5 of 5" in m.prediction_error_confidence_basis


def test_missing_observed_at_omitted_and_traced_as_unknown_age():
    nodes = _nodes(chat_age=5)
    nodes[2] = _node("chat", 0.834, 0, with_temporal=False)
    pe, _, omitted = _worker()._fresh_prediction_error_and_evidence_by_domain(nodes, now=NOW)
    assert omitted == {"chat": None} and "chat" not in pe
    assert "chat(age unknown)" in _conf(pe, omitted).prediction_error_confidence_basis


def test_unparseable_observed_at_string_omitted():
    n = _node("chat", 0.5, 0)
    n.temporal = SimpleNamespace(observed_at="garbage")
    _, _, omitted = _worker()._fresh_prediction_error_and_evidence_by_domain([n], now=NOW)
    assert omitted == {"chat": None}


def test_setting_off_restores_old_behavior():
    w = _worker(omit=False)
    pe, ev, omitted = w._fresh_prediction_error_and_evidence_by_domain(_nodes(), now=NOW)
    old_pe, old_ev = w._brain_frame_prediction_error_and_evidence_by_domain(_nodes())
    assert omitted == {} and pe == old_pe and ev == old_ev and pe["chat"] == 0.834


def test_stored_node_values_not_mutated():
    nodes = _nodes()
    _worker()._fresh_prediction_error_and_evidence_by_domain(nodes, now=NOW)
    assert nodes[2].metadata["prediction_error"] == 0.834


def test_trend_buffer_not_reappended_with_stale_value(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("SUBSTRATE_ATTENTION_SELF_MODEL_TICK_ENABLED", "true")
    monkeypatch.setenv("ORION_ATTENTION_BROADCAST_ENABLED", "true")
    import app.settings as settings_mod

    settings_mod._settings = None
    w = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    w._settings = settings_mod.get_settings()
    w._substrate_graph_store = None
    w._store = MagicMock()
    w._store.get_latest_field_attention_frame.return_value = None
    w._attention_self_model_trend_buffer = deque(maxlen=4)
    fresh = datetime.now(timezone.utc)
    nodes = {}
    for d, v, age in [("biometrics", 0.4, 1), ("chat", 0.834, 7200)]:
        nodes[f"node:substrate.{d}"] = SimpleNamespace(
            node_id=f"node:substrate.{d}", metadata={"prediction_error": v},
            temporal=SimpleNamespace(observed_at=fresh - timedelta(seconds=age)),
        )
    store = MagicMock()
    store.snapshot.return_value = SimpleNamespace(nodes=nodes)
    with patch("orion.substrate.graphdb_store.build_substrate_store_from_env", return_value=store):
        w._attention_broadcast_tick()
        w._attention_broadcast_tick()
    assert all("chat" not in snap for snap in w._attention_self_model_trend_buffer)
    assert len(w._attention_self_model_trend_buffer) == 2
    model = w._store.save_attention_self_model.call_args.args[0]
    assert "chat" not in model.prediction_error_by_domain


def test_staleness_horizon_is_the_shared_substrate_one():
    from orion.substrate.endogenous_curiosity import _PREDICTION_ERROR_DECAY_HORIZON_SECONDS
    from orion.substrate.prediction_error_freshness import PE_STALENESS_HORIZON_SEC

    assert PE_STALENESS_HORIZON_SEC == _PREDICTION_ERROR_DECAY_HORIZON_SECONDS == 1800


def test_persisted_row_with_omission_parses_in_equilibrium_reader():
    sys.path.insert(0, str(REPO_ROOT / "services" / "orion-equilibrium-service"))
    from app.attention_self_model_reader import parse_confidence_samples

    pe, _, omitted = _worker()._fresh_prediction_error_and_evidence_by_domain(_nodes(), now=NOW)
    ok = _conf(pe, omitted).model_dump(mode="json")
    nodes = [_node(d, 0.5, 9999) for d in DOMAINS]
    pe2, _, om2 = _worker()._fresh_prediction_error_and_evidence_by_domain(nodes, now=NOW)
    none_row = _conf(pe2, om2).model_dump(mode="json")
    samples = parse_confidence_samples([(ok, NOW), (none_row, NOW + timedelta(seconds=30))])
    assert [round(s.value, 4) for s in samples] == [round(ok["prediction_error_confidence"], 4)]
