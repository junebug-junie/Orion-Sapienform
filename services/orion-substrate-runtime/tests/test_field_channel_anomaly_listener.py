"""Unit tests for the field-channel-anomaly bus listener and the
prediction_error_by_domain extractor added to wire real
prediction_error_confidence / field_anomaly into brain frames.

Constructs the worker via __new__, mirroring test_embodiment_c_hook.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from app.worker import BiometricsSubstrateWorker

from orion.schemas.telemetry.field_channel_anomaly_score import FieldChannelAnomalyScoreV1


def _make_worker(monkeypatch) -> BiometricsSubstrateWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = settings_mod.get_settings()
    worker._bus = MagicMock()
    worker._latest_field_anomaly = None
    worker._latest_field_anomaly_at = None
    return worker


def _score_payload(**overrides) -> dict:
    payload = {
        "correlation_id": "corr-1",
        "encoder_id": "mood-arc-encoder:v3",
        "encoder_version": "v3",
        "recon_loss": 0.012,
        "recon_error_p95": 0.0003,
        "threshold_multiplier": 3.0,
        "threshold": 0.0009,
        "anomalous": True,
        "window_start": "2026-07-28T20:00:00+00:00",
        "window_end": "2026-07-28T20:01:00+00:00",
        "window_size": 30,
    }
    payload.update(overrides)
    return payload


def test_cache_field_channel_anomaly_stores_valid_score(monkeypatch):
    worker = _make_worker(monkeypatch)
    bus = MagicMock()
    bus.codec.decode.return_value = MagicMock(
        ok=True, envelope=SimpleNamespace(payload=_score_payload())
    )
    worker._bus = bus

    worker._cache_field_channel_anomaly_message({"data": b"irrelevant, decode is mocked"})

    assert isinstance(worker._latest_field_anomaly, FieldChannelAnomalyScoreV1)
    assert worker._latest_field_anomaly.recon_loss == 0.012
    assert worker._latest_field_anomaly_at is not None


def test_cache_field_channel_anomaly_fails_open_on_bad_decode(monkeypatch):
    worker = _make_worker(monkeypatch)
    bus = MagicMock()
    bus.codec.decode.return_value = MagicMock(ok=False, error="boom")
    worker._bus = bus

    worker._cache_field_channel_anomaly_message({"data": b"garbage"})
    assert worker._latest_field_anomaly is None


def test_cache_field_channel_anomaly_fails_open_on_invalid_payload(monkeypatch):
    worker = _make_worker(monkeypatch)
    bus = MagicMock()
    bus.codec.decode.return_value = MagicMock(
        ok=True, envelope=SimpleNamespace(payload={"not": "a valid score"})
    )
    worker._bus = bus

    worker._cache_field_channel_anomaly_message({"data": b"irrelevant"})
    assert worker._latest_field_anomaly is None


def _node(node_id, prediction_error=None):
    return SimpleNamespace(node_id=node_id, metadata={"prediction_error": prediction_error})


def test_prediction_error_by_domain_reads_fixed_nodes(monkeypatch):
    worker = _make_worker(monkeypatch)
    nodes = [
        _node("node:substrate.execution", 0.001),
        _node("node:substrate.biometrics", 0.002),
        _node("node:substrate.bus_synaptic", 1.0),
        _node("node:substrate.harness_closure", 0.65),  # not a recognized domain
        _node("sub-concept-seed-orion", None),  # unrelated node, no prediction_error
    ]
    out = worker._brain_frame_prediction_error_by_domain(nodes)
    assert out == {"execution": 0.001, "biometrics": 0.002, "bus_synaptic": 1.0}


def test_prediction_error_by_domain_empty_when_no_nodes(monkeypatch):
    worker = _make_worker(monkeypatch)
    assert worker._brain_frame_prediction_error_by_domain([]) == {}


def test_prediction_error_by_domain_skips_missing_values(monkeypatch):
    worker = _make_worker(monkeypatch)
    nodes = [_node("node:substrate.execution", None)]
    assert worker._brain_frame_prediction_error_by_domain(nodes) == {}
