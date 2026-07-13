from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

import app.worker as worker
from orion.schemas.telemetry.mood_arc import MoodArcCorpusRowV1
from test_inner_state_emit import _mock_healthy_substrate_reads, _self_state_payload
from test_phi_reward_emit import _envelope, _reset_inner_state, _write_tiny_encoder


class _Bus:
    enabled = True

    def __init__(self) -> None:
        self.published: dict[str, object] = {}

    async def publish(self, channel, env):
        self.published[channel] = env


def _read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip("\n").splitlines()
    return [json.loads(line) for line in lines if line]


@pytest.mark.asyncio
async def test_mood_arc_corpus_appends_on_proxy_source(monkeypatch, tmp_path) -> None:
    corpus_path = tmp_path / "mood_arc.jsonl"

    monkeypatch.setattr(worker.settings, "orion_phi_encoder_enabled", True, raising=False)
    monkeypatch.setattr(worker.settings, "orion_phi_encoder_weights", str(tmp_path), raising=False)
    monkeypatch.setattr(worker.settings, "inner_features_version", "seed-v2", raising=False)
    monkeypatch.setattr(worker.settings, "mood_arc_corpus_path", str(corpus_path), raising=False)
    monkeypatch.setattr(worker, "_MOOD_ARC_SINK", worker.InnerStateCorpusSink(str(corpus_path)), raising=False)

    # Nonzero, asymmetric probes -> golden_overrides carries "valence" ->
    # valence_source == "proxy" (mirrors test_phi_reward_emit.py's own
    # proxy-source setup).
    probes = {
        "z0": {"agency_readiness": 0.68},
        "z1": {"agency_readiness": -0.45},
    }
    _write_tiny_encoder(tmp_path, probes=probes, encoder_id="test-probes", encoder_version="v0-probes")

    bus = _Bus()
    monkeypatch.setattr(worker, "_pub_bus", bus, raising=False)
    monkeypatch.setattr(worker.manager, "broadcast", AsyncMock(), raising=False)
    _reset_inner_state(monkeypatch, tmp_path)
    _mock_healthy_substrate_reads(monkeypatch)

    await worker.handle_self_state(_envelope())

    rows = _read_jsonl(corpus_path)
    assert len(rows) == 1

    snap_env = bus.published[worker.settings.channel_spark_state_snapshot]
    snap_payload = snap_env.payload
    phi_dict = snap_payload.phi if hasattr(snap_payload, "phi") else snap_payload["phi"]
    valence_source = snap_payload.metadata["valence_source"]
    dominant_node = (
        snap_payload.dominant_node if hasattr(snap_payload, "dominant_node") else snap_payload["dominant_node"]
    )

    assert valence_source == "proxy"
    row = rows[0]
    assert row["coherence"] == phi_dict["coherence"]
    assert row["energy"] == phi_dict["energy"]
    assert row["novelty"] == phi_dict["novelty"]
    assert row["valence"] == phi_dict["valence"]
    assert row["valence_source"] == "proxy"
    assert row["dominant_node"] == dominant_node


@pytest.mark.asyncio
async def test_mood_arc_corpus_appends_on_heuristic_source(monkeypatch, tmp_path) -> None:
    # Fallback ticks (no encoder configured -> valence_source == "heuristic")
    # are still real, clean data and must not be excluded -- excluding them
    # would bias the corpus toward encoder-healthy periods only.
    corpus_path = tmp_path / "mood_arc.jsonl"

    monkeypatch.setattr(worker.settings, "orion_phi_encoder_enabled", False, raising=False)
    monkeypatch.setattr(worker.settings, "mood_arc_corpus_path", str(corpus_path), raising=False)
    monkeypatch.setattr(worker, "_MOOD_ARC_SINK", worker.InnerStateCorpusSink(str(corpus_path)), raising=False)

    bus = _Bus()
    monkeypatch.setattr(worker, "_pub_bus", bus, raising=False)
    monkeypatch.setattr(worker.manager, "broadcast", AsyncMock(), raising=False)
    _reset_inner_state(monkeypatch, tmp_path)
    _mock_healthy_substrate_reads(monkeypatch)

    await worker.handle_self_state(_envelope())

    rows = _read_jsonl(corpus_path)
    assert len(rows) == 1, "fallback (heuristic) ticks must still be appended -- not excluded"
    assert rows[0]["valence_source"] == "heuristic"

    snap_env = bus.published[worker.settings.channel_spark_state_snapshot]
    snap_payload = snap_env.payload
    assert snap_payload.metadata["valence_source"] == "heuristic"


@pytest.mark.asyncio
async def test_mood_arc_corpus_appends_even_when_pub_bus_disabled(monkeypatch, tmp_path) -> None:
    # 2026-07-13, found by code review: the append call originally sat
    # after the `if not (_pub_bus and _pub_bus.enabled): return` gate,
    # silently coupling this training-data sink to bus health for no
    # reason -- a bus outage would fragment the corpus into gap-broken
    # windows with no warning. Moved before that gate; this regression
    # test pins the fix by disabling the bus entirely and confirming a
    # row is still written.
    corpus_path = tmp_path / "mood_arc.jsonl"

    monkeypatch.setattr(worker.settings, "orion_phi_encoder_enabled", False, raising=False)
    monkeypatch.setattr(worker.settings, "mood_arc_corpus_path", str(corpus_path), raising=False)
    monkeypatch.setattr(worker, "_MOOD_ARC_SINK", worker.InnerStateCorpusSink(str(corpus_path)), raising=False)

    class _DisabledBus:
        enabled = False

        async def publish(self, channel, env):
            raise AssertionError("must not publish when disabled")

    monkeypatch.setattr(worker, "_pub_bus", _DisabledBus(), raising=False)
    monkeypatch.setattr(worker.manager, "broadcast", AsyncMock(), raising=False)
    _reset_inner_state(monkeypatch, tmp_path)
    _mock_healthy_substrate_reads(monkeypatch)

    await worker.handle_self_state(_envelope())

    rows = _read_jsonl(corpus_path)
    assert len(rows) == 1, "mood-arc collection must not depend on _pub_bus.enabled"
    assert rows[0]["valence_source"] == "heuristic"


@pytest.mark.asyncio
async def test_mood_arc_corpus_construction_failure_does_not_abort_tick(monkeypatch, tmp_path) -> None:
    # 2026-07-13, found by code review: the original `except OSError` only
    # wrapped the file write, not MoodArcCorpusRowV1(...) construction --
    # a pydantic ValidationError (or any other non-OSError exception) would
    # propagate past it and abort the rest of handle_self_state, including
    # the real, consumed spark_state_snapshot publish further down. Fixed
    # via `except Exception`; this test forces a construction-time failure
    # and confirms the snapshot still publishes.
    corpus_path = tmp_path / "mood_arc.jsonl"

    monkeypatch.setattr(worker.settings, "orion_phi_encoder_enabled", False, raising=False)
    monkeypatch.setattr(worker.settings, "mood_arc_corpus_path", str(corpus_path), raising=False)
    monkeypatch.setattr(worker, "_MOOD_ARC_SINK", worker.InnerStateCorpusSink(str(corpus_path)), raising=False)

    def _raise(*args, **kwargs):
        raise ValueError("simulated corpus-row construction failure")

    monkeypatch.setattr(worker, "MoodArcCorpusRowV1", _raise, raising=False)

    bus = _Bus()
    monkeypatch.setattr(worker, "_pub_bus", bus, raising=False)
    monkeypatch.setattr(worker.manager, "broadcast", AsyncMock(), raising=False)
    _reset_inner_state(monkeypatch, tmp_path)
    _mock_healthy_substrate_reads(monkeypatch)

    await worker.handle_self_state(_envelope())

    assert not corpus_path.exists(), "the simulated failure should have prevented any write"
    assert worker.settings.channel_spark_state_snapshot in bus.published, (
        "a mood-arc corpus construction failure must not abort the real, "
        "consumed spark_state_snapshot publish"
    )


@pytest.mark.asyncio
async def test_mood_arc_corpus_disabled_when_path_empty(monkeypatch, tmp_path) -> None:
    # mood_arc_corpus_path == "" (the default) -> no-op, no file created, no raise.
    corpus_path = tmp_path / "mood_arc.jsonl"

    monkeypatch.setattr(worker.settings, "orion_phi_encoder_enabled", False, raising=False)
    monkeypatch.setattr(worker.settings, "mood_arc_corpus_path", "", raising=False)
    monkeypatch.setattr(worker, "_MOOD_ARC_SINK", worker.InnerStateCorpusSink(""), raising=False)

    bus = _Bus()
    monkeypatch.setattr(worker, "_pub_bus", bus, raising=False)
    monkeypatch.setattr(worker.manager, "broadcast", AsyncMock(), raising=False)
    _reset_inner_state(monkeypatch, tmp_path)
    _mock_healthy_substrate_reads(monkeypatch)

    await worker.handle_self_state(_envelope())

    assert not corpus_path.exists()


def test_mood_arc_corpus_row_schema_matches_phi_now_keys() -> None:
    ss = worker.SelfStateV1.model_validate(_self_state_payload())
    phi_now = worker._phi_from_self_state(ss)

    assert set(phi_now.keys()) >= {"coherence", "energy", "novelty", "valence"}

    row_float_fields = {
        name
        for name, field in MoodArcCorpusRowV1.model_fields.items()
        if field.annotation is float
    }
    assert row_float_fields == {"coherence", "energy", "novelty", "valence"}
