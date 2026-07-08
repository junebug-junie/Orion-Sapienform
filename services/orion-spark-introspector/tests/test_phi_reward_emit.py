from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import numpy as np
import pytest

import app.worker as worker
from app.inner_state import COGNITIVE_FEATURE_NAMES, FELT_DIMENSIONS
from app.substrate_reads import ExecutionTrajectorySnapshot, GrammarTruthSnapshot
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.phi_encoder import (
    CorpusStatsV1,
    PhiEncoderManifestV1,
    PhiIntrinsicRewardV1,
    TrainingStatsV1,
)
from test_inner_state_emit import (
    _NOW,
    _mock_healthy_substrate_reads,
    _self_state_payload,
)


def _felt_input_features() -> list[str]:
    # seed-v2 always emits cognitive slots (zeroed when trajectory dark).
    return list(FELT_DIMENSIONS) + ["overall_intensity"] + list(COGNITIVE_FEATURE_NAMES)


def _write_tiny_encoder(tmp_path) -> None:
    feats = _felt_input_features()
    manifest = PhiEncoderManifestV1(
        encoder_id="test",
        encoder_version="v0",
        status="candidate",
        architecture="mlp_shallow_v1",
        features_version="seed-v2",
        input_features=feats,
        hidden_dim=4,
        latent_dim=2,
        corpus=CorpusStatsV1(corpus_path="x", row_count=1, excluded_degenerate=0),
        training=TrainingStatsV1(
            epochs=1,
            final_loss=0.1,
            held_out_loss=0.1,
            recon_error_p50=0.1,
            recon_error_p95=0.2,
        ),
        git_sha="test",
        trained_at=datetime.now(timezone.utc),
    )
    d_in, h, d_lat = len(feats), 4, 2
    np.savez(
        tmp_path / "weights.npz",
        W1=np.random.randn(d_in, h).astype(np.float64) * 0.01,
        b1=np.zeros(h),
        W2=np.random.randn(h, d_lat).astype(np.float64) * 0.01,
        b2=np.zeros(d_lat),
        W3=np.random.randn(d_lat, d_in).astype(np.float64) * 0.01,
        b3=np.zeros(d_in),
        w_phi=np.array([0.1, 0.2]),
        b_phi=np.array(0.0),
    )
    (tmp_path / "manifest.json").write_text(manifest.model_dump_json())


def _reset_inner_state(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(worker.settings, "inner_features_corpus_path", str(tmp_path / "c.jsonl"), raising=False)
    monkeypatch.setattr(worker, "_INNER_SCALER", worker._new_inner_scaler(), raising=False)
    monkeypatch.setattr(worker, "_INNER_PREV_FELT", None, raising=False)
    monkeypatch.setattr(worker, "_INNER_PREV_HEADLINE", None, raising=False)
    monkeypatch.setattr(worker, "_INNER_DEGENERATE_STREAK", 0, raising=False)
    monkeypatch.setattr(worker, "_INNER_LAST_HEADLINE", None, raising=False)
    monkeypatch.setattr(worker, "_PHI_ENCODER", None, raising=False)
    monkeypatch.setattr(worker, "_PHI_PREV_PHI", None, raising=False)
    monkeypatch.setattr(worker, "_PHI_PREV_RECON", None, raising=False)
    monkeypatch.setattr(worker, "_SUBSTRATE_CACHE", None, raising=False)


def _envelope() -> BaseEnvelope:
    return BaseEnvelope(
        kind="substrate.self_state.v1",
        source=ServiceRef(name="substrate-runtime", node="athena"),
        payload=_self_state_payload(),
    )


@pytest.mark.asyncio
async def test_phi_reward_emitted_when_encoder_ok(monkeypatch, tmp_path) -> None:
    published = []

    class _Bus:
        enabled = True

        async def publish(self, channel, env):
            published.append((channel, env))

    monkeypatch.setattr(worker.settings, "orion_phi_encoder_enabled", True, raising=False)
    monkeypatch.setattr(worker.settings, "orion_phi_encoder_weights", str(tmp_path), raising=False)
    _write_tiny_encoder(tmp_path)
    monkeypatch.setattr(worker, "_pub_bus", _Bus(), raising=False)
    monkeypatch.setattr(worker.manager, "broadcast", AsyncMock(), raising=False)
    _reset_inner_state(monkeypatch, tmp_path)
    _mock_healthy_substrate_reads(monkeypatch)

    await worker.handle_self_state(_envelope())

    reward_channels = [c for c, _ in published if c == worker.settings.channel_phi_reward]
    assert reward_channels, "expected phi_reward publish"

    reward_env = next(env for c, env in published if c == worker.settings.channel_phi_reward)
    reward = PhiIntrinsicRewardV1.model_validate(reward_env.payload)
    assert 0.0 <= reward.phi <= 1.0
    assert reward.delta_phi is not None
    assert reward.encoder_version == "v0"
    assert reward.features_version == "seed-v2"
    assert reward.phi_health == "ok"
    assert reward.grammar_truth_degraded is False


@pytest.mark.asyncio
async def test_phi_reward_suppressed_when_frozen(monkeypatch, tmp_path) -> None:
    published = []

    class _Bus:
        enabled = True

        async def publish(self, channel, env):
            published.append(channel)

    monkeypatch.setattr(worker.settings, "orion_phi_encoder_enabled", True, raising=False)
    monkeypatch.setattr(worker.settings, "orion_phi_encoder_weights", str(tmp_path), raising=False)
    _write_tiny_encoder(tmp_path)
    monkeypatch.setattr(worker, "_pub_bus", _Bus(), raising=False)
    monkeypatch.setattr(worker.manager, "broadcast", AsyncMock(), raising=False)
    _reset_inner_state(monkeypatch, tmp_path)
    monkeypatch.setattr(worker, "_SUBSTRATE_CACHE", None, raising=False)
    monkeypatch.setattr(
        worker,
        "fetch_grammar_truth",
        AsyncMock(
            return_value=GrammarTruthSnapshot(
                degraded=True,
                degraded_reasons=["cursor_lag:execution_grammar_reducer"],
                enabled_reducers={"execution_trajectory": True},
                reducer_health_by_name={},
            )
        ),
    )
    monkeypatch.setattr(
        worker,
        "fetch_execution_trajectory",
        AsyncMock(return_value=ExecutionTrajectorySnapshot(ok=False, projection=None)),
    )

    await worker.handle_self_state(_envelope())

    assert worker.settings.channel_phi_reward not in published
