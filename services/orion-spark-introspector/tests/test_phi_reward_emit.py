from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import math

import numpy as np
import pytest

import app.worker as worker
from app.inner_state import COGNITIVE_FEATURE_NAMES
from app.substrate_reads import ExecutionTrajectorySnapshot, GrammarTruthSnapshot
from orion.schemas.telemetry.phi_encoder import (
    CorpusStatsV1,
    PhiEncoderManifestV1,
    PhiIntrinsicRewardV1,
    TrainingStatsV1,
)
from test_inner_state_emit import _mock_healthy_substrate_reads

_NOW = datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)


def _felt_input_features() -> list[str]:
    """2026-07-22 (SelfStateV1 burn): was FELT_DIMENSIONS + overall_intensity
    + COGNITIVE_FEATURE_NAMES -- SelfStateV1's FELT_DIMENSIONS no longer
    exist. These tests use features_version="seed-v2", which hits
    build_inner_state_features()'s non-seed-v4/v5 branch
    (cognitive_features_from_trajectory), whose only real output is the 4
    COGNITIVE_FEATURE_NAMES -- so that's the synthetic encoder's whole input
    space now."""
    return list(COGNITIVE_FEATURE_NAMES)


def _write_tiny_encoder(
    tmp_path,
    *,
    probes: dict[str, dict[str, float]] | None = None,
    encoder_id: str = "test",
    encoder_version: str = "v0",
) -> None:
    feats = _felt_input_features()
    manifest = PhiEncoderManifestV1(
        encoder_id=encoder_id,
        encoder_version=encoder_version,
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
        # Nonzero, asymmetric weights (when probes is passed) so the two
        # latent dims produce a distinguishable, non-degenerate proxy value.
        probes=probes or {},
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
    monkeypatch.setattr(worker, "_PHI_PREV_VALENCE_SOURCE", None, raising=False)
    monkeypatch.setattr(worker, "_SUBSTRATE_CACHE", None, raising=False)


@pytest.mark.asyncio
async def test_phi_reward_emitted_when_encoder_ok(monkeypatch, tmp_path) -> None:
    published = []

    class _Bus:
        enabled = True

        async def publish(self, channel, env):
            published.append((channel, env))

    monkeypatch.setattr(worker.settings, "orion_phi_encoder_enabled", True, raising=False)
    monkeypatch.setattr(worker.settings, "orion_phi_encoder_weights", str(tmp_path), raising=False)
    monkeypatch.setattr(worker.settings, "inner_features_version", "seed-v2", raising=False)
    _write_tiny_encoder(tmp_path)
    monkeypatch.setattr(worker, "_pub_bus", _Bus(), raising=False)
    monkeypatch.setattr(worker.manager, "broadcast", AsyncMock(), raising=False)
    _reset_inner_state(monkeypatch, tmp_path)
    _mock_healthy_substrate_reads(monkeypatch)

    await worker.run_inner_state_tick()

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

    await worker.run_inner_state_tick()

    assert worker.settings.channel_phi_reward not in published


@pytest.mark.asyncio
async def test_golden_phi_overrides_coherence_energy_novelty_valence_falls_back_without_probes(
    monkeypatch, tmp_path
) -> None:
    # 2026-07-12: the trained encoder's output must replace the heuristic
    # baseline (TISSUE.phi()-based since the 2026-07-22 SelfStateV1 burn) for
    # coherence/energy/novelty wherever it reaches orion-cortex-exec's
    # metacognition prompts (via SparkStateSnapshotV1.phi -> spark_narrative.py).
    # 2026-07-13: valence now also has a trained-probe override
    # (_agency_valence_proxy), but only when the loaded manifest actually
    # carries a probes map. _write_tiny_encoder's manifest below has no
    # probes (default_factory=dict -> {}), so this test still exercises the
    # fallback path -- valence stays on the heuristic. See
    # test_golden_phi_overrides_valence_uses_agency_probe below for the
    # probe-present path.
    published: dict[str, object] = {}

    class _Bus:
        enabled = True

        async def publish(self, channel, env):
            published[channel] = env

    monkeypatch.setattr(worker.settings, "orion_phi_encoder_enabled", True, raising=False)
    monkeypatch.setattr(worker.settings, "orion_phi_encoder_weights", str(tmp_path), raising=False)
    monkeypatch.setattr(worker.settings, "inner_features_version", "seed-v2", raising=False)
    _write_tiny_encoder(tmp_path)
    monkeypatch.setattr(worker, "_pub_bus", _Bus(), raising=False)
    monkeypatch.setattr(worker.manager, "broadcast", AsyncMock(), raising=False)
    _reset_inner_state(monkeypatch, tmp_path)
    # A known, nonzero previous-tick phi so delta_phi (and therefore the
    # "energy" mapping) is actually exercised -- _reset_inner_state alone
    # leaves _PHI_PREV_PHI=None, which forces delta_phi=0.0 by the cold-start
    # convention and would let a broken energy mapping pass trivially
    # (found by code review, 2026-07-12).
    monkeypatch.setattr(worker, "_PHI_PREV_PHI", 0.2, raising=False)
    _mock_healthy_substrate_reads(monkeypatch)

    heuristic_valence = worker._get_phi_stats()["valence"]

    await worker.run_inner_state_tick()

    reward_env = published[worker.settings.channel_phi_reward]
    reward = PhiIntrinsicRewardV1.model_validate(reward_env.payload)
    assert reward.delta_phi != 0.0, "test setup must exercise a nonzero delta_phi"

    snap_env = published[worker.settings.channel_spark_state_snapshot]
    snap_payload = snap_env.payload
    phi_dict = snap_payload.phi if hasattr(snap_payload, "phi") else snap_payload["phi"]

    assert phi_dict["coherence"] == round(reward.phi, 4)
    assert phi_dict["energy"] == round(min(1.0, abs(reward.delta_phi)), 4)
    expected_novelty = round(min(1.0, reward.recon_error / max(0.2, 1e-6)), 4)
    assert phi_dict["novelty"] == expected_novelty
    # No probes on this manifest -> _agency_valence_proxy returns None ->
    # valence falls back to the heuristic's value, not overridden or zeroed.
    assert phi_dict["valence"] == heuristic_valence


def test_agency_valence_proxy_none_without_probes() -> None:
    assert worker._agency_valence_proxy({"z0": 0.5}, {}) is None
    assert worker._agency_valence_proxy({"z0": 0.5}, None) is None
    assert worker._agency_valence_proxy(None, {"z0": {"agency_readiness": 0.6}}) is None


def test_agency_valence_proxy_zero_weight_returns_none() -> None:
    # Every latent's agency_readiness weight is exactly 0 (or absent) --
    # weight_total is 0, must not divide by zero.
    latent = {"z0": 0.9, "z1": -0.4}
    probes = {"z0": {"agency_readiness": 0.0}, "z1": {"coherence": 0.5}}
    assert worker._agency_valence_proxy(latent, probes) is None


def test_agency_valence_proxy_noise_level_weight_returns_none() -> None:
    # 2026-07-13, found by code review: a single noise-level residual weight
    # (e.g. 0.0001, well above a naive near-zero guard but not real signal)
    # combined with a large latent activation must not saturate tanh to a
    # maximally-confident +-1 on statistically meaningless evidence. Real
    # agency_readiness correlations in the active manifest are 0.35-0.69;
    # weight_total=0.0001 is far below the 0.05 floor.
    latent = {"z0": 500.0}
    probes = {"z0": {"agency_readiness": 0.0001}}
    assert worker._agency_valence_proxy(latent, probes) is None


def test_agency_valence_proxy_ignores_nonfinite_latent_or_weight() -> None:
    # 2026-07-13, found by code review: a corrupted weights.npz or an
    # unbounded linear layer producing NaN/Inf in a latent activation (or,
    # defensively, a probe weight) must not propagate into a published bus
    # value.
    probes = {
        "z0": {"agency_readiness": 0.68},
        "z1": {"agency_readiness": 0.65},
    }
    latent_with_nan = {"z0": float("nan"), "z1": 1.0}
    result = worker._agency_valence_proxy(latent_with_nan, probes)
    assert result is not None
    assert math.isfinite(result)

    latent_with_inf = {"z0": float("inf"), "z1": 1.0}
    result_inf = worker._agency_valence_proxy(latent_with_inf, probes)
    assert result_inf is not None
    assert math.isfinite(result_inf)

    # Every latent is non-finite -> no usable evidence at all -> None, not NaN.
    all_nan = {"z0": float("nan"), "z1": float("nan")}
    assert worker._agency_valence_proxy(all_nan, probes) is None


def test_agency_valence_proxy_matches_real_encoder_probes() -> None:
    # Weight literals copied from the active v20260712-seedv4-postfix
    # manifest's probes.json as of 2026-07-13 (see inner_state_registry.py's
    # phi_heuristic.valence entry) -- not re-read from that live file here,
    # so this only pins the algorithm's math against a known-real snapshot;
    # it does not re-verify the snapshot is still current. A strong positive
    # agency signal across the latent space should land close to +1 after
    # the tanh squash, not near 0.
    probes = {
        "z0": {"agency_readiness": 0.6858},
        "z1": {"agency_readiness": -0.4526},
        "z3": {"agency_readiness": 0.6773},
        "z7": {"agency_readiness": 0.6188},
    }
    latent = {"z0": 2.0, "z1": -2.0, "z3": 2.0, "z7": 2.0}
    result = worker._agency_valence_proxy(latent, probes)
    assert result is not None
    assert result > 0.9

    # Flip the sign of every latent activation -> flips the proxy's sign too.
    flipped = {k: -v for k, v in latent.items()}
    flipped_result = worker._agency_valence_proxy(flipped, probes)
    assert flipped_result is not None
    assert flipped_result < -0.9


@pytest.mark.asyncio
async def test_golden_phi_overrides_valence_uses_agency_probe(monkeypatch, tmp_path) -> None:
    # 2026-07-13: with a manifest carrying a real agency_readiness probe
    # column, the golden-phi override path must replace the heuristic
    # valence with _agency_valence_proxy's output, not silently keep the
    # heuristic (the bug this whole fix corrects: an unverified "no trained
    # analog" claim was treated as final instead of checked).
    published: dict[str, object] = {}

    class _Bus:
        enabled = True

        async def publish(self, channel, env):
            published[channel] = env

    monkeypatch.setattr(worker.settings, "orion_phi_encoder_enabled", True, raising=False)
    monkeypatch.setattr(worker.settings, "orion_phi_encoder_weights", str(tmp_path), raising=False)
    monkeypatch.setattr(worker.settings, "inner_features_version", "seed-v2", raising=False)

    # Nonzero, asymmetric weights so the two latent dims produce a
    # distinguishable, non-degenerate proxy value.
    probes = {
        "z0": {"agency_readiness": 0.68},
        "z1": {"agency_readiness": -0.45},
    }
    _write_tiny_encoder(tmp_path, probes=probes, encoder_id="test-probes", encoder_version="v0-probes")

    monkeypatch.setattr(worker, "_pub_bus", _Bus(), raising=False)
    monkeypatch.setattr(worker.manager, "broadcast", AsyncMock(), raising=False)
    _reset_inner_state(monkeypatch, tmp_path)
    _mock_healthy_substrate_reads(monkeypatch)

    heuristic_valence = worker._get_phi_stats()["valence"]

    await worker.run_inner_state_tick()

    reward_env = published[worker.settings.channel_phi_reward]
    reward = PhiIntrinsicRewardV1.model_validate(reward_env.payload)

    snap_env = published[worker.settings.channel_spark_state_snapshot]
    snap_payload = snap_env.payload
    phi_dict = snap_payload.phi if hasattr(snap_payload, "phi") else snap_payload["phi"]

    expected = worker._agency_valence_proxy(reward.latent, probes)
    assert expected is not None
    assert phi_dict["valence"] == expected
    assert phi_dict["valence"] != heuristic_valence, (
        "probe manifest present -- valence must not silently stay on the heuristic"
    )


@pytest.mark.asyncio
async def test_phi_prev_resets_across_a_skipped_tick(monkeypatch, tmp_path) -> None:
    # Regression for the gap-reset fix (found by code review, 2026-07-12): a
    # skipped tick (grammar-truth degraded here) between two healthy encoder
    # ticks must NOT leave delta_phi spanning both ticks on the next healthy
    # one -- _PHI_PREV_PHI must reset to None during the skip, so the next
    # healthy tick's delta_phi is a fresh 0.0 (cold-start convention), not
    # tick3.phi - tick1.phi.
    published: dict[str, object] = {}

    class _Bus:
        enabled = True

        async def publish(self, channel, env):
            published[channel] = env

    monkeypatch.setattr(worker.settings, "orion_phi_encoder_enabled", True, raising=False)
    monkeypatch.setattr(worker.settings, "orion_phi_encoder_weights", str(tmp_path), raising=False)
    monkeypatch.setattr(worker.settings, "inner_features_version", "seed-v2", raising=False)
    _write_tiny_encoder(tmp_path)
    monkeypatch.setattr(worker, "_pub_bus", _Bus(), raising=False)
    monkeypatch.setattr(worker.manager, "broadcast", AsyncMock(), raising=False)
    _reset_inner_state(monkeypatch, tmp_path)

    # Tick 1: healthy.
    _mock_healthy_substrate_reads(monkeypatch)
    await worker.run_inner_state_tick()
    assert worker._PHI_PREV_PHI is not None, "tick 1 should have set a baseline"

    # Tick 2: skipped (grammar-truth degraded) -- must reset the baseline.
    # _SUBSTRATE_CACHE must be cleared again here (not just at the top of the
    # test): tick 1 populated it with a "healthy" result via
    # _mock_healthy_substrate_reads, and _fetch_substrate_grammar_truth
    # serves from that cache before ever calling the (re-)mocked
    # fetch_grammar_truth below -- without this, tick 2 would silently reuse
    # tick 1's cached healthy read instead of the degraded one being set up.
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
    await worker.run_inner_state_tick()
    assert worker._PHI_PREV_PHI is None, "skipped tick must reset the baseline"

    # Tick 3: healthy again -- delta_phi must be a fresh 0.0, not spanning tick 2.
    _mock_healthy_substrate_reads(monkeypatch)
    await worker.run_inner_state_tick()

    reward_env = published[worker.settings.channel_phi_reward]
    reward = PhiIntrinsicRewardV1.model_validate(reward_env.payload)
    assert reward.delta_phi == 0.0


@pytest.mark.asyncio
async def test_turn_effect_valence_delta_suppressed_across_a_source_swap(monkeypatch, tmp_path) -> None:
    # 2026-07-13, found by code review (cross-file trace): valence can be
    # produced by two independently-computed, uncalibrated formulas (the
    # TISSUE.phi()-based heuristic vs _agency_valence_proxy) depending on
    # whether the loaded encoder manifest carries probes. A raw diff across a
    # tick where the formula swaps is not evidence of real state change --
    # confirmed reachable via this snapshot's metadata.turn_effect, which
    # spark_phi_hint/spark_phi_narrative (orion-cortex-exec) read straight
    # into live metacognition prompts. This must be suppressed to exactly
    # 0.0, not just "small".
    published: dict[str, object] = {}

    class _Bus:
        enabled = True

        async def publish(self, channel, env):
            published[channel] = env

    monkeypatch.setattr(worker.settings, "orion_phi_encoder_enabled", True, raising=False)
    monkeypatch.setattr(worker.settings, "orion_phi_encoder_weights", str(tmp_path), raising=False)
    monkeypatch.setattr(worker.settings, "inner_features_version", "seed-v2", raising=False)
    monkeypatch.setattr(worker, "_pub_bus", _Bus(), raising=False)
    monkeypatch.setattr(worker.manager, "broadcast", AsyncMock(), raising=False)
    _reset_inner_state(monkeypatch, tmp_path)
    _mock_healthy_substrate_reads(monkeypatch)

    # Tick 1: no probes on the manifest -> valence_source == "heuristic".
    _write_tiny_encoder(tmp_path)
    await worker.run_inner_state_tick()
    assert worker._PHI_PREV_VALENCE_SOURCE == "heuristic"

    # Tick 2: same weights, but reload with a manifest carrying a strong,
    # asymmetric agency_readiness probe -> valence_source flips to "proxy",
    # and the proxy's tanh-squashed output will differ sharply from the
    # heuristic (by construction, per test_agency_valence_proxy_matches_
    # real_encoder_probes).
    monkeypatch.setattr(worker, "_PHI_ENCODER", None, raising=False)
    _write_tiny_encoder(
        tmp_path,
        probes={"z0": {"agency_readiness": 0.9}, "z1": {"agency_readiness": -0.9}},
        encoder_id="test-probes-2",
        encoder_version="v0-probes-2",
    )
    await worker.run_inner_state_tick()
    assert worker._PHI_PREV_VALENCE_SOURCE == "proxy"

    snap_env = published[worker.settings.channel_spark_state_snapshot]
    snap_payload = snap_env.payload
    turn_effect = snap_payload.metadata["turn_effect"]["turn"]
    assert turn_effect["valence"] == 0.0, (
        "a formula swap alone must not be reported as a real valence delta"
    )
    assert snap_payload.metadata["valence_source"] == "proxy"


@pytest.mark.asyncio
async def test_spark_snapshot_dominant_node_always_none(monkeypatch, tmp_path) -> None:
    """2026-07-22 (SelfStateV1 burn): dominant_node/dominant_node_reason are a
    disclosed regression from this burn (see the comment above
    _get_phi_stats() in app/worker.py) -- their only input,
    SelfStateV1.dominant_attention_target_details, no longer exists. Was
    test_spark_snapshot_dominant_node_none_when_encoder_skipped (true only
    when the encoder was off); now true unconditionally, so this test covers
    the encoder-on case too rather than just the skipped one."""
    published: dict[str, object] = {}

    class _Bus:
        enabled = True

        async def publish(self, channel, env):
            published[channel] = env

    monkeypatch.setattr(worker.settings, "orion_phi_encoder_enabled", True, raising=False)
    monkeypatch.setattr(worker.settings, "orion_phi_encoder_weights", str(tmp_path), raising=False)
    monkeypatch.setattr(worker.settings, "inner_features_version", "seed-v2", raising=False)
    _write_tiny_encoder(tmp_path)
    monkeypatch.setattr(worker, "_pub_bus", _Bus(), raising=False)
    monkeypatch.setattr(worker.manager, "broadcast", AsyncMock(), raising=False)
    _reset_inner_state(monkeypatch, tmp_path)
    _mock_healthy_substrate_reads(monkeypatch)

    await worker.run_inner_state_tick()

    reward_env = published.get(worker.settings.channel_phi_reward)
    if reward_env is not None:
        reward = PhiIntrinsicRewardV1.model_validate(reward_env.payload)
        assert reward.dominant_node is None
        assert reward.dominant_node_reason is None

    snap_env = published[worker.settings.channel_spark_state_snapshot]
    snap_payload = snap_env.payload
    snap_dominant_node = (
        snap_payload.dominant_node if hasattr(snap_payload, "dominant_node") else snap_payload["dominant_node"]
    )
    assert snap_dominant_node is None
