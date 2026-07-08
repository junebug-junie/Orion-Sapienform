from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from orion.schemas.telemetry.inner_state import InnerFeatureV1, InnerStateFeaturesV1
from orion.schemas.telemetry.phi_encoder import (
    AttributionV1,
    CorpusStatsV1,
    PhiEncoderManifestV1,
    TrainingStatsV1,
)

_REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "spark_phi_encoder",
    _REPO / "services" / "orion-spark-introspector" / "app" / "phi_encoder.py",
)
_phi_encoder = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _phi_encoder
_SPEC.loader.exec_module(_phi_encoder)

PhiEncoderRuntime = _phi_encoder.PhiEncoderRuntime
PhiForwardResult = _phi_encoder.PhiForwardResult


def _tiny_manifest(input_features: list[str]) -> PhiEncoderManifestV1:
    return PhiEncoderManifestV1(
        encoder_id="test",
        encoder_version="v0",
        status="candidate",
        architecture="mlp_shallow_v1",
        features_version="seed-v2",
        input_features=input_features,
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


def _write_weights(tmp_path: Path, d_in: int, h: int, d_lat: int) -> None:
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


def test_mlp_forward_phi_bounded(tmp_path: Path) -> None:
    feats = ["coherence", "overall_intensity"]
    manifest = _tiny_manifest(feats)
    d_in, h, d_lat = 2, 4, 2
    _write_weights(tmp_path, d_in, h, d_lat)
    (tmp_path / "manifest.json").write_text(manifest.model_dump_json())
    rt = PhiEncoderRuntime.load(tmp_path, expected_features_version="seed-v2")
    assert rt is not None
    x = np.array([0.1, -0.2])
    out = rt.forward(x)
    assert isinstance(out, PhiForwardResult)
    assert 0.0 <= out.phi <= 1.0
    assert out.recon_error >= 0.0
    assert len(out.latent) == 2
    assert len(out.attribution_top) == 2
    assert all(isinstance(a, AttributionV1) for a in out.attribution_top)
    assert {a.feature for a in out.attribution_top} == set(feats)


def test_manifest_version_mismatch_returns_none(tmp_path: Path) -> None:
    manifest = _tiny_manifest(["coherence"])
    manifest = manifest.model_copy(update={"features_version": "seed-v1"})
    (tmp_path / "manifest.json").write_text(manifest.model_dump_json())
    np.savez(tmp_path / "weights.npz", W1=np.zeros((1, 1)))
    assert PhiEncoderRuntime.load(tmp_path, expected_features_version="seed-v2") is None


def test_manifest_weight_shape_mismatch_returns_none(tmp_path: Path) -> None:
    feats = ["coherence", "overall_intensity"]
    manifest = _tiny_manifest(feats)
    (tmp_path / "manifest.json").write_text(manifest.model_dump_json())
    # Wrong W1 shape vs hidden_dim/latent_dim in manifest.
    np.savez(
        tmp_path / "weights.npz",
        W1=np.zeros((2, 2)),
        b1=np.zeros(2),
        W2=np.zeros((2, 2)),
        b2=np.zeros(2),
        W3=np.zeros((2, 2)),
        b3=np.zeros(2),
        w_phi=np.array([0.1, 0.2]),
        b_phi=np.array(0.0),
    )
    assert PhiEncoderRuntime.load(tmp_path, expected_features_version="seed-v2") is None


def test_feature_vector_from_inner_orders_manifest_features(tmp_path: Path) -> None:
    feats = ["overall_intensity", "coherence"]
    manifest = _tiny_manifest(feats)
    d_in, h, d_lat = 2, 4, 2
    _write_weights(tmp_path, d_in, h, d_lat)
    (tmp_path / "manifest.json").write_text(manifest.model_dump_json())
    rt = PhiEncoderRuntime.load(tmp_path, expected_features_version="seed-v2")
    assert rt is not None
    inner = InnerStateFeaturesV1(
        features_version="seed-v2",
        generated_at=datetime.now(timezone.utc),
        features=[
            InnerFeatureV1(
                name="coherence",
                raw_value=0.8,
                scaled_value=0.3,
                source="test",
            ),
            InnerFeatureV1(
                name="overall_intensity",
                raw_value=0.5,
                scaled_value=-0.1,
                source="test",
            ),
        ],
    )
    x = rt.feature_vector_from_inner(inner)
    np.testing.assert_allclose(x, np.array([-0.1, 0.3]))
    out = rt.forward(x)
    assert out.attribution_top[0].raw_value in {0.5, 0.8}
    assert out.attribution_top[0].scaled_value in {-0.1, 0.3}
