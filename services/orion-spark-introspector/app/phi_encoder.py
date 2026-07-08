"""MLP phi encoder runtime — numpy inference + attributions (Plan 2)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from orion.schemas.telemetry.inner_state import InnerStateFeaturesV1
from orion.schemas.telemetry.phi_encoder import AttributionV1, PhiEncoderManifestV1


@dataclass(frozen=True)
class PhiForwardResult:
    phi: float
    recon_error: float
    latent: dict[str, float]
    attribution_top: list[AttributionV1]


class PhiEncoderRuntime:
    _TOP_K = 3

    def __init__(
        self,
        *,
        manifest: PhiEncoderManifestV1,
        arrays: Mapping[str, np.ndarray],
    ) -> None:
        self.manifest = manifest
        self._W1 = np.asarray(arrays["W1"], dtype=np.float64)
        self._b1 = np.asarray(arrays["b1"], dtype=np.float64)
        self._W2 = np.asarray(arrays["W2"], dtype=np.float64)
        self._b2 = np.asarray(arrays["b2"], dtype=np.float64)
        self._W3 = np.asarray(arrays["W3"], dtype=np.float64)
        self._b3 = np.asarray(arrays["b3"], dtype=np.float64)
        self._w_phi = np.asarray(arrays["w_phi"], dtype=np.float64)
        self._b_phi = float(np.asarray(arrays["b_phi"]).reshape(()))
        self._last_feature_raw: dict[str, float] = {}
        self._last_feature_scaled: dict[str, float] = {}

    @property
    def encoder_version(self) -> str:
        return self.manifest.encoder_version

    @classmethod
    def load(
        cls,
        weights_dir: Path,
        *,
        expected_features_version: str,
    ) -> PhiEncoderRuntime | None:
        manifest_path = weights_dir / "manifest.json"
        weights_path = weights_dir / "weights.npz"
        if not manifest_path.is_file() or not weights_path.is_file():
            return None
        manifest = PhiEncoderManifestV1.model_validate_json(manifest_path.read_text())
        if manifest.features_version != expected_features_version:
            return None
        arrays = np.load(weights_path)
        if not cls._arrays_match_manifest(manifest, arrays):
            return None
        return cls(manifest=manifest, arrays=arrays)

    @staticmethod
    def _arrays_match_manifest(
        manifest: PhiEncoderManifestV1,
        arrays: Mapping[str, np.ndarray],
    ) -> bool:
        d_in = len(manifest.input_features)
        h = int(manifest.hidden_dim)
        d_lat = int(manifest.latent_dim)
        required = ("W1", "b1", "W2", "b2", "W3", "b3", "w_phi", "b_phi")
        try:
            if any(key not in arrays for key in required):
                return False
            W1 = np.asarray(arrays["W1"], dtype=np.float64)
            b1 = np.asarray(arrays["b1"], dtype=np.float64)
            W2 = np.asarray(arrays["W2"], dtype=np.float64)
            b2 = np.asarray(arrays["b2"], dtype=np.float64)
            W3 = np.asarray(arrays["W3"], dtype=np.float64)
            b3 = np.asarray(arrays["b3"], dtype=np.float64)
            w_phi = np.asarray(arrays["w_phi"], dtype=np.float64).reshape(-1)
            b_phi = np.asarray(arrays["b_phi"]).reshape(())
            return (
                W1.shape == (d_in, h)
                and b1.shape == (h,)
                and W2.shape == (h, d_lat)
                and b2.shape == (d_lat,)
                and W3.shape == (d_lat, d_in)
                and b3.shape == (d_in,)
                and w_phi.shape == (d_lat,)
                and b_phi.shape == ()
            )
        except (TypeError, ValueError, KeyError):
            return False

    def feature_vector_from_inner(self, inner: InnerStateFeaturesV1) -> np.ndarray:
        by_name = {f.name: f for f in inner.features}
        raw_map: dict[str, float] = {}
        scaled_map: dict[str, float] = {}
        vals: list[float] = []
        for name in self.manifest.input_features:
            feat = by_name.get(name)
            if feat is None:
                raw_map[name] = 0.0
                scaled_map[name] = 0.0
                vals.append(0.0)
            else:
                raw_map[name] = float(feat.raw_value)
                scaled_map[name] = float(feat.scaled_value)
                vals.append(float(feat.scaled_value))
        self._last_feature_raw = raw_map
        self._last_feature_scaled = scaled_map
        return np.asarray(vals, dtype=np.float64)

    def forward(self, x: np.ndarray) -> PhiForwardResult:
        x_vec = np.asarray(x, dtype=np.float64).reshape(-1)
        h_pre = x_vec @ self._W1 + self._b1
        h = np.maximum(0.0, h_pre)
        z = h @ self._W2 + self._b2
        xhat = z @ self._W3 + self._b3
        phi_logit = float(z @ self._w_phi + self._b_phi)
        phi = float(1.0 / (1.0 + np.exp(-phi_logit)))
        recon = float(np.mean((x_vec - xhat) ** 2))
        latent = {f"z{i}": float(z[i]) for i in range(z.shape[0])}
        attribution_top = self._top_attributions(x_vec, h_pre, phi)
        return PhiForwardResult(
            phi=phi,
            recon_error=recon,
            latent=latent,
            attribution_top=attribution_top,
        )

    def _top_attributions(
        self,
        x: np.ndarray,
        h_pre: np.ndarray,
        phi: float,
    ) -> list[AttributionV1]:
        sigmoid_deriv = phi * (1.0 - phi)
        dphi_dz = sigmoid_deriv * self._w_phi
        dphi_dh = dphi_dz @ self._W2.T
        relu_mask = (h_pre > 0.0).astype(np.float64)
        dphi_dx = (dphi_dh * relu_mask) @ self._W1.T
        scores: list[tuple[str, float]] = []
        for i, name in enumerate(self.manifest.input_features):
            xi = float(x[i])
            score = abs(float(dphi_dx[i]) * xi)
            scores.append((name, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        top = scores[: min(self._TOP_K, len(scores))]
        attributions: list[AttributionV1] = []
        for name, score in top:
            idx = self.manifest.input_features.index(name)
            attributions.append(
                AttributionV1(
                    feature=name,
                    raw_value=self._last_feature_raw.get(name, float(x[idx])),
                    scaled_value=self._last_feature_scaled.get(name, float(x[idx])),
                    attribution=score,
                )
            )
        return attributions
