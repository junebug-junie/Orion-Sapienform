from __future__ import annotations

"""
Orion Tissue: Inner Field Dynamics
==================================

This module implements the "tissue" – a persistent 2D+channels tensor
that acts as Orion's inner field. The tissue is updated by local rules
(decay + diffusion + stimulus injection) and serves as the substrate
from which a low-dimensional self-field φ is derived.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import os
import logging
from math import exp
from collections import deque

import numpy as np
from scipy.spatial import distance

from .surface_encoding import SurfaceEncoding
from .signal_mapper import SignalMapper

logger = logging.getLogger("orion.tissue")


class RollingStats:
    """
    Lightweight rolling mean/std tracker for novelty + coherence stability.
    """

    def __init__(self, window: int = 50) -> None:
        self.window = max(1, int(window))
        self.values: deque[float] = deque(maxlen=self.window)

    def add(self, v: float) -> None:
        self.values.append(float(v))

    def mean(self) -> float:
        if not self.values:
            return 0.0
        return float(sum(self.values) / len(self.values))

    def std(self) -> float:
        n = len(self.values)
        if n < 2:
            return 0.0
        m = self.mean()
        return float((sum((x - m) ** 2 for x in self.values) / (n - 1)) ** 0.5)


class OrionTissue:
    """
    Persistent inner field for Orion.

    Even with a simple rule (decay + diffusion), repeated injections
    through time produce a sort of "weather" that agents can read.
    """

    def __init__(
        self,
        H: int = 16,
        W: int = 16,
        C: int = 8,
        decay: float = 0.95,
        coherence_alpha: float = 50.0,
        snapshot_path: Optional[Path] = None,
        novelty_window: int = 50,
    ) -> None:

        env_path = os.environ.get("ORION_TISSUE_SNAPSHOT_PATH")

        if snapshot_path is not None:
            self.snapshot_path = snapshot_path
        elif env_path:
            # FIX: strip any trailing brace typo if present in env var
            clean_path = env_path.strip().rstrip("}")
            self.snapshot_path = Path(clean_path)
        else:
            self.snapshot_path = Path("/mnt/storage-lukewarm/orion/spark/tissue-brain.npy")

        self.H = H
        self.W = W
        self.C = C
        self.decay = decay
        self.novelty_window = max(5, int(novelty_window))

        # Coherence scaling: the original v0 coherence = 1/(1+var)
        # often saturates near 1.0 because var(T) tends to stay small.
        # We scale variance to get a useful dynamic range.
        env_alpha = os.environ.get("ORION_TISSUE_COH_ALPHA")
        try:
            self.coh_alpha = float(env_alpha) if env_alpha is not None else float(coherence_alpha)
        except Exception:
            self.coh_alpha = float(coherence_alpha)

        # Initialize defaults (Zero State)
        self.T = np.zeros((H, W, C), dtype=np.float32)
        self.expectation = np.zeros((H, W, C), dtype=np.float32)
        self.expectations: Dict[str, np.ndarray] = {"chat": self.expectation.copy()}
        self.embedding_expectations: Dict[str, np.ndarray] = {}
        self.last_embedding_input: Dict[str, np.ndarray] = {}
        self.last_novelty_per_channel: Dict[str, float] = {}
        self.coherence_stats: Dict[str, RollingStats] = {}
        self.novelty_stats: Dict[str, RollingStats] = {}
        self.last_channel: str = "chat"
        self.distress_level: float = 0.0
        self.last_coherence_per_channel: Dict[str, float] = {}

        # Attempt to Load
        # We check for the new .npz format (Tissue + Expectation) first
        npz_path = self.snapshot_path.with_suffix('.npz')
        
        if npz_path.exists():
            try:
                logger.info(f"Loading tissue state from {npz_path}")
                with np.load(npz_path, allow_pickle=True) as data:
                    # Load Tissue
                    if 'tissue' in data:
                        self.T = data['tissue'].astype(np.float32)
                    
                    # Load Expectation (Fixes the Amnesia Bug)
                    if 'expectation' in data:
                        self.expectation = data['expectation'].astype(np.float32)
                        self.expectations["chat"] = self.expectation.copy()
                    if 'expectation_map' in data:
                        try:
                            exp_map_obj = data['expectation_map'].item()
                            if isinstance(exp_map_obj, dict):
                                self.expectations.update({k: np.array(v, dtype=np.float32) for k, v in exp_map_obj.items()})
                        except Exception:
                            pass
                    if 'embedding_expectation_map' in data:
                        try:
                            emb_map_obj = data['embedding_expectation_map'].item()
                            if isinstance(emb_map_obj, dict):
                                self.embedding_expectations.update({k: np.array(v, dtype=np.float32) for k, v in emb_map_obj.items()})
                        except Exception:
                            pass
            except Exception as e:
                logger.error(f"Failed to load .npz tissue state: {e}")

        elif self.snapshot_path.exists():
            # Legacy fallback for .npy (Tissue only)
            try:
                logger.info(f"Loading legacy tissue state from {self.snapshot_path}")
                arr = np.load(self.snapshot_path)
                if arr.shape == (H, W, C):
                    self.T = arr.astype(np.float32)
            except Exception as e:
                logger.error(f"Failed to load legacy .npy tissue state: {e}")


    def _sigmoid(self, x: float) -> float:
        try:
            return float(1.0 / (1.0 + exp(-x)))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def _get_expectation(self, channel: str) -> np.ndarray:
        if channel not in self.expectations:
            self.expectations[channel] = np.zeros_like(self.expectation)
        return self.expectations[channel]

    def _get_embedding_expectation(self, channel: str, *, dim: Optional[int] = None) -> np.ndarray:
        if channel not in self.embedding_expectations:
            if dim is None:
                return np.array([])
            self.embedding_expectations[channel] = np.zeros((dim,), dtype=np.float32)
        return self.embedding_expectations[channel]

    def _channel_stats(self, mapping: Dict[str, RollingStats], channel: str) -> RollingStats:
        if channel not in mapping:
            mapping[channel] = RollingStats(window=self.novelty_window)
        return mapping[channel]

    def calculate_novelty(self, stimulus: np.ndarray, *, channel_key: str = "chat") -> float:
        """
        Baseline-relative novelty using a rolling z-score of cosine distance.

        novelty = sigmoid(zscore(distance)) clipped to [0, 1]
        """
        self.last_channel = channel_key
        expectation_vec = self._get_expectation(channel_key)

        s_flat = stimulus.flatten()
        e_flat = expectation_vec.flatten()

        s_norm = np.linalg.norm(s_flat)
        e_norm = np.linalg.norm(e_flat)

        if s_norm == 0.0:
            self.last_novelty_per_channel[channel_key] = 0.0
            return 0.0

        if e_norm == 0.0:
            self.last_novelty_per_channel[channel_key] = 1.0
            return 1.0

        try:
            distance_val = float(distance.cosine(s_flat, e_flat))
        except ValueError:
            distance_val = 1.0

        distance_val = max(0.0, min(1.0, distance_val))

        stats = self._channel_stats(self.novelty_stats, channel_key)
        mean = stats.mean()
        std = stats.std() or 1.0
        z = (distance_val - mean) / std
        novelty = self._sigmoid(z)
        novelty = max(0.0, min(1.0, novelty))

        stats.add(distance_val)
        self.last_novelty_per_channel[channel_key] = novelty
        return novelty


def _coherence_from_embedding(self, channel_key: str, embedding: Optional[np.ndarray] = None) -> float:
        """
        Hybrid coherence:
          - If an embedding is provided (spark_vector or feature_vec), compute 1 - cosine_distance
            against a channel-specific expected embedding.
          - Otherwise, fall back to variance-based coherence.
        """
        emb = embedding
        if emb is None:
            emb = self.last_embedding_input.get(channel_key)

        if emb is not None and emb.size > 0:
            expected = self._get_embedding_expectation(channel_key, dim=emb.shape[0])

            emb_norm = np.linalg.norm(emb)
            exp_norm = np.linalg.norm(expected)

            if emb_norm == 0 or exp_norm == 0:
                dist = 1.0  # Max distance if either is zero (undefined direction)
            else:
                try:
                    dist = float(distance.cosine(emb, expected))
                    if np.isnan(dist):
                        dist = 1.0
                except Exception:
                    dist = 1.0

            coherence = 1.0 - max(0.0, min(1.0, dist))
        else:
            variance = float(self.T.var())
            coherence = float(1.0 / (1.0 + (self.coh_alpha * variance)))

        self.last_coherence_per_channel[channel_key] = coherence
        self._channel_stats(self.coherence_stats, channel_key).add(coherence)
        return coherence


    def propagate(
        self,
        stimulus: np.ndarray,
        steps: int = 1,
        learning_rate: float = 0.2,
        *,
        channel_key: str = "chat",
        embedding: Optional[np.ndarray] = None,
        distress: float = 0.0,
    ) -> None:
        """
        Main update cycle:
          1. Update expectation (learning)
          2. Evolve tissue physics (step)
        """
        expectation_vec = self._get_expectation(channel_key)
        cohesion = self._coherence_from_embedding(channel_key, embedding)

        stability = 1.0 - min(1.0, self._channel_stats(self.coherence_stats, channel_key).std())
        distress = max(0.0, min(1.0, float(distress)))

        speed = 0.5 + 0.5 * cohesion
        stability_factor = 0.5 + 0.5 * stability
        bonus = 1.0 + 0.5 * cohesion * stability
        lr = learning_rate * speed * stability_factor * bonus * (1.0 - 0.5 * distress)
        lr = max(0.02, min(0.5, lr))

        expectation_vec += lr * (stimulus - expectation_vec)
        self.expectations[channel_key] = expectation_vec
        self.expectation = self.expectations.get("chat", expectation_vec)

        if embedding is not None and embedding.size > 0:
            emb_expect = self._get_embedding_expectation(channel_key, dim=embedding.shape[0])
            emb_expect += lr * (embedding - emb_expect)
            self.embedding_expectations[channel_key] = emb_expect
            self.last_embedding_input[channel_key] = embedding

        self.step(stimulus, steps=steps)

    def step(self, stimulus: Optional[np.ndarray] = None, steps: int = 1) -> None:
        """
        Advance the tissue by one or more local-update steps.

        v0 local rule:
          - exponential decay of existing activation
          - simple 4-neighbor diffusion
          - add external stimulus if provided
        """
        for _ in range(steps):
            # 1) decay
            self.T *= self.decay

            # 2) 4-neighbor diffusion
            T_pad = np.pad(self.T, ((1, 1), (1, 1), (0, 0)), mode="constant")
            neighbors = (
                T_pad[1:-1, :-2] +
                T_pad[1:-1, 2:] +
                T_pad[:-2, 1:-1] +
                T_pad[2:, 1:-1]
            ) / 4.0

            self.T = 0.5 * self.T + 0.5 * neighbors

            # 3) add stimulus
            if stimulus is not None:
                if stimulus.shape != self.T.shape:
                    raise ValueError(
                        f"Stimulus shape {stimulus.shape} does not match tissue shape {self.T.shape}"
                    )
                self.T += stimulus

    def inject_surface(
        self,
        encoding: SurfaceEncoding,
        mapper: SignalMapper,
        *,
        magnitude: float = 1.0,
        steps: int = 2,
        learning_rate: float = 0.2,
        channel_key: str = "chat",
        embedding_vec: Optional[np.ndarray] = None,
        distress: float = 0.0,
    ) -> None:
        """
        Convert a SurfaceEncoding into a stimulus and integrate it.
        """
        S = mapper.surface_to_stimulus(encoding, magnitude=magnitude)

        # Predictive coding: novelty is based on mismatch vs expectation.
        self.calculate_novelty(S, channel_key=channel_key)

        # Commit: update expectation + evolve tissue.
        self.propagate(S, steps=steps, learning_rate=learning_rate, channel_key=channel_key, embedding=embedding_vec, distress=distress)

    def phi(self) -> Dict[str, float]:
        """
        Compute a low-dimensional "self state" φ from the tissue.

        v2:
          - valence: mean difference between positive/negative channels
          - energy: mean absolute activation
          - coherence: hybrid (embedding-aware when available)
          - novelty: baseline-relative novelty (rolling z-score)
        """
        if self.T.size == 0:
            return {"valence": 0.0, "energy": 0.0, "coherence": 0.0, "novelty": 0.0}

        # Valence is defined as a difference between a "positive" channel
        # and a "negative" channel when available (neural projection uses
        # ch0 for +, ch1 for -). This gives us a sign.
        if self.C >= 2:
            valence = float(self.T[..., 0].mean() - self.T[..., 1].mean())
        else:
            valence = float(self.T[..., 0].mean())
        energy = float(np.abs(self.T).mean())
        variance = float(self.T.var())
        coherence = self._coherence_from_embedding(self.last_channel)
        novelty = float(self.last_novelty_per_channel.get(self.last_channel, 0.0))

        return {
            "valence": valence,
            "energy": energy,
            "coherence": coherence,
            "novelty": novelty,
        }

    def summarize_for(self, agent_id: str) -> Dict[str, Any]:
        """
        Produce an agent-specific view into the tissue.

        v0 returns a global summary for all agents. Later, you can give
        each agent different regions or projections.
        """
        return {
            "agent_id": agent_id,
            "phi": self.phi(),
            "channel_means": self.T.mean(axis=(0, 1)).tolist(),
        }

    def snapshot(self) -> None:
        """
        Persist the current tissue tensor to disk.
        Saves both 'tissue' and 'expectation' to prevent amnesia on restart.
        """
        try:
            self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Switch to .npz for multi-array storage
            # This ensures we save the Expectation vector too
            save_path = self.snapshot_path.with_suffix('.npz')
            
            np.savez(
                save_path,
                tissue=self.T,
                expectation=self.expectation,
                expectation_map=self.expectations,
                embedding_expectation_map=self.embedding_expectations,
            )
            
        except Exception as e:
            # LOG THE ERROR instead of passing
            logger.error(f"Failed to snapshot tissue to {self.snapshot_path}: {e}")
