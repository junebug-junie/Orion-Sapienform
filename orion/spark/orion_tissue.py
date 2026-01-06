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
from typing import Optional, Dict, Any
import os
import logging

import numpy as np
from scipy.spatial import distance

from .surface_encoding import SurfaceEncoding
from .signal_mapper import SignalMapper

logger = logging.getLogger("orion.tissue")


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
        self.last_novelty = 0.0

        # Attempt to Load
        # We check for the new .npz format (Tissue + Expectation) first
        npz_path = self.snapshot_path.with_suffix('.npz')
        
        if npz_path.exists():
            try:
                logger.info(f"Loading tissue state from {npz_path}")
                with np.load(npz_path) as data:
                    # Load Tissue
                    if 'tissue' in data:
                        self.T = data['tissue'].astype(np.float32)
                    
                    # Load Expectation (Fixes the Amnesia Bug)
                    if 'expectation' in data:
                        self.expectation = data['expectation'].astype(np.float32)
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


    def calculate_novelty(self, stimulus: np.ndarray) -> float:
        """
        Calculate novelty as the cosine distance between the incoming stimulus
        and the current expectation state.

        Returns a value in [0.0, 1.0].
        """
        # Flatten for vector comparison
        s_flat = stimulus.flatten()
        e_flat = self.expectation.flatten()

        # Handle zero vectors to avoid NaN
        s_norm = np.linalg.norm(s_flat)
        e_norm = np.linalg.norm(e_flat)

        if s_norm == 0.0:
            # No stimulus => no novelty
            self.last_novelty = 0.0
            return 0.0

        if e_norm == 0.0:
            # Stimulus exists but no expectation => max novelty (cold start)
            self.last_novelty = 1.0
            return 1.0

        # Cosine distance: 1 - cosine_similarity
        try:
            d = float(distance.cosine(s_flat, e_flat))
        except ValueError:
            # Fallback for safety
            d = 1.0

        # Clamp to [0, 1] just in case
        d = max(0.0, min(1.0, d))

        self.last_novelty = d
        return d

    def propagate(
        self,
        stimulus: np.ndarray,
        steps: int = 1,
        learning_rate: float = 0.2
    ) -> None:
        """
        Main update cycle:
          1. Update expectation (learning)
          2. Evolve tissue physics (step)
        """
        # Hebbian / EMA update of expectation towards the new stimulus
        # We do this *after* novelty calculation (which should happen before propagate call)
        # But here we assume we are committing the stimulus to memory.
        self.expectation += learning_rate * (stimulus - self.expectation)

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
    ) -> None:
        """
        Convert a SurfaceEncoding into a stimulus and integrate it.
        """
        S = mapper.surface_to_stimulus(encoding, magnitude=magnitude)

        # Predictive coding: novelty is based on mismatch vs expectation.
        self.calculate_novelty(S)

        # Commit: update expectation + evolve tissue.
        self.propagate(S, steps=steps, learning_rate=learning_rate)

    def phi(self) -> Dict[str, float]:
        """
        Compute a low-dimensional "self state" φ from the tissue.

        v0:
          - valence: mean of channel 0
          - energy: mean absolute activation
          - coherence: 1 / (1 + var(T))
          - novelty: derived from predictive coding error (cosine distance)
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
        coherence = float(1.0 / (1.0 + (self.coh_alpha * variance)))
        novelty = float(self.last_novelty)

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
            
            np.savez(save_path, tissue=self.T, expectation=self.expectation)
            
        except Exception as e:
            # LOG THE ERROR instead of passing
            logger.error(f"Failed to snapshot tissue to {self.snapshot_path}: {e}")
