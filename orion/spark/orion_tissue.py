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

import numpy as np

from .surface_encoding import SurfaceEncoding
from .signal_mapper import SignalMapper


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
        snapshot_path: Optional[Path] = None,
    ) -> None:
        self.H = H
        self.W = W
        self.C = C
        self.decay = decay
        self.snapshot_path = snapshot_path or Path("/tmp/orion_tissue.npy")

        if self.snapshot_path.exists():
            try:
                arr = np.load(self.snapshot_path)
                if arr.shape == (H, W, C):
                    self.T = arr.astype(np.float32)
                else:
                    self.T = np.zeros((H, W, C), dtype=np.float32)
            except Exception:
                self.T = np.zeros((H, W, C), dtype=np.float32)
        else:
            self.T = np.zeros((H, W, C), dtype=np.float32)

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
    ) -> None:
        """
        Convert a SurfaceEncoding into a stimulus and integrate it.
        """
        S = mapper.surface_to_stimulus(encoding, magnitude=magnitude)
        self.step(S, steps=steps)

    def phi(self) -> Dict[str, float]:
        """
        Compute a low-dimensional "self state" φ from the tissue.

        v0:
          - valence: mean of channel 0
          - energy: mean absolute activation
          - coherence: 1 / (1 + var(T))
          - novelty: placeholder (to be wired to a baseline later)
        """
        if self.T.size == 0:
            return {"valence": 0.0, "energy": 0.0, "coherence": 0.0, "novelty": 0.0}

        valence = float(self.T[..., 0].mean())
        energy = float(np.abs(self.T).mean())
        variance = float(self.T.var())
        coherence = float(1.0 / (1.0 + variance))
        novelty = 0.0

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
        """
        try:
            self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.snapshot_path, self.T)
        except Exception:
            # fail-soft; not critical path
            pass
