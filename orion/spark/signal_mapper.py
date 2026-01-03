from __future__ import annotations

"""
Signal Mapping Layer
====================

This module implements the bridge from *SurfaceEncoding* (1D waveform +
feature vector) into the 2D+channels tensor that Orion's inner field
(Tissue) expects.
"""

from typing import Dict, List, Optional

import numpy as np

from .surface_encoding import SurfaceEncoding


class SignalMapper:
    """
    Map surface encodings to stimulus tensors for the OrionTissue.

    The mapping is intentionally explicit and tunable. Over time you can
    replace this with a learned mapper without touching the rest of the
    Spark Engine.
    """

    def __init__(self, H: int = 16, W: int = 16, C: int = 8) -> None:
        self.H = H
        self.W = W
        self.C = C

        # TAG -> channel mapping (Legacy / Fallback).
        self.tag_to_channel: Dict[str, int] = {
            # Embodiment & physical state
            "pain": 0,
            "body": 0,
            "health": 0,

            # Work / money / career arcs
            "career": 1,
            "money": 1,
            "work": 1,

            # Infrastructure / system health
            "system_error": 2,
            "infra": 2,

            # Relationships / emotion
            "relationship": 3,
            "family": 3,

            # Identity anchors
            "juniper": 4,
            "orion": 5,
        }

        # Neural Projection: Fixed Random Projection Matrix (seed=42)
        # Input: 768 dim (embedding) -> Output: 256 dim (16x16 grid)
        # We assume flattening the grid H*W = 256.
        # We also need to decide how to map to channels.
        # Strategy: Project to [H*W] space.
        # Positive activation -> Channel 0 (Safety/Context)
        # Negative activation -> Channel 1 (Novelty/Stimulus)

        self.projection_input_dim = 768 # Standard embedding size
        self.projection_output_dim = self.H * self.W

        rng = np.random.RandomState(42)
        # Gaussian Random Projection Matrix
        self.projection_matrix = rng.randn(self.projection_input_dim, self.projection_output_dim).astype(np.float32)
        # Normalize columns
        self.projection_matrix /= np.linalg.norm(self.projection_matrix, axis=0)


    def surface_to_stimulus(
        self,
        encoding: SurfaceEncoding,
        *,
        magnitude: float = 1.0,
    ) -> np.ndarray:
        """
        Convert a SurfaceEncoding into a stimulus tensor S[H, W, C].

        Neural Projection Strategy:
        1. If encoding has `spark_vector`, project it onto the HxW grid.
        2. Map + activations to Ch 0, - activations to Ch 1.

        Fallback Strategy (Legacy):
        1. Pick a quadrant by modality.
        2. Paint the waveform along X in channel 0.
        3. For each tag, bump an associated channel at the region center.
        """
        S = np.zeros((self.H, self.W, self.C), dtype=np.float32)

        # ---------------------------------------------------------
        # Path A: Neural Projection (if vector available)
        # ---------------------------------------------------------
        if encoding.spark_vector is not None and len(encoding.spark_vector) > 0:
            vec = np.array(encoding.spark_vector, dtype=np.float32)

            # Handle dimension mismatch if vector isn't 768
            # Simple truncation or padding
            if vec.shape[0] != self.projection_input_dim:
                if vec.shape[0] > self.projection_input_dim:
                     vec = vec[:self.projection_input_dim]
                else:
                    padded = np.zeros(self.projection_input_dim, dtype=np.float32)
                    padded[:vec.shape[0]] = vec
                    vec = padded

            # Project: [1, 768] @ [768, 256] -> [1, 256]
            activations = vec @ self.projection_matrix

            # Normalize to 0.0 - 1.0 range (roughly)
            # Standard deviation normalization
            activations = activations / (np.std(activations) + 1e-6)

            # Reshape to grid
            grid = activations.reshape((self.H, self.W))

            # Paint channels
            # Positive -> Channel 0
            S[:, :, 0] += np.maximum(grid, 0) * magnitude

            # Negative -> Channel 1 (absolute value)
            S[:, :, 1] += np.abs(np.minimum(grid, 0)) * magnitude

            return S

        # ---------------------------------------------------------
        # Path B: Legacy Tag/Waveform Heuristics
        # ---------------------------------------------------------

        # 1) Region choice by modality.
        if encoding.modality == "chat":
            x0, x1 = 0, self.H // 2
            y0, y1 = 0, self.W // 2
        elif encoding.modality == "biometrics":
            x0, x1 = self.H // 2, self.H
            y0, y1 = 0, self.W // 2
        elif encoding.modality == "vision":
            x0, x1 = 0, self.H // 2
            y0, y1 = self.W // 2, self.W
        else:
            x0, x1 = self.H // 2, self.H
            y0, y1 = self.W // 2, self.W

        # 2) Paint waveform.
        w = encoding.waveform.astype(np.float32)
        L = min(len(w), x1 - x0)
        row = (y0 + y1) // 2

        for i in range(L):
            x = x0 + i
            S[x, row, 0] += float(w[i]) * magnitude

        # 3) Tag-based bumps.
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        for tag in encoding.channel_tags:
            ch = self.tag_to_channel.get(tag)
            if ch is None or ch >= self.C:
                continue
            S[cx, cy, ch] += magnitude

        return S
