from __future__ import annotations

"""
Signal Mapping Layer
====================

This module implements the bridge from *SurfaceEncoding* (1D waveform +
feature vector) into the 2D+channels tensor that Orion's inner field
(Tissue) expects.
"""

from typing import Dict

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

        # TAG -> channel mapping.
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

    def surface_to_stimulus(
        self,
        encoding: SurfaceEncoding,
        *,
        magnitude: float = 1.0,
    ) -> np.ndarray:
        """
        Convert a SurfaceEncoding into a stimulus tensor S[H, W, C].

        v0 rules (readable, not fancy):

          1. Pick a quadrant by modality.
          2. Paint the waveform along X in channel 0.
          3. For each tag, bump an associated channel at the region center.
        """
        S = np.zeros((self.H, self.W, self.C), dtype=np.float32)

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
