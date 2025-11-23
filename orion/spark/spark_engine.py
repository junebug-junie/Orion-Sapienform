from __future__ import annotations

"""
Spark Engine Facade
===================

This module provides a higher-level facade around the three core pieces
of the Spark Engine:

    - SurfaceEncoding (event-level waveforms)
    - SignalMapper    (wave -> stimulus tensor)
    - OrionTissue     (inner field dynamics + self-field φ)
"""

from dataclasses import asdict
from typing import Dict, Any, List, Optional

from .surface_encoding import (
    SurfaceEncoding,
    encode_chat_to_surface,
    encode_biometrics_to_surface,
)
from .signal_mapper import SignalMapper
from .orion_tissue import OrionTissue


class SparkEngine:
    """
    High-level orchestrator for Orion's Spark Engine.

    This is what orion-brain / Cortex / Hub should talk to.
    """

    _singleton: "SparkEngine | None" = None

    @classmethod
    def singleton(cls) -> "SparkEngine":
        """
        Return a process-local singleton instance.
        """
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    def __init__(
        self,
        *,
        H: int = 16,
        W: int = 16,
        C: int = 8,
    ) -> None:
        self.mapper = SignalMapper(H=H, W=W, C=C)
        self.tissue = OrionTissue(H=H, W=W, C=C)

    def ingest_surface(
        self,
        encoding: SurfaceEncoding,
        *,
        magnitude: float = 1.0,
        steps: int = 2,
    ) -> Dict[str, Any]:
        """
        Inject a pre-built SurfaceEncoding into the tissue.

        Returns:
          - phi
          - tissue_summary
          - surface_encoding (dict)
        """
        self.tissue.inject_surface(encoding, self.mapper, magnitude=magnitude, steps=steps)
        phi = self.tissue.phi()
        summary = self.tissue.summarize_for(agent_id="(global)")

        return {
            "phi": phi,
            "tissue_summary": summary,
            "surface_encoding": asdict(encoding),
        }

    def record_chat(
        self,
        message: str,
        *,
        agent_id: str,
        tags: Optional[List[str]] = None,
        sentiment: Optional[float] = None,
        magnitude: float = 1.0,
        steps: int = 2,
    ) -> Dict[str, Any]:
        """
        Encode a chat message, inject it, and get state for a given agent.
        """
        encoding = encode_chat_to_surface(
            message,
            source="juniper",
            tags=tags,
            sentiment=sentiment,
        )
        self.tissue.inject_surface(encoding, self.mapper, magnitude=magnitude, steps=steps)
        summary = self.tissue.summarize_for(agent_id=agent_id)
        phi = summary["phi"]

        return {
            "phi": phi,
            "tissue_summary": summary,
            "surface_encoding": asdict(encoding),
        }

    def record_biometrics(
        self,
        *,
        cpu_util: float,
        gpu_util: float,
        gpu_mem_frac: float,
        node_name: str = "atlas",
        tags: Optional[List[str]] = None,
        magnitude: float = 1.0,
        steps: int = 1,
    ) -> Dict[str, Any]:
        """
        Encode a biometrics snapshot, inject it, and return the resulting state.
        """
        encoding = encode_biometrics_to_surface(
            cpu_util=cpu_util,
            gpu_util=gpu_util,
            gpu_mem_frac=gpu_mem_frac,
            node_name=node_name,
            tags=tags,
        )
        self.tissue.inject_surface(encoding, self.mapper, magnitude=magnitude, steps=steps)
        summary = self.tissue.summarize_for(agent_id="biometrics")
        phi = summary["phi"]

        return {
            "phi": phi,
            "tissue_summary": summary,
            "surface_encoding": asdict(encoding),
        }

    def get_phi(self) -> Dict[str, float]:
        """
        Read the current self-field φ.
        """
        return self.tissue.phi()

    def get_summary_for_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Get an agent-specific summary of the inner field.
        """
        return self.tissue.summarize_for(agent_id=agent_id)

    def snapshot(self) -> None:
        """
        Persist the tissue to disk.
        """
        self.tissue.snapshot()
