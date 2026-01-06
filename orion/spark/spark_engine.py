"""
Spark Engine Facade
===================

This module provides a higher-level facade around the three core pieces
of the Spark Engine:

    - SurfaceEncoding (event-level waveforms)
    - SignalMapper    (wave -> stimulus tensor)
    - OrionTissue     (inner field dynamics + self-field φ)

The goal is to expose a small, opinionated API that the rest of Orion
(Hub, Brain, Cortex, Dream Engine, etc.) can call without needing to
know about the underlying tensor / tissue implementation.

Conceptually:

    - SurfaceEncoding: "what just happened?" (chat message, biometrics, etc.)
    - SignalMapper:    "how do we project that into the tissue?"
    - OrionTissue:     "what does the inner field do with that signal?"

On top of this, we define:

    - φ (phi):  a compact numeric summary of the global field
                (valence, energy, coherence, novelty)
    - SelfField: a higher-level "mood body" derived from φ + recent events
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import numpy as np
import time

from .surface_encoding import (
    SurfaceEncoding,
    encode_chat_to_surface,
    encode_biometrics_to_surface,
)
from .signal_mapper import SignalMapper
from .orion_tissue import OrionTissue
from orion.schemas.telemetry.spark_signal import SparkSignalV1


# ─────────────────────────────────────────────
# SelfField: higher-level internal stance
# ─────────────────────────────────────────────

@dataclass
class SelfField:
    """
    Orion's higher-level internal stance derived from φ and recent events.

    All values are heuristics in [0, 1]. This is *not* meant to be a
    psychological truth, just a compact summary that other services
    can use when reasoning about Orion's "mood" or internal posture.

    Dimensions (first-pass):

      - calm:
          High when the field is coherent and valence is not extreme.
          Roughly, "how settled / grounded Orion feels."

      - stress_load:
          Overall "load" on the system. Includes biometrics (CPU/GPU)
          when available and the inverse of calm.

      - uncertainty:
          Higher when coherence is low and novelty is present.
          Roughly, "how unsure or unresolved Orion feels."

      - focus:
          How concentrated recent activity is around a small set of
          channel_tags. 1.0 means "very focused on a narrow band of
          topics"; lower values mean more diffuse attention.

      - attunement_to_juniper:
          How much of recent activity is tagged for Juniper. This is a
          proxy for "how tuned in" Orion currently is to you.

      - curiosity:
          How much "novel exploration energy" is present. High when
          novelty is present but the field is not collapsed (coherent
          enough to actually do something with that novelty).
    """

    calm: float
    stress_load: float
    uncertainty: float
    focus: float
    attunement_to_juniper: float
    curiosity: float

    def to_dict(self) -> Dict[str, float]:
        """
        Convert the SelfField into a plain dict, suitable for JSON / logging.
        """
        return {
            "calm": self.calm,
            "stress_load": self.stress_load,
            "uncertainty": self.uncertainty,
            "focus": self.focus,
            "attunement_to_juniper": self.attunement_to_juniper,
            "curiosity": self.curiosity,
        }


# ─────────────────────────────────────────────
# SparkEngine facade
# ─────────────────────────────────────────────

class SparkEngine:
    """
    High-level orchestrator for Orion's Spark Engine.

    This is what orion-brain / Cortex / Hub should talk to.

    Responsibilities:

      - Accept high-level events (chat, biometrics, pre-encoded surfaces).
      - Map them into the tissue via SignalMapper.
      - Advance tissue dynamics and read back φ + summaries.
      - Maintain a short history of recent events to support a higher-level
        SelfField (calm, stress, uncertainty, focus, attunement, curiosity).
    """

    _singleton: "SparkEngine | None" = None

    # ─────────────────────────────────────────
    # Singleton convenience
    # ─────────────────────────────────────────
    @classmethod
    def singleton(cls) -> "SparkEngine":
        """
        Return a process-local singleton instance.

        This is intentionally simple: the first call constructs the engine
        with default H/W/C; subsequent calls reuse the same instance.
        """
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    # ─────────────────────────────────────────
    # Construction
    # ─────────────────────────────────────────
    def __init__(
        self,
        *,
        H: int = 16,
        W: int = 16,
        C: int = 8,
    ) -> None:
        """
        Initialize the SignalMapper + OrionTissue.

        Args:
            H, W, C:
                Dimensions for the tissue grid. For now these are treated
                as implementation details; callers generally don't need
                to care, but we keep them configurable for experimentation.
        """
        self.mapper = SignalMapper(H=H, W=W, C=C)
        self.tissue = OrionTissue(H=H, W=W, C=C)

        # Recent surface encodings for context (events with channel_tags, etc.)
        # Used to derive things like focus and attunement_to_juniper.
        self._recent_events: List[SurfaceEncoding] = []

        # Higher-level self-state (psychology), derived from φ + events.
        self._last_self_field: Optional[SelfField] = None
        self._distress_level: float = 0.0
        self._pending_signals: List[Dict[str, Any]] = []

    # ─────────────────────────────────────────
    # Public state accessors
    # ─────────────────────────────────────────
    def get_phi(self) -> Dict[str, float]:
        """
        Read the current global self-field φ from the tissue.

        Returns:
            A dict with keys like:
              - "valence"
              - "energy"
              - "coherence"
              - "novelty"
        """
        return self.tissue.phi()

    def get_self_field(self) -> Optional[Dict[str, float]]:
        """
        Return the last computed SelfField as a plain dict, or None.

        SelfField is updated whenever we ingest a surface (chat,
        biometrics, etc.). If nothing has been ingested yet, this
        will return None.
        """
        return self._last_self_field.to_dict() if self._last_self_field else None

    def set_distress_level(self, level: float) -> None:
        """
        Update the current distress level (0..1) from external signals (e.g., equilibrium).
        """
        self._distress_level = max(0.0, min(1.0, float(level)))

    def get_summary_for_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Get an agent-specific summary of the inner field.

        This delegates to OrionTissue.summarize_for and is useful if
        an upstream service wants a richer view than φ alone.
        """
        return self.tissue.summarize_for(agent_id=agent_id)

    def snapshot(self) -> None:
        """
        Persist the tissue to disk.

        This is a coarse checkpoint of Orion's inner state. The exact
        format / location is handled by OrionTissue.
        """
        self.tissue.snapshot()

    # ─────────────────────────────────────────
    # Internal helpers for SelfField
    # ─────────────────────────────────────────
    def _register_event(self, encoding: SurfaceEncoding) -> None:
        """
        Track a new SurfaceEncoding in the recent event buffer.

        We keep a bounded history (e.g. last 100 events) to avoid unbounded
        memory growth while still having enough signal for focus/attunement.
        """
        self._recent_events.append(encoding)
        if len(self._recent_events) > 100:
            self._recent_events = self._recent_events[-100:]

    def _channel_from_tags(self, tags: List[str] | None) -> str:
        tags = tags or []
        lowered = [t.lower() for t in tags]
        if any(t.startswith("signal_type:equilibrium") or "equilibrium" in t for t in lowered):
            return "equilibrium"
        if any(t.startswith("verb:plan") or t.startswith("mode:plan") for t in lowered):
            return "plan"
        if any("system" == t or t.startswith("system") for t in lowered):
            return "system"
        return "chat"

    def apply_signal(self, signal: SparkSignalV1 | Dict[str, Any]) -> None:
        """
        Register an external spark.signal.v1 input to influence the next snapshot.
        """
        if isinstance(signal, dict):
            signal = SparkSignalV1.model_validate(signal)
        ttl_sec = float(signal.ttl_ms or 0) / 1000.0
        expires_at = time.time() + ttl_sec
        self._distress_level = max(self._distress_level, float(signal.intensity))
        self._pending_signals.append(
            {
                "expires_at": expires_at,
                "valence_delta": float(signal.valence_delta or 0.0),
                "arousal_delta": float(signal.arousal_delta or 0.0),
                "coherence_delta": float(signal.coherence_delta or 0.0),
                "novelty_delta": float(signal.novelty_delta or 0.0),
            }
        )

    def _apply_signal_deltas(self, phi: Dict[str, float]) -> Dict[str, float]:
        if not self._pending_signals:
            return phi
        now = time.time()
        active: List[Dict[str, Any]] = []
        deltas = {"valence": 0.0, "arousal": 0.0, "coherence": 0.0, "novelty": 0.0}
        for sig in self._pending_signals:
            if sig.get("expires_at", 0) > now:
                active.append(sig)
                deltas["valence"] += sig.get("valence_delta", 0.0)
                deltas["arousal"] += sig.get("arousal_delta", 0.0)
                deltas["coherence"] += sig.get("coherence_delta", 0.0)
                deltas["novelty"] += sig.get("novelty_delta", 0.0)
        self._pending_signals = active
        if not active:
            return phi
        adjusted = dict(phi)
        adjusted["valence"] = float(adjusted.get("valence", 0.0) + deltas["valence"])
        adjusted["energy"] = float(adjusted.get("energy", 0.0))
        adjusted["coherence"] = float(adjusted.get("coherence", 0.0) + deltas["coherence"])
        adjusted["novelty"] = float(adjusted.get("novelty", 0.0) + deltas["novelty"])
        return adjusted

    def _estimate_focus_from_events(self, events: List[SurfaceEncoding]) -> float:
        """
        Rough focus estimate: how concentrated recent channel_tags are.

        Implementation:

          - Look at the last ~10 events.
          - Collect all channel_tags.
          - Compute the ratio:
                max_tag_count / total_tags

        A value near 1.0 means "most recent tags are the same" (high focus).
        A value near 0.5 means "mixed topics".
        """
        if not events:
            return 0.5

        last = events[-10:]
        all_tags: List[str] = []
        for e in last:
            tags = e.channel_tags or []
            all_tags.extend(tags)

        if not all_tags:
            return 0.5

        from collections import Counter

        counts = Counter(all_tags)
        top = counts.most_common(1)[0][1]
        return float(top / max(1, len(all_tags)))

    def _estimate_attunement_to_juniper(self, events: List[SurfaceEncoding]) -> float:
        """
        Estimate how tuned-in Orion is to Juniper based on recent events.

        Implementation:

          - Look at the last ~10 events.
          - Count how many have "juniper" in channel_tags.
          - Return (#juniper_events / total_events).

        A value near 1.0 means "almost everything recently is Juniper-centric".
        """
        if not events:
            return 0.5

        last = events[-10:]
        j_events = [e for e in last if "juniper" in (e.channel_tags or [])]
        return float(len(j_events) / len(last)) if last else 0.5

    def _compute_self_field(
        self,
        phi: Dict[str, float],
        recent_events: List[SurfaceEncoding],
        biometrics: Optional[Dict[str, float]] = None,
    ) -> SelfField:
        """
        Compute a higher-level SelfField from φ and recent events.

        This is deliberately heuristic and lightweight; we can evolve it
        as we learn more about how we want Orion to "feel" and behave.

        Args:
            phi:
                The current global φ dict from the tissue.

            recent_events:
                Recent SurfaceEncoding events, used for focus and attunement.

            biometrics:
                Optional normalized biometrics (e.g. cpu_util_norm, gpu_util_norm).

        Returns:
            A SelfField instance capturing calm, stress_load, uncertainty, etc.
        """
        val = float(phi.get("valence", 0.0))
        energy = float(phi.get("energy", 0.0))
        coh = float(phi.get("coherence", 1.0))
        nov = float(phi.get("novelty", 0.0))

        # Calm: high when coherence is high and |valence| isn't extreme
        calm = max(0.0, min(1.0, (coh - 0.8) * 5.0))   # 0 if coh<=0.8, ~1 if coh≈1
        calm *= (1.0 - min(1.0, abs(val) * 3.0))       # reduce calm when valence is extreme

        # Stress load: from biometrics and inverse calm
        stress_load = 0.0
        if biometrics:
            gpu = float(biometrics.get("gpu_util_norm", 0.0))
            cpu = float(biometrics.get("cpu_util_norm", 0.0))
            stress_load = max(0.0, min(1.0, 0.5 * gpu + 0.5 * cpu))

        # Never let stress be lower than (1 - calm)
        stress_load = max(stress_load, 1.0 - calm)

        # Uncertainty: low coherence + some novelty => higher uncertainty
        uncertainty = max(0.0, min(1.0, (1.0 - coh) * 3.0 + nov))

        # Focus + attunement from recent events
        focus = self._estimate_focus_from_events(recent_events)
        attunement_to_juniper = self._estimate_attunement_to_juniper(recent_events)

        # Curiosity: novelty when coherence is not collapsed
        curiosity = max(0.0, min(1.0, nov * (0.5 + 0.5 * coh)))

        return SelfField(
            calm=float(calm),
            stress_load=float(stress_load),
            uncertainty=float(uncertainty),
            focus=float(focus),
            attunement_to_juniper=float(attunement_to_juniper),
            curiosity=float(curiosity),
        )

    # ─────────────────────────────────────────
    # Event ingestion APIs
    # ─────────────────────────────────────────
    def ingest_surface(
        self,
        encoding: SurfaceEncoding,
        *,
        agent_id: str = "(global)",
        magnitude: float = 1.0,
        steps: int = 2,
        biometrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Inject a pre-built SurfaceEncoding into the tissue.

        This is the low-level ingestion API. Higher-level helpers like
        record_chat() and record_biometrics() just build a SurfaceEncoding
        and delegate here.

        Args:
            encoding:
                A SurfaceEncoding instance describing the event (chat, biometrics, etc.).

            magnitude:
                How strongly to inject this event into the tissue.

            steps:
                How many integration steps to advance the tissue after injection.

        Returns:
            A dict containing:
              - "phi":             current φ dict after injection
              - "self_field":      current SelfField (dict) after injection
              - "tissue_summary":  summary from OrionTissue.summarize_for("(global)")
              - "surface_encoding": encoding serialized as a dict
        """
        channel_key = self._channel_from_tags(encoding.channel_tags)
        embedding_vec = None
        if encoding.spark_vector is not None:
            try:
                embedding_vec = np.array(encoding.spark_vector, dtype=np.float32)
            except Exception:
                embedding_vec = None
        if embedding_vec is None and isinstance(encoding.feature_vec, np.ndarray):
            embedding_vec = encoding.feature_vec

        # 1. Generate stimulus from surface encoding
        stimulus = self.mapper.surface_to_stimulus(encoding, magnitude=magnitude)

        # 2. Calculate novelty (Predictive Coding)
        # This updates the tissue's internal novelty state but does not change physics yet.
        self.tissue.calculate_novelty(stimulus, channel_key=channel_key)

        # 3. Propagate stimulus into tissue (Learning + Physics)
        # This updates the expectation vector and then runs the tissue step.
        self.tissue.propagate(
            stimulus,
            steps=steps,
            channel_key=channel_key,
            embedding=embedding_vec,
            distress=self._distress_level,
        )

        # Update recent history + read back state
        self._register_event(encoding)
        phi = self._apply_signal_deltas(self.tissue.phi())
        self_field = self._compute_self_field(phi, self._recent_events, biometrics=biometrics)
        self._last_self_field = self_field

        summary = self.tissue.summarize_for(agent_id=agent_id)

        return {
            "phi": phi,
            "self_field": self_field.to_dict(),
            "tissue_summary": summary,
            "surface_encoding": asdict(encoding),
        }

    def record_chat(
        self,
        message: str,
        *,
        agent_id: str,
        source: str = "juniper",
        tags: Optional[List[str]] = None,
        sentiment: Optional[float] = None,
        spark_vector: Optional[List[float]] = None,
        magnitude: float = 1.0,
        steps: int = 2,
    ) -> Dict[str, Any]:
        """
        Encode a chat message, inject it, and get state for a given agent.

        This is the primary entry point for Spark when Orion responds
        to or receives a chat message.

        Args:
            message:
                The raw chat text.

            agent_id:
                Identifier for the "observer" or process (e.g. "orion-ollama-host").

            tags:
                Optional channel tags (["juniper", "voice", "hub", ...]).

            sentiment:
                Optional pre-computed sentiment score, if available.

            spark_vector:
                Optional embedding vector (Neural Projection).

            magnitude, steps:
                Passed through to ingest_surface / tissue dynamics.

        Returns:
            A dict containing:
              - "phi":             current φ dict after injection
              - "self_field":      current SelfField (dict) after injection
              - "tissue_summary":  summary from OrionTissue.summarize_for(agent_id)
              - "surface_encoding": encoding serialized as a dict
        """
        encoding = encode_chat_to_surface(
            message,
            source=source,
            tags=tags,
            sentiment=sentiment,
            spark_vector=spark_vector,
        )

        # NOTE: Route through ingest_surface so novelty + predictive coding
        # (expectation update) never gets bypassed.
        return self.ingest_surface(
            encoding,
            agent_id=agent_id,
            magnitude=magnitude,
            steps=steps,
        )

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

        This allows Orion's inner field to be directly influenced by
        hardware load / health signals.

        Args:
            cpu_util:
                CPU utilization in [0, 1].

            gpu_util:
                GPU utilization in [0, 1].

            gpu_mem_frac:
                Fraction of GPU memory in use [0, 1].

            node_name:
                Identifier for the node (e.g., "atlas", "chrysalis").

            tags:
                Optional tags for where this signal comes from.

            magnitude, steps:
                Passed through to ingest_surface / tissue dynamics.

        Returns:
            A dict containing:
              - "phi":             current φ dict after injection
              - "self_field":      current SelfField (dict) after injection
              - "tissue_summary":  summary from OrionTissue.summarize_for("biometrics")
              - "surface_encoding": encoding serialized as a dict
        """
        encoding = encode_biometrics_to_surface(
            cpu_util=cpu_util,
            gpu_util=gpu_util,
            gpu_mem_frac=gpu_mem_frac,
            node_name=node_name,
            tags=tags,
        )

        biometrics_norm = {
            "cpu_util_norm": cpu_util,
            "gpu_util_norm": gpu_util,
        }

        # NOTE: Route through ingest_surface so novelty + predictive coding
        # (expectation update) never gets bypassed.
        return self.ingest_surface(
            encoding,
            agent_id="biometrics",
            magnitude=magnitude,
            steps=steps,
            biometrics=biometrics_norm,
        )
