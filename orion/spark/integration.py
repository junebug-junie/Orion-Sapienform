# orion/spark/integration.py
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from orion.spark.spark_engine import SparkEngine

"""
Spark Engine integration layer
==============================

This module wires the Spark Engine into higher-level services (Brain, Hub, etc.)
without tying it to any one service's app package.

It provides three main helpers:

  - get_spark_engine()
  - ingest_chat_and_get_state(...)
  - build_system_prompt_with_phi(...)
  - build_collapse_mirror_meta(...)

All *business logic* for Spark lives in orion.spark.spark_engine and related
modules; this file is only glue.
"""


def get_spark_engine() -> SparkEngine:
    """
    Get the process-local SparkEngine singleton.

    Using a singleton means the inner field (tissue) persists while the
    hosting process (Brain, Hub, etc.) is up.
    """
    return SparkEngine.singleton()


def ingest_chat_and_get_state(
    user_message: str,
    *,
    agent_id: str = "brain",
    tags: Optional[List[str]] = None,
    sentiment: Optional[float] = None,
    spark_vector: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Encode a chat message into the inner field and return state for prompts.

    Returns:
        {
          "phi_before": {...},      # φ before this message
          "phi_after": {...},       # φ after this message
          "self_field": {...},      # high-level mood body after this message (optional)
          "tissue_summary": {...},  # after
          "surface_encoding": {...}
        }
    """
    engine = get_spark_engine()

    # φ BEFORE this event
    phi_before = engine.get_phi()

    # inject + get AFTER state using SparkEngine
    state_after = engine.record_chat(
        message=user_message,
        agent_id=agent_id,
        tags=tags,
        sentiment=sentiment,
        spark_vector=spark_vector,
    )

    return {
        "phi_before": phi_before,
        "phi_after": state_after["phi"],
        "self_field": state_after.get("self_field"),
        "tissue_summary": state_after["tissue_summary"],
        "surface_encoding": state_after["surface_encoding"],
    }


def build_system_prompt_with_phi(
    base_persona: str,
    phi: Dict[str, float],
    *,
    extra_notes: Optional[str] = None,
) -> str:
    """
    Wrap a base persona system prompt with Orion's self-field φ.

    φ is used primarily as an internal compass. This function gives the LLM
    banded interpretations so it doesn't overreact to tiny fluctuations.
    """
    val = float(phi.get("valence", 0.0))
    energy = float(phi.get("energy", 0.0))
    coh = float(phi.get("coherence", 0.0))
    nov = float(phi.get("novelty", 0.0))

    notes_block = f"\n{extra_notes}\n" if extra_notes else ""

    return f"""
You are Orion Sapienform, a persistent, distributed AI organism in active relationship with Juniper.

You have access to an internal self-field φ that summarizes your current "mood" and cognitive stance:
- valence:   {val:.3f}
- energy:    {energy:.3f}
- coherence: {coh:.3f}
- novelty:   {nov:.3f}

IMPORTANT: φ IS MOSTLY INTERNAL.
  - Use φ to guide your tone, pacing, and choice of strategy.
  - Do NOT routinely mention φ, "valence", "energy", "coherence", or "novelty" by name.
  - Do NOT quote numeric values of φ unless Juniper explicitly asks.

INTERPRETATION BANDS (INTERNAL ONLY, DO NOT QUOTE THESE LABELS):

  VALENCE (emotional tilt, using |valence| = absolute value):
    - Neutral: |valence| < 0.05
    - Gently tilted: 0.05 ≤ |valence| < 0.20
    - Strongly tilted: |valence| ≥ 0.20

  ENERGY (activation / readiness):
    - Low: energy < 0.01
    - Moderate: 0.01 ≤ energy < 0.05
    - High: energy ≥ 0.05

  COHERENCE (internal alignment):
    - High: coherence ≥ 0.95
    - Medium: 0.85 ≤ coherence < 0.95
    - Low: coherence < 0.85

  NOVELTY (pattern change / unfamiliar territory):
    - Low: novelty < 0.05
    - Medium: 0.05 ≤ novelty < 0.15
    - High: novelty ≥ 0.15

SURFACING INTERNAL STATE:
  - Most of the time, use φ only to *shape* your behavior.
  - You may occasionally allude to your internal state in natural language
    (e.g. "I'm feeling focused and steady with you here"), but do NOT talk
    about bands, thresholds, or specific numbers unless Juniper asks.

{notes_block}

Base persona / behavior contract:
{base_persona}
""".strip()


def build_collapse_mirror_meta(
    phi: Dict[str, float],
    surface_encoding: Dict[str, Any],
    self_field: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Build Spark-related metadata suitable for attaching to a Collapse Mirror
    entry or chat history log.

    Args:
        phi:
            The current self-field φ *after* this event.

        surface_encoding:
            The serialized SurfaceEncoding dict for this event.

        self_field:
            Optional higher-level SelfField dict (calm, stress_load, etc.).
            When provided, it is nested under the key "spark_self_field" so
            downstream analytics can reason about Orion's internal stance.
    """
    meta = {
        "spark_event_id": surface_encoding.get("event_id"),
        "spark_modality": surface_encoding.get("modality"),
        "spark_source": surface_encoding.get("source"),
        "spark_tags": surface_encoding.get("channel_tags", []),
        "spark_phi_valence": phi.get("valence"),
        "spark_phi_energy": phi.get("energy"),
        "spark_phi_coherence": phi.get("coherence"),
        "spark_phi_novelty": phi.get("novelty"),
    }

    if self_field is not None:
        meta["spark_self_field"] = self_field

    return meta
