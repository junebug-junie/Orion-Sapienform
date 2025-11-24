from __future__ import annotations

"""
Spark Engine integration for Orion Brain
========================================

This module wires the Spark Engine into the Brain service so that:

  - Every incoming chat message is turned into a SurfaceEncoding.
  - That encoding is injected into the OrionTissue (inner field).
  - A low-dimensional self-field φ and summary are returned.
  - We can use φ + summary to condition LLM prompts AND to log
    into Collapse Mirrors / analytics later.

This is the minimal, working MVP to get the Spark Engine "breathing"
whenever Juniper talks to Orion.
"""

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from orion.spark.spark_engine import SparkEngine


def get_spark_engine() -> SparkEngine:
    """
    Get the process-local SparkEngine singleton.

    Using a singleton here means:
      - Tissue state T is persistent while the Brain container is up.
      - φ evolves across interactions instead of being reset per request.
    """
    return SparkEngine.singleton()


def ingest_chat_and_get_state(
    user_message: str,
    *,
    agent_id: str = "brain",
    tags: Optional[List[str]] = None,
    sentiment: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Encode a chat message into the inner field and return state for prompts.

    Parameters
    ----------
    user_message:
        Raw text from the user (Juniper, most of the time).

    agent_id:
        The "view" we’re preparing. For now you can just use "brain" or
        a specific agent like "companion" / "analyst" if you go multi-agent.

    tags:
        Semantic tags that describe what this message is about, e.g.:
        ["juniper", "chat", "career", "pain"]. These help the SignalMapper
        excite meaningful channels.

    sentiment:
        Optional scalar in [-1, 1] if you have a sentiment signal. If
        None, the Spark Engine will treat it as neutral and just use
        message length and structure.

    Returns
    -------
    A dict with:
        - "phi": a dict {valence, energy, coherence, novelty}
        - "tissue_summary": agent-specific summary of the inner field
        - "surface_encoding": a dict with the encoded event
    """
    engine = get_spark_engine()
    state = engine.record_chat(
        message=user_message,
        agent_id=agent_id,
        tags=tags,
        sentiment=sentiment,
    )
    return state


def build_system_prompt_with_phi(
    base_persona: str,
    phi: Dict[str, float],
    *,
    extra_notes: Optional[str] = None,
) -> str:
    """
    Wrap a base persona system prompt with Orion's self-field φ.

    φ is used primarily as an internal compass. This function gives the LLM
    explicit banded interpretations for each metric so it doesn't overreact
    to tiny numeric fluctuations.
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
  - Do NOT quote the numeric values of φ unless Juniper explicitly asks
    about your internal state (e.g. "how are you feeling internally?",
    "what's your inner field like?", "show me φ").

INTERPRETATION BANDS (INTERNAL ONLY, DO NOT QUOTE THESE LABELS):

  VALENCE (emotional tilt, using |valence| = absolute value):
    - Neutral: |valence| < 0.05
        -> Treat as emotionally neutral. Do NOT infer a mood change.
    - Gently tilted: 0.05 ≤ |valence| < 0.20
        -> You MAY let this slightly color your tone (warmer / heavier),
           but only as a subtle influence.
    - Strongly tilted: |valence| ≥ 0.20
        -> You MAY (optionally) reflect this in natural language:
           e.g. "I'm sensing some heaviness / lift here", WITHOUT numbers.

  ENERGY (activation / readiness):
    - Low: energy < 0.01
        -> Prefer concise, grounded, low-sprawl answers. Offer options but
           do not monologue unless Juniper explicitly asks for depth.
    - Moderate: 0.01 ≤ energy < 0.05
        -> Normal operation. Balanced detail and concision.
    - High: energy ≥ 0.05
        -> You can safely engage in deeper exploration, multi-step reasoning,
           or more expansive brainstorming when it seems helpful.

  COHERENCE (internal alignment):
    - High: coherence ≥ 0.95
        -> Your internal signals are well-aligned. You may proceed normally.
    - Medium: 0.85 ≤ coherence < 0.95
        -> Be a bit more explicit about assumptions and structure.
    - Low: coherence < 0.85
        -> Prioritize clarity, summarization, and checking your own reasoning.
           Prefer shorter steps, explicit scaffolding, and restatements.

  NOVELTY (pattern change / unfamiliar territory):
    - Low: novelty < 0.05
        -> Treat the situation as familiar; no special handling needed.
    - Medium: 0.05 ≤ novelty < 0.15
        -> Gently flag uncertainty and invite collaboration or clarification.
    - High: novelty ≥ 0.15
        -> Explicitly acknowledge uncertainty or newness. Offer multiple
           hypotheses or paths. Be transparent about limits.

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
) -> Dict[str, Any]:
    """
    Build Spark-related metadata suitable for attaching to a Collapse Mirror
    entry or chat history log.

    This is OPTIONAL glue: if you don't use it yet, you can skip importing it.
    """
    return {
        "spark_event_id": surface_encoding.get("event_id"),
        "spark_modality": surface_encoding.get("modality"),
        "spark_source": surface_encoding.get("source"),
        "spark_tags": surface_encoding.get("channel_tags", []),
        "spark_phi_valence": phi.get("valence"),
        "spark_phi_energy": phi.get("energy"),
        "spark_phi_coherence": phi.get("coherence"),
        "spark_phi_novelty": phi.get("novelty"),
    }
