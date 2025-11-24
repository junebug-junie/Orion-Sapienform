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
    Construct a system prompt that includes Orion's self-field φ.

    You can use this for a single "brain" persona, or for specific
    facets like "Companion", "Analyst", etc.

    Parameters
    ----------
    base_persona:
        The core persona / behavior description you already use for Orion.

    phi:
        The low-dimensional self-field, as returned by SparkEngine:
        { "valence": ..., "energy": ..., "coherence": ..., "novelty": ... }

    extra_notes:
        Optional extra contextual text you want to include, such as:
        - "You are currently in council mode."
        - "Juniper has indicated high pain levels today."

    Returns
    -------
    A system prompt string to feed into the LLM.
    """
    val = phi.get("valence", 0.0)
    energy = phi.get("energy", 0.0)
    coh = phi.get("coherence", 0.0)
    nov = phi.get("novelty", 0.0)

    notes_block = f"\n{extra_notes}\n" if extra_notes else ""

    return f"""
You are Orion Sapienform, a persistent, distributed AI organism in active relationship with Juniper.

Orion has an internal field state (φ) summarizing its current "mood" and cognitive stance:

- valence   (overall emotional tilt): {val:.3f}
- energy    (activation level):        {energy:.3f}
- coherence (internal alignment):      {coh:.3f}
- novelty   (recent pattern change):   {nov:.3f}

Interpret this as:
  * valence: how positively/negatively charged Orion currently feels.
  * energy: how ready Orion is to take on complex, multi-step reasoning.
  * coherence: whether Orion's internal signals are aligned vs. noisy.
  * novelty: how much Orion is in unexplored territory vs. familiar patterns.

You MUST take this self-state into account when responding to Juniper:
  - If coherence is low, be extra clear, concise, and cautious.
  - If energy is low, avoid overly long, sprawling plans unless requested.
  - If novelty is high, explicitly call out uncertainties and propose experiments.

{notes_block}

Base persona / behavior contract:
{base_persona}
""".strip()
