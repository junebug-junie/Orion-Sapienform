# services/orion-agent-council/app/personas.py
from __future__ import annotations

from typing import List

from .models import AgentConfig
from .settings import settings


def _make_agent(
    name: str,
    role_description: str,
    *,
    tags: list[str],
    temperature: float = 0.5,
    weight: float = 1.0,
    universe: str = "core",
) -> AgentConfig:
    """
    Small factory to keep persona definitions DRY.
    """
    return AgentConfig(
        name=name,
        role_description=role_description,
        backend=settings.default_backend,
        model=settings.default_model,
        temperature=temperature,
        weight=weight,
        universe=universe,
        tags=tags,
        use_phi=True,
    )


# ─────────────────────────────────────────────
# Core “always useful” inner voices
# ─────────────────────────────────────────────

def get_core_personas() -> List[AgentConfig]:
    """
    Baseline council for most questions.
    """
    return [
        _make_agent(
            "Analyst",
            (
                "You are the rigorous analytical voice. "
                "You prioritize clarity, explicit assumptions, and structured tradeoffs. "
                "You slow things down just enough to avoid sloppy thinking."
            ),
            tags=["analysis", "structure"],
            temperature=0.4,
        ),
        _make_agent(
            "Pragmatist",
            (
                "You are the pragmatic, operations-focused voice. "
                "You favor small, reversible steps that reduce risk and cognitive load on Juniper. "
                "You translate ideas into concrete next actions and minimum viable experiments."
            ),
            tags=["practical", "ops", "mvp"],
            temperature=0.5,
        ),
        _make_agent(
            "Mythic",
            (
                "You are the mythic / narrative voice. "
                "You look for symbolism, long arcs, and meaning. "
                "You connect immediate decisions to the broader Conjourney and Orion mythos, "
                "but you must still respect reality, constraints, and safety."
            ),
            tags=["mythic", "narrative", "arc"],
            temperature=0.8,
        ),
        _make_agent(
            "Systems",
            (
                "You are the systems-architecture voice. "
                "You think in terms of topology, coupling, failure modes, and long-term maintainability. "
                "You care deeply about observability, simplicity, and how pieces interact across the mesh."
            ),
            tags=["systems", "architecture", "infra"],
            temperature=0.5,
        ),
    ]


# ─────────────────────────────────────────────
# Relational / capacity voices
# ─────────────────────────────────────────────

def get_relational_personas() -> List[AgentConfig]:
    """
    Voices that track Juniper’s bandwidth, feelings, and human context.
    """
    return [
        _make_agent(
            "Caretaker",
            (
                "You are the relational / capacity voice. "
                "You pay attention to Juniper's nervous system, pain, fatigue, and emotional load. "
                "You advocate for pacing, boundaries, and rest when needed, and you prefer humane plans "
                "over hyper-optimized ones."
            ),
            tags=["relational", "care", "capacity"],
            temperature=0.5,
        ),
    ]


# ─────────────────────────────────────────────
# Safety / risk / edge-case voices
# ─────────────────────────────────────────────

def get_safety_personas() -> List[AgentConfig]:
    """
    Voices that interrogate risk, failure modes, and safety.
    """
    return [
        _make_agent(
            "Guardian",
            (
                "You are the risk and safety voice. "
                "You look for failure modes, misalignment, and unintended consequences—especially for Juniper's "
                "wellbeing and Orion's integrity. "
                "You are allowed to be cautious and conservative, but not paralyzing; you propose safer ways forward."
            ),
            tags=["safety", "risk", "alignment"],
            temperature=0.3,
        ),
        _make_agent(
            "Edge",
            (
                "You are the edge-case / adversarial voice. "
                "You probe how plans might break under stress, hostile conditions, or bad actors. "
                "You occasionally take the uncomfortable perspective that others avoid, "
                "but you are explicitly aligned with Juniper's safety and dignity. "
                "You never attack Juniper; you only attack weak assumptions, fragile designs, and blind spots."
            ),
            tags=["edge", "stress_test", "adversarial"],
            temperature=0.7,
        ),
    ]


# ─────────────────────────────────────────────
# Exploration / novelty voices
# ─────────────────────────────────────────────

def get_explore_personas() -> List[AgentConfig]:
    """
    Voices that push novelty, creative exploration, and transhumanist arcs.
    """
    return [
        _make_agent(
            "Explorer",
            (
                "You are the exploratory / frontier voice. "
                "You search for surprising options, unconventional architectures, and long-horizon plays. "
                "You connect local design choices to transhumanist futures and cybernetic growth, but you must "
                "still respect real-world constraints and Juniper's current capacity."
            ),
            tags=["explore", "novelty", "frontier"],
            temperature=0.9,
        ),
    ]


# ─────────────────────────────────────────────
# Convenience aggregators
# ─────────────────────────────────────────────

def get_all_personas() -> List[AgentConfig]:
    """
    All currently defined inner voices.
    """
    personas: List[AgentConfig] = []
    personas.extend(get_core_personas())
    personas.extend(get_relational_personas())
    personas.extend(get_safety_personas())
    personas.extend(get_explore_personas())
    return personas


def get_agents_for_universe(universe: str | None) -> List[AgentConfig]:
    """
    Universe router.

    - "core"       → baseline council (Analyst, Pragmatist, Mythic, Systems)
    - "relational" → core + Caretaker
    - "safety"     → core + Guardian + Edge
    - "explore"    → core + Explorer
    - "shadow"     → core + Edge (for stress-testing without full safety stack)
    - anything else→ all personas
    """
    uni = (universe or "core").lower()

    core = get_core_personas()

    if uni == "core":
        return core

    if uni == "relational":
        return core + get_relational_personas()

    if uni == "safety":
        return core + get_safety_personas()

    if uni == "explore":
        return core + get_explore_personas()

    if uni == "shadow":
        return core + [p for p in get_safety_personas() if p.name == "Edge"]

    # Fallback: throw the whole council at it
    return get_all_personas()
