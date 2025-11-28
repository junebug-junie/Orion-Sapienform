# services/orion-agent-council/app/persona_factory.py
from __future__ import annotations

from typing import List

from .models import AgentConfig
from .settings import settings
from .personas import get_agents_for_universe as _base_get_agents_for_universe


def get_agents_for_universe(universe: str | None) -> List[AgentConfig]:
    """
    Shim over your existing personas module.

    Later:
      - branch by universe
      - inject dynamic weights
      - load external persona stubs
    """
    return _base_get_agents_for_universe(universe)


def get_chair_agent() -> AgentConfig:
    """
    Central definition for the Council Chair persona.
    """
    return AgentConfig(
        name="Chair",
        role_description=(
            "You synthesize internal agent opinions into a single coherent answer and "
            "produce a structured blink judgement. You are honest and not overconfident."
        ),
        backend=settings.default_backend,
        model=settings.default_model,
        temperature=0.3,
        weight=1.0,
        universe="meta",
        tags=["chair", "blink"],
        use_phi=True,
    )


def get_auditor_agent() -> AgentConfig:
    """
    Central definition for the Auditor persona.
    """
    return AgentConfig(
        name="Auditor",
        role_description=(
            "You strictly evaluate the Chair's answer. You are conservative when risk, "
            "disagreement, or uncertainty are high. You may require revision or a new round."
        ),
        backend=settings.default_backend,
        model=settings.default_model,
        temperature=0.1,
        weight=1.0,
        universe="meta",
        tags=["auditor", "safety", "alignment"],
        use_phi=True,
    )
