from __future__ import annotations

from typing import List

from app.models import AgentConfig
from app.personas import load_all_personas

import logging
logger = logging.getLogger("agent-council.registry")


class UniverseRegistry:
    """
    Factory / registry that knows which agents live in which 'universe'.

    - Loads all personas from app.personas.*
    - Filters by AgentConfig.universe
    """

    def __init__(self) -> None:
        self._agents: List[AgentConfig] = load_all_personas()
        logger.info("UniverseRegistry initialized with %d agents", len(self._agents))

    @property
    def agents(self) -> List[AgentConfig]:
        return list(self._agents)

    def get_agents_for_universe(self, universe: str | None) -> List[AgentConfig]:
        uni = universe or "core"
        subset = [a for a in self._agents if a.universe == uni]
        if not subset:
            logger.warning(
                "No agents found for universe '%s'; falling back to all (%d)",
                uni,
                len(self._agents),
            )
            return self.agents
        return subset
