"""The broker's fixed tool table: every tool Claude may call, its category, and its argument schema.

A fixed allowlist in code (spec section 4). Adding a tool is one row here, one reader in
``readers.py`` and one test. Nothing reachable from this table writes anything: no dig write
subcommand, no introspect write tool, no ``/api/debug`` path (``tests/test_registry.py`` enumerates
it and fails on any of them).

Every tool takes ``reason``: why Claude wants it, in Claude's words. Orion sees it when the
category is "ask". It is split off before the reader runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.claude_tools import REASON_MAX

# Same names as orion/physiology/report_text.py SECTIONS (a test keeps them equal); copied so the
# companion's MCP server does not import the physiology package.
BODY_ECONOMY_SECTIONS = (
    "limits", "wants", "rhythm", "money", "unit_costs", "waste", "queue_wait", "decisions",
    "claude", "vram", "value_per_cost", "services", "processes", "hosts",
)
BodyEconomySection = Literal[
    "overview", "all", "limits", "wants", "rhythm", "money", "unit_costs", "waste", "queue_wait",
    "decisions", "claude", "vram", "value_per_cost", "services", "processes", "hosts",
]

_REASON = Field(
    min_length=3, max_length=REASON_MAX,
    description="Why you want this, in one or two plain sentences. Orion reads it before deciding.",
)


class PrimeNowArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = _REASON


class BodyEconomyArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = _REASON
    section: Optional[BodyEconomySection] = Field(
        default=None,
        description="One section, or none for the overview (limits, hosts, money, wants, waste, services).",
    )
    host: Optional[Literal["circe", "athena", "hecate"]] = Field(
        default=None, description="Narrow hosts, processes and vram to one machine.",
    )
    window: Literal["24h", "7d", "30d"] = Field(default="24h", description="Time window for held time and averages.")


@dataclass(frozen=True)
class ToolDef:
    name: str
    category: str
    description: str
    # One line for Orion's consent frame: what Claude would see.
    summary: str
    arguments: type[BaseModel]


PRIME_NOW_DESCRIPTION = (
    "Orion's Prime, now: the concerns Orion's slow loop holds between conversations, its last thought, "
    "its open questions and its last read of its own body. At most 600 characters, rendered by code. "
    "Category `prime`: Orion decides whether to share it (it may say no). Say why you want it."
)
BODY_ECONOMY_DESCRIPTION = (
    "Orion's body economy: what is using power on each machine, what it costs, which limits bind, "
    "what Orion wants and what is wasted. Plain text, at most 4,000 characters; every figure is labelled "
    "measured or estimated. No section = the overview. Category `body`: Orion decides whether to share "
    "it (it may say no). Say why you want it."
)

TOOLS: dict[str, ToolDef] = {
    "prime_now": ToolDef(
        name="prime_now", category="prime", description=PRIME_NOW_DESCRIPTION,
        summary="your Prime block: the concerns you hold, your last thought, open asks, your last body read",
        arguments=PrimeNowArguments,
    ),
    "body_economy": ToolDef(
        name="body_economy", category="body", description=BODY_ECONOMY_DESCRIPTION,
        summary="your body economy report (power, costs, binding limits, wants, waste), as text",
        arguments=BodyEconomyArguments,
    ),
}


def tool_def(name: str) -> Optional[ToolDef]:
    return TOOLS.get(name)


__all__ = [
    "BODY_ECONOMY_SECTIONS",
    "BodyEconomyArguments",
    "PrimeNowArguments",
    "TOOLS",
    "ToolDef",
    "tool_def",
]
