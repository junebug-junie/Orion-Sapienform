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
from orion.schemas.introspect import CuriosityArguments, DreamsArguments, OrionDayArguments, ReadingResultArguments

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


class ConsultsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = _REASON
    consult_id: Optional[str] = Field(
        default=None, pattern=r"^[0-9a-f]{12}$",
        description="The 12 hex characters after 'ref consult' on a Hub card, or none for the newest ten.",
    )


class ArcArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = _REASON
    id: str = Field(
        pattern=r"^(?:[0-9a-f]{16}|act_[0-9a-f]{24})$",
        description="An arc id (16 hex characters) or an act id (act_ + 24 hex characters), as Orion cites them.",
    )


# Orion's own introspect read tools, with Claude's ``reason`` added. The reader drops ``reason`` and
# re-validates against the parent model, so the parent's limits are the only limits.
class ClaudeDreamsArguments(DreamsArguments):
    reason: str = _REASON


class ClaudeCuriosityArguments(CuriosityArguments):
    reason: str = _REASON


class ClaudeReadingResultArguments(ReadingResultArguments):
    reason: str = _REASON


class ClaudeOrionDayArguments(OrionDayArguments):
    reason: str = _REASON


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

_ASK = " Orion decides whether to share it (it may say no). Say why you want it."
CONSULTS_DESCRIPTION = (
    "A consult: a conversation Orion had with Claude before bringing a question or proposal to Juniper "
    "(Hub cards say 'ref consult <id>'). With consult_id: its question, status, Juniper's decision and its "
    "turns (each cut to 600 characters). Without: the newest ten. Category `consults`." + _ASK
)
ARC_DESCRIPTION = (
    "One Temporal Self arc (16 hex id) or one act ledger row (act_ + 24 hex), opened with the same resolver "
    "Orion uses before citing it. Category `self`." + _ASK
)
DREAMS_DESCRIPTION = (
    "Orion's dreams: nightly dream stories and the speculative links a sleep cycle proposed between two "
    "memories. query=<topic> finds them by meaning, dream_id=<id> opens one, nothing gives the newest. "
    "Dreams are not established fact. Category `memory`." + _ASK
)
CURIOSITY_DESCRIPTION = (
    "Orion's curiosity runs: the sittings where Orion chose something to look into, what it concluded and "
    "how the run ended. query=<topic> by meaning, run_id=<id> for one write-up, nothing for the newest. "
    "Conclusions are Orion's at the time, not settled fact. Category `memory`." + _ASK
)
READING_RESULTS_DESCRIPTION = (
    "What Orion learned from sources it read (web pages, papers). query=<topic> by meaning, url or "
    "request_id for one reading, nothing for the newest. Source-attributed, not settled belief. "
    "Category `memory`." + _ASK
)
ORION_DAY_DESCRIPTION = (
    "Orion's Day letters (Orion's nightly letter to Juniper) and the records behind them. part=list (default) "
    "is the outline; part=note or carry_forward with index=<N> gives exact words; part=section with "
    "section=<name> gives the records behind one section; query=<topic> searches letters by meaning; "
    "letter_date=YYYY-MM-DD picks a letter. Category `self`." + _ASK
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
    "consults": ToolDef(
        name="consults", category="consults", description=CONSULTS_DESCRIPTION,
        summary="one of your consults with Claude (question, status, Juniper's decision, the turns), or the newest ten",
        arguments=ConsultsArguments,
    ),
    "arc": ToolDef(
        name="arc", category="self", description=ARC_DESCRIPTION,
        summary="one of your Temporal Self arcs or one act ledger row",
        arguments=ArcArguments,
    ),
    "dreams": ToolDef(
        name="dreams", category="memory", description=DREAMS_DESCRIPTION,
        summary="your dreams (stories and sleep-cycle links), full text up to 1,200 characters each",
        arguments=ClaudeDreamsArguments,
    ),
    "curiosity": ToolDef(
        name="curiosity", category="memory", description=CURIOSITY_DESCRIPTION,
        summary="your curiosity runs: what you chose to look into and what you concluded",
        arguments=ClaudeCuriosityArguments,
    ),
    "reading_results": ToolDef(
        name="reading_results", category="memory", description=READING_RESULTS_DESCRIPTION,
        summary="what you learned from sources you read",
        arguments=ClaudeReadingResultArguments,
    ),
    "orion_day": ToolDef(
        name="orion_day", category="self", description=ORION_DAY_DESCRIPTION,
        summary="your Orion's Day letters to Juniper and the records behind them",
        arguments=ClaudeOrionDayArguments,
    ),
}

# The introspect read tools: Claude's tool name -> (Orion's introspect tool, its own argument model).
INTROSPECT_TOOLS: dict[str, tuple[str, type[BaseModel]]] = {
    "dreams": ("dreams", DreamsArguments),
    "curiosity": ("curiosity", CuriosityArguments),
    "reading_results": ("reading_results", ReadingResultArguments),
    "orion_day": ("orion_day", OrionDayArguments),
}


def tool_def(name: str) -> Optional[ToolDef]:
    return TOOLS.get(name)


__all__ = [
    "BODY_ECONOMY_SECTIONS",
    "INTROSPECT_TOOLS",
    "ArcArguments",
    "BodyEconomyArguments",
    "ConsultsArguments",
    "PrimeNowArguments",
    "TOOLS",
    "ToolDef",
    "tool_def",
]
