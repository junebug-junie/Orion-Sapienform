"""Request identity propagated from Orch MindRun into each LLM gateway call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4


@dataclass(frozen=True)
class MindLLMRequestContext:
    correlation_id: str
    mind_run_id: str
    phase_name: str
    session_id: str | None = None
    trace_id: str | None = None
    router_profile_id: str = "default"
    trigger: str = "user_turn"
    causality_chain: list[Any] | None = None

    def envelope_correlation_id(self) -> UUID:
        try:
            return UUID(str(self.correlation_id))
        except ValueError:
            return uuid4()

    def trace_baggage(self) -> dict[str, Any]:
        baggage: dict[str, Any] = {
            "mind_run_id": self.mind_run_id,
            "mind_phase": self.phase_name,
            "mind_router_profile_id": self.router_profile_id,
            "mind_trigger": self.trigger,
        }
        if self.session_id:
            baggage["session_id"] = self.session_id
        if self.trace_id:
            baggage["trace_id"] = self.trace_id
        return baggage
