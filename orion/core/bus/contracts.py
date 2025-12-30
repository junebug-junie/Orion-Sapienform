from __future__ import annotations

"""
Authoritative contract + channel map for the Orion cognitive bus.

These constants are the single source of truth for:
• Bus channels (intake + reply prefixes) used by every cognitive service.
• Envelope kinds used on those channels.
• Reply-channel construction rules.
"""

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class BusChannels:
    """Canonical bus channels used across the cognitive pipeline."""

    # Client/Hub -> Cortex-Orch
    cortex_request: str = "orion-cortex:request"
    cortex_reply_prefix: str = "orion-cortex:reply:"

    # Orch -> Cortex-Exec
    exec_request: str = "orion-cortex-exec:request"
    exec_reply_prefix: str = "orion-cortex-exec:reply:"

    # Exec -> Workers (all workers live under the exec namespace)
    llm_intake: str = "orion-exec:request:LLMGatewayService"
    recall_intake: str = "orion-exec:request:RecallService"
    agent_chain_intake: str = "orion-exec:request:AgentChainService"
    planner_intake: str = "orion-exec:request:PlannerReactService"
    council_intake: str = "orion-exec:request:AgentCouncilService"


@dataclass(frozen=True)
class EnvelopeKinds:
    """Canonical envelope kinds (use these exact strings)."""

    cortex_orch_request: str = "cortex.orch.request"
    cortex_orch_result: str = "cortex.orch.result"

    cortex_exec_request: str = "cortex.exec.request"
    cortex_exec_result: str = "cortex.exec.result"

    llm_chat_request: str = "llm.chat.request"
    llm_chat_result: str = "llm.chat.result"

    recall_query_request: str = "recall.query.request"
    recall_query_result: str = "recall.query.result"

    agent_chain_request: str = "agent.chain.request"
    agent_chain_result: str = "agent.chain.result"

    planner_request: str = "agent.planner.request"
    planner_result: str = "agent.planner.result"

    council_request: str = "agent.council.request"
    council_result: str = "agent.council.result"


CHANNELS = BusChannels()
KINDS = EnvelopeKinds()


def reply_channel(prefix: str, correlation_id: UUID | str) -> str:
    """Construct a reply channel using the canonical prefix + correlation id."""
    return f"{prefix}{correlation_id}"

