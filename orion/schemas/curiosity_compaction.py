"""Weekly/monthly Curiosity consolidation, admitted by the broader sleep arc.

ADDITIVE DurableWorkflowV1 value: update state consumers and cortex validators
before enabling the dream producer. No new bus channel; uses durable ingress,
the existing cortex verb rail, and journal.entry.write.v1.
"""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

CURIOSITY_COMPACTION_WORKFLOW = "curiosity.compaction"
CURIOSITY_COMPACTION_VERB = "curiosity_compactor_digest_v1"


class CuriosityCompactionBriefV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    compaction_id: str = Field(pattern=r"^curiosity-compaction:(weekly|monthly):\d{4}-\d{2}-\d{2}$")
    llm_route: Literal["agent"] = "agent"
    timeout_sec: float = Field(default=600, gt=30)
    session_id: str = "curiosity-compaction"
