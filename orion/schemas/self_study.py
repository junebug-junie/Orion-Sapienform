from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

# Self-study Layer 1 (the daily repo/hardware/behavior scan that fed Self Atlas) was retired
# in 2026-11 (docs/superpowers/specs/2026-10-30-self-atlas-replacement-proposal.md, Patch B),
# after the 2026-10-30 fold into curiosity retired the rest. Its item, snapshot and
# inspect-result models went with it. The only thing left here is the
# write-status shape the analysis sensor (skills.self_study.analyze.v1) still reports.
SelfWritebackState = Literal["written", "skipped", "failed"]


class SelfWritebackStatusV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # "journal" is the analysis sensor's one journal entry. The "self_knowledge_items" target
    # was retired with Self Atlas in 2026-11; "graph" and "self_concept_history" on 2026-10-30.
    target: Literal["journal"]
    status: SelfWritebackState
    authoritative: bool
    channel: str | None = None
    idempotency_key: str | None = None
    append_only: bool = False
    detail: str | None = None
