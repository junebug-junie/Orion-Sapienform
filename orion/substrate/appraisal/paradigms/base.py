from __future__ import annotations

from typing import Protocol

from orion.schemas.pre_turn_appraisal import PreTurnAppraisalRequestV1, TurnAppraisalParadigmSliceV1


class AppraisalParadigm(Protocol):
    name: str

    async def run(self, req: PreTurnAppraisalRequestV1) -> TurnAppraisalParadigmSliceV1: ...
