from __future__ import annotations

import logging
from typing import Any, List, Tuple

from pydantic import BaseModel

from orion.core.verbs.base import BaseVerb, VerbContext
from orion.core.verbs.models import VerbEffectV1
from orion.core.verbs.registry import verb
from orion.schemas.cortex.schemas import PlanExecutionRequest, PlanExecutionResult

from .router import PlanRouter
from .settings import settings

logger = logging.getLogger("orion.cortex.exec.verb_adapters")


class LegacyPlanOutput(BaseModel):
    result: PlanExecutionResult


@verb("legacy.plan")
class LegacyPlanVerb(BaseVerb[PlanExecutionRequest, LegacyPlanOutput]):
    input_model = PlanExecutionRequest
    output_model = LegacyPlanOutput

    async def execute(
        self,
        ctx: VerbContext,
        payload: PlanExecutionRequest,
    ) -> Tuple[LegacyPlanOutput, List[VerbEffectV1]]:
        bus = ctx.meta.get("bus")
        source = ctx.meta.get("source")
        correlation_id = str(ctx.meta.get("correlation_id") or payload.args.request_id or "unknown")

        if bus is None or source is None:
            logger.error("LegacyPlanVerb missing bus or source in context meta.")
            return LegacyPlanOutput(
                result=PlanExecutionResult(
                    verb_name=payload.plan.verb_name,
                    request_id=payload.args.request_id,
                    status="fail",
                    blocked=False,
                    blocked_reason=None,
                    steps=[],
                    mode=(payload.args.extra or {}).get("mode"),
                    final_text=None,
                    memory_used=False,
                    recall_debug={},
                    error="missing_execution_context",
                )
            ), []

        payload_context = payload.context or {}
        ctx_payload = {
            **payload_context,
            **(payload.args.extra or {}),
            "user_id": payload.args.user_id,
            "trigger_source": payload.args.trigger_source,
        }

        diagnostic = False
        try:
            extra = payload.args.extra or {}
            options = extra.get("options") if isinstance(extra, dict) else {}
            diagnostic = bool(
                settings.diagnostic_mode
                or extra.get("diagnostic")
                or (isinstance(options, dict) and (options.get("diagnostic") or options.get("diagnostic_mode")))
            )
        except Exception:
            diagnostic = settings.diagnostic_mode

        if diagnostic:
            ctx_payload["diagnostic"] = True

        router = PlanRouter()
        result = await router.run_plan(
            bus,
            source=source,
            req=payload,
            correlation_id=correlation_id,
            ctx=ctx_payload,
        )
        return LegacyPlanOutput(result=result), []
