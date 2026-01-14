from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Tuple

import yaml
import orion
from pydantic import BaseModel

from orion.core.verbs.base import BaseVerb, VerbContext
from orion.core.verbs.models import VerbEffectV1
from orion.core.verbs.registry import verb
from orion.schemas.cortex.schemas import PlanExecutionRequest, PlanExecutionResult

from .router import PlanRouter
from .settings import settings

logger = logging.getLogger("orion.cortex.exec.verb_adapters")
VERBS_DIR = Path(orion.__file__).resolve().parent / "cognition" / "verbs"


@lru_cache(maxsize=128)
def _load_verb_recall_profile(verb_name: str | None) -> str | None:
    if not verb_name:
        return None
    path = VERBS_DIR / f"{verb_name}.yaml"
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        logger.warning("Failed to parse verb yaml for recall profile: %s", path)
        return None
    profile = data.get("recall_profile")
    if isinstance(profile, str):
        profile = profile.strip()
    return profile or None


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
        recall_cfg = ctx_payload.get("recall")
        if not isinstance(recall_cfg, dict):
            recall_cfg = {}

        verb_name = payload.plan.verb_name or ctx.meta.get("verb")
        recall_profile = _load_verb_recall_profile(verb_name)
        if recall_profile and not recall_cfg.get("profile"):
            recall_cfg = {**recall_cfg, "profile": recall_profile}
            ctx_payload["recall"] = recall_cfg

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
