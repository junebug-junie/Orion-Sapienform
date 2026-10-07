"""The plan ctx cortex-exec builds from an exec request.

Shared by ``main.handle`` (every plan request) and the stance_context_prepare
handler (``app/stance_prepare.py``), which must build stance_react's context
from exactly the ctx the later stance_react request will produce.
"""

from __future__ import annotations

from typing import Any, Dict

from orion.schemas.cortex.schemas import PlanExecutionRequest


def build_exec_ctx(
    *,
    payload_context: Dict[str, Any] | None,
    req: PlanExecutionRequest,
    trace_id: str,
    parent_event_id: Any,
    corr_id: str,
) -> Dict[str, Any]:
    plan_metadata = req.plan.metadata if isinstance(req.plan.metadata, dict) else {}
    ctx: Dict[str, Any] = {
        **(payload_context or {}),
        **(req.args.extra or {}),
        "user_id": req.args.user_id,
        "trigger_source": req.args.trigger_source,
        "trace_id": trace_id,
        "parent_event_id": parent_event_id,
        "correlation_id": corr_id,
        "plan_metadata": plan_metadata,
    }
    if "personality_file" in plan_metadata:
        # Preserve declaration state (including empty string) for precise identity fallback diagnostics.
        ctx["personality_file"] = plan_metadata.get("personality_file")
    ctx.setdefault("trigger_correlation_id", ctx.get("chat_correlation_id") or corr_id)
    ctx.setdefault("trigger_trace_id", trace_id)
    return ctx
