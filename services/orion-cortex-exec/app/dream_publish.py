"""
Build and envelope `dream.result.v1` after a successful `dream_cycle` plan run.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.cortex.schemas import PlanExecutionResult
from orion.schemas.telemetry.dream import DreamResultV1

logger = logging.getLogger("orion.cortex.exec.dream_publish")


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _parse_llm_dream_json(final_text: Optional[str]) -> Dict[str, Any]:
    if not final_text:
        return {}
    raw = _strip_json_fence(final_text)
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _recall_profile_from_debug(recall_debug: Any) -> str:
    if isinstance(recall_debug, dict):
        prof = recall_debug.get("profile") or recall_debug.get("resolved_profile")
        if isinstance(prof, str) and prof.strip():
            return prof.strip()
    return "dream.v1"


def build_dream_result_v1(
    res: PlanExecutionResult,
    *,
    correlation_id: str,
    trace_id: Optional[str],
    context: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[DreamResultV1]:
    if (res.verb_name or "").strip() != "dream_cycle":
        return None
    if res.status != "success" or res.blocked:
        return None

    extra = extra or {}
    meta = context.get("metadata") if isinstance(context.get("metadata"), dict) else {}
    trigger_meta: Dict[str, Any] = {}
    if isinstance(meta, dict) and isinstance(meta.get("dream_trigger"), dict):
        trigger_meta = dict(meta["dream_trigger"])
    dream_mode = "standard"
    if isinstance(meta, dict) and meta.get("dream_mode"):
        dream_mode = str(meta["dream_mode"])
    elif extra.get("dream_mode"):
        dream_mode = str(extra["dream_mode"])

    profile = _recall_profile_from_debug(res.recall_debug)
    if isinstance(meta, dict) and isinstance(meta.get("dream_trigger"), dict):
        p = meta["dream_trigger"].get("profile")
        if isinstance(p, str) and p.strip():
            profile = p.strip()

    parsed = _parse_llm_dream_json(res.final_text)
    narrative = (parsed.get("narrative") or res.final_text or "").strip()
    tldr = parsed.get("tldr")
    if not tldr and narrative:
        tldr = narrative[:400]

    themes = parsed.get("themes")
    if not isinstance(themes, list):
        themes = []

    symbols = parsed.get("symbols")
    if not isinstance(symbols, dict):
        symbols = {}

    fragments = parsed.get("fragments")
    if not isinstance(fragments, list):
        fragments = []

    return DreamResultV1(
        mode=dream_mode,
        profile=profile,
        trigger=trigger_meta,
        tldr=tldr,
        themes=[str(x) for x in themes if x is not None],
        symbols={str(k): str(v) for k, v in symbols.items() if v is not None},
        narrative=narrative or None,
        fragments=fragments,
        metrics={},
        recall_debug=res.recall_debug if isinstance(res.recall_debug, dict) else {},
        source_context={"verb": res.verb_name, "request_id": res.request_id},
        correlation_id=correlation_id,
        trace_id=trace_id,
    )


def build_dream_publish_envelope(
    *,
    source: ServiceRef,
    causality_chain: List[Any],
    correlation_id: str | UUID,
    res: PlanExecutionResult,
    context: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[BaseEnvelope]:
    cid_str = str(correlation_id)
    trace_id = None
    if isinstance(context, dict):
        trace_id = context.get("trace_id")
    dr = build_dream_result_v1(
        res,
        correlation_id=cid_str,
        trace_id=str(trace_id) if trace_id else None,
        context=context,
        extra=extra,
    )
    if dr is None:
        return None
    corr_uuid = UUID(cid_str) if isinstance(correlation_id, str) else correlation_id
    return BaseEnvelope(
        kind="dream.result.v1",
        source=source,
        correlation_id=corr_uuid,
        causality_chain=list(causality_chain or []),
        payload=dr.model_dump(mode="json"),
    )
