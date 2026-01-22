# services/orion-cortex-exec/app/executor.py
from __future__ import annotations

"""
Core execution engine for cortex-exec.
Handles recall, planner-react, agent-chain, and LLM Gateway hops over the bus.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple
from uuid import uuid4

from jinja2 import Environment

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef
from orion.core.contracts.recall import RecallQueryV1

from orion.schemas.agents.schemas import AgentChainRequest, DeliberationRequest
from orion.core.verbs import VerbResultV1
from orion.schemas.collapse_mirror import CollapseMirrorEntryV2, find_collapse_entry, normalize_collapse_entry
from orion.schemas.cortex.schemas import ExecutionStep, StepExecutionResult
from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1
from orion.schemas.telemetry.spark_signal import SparkSignalV1
from orion.schemas.pad.v1 import KIND_PAD_RPC_REQUEST_V1, PadRpcRequestV1, PadRpcResponseV1
from orion.schemas.telemetry.spark import SparkStateSnapshotV1, SparkTelemetryPayload
from orion.schemas.telemetry.system_health import EquilibriumSnapshotV1
from orion.schemas.state.contracts import StateGetLatestRequest, StateLatestReply

from .settings import settings
from .clients import AgentChainClient, LLMGatewayClient, RecallClient, PlannerReactClient
from .trace_cache import get_trace_cache
from .spark_narrative import spark_phi_hint, spark_phi_narrative

logger = logging.getLogger("orion.cortex.exec")


def _default_biometrics_context(*, status: str, reason: str) -> Dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "as_of": None,
        "freshness_s": None,
        "constraint": "NONE",
        "cluster": {
            "composite": {"strain": 0.0, "homeostasis": 0.0, "stability": 1.0},
            "trend": {
                "strain": {"trend": 0.5, "volatility": 0.0, "spike_rate": 0.0},
                "homeostasis": {"trend": 0.5, "volatility": 0.0, "spike_rate": 0.0},
                "stability": {"trend": 0.5, "volatility": 0.0, "spike_rate": 0.0},
            },
        },
        "nodes": {},
        "summary": None,
        "induction": None,
    }


def _metacog_messages(prompt: str) -> List[Dict[str, Any]]:
    """
    Force "schema-mode": system prompt + explicit user instruction.
    Avoids the model going into explanation/chat mode.
    """
    return [
        {"role": "system", "content": str(prompt or "").strip() or " "},
        {
            "role": "user",
            "content": (
                "Return ONE valid JSON object only (CollapseMirrorEntryV2). "
                "No explanation, no markdown, no extra text."
            ),
        },
    ]


def _fallback_metacog_draft(ctx: Dict[str, Any]) -> CollapseMirrorEntryV2:
    """
    If the LLM returns non-JSON, produce a valid baseline draft so the pipeline continues.
    """
    trig = ctx.get("trigger") or {}
    trigger_kind = str(trig.get("trigger_kind") or "unknown")
    reason = str(trig.get("reason") or "unknown")
    pressure = trig.get("pressure")
    zen_state = str(trig.get("zen_state") or "unknown")

    hint = ctx.get("phi_hint") or {}
    vb = str(hint.get("valence_band") or "unknown")
    vd = str(hint.get("valence_dir") or "unknown")
    eb = str(hint.get("energy_band") or "unknown")
    cb = str(hint.get("coherence_band") or "unknown")
    nb = str(hint.get("novelty_band") or "unknown")

    # crude but stable type guess
    typ = "idle"
    if cb == "low" or nb == "high":
        typ = "turbulence"
    elif eb in ("moderate", "high") and cb in ("medium", "high"):
        typ = "flow"

    observer_state = [
        "metacog",
        f"energy:{eb}",
        f"clarity:{cb}",
        f"overload:{nb}",
        f"valence:{vb}",
    ]

    field_res = f"φ:{vb}-{vd}, energy:{eb}, clarity:{cb}, overload:{nb}; zen={zen_state}"

    entry = CollapseMirrorEntryV2(
        event_id=f"collapse_{uuid4().hex}",
        id=None,
        observer="orion",
        trigger=trigger_kind,
        observer_state=observer_state,
        field_resonance=field_res,
        type=typ,
        emergent_entity="Fallback Baseline",
        summary=f"Fallback mirror draft. Trigger={trigger_kind} ({reason}); {field_res}.",
        mantra="Compress truth; keep the imprint.",
        causal_echo=None,
        timestamp=None,  # system fills
        environment=None,
        snapshot_kind="baseline",
        what_changed_summary="fallback_generated",
        what_changed={
            "summary": "LLM returned non-JSON; fallback draft constructed.",
            "previous_state": None,
            "new_state": None,
            "evidence": [
                f"trigger={trigger_kind}",
                f"pressure={pressure}",
                f"phi={vb}-{vd} energy={eb} clarity={cb} overload={nb}",
            ],
        },
        pattern_candidate=None,
        resonance_signature=f"{typ}: Fallback Baseline | Δ:fallback_generated | →observe",
        change_type=None,
        change_type_scores={},
        tag_scores={},
        tags=[typ] if typ != "idle" else [],
        numeric_sisters={
            "valence": None,
            "arousal": None,
            "clarity": None,
            "overload": None,
            "risk_score": None,
        },
        causal_density={"label": None, "score": None, "rationale": None},
        is_causally_dense=False,
        epistemic_status="observed",
        visibility="internal",
        redaction_level="low",
        source_service=None,
        source_node=None,
    )
    return entry.with_defaults()


def _trace_meta_from_ctx(
    ctx: Dict[str, Any],
    *,
    event_id: str,
    parent_event_id: str | None,
    source: ServiceRef,
) -> Dict[str, Any]:
    trace_id = str(ctx.get("trace_id") or ctx.get("correlation_id") or "")
    return {
        "trace_id": trace_id,
        "event_id": event_id,
        "parent_event_id": parent_event_id,
        "source_service": source.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }



def _render_prompt(template_str: str, ctx: Dict[str, Any]) -> str:
    env = Environment(autoescape=False)

    # [FIX] Defensive coding: Prevent Jinja crash on missing globals
    render_ctx = ctx.copy()
    defaults = {
        "prompt_templates": {},
        "collapse_entry": {"event_id": "unknown_missing_draft"},
        "collapse_json": "{}",
        "trigger": {"trigger_kind": "unknown", "reason": "unknown", "pressure": 0.0, "zen_state": "unknown"},
        "context_summary": "Context missing.",
        "spark_state_json": "{}",
        "spark_phi_narrative": "",
        "phi_hint": None,
    }
    for k, v in defaults.items():
        if k not in render_ctx:
            render_ctx[k] = v

    tmpl = env.from_string(template_str or "")
    return tmpl.render(**render_ctx)


def _last_user_message(ctx: Dict[str, Any]) -> str:
    msgs = ctx.get("messages") or []
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content") or "")
            if isinstance(m, LLMMessage) and getattr(m, "role", None) == "user":
                return str(getattr(m, "content", "") or "")
            if hasattr(m, "model_dump"):
                try:
                    d = m.model_dump(mode="json")
                    if isinstance(d, dict) and d.get("role") == "user":
                        return str(d.get("content") or "")
                except Exception:
                    pass
    return str(ctx.get("user_message") or "")


def _json_sanitize(obj: Any, *, _seen: set[int] | None = None, _depth: int = 0, _max_depth: int = 10) -> Any:
    if _seen is None:
        _seen = set()

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    oid = id(obj)
    if oid in _seen:
        return "<circular_ref>"
    if _depth >= _max_depth:
        return "<max_depth>"

    if isinstance(obj, dict):
        _seen.add(oid)
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            out[str(k)] = _json_sanitize(v, _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth)
        _seen.remove(oid)
        return out

    if isinstance(obj, (list, tuple, set)):
        _seen.add(oid)
        out_list = [_json_sanitize(v, _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth) for v in obj]
        _seen.remove(oid)
        return out_list

    if isinstance(obj, LLMMessage):
        return _json_sanitize(obj.model_dump(mode="json"), _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth)

    if hasattr(obj, "model_dump"):
        try:
            return _json_sanitize(obj.model_dump(mode="json"), _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth)
        except Exception:
            return str(obj)

    return str(obj)


def _build_hop_messages(
    *,
    prompt: str,
    ctx_messages: Any,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []

    raw_msgs = ctx_messages or []
    if isinstance(raw_msgs, list):
        for m in raw_msgs:
            if isinstance(m, dict):
                normalized.append(_json_sanitize(m))
            elif isinstance(m, LLMMessage):
                normalized.append(_json_sanitize(m.model_dump(mode="json")))
            elif hasattr(m, "model_dump"):
                try:
                    normalized.append(_json_sanitize(m.model_dump(mode="json")))
                except Exception:
                    pass

    guardrail = "If no memory is provided, do not claim prior work; say you don't know."
    prompt_content = str(prompt or "").strip()
    if guardrail.lower() not in prompt_content.lower():
        prompt_content = f"{prompt_content}\n\n{guardrail}".strip()

    if prompt_content:
        sys_msg = {"role": "system", "content": prompt_content}
        if normalized and isinstance(normalized[0], dict) and normalized[0].get("role") == "system":
            normalized[0] = sys_msg
        else:
            normalized = [sys_msg] + normalized

    if not normalized:
        content = prompt_content or " "
        normalized = [{"role": "user", "content": content}]

    return normalized


def _extract_llm_text(res: Any) -> str:
    """Safely extract text content from various LLM result shapes."""
    if not res:
        return ""

    if isinstance(res, dict):
        try:
            return json.dumps(res)
        except Exception:
            return str(res)

    if hasattr(res, "choices") and res.choices:
        try:
            return str(res.choices[0].message.content)
        except (AttributeError, IndexError):
            pass

    if hasattr(res, "content") and res.content:
        return str(res.content)

    if hasattr(res, "message") and res.message:
        if hasattr(res.message, "content"):
            return str(res.message.content)
        if isinstance(res.message, dict):
            return str(res.message.get("content", ""))

    if hasattr(res, "model_dump"):
        try:
            d = res.model_dump(mode="json")
            if "content" in d:
                return str(d["content"])
            if "choices" in d and d["choices"]:
                return str(d["choices"][0]["message"]["content"])
        except Exception:
            pass

    if isinstance(res, list):
        try:
            return json.dumps(res)
        except Exception:
            return str(res)

    return str(res)


def _loose_json_extract(text: str) -> Dict[str, Any] | None:
    """
    Fallback extraction when strict find_collapse_entry fails.
    Finds the first outer { and last } and tries to parse.
    """
    if not text:
        return None
    extracted = _extract_first_json_object(text)
    if isinstance(extracted, dict):
        return extracted
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            json_str = text[start : end + 1]
            return json.loads(json_str)
    except Exception:
        pass
    return None


def _extract_first_json_object(text: str) -> Dict[str, Any] | None:
    if not text or "{" not in text:
        return None

    starts = [i for i, ch in enumerate(text) if ch == "{"]

    for s in starts:
        depth = 0
        in_str = False
        esc = False

        for e in range(s, len(text)):
            ch = text[e]

            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[s : e + 1].strip()
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        pass
                    break

    return None


async def run_recall_step(
    bus: OrionBusAsync,
    *,
    source: ServiceRef,
    ctx: Dict[str, Any],
    correlation_id: str,
    recall_cfg: Dict[str, Any],
    recall_profile: str | None = None,
    step_name: str = "recall",
    step_order: int = -1,
    diagnostic: bool = False,
) -> Tuple[StepExecutionResult, Dict[str, Any], str]:
    t0 = time.time()
    recall_client = RecallClient(bus)
    reply_channel = f"orion:exec:result:RecallService:{uuid4()}"

    recall_timeout = float(settings.step_timeout_ms) / 1000.0

    fragment = _last_user_message(ctx) or ""
    trace_val = ctx.get("trace_id") or recall_cfg.get("trace_id") or correlation_id
    req = RecallQueryV1(
        fragment=fragment,
        verb=str(ctx.get("verb") or recall_cfg.get("verb") or "unknown"),
        intent=ctx.get("intent"),
        session_id=ctx.get("session_id"),
        node_id=ctx.get("node_id"),
        profile=recall_profile or recall_cfg.get("profile") or "reflect.v1",
        reply_to=reply_channel,
    )

    logs: List[str] = [f"rpc -> RecallService (profile={req.profile})"]
    debug: Dict[str, Any] = {}
    try:
        res = await recall_client.query(
            source=source,
            req=req,
            correlation_id=correlation_id,
            reply_to=reply_channel,
            timeout_sec=recall_timeout,
        )
        bundle = res.bundle
        debug = {
            "count": len(bundle.items),
            "profile": req.profile,
            "error": None,
        }
        memory_digest = bundle.rendered if hasattr(bundle, "rendered") else ""
        ctx["memory_digest"] = memory_digest
        ctx["memory_bundle"] = bundle.model_dump(mode="json")
        ctx["memory_used"] = True
        ctx["recall_fragments"] = [i.model_dump(mode="json") for i in bundle.items]
        logs.append(f"ok <- RecallService ({len(bundle.items)} items)")

        return (
            StepExecutionResult(
                status="success",
                verb_name=str(ctx.get("verb") or "unknown"),
                step_name=step_name,
                order=step_order,
                result={"RecallService": debug},
                latency_ms=int((time.time() - t0) * 1000),
                node=settings.node_name,
                logs=logs,
            ),
            debug,
            memory_digest,
        )
    except Exception as e:
        logs.append(f"exception <- RecallService: {e}")
        debug["error"] = str(e)
        return (
            StepExecutionResult(
                status="fail",
                verb_name=str(ctx.get("verb") or "unknown"),
                step_name=step_name,
                order=step_order,
                result={"RecallService": debug},
                latency_ms=int((time.time() - t0) * 1000),
                node=settings.node_name,
                logs=logs,
                error=str(e),
            ),
            debug,
            "",
        )


async def call_step_services(
    bus: OrionBusAsync,
    *,
    source: ServiceRef,
    step: ExecutionStep,
    ctx: Dict[str, Any],
    correlation_id: str,
    diagnostic: bool = False,
) -> StepExecutionResult:
    t0 = time.time()
    logs: List[str] = []
    merged_result: Dict[str, Any] = {}
    spark_vector: list[float] | None = None

    logger.info(f"--- EXEC STEP '{step.step_name}' START ---")
    logger.info(f"Context Keys available: {list(ctx.keys())}")

    step_timeout_sec = (step.timeout_ms or 60000) / 1000.0
    effective_timeout = step_timeout_sec

    llm_client = LLMGatewayClient(bus)
    planner_client = PlannerReactClient(bus)
    agent_client = AgentChainClient(bus)

    for service in step.services:
        reply_channel = f"orion:exec:result:{service}:{uuid4()}"

        # IMPORTANT: render prompts per-service so MetacogContextService mutations take effect
        prompt = _render_prompt(step.prompt_template or "", ctx) if step.prompt_template else ""

        # ---- DEBUG BY EYE ----
        if service in {"MetacogDraftService", "MetacogEnrichService"}:
            logger.info(f"[PROMPT] service={service} chars={len(prompt)}")
            logger.info(f"[PROMPT_HEAD] {prompt[:500]!r}")
            logger.info(f"[PROMPT_TAIL] {prompt[-500:]!r}")
            # also useful: show which ctx keys exist
            logger.info(f"[CTX_KEYS] {sorted(list(ctx.keys()))}")
            # show lengths of the likely “balloon” fields
            for k in ("context_summary", "spark_state_json", "collapse_json", "memory_digest"):
                v = ctx.get(k)
                if isinstance(v, str):
                    logger.info(f"[CTX_LEN] {k}={len(v)}")
         # ----------------------


        if service in {"MetacogDraftService", "MetacogEnrichService"}:
            debug_prompt = (prompt[:200] + "...") if len(prompt) > 200 else prompt
            logger.info(f"Rendered Prompt[{service}]: {debug_prompt!r}")

        try:
            if service == "MetacogDraftService":
                logs.append("exec -> MetacogDraftService (LLM + Parse)")

                req_model = ctx.get("model") or ctx.get("llm_model") or None
                messages_payload = _metacog_messages(prompt)
                #messages_payload = _build_hop_messages(prompt=prompt, ctx_messages=ctx.get("messages"))

                request_object = ChatRequestPayload(
                    model=req_model,
                    messages=messages_payload,
                    raw_user_text=ctx.get("raw_user_text") or _last_user_message(ctx),
                    route="metacog",
                    options={
                        "temperature": 0.8,
                        "max_tokens": 1024,
                        "response_format": {"type": "json_object"},
                        "stream": False,
                    },
                )

                llm_res = await llm_client.chat(
                    source=source,
                    req=request_object,
                    correlation_id=correlation_id,
                    reply_to=reply_channel,
                    timeout_sec=effective_timeout,
                )

                try:
                    raw_content = _extract_llm_text(llm_res)

                    # 1) Try strict find
                    parsed = find_collapse_entry(raw_content)

                    # 2) Fallback: loose extraction
                    if not parsed:
                        parsed = _loose_json_extract(raw_content)

                    # 3) Handle Result or Fallback
                    if not parsed:
                        logger.error(f"MetacogDraftService: No JSON found. Raw Content: {raw_content!r}")
                        # Use the fallback model directly
                        entry = _fallback_metacog_draft(ctx)
                    else:
                        # We have a dictionary, ensure required fields exist before normalizing
                        parsed["observer"] = "orion"
                        parsed["visibility"] = "internal"
                        parsed["epistemic_status"] = "observed"

                        # Convert dictionary to Pydantic model
                        entry = normalize_collapse_entry(parsed)

                    # At this point, 'entry' is guaranteed to be a CollapseMirrorEntryV2 object
                    entry_dict = entry.model_dump(mode="json")

                    # Update Context
                    ctx["collapse_entry"] = entry_dict
                    ctx["collapse_json"] = json.dumps(entry_dict, ensure_ascii=False)

                    merged_result[service] = {"ok": True, "event_id": entry.event_id}
                    logs.append("ok <- MetacogDraftService")

                except Exception as e:
                    logger.error(f"MetacogDraftService FAILED: {e}")
                    logs.append(f"error <- MetacogDraftService parsing: {e}")
                    merged_result[service] = {"ok": False, "error": str(e)}


                continue

            if service == "MetacogEnrichService":
                logs.append("exec -> MetacogEnrichService (LLM + Merge)")

                req_model = ctx.get("model") or ctx.get("llm_model") or None
                #messages_payload = _build_hop_messages(prompt=prompt, ctx_messages=ctx.get("messages"))
                messages_payload = _metacog_messages(prompt)

                request_object = ChatRequestPayload(
                    model=req_model,
                    messages=messages_payload,
                    raw_user_text="metacog_enrich",
                    route="metacog",
                    options={
                        "temperature": 0.5,
                        "max_tokens": 1024,
                        "stream": False,
                        "response_format": {"type": "json_object"}
                    },
                )

                llm_res = await llm_client.chat(
                    source=source,
                    req=request_object,
                    correlation_id=correlation_id,
                    reply_to=reply_channel,
                    timeout_sec=effective_timeout,
                )

                try:
                    raw_content = _extract_llm_text(llm_res)

                    # 1) strict find
                    patch = find_collapse_entry(raw_content)

                    # 2) fallback loose extract
                    if not patch:
                        patch = _loose_json_extract(raw_content)

                    if isinstance(patch, dict) and isinstance(patch.get("draft"), dict):
                        patch = patch["draft"]

                    if not patch:
                        logger.warning(f"MetacogEnrichService: No JSON found. Raw: {raw_content!r}")
                        patch = {}

                    draft_data = ctx.get("collapse_entry")
                    if not draft_data:
                        raise ValueError("No draft entry found in context")

                    draft = CollapseMirrorEntryV2.model_validate(draft_data)
                    final_dict = draft.model_dump(mode="json")

                    # Allow enricher to supply these (and only these) fields
                    for k in [
                        "numeric_sisters",
                        "causal_density",
                        "is_causally_dense",
                        "what_changed",
                        "what_changed_summary",
                        "change_type",
                        "change_type_scores",
                        "tag_scores",
                        "tags",
                        "state_snapshot",
                        "pattern_candidate",
                        "resonance_signature",
                    ]:
                        if k in patch and patch[k] is not None:
                            final_dict[k] = patch[k]

                    # Coerce resonance_signature to string if LLM returned an object
                    rs = final_dict.get("resonance_signature")
                    if rs is not None and not isinstance(rs, str):
                        try:
                            final_dict["resonance_signature"] = json.dumps(rs)
                        except Exception:
                            final_dict["resonance_signature"] = str(rs)

                    if isinstance(final_dict.get("change_type"), dict):
                        trace_id = ctx.get("trace_id") or correlation_id
                        logger.warning(
                            "MetacogEnrichService: change_type emitted as dict "
                            f"correlation_id={correlation_id} trace_id={trace_id}"
                        )

                    final_entry = CollapseMirrorEntryV2.model_validate(final_dict)

                    ctx["final_entry"] = final_entry.model_dump(mode="json")
                    merged_result[service] = {"ok": True, "event_id": final_entry.event_id}
                    logs.append("ok <- MetacogEnrichService")

                except Exception as e:
                    logger.error(f"MetacogEnrichService FAILED: {e}")
                    logs.append(f"error <- MetacogEnrichService: {e}")
                    merged_result[service] = {"ok": False, "error": str(e)}

                continue

            if service == "MetacogPublishService":
                logs.append("exec -> MetacogPublishService")

                final_data = ctx.get("final_entry") or ctx.get("collapse_entry")
                if not final_data:
                    logs.append("skip <- MetacogPublishService (no entry)")
                    merged_result[service] = {"ok": False, "reason": "no_entry"}
                    continue

                try:
                    entry = CollapseMirrorEntryV2.model_validate(final_data)
                    event_id = str(uuid4())
                    trace_meta = _trace_meta_from_ctx(
                        ctx,
                        event_id=event_id,
                        parent_event_id=ctx.get("parent_event_id"),
                        source=source,
                    )

                    env = BaseEnvelope(
                        kind="collapse.mirror.entry.v2",
                        source=source,
                        correlation_id=correlation_id,
                        trace=trace_meta,
                        payload=entry.model_dump(mode="json"),
                    )

                    await bus.publish(settings.channel_collapse_sql_write, env)
                    logger.info(
                        f"MetacogPublishService published channel={settings.channel_collapse_sql_write} "
                        f"trace_id={trace_meta.get('trace_id')} event_id={event_id}"
                    )
                    merged_result[service] = {
                        "ok": True,
                        "published": True,
                        "channel": settings.channel_collapse_sql_write,
                        "event_id": entry.event_id,
                    }
                    logs.append("ok <- MetacogPublishService (SQL)")

                except Exception as e:
                    logger.error(f"MetacogPublishService FAILED: {e}")
                    logs.append(f"error <- MetacogPublishService: {e}")
                    merged_result[service] = {"ok": False, "error": str(e)}

                continue

            if service == "MetacogContextService":
                logs.append("exec -> MetacogContextService")

                trigger_data = ctx.get("trigger") or ctx.get("args", {}).get("trigger", {})

                try:
                    trigger = MetacogTriggerV1.model_validate(trigger_data)
                except Exception:
                    trigger = MetacogTriggerV1(trigger_kind="unknown", reason="deserialization_failed")

                pad_summary = "unknown"
                spark_line = "unknown"
                trace_summary = "unknown"

                # Landing Pad
                pad_reply_channel = f"orion:exec:result:PadRpc:{uuid4()}"
                pad_req = PadRpcRequestV1(
                    request_id=correlation_id,
                    reply_channel=pad_reply_channel,
                    method="get_latest_frame",
                    args={},
                )
                pad_env = BaseEnvelope(
                    kind=KIND_PAD_RPC_REQUEST_V1,
                    source=source,
                    correlation_id=correlation_id,
                    reply_to=pad_reply_channel,
                    payload=pad_req.model_dump(mode="json"),
                )
                try:
                    pad_msg = await bus.rpc_request(
                        settings.channel_pad_rpc_request,
                        pad_env,
                        reply_channel=pad_reply_channel,
                        timeout_sec=20.0,
                    )
                    pad_dec = bus.codec.decode(pad_msg.get("data"))
                    if pad_dec.ok:
                        pad_res = PadRpcResponseV1.model_validate(pad_dec.envelope.payload)
                        pad_summary = str(pad_res.result)
                except Exception as e:
                    logger.warning("MetacogContextService pad RPC failed: %s", e)
                    pad_summary = f"error: {e}"

                # Spark (State Service)
                state_reply_channel = f"orion:exec:result:StateService:{uuid4()}"
                state_req = StateGetLatestRequest(scope="global")
                state_env = BaseEnvelope(
                    kind="state.get_latest.v1",
                    source=source,
                    correlation_id=correlation_id,
                    reply_to=state_reply_channel,
                    payload=state_req.model_dump(mode="json"),
                )
                biometrics_context = _default_biometrics_context(
                    status="NO_SIGNAL",
                    reason="no_state_reply",
                )
                try:
                    state_msg = await bus.rpc_request(
                        settings.channel_state_request,
                        state_env,
                        reply_channel=state_reply_channel,
                        timeout_sec=20.0,
                    )
                    state_dec = bus.codec.decode(state_msg.get("data"))
                    if state_dec.ok:
                        state_res = StateLatestReply.model_validate(state_dec.envelope.payload)
                        if state_res.biometrics:
                            if hasattr(state_res.biometrics, "model_dump"):
                                raw_biometrics = state_res.biometrics.model_dump(mode="json")
                            elif isinstance(state_res.biometrics, dict):
                                raw_biometrics = state_res.biometrics
                            else:
                                raw_biometrics = {}
                            biometrics_context = _default_biometrics_context(
                                status="OK",
                                reason="state_service",
                            )
                            if isinstance(raw_biometrics, dict):
                                biometrics_context["summary"] = raw_biometrics.get("summary")
                                biometrics_context["induction"] = raw_biometrics.get("induction")
                                if raw_biometrics.get("constraint"):
                                    biometrics_context["constraint"] = raw_biometrics.get("constraint")
                                if raw_biometrics.get("freshness_s") is not None:
                                    biometrics_context["freshness_s"] = raw_biometrics.get("freshness_s")
                                if raw_biometrics.get("as_of") is not None:
                                    biometrics_context["as_of"] = raw_biometrics.get("as_of")
                                if raw_biometrics.get("status"):
                                    biometrics_context["status"] = raw_biometrics.get("status")
                                if raw_biometrics.get("reason"):
                                    biometrics_context["reason"] = raw_biometrics.get("reason")
                                nodes = raw_biometrics.get("nodes")
                                if isinstance(nodes, dict):
                                    biometrics_context["nodes"] = nodes
                        ctx["biometrics"] = biometrics_context
                        ctx["biometrics_json"] = json.dumps(biometrics_context, indent=2)
                        if state_res.ok and state_res.snapshot:
                            snap_obj = state_res.snapshot

                            spark_snap: SparkStateSnapshotV1 | None = None
                            try:
                                if isinstance(snap_obj, SparkStateSnapshotV1):
                                    spark_snap = snap_obj
                                elif isinstance(snap_obj, dict):
                                    spark_snap = SparkStateSnapshotV1.model_validate(snap_obj)
                                elif hasattr(snap_obj, "model_dump"):
                                    spark_snap = SparkStateSnapshotV1.model_validate(snap_obj.model_dump(mode="json"))
                            except Exception as exc:
                                spark_snap = None
                                spark_line = f"snapshot_unparseable: {exc}"

                            if spark_snap:
                                # Provide prompt-ready vars
                                phi_hint = spark_phi_hint(spark_snap)
                                ctx["phi_hint"] = phi_hint
                                ctx["spark_phi_narrative"] = spark_phi_narrative(spark_snap)
                                ctx["spark_state_json"] = spark_snap.model_dump_json()

                                spark_line = (
                                    f"φ={phi_hint.get('valence_band','?')}-{phi_hint.get('valence_dir','?')}, "
                                    f"energy={phi_hint.get('energy_band','?')}, "
                                    f"clarity={phi_hint.get('coherence_band','?')}, "
                                    f"overload={phi_hint.get('novelty_band','?')} "
                                    f"(seq={spark_snap.seq}, ts={spark_snap.snapshot_ts})"
                                )
                        else:
                            spark_line = f"stale/missing (status={state_res.status})"
                except Exception as e:
                    logger.warning("MetacogContextService state RPC failed: %s", e)
                    spark_line = f"error: {e}"
                    biometrics_context = _default_biometrics_context(
                        status="NO_SIGNAL",
                        reason=f"state_rpc_error:{e}",
                    )
                    ctx["biometrics"] = biometrics_context
                    ctx["biometrics_json"] = json.dumps(biometrics_context, indent=2)
                if "biometrics" not in ctx:
                    ctx["biometrics"] = biometrics_context
                    ctx["biometrics_json"] = json.dumps(biometrics_context, indent=2)

                # Recent Traces
                recent_traces = get_trace_cache().get_recent(5)
                if recent_traces:
                    trace_summary = "\n".join(
                        [f"- [{t.mode}] {t.verb}: {(t.final_text or '')[:100]}..." for t in recent_traces]
                    )
                else:
                    trace_summary = "None available."

                # Keep context_summary human-readable (spark JSON is separate in spark_state_json)
                pad_short = pad_summary
                if isinstance(pad_short, str) and len(pad_short) > 500:
                    pad_short = pad_short[:500] + "...(truncated)"

                biometrics_context = ctx.get("biometrics") or {}
                summary = biometrics_context.get("summary") or {}
                composites = summary.get("composites") or {}
                pressures = summary.get("pressures") or {}
                strain = composites.get("strain")
                biometrics_line = f"status={biometrics_context.get('status','missing')}"
                if strain is not None:
                    biometrics_line += f", strain={float(strain):.2f}"
                if pressures:
                    cpu_p = pressures.get("cpu")
                    gpu_p = pressures.get("gpu_util")
                    if cpu_p is not None:
                        biometrics_line += f", cpu={float(cpu_p):.2f}"
                    if gpu_p is not None:
                        biometrics_line += f", gpu={float(gpu_p):.2f}"

                summary_text = (
                    f"Trigger: {trigger.trigger_kind} ({trigger.reason})\n"
                    f"Pressure: {trigger.pressure}\n"
                    f"Landing Pad: {pad_short}\n"
                    f"Spark: {spark_line}\n"
                    f"Biometrics: {biometrics_line}\n"
                    f"Recent Traces:\n{trace_summary}\n"
                )

                ctx["trigger"] = trigger.model_dump(mode="json")
                ctx["context_summary"] = summary_text
                merged_result[service] = {"ok": True, "summary_len": len(summary_text)}
                logs.append("ok <- MetacogContextService")
                continue

            if service == "RecallService":
                logs.append(f"rpc -> RecallService (reply={reply_channel}, profile={step.recall_profile})")
                recall_step, recall_debug, memory_digest = await run_recall_step(
                    bus,
                    source=source,
                    ctx=ctx,
                    correlation_id=correlation_id,
                    recall_cfg=ctx.get("recall") or {},
                    recall_profile=step.recall_profile,
                    step_name=step.step_name,
                    step_order=step.order,
                    diagnostic=diagnostic,
                )
                merged_result["RecallService"] = recall_debug
                logs.extend(recall_step.logs)
                return recall_step

            if service == "LLMGatewayService":
                req_model = ctx.get("model") or ctx.get("llm_model") or None
                memory_digest = (ctx.get("memory_digest") or "").strip()
                if memory_digest:
                    prompt = f"{prompt}\n\n# RELEVANT MEMORY (retrieved)\n{memory_digest}\n"
                if diagnostic:
                    logger.info(
                        "memory_digest_present=%s memory_digest_chars=%s",
                        bool(memory_digest),
                        len(memory_digest),
                    )
                messages_payload = _build_hop_messages(prompt=prompt, ctx_messages=ctx.get("messages"))

                request_object = ChatRequestPayload(
                    model=req_model,
                    messages=messages_payload,
                    raw_user_text=ctx.get("raw_user_text") or _last_user_message(ctx),
                    options={
                        "temperature": float(ctx.get("temperature", 0.7)),
                        "max_tokens": int(ctx.get("max_tokens", 512)),
                        "stream": False,
                    },
                )

                logs.append(f"rpc -> LLMGateway via client (timeout={effective_timeout}s)")
                result_object = await llm_client.chat(
                    source=source,
                    req=request_object,
                    correlation_id=correlation_id,
                    reply_to=reply_channel,
                    timeout_sec=effective_timeout,
                )

                merged_result[service] = result_object.model_dump(mode="json")

                if spark_vector is None:
                    try:
                        _sv = getattr(result_object, "spark_vector", None)
                    except Exception:
                        _sv = None
                    if _sv:
                        spark_vector = _sv
                logs.append(f"ok <- {service}")

            elif service == "AgentChainService":
                hop_msgs = _build_hop_messages(prompt=prompt, ctx_messages=ctx.get("messages"))

                agent_req = AgentChainRequest(
                    text=_last_user_message(ctx),
                    mode=ctx.get("mode") or "agent",
                    session_id=ctx.get("session_id"),
                    user_id=ctx.get("user_id"),
                    messages=[LLMMessage(**m) if not isinstance(m, LLMMessage) else m for m in (hop_msgs or [])],
                    packs=ctx.get("packs") or [],
                )
                logs.append(f"rpc -> AgentChainService (reply={reply_channel}, timeout={effective_timeout}s)")
                agent_res = await agent_client.run_chain(
                    source=source,
                    req=agent_req,
                    correlation_id=correlation_id,
                    reply_to=reply_channel,
                    timeout_sec=effective_timeout,
                )
                merged_result[service] = agent_res.model_dump(mode="json")
                logs.append("ok <- AgentChainService")

            elif service == "PlannerReactService":
                planner_req = PlannerRequest(
                    request_id=str(correlation_id),
                    caller="cortex-exec",
                    goal=Goal(description=_last_user_message(ctx), metadata={"verb": step.verb_name}),
                    context=ContextBlock(conversation_history=[LLMMessage(**m) for m in (ctx.get("messages") or [])]),
                    toolset=[],
                )
                logs.append(f"rpc -> PlannerReactService (timeout={effective_timeout}s)")
                planner_res = await planner_client.plan(
                    source=source,
                    req=planner_req,
                    correlation_id=correlation_id,
                    reply_to=reply_channel,
                    timeout_sec=effective_timeout,
                )
                merged_result[service] = planner_res.model_dump(mode="json")
                logs.append("ok <- PlannerReactService")
                ctx.setdefault("planner_trace", planner_res.model_dump(mode="json"))

            elif service == "CouncilService":
                council_req = DeliberationRequest(
                    prompt=_last_user_message(ctx),
                    history=ctx.get("messages") or [],
                    tags=ctx.get("packs") or [],
                    universe=ctx.get("mode") or "agent",
                    response_channel=reply_channel,
                )

                env = BaseEnvelope(
                    kind="council.request",
                    source=source,
                    correlation_id=correlation_id,
                    reply_to=reply_channel,
                    payload=council_req.model_dump(mode="json"),
                )

                logs.append(f"rpc -> CouncilService reply={reply_channel}")
                msg = await bus.rpc_request(
                    settings.channel_council_intake,
                    env,
                    reply_channel=reply_channel,
                    timeout_sec=step_timeout_sec,
                )
                decoded = bus.codec.decode(msg.get("data"))
                if not decoded.ok:
                    raise RuntimeError(f"CouncilService decode failed: {decoded.error}")

                payload = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
                merged_result[service] = payload
                logs.append("ok <- CouncilService")

            elif service == "VerbRequestService":
                candidate = None
                if isinstance(ctx.get("collapse_entry"), dict):
                    candidate = ctx.get("collapse_entry")
                elif isinstance(ctx.get("collapse_json"), str):
                    try:
                        candidate = json.loads(ctx.get("collapse_json"))
                    except Exception:
                        candidate = None

                if not candidate:
                    candidate = find_collapse_entry(ctx.get("prior_step_results"))

                if not isinstance(candidate, dict):
                    merged_result[service] = {"ok": False, "reason": "no_collapse_entry"}
                    logs.append("skip <- VerbRequestService (no entry)")
                elif str(candidate.get("type", "")).strip().lower() == "noop":
                    merged_result[service] = {"ok": True, "reason": "noop"}
                    logs.append("skip <- VerbRequestService (noop)")
                else:
                    try:
                        entry = CollapseMirrorEntryV2.model_validate(candidate)
                    except Exception as exc:
                        merged_result[service] = {"ok": False, "reason": "invalid_collapse_entry", "error": str(exc)}
                        logs.append("skip <- VerbRequestService (invalid entry)")
                    else:
                        event_id = str(uuid4())
                        trace_meta = _trace_meta_from_ctx(
                            ctx,
                            event_id=event_id,
                            parent_event_id=ctx.get("parent_event_id"),
                            source=source,
                        )
                        envelope = BaseEnvelope(
                            kind="collapse.mirror.intake",
                            source=source,
                            correlation_id=correlation_id,
                            trace=trace_meta,
                            payload=entry.model_dump(mode="json"),
                        )
                        await bus.publish(settings.channel_collapse_intake, envelope)
                        logger.info(
                            f"VerbRequestService published channel={settings.channel_collapse_intake} "
                            f"trace_id={trace_meta.get('trace_id')} event_id={event_id}"
                        )
                        merged_result[service] = {
                            "ok": True,
                            "published": True,
                            "channel": settings.channel_collapse_intake,
                            "event_id": entry.event_id,
                        }
                        logs.append("publish -> orion:collapse:intake")

            elif service == "MetaTagsService":
                tick = ctx.get("metacognition_tick") or {}
                if not isinstance(tick, dict):
                    tick = {"raw": str(tick)}

                text = ""
                for k in ("summary", "mantra", "trigger"):
                    v = tick.get(k)
                    if isinstance(v, str) and v.strip():
                        text += v.strip() + "\n"
                if not text.strip():
                    text = str(tick)

                req_id = str(tick.get("tick_id") or tick.get("event_id") or correlation_id)

                req_payload = {
                    "id": req_id,
                    "text": text,
                    "kind": "metacognition.tick.v1",
                    "raw": tick,
                }

                env = BaseEnvelope(
                    kind="meta_tags.request.v1",
                    source=source,
                    correlation_id=correlation_id,
                    reply_to=reply_channel,
                    payload=req_payload,
                )

                logs.append(f"rpc -> MetaTagsService (reply={reply_channel})")
                msg = await bus.rpc_request(
                    "orion:exec:request:MetaTagsService",
                    env,
                    reply_channel=reply_channel,
                    timeout_sec=effective_timeout,
                )

                decoded = bus.codec.decode(msg.get("data"))
                if not decoded.ok or not decoded.envelope:
                    raise RuntimeError(f"MetaTagsService decode failed: {decoded.error}")

                merged_result["MetaTagsService"] = (
                    decoded.envelope.payload
                    if isinstance(decoded.envelope.payload, dict)
                    else {"raw": str(decoded.envelope.payload)}
                )
                logs.append("ok <- MetaTagsService")
                continue

            else:
                logs.append(f"skip <- {service} (generic path not implemented in example)")

        except Exception as e:
            logs.append(f"exception <- {service}: {e}")
            logger.error(f"Service {service} failed: {e}")
            return StepExecutionResult(
                status="fail",
                verb_name=step.verb_name,
                step_name=step.step_name,
                order=step.order,
                result=merged_result,
                latency_ms=int((time.time() - t0) * 1000),
                node=settings.node_name,
                logs=logs,
                error=f"{service}: {e}",
            )

    return StepExecutionResult(
        status="success",
        verb_name=step.verb_name,
        step_name=step.step_name,
        order=step.order,
        result=merged_result,
        spark_vector=spark_vector,
        latency_ms=int((time.time() - t0) * 1000),
        node=settings.node_name,
        logs=logs,
    )
