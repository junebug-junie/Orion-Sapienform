from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo
from uuid import uuid4

import yaml
import orion
from pydantic import BaseModel, Field

from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, ChatResultPayload, LLMMessage, ServiceRef
from orion.core.contracts.recall import RecallQueryV1, RecallReplyV1
from orion.core.verbs.base import BaseVerb, VerbContext
from orion.core.verbs.models import VerbEffectV1
from orion.core.verbs.registry import verb
from orion.schemas.collapse_mirror import CollapseMirrorEntryV2
from orion.schemas.cortex.schemas import PlanExecutionRequest, PlanExecutionResult
from orion.schemas.pad.v1 import PadRpcRequestV1, PadRpcResponseV1
from orion.schemas.notify import NotificationRequest
from orion.schemas.self_study import (
    SelfStudyConsumerContextV1,
    SelfStudyRetrieveRequestV1,
)
from orion.notify.client import NotifyClient

from .router import PlanRouter
from .self_study import run_self_concept_induce, run_self_concept_reflect, run_self_repo_inspect, run_self_retrieve
from .self_study_policy import (
    build_self_study_consumer_context,
    build_self_study_consumer_request,
    render_self_study_consumer_context,
    resolve_self_study_consumer_policy,
)
from .settings import settings

logger = logging.getLogger("orion.cortex.exec.verb_adapters")
VERBS_DIR = Path(orion.__file__).resolve().parent / "cognition" / "verbs"


def _self_study_config_from_payload(payload: PlanExecutionRequest) -> Dict[str, Any]:
    metadata = _metadata_from_payload(payload)
    if isinstance(metadata.get("self_study"), dict):
        return dict(metadata.get("self_study") or {})
    extra = payload.args.extra or {}
    options = extra.get("options") if isinstance(extra, dict) else {}
    if isinstance(options, dict) and isinstance(options.get("self_study"), dict):
        return dict(options.get("self_study") or {})
    return {}


async def _resolve_self_study_context(
    *,
    consumer_name: str,
    output_mode: str | None,
    payload: PlanExecutionRequest,
    correlation_id: str,
    source: ServiceRef | None,
) -> SelfStudyConsumerContextV1:
    config = _self_study_config_from_payload(payload)
    decision = resolve_self_study_consumer_policy(
        consumer_name=consumer_name,
        output_mode=output_mode,
        config=config,
    )
    notes: List[str] = []
    result = None

    if decision.enabled:
        try:
            request = build_self_study_consumer_request(decision, config)
            result = await run_self_retrieve(
                request=request,
                bus=None,
                source=source,
                correlation_id=correlation_id,
            )
            notes.append(f"self_study_consulted mode={request.retrieval_mode}")
        except Exception as exc:
            notes.append(f"self_study_unavailable:{exc}")
            logger.warning(
                "self_study_consumer_unavailable consumer=%s corr=%s error=%s",
                consumer_name,
                correlation_id,
                exc,
            )
    else:
        notes.append(f"self_study_disabled:{decision.policy_reason}")

    return build_self_study_consumer_context(decision, result=result, notes=notes)


def _self_study_payload(context: SelfStudyConsumerContextV1) -> Dict[str, Any]:
    rendered = render_self_study_consumer_context(context)
    return {
        "consulted": context.consulted,
        "used": context.used,
        "consumer_name": context.consumer_name,
        "consumer_kind": context.consumer_kind,
        "retrieval_mode": context.retrieval_mode,
        "policy_reason": context.policy_reason,
        "policy_decision": context.policy_decision.model_dump(mode="json"),
        "notes": list(context.notes),
        "rendered": rendered,
        "result": context.result.model_dump(mode="json") if context.result is not None else None,
    }


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
        plan_metadata = payload.plan.metadata if isinstance(payload.plan.metadata, dict) else {}
        ctx_payload = {
            **payload_context,
            **(payload.args.extra or {}),
            "user_id": payload.args.user_id,
            "trigger_source": payload.args.trigger_source,
            "plan_metadata": plan_metadata,
        }
        if "personality_file" in plan_metadata:
            # Preserve personality declaration from verb profile metadata all the way into execution ctx.
            ctx_payload["personality_file"] = plan_metadata.get("personality_file")
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

        self_study_context = await _resolve_self_study_context(
            consumer_name="legacy.plan",
            output_mode=str(payload_context.get("output_mode") or extra.get("output_mode") or ""),
            payload=payload,
            correlation_id=correlation_id,
            source=source,
        )
        self_study_payload = _self_study_payload(self_study_context)
        ctx_payload["self_study"] = self_study_payload
        ctx_payload["self_study_rendered"] = self_study_payload["rendered"]
        if self_study_context.used:
            ctx_payload.setdefault("messages", []).append(
                {
                    "role": "system",
                    "content": self_study_payload["rendered"],
                }
            )

        router = PlanRouter()
        result = await router.run_plan(
            bus,
            source=source,
            req=payload,
            correlation_id=correlation_id,
            ctx=ctx_payload,
        )
        result.recall_debug["self_study"] = self_study_payload
        return LegacyPlanOutput(result=result), []


class JuniperCollapseActionOutput(BaseModel):
    ok: bool = True
    status: str = "success"
    final_text: str | None = None
    message_preview: str | None = None
    notification_id: str | None = None
    memory_used: bool = False
    recall_debug: Dict[str, Any] = Field(default_factory=dict)
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timings: Dict[str, int] = Field(default_factory=dict)
    error: Dict[str, Any] | None = None


def _actions_source(source: ServiceRef | None) -> ServiceRef:
    if isinstance(source, ServiceRef):
        return source
    return ServiceRef(name=settings.service_name, version=settings.service_version, node=settings.node_name)


def _collapse_to_fragment(entry: CollapseMirrorEntryV2) -> str:
    parts = [f"Trigger: {entry.trigger}", f"Summary: {entry.summary}"]
    if entry.what_changed_summary:
        parts.append(f"What changed: {entry.what_changed_summary}")
    if entry.observer_state:
        parts.append("Observer state: " + "; ".join(entry.observer_state))
    if entry.emergent_entity:
        parts.append(f"Emergent entity: {entry.emergent_entity}")
    if entry.mantra:
        parts.append(f"Mantra: {entry.mantra}")
    return "\n".join([p for p in parts if p])


def _collapse_to_markdown(entry: CollapseMirrorEntryV2) -> str:
    lines = [
        "### Collapse Mirror",
        f"- **observer**: {entry.observer}",
        f"- **type**: {entry.type}",
        f"- **emergent_entity**: {entry.emergent_entity}",
    ]
    if entry.tags:
        lines.append(f"- **tags**: {', '.join(entry.tags)}")
    lines.extend(["", f"**Trigger:** {entry.trigger}", "", f"**Summary:** {entry.summary}"])
    if entry.what_changed_summary:
        lines.extend(["", f"**What changed:** {entry.what_changed_summary}"])
    if entry.observer_state:
        lines.extend(["", "**Observer state:**", *[f"- {item}" for item in entry.observer_state]])
    lines.extend(["", f"**Mantra:** {entry.mantra}"])
    return "\n".join(lines).strip() + "\n"


def _system_prompt() -> str:
    return (
        "You are Orion. A Collapse Mirror entry was authored by Juniper. "
        "Do two things and use the exact delimiters below.\n\n"
        "[INTROSPECT]\n"
        "Write a brief introspect+synthesize view (private, not addressed to Juniper).\n"
        "[/INTROSPECT]\n\n"
        "[MESSAGE]\n"
        "Write a supportive, specific message addressed to Juniper. "
        "Be concise, grounded in the mirror and relevant memory.\n"
        "[/MESSAGE]\n"
    )


_SECTION_RE = re.compile(r"\[(INTROSPECT|MESSAGE)\]\s*(.*?)\s*\[/\1\]", flags=re.DOTALL | re.IGNORECASE)


def _extract_sections(text: str) -> tuple[str, str]:
    introspect = ""
    message = ""
    if not text:
        return introspect, message
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return "", text.strip()
    for match in matches:
        label = (match.group(1) or "").strip().lower()
        content = (match.group(2) or "").strip()
        if label == "introspect":
            introspect = content
        elif label == "message":
            message = content
    return introspect, message or text.strip()


def _preview_text(message: str, max_len: int = 280) -> str:
    msg = (message or "").strip()
    return msg if len(msg) <= max_len else msg[: max_len - 1].rstrip() + "…"


def _metadata_from_payload(payload: PlanExecutionRequest) -> Dict[str, Any]:
    metadata = payload.context.get("metadata") if isinstance(payload.context, dict) else {}
    return metadata if isinstance(metadata, dict) else {}


def _decode_recall(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    try:
        reply = RecallReplyV1.model_validate(payload)
        return reply.bundle.rendered, reply.model_dump(mode="json")
    except Exception:
        bundle = payload.get("bundle") or {}
        return str(bundle.get("rendered") or ""), payload


def _decode_llm(payload: dict[str, Any]) -> str:
    try:
        return ChatResultPayload.model_validate(payload).text
    except Exception:
        return str(payload.get("content") or payload.get("text") or "")


def _build_notify_request(*, entry: CollapseMirrorEntryV2, metadata: Dict[str, Any], correlation_id: str, introspect_text: str, message_text: str) -> NotificationRequest:
    session_id = str(metadata.get("session_id") or "collapse_mirror")
    recipient_group = str(metadata.get("recipient_group") or "juniper_primary")
    dedupe_key = str(metadata.get("notify_dedupe_key") or f"actions:collapse_reply:{entry.event_id}")
    dedupe_window_seconds = int(metadata.get("notify_dedupe_window_seconds") or 86400)
    body_md = "## Orion — Collapse Mirror\n\n" + message_text.strip() + "\n"
    if introspect_text.strip():
        body_md += "\n---\n\n<details><summary>Introspect</summary>\n\n" + introspect_text.strip() + "\n\n</details>\n"
    return NotificationRequest(
        source_service=settings.service_name,
        event_kind="orion.chat.message",
        severity="info",
        title="Orion — Collapse Mirror",
        body_text=message_text.strip(),
        body_md=body_md,
        recipient_group=recipient_group,
        session_id=session_id,
        correlation_id=correlation_id,
        dedupe_key=dedupe_key,
        dedupe_window_seconds=dedupe_window_seconds,
        tags=["chat", "message", "actions", "collapse"],
        context={
            "action_name": metadata.get("action_name") or "respond_to_juniper_collapse_mirror.v1",
            "collapse_event_id": entry.event_id,
            "collapse_id": entry.id,
            "collapse_type": entry.type,
            "collapse_tags": list(entry.tags or []),
            "collapse_emergent_entity": entry.emergent_entity,
            "preview_text": _preview_text(message_text),
        },
    )


@verb("actions.respond_to_juniper_collapse_mirror.v1")
class RespondToJuniperCollapseMirrorVerb(BaseVerb[PlanExecutionRequest, JuniperCollapseActionOutput]):
    input_model = PlanExecutionRequest
    output_model = JuniperCollapseActionOutput

    async def execute(self, ctx: VerbContext, payload: PlanExecutionRequest) -> Tuple[JuniperCollapseActionOutput, List[VerbEffectV1]]:
        bus = ctx.meta.get("bus")
        source = _actions_source(ctx.meta.get("source"))
        correlation_id = str(ctx.meta.get("correlation_id") or payload.args.request_id or uuid4())
        if bus is None:
            return JuniperCollapseActionOutput(ok=False, status="fail", error={"message": "missing_bus"}), []

        metadata = _metadata_from_payload(payload)
        raw_entry = metadata.get("collapse_entry")
        if not isinstance(raw_entry, dict):
            return JuniperCollapseActionOutput(ok=False, status="fail", error={"message": "missing_collapse_entry"}), []
        entry = CollapseMirrorEntryV2.model_validate(raw_entry)
        output_mode = str(payload.context.get("output_mode") or metadata.get("output_mode") or "reflective_depth")
        self_study_context = await _resolve_self_study_context(
            consumer_name="actions.respond_to_juniper_collapse_mirror.v1",
            output_mode=output_mode,
            payload=payload,
            correlation_id=correlation_id,
            source=source,
        )
        self_study_payload = _self_study_payload(self_study_context)

        logger.info("running verb actions.respond_to_juniper_collapse_mirror.v1 corr=%s event_id=%s", correlation_id, entry.event_id)

        recall_reply = f"orion:exec:result:RecallService:{uuid4()}"
        recall_env = BaseEnvelope(
            kind="recall.query.v1",
            source=source,
            correlation_id=correlation_id,
            reply_to=recall_reply,
            payload=RecallQueryV1(
                fragment=str(metadata.get("recall_fragment") or _collapse_to_fragment(entry)),
                profile=str(metadata.get("recall_profile") or "reflect.v1"),
                session_id=str(metadata.get("session_id") or "collapse_mirror"),
                node_id=settings.node_name,
                verb="collapse_mirror",
                intent="respond_to_juniper",
            ).model_dump(mode="json"),
        )
        recall_msg = await bus.rpc_request(settings.channel_recall_intake, recall_env, reply_channel=recall_reply, timeout_sec=60.0)
        recall_decoded = bus.codec.decode(recall_msg.get("data"))
        if not recall_decoded.ok or recall_decoded.envelope is None:
            return JuniperCollapseActionOutput(ok=False, status="fail", error={"message": f"recall_decode_failed:{recall_decoded.error}"}), []
        memory_rendered, recall_debug = _decode_recall(recall_decoded.envelope.payload if isinstance(recall_decoded.envelope.payload, dict) else {})

        llm_reply = f"orion:exec:result:LLMGatewayService:{uuid4()}"
        llm_env = BaseEnvelope(
            kind="llm.chat.request",
            source=source,
            correlation_id=correlation_id,
            reply_to=llm_reply,
            payload=ChatRequestPayload(
                messages=[
                    LLMMessage(role="system", content=_system_prompt()),
                    LLMMessage(
                        role="user",
                        content=(
                            _collapse_to_markdown(entry)
                            + "\nRELEVANT MEMORY\n"
                            + (memory_rendered or "").strip()
                            + "\n\n"
                            + self_study_payload["rendered"]
                            + "\n"
                        ),
                    ),
                ],
                raw_user_text=entry.summary,
                route="chat",
                options={"max_tokens": 512, "temperature": 0.3},
                session_id=str(metadata.get("session_id") or "collapse_mirror"),
                user_id=None,
            ).model_dump(mode="json"),
        )
        llm_msg = await bus.rpc_request(settings.channel_llm_intake, llm_env, reply_channel=llm_reply, timeout_sec=200.0)
        llm_decoded = bus.codec.decode(llm_msg.get("data"))
        if not llm_decoded.ok or llm_decoded.envelope is None:
            return JuniperCollapseActionOutput(ok=False, status="fail", error={"message": f"llm_decode_failed:{llm_decoded.error}"}), []
        llm_text = _decode_llm(llm_decoded.envelope.payload if isinstance(llm_decoded.envelope.payload, dict) else {})
        introspect_text, message_text = _extract_sections(llm_text)

        notify_request = _build_notify_request(
            entry=entry,
            metadata=metadata,
            correlation_id=correlation_id,
            introspect_text=introspect_text,
            message_text=message_text,
        )
        accepted = await asyncio.to_thread(
            NotifyClient(base_url=settings.notify_url, api_token=settings.notify_api_token, timeout=10).send,
            notify_request,
        )

        return JuniperCollapseActionOutput(
            ok=bool(accepted.ok),
            status="success" if accepted.ok else "fail",
            final_text=message_text.strip(),
            message_preview=_preview_text(message_text),
            notification_id=str(accepted.notification_id) if accepted.notification_id else None,
            memory_used=bool(memory_rendered.strip()),
            recall_debug={**recall_debug, "self_study": self_study_payload},
            metadata={"notify_status": accepted.status, "self_study": self_study_payload},
            timings={},
            error=None if accepted.ok else {"message": accepted.detail or "notify_failed"},
        ), []


class SkillVerbOutput(BaseModel):
    ok: bool = True
    status: str = "success"
    final_text: str | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Dict[str, Any] | None = None


class NotifyChatMessageOutput(SkillVerbOutput):
    notification_id: str | None = None


class SafeCommandRunner:
    def __init__(self, *, allowed_commands: set[str], timeout_sec: float) -> None:
        self.allowed_commands = set(allowed_commands)
        self.timeout_sec = float(timeout_sec)

    def run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        if not command:
            raise PermissionError("empty_command")
        binary = str(command[0]).strip()
        if binary not in self.allowed_commands:
            raise PermissionError(f"command_not_allowlisted:{binary}")
        resolved = shutil.which(binary)
        if not resolved:
            raise FileNotFoundError(binary)
        return subprocess.run(
            [resolved, *command[1:]],
            capture_output=True,
            text=True,
            timeout=self.timeout_sec,
            check=False,
        )


def _skill_args(payload: PlanExecutionRequest) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    metadata = _metadata_from_payload(payload)
    if isinstance(metadata.get("skill_args"), dict):
        out.update(metadata.get("skill_args") or {})
    extra = payload.args.extra or {}
    if isinstance(extra.get("skill_args"), dict):
        out.update(extra.get("skill_args") or {})
    return out


def _skill_result_output(*, skill_name: str, result: Dict[str, Any], ok: bool = True, status: str = "success", error: Dict[str, Any] | None = None) -> SkillVerbOutput:
    return SkillVerbOutput(
        ok=ok,
        status=status,
        final_text=json.dumps(result, sort_keys=True),
        metadata={"skill_name": skill_name, "skill_result": result},
        error=error,
    )


def _http_json_get(url: str, *, timeout_sec: float) -> Dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout_sec) as response:  # noqa: S310
        data = response.read().decode("utf-8")
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("http_json_not_object")
    return payload


def _parse_nvidia_smi_csv(text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 9:
            continue
        memory_used = float(parts[5])
        memory_total = float(parts[6]) if float(parts[6]) else 0.0
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "uuid": parts[2],
                "temperature_c": float(parts[3]),
                "utilization_gpu_pct": float(parts[4]),
                "memory_used_mb": memory_used,
                "memory_total_mb": memory_total,
                "memory_used_ratio": (memory_used / memory_total) if memory_total else 0.0,
                "power_draw_w": float(parts[7]) if parts[7] not in {"[N/A]", "N/A", ""} else None,
                "pstate": parts[8],
            }
        )
    return rows


def _map_docker_engine_containers(containers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mapped: List[Dict[str, Any]] = []
    for container in containers:
        ports = container.get("Ports") if isinstance(container.get("Ports"), list) else []
        mapped.append(
            {
                "id": container.get("Id"),
                "name": ((container.get("Names") or [None])[0] or "").lstrip("/"),
                "image": container.get("Image"),
                "state": container.get("State"),
                "status": container.get("Status"),
                "command": container.get("Command"),
                "ports": [
                    {
                        "private_port": port.get("PrivatePort"),
                        "public_port": port.get("PublicPort"),
                        "type": port.get("Type"),
                    }
                    for port in ports
                ],
            }
        )
    return mapped


def _parse_docker_ps_lines(text: str) -> List[Dict[str, Any]]:
    containers: List[Dict[str, Any]] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        containers.append(
            {
                "id": payload.get("ID") or payload.get("ID".lower()) or payload.get("Id"),
                "name": payload.get("Names") or payload.get("Name"),
                "image": payload.get("Image"),
                "state": payload.get("State"),
                "status": payload.get("Status"),
                "command": payload.get("Command"),
                "ports": payload.get("Ports"),
            }
        )
    return containers


def _read_docker_engine_containers(sock_path: str) -> List[Dict[str, Any]]:
    request = b"GET /containers/json?all=1 HTTP/1.1\r\nHost: docker\r\nConnection: close\r\n\r\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(3.0)
        client.connect(sock_path)
        client.sendall(request)
        chunks: list[bytes] = []
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    response = b"".join(chunks)
    header_blob, _, body = response.partition(b"\r\n\r\n")
    status_line = header_blob.splitlines()[0].decode("utf-8", errors="ignore") if header_blob else ""
    if " 200 " not in f" {status_line} ":
        raise RuntimeError(f"docker_engine_http_error:{status_line}")
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("docker_engine_payload_invalid")
    return payload


def _normalize_biometrics_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    nodes = snapshot.get("nodes") if isinstance(snapshot.get("nodes"), dict) else {}
    cluster = snapshot.get("cluster") if isinstance(snapshot.get("cluster"), dict) else {}
    composite = cluster.get("composite") if isinstance(cluster.get("composite"), dict) else {}
    trend = cluster.get("trend") if isinstance(cluster.get("trend"), dict) else {}
    return {
        "status": snapshot.get("status") or "NO_SIGNAL",
        "reason": snapshot.get("reason"),
        "as_of": snapshot.get("as_of"),
        "freshness_s": snapshot.get("freshness_s"),
        "constraint": snapshot.get("constraint") or "NONE",
        "cluster": {"composite": composite, "trend": trend},
        "nodes": nodes,
    }


def _threshold_findings(*, biometrics_snapshot: Dict[str, Any] | None = None, gpu_snapshot: Dict[str, Any] | None = None, gpu_mem_threshold: float | None = None, biometrics_stability_threshold: float | None = None) -> List[str]:
    findings: List[str] = []
    if biometrics_snapshot and biometrics_stability_threshold is not None:
        cluster = biometrics_snapshot.get("cluster") if isinstance(biometrics_snapshot.get("cluster"), dict) else {}
        composite = cluster.get("composite") if isinstance(cluster.get("composite"), dict) else {}
        stability = composite.get("stability")
        try:
            if stability is not None and float(stability) < float(biometrics_stability_threshold):
                findings.append(f"biometrics_stability_below:{float(stability):.3f}")
        except Exception:
            pass
    if gpu_snapshot and gpu_mem_threshold is not None:
        gpus = gpu_snapshot.get("gpus") if isinstance(gpu_snapshot.get("gpus"), list) else []
        for gpu in gpus:
            try:
                ratio = float(gpu.get("memory_used_ratio") or 0.0)
            except Exception:
                ratio = 0.0
            if ratio > float(gpu_mem_threshold):
                findings.append(f"gpu_mem_above:{gpu.get('name') or gpu.get('index')}:{ratio:.3f}")
    return findings


def _notify_payload_from_findings(findings: List[str], *, metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "recipient_group": str(metadata.get("recipient_group") or "operators"),
        "session_id": str(metadata.get("session_id") or "skills_scheduler"),
        "title": "Orion Skills Threshold Alert",
        "body_text": "Threshold findings: " + "; ".join(findings),
        "dedupe_key": str(metadata.get("dedupe_key") or ("actions:skills:threshold:" + (findings[0] if findings else "none"))),
        "tags": ["actions", "skills", "threshold"],
    }


def _pad_rpc_request(method: str, *, correlation_id: str, reply_channel: str, source: ServiceRef, args: Optional[Dict[str, Any]] = None) -> BaseEnvelope:
    req = PadRpcRequestV1(
        request_id=correlation_id,
        reply_channel=reply_channel,
        method=method,
        args=args or {},
    )
    return BaseEnvelope(
        kind="PadRpcRequestV1",
        source=source,
        correlation_id=correlation_id,
        reply_to=reply_channel,
        payload=req.model_dump(mode="json"),
    )


@verb("skills.system.time_now.v1")
class TimeNowVerb(BaseVerb[PlanExecutionRequest, SkillVerbOutput]):
    input_model = PlanExecutionRequest
    output_model = SkillVerbOutput

    async def execute(self, ctx: VerbContext, payload: PlanExecutionRequest) -> Tuple[SkillVerbOutput, List[VerbEffectV1]]:
        skill_args = _skill_args(payload)
        tz_name = str(skill_args.get("timezone") or os.getenv("ORION_TZ") or settings.orion_tz or "America/Denver")
        now_utc = datetime.now(timezone.utc)
        now_local = now_utc.astimezone(ZoneInfo(tz_name))
        result = {
            "timezone": tz_name,
            "local_iso": now_local.isoformat(),
            "utc_iso": now_utc.isoformat(),
            "local_date": now_local.date().isoformat(),
            "local_time": now_local.time().replace(microsecond=0).isoformat(),
        }
        return _skill_result_output(skill_name="skills.system.time_now.v1", result=result), []


@verb("skills.gpu.nvidia_smi_snapshot.v1")
class NvidiaSmiSnapshotVerb(BaseVerb[PlanExecutionRequest, SkillVerbOutput]):
    input_model = PlanExecutionRequest
    output_model = SkillVerbOutput

    async def execute(self, ctx: VerbContext, payload: PlanExecutionRequest) -> Tuple[SkillVerbOutput, List[VerbEffectV1]]:
        runner = SafeCommandRunner(allowed_commands={"nvidia-smi"}, timeout_sec=settings.skills_command_timeout_sec)
        command = [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,pstate",
            "--format=csv,noheader,nounits",
        ]
        try:
            proc = await asyncio.to_thread(runner.run, command)
        except FileNotFoundError:
            result = {"available": False, "reason": "nvidia_smi_not_installed", "gpus": []}
            return _skill_result_output(skill_name="skills.gpu.nvidia_smi_snapshot.v1", result=result, ok=False, status="unavailable", error={"message": result["reason"]}), []
        except PermissionError as exc:
            result = {"available": False, "reason": str(exc), "gpus": []}
            return _skill_result_output(skill_name="skills.gpu.nvidia_smi_snapshot.v1", result=result, ok=False, status="blocked", error={"message": str(exc)}), []
        if proc.returncode != 0:
            result = {"available": False, "reason": (proc.stderr or proc.stdout or "nvidia_smi_failed").strip(), "gpus": []}
            return _skill_result_output(skill_name="skills.gpu.nvidia_smi_snapshot.v1", result=result, ok=False, status="unavailable", error={"message": result["reason"]}), []
        gpus = _parse_nvidia_smi_csv(proc.stdout)
        result = {"available": True, "gpu_count": len(gpus), "gpus": gpus}
        return _skill_result_output(skill_name="skills.gpu.nvidia_smi_snapshot.v1", result=result), []


@verb("skills.docker.ps_status.v1")
class DockerPsStatusVerb(BaseVerb[PlanExecutionRequest, SkillVerbOutput]):
    input_model = PlanExecutionRequest
    output_model = SkillVerbOutput

    async def execute(self, ctx: VerbContext, payload: PlanExecutionRequest) -> Tuple[SkillVerbOutput, List[VerbEffectV1]]:
        backend = None
        containers: List[Dict[str, Any]] = []
        sock_path = settings.docker_sock_path
        try:
            if sock_path and os.path.exists(sock_path):
                backend = "engine_api"
                raw = await asyncio.to_thread(_read_docker_engine_containers, sock_path)
                containers = _map_docker_engine_containers(raw)
        except Exception:
            backend = None
            containers = []
        if backend is None:
            runner = SafeCommandRunner(allowed_commands={"docker"}, timeout_sec=settings.skills_command_timeout_sec)
            try:
                proc = await asyncio.to_thread(runner.run, ["docker", "ps", "--format", "{{json .}}", "-a"])
            except (PermissionError, FileNotFoundError) as exc:
                result = {"available": False, "backend": None, "reason": str(exc), "containers": []}
                return _skill_result_output(skill_name="skills.docker.ps_status.v1", result=result, ok=False, status="unavailable", error={"message": str(exc)}), []
            if proc.returncode != 0:
                result = {"available": False, "backend": None, "reason": (proc.stderr or proc.stdout or "docker_ps_failed").strip(), "containers": []}
                return _skill_result_output(skill_name="skills.docker.ps_status.v1", result=result, ok=False, status="unavailable", error={"message": result["reason"]}), []
            backend = "docker_ps"
            containers = _parse_docker_ps_lines(proc.stdout)
        result = {"available": True, "backend": backend, "container_count": len(containers), "containers": containers}
        return _skill_result_output(skill_name="skills.docker.ps_status.v1", result=result), []


@verb("skills.biometrics.snapshot.v1")
class BiometricsSnapshotVerb(BaseVerb[PlanExecutionRequest, SkillVerbOutput]):
    input_model = PlanExecutionRequest
    output_model = SkillVerbOutput

    async def execute(self, ctx: VerbContext, payload: PlanExecutionRequest) -> Tuple[SkillVerbOutput, List[VerbEffectV1]]:
        base_url = str(settings.biometrics_service_url).rstrip("/")
        try:
            raw = await asyncio.to_thread(_http_json_get, f"{base_url}/snapshot", timeout_sec=float(settings.biometrics_http_timeout_sec))
        except Exception as exc:
            result = {"available": False, "reason": str(exc)}
            return _skill_result_output(skill_name="skills.biometrics.snapshot.v1", result=result, ok=False, status="unavailable", error={"message": str(exc)}), []
        result = _normalize_biometrics_snapshot(raw)
        return _skill_result_output(skill_name="skills.biometrics.snapshot.v1", result=result), []


@verb("skills.biometrics.raw_recent.v1")
class BiometricsRawRecentVerb(BaseVerb[PlanExecutionRequest, SkillVerbOutput]):
    input_model = PlanExecutionRequest
    output_model = SkillVerbOutput

    async def execute(self, ctx: VerbContext, payload: PlanExecutionRequest) -> Tuple[SkillVerbOutput, List[VerbEffectV1]]:
        skill_args = _skill_args(payload)
        query = urlencode({k: v for k, v in {"limit": skill_args.get("limit", 10), "node": skill_args.get("node")}.items() if v is not None})
        base_url = str(settings.biometrics_service_url).rstrip("/")
        url = f"{base_url}/raw/recent" + (f"?{query}" if query else "")
        try:
            raw = await asyncio.to_thread(_http_json_get, url, timeout_sec=float(settings.biometrics_http_timeout_sec))
        except Exception as exc:
            result = {"available": False, "reason": str(exc), "items": []}
            return _skill_result_output(skill_name="skills.biometrics.raw_recent.v1", result=result, ok=False, status="unavailable", error={"message": str(exc)}), []
        return _skill_result_output(skill_name="skills.biometrics.raw_recent.v1", result=raw), []


@verb("skills.landing_pad.metrics_snapshot.v1")
class LandingPadMetricsSnapshotVerb(BaseVerb[PlanExecutionRequest, SkillVerbOutput]):
    input_model = PlanExecutionRequest
    output_model = SkillVerbOutput

    async def execute(self, ctx: VerbContext, payload: PlanExecutionRequest) -> Tuple[SkillVerbOutput, List[VerbEffectV1]]:
        bus = ctx.meta.get("bus")
        source = _actions_source(ctx.meta.get("source"))
        correlation_id = str(ctx.meta.get("correlation_id") or payload.args.request_id or str(uuid4()))
        if bus is None:
            return _skill_result_output(skill_name="skills.landing_pad.metrics_snapshot.v1", result={"available": False, "reason": "missing_bus"}, ok=False, status="fail", error={"message": "missing_bus"}), []
        reply_channel = f"{settings.channel_pad_rpc_reply_prefix}:{uuid4()}"
        env = _pad_rpc_request("get_stats", correlation_id=correlation_id, reply_channel=reply_channel, source=source, args={})
        msg = await bus.rpc_request(settings.channel_pad_rpc_request, env, reply_channel=reply_channel, timeout_sec=20.0)
        decoded = bus.codec.decode(msg.get("data"))
        if not decoded.ok or decoded.envelope is None:
            return _skill_result_output(skill_name="skills.landing_pad.metrics_snapshot.v1", result={"available": False, "reason": decoded.error}, ok=False, status="fail", error={"message": str(decoded.error)}), []
        response = PadRpcResponseV1.model_validate(decoded.envelope.payload)
        if not response.ok:
            return _skill_result_output(skill_name="skills.landing_pad.metrics_snapshot.v1", result={"available": False, "reason": response.error}, ok=False, status="fail", error={"message": str(response.error)}), []
        result = response.result or {}
        return _skill_result_output(skill_name="skills.landing_pad.metrics_snapshot.v1", result=result), []


@verb("skills.landing_pad.last_events.v1")
class LandingPadLastEventsVerb(BaseVerb[PlanExecutionRequest, SkillVerbOutput]):
    input_model = PlanExecutionRequest
    output_model = SkillVerbOutput

    async def execute(self, ctx: VerbContext, payload: PlanExecutionRequest) -> Tuple[SkillVerbOutput, List[VerbEffectV1]]:
        bus = ctx.meta.get("bus")
        source = _actions_source(ctx.meta.get("source"))
        correlation_id = str(ctx.meta.get("correlation_id") or payload.args.request_id or str(uuid4()))
        skill_args = _skill_args(payload)
        if bus is None:
            return _skill_result_output(skill_name="skills.landing_pad.last_events.v1", result={"available": False, "reason": "missing_bus", "events": []}, ok=False, status="fail", error={"message": "missing_bus"}), []
        limit = int(skill_args.get("limit") or 10)
        reply_channel = f"{settings.channel_pad_rpc_reply_prefix}:{uuid4()}"
        env = _pad_rpc_request("get_salient_events", correlation_id=correlation_id, reply_channel=reply_channel, source=source, args={"limit": limit})
        msg = await bus.rpc_request(settings.channel_pad_rpc_request, env, reply_channel=reply_channel, timeout_sec=20.0)
        decoded = bus.codec.decode(msg.get("data"))
        if not decoded.ok or decoded.envelope is None:
            return _skill_result_output(skill_name="skills.landing_pad.last_events.v1", result={"available": False, "reason": decoded.error, "events": []}, ok=False, status="fail", error={"message": str(decoded.error)}), []
        response = PadRpcResponseV1.model_validate(decoded.envelope.payload)
        if not response.ok:
            return _skill_result_output(skill_name="skills.landing_pad.last_events.v1", result={"available": False, "reason": response.error, "events": []}, ok=False, status="fail", error={"message": str(response.error)}), []
        events = response.result.get("events") if isinstance(response.result, dict) else []
        if not isinstance(events, list):
            events = []
        min_salience = skill_args.get("min_salience")
        event_type = skill_args.get("type")
        source_service = skill_args.get("source_service")
        filtered = []
        for event in events:
            if not isinstance(event, dict):
                continue
            if min_salience is not None:
                try:
                    if float(event.get("salience") or 0.0) < float(min_salience):
                        continue
                except Exception:
                    continue
            if event_type and str(event.get("type") or "") != str(event_type):
                continue
            if source_service and str(event.get("source_service") or "") != str(source_service):
                continue
            filtered.append(event)
        result = {"available": True, "events": filtered, "count": len(filtered)}
        return _skill_result_output(skill_name="skills.landing_pad.last_events.v1", result=result), []


@verb("self_repo_inspect")
class SelfRepoInspectVerb(BaseVerb[PlanExecutionRequest, SkillVerbOutput]):
    input_model = PlanExecutionRequest
    output_model = SkillVerbOutput

    async def execute(self, ctx: VerbContext, payload: PlanExecutionRequest) -> Tuple[SkillVerbOutput, List[VerbEffectV1]]:
        correlation_id = str(ctx.meta.get("correlation_id") or payload.args.request_id or str(uuid4()))
        result = await run_self_repo_inspect(
            bus=ctx.meta.get("bus"),
            source=_actions_source(ctx.meta.get("source")),
            correlation_id=correlation_id,
        )
        data = result.model_dump(mode="json")
        return _skill_result_output(skill_name="self_repo_inspect", result=data), []


@verb("self_concept_induce")
class SelfConceptInduceVerb(BaseVerb[PlanExecutionRequest, SkillVerbOutput]):
    input_model = PlanExecutionRequest
    output_model = SkillVerbOutput

    async def execute(self, ctx: VerbContext, payload: PlanExecutionRequest) -> Tuple[SkillVerbOutput, List[VerbEffectV1]]:
        correlation_id = str(ctx.meta.get("correlation_id") or payload.args.request_id or str(uuid4()))
        result = await run_self_concept_induce(
            bus=ctx.meta.get("bus"),
            source=_actions_source(ctx.meta.get("source")),
            correlation_id=correlation_id,
        )
        data = result.model_dump(mode="json")
        return _skill_result_output(skill_name="self_concept_induce", result=data), []


@verb("self_concept_reflect")
class SelfConceptReflectVerb(BaseVerb[PlanExecutionRequest, SkillVerbOutput]):
    input_model = PlanExecutionRequest
    output_model = SkillVerbOutput

    async def execute(self, ctx: VerbContext, payload: PlanExecutionRequest) -> Tuple[SkillVerbOutput, List[VerbEffectV1]]:
        correlation_id = str(ctx.meta.get("correlation_id") or payload.args.request_id or str(uuid4()))
        result = await run_self_concept_reflect(
            bus=ctx.meta.get("bus"),
            source=_actions_source(ctx.meta.get("source")),
            correlation_id=correlation_id,
        )
        data = result.model_dump(mode="json")
        return _skill_result_output(skill_name="self_concept_reflect", result=data), []


@verb("self_retrieve")
class SelfRetrieveVerb(BaseVerb[PlanExecutionRequest, SkillVerbOutput]):
    input_model = PlanExecutionRequest
    output_model = SkillVerbOutput

    async def execute(self, ctx: VerbContext, payload: PlanExecutionRequest) -> Tuple[SkillVerbOutput, List[VerbEffectV1]]:
        skill_args = _skill_args(payload)
        if not skill_args and isinstance(payload.args.extra, dict):
            skill_args = dict(payload.args.extra)
        request = SelfStudyRetrieveRequestV1.model_validate(skill_args)
        result = await run_self_retrieve(
            request=request,
            bus=ctx.meta.get("bus"),
            source=_actions_source(ctx.meta.get("source")),
            correlation_id=str(ctx.meta.get("correlation_id") or payload.args.request_id or str(uuid4())),
        )
        data = result.model_dump(mode="json")
        return _skill_result_output(skill_name="self_retrieve", result=data), []


@verb("skills.system.notify_chat_message.v1")
class NotifyChatMessageVerb(BaseVerb[PlanExecutionRequest, NotifyChatMessageOutput]):
    input_model = PlanExecutionRequest
    output_model = NotifyChatMessageOutput

    async def execute(self, ctx: VerbContext, payload: PlanExecutionRequest) -> Tuple[NotifyChatMessageOutput, List[VerbEffectV1]]:
        metadata = _metadata_from_payload(payload)
        skill_args = _skill_args(payload)
        notify_request = NotificationRequest(
            source_service=settings.service_name,
            event_kind="orion.chat.message",
            severity="warning",
            title=str(skill_args.get("title") or "Orion Alert"),
            body_text=str(skill_args.get("body_text") or ""),
            body_md=str(skill_args.get("body_md") or (skill_args.get("body_text") or "")),
            recipient_group=str(skill_args.get("recipient_group") or metadata.get("recipient_group") or "operators"),
            session_id=str(skill_args.get("session_id") or metadata.get("session_id") or "skills_scheduler"),
            correlation_id=str(ctx.meta.get("correlation_id") or payload.args.request_id or str(uuid4())),
            dedupe_key=str(skill_args.get("dedupe_key") or metadata.get("dedupe_key") or f"skills:notify:{uuid4()}"),
            dedupe_window_seconds=int(skill_args.get("dedupe_window_seconds") or 3600),
            tags=list(skill_args.get("tags") or ["skills", "alert"]),
            context={"skill_args": skill_args},
        )
        accepted = await asyncio.to_thread(
            NotifyClient(base_url=settings.notify_url, api_token=settings.notify_api_token, timeout=10).send,
            notify_request,
        )
        base = NotifyChatMessageOutput(
            ok=bool(accepted.ok),
            status="success" if accepted.ok else "fail",
            final_text=notify_request.body_text,
            metadata={"skill_name": "skills.system.notify_chat_message.v1", "notify_status": accepted.status},
            error=None if accepted.ok else {"message": accepted.detail or "notify_failed"},
            notification_id=str(accepted.notification_id) if accepted.notification_id else None,
        )
        return base, []
