#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_PROOF_DIR = REPO_ROOT / "docs" / "postflight" / "proof" / "live"

# Ensure repo root import resolution when launched as `python scripts/...`.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
script_dir = str(Path(__file__).resolve().parent)
if script_dir in sys.path:
    sys.path.remove(script_dir)

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.cortex.contracts import CortexChatRequest

DISCORD_PROMPT = "Please provide instructions on how to deploy you onto Discord."
SUPERVISOR_PROMPT = (
    "Give concise deployment instructions for Orion on Discord. "
    "Include a brief testing or verification step. "
    "Avoid managerial planning language; produce concrete steps."
)


POSITIVE_PATTERNS = [
    r"discord",
    r"(developer portal|application|app setup|bot setup|create .*bot)",
    r"(token|env var|environment variable|DISCORD_BOT_TOKEN)",
    r"(intent|permission|oauth|invite)",
    r"(deploy|hosting|host|process|systemd|docker)",
    r"(test|troubleshoot|debug|verify)",
]

NEGATIVE_PATTERNS = [
    r"gather requirements",
    r"create a guide",
    r"review and refine",
    r"test deployment, then refine",
    r"purely managerial",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _service_ref() -> ServiceRef:
    return ServiceRef(name="answer-depth-live-proof", version="0.0.1", node="local")


def _extract(obj: Any, *path: str) -> Any:
    cur = obj
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _quality_checks(
    final_text: str, *, min_positive_patterns: Optional[int] = None
) -> Dict[str, Any]:
    text = final_text or ""
    lower = text.lower()
    positive = {pat: bool(re.search(pat, lower, flags=re.IGNORECASE)) for pat in POSITIVE_PATTERNS}
    negative = {pat: bool(re.search(pat, lower, flags=re.IGNORECASE)) for pat in NEGATIVE_PATTERNS}
    positive_count = sum(1 for v in positive.values() if v)
    required = min_positive_patterns if min_positive_patterns is not None else len(POSITIVE_PATTERNS)
    positive_pass = positive_count >= required
    negative_pass = not any(negative.values())
    return {
        "positive_checks": positive,
        "negative_checks": negative,
        "positive_pass": positive_pass,
        "negative_pass": negative_pass,
        "overall_pass": positive_pass and negative_pass,
    }


@dataclass
class LiveScenario:
    name: str
    prompt: str
    mode: str = "agent"
    force_agent_chain: bool = True
    route_intent: str = "none"


class ProbeCollector:
    def __init__(self, bus: OrionBusAsync, *, patterns: Iterable[str], corr_id: str):
        self.bus = bus
        self.patterns = list(patterns)
        self.corr_id = str(corr_id)
        self.events: List[Dict[str, Any]] = []
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._ready = asyncio.Event()

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())
        await asyncio.wait_for(self._ready.wait(), timeout=3.0)

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=3.0)
            except asyncio.TimeoutError:
                self._task.cancel()

    async def _run(self) -> None:
        async with self.bus.subscribe(*self.patterns, patterns=True) as pubsub:
            self._ready.set()
            while not self._stop.is_set():
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.25)
                if not msg or msg.get("type") not in ("message", "pmessage"):
                    continue
                decoded = self.bus.codec.decode(msg.get("data"))
                if not decoded.ok:
                    continue
                env = decoded.envelope
                if str(env.correlation_id) != self.corr_id:
                    continue
                channel = msg.get("channel")
                if hasattr(channel, "decode"):
                    channel = channel.decode("utf-8")
                payload = env.payload if isinstance(env.payload, dict) else {}
                self.events.append(
                    {
                        "ts": _now_iso(),
                        "channel": channel,
                        "kind": env.kind,
                        "correlation_id": str(env.correlation_id),
                        "reply_to": env.reply_to,
                        "payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
                    }
                )


async def _run_gateway_chat(
    *,
    bus_url: str,
    scenario: LiveScenario,
    timeout_sec: float,
    extra_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    corr = str(uuid4())
    reply_channel = f"orion:cortex:gateway:result:{corr}"
    payload = CortexChatRequest(
        prompt=scenario.prompt,
        mode=scenario.mode,
        route_intent=scenario.route_intent,
        options={
            "force_agent_chain": scenario.force_agent_chain,
            **(extra_options or {}),
        },
    ).model_dump(mode="json")

    env = BaseEnvelope(
        kind="cortex.gateway.chat.request",
        source=_service_ref(),
        correlation_id=corr,
        reply_to=reply_channel,
        payload=payload,
    )

    bus = OrionBusAsync(url=bus_url)
    await bus.connect()
    probe = ProbeCollector(
        bus,
        patterns=[
            "orion:cortex:*",
            "orion:verb:*",
            "orion:exec:request:*",
            "orion:exec:result:*",
            "orion:llm:*",
            "orion:cognition:trace",
        ],
        corr_id=corr,
    )
    await probe.start()

    started = _now_iso()
    error: Optional[str] = None
    raw_payload: Optional[Dict[str, Any]] = None
    result_kind: Optional[str] = None
    try:
        msg = await bus.rpc_request(
            "orion:cortex:gateway:request",
            env,
            reply_channel=reply_channel,
            timeout_sec=timeout_sec,
        )
        dec = bus.codec.decode(msg.get("data"))
        if not dec.ok:
            error = f"decode_failed:{dec.error}"
        else:
            result_kind = dec.envelope.kind
            raw_payload = dec.envelope.payload if isinstance(dec.envelope.payload, dict) else {}
    except Exception as e:
        error = str(e)
    finally:
        await asyncio.sleep(0.4)
        await probe.stop()
        await bus.close()

    finished = _now_iso()
    return {
        "entrypoint": "gateway",
        "scenario": scenario.name,
        "started_at": started,
        "finished_at": finished,
        "correlation_id": corr,
        "request_channel": "orion:cortex:gateway:request",
        "reply_channel": reply_channel,
        "result_kind": result_kind,
        "request_payload": payload,
        "response_payload": raw_payload,
        "error": error,
        "probe_events": probe.events,
    }


async def _run_orch_chat(
    *,
    bus_url: str,
    scenario: LiveScenario,
    timeout_sec: float,
    extra_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from orion.schemas.cortex.contracts import (
        CortexClientContext,
        CortexClientRequest,
        LLMMessage,
        RecallDirective,
    )

    corr = str(uuid4())
    reply_channel = f"orion:cortex:result:{corr}"
    req = CortexClientRequest(
        mode=scenario.mode,
        verb="agent_runtime" if scenario.mode in {"agent", "council"} else "chat_general",
        packs=["executive_pack"],
        options={
            "force_agent_chain": scenario.force_agent_chain,
            "supervised": True,
            "diagnostic": True,
            **(extra_options or {}),
        },
        recall=RecallDirective(enabled=False),
        context=CortexClientContext(
            messages=[LLMMessage(role="user", content=scenario.prompt)],
            raw_user_text=scenario.prompt,
            user_message=scenario.prompt,
            session_id=f"live-proof-{scenario.name}",
            user_id="live-proof",
        ),
    )
    payload = req.model_dump(mode="json")
    env = BaseEnvelope(
        kind="cortex.orch.request",
        source=_service_ref(),
        correlation_id=corr,
        reply_to=reply_channel,
        payload=payload,
    )

    bus = OrionBusAsync(url=bus_url)
    await bus.connect()
    probe = ProbeCollector(
        bus,
        patterns=[
            "orion:cortex:*",
            "orion:verb:*",
            "orion:exec:request:*",
            "orion:exec:result:*",
            "orion:llm:*",
            "orion:cognition:trace",
        ],
        corr_id=corr,
    )
    await probe.start()

    started = _now_iso()
    error: Optional[str] = None
    raw_payload: Optional[Dict[str, Any]] = None
    result_kind: Optional[str] = None
    try:
        msg = await bus.rpc_request(
            "orion:cortex:request",
            env,
            reply_channel=reply_channel,
            timeout_sec=timeout_sec,
        )
        dec = bus.codec.decode(msg.get("data"))
        if not dec.ok:
            error = f"decode_failed:{dec.error}"
        else:
            result_kind = dec.envelope.kind
            raw_payload = dec.envelope.payload if isinstance(dec.envelope.payload, dict) else {}
    except Exception as e:
        error = str(e)
    finally:
        await asyncio.sleep(0.4)
        await probe.stop()
        await bus.close()

    finished = _now_iso()
    # Normalize to the same schema expected by _summarize_live_result.
    response_payload: Dict[str, Any] | None = None
    if isinstance(raw_payload, dict):
        response_payload = {
            "cortex_result": raw_payload,
            "final_text": raw_payload.get("final_text"),
        }
    return {
        "entrypoint": "orch",
        "scenario": scenario.name,
        "started_at": started,
        "finished_at": finished,
        "correlation_id": corr,
        "request_channel": "orion:cortex:request",
        "reply_channel": reply_channel,
        "result_kind": result_kind,
        "request_payload": payload,
        "response_payload": response_payload,
        "error": error,
        "probe_events": probe.events,
    }


def _summarize_live_result(raw: Dict[str, Any], *, scenario_name: str) -> Dict[str, Any]:
    payload = raw.get("response_payload") or {}
    # Gateway wraps result as CortexChatResult(cortex_result=..., final_text=...)
    if isinstance(payload, dict):
        cr = payload.get("cortex_result")
        # Some gateway routes may return CortexClientResult directly (without outer wrapper).
        if isinstance(cr, dict):
            cortex_result = cr
            final_text = payload.get("final_text")
        else:
            cortex_result = payload if ("status" in payload and "mode" in payload and "verb" in payload) else {}
            final_text = payload.get("final_text")
    else:
        cortex_result = {}
        final_text = None
    if not final_text and isinstance(cortex_result, dict):
        final_text = cortex_result.get("final_text")
    metadata = cortex_result.get("metadata") if isinstance(cortex_result, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}

    # Prefer first-class metadata.answer_depth from Orch (Pass 5)
    answer_depth = metadata.get("answer_depth") if isinstance(metadata, dict) else None
    if isinstance(answer_depth, dict):
        output_mode = answer_depth.get("output_mode")
        response_profile = answer_depth.get("response_profile")
        packs = answer_depth.get("packs")
        resolved_tool_ids = answer_depth.get("resolved_tool_ids")
        triage_blocked = answer_depth.get("triage_blocked_post_step0")
        repeated_plan_escalated = answer_depth.get("repeated_plan_action_escalation")
        finalize_invoked = answer_depth.get("finalize_response_invoked")
        quality_rewrite = answer_depth.get("quality_evaluator_rewrite")
    else:
        # Fallback: extract from steps[i].result.AgentChainService.runtime_debug
        live_runtime_debug = _extract(cortex_result, "steps")
        runtime_debug_guess = None
        if isinstance(live_runtime_debug, list):
            for step in live_runtime_debug:
                if not isinstance(step, dict):
                    continue
                result = step.get("result") or {}
                if isinstance(result, dict):
                    ac = result.get("AgentChainService")
                    if isinstance(ac, dict):
                        runtime_debug_guess = ac.get("runtime_debug")
                        if runtime_debug_guess:
                            break
        if isinstance(runtime_debug_guess, dict):
            output_mode = runtime_debug_guess.get("output_mode")
            response_profile = runtime_debug_guess.get("response_profile")
            packs = runtime_debug_guess.get("packs")
            resolved_tool_ids = runtime_debug_guess.get("resolved_tool_ids")
            triage_blocked = runtime_debug_guess.get("triage_blocked_post_step0")
            repeated_plan_escalated = runtime_debug_guess.get("repeated_plan_action_escalation")
            finalize_invoked = runtime_debug_guess.get("finalize_response_invoked")
            quality_rewrite = runtime_debug_guess.get("quality_evaluator_rewrite")
        else:
            output_mode = None
            response_profile = None
            packs = None
            resolved_tool_ids = None
            triage_blocked = None
            repeated_plan_escalated = None
            finalize_invoked = None
            quality_rewrite = None

    # Supervisor scenario uses relaxed gate (4 of 6) for concise answers
    min_pos = 4 if scenario_name == "supervisor_meta_plan_live" else None
    quality = _quality_checks(str(final_text or ""), min_positive_patterns=min_pos)
    unavailable: Dict[str, str] = {}
    if output_mode is None and response_profile is None and packs is None and resolved_tool_ids is None:
        if not isinstance(answer_depth, dict):
            unavailable["runtime_debug"] = "metadata.answer_depth and AgentChain runtime_debug not present."

    tool_sequence = None

    # Infer some tool sequence from exec steps and probe events.
    if isinstance(cortex_result.get("steps"), list):
        tool_sequence = [str(s.get("step_name")) for s in cortex_result["steps"] if isinstance(s, dict) and s.get("step_name")]
    if not tool_sequence:
        unavailable["tool_sequence"] = "No step_name sequence available from cortex_result.steps."

    events = raw.get("probe_events") or []
    saw_orch = any(e.get("kind") == "cortex.orch.request" for e in events)
    saw_exec = any(e.get("kind") == "cortex.exec.request" for e in events)
    saw_verb_req = any(e.get("kind") == "verb.request" for e in events)
    saw_planner = any(e.get("kind") == "agent.planner.request" for e in events)
    saw_agent_chain = any(e.get("kind") == "agent.chain.request" for e in events)
    saw_llm = any(e.get("kind") == "llm.chat.request" for e in events)
    # Orch->verb->exec is internal; we require orch + planner + llm + agent_chain for live path proof.
    real_orch_path = saw_orch and saw_verb_req and (saw_planner or saw_agent_chain) and saw_llm

    pass_checks = {
        "real_orch_path_observed": real_orch_path,
        "plannerreact_bus_observed": saw_planner,
        "llm_bus_observed": saw_llm,
        "output_mode_expected": output_mode == "implementation_guide" if output_mode is not None else None,
        "response_profile_expected": response_profile == "technical_delivery" if response_profile is not None else None,
        "delivery_pack_active": ("delivery_pack" in (packs or [])) if packs is not None else None,
        "delivery_verbs_visible": (
            isinstance(resolved_tool_ids, list)
            and ("write_guide" in resolved_tool_ids or "finalize_response" in resolved_tool_ids)
        )
        if resolved_tool_ids is not None
        else None,
        "triage_not_after_step0": triage_blocked if triage_blocked is not None else None,
        "repeated_plan_action_not_shallow": repeated_plan_escalated if repeated_plan_escalated is not None else None,
        "finalization_when_needed": finalize_invoked if finalize_invoked is not None else None,
        "quality_evaluator_rewrite": quality_rewrite if quality_rewrite is not None else None,
        "answer_quality_concrete": quality["overall_pass"],
    }

    overall_pass = (
        raw.get("error") is None
        and pass_checks["real_orch_path_observed"] is True
        and pass_checks["plannerreact_bus_observed"] is True
        and pass_checks["llm_bus_observed"] is True
        and quality["overall_pass"] is True
    )

    return {
        "scenario": scenario_name,
        "timestamp": _now_iso(),
        "correlation_id": raw.get("correlation_id"),
        "request_channel": raw.get("request_channel"),
        "entrypoint": raw.get("entrypoint"),
        "reply_channel": raw.get("reply_channel"),
        "result_kind": raw.get("result_kind"),
        "error": raw.get("error"),
        "request_text": _extract(raw, "request_payload", "prompt"),
        "output_mode": output_mode,
        "response_profile": response_profile,
        "packs": packs,
        "resolved_tool_ids": resolved_tool_ids,
        "tool_sequence": tool_sequence,
        "triage_blocked_post_step0": triage_blocked,
        "repeated_plan_action_escalation": repeated_plan_escalated,
        "finalize_response_invoked": finalize_invoked,
        "quality_evaluator_rewrite": quality_rewrite,
        "answer_excerpt": (str(final_text or "")[:1600]),
        "quality_checks": quality,
        "pass_checks": pass_checks,
        "overall_pass": overall_pass,
        "path_observed": {
            "gateway_request_kind": any(e.get("kind") == "cortex.gateway.chat.request" for e in events),
            "orch_request_kind": saw_orch,
            "exec_request_kind": saw_exec,
            "verb_request_kind": saw_verb_req,
            "planner_request_kind": saw_planner,
            "agent_chain_request_kind": saw_agent_chain,
            "llm_request_kind": saw_llm,
        },
        "probe_events_excerpt": events[:120],
        "unavailable_fields": unavailable,
        "raw_response_payload": raw.get("response_payload"),
    }


def _write_evidence(base_name: str, evidence: Dict[str, Any]) -> None:
    LIVE_PROOF_DIR.mkdir(parents=True, exist_ok=True)
    json_path = LIVE_PROOF_DIR / f"{base_name}.json"
    md_path = LIVE_PROOF_DIR / f"{base_name}.md"
    json_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")

    checks = evidence.get("pass_checks") or {}
    md = [
        f"# {base_name.replace('_', ' ').title()}",
        "",
        f"- timestamp: `{evidence.get('timestamp')}`",
        f"- correlation_id: `{evidence.get('correlation_id')}`",
        f"- request_channel: `{evidence.get('request_channel')}`",
        f"- reply_channel: `{evidence.get('reply_channel')}`",
        f"- result_kind: `{evidence.get('result_kind')}`",
        f"- overall_pass: `{evidence.get('overall_pass')}`",
        "",
        "## Runtime Signals",
        f"- output_mode: `{evidence.get('output_mode')}`",
        f"- response_profile: `{evidence.get('response_profile')}`",
        f"- packs: `{evidence.get('packs')}`",
        f"- resolved_tool_ids (excerpt): `{(evidence.get('resolved_tool_ids') or [])[:30]}`",
        f"- tool_sequence: `{evidence.get('tool_sequence')}`",
        f"- triage_blocked_post_step0: `{evidence.get('triage_blocked_post_step0')}`",
        f"- repeated_plan_action_escalation: `{evidence.get('repeated_plan_action_escalation')}`",
        f"- finalize_response_invoked: `{evidence.get('finalize_response_invoked')}`",
        "",
        "## Path Observed",
        f"- {json.dumps(evidence.get('path_observed') or {}, ensure_ascii=False)}",
        "",
        "## Pass Checks",
        "```json",
        json.dumps(checks, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Quality Checks",
        "```json",
        json.dumps(evidence.get("quality_checks") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Final Answer Excerpt",
        "```text",
        evidence.get("answer_excerpt") or "",
        "```",
    ]
    if evidence.get("error"):
        md.extend(
            [
                "",
                "## Error",
                "```text",
                str(evidence["error"]),
                "```",
            ]
        )
    if evidence.get("unavailable_fields"):
        md.extend(
            [
                "",
                "## Unavailable Fields",
                "```json",
                json.dumps(evidence["unavailable_fields"], indent=2, ensure_ascii=False),
                "```",
            ]
        )
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")


async def _run_one(
    *,
    bus_url: str,
    scenario: LiveScenario,
    timeout_sec: float,
    base_name: str,
    extra_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    raw = await _run_gateway_chat(
        bus_url=bus_url,
        scenario=scenario,
        timeout_sec=timeout_sec,
        extra_options=extra_options,
    )
    gateway_error = raw.get("error")
    if gateway_error:
        # Gateway unavailable in this environment? Fall back to real Orch entry and record this explicitly.
        raw = await _run_orch_chat(
            bus_url=bus_url,
            scenario=scenario,
            timeout_sec=timeout_sec,
            extra_options=extra_options,
        )
        if isinstance(raw, dict):
            raw.setdefault("fallback_from_gateway_error", gateway_error)
    ev = _summarize_live_result(raw, scenario_name=scenario.name)
    if raw.get("fallback_from_gateway_error"):
        ev.setdefault("unavailable_fields", {})
        ev["unavailable_fields"]["gateway_path"] = (
            "Gateway path timed out in this run; evidence captured via real Orch path fallback."
        )
        ev["gateway_error"] = raw.get("fallback_from_gateway_error")
    _write_evidence(base_name, ev)
    return ev


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live answer-depth proof through real Cortex Gateway/Orch path.")
    parser.add_argument("--bus-url", default=os.getenv("ORION_BUS_URL", "redis://100.92.216.81:6379/0"))
    parser.add_argument("--timeout-sec", type=float, default=240.0)
    parser.add_argument(
        "--scenario",
        choices=["all", "discord", "supervisor"],
        default="all",
        help="Run one or both live scenarios.",
    )
    parser.add_argument(
        "--allow-partial-pass",
        action="store_true",
        help="Exit 0 even if checks fail; evidence is still emitted.",
    )
    return parser.parse_args(argv)


async def _amain(argv: List[str]) -> int:
    args = _parse_args(argv)
    print(f"[live-proof] bus_url={args.bus_url}")
    print(f"[live-proof] evidence_dir={LIVE_PROOF_DIR}")

    results: List[Dict[str, Any]] = []
    if args.scenario in {"all", "discord"}:
        results.append(
            await _run_one(
                bus_url=args.bus_url,
                scenario=LiveScenario(name="discord_deploy_live", prompt=DISCORD_PROMPT),
                timeout_sec=args.timeout_sec,
                base_name="discord_deploy_live_evidence",
                extra_options={"supervised": True, "diagnostic": True},
            )
        )
    if args.scenario in {"all", "supervisor"}:
        results.append(
            await _run_one(
                bus_url=args.bus_url,
                scenario=LiveScenario(name="supervisor_meta_plan_live", prompt=SUPERVISOR_PROMPT),
                timeout_sec=args.timeout_sec,
                base_name="supervisor_meta_plan_live_evidence",
                extra_options={"supervised": True, "diagnostic": True, "max_steps": 3},
            )
        )

    summary = {
        "timestamp": _now_iso(),
        "bus_url": args.bus_url,
        "results": [
            {
                "scenario": r.get("scenario"),
                "correlation_id": r.get("correlation_id"),
                "overall_pass": r.get("overall_pass"),
                "error": r.get("error"),
            }
            for r in results
        ],
    }
    LIVE_PROOF_DIR.mkdir(parents=True, exist_ok=True)
    (LIVE_PROOF_DIR / "live_proof_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    failed = [r for r in results if not r.get("overall_pass")]
    if failed and not args.allow_partial_pass:
        return 1
    return 0


def main() -> int:
    return asyncio.run(_amain(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())

