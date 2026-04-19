#!/usr/bin/env python3
"""
Live bus matrix: skill vs chat_general vs agent_runtime through cortex-gateway.

Requires: ORION_BUS_URL (default redis://127.0.0.1:6379/0), reachable Redis, and
live cortex-gateway + cortex-orch + cortex-exec (+ planner-react, agent-chain, LLM as configured).

Results reflect whatever **cortex-orch** revision is deployed against that bus; after changing
`services/orion-cortex-orch/app/decision_router.py`, redeploy orch before expecting updated
auto-route rows (e.g. coaching → chat_general).

Usage:
  ORION_BUS_URL=redis://host:6379/0 python scripts/run_agent_flow_live_matrix.py
  AGENT_FLOW_TIMEOUT_SEC=600 python scripts/run_agent_flow_live_matrix.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
_orig_cwd = os.getcwd()
# When cwd is the repo, '' on sys.path makes `import platform` resolve repo `platform.py` and breaks `uuid`.
while sys.path and sys.path[0] in ("", str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    sys.path.pop(0)
from uuid import uuid4

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.cortex.contracts import CortexChatRequest


def _service_ref() -> ServiceRef:
    return ServiceRef(name="agent-flow-live-matrix", version="0.0.1", node="local")


@dataclass
class MatrixRow:
    label: str
    expected_lane: str  # skill | chat_general | agent_runtime
    prompt: str
    mode: str
    route_intent: str
    verb: Optional[str] = None
    recall_enabled: bool = False
    routing_note: str = ""


@dataclass
class ProbeCollector:
    bus: OrionBusAsync
    patterns: List[str]
    corr_id: str
    events: List[Dict[str, Any]] = field(default_factory=list)
    _task: Optional[asyncio.Task] = None
    _stop: asyncio.Event = field(default_factory=asyncio.Event)
    _ready: asyncio.Event = field(default_factory=asyncio.Event)

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())
        await asyncio.wait_for(self._ready.wait(), timeout=5.0)

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
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
                        "channel": channel,
                        "kind": env.kind,
                        "source_service": env.source.name if env.source else None,
                    }
                )


def _extract_cortex_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    cr = payload.get("cortex_result")
    if isinstance(cr, dict):
        return cr
    if "status" in payload and "verb" in payload:
        return payload
    return {}


def _summarize(row: MatrixRow, raw: Dict[str, Any], elapsed_ms: float) -> Dict[str, Any]:
    events = raw.get("probe_events") or []
    err = raw.get("error")
    corr = raw.get("correlation_id")
    reply_wait = raw.get("reply_channel")
    payload = raw.get("response_payload") or {}
    cr = _extract_cortex_result(payload)
    metadata = cr.get("metadata") if isinstance(cr, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    auto_route = metadata.get("auto_route") if isinstance(metadata.get("auto_route"), dict) else {}
    exec_depth = auto_route.get("execution_depth")
    trace_verb = metadata.get("trace_verb") or cr.get("verb")
    selected_verb = cr.get("verb")
    mode = cr.get("mode")
    final_text = payload.get("final_text") or cr.get("final_text")
    steps = cr.get("steps") if isinstance(cr.get("steps"), list) else []
    step_names = [str(s.get("step_name")) for s in steps if isinstance(s, dict) and s.get("step_name")]
    planner_invoked = any(e.get("kind") == "agent.planner.request" for e in events)
    chain_invoked = any(e.get("kind") == "agent.chain.request" for e in events)
    result_channel = None
    if not err:
        for e in reversed(events):
            if e.get("kind") == "cortex.gateway.chat.result":
                result_channel = e.get("channel")
                break
    preview = (final_text or "")[:400].replace("\n", " ")
    return {
        "label": row.label,
        "expected_lane": row.expected_lane,
        "routing_note": row.routing_note,
        "correlation_id": corr,
        "selected_verb": selected_verb,
        "effective_verb": trace_verb,
        "execution_depth": exec_depth,
        "mode": mode,
        "planner_invoked": planner_invoked,
        "agent_chain_invoked": chain_invoked,
        "reply_channel_waited": reply_wait,
        "final_result_channel": result_channel,
        "elapsed_ms": round(elapsed_ms, 1),
        "error_or_timeout": err,
        "final_text_preview": preview,
        "step_names": step_names,
    }


async def _one_row(
    bus_url: str,
    row: MatrixRow,
    gateway_request_channel: str,
    timeout_sec: float,
) -> Dict[str, Any]:
    corr = str(uuid4())
    reply_channel = f"orion:cortex:gateway:result:{corr}"
    opts: Dict[str, Any] = {}
    if row.recall_enabled is False:
        opts["route_intent"] = row.route_intent
    payload = CortexChatRequest(
        prompt=row.prompt,
        mode=row.mode,  # type: ignore[arg-type]
        route_intent=row.route_intent,  # type: ignore[arg-type]
        verb=row.verb,
        packs=["executive_pack"],
        options=opts or None,
        recall=(
            {"enabled": False, "required": False, "mode": "hybrid"}
            if not row.recall_enabled
            else {"enabled": True, "required": False, "mode": "hybrid", "profile": None}
        ),
    ).model_dump(mode="json")

    env = BaseEnvelope(
        kind="cortex.gateway.chat.request",
        source=_service_ref(),
        correlation_id=corr,
        reply_to=reply_channel,
        payload=payload,
    )
    bus = OrionBusAsync(url=bus_url)
    t0 = time.perf_counter()
    error: Optional[str] = None
    response_payload: Optional[Dict[str, Any]] = None
    probe_events: List[Dict[str, Any]] = []
    try:
        await bus.connect()
        probe = ProbeCollector(
            bus,
            patterns=[
                "orion:cortex:*",
                "orion:verb:*",
                "orion:exec:request:*",
                "orion:exec:result:*",
                "orion:llm:*",
            ],
            corr_id=corr,
        )
        await probe.start()
        msg = await bus.rpc_request(
            gateway_request_channel,
            env,
            reply_channel=reply_channel,
            timeout_sec=timeout_sec,
        )
        dec = bus.codec.decode(msg.get("data"))
        if not dec.ok:
            error = f"decode_failed:{dec.error}"
        else:
            response_payload = dec.envelope.payload if isinstance(dec.envelope.payload, dict) else {}
    except Exception as e:
        error = str(e)
    finally:
        if "probe" in locals():
            await asyncio.sleep(0.5)
            await probe.stop()
            probe_events = list(probe.events)
        await bus.close()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    raw = {
        "correlation_id": corr,
        "reply_channel": reply_channel,
        "response_payload": response_payload,
        "error": error,
        "probe_events": probe_events,
    }
    return _summarize(row, raw, elapsed_ms)


def main() -> int:
    try:
        os.chdir(_orig_cwd)
    except OSError:
        pass
    bus_url = os.environ.get("ORION_BUS_URL", "redis://127.0.0.1:6379/0")
    gateway_ch = os.environ.get("CORTEX_GATEWAY_REQUEST_CHANNEL", "orion:cortex:gateway:request")
    timeout_sec = float(os.environ.get("AGENT_FLOW_TIMEOUT_SEC", "600"))

    rows = [
        MatrixRow(
            label="skill_time",
            expected_lane="skill",
            prompt="What time is it right now?",
            mode="brain",
            route_intent="none",
            verb="skills.system.time_now.v1",
            recall_enabled=False,
        ),
        MatrixRow(
            label="chat_general_motivation",
            expected_lane="chat_general",
            prompt="How do I motivate myself to do stuff?",
            mode="auto",
            route_intent="auto",
            recall_enabled=False,
        ),
        MatrixRow(
            label="agent_runtime_docker_debug",
            expected_lane="agent_runtime",
            prompt="How do I debug why my docker compose stack is hanging?",
            mode="auto",
            route_intent="auto",
            recall_enabled=False,
        ),
        MatrixRow(
            label="agent_runtime_logs_ops",
            expected_lane="agent_runtime",
            prompt="Analyze these logs and tell me the first bad hop",
            mode="agent",
            route_intent="none",
            verb="agent_runtime",
            recall_enabled=False,
            routing_note="explicit_agent_runtime (auto-router would choose analyze_text for this wording)",
        ),
        MatrixRow(
            label="agent_runtime_workflow_design",
            expected_lane="agent_runtime",
            prompt="Help me design a workflow for log triage with planner and agent chain",
            mode="auto",
            route_intent="auto",
            recall_enabled=False,
        ),
    ]

    async def _run_all() -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for row in rows:
            out.append(await _one_row(bus_url, row, gateway_ch, timeout_sec))
        return out

    results = asyncio.run(_run_all())
    print(json.dumps({"bus_url": bus_url, "gateway_request_channel": gateway_ch, "timeout_sec": timeout_sec, "rows": results}, indent=2))

    failures = [r for r in results if r.get("error_or_timeout")]
    agent_rows = [r for r in results if r["expected_lane"] == "agent_runtime"]
    agent_fail = [r for r in agent_rows if r.get("error_or_timeout")]
    if agent_fail:
        return 3
    if failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
