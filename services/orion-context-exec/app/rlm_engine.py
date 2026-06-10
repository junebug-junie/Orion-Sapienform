from __future__ import annotations

import asyncio
import logging
from typing import Any

from orion.schemas.context_exec import ContextExecRequestV1

from .callable_namespace import ContextNamespace

logger = logging.getLogger("orion-context-exec.rlm_engine")


class RLMEngine:
    async def run(self, request: ContextExecRequestV1, namespace: ContextNamespace) -> Any:
        raise NotImplementedError


class FakeRLMEngine(RLMEngine):
    """Deterministic engine for tests — no model dependency."""

    async def run(self, request: ContextExecRequestV1, namespace: ContextNamespace) -> Any:
        mode = request.mode
        text = request.text.lower()

        if mode == "belief_provenance" or "denver" in text:
            memory_hits = namespace.memory.search_claims("Denver", limit=20)
            trace_hits = namespace.traces.search(query="Denver")
            report = {
                "claim": "User is from Denver",
                "status": "contradicted" if memory_hits else "unknown",
                "likely_origin": "memory/recall cross-check",
                "confidence": 0.72 if memory_hits else 0.2,
                "recommended_action": "mark_uncertain",
                "findings": [],
                "missing_evidence": [],
            }
            for hit in memory_hits:
                report["findings"].append(
                    {
                        "claim": str(hit.get("claim") or "Denver location mention"),
                        "evidence_type": "user_statement",
                        "source_ref": hit.get("source_ref"),
                        "verified": bool(hit.get("verified")),
                        "confidence": float(hit.get("confidence") or 0.5),
                        "scope": "fact",
                    }
                )
            for th in trace_hits:
                report["findings"].append(
                    {
                        "claim": str(th.get("snippet") or "trace hit"),
                        "evidence_type": "runtime_log",
                        "source_ref": th.get("handle"),
                        "verified": True,
                        "confidence": 0.8,
                        "scope": "fact",
                    }
                )
            namespace.set_local("report", report)
            namespace.FINAL_VAR("report")
            return namespace.get_final()

        if mode == "trace_autopsy":
            corr = None
            for token in text.split():
                if token.isdigit() or token.startswith("corr"):
                    corr = token.replace("corr", "").strip()
                    break
            report = {
                "target": corr or request.text[:80],
                "status": "partial",
                "failure_chain": ["rpc timeout", "fail open"],
                "root_cause": "parent RPC timeout before child reply",
                "contributing_factors": ["step_timeout_ms too low"],
                "evidence": [],
                "recommended_patch": "Increase context-exec budget or fix reply channel wiring",
            }
            namespace.set_local("report", report)
            namespace.FINAL_VAR("report")
            return namespace.get_final()

        if mode == "repo_impact_analysis":
            hits = namespace.repo.grep("agent.chain|AgentChainClient", limit=20)
            report = {
                "proposed_change": request.text[:500],
                "status": "analyzed" if hits else "insufficient_grounding",
                "affected_paths": [h.get("path") for h in hits if isinstance(h, dict) and h.get("path")][:20],
                "breaking_surfaces": ["cortex-exec AgentChainClient hop"],
                "compatibility_shims": ["AgentChainResult-compatible context-exec reply"],
                "tests_to_add_or_update": ["test_context_exec_depth2_routing"],
                "migration_steps": ["feature flag CONTEXT_EXEC_ENABLED"],
                "risk": "medium",
                "findings": [],
            }
            namespace.set_local("report", report)
            namespace.FINAL_VAR("report")
            return namespace.get_final()

        report = {
            "summary": f"Investigation complete for: {request.text[:200]}",
            "mode": mode,
        }
        namespace.set_local("report", report)
        namespace.FINAL_VAR("report")
        return namespace.get_final()


class TimeoutRLMEngine(RLMEngine):
    async def run(self, request: ContextExecRequestV1, namespace: ContextNamespace) -> Any:
        await asyncio.sleep(3600)
        return None


def build_engine(name: str) -> RLMEngine:
    if name == "timeout":
        return TimeoutRLMEngine()
    return FakeRLMEngine()
