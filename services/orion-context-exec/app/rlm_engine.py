from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from orion.schemas.context_exec import ContextExecRequestV1

from .callable_namespace import ContextNamespace
from .organ_runtime import OrganRuntime
from .settings import settings

logger = logging.getLogger("orion-context-exec.rlm_engine")


class RLMEngine:
    async def run(
        self,
        request: ContextExecRequestV1,
        namespace: ContextNamespace,
        *,
        organ_runtime: OrganRuntime | None = None,
    ) -> Any:
        raise NotImplementedError


def _extract_corr_id(text: str) -> str | None:
    for token in text.split():
        cleaned = token.strip(" ,.;:")
        if cleaned.lower().startswith("corr"):
            tail = cleaned[4:].strip(" :#")
            if tail:
                return tail
        if re.fullmatch(r"[0-9a-f-]{8,}", cleaned, flags=re.I):
            return cleaned
    return None


class FakeRLMEngine(RLMEngine):
    """Deterministic engine for tests — no model dependency."""

    async def run(
        self,
        request: ContextExecRequestV1,
        namespace: ContextNamespace,
        *,
        organ_runtime: OrganRuntime | None = None,
    ) -> Any:
        mode = request.mode
        text = request.text.lower()
        organ_cache = getattr(namespace, "_organ_cache", None)
        runtime = organ_runtime or getattr(namespace, "_organ_runtime", None)

        if mode == "belief_provenance" or "denver" in text:
            recall_result: dict[str, Any] = {"hits": []}
            trace_hits: list[dict[str, Any]] = []
            if runtime is not None:
                recall_result = await runtime.recall_query(request.text, limit=12)
                trace_hits = await runtime.traces_search(query="Denver", limit=20)
            if organ_cache is not None:
                organ_cache["recall"] = recall_result
                organ_cache["traces"] = trace_hits
            memory_hits = namespace.memory.search_claims("Denver", limit=20)
            if not memory_hits and recall_result.get("hits"):
                memory_hits = [
                    {
                        "claim": str(h.get("snippet") or h.get("title") or "recall hit"),
                        "source_ref": h.get("source_ref") or h.get("id"),
                        "verified": True,
                        "confidence": float(h.get("score") or 0.5),
                    }
                    for h in recall_result.get("hits") or []
                ]
            if not trace_hits:
                trace_hits = namespace.traces.search(query="Denver")
            trace_hits = [
                th
                for th in trace_hits
                if not (
                    str(th.get("source")) == "context_exec"
                    and str(th.get("kind", "")).startswith("context.exec.")
                )
            ]
            report = {
                "claim": "User is from Denver",
                "status": "supported" if (memory_hits or trace_hits) else "unknown",
                "likely_origin": "memory/recall cross-check",
                "confidence": 0.72 if memory_hits else 0.2,
                "recommended_action": "mark_uncertain",
                "findings": [],
                "missing_evidence": [],
                "source_chain": [],
            }
            if not memory_hits and not trace_hits:
                report["missing_evidence"].append("no recall or trace evidence found")
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
                report["source_chain"].append(
                    {"kind": "recall", "source_ref": hit.get("source_ref"), "claim": hit.get("claim")}
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
                report["source_chain"].append(
                    {
                        "kind": "trace",
                        "source": th.get("source"),
                        "corr_id": th.get("corr_id"),
                        "handle": th.get("handle"),
                    }
                )
            namespace.set_local("report", report)
            namespace.FINAL_VAR("report")
            return namespace.get_final()

        if mode == "trace_autopsy":
            corr = _extract_corr_id(request.text)
            trace_hits: list[dict[str, Any]] = []
            if runtime is not None and corr:
                trace_hits = await runtime.traces_search(corr_id=corr, limit=40)
            if organ_cache is not None:
                organ_cache["traces"] = trace_hits
            if not trace_hits:
                trace_hits = namespace.traces.search(corr_id=corr, limit=40)
            evidence = []
            failure_chain = []
            for th in trace_hits:
                snippet = str(th.get("snippet") or th.get("kind") or "trace hit")
                failure_chain.append(snippet)
                evidence.append(
                    {
                        "claim": snippet,
                        "evidence_type": "runtime_log",
                        "source_ref": th.get("handle"),
                        "verified": True,
                        "confidence": 0.85,
                        "scope": "fact",
                    }
                )
            status = "explained" if trace_hits else "unknown"
            root_cause = failure_chain[0] if failure_chain else None
            if not trace_hits:
                root_cause = "no trace evidence found for correlation id"
            report = {
                "target": corr or request.text[:80],
                "status": status,
                "failure_chain": failure_chain or (["no trace evidence found"] if not corr else ["corr not resolved"]),
                "root_cause": root_cause,
                "contributing_factors": [],
                "evidence": evidence,
                "recommended_patch": None
                if trace_hits
                else "Collect Redis/Cortex trace evidence for the correlation id",
            }
            namespace.set_local("report", report)
            namespace.FINAL_VAR("report")
            return namespace.get_final()

        if mode == "repo_impact_analysis":
            if runtime is not None and request.permissions.read_repo:
                hits = runtime.repo_grep("agent.chain|AgentChainClient|AgentChainService", limit=30)
            else:
                hits = namespace.repo.grep("agent.chain|AgentChainClient|AgentChainService", limit=30)
            affected = [h.get("path") for h in hits if isinstance(h, dict) and h.get("path")][:20]
            report = {
                "proposed_change": request.text[:500],
                "status": "analyzed" if hits else "insufficient_grounding",
                "affected_paths": affected,
                "breaking_surfaces": ["cortex-exec AgentChainClient hop"] if hits else [],
                "compatibility_shims": ["AgentChainResult-compatible context-exec reply"],
                "tests_to_add_or_update": ["test_context_exec_depth2_routing"],
                "migration_steps": ["feature flag CONTEXT_EXEC_ENABLED"],
                "risk": "medium" if hits else "unknown",
                "findings": [
                    {
                        "claim": f"{h.get('path')}:{h.get('line_start')} {h.get('snippet', '')[:120]}",
                        "evidence_type": "repo_file",
                        "source_ref": h.get("source_ref"),
                        "verified": True,
                        "confidence": 0.9,
                        "scope": "fact",
                    }
                    for h in hits
                    if isinstance(h, dict)
                ],
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
    async def run(
        self,
        request: ContextExecRequestV1,
        namespace: ContextNamespace,
        *,
        organ_runtime: OrganRuntime | None = None,
    ) -> Any:
        await asyncio.sleep(3600)
        return None


def build_engine(name: str) -> RLMEngine:
    if name == "timeout":
        return TimeoutRLMEngine()
    return FakeRLMEngine()
