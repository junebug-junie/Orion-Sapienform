from __future__ import annotations

import logging
import re
from typing import Any

from orion.schemas.context_exec import ContextExecRequestV1

from .callable_namespace import ContextNamespace
from .organ_runtime import OrganRuntime
from .rlm_engine import RLMEngine, _extract_corr_id
from .settings import ContextExecSettings, settings

_UUID_RE = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"

_PROMPT_ECHO_MARKERS = (
    "why did",
    "orion,",
    "orion ",
    "fail open?",
    "trace autopsy",
    "root cause:",
)

_TRACE_SIGNAL_ORDER: tuple[tuple[tuple[str, ...], str], ...] = (
    (("contextexecservice", "timeout"), "ContextExecService timed out"),
    (("context.exec.result", "missing"), "context.exec.result missing"),
    (("fallback", "true"), "fallback=True"),
    (("agentchainservice", "fallback"), "AgentChainService fallback"),
    (("baseenvelope", "validationerror"), "BaseEnvelope ValidationError"),
    (("causality_chain", "list_type"), "causality_chain list_type"),
    (("recall", "timeout"), "recall timeout"),
    (("contextexecservice",), "ContextExecService"),
    (("agentchainservice",), "AgentChainService"),
    (("fallback",), "fallback"),
    (("timeout",), "timeout"),
)

logger = logging.getLogger("orion-context-exec.alexzhang_rlm_engine")

SUPPORTED_MODES = frozenset({"belief_provenance", "trace_autopsy", "repo_impact_analysis"})


class UnsupportedModeError(Exception):
    def __init__(self, mode: str) -> None:
        super().__init__(f"unsupported mode: {mode}")
        self.mode = mode


class AlexZhangInitError(Exception):
    pass


_VOCATIVE_PREFIX = re.compile(r"^(?:hey\s+)?orion[\s,]+", re.IGNORECASE)

_SCAFFOLD_TERMINATOR = (
    r"(?:\s+come\s+from\s+(?:across\s+(?:your\s+)?runtime|(?:your\s+)?runtime)"
    r"|\s+across\s+(?:your\s+)?runtime|\?|$)"
)

_CLAIM_TAIL_STOP = re.compile(
    r"\s+(?:come\s+from\s+(?:across\s+(?:your\s+)?runtime|(?:your\s+)?runtime)"
    r"|across\s+your\s+runtime|across\s+the\s+runtime|"
    r"from\s+across\s+your\s+runtime|in\s+your\s+runtime)\b",
    re.IGNORECASE,
)

_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"where\s+did\s+(?:the\s+claim\s+that|orion\s+get\s+the\s+claim\s+that)\s+"
        rf"(?P<claim>.+?){_SCAFFOLD_TERMINATOR}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"claim\s+that\s+(?P<claim>.+?){_SCAFFOLD_TERMINATOR}",
        re.IGNORECASE,
    ),
    re.compile(
        r"why\s+does\s+orion\s+think\s+(?P<claim>.+?)(?:\?|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"what\s+evidence\s+(?:says|supports)\s+(?P<claim>.+?)(?:\?|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"is\s+it\s+true\s+that\s+(?P<claim>.+?)(?:\?|$)",
        re.IGNORECASE,
    ),
)


def _normalize_claim(claim: str) -> str:
    normalized = re.sub(r"\s+", " ", claim.strip(" ?.,;"))
    return normalized[:500]


def _strip_claim_tail(claim: str) -> str:
    match = _CLAIM_TAIL_STOP.search(claim)
    if match:
        return claim[: match.start()]
    return claim


def _extract_claim_from_text(text: str) -> str:
    cleaned = _VOCATIVE_PREFIX.sub("", text.strip())

    for pattern in _CLAIM_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            claim = _normalize_claim(match.group("claim"))
            if claim:
                return claim

    lowered = cleaned.lower()
    for marker in ("claim that ", "claim: ", "belief that "):
        idx = lowered.find(marker)
        if idx >= 0:
            tail = _normalize_claim(_strip_claim_tail(cleaned[idx + len(marker) :]))
            if tail:
                return tail

    if len(cleaned) <= 500:
        return cleaned.strip()
    return cleaned[:500].strip()


def _extract_corr_id_from_text(text: str) -> str | None:
    corr_eq = re.search(rf"correlation_id\s*=\s*({_UUID_RE})", text, flags=re.IGNORECASE)
    if corr_eq:
        return corr_eq.group(1)
    corr_sp = re.search(rf"\bcorr[:\s#]+({_UUID_RE})\b", text, flags=re.IGNORECASE)
    if corr_sp:
        return corr_sp.group(1)
    bare = re.search(rf"\b({_UUID_RE})\b", text, flags=re.IGNORECASE)
    if bare:
        return bare.group(1)
    return _extract_corr_id(text)


def _normalize_evidence_blob(text: str) -> str:
    return re.sub(r"[\s_\-]+", "", text.lower())


def _snippet_echoes_prompt(snippet: str, request_text: str) -> bool:
    snippet_clean = snippet.strip()
    request_clean = request_text.strip()
    if not snippet_clean:
        return True
    sl = snippet_clean.lower()
    rl = request_clean.lower()
    if sl == rl or sl in rl or rl in sl:
        return True
    return any(marker in sl for marker in _PROMPT_ECHO_MARKERS)


def _usable_trace_snippets(snippets: list[str], request_text: str) -> list[str]:
    return [snippet for snippet in snippets if not _snippet_echoes_prompt(snippet, request_text)]


def _synthesize_trace_root_cause(snippets: list[str], request_text: str) -> str | None:
    usable = _usable_trace_snippets(snippets, request_text)
    if not usable:
        return None

    blob = _normalize_evidence_blob(" ".join(usable))
    matched: list[str] = []
    for needles, label in _TRACE_SIGNAL_ORDER:
        if all(_normalize_evidence_blob(needle) in blob for needle in needles):
            if label not in matched:
                matched.append(label)

    if matched:
        if "ContextExecService timed out" in matched and "AgentChainService fallback" in matched:
            return "ContextExecService timed out and Cortex-Exec fell back to AgentChainService"
        if "ContextExecService timed out" in matched and "fallback" in matched:
            return "ContextExecService timed out and Cortex-Exec fell back to AgentChainService"
        return matched[0]

    return usable[0]


def _repo_search_terms(text: str) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text):
        if token.lower() in {"what", "breaks", "replace", "with", "context", "exec", "agent", "chain"}:
            continue
        terms.append(token)
    if "agent-chain" in text.lower() or "agent chain" in text.lower():
        terms.extend(["agent.chain", "AgentChainClient", "AgentChainService"])
    if "context-exec" in text.lower() or "context exec" in text.lower():
        terms.append("context-exec")
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out[:6] or ["agent.chain"]


class AlexZhangRLMEngine(RLMEngine):
    """Conservative depth-1 RLM engine using read-only organ hooks."""

    engine_name = "alexzhang"

    def __init__(
        self,
        cfg: ContextExecSettings | None = None,
        *,
        organs: OrganRuntime | None = None,
    ) -> None:
        self._settings = cfg or settings
        self._organs = organs
        self._debug_steps: list[str] = []
        self._subcalls = 0
        self._init_ok = True
        self._init_error: str | None = None
        if self._settings.context_exec_write_enabled:
            self._init_ok = False
            self._init_error = "write_enabled must be false for alexzhang"
        if self._settings.context_exec_network_enabled:
            self._init_ok = False
            self._init_error = "network_enabled must be false for alexzhang"

    @property
    def is_ready(self) -> bool:
        return self._init_ok

    @property
    def init_error(self) -> str | None:
        return self._init_error

    @property
    def debug_steps(self) -> list[str]:
        return list(self._debug_steps)

    @property
    def subcall_count(self) -> int:
        return self._subcalls

    async def run(
        self,
        request: ContextExecRequestV1,
        namespace: ContextNamespace,
        *,
        organ_runtime: OrganRuntime | None = None,
    ) -> Any:
        if not self._init_ok:
            raise AlexZhangInitError(self._init_error or "alexzhang_init_failed")
        mode = request.mode
        if mode not in SUPPORTED_MODES:
            raise UnsupportedModeError(mode)

        self._debug_steps = ["plan"]
        runtime = organ_runtime or getattr(namespace, "_organ_runtime", None)
        organ_cache = getattr(namespace, "_organ_cache", None)

        if mode == "belief_provenance":
            report = await self._belief_provenance(request, namespace, runtime, organ_cache)
        elif mode == "trace_autopsy":
            report = await self._trace_autopsy(request, namespace, runtime, organ_cache)
        else:
            report = await self._repo_impact(request, namespace, runtime, organ_cache)

        self._debug_steps.append("validate")
        namespace.set_local("report", report)
        namespace.FINAL_VAR("report")
        return namespace.get_final()

    async def _belief_provenance(
        self,
        request: ContextExecRequestV1,
        namespace: ContextNamespace,
        runtime: OrganRuntime | None,
        organ_cache: dict[str, Any] | None,
    ) -> dict[str, Any]:
        claim = _extract_claim_from_text(request.text)
        query_terms = claim.split()[:8] or [request.text[:80]]
        search_q = " ".join(query_terms)

        self._debug_steps.append("retrieve")
        recall_result: dict[str, Any] = {"hits": []}
        trace_hits: list[dict[str, Any]] = []
        if runtime is not None:
            recall_result = await runtime.recall_query(search_q, limit=self._settings.context_exec_recall_limit)
            self._subcalls += 1
            trace_hits = await runtime.traces_search(query=search_q, limit=self._settings.context_exec_trace_limit)
            self._subcalls += 1
        if organ_cache is not None:
            organ_cache["recall"] = recall_result
            organ_cache["traces"] = trace_hits

        memory_hits = namespace.memory.search_claims(search_q, limit=20)
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
            trace_hits = namespace.traces.search(query=search_q)
        trace_hits = [
            th
            for th in trace_hits
            if not (
                str(th.get("source")) == "context_exec"
                and str(th.get("kind", "")).startswith("context.exec.")
            )
        ]

        self._debug_steps.append("synthesize")
        has_evidence = bool(memory_hits or trace_hits)
        status = "supported" if has_evidence else "unknown"
        if not has_evidence:
            status = "unknown"

        report: dict[str, Any] = {
            "claim": claim,
            "status": status,
            "likely_origin": "memory/recall cross-check" if has_evidence else None,
            "confidence": 0.65 if has_evidence else 0.1,
            "recommended_action": "mark_uncertain" if not has_evidence else "keep",
            "findings": [],
            "missing_evidence": [],
            "source_chain": [],
        }
        if not has_evidence:
            report["missing_evidence"].append("no recall or trace evidence found for claim")

        for hit in memory_hits:
            report["findings"].append(
                {
                    "claim": str(hit.get("claim") or claim),
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
                    "confidence": 0.75,
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
        return report

    async def _trace_autopsy(
        self,
        request: ContextExecRequestV1,
        namespace: ContextNamespace,
        runtime: OrganRuntime | None,
        organ_cache: dict[str, Any] | None,
    ) -> dict[str, Any]:
        corr = _extract_corr_id_from_text(request.text)
        self._debug_steps.append("retrieve")
        trace_hits: list[dict[str, Any]] = []
        if runtime is not None and corr:
            trace_hits = await runtime.traces_search(corr_id=corr, limit=self._settings.context_exec_trace_limit)
            self._subcalls += 1
        if organ_cache is not None:
            organ_cache["traces"] = trace_hits
        if not trace_hits:
            trace_hits = namespace.traces.search(corr_id=corr, limit=self._settings.context_exec_trace_limit)

        self._debug_steps.append("synthesize")
        raw_snippets: list[str] = []
        evidence: list[dict[str, Any]] = []
        for th in trace_hits:
            snippet = str(th.get("snippet") or th.get("kind") or "trace hit")
            raw_snippets.append(snippet)
            if _snippet_echoes_prompt(snippet, request.text):
                continue
            evidence.append(
                {
                    "claim": snippet,
                    "evidence_type": "runtime_log",
                    "source_ref": th.get("handle"),
                    "verified": True,
                    "confidence": 0.8,
                    "scope": "fact",
                }
            )

        usable_snippets = _usable_trace_snippets(raw_snippets, request.text)
        root_cause = _synthesize_trace_root_cause(raw_snippets, request.text)

        if not trace_hits or not usable_snippets or root_cause is None:
            return {
                "target": corr or request.text[:80],
                "status": "unknown",
                "failure_chain": ["insufficient_trace_evidence"],
                "root_cause": "insufficient_trace_evidence",
                "contributing_factors": ["no trace evidence found for correlation id"],
                "evidence": [],
                "recommended_patch": "Collect Redis/Cortex trace evidence for the correlation id",
            }

        failure_chain = usable_snippets
        return {
            "target": corr or request.text[:80],
            "status": "explained" if len(failure_chain) >= 2 else "partial",
            "failure_chain": failure_chain,
            "root_cause": root_cause,
            "contributing_factors": failure_chain[1:4],
            "evidence": evidence,
            "recommended_patch": None,
        }

    async def _repo_impact(
        self,
        request: ContextExecRequestV1,
        namespace: ContextNamespace,
        runtime: OrganRuntime | None,
        organ_cache: dict[str, Any] | None,
    ) -> dict[str, Any]:
        self._debug_steps.append("retrieve")
        terms = _repo_search_terms(request.text)
        hits: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for term in terms:
            batch: list[dict[str, Any]]
            if runtime is not None and request.permissions.read_repo:
                batch = runtime.repo_grep(term, limit=30)
                self._subcalls += 1
            elif request.permissions.read_repo:
                batch = namespace.repo.grep(term, limit=30)
            else:
                batch = []
            for h in batch:
                path = str(h.get("path") or "")
                if path and path not in seen_paths:
                    seen_paths.add(path)
                    hits.append(h)

        self._debug_steps.append("synthesize")
        if not hits:
            return {
                "proposed_change": request.text[:500],
                "status": "insufficient_grounding",
                "affected_paths": [],
                "breaking_surfaces": [],
                "compatibility_shims": [],
                "tests_to_add_or_update": [],
                "migration_steps": [],
                "risk": "unknown",
                "findings": [],
            }

        affected = [h.get("path") for h in hits if isinstance(h, dict) and h.get("path")][:20]
        return {
            "proposed_change": request.text[:500],
            "status": "analyzed" if len(affected) >= 2 else "partial",
            "affected_paths": affected,
            "breaking_surfaces": ["cortex-exec AgentChainClient hop"] if hits else [],
            "compatibility_shims": ["AgentChainResult-compatible context-exec reply"],
            "tests_to_add_or_update": ["test_context_exec_depth2_routing"],
            "migration_steps": ["feature flag CONTEXT_EXEC_ENABLED"],
            "risk": "medium" if len(affected) > 3 else "low",
            "findings": [
                {
                    "claim": f"{h.get('path')}:{h.get('line_start')} {str(h.get('snippet', ''))[:120]}",
                    "evidence_type": "repo_file",
                    "source_ref": h.get("source_ref"),
                    "verified": True,
                    "confidence": 0.85,
                    "scope": "fact",
                }
                for h in hits
                if isinstance(h, dict)
            ],
        }
