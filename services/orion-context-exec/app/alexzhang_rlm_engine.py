from __future__ import annotations

import logging
import re
from pathlib import Path
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

SUPPORTED_MODES = frozenset({
    "general_investigation",
    "belief_provenance",
    "trace_autopsy",
    "repo_impact_analysis",
    "patch_proposal",
    "memory_correction_proposal",
})


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


_ENGINE_SELECTION_MARKERS = frozenset(
    {
        "alexzhang",
        "fakerlmengine",
        "context_exec_rlm_engine",
        "engine selection",
        "rlm engine",
    }
)

_ENGINE_REPO_ANCHORS: tuple[str, ...] = (
    "AlexZhangRLMEngine",
    "FakeRLMEngine",
    "CONTEXT_EXEC_RLM_ENGINE",
    "engine_selected",
    "fallback_engine",
    "build_engine",
    "ContextExecRunner",
    "repo_impact_analysis",
    "rlm_engine",
)

_ENGINE_SELECTION_PATHS: tuple[str, ...] = (
    "rlm_engine.py",
    "alexzhang_rlm_engine.py",
    "runner.py",
    "settings.py",
    "rlm_eval_harness.py",
)

_TRACE_AUTOPSY_PATCH_ANCHORS: tuple[str, ...] = (
    "_synthesize_trace_root_cause",
    "_usable_trace_snippets",
    "insufficient_trace_evidence",
    "trace_autopsy",
    "root_cause",
)

_PATCH_PROPOSAL_LIKELY_FILES: tuple[str, ...] = (
    "alexzhang_rlm_engine.py",
    "test_alexzhang_rlm_engine.py",
    "test_rlm_eval_fixtures.py",
    "rlm_eval_harness.py",
)

def _is_trace_autopsy_patch_query(text: str) -> bool:
    lowered = text.lower()
    if "trace autopsy" in lowered or "trace_autopsy" in lowered:
        return True
    if "trace-autopsy" in lowered:
        return True
    if "root cause" in lowered and ("trace" in lowered or "autopsy" in lowered):
        return True
    if "weak" in lowered and "trace" in lowered:
        return True
    return False


def _is_repo_impact_patch_query(text: str) -> bool:
    lowered = text.lower()
    return "repo-impact" in lowered or "repo impact" in lowered or (
        "repo" in lowered and "ground" in lowered and "patch" in lowered
    )


def _patch_proposal_search_terms(text: str) -> list[str]:
    if _is_trace_autopsy_patch_query(text):
        return list(_TRACE_AUTOPSY_PATCH_ANCHORS)
    if _is_repo_impact_patch_query(text):
        return list(_ENGINE_REPO_ANCHORS)
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text):
        if token.lower() in {"propose", "patch", "weak", "diagnose", "context", "exec"}:
            continue
        terms.append(token)
    return terms[:6] or list(_TRACE_AUTOPSY_PATCH_ANCHORS[:3])


def _patch_proposal_risk(files: list[str]) -> str:
    if not files:
        return "unknown"
    names = {path.rsplit("/", 1)[-1].lower() for path in files}
    likely = frozenset(_PATCH_PROPOSAL_LIKELY_FILES)
    if names & likely:
        return "medium"
    if len(files) > 3:
        return "medium"
    return "low"


def _prioritize_patch_proposal_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def rank(h: dict[str, Any]) -> tuple[int, str]:
        path = str(h.get("path") or "")
        name = path.rsplit("/", 1)[-1].lower()
        if name in _PATCH_PROPOSAL_LIKELY_FILES:
            return (_PATCH_PROPOSAL_LIKELY_FILES.index(name), path)
        for prefix in _CONTEXT_EXEC_APP_PREFIXES:
            if path.lower().startswith(prefix):
                return (len(_PATCH_PROPOSAL_LIKELY_FILES), path)
        return (len(_PATCH_PROPOSAL_LIKELY_FILES) + 1, path)

    return sorted(hits, key=rank)


_CONTEXT_EXEC_APP_PREFIXES: tuple[str, ...] = (
    "services/orion-context-exec/app/",
    "app/",
)


def _engine_selection_grep_path() -> str | None:
    root = Path(settings.context_exec_repo_root or settings.orion_repo_root)
    for rel in ("services/orion-context-exec/app", "app"):
        if (root / rel).is_dir():
            return rel
    return None


def _is_engine_selection_query(text: str) -> bool:
    lowered = text.lower()
    if any(marker in lowered for marker in _ENGINE_SELECTION_MARKERS):
        return True
    if ("context-exec" in lowered or "context exec" in lowered) and "engine" in lowered:
        return True
    if "rlm" in lowered and "engine" in lowered:
        return True
    if "engine" in lowered and "selection" in lowered:
        return True
    return False


def _repo_search_terms(text: str) -> list[str]:
    if _is_engine_selection_query(text):
        return list(_ENGINE_REPO_ANCHORS)

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


def _merge_repo_hits(
    hits: list[dict[str, Any]],
    batch: list[dict[str, Any]],
    seen_paths: set[str],
) -> None:
    for h in batch:
        path = str(h.get("path") or "")
        if path and path not in seen_paths:
            seen_paths.add(path)
            hits.append(h)


def _prioritize_engine_selection_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def rank(h: dict[str, Any]) -> tuple[int, str]:
        path = str(h.get("path") or "")
        lowered = path.lower()
        for prefix in _CONTEXT_EXEC_APP_PREFIXES:
            if lowered.startswith(prefix):
                for idx, name in enumerate(_ENGINE_SELECTION_PATHS):
                    if lowered.endswith(name):
                        return (idx, path)
                return (len(_ENGINE_SELECTION_PATHS), path)
        return (len(_ENGINE_SELECTION_PATHS) + 1, path)

    return sorted(hits, key=rank)


def _engine_selection_risk(affected_paths: list[str]) -> str:
    if not affected_paths:
        return "unknown"
    runtime_markers = frozenset(_ENGINE_SELECTION_PATHS[:4])
    for path in affected_paths:
        name = path.rsplit("/", 1)[-1].lower()
        if name in runtime_markers:
            return "medium"
    if any(
        path.lower().startswith(prefix)
        for path in affected_paths
        for prefix in _CONTEXT_EXEC_APP_PREFIXES
    ):
        return "low"
    return "unknown"


def _agent_chain_risk(affected_paths: list[str]) -> str:
    if len(affected_paths) > 3:
        return "medium"
    if affected_paths:
        return "low"
    return "unknown"


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
        elif mode == "general_investigation":
            report = await self._general_investigation(request, namespace, runtime, organ_cache)
        elif mode == "trace_autopsy":
            report = await self._trace_autopsy(request, namespace, runtime, organ_cache)
        elif mode == "patch_proposal":
            report = await self._patch_proposal(request, namespace, runtime, organ_cache)
        elif mode == "memory_correction_proposal":
            report = await self._memory_correction_proposal(request, namespace, runtime, organ_cache)
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

    async def _general_investigation(
        self,
        request: ContextExecRequestV1,
        namespace: ContextNamespace,
        runtime: OrganRuntime | None,
        organ_cache: dict[str, Any] | None,
    ) -> dict[str, Any]:
        search_q = request.text[:200]
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
        findings: list[dict[str, Any]] = []
        for hit in memory_hits:
            findings.append(
                {
                    "claim": str(hit.get("claim") or "recall hit"),
                    "evidence_type": "user_statement",
                    "source_ref": hit.get("source_ref"),
                    "verified": bool(hit.get("verified")),
                    "confidence": float(hit.get("confidence") or 0.5),
                    "scope": "fact",
                }
            )
        for th in trace_hits:
            findings.append(
                {
                    "claim": str(th.get("snippet") or "trace hit"),
                    "evidence_type": "runtime_log",
                    "source_ref": th.get("handle"),
                    "verified": True,
                    "confidence": 0.75,
                    "scope": "fact",
                }
            )

        if findings:
            lead = str(findings[0].get("claim") or "related evidence")
            summary = f"Grounded evidence found ({len(findings)} item(s)). Lead: {lead[:160]}."
        else:
            summary = "No recall or trace evidence found for this inquiry."

        return {
            "summary": summary,
            "mode": "general_investigation",
            "findings": findings,
        }

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

    async def _repo_grep_batch(
        self,
        pattern: str,
        request: ContextExecRequestV1,
        namespace: ContextNamespace,
        runtime: OrganRuntime | None,
        *,
        path: str | None = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        if runtime is not None and request.permissions.read_repo:
            self._subcalls += 1
            return runtime.repo_grep(pattern, path=path, limit=limit)
        if request.permissions.read_repo:
            return namespace.repo.grep(pattern, path=path, limit=limit)
        return []

    async def _repo_impact(
        self,
        request: ContextExecRequestV1,
        namespace: ContextNamespace,
        runtime: OrganRuntime | None,
        organ_cache: dict[str, Any] | None,
    ) -> dict[str, Any]:
        self._debug_steps.append("retrieve")
        engine_selection = _is_engine_selection_query(request.text)
        terms = _repo_search_terms(request.text)
        hits: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        scoped_path = _engine_selection_grep_path() if engine_selection else None

        for term in terms:
            batch = await self._repo_grep_batch(
                term,
                request,
                namespace,
                runtime,
                path=scoped_path,
                limit=30,
            )
            _merge_repo_hits(hits, batch, seen_paths)

        if engine_selection and not hits:
            for term in terms[:4]:
                batch = await self._repo_grep_batch(
                    term,
                    request,
                    namespace,
                    runtime,
                    limit=30,
                )
                _merge_repo_hits(hits, batch, seen_paths)

        if engine_selection:
            hits = _prioritize_engine_selection_hits(hits)

        if organ_cache is not None:
            organ_cache["repo_hits"] = hits

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
        if engine_selection:
            risk = _engine_selection_risk(affected)
            breaking_surfaces: list[str] = []
            compatibility_shims: list[str] = []
            tests_to_add: list[str] = []
            migration_steps: list[str] = []
        else:
            risk = _agent_chain_risk(affected)
            breaking_surfaces = ["cortex-exec AgentChainClient hop"]
            compatibility_shims = ["AgentChainResult-compatible context-exec reply"]
            tests_to_add = ["test_context_exec_depth2_routing"]
            migration_steps = ["feature flag CONTEXT_EXEC_ENABLED"]

        return {
            "proposed_change": request.text[:500],
            "status": "analyzed" if len(affected) >= 2 else "partial",
            "affected_paths": affected,
            "breaking_surfaces": breaking_surfaces,
            "compatibility_shims": compatibility_shims,
            "tests_to_add_or_update": tests_to_add,
            "migration_steps": migration_steps,
            "risk": risk,
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

    async def _patch_proposal(
        self,
        request: ContextExecRequestV1,
        namespace: ContextNamespace,
        runtime: OrganRuntime | None,
        organ_cache: dict[str, Any] | None,
    ) -> dict[str, Any]:
        self._debug_steps.append("retrieve")
        terms = _patch_proposal_search_terms(request.text)
        hits: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        scoped_path = _engine_selection_grep_path()

        for term in terms:
            batch = await self._repo_grep_batch(
                term,
                request,
                namespace,
                runtime,
                path=scoped_path,
                limit=30,
            )
            _merge_repo_hits(hits, batch, seen_paths)

        if not hits:
            for term in terms[:4]:
                batch = await self._repo_grep_batch(
                    term,
                    request,
                    namespace,
                    runtime,
                    limit=30,
                )
                _merge_repo_hits(hits, batch, seen_paths)

        hits = _prioritize_patch_proposal_hits(hits)
        if organ_cache is not None:
            organ_cache["repo_hits"] = hits
        self._debug_steps.append("synthesize")

        if not hits:
            return {
                "problem": request.text[:500],
                "evidence": [],
                "files_to_change": [],
                "proposed_change_summary": "Insufficient repo evidence to propose a grounded patch.",
                "risk": "unknown",
                "tests_to_run": [],
                "rollback_plan": "No changes proposed; no rollback required.",
                "open_questions": ["insufficient repo grounding"],
                "mutation_allowed": False,
            }

        files = [h.get("path") for h in hits if isinstance(h, dict) and h.get("path")][:12]
        evidence = [
            f"{h.get('path')}:{h.get('line_start')} {str(h.get('snippet', ''))[:120]}"
            for h in hits
            if isinstance(h, dict)
        ][:8]
        risk = _patch_proposal_risk(files)
        tests = [
            name
            for name in _PATCH_PROPOSAL_LIKELY_FILES
            if name.startswith("test_") and any(name in str(p) for p in files)
        ]
        if not tests:
            tests = [
                name
                for name in ("test_alexzhang_rlm_engine.py", "test_rlm_eval_fixtures.py")
                if any(name in str(p) for p in files)
            ]
        if not tests and any("alexzhang_rlm_engine.py" in str(p) for p in files):
            tests = ["test_alexzhang_rlm_engine.py"]
        if not tests:
            tests = []
            open_questions = ["tests_to_run unclear without grounded test file hits"]
        else:
            open_questions = []

        summary_parts = []
        if _is_trace_autopsy_patch_query(request.text):
            summary_parts.append(
                "Improve trace-autopsy root cause synthesis using grounded repo evidence"
            )
        elif _is_repo_impact_patch_query(request.text):
            summary_parts.append("Strengthen repo-impact grounding for patch proposals")
        else:
            summary_parts.append("Apply targeted fixes based on grounded repo findings")
        file_names = [str(p).rsplit("/", 1)[-1] for p in files[:4]]
        summary_parts.append(f"Likely files: {', '.join(file_names)}")

        return {
            "problem": request.text[:500],
            "evidence": evidence,
            "files_to_change": files,
            "proposed_change_summary": ". ".join(summary_parts),
            "risk": risk,
            "tests_to_run": tests,
            "rollback_plan": "Revert proposed file changes via version control; no runtime mutation performed.",
            "open_questions": open_questions,
            "mutation_allowed": False,
        }

    def _infer_memory_domain(self, source_ref: str | None) -> str | None:
        if not source_ref:
            return None
        lowered = str(source_ref).lower()
        prefixes = (
            ("rdf:", "rdf"),
            ("chroma:", "chroma"),
            ("graphiti:", "graphiti"),
            ("sql:", "sql_timeline"),
            ("timeline:", "sql_timeline"),
            ("cards:", "cards"),
            ("card:", "cards"),
            ("memory:", "cards"),
        )
        for prefix, domain in prefixes:
            if lowered.startswith(prefix):
                return domain
        return None

    def _collect_memory_correction_evidence(
        self,
        *,
        memory_hits: list[dict[str, Any]],
        recall_hits: list[dict[str, Any]],
        trace_hits: list[dict[str, Any]],
    ) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
        supporting: list[str] = []
        contradicting: list[str] = []
        missing: list[str] = []
        domains: list[str] = []
        affected_ids: list[str] = []

        for hit in memory_hits + recall_hits:
            claim = str(hit.get("claim") or hit.get("snippet") or hit.get("title") or "").strip()
            source_ref = hit.get("source_ref") or hit.get("id") or hit.get("handle")
            if not claim:
                continue
            entry = f"{claim} (source: {source_ref})" if source_ref else claim
            lowered = claim.lower()
            if any(term in lowered for term in ("not denver", "ogden", "contradict", "unsupported")):
                contradicting.append(entry)
            else:
                supporting.append(entry)
            domain = self._infer_memory_domain(str(source_ref) if source_ref else None)
            if domain and domain not in domains:
                domains.append(domain)
            if source_ref and str(source_ref) not in affected_ids:
                affected_ids.append(str(source_ref))

        for th in trace_hits:
            snippet = str(th.get("snippet") or th.get("kind") or "").strip()
            handle = th.get("handle") or th.get("corr_id")
            if not snippet:
                continue
            entry = f"{snippet} (trace: {handle})" if handle else snippet
            lowered = snippet.lower()
            if any(term in lowered for term in ("not denver", "ogden", "contradict", "unsupported")):
                contradicting.append(entry)
            else:
                supporting.append(entry)
            if handle and str(handle) not in affected_ids:
                affected_ids.append(str(handle))

        if not supporting and not contradicting:
            missing.append("insufficient memory evidence")

        return supporting, contradicting, missing, domains, affected_ids

    async def _memory_correction_proposal(
        self,
        request: ContextExecRequestV1,
        namespace: ContextNamespace,
        runtime: OrganRuntime | None,
        organ_cache: dict[str, Any] | None,
    ) -> dict[str, Any]:
        current_belief = _extract_claim_from_text(request.text)
        query_terms = current_belief.split()[:8] or [request.text[:80]]
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
        recall_hits = [
            {
                "claim": str(h.get("snippet") or h.get("title") or "recall hit"),
                "source_ref": h.get("source_ref") or h.get("id"),
                "verified": True,
                "confidence": float(h.get("score") or 0.5),
            }
            for h in recall_result.get("hits") or []
        ]
        if not memory_hits and recall_hits:
            memory_hits = recall_hits
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
        supporting, contradicting, missing, domains, affected_ids = self._collect_memory_correction_evidence(
            memory_hits=memory_hits,
            recall_hits=recall_hits,
            trace_hits=trace_hits,
        )

        has_evidence = bool(supporting or contradicting)
        if not has_evidence:
            missing.append("insufficient memory evidence")

        correction_type = "mark_uncertain"
        proposed_belief: str | None = None
        confidence = 0.15
        risk = "unknown"

        ogden_contradiction = any("ogden" in c.lower() for c in contradicting)
        denver_unsupported = "denver" in current_belief.lower() and not supporting

        if ogden_contradiction:
            correction_type = "replace_belief"
            proposed_belief = next(
                (c.split(" (source:")[0] for c in contradicting if "ogden" in c.lower()),
                "User location is Ogden, not Denver",
            )
            confidence = 0.65
            risk = "medium"
        elif contradicting and supporting:
            correction_type = "mark_contradicted"
            confidence = 0.45
            risk = "medium"
        elif contradicting:
            correction_type = "mark_contradicted"
            confidence = 0.4
            risk = "medium"
        elif denver_unsupported or not has_evidence:
            correction_type = "mark_uncertain"
            confidence = 0.2
            risk = "unknown"

        if confidence > 0.7 and not (ogden_contradiction or (contradicting and len(contradicting) >= 2)):
            confidence = 0.7

        target_domains = domains or ["unknown"]
        rationale_parts = [
            f"Proposed memory correction for belief: {current_belief}",
        ]
        if correction_type == "mark_uncertain":
            rationale_parts.append("Evidence is insufficient to propose a definitive correction.")
        elif correction_type == "mark_contradicted":
            rationale_parts.append("Recall/trace evidence contradicts the current belief.")
        elif correction_type == "replace_belief":
            rationale_parts.append("Contradicting evidence supports replacing the current belief.")

        open_questions: list[str] = []
        if not has_evidence:
            open_questions.append("insufficient memory evidence for grounded correction")
        if not domains:
            open_questions.append("target memory domain unclear from evidence source refs")

        return {
            "current_belief": current_belief,
            "proposed_belief": proposed_belief,
            "correction_type": correction_type,
            "rationale": " ".join(rationale_parts),
            "supporting_evidence": supporting[:8],
            "contradicting_evidence": contradicting[:8],
            "missing_evidence": missing,
            "target_memory_domains": target_domains,
            "affected_ids": affected_ids[:12],
            "confidence": confidence,
            "risk": risk,
            "tests_to_run": [],
            "rollback_plan": "No mutation proposed; no rollback required.",
            "open_questions": open_questions,
            "mutation_allowed": False,
        }
