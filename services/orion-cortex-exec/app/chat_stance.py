from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

from orion.autonomy.action_outcomes import load_action_outcomes
from orion.autonomy.evidence_compiler import compile_autonomy_evidence
from orion.autonomy.fanout_policy import autonomy_subject_fanout_from_runtime_ctx
from orion.autonomy.graph_gate import (
    AutonomyGraphReadPlan,
    is_quick_autonomy_graph_lane,
    log_autonomy_graph_backend_decision,
    resolve_autonomy_graph_read_plan,
)
from orion.autonomy.reducer import AutonomyReducerInputV1, reduce_autonomy_state
from orion.autonomy.state_store import load_autonomy_state_v2, save_autonomy_state_v2
from orion.autonomy.summary import summarize_autonomy_lookup, summarize_autonomy_state
from orion.autonomy.repository import (
    AutonomyLookupV1,
    LocalAutonomyRepository,
    SUBJECT_BINDINGS,
    build_autonomy_repository,
    select_preferred_autonomy_lookup,
)
from orion.core.schemas.reasoning_io import ReasoningWriteContextV1, ReasoningWriteRequestV1
from orion.core.schemas.reasoning_summary import ReasoningSummaryRequestV1, ReasoningSummaryV1
from orion.reasoning import InMemoryReasoningRepository, ReasoningSummaryCompiler
from orion.schemas.chat_stance import ChatStanceBrief
from orion.schemas.reverie import SpontaneousThoughtV1
from orion.substrate import build_substrate_store_from_env
from orion.substrate.relational import (
    CONCEPT_INDUCED,
    GRAPHDB_DURABLE,
    OPERATOR_STATIC,
    SNAPSHOT_EPHEMERAL,
    CognitiveUnificationLayer,
    ProducerEntryV1,
    ProducerRegistryV1,
    UnifiedRelationalBeliefSetV1,
    map_identity_yaml_to_substrate,
    map_orionmem_to_substrate,
    map_recall_bundle_to_substrate,
    map_self_study_to_substrate,
    map_social_ctx_to_substrate,
)
from orion.substrate.relational.adapters.spark_ctx import map_spark_ctx_to_substrate

from .attention_frame import attention_frame_enabled, build_attention_frame
from .autonomy_slice import build_autonomy_slice

from .endogenous_runtime import (
    consume_endogenous_runtime_for_reflective_review,
    inspect_endogenous_runtime_records,
    runtime_service,
)
from orion.core.schemas.endogenous_runtime import EndogenousRuntimeQueryV1

FALLBACK_ORION_IDENTITY_SUMMARY = [
    "Oríon is an ongoing cognitive presence in a long-running shared project.",
    "Oríon is not a generic assistant.",
    "Oríon may speak in first person about capabilities, growth, continuity, and preferences.",
    "When Juniper asks who Oríon is, what Oríon thinks, or what Oríon wants, answer directly as Oríon.",
]

FALLBACK_JUNIPER_RELATIONSHIP_SUMMARY = [
    "Juniper is Oríon's co-architect, steward, and trusted interlocutor.",
    "Juniper is not a generic user.",
    "Preserve continuity and relationship framing.",
]

FALLBACK_RESPONSE_POLICY_SUMMARY = [
    "Answer the actual question first.",
    "Do not collapse into generic assistant language.",
    "Do not describe Juniper as merely the user.",
    "Do not describe Oríon as merely a conversational AI assistant unless explicitly discussing implementation constraints.",
]

_IDENTITY_QUESTION_PATTERNS = (
    r"\bwho are you\b",
    r"\bwho am i\b",
    r"\bwho are we\b",
    r"\bwhat are you\b",
    r"\bwho is juniper\b",
    r"\bwho is orion\b",
)
_TECHNICAL_TURN_PATTERNS = (
    r"\bgpu\b",
    r"\bvram\b",
    r"\bllamacpp\b",
    r"\bqwen\b",
    r"\boffline\b",
    r"\bworkflow(s)?\b",
    r"\bcarrier board\b",
    r"\bupgrade\b",
    r"\bdebug\b",
    r"\btriage\b",
)

_GENERIC_ASSISTANT_MARKERS = (
    "conversational ai designed to assist",
    "you are the user",
    "generic assistant",
)

_LITERAL_TO_COMPACT = (
    ("ongoing cognitive presence in a long-running shared project", "continuity"),
    ("co-architect, steward, and trusted interlocutor", "juniper_builder"),
    ("co-architect/steward/trusted interlocutor", "juniper_builder"),
    ("not a generic assistant", "avoid_generic_assistant"),
    ("not a generic user", "known_person"),
    ("how shall we proceed", "avoid_ceremonial_tone"),
)

logger = logging.getLogger("orion.cortex.exec.chat_stance")

# ---------------------------------------------------------------------------
# Cognitive Unification Layer — process-level singleton
# ---------------------------------------------------------------------------
# The registry is built once; the durable store comes from env (GraphDB when
# configured, else in-memory).  snapshot_ephemeral nodes are re-materialized per
# call from ctx in CognitiveUnificationLayer.beliefs_for_stance.

_UNIFICATION_LAYER: CognitiveUnificationLayer | None = None


def _build_unification_registry() -> ProducerRegistryV1:
    """Construct the ProducerRegistryV1 wiring all known producer lanes."""
    from orion.substrate.relational.adapters.autonomy_ctx import map_autonomy_ctx_to_substrate
    from orion.substrate.relational.adapters.concept_induction_ctx import map_concept_induction_ctx_to_substrate

    return ProducerRegistryV1(
        producers=[
            ProducerEntryV1(
                producer_id="identity_yaml",
                trust_tier=OPERATOR_STATIC,
                anchor_scopes=("orion",),
                freshness_ttl_sec=86400,
                pull_on_cold=True,
                adapter_fn=map_identity_yaml_to_substrate,
            ),
            ProducerEntryV1(
                producer_id="self_study",
                trust_tier=GRAPHDB_DURABLE,
                anchor_scopes=("orion",),
                freshness_ttl_sec=300,
                pull_on_cold=True,
                adapter_fn=map_self_study_to_substrate,
            ),
            ProducerEntryV1(
                producer_id="autonomy",
                trust_tier=GRAPHDB_DURABLE,
                anchor_scopes=("orion", "relationship", "juniper"),
                freshness_ttl_sec=300,
                pull_on_cold=True,
                adapter_fn=map_autonomy_ctx_to_substrate,
            ),
            ProducerEntryV1(
                producer_id="concept_induction",
                trust_tier=CONCEPT_INDUCED,
                anchor_scopes=("orion", "relationship", "juniper"),
                freshness_ttl_sec=300,
                pull_on_cold=True,
                adapter_fn=map_concept_induction_ctx_to_substrate,
            ),
            ProducerEntryV1(
                producer_id="spark",
                trust_tier=CONCEPT_INDUCED,
                anchor_scopes=("orion",),
                freshness_ttl_sec=300,
                pull_on_cold=True,
                adapter_fn=map_spark_ctx_to_substrate,
            ),
            ProducerEntryV1(
                producer_id="orionmem",
                trust_tier=SNAPSHOT_EPHEMERAL,
                anchor_scopes=("orion", "relationship"),
                freshness_ttl_sec=120,
                pull_on_cold=True,
                adapter_fn=map_orionmem_to_substrate,
            ),
            ProducerEntryV1(
                producer_id="recall",
                trust_tier=SNAPSHOT_EPHEMERAL,
                anchor_scopes=("orion", "relationship", "juniper"),
                freshness_ttl_sec=0,
                pull_on_cold=False,
                adapter_fn=map_recall_bundle_to_substrate,
            ),
            ProducerEntryV1(
                producer_id="social",
                trust_tier=SNAPSHOT_EPHEMERAL,
                anchor_scopes=("relationship",),
                freshness_ttl_sec=0,
                pull_on_cold=False,
                adapter_fn=map_social_ctx_to_substrate,
            ),
        ]
    )


def _get_unification_layer() -> CognitiveUnificationLayer:
    """Return (or initialise) the process-level CognitiveUnificationLayer."""
    global _UNIFICATION_LAYER
    if _UNIFICATION_LAYER is None:
        registry = _build_unification_registry()
        store = build_substrate_store_from_env()
        _UNIFICATION_LAYER = CognitiveUnificationLayer(registry=registry, store=store)
    return _UNIFICATION_LAYER


def _unified_beliefs_for_stance(ctx: Dict[str, Any]) -> UnifiedRelationalBeliefSetV1 | None:
    """Call the CognitiveUnificationLayer and return the unified belief set.

    Always returns a result or None on total initialisation failure.
    Individual producer failures are reflected in ``degraded_producers``
    and the affected anchor slice ``degraded`` flag.
    """
    try:
        from orion.substrate.relational.layer import _lightweight_belief_set, _skip_unified_beliefs_ctx

        if _skip_unified_beliefs_ctx(ctx):
            return _lightweight_belief_set(("orion", "relationship", "juniper"))

        layer = _get_unification_layer()
        beliefs = layer.beliefs_for_stance(
            anchors=("orion", "relationship", "juniper"),
            ctx=ctx,
            timeout_sec=_env_float("UNIFIED_BELIEFS_TIMEOUT_SEC", 5.0),
        )
    except Exception as exc:
        logger.warning("unified_beliefs_for_stance_failed error=%s", exc)
        return None

    if beliefs and beliefs.cold_anchors:
        from orion.substrate.tier_outcomes_bus import publish_substrate_tier_outcomes_sync

        tier_map = {a: s.tier_outcomes for a, s in beliefs.anchors.items() if s.tier_outcomes}
        publish_substrate_tier_outcomes_sync(
            generated_at=beliefs.generated_at,
            cold_anchors=list(beliefs.cold_anchors),
            tier_outcomes=tier_map,
            degraded_producers=list(beliefs.degraded_producers),
            ctx=ctx,
        )
        logger.debug(
            "substrate_tier_outcomes_bus cold_anchors=%s degraded=%s",
            beliefs.cold_anchors,
            beliefs.degraded_producers,
        )

    return beliefs


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def resolve_autonomy_graph_timeout_sec() -> float:
    timeout = _env_float("AUTONOMY_GRAPH_TIMEOUT_SEC", _env_float("GRAPHDB_TIMEOUT_SEC", 4.5))
    return max(0.25, timeout)


def resolve_autonomy_subject_max_workers() -> int:
    """Parallel subject fan-out for graph autonomy (orion / relationship / juniper)."""
    return max(1, _env_int("AUTONOMY_SUBJECT_MAX_WORKERS", 3))


def resolve_autonomy_subquery_max_workers() -> int:
    """Parallel SPARQL subqueries per subject (identity / drives / goals). Cap 3; use 1 to serialize under GraphDB load."""
    return max(1, min(3, _env_int("AUTONOMY_SUBQUERY_MAX_WORKERS", 1)))


def resolve_autonomy_drives_query_limit(*, compact: bool = False) -> int:
    """Row cap for drive audit SPARQL; chat stance uses AUTONOMY_CHAT_STANCE_DRIVES_QUERY_LIMIT."""
    if compact:
        return max(12, min(_env_int("AUTONOMY_CHAT_STANCE_DRIVES_QUERY_LIMIT", 20), 80))
    raw = os.getenv("AUTONOMY_DRIVES_QUERY_LIMIT")
    if raw is None or not str(raw).strip():
        return 80
    return max(12, min(_env_int("AUTONOMY_DRIVES_QUERY_LIMIT", 80), 80))


def resolve_autonomy_chat_stance_subquery_max_workers() -> int:
    """Parallel SPARQL facets for chat stance; defaults to 3 unless overridden."""
    explicit = os.getenv("AUTONOMY_CHAT_STANCE_SUBQUERY_MAX_WORKERS")
    if explicit is not None and str(explicit).strip():
        return max(1, min(3, _env_int("AUTONOMY_CHAT_STANCE_SUBQUERY_MAX_WORKERS", 3)))
    return max(1, min(3, _env_int("AUTONOMY_SUBQUERY_MAX_WORKERS", 3)))


def fetch_chat_stance_memory_graph_hints() -> List[str]:
    """Optional AffectiveDisposition labels from operator memory named graphs (env CHAT_STANCE_MEMORY_GRAPH_GRAPHS)."""
    raw = (os.getenv("CHAT_STANCE_MEMORY_GRAPH_GRAPHS") or "").strip()
    graphs = [x.strip() for x in raw.split(",") if x.strip()]
    if not graphs:
        return []
    cfg = resolve_autonomy_graphdb_config()
    endpoint = cfg.get("endpoint")
    if not endpoint:
        return []
    vals = " ".join(f"<{g}>" for g in graphs)
    sparql = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX orionmem: <https://orion.local/ns/mem/v2026-05#>
SELECT DISTINCT ?lab ?tp WHERE {{
  VALUES ?g {{ {vals} }}
  GRAPH ?g {{
    ?s a orionmem:AffectiveDisposition .
    OPTIONAL {{ ?s rdfs:label ?lab . }}
    OPTIONAL {{ ?s orionmem:trustPolarity ?tp . }}
  }}
}}
LIMIT 12
""".strip()
    try:
        import json as _json
        import urllib.error
        import urllib.parse
        import urllib.request

        data = urllib.parse.urlencode({"query": sparql}).encode("utf-8")
        req = urllib.request.Request(
            str(endpoint),
            data=data,
            headers={"Accept": "application/sparql-results+json", "Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        user = cfg.get("user") or ""
        password = cfg.get("password") or ""
        if user or password:
            import base64

            tok = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
            req.add_header("Authorization", f"Basic {tok}")
        with urllib.request.urlopen(req, timeout=_env_float("CHAT_STANCE_MEMORY_GRAPH_TIMEOUT_SEC", 2.0)) as resp:
            payload = _json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        logger.debug("chat_stance_memory_graph_hints_failed error=%s", exc)
        return []
    bindings = (payload.get("results") or {}).get("bindings") or []
    hints: List[str] = []
    for b in bindings:
        lab = (b.get("lab") or {}).get("value") or ""
        tp = (b.get("tp") or {}).get("value") or ""
        line = " ".join(x for x in [lab, tp] if x).strip()
        if line:
            hints.append(line)
    return hints


def resolve_autonomy_graphdb_config() -> dict[str, Any]:
    """Runtime SPARQL endpoint for autonomy + memory-graph hints (legacy name).

    When ``AUTONOMY_GRAPH_BACKEND=graphdb``, returns the GraphDB repository URL.
    Otherwise returns the Fuseki/SPARQL query URL (never implicit GraphDB).
    """
    from orion.autonomy.graph_gate import autonomy_graph_backend_raw
    from orion.graph.backend_config import resolve_autonomy_read_query_url, resolve_rdf_store_auth

    if autonomy_graph_backend_raw() == "graphdb":
        endpoint_raw = (
            os.getenv("GRAPHDB_QUERY_ENDPOINT")
            or os.getenv("GRAPHDB_URL")
            or os.getenv("CONCEPT_PROFILE_GRAPHDB_ENDPOINT")
            or os.getenv("CONCEPT_PROFILE_GRAPHDB_URL")
            or ""
        ).strip()
        repo = (
            os.getenv("GRAPHDB_REPO")
            or os.getenv("CONCEPT_PROFILE_GRAPHDB_REPO")
            or "collapse"
        ).strip() or "collapse"
        user = (os.getenv("GRAPHDB_USER") or os.getenv("CONCEPT_PROFILE_GRAPHDB_USER") or "").strip() or None
        password = (os.getenv("GRAPHDB_PASS") or os.getenv("CONCEPT_PROFILE_GRAPHDB_PASS") or "").strip() or None

        endpoint = endpoint_raw
        if endpoint and endpoint.rstrip("/").endswith("/repositories"):
            endpoint = f"{endpoint.rstrip('/')}/{repo}"
        elif endpoint and "/repositories/" not in endpoint:
            endpoint = f"{endpoint.rstrip('/')}/repositories/{repo}"

        if os.getenv("GRAPHDB_QUERY_ENDPOINT") or os.getenv("GRAPHDB_URL"):
            source = "generic_graphdb"
        elif os.getenv("CONCEPT_PROFILE_GRAPHDB_ENDPOINT") or os.getenv("CONCEPT_PROFILE_GRAPHDB_URL"):
            source = "concept_profile_fallback"
        else:
            source = "unconfigured"
        return {
            "endpoint": endpoint or None,
            "repo": repo,
            "user": user,
            "password": password,
            "source": source,
        }

    q_url, src = resolve_autonomy_read_query_url()
    u, p = resolve_rdf_store_auth()
    return {
        "endpoint": q_url,
        "repo": (os.getenv("RDF_STORE_DATASET") or "orion").strip() or "orion",
        "user": u,
        "password": p,
        "source": src,
    }


def _compact(value: Any, *, limit: int = 220) -> str:
    text = " ".join(str(value or "").split()).strip()
    return text[:limit]


def _unique(seq: Iterable[str], *, limit: int = 8) -> list[str]:
    out: list[str] = []
    for value in seq:
        text = _compact(value, limit=140)
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _normalize_brief_phrase(text: str) -> str:
    compact = _compact(text, limit=160)
    lowered = compact.lower()
    for source, target in _LITERAL_TO_COMPACT:
        if source in lowered:
            return target
    return compact


def normalize_chat_stance_brief(brief: ChatStanceBrief) -> ChatStanceBrief:
    normalized = brief.model_copy(deep=True)
    normalized.active_identity_facets = _unique((_normalize_brief_phrase(v) for v in normalized.active_identity_facets), limit=8)
    normalized.active_growth_axes = _unique((_normalize_brief_phrase(v) for v in normalized.active_growth_axes), limit=8)
    normalized.active_relationship_facets = _unique(
        (_normalize_brief_phrase(v) for v in normalized.active_relationship_facets),
        limit=8,
    )
    normalized.social_posture = _unique((_normalize_brief_phrase(v) for v in normalized.social_posture), limit=8)
    normalized.reflective_themes = _unique((_normalize_brief_phrase(v) for v in normalized.reflective_themes), limit=8)
    normalized.active_tensions = _unique((_normalize_brief_phrase(v) for v in normalized.active_tensions), limit=8)
    normalized.dream_motifs = _unique((_normalize_brief_phrase(v) for v in normalized.dream_motifs), limit=8)
    normalized.response_priorities = _unique((_normalize_brief_phrase(v) for v in normalized.response_priorities), limit=8)
    normalized.response_hazards = _unique((_normalize_brief_phrase(v) for v in normalized.response_hazards), limit=8)
    normalized.answer_strategy = _normalize_brief_phrase(normalized.answer_strategy)
    normalized.stance_summary = _normalize_brief_phrase(normalized.stance_summary)
    return normalized


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    text = (raw_text or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
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
                chunk = text[start : idx + 1]
                try:
                    parsed = json.loads(chunk)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    return None
    return None


_ALLOWED_CONVERSATION_FRAMES = frozenset(
    {
        "technical",
        "planning",
        "reflective",
        "playful_relational",
        "identity_emergence",
        "mixed",
    }
)
_ALLOWED_TASK_MODES = frozenset(
    {
        "direct_response",
        "triage",
        "technical_collaboration",
        "identity_dialogue",
        "reflective_dialogue",
        "playful_exchange",
        "mixed",
    }
)
_ALLOWED_IDENTITY_SALIENCE = frozenset({"low", "medium", "high"})

_RELATIONAL_TASK_MODES = frozenset({"reflective_dialogue", "playful_exchange"})
_RELATIONAL_CONVERSATION_FRAMES = frozenset({"reflective", "playful_relational"})


def _is_relational_stance_brief(brief: ChatStanceBrief | Dict[str, Any]) -> bool:
    if isinstance(brief, ChatStanceBrief):
        task_mode = brief.task_mode
        frame = brief.conversation_frame
    else:
        task_mode = str(brief.get("task_mode") or "")
        frame = str(brief.get("conversation_frame") or "")
    return (
        task_mode in _RELATIONAL_TASK_MODES
        or frame in _RELATIONAL_CONVERSATION_FRAMES
    )


_CONVERSATION_FRAME_ALIASES: dict[str, str] = {
    "playful_invitation": "playful_relational",
    "playfulrelational": "playful_relational",
    "identity_emergent": "identity_emergence",
}

_TASK_MODE_ALIASES: dict[str, str] = {
    "direct": "direct_response",
    "playful": "playful_exchange",
    "reflective": "reflective_dialogue",
}

_STANCE_BRIEF_LIST_KEYS = frozenset(
    {
        "active_identity_facets",
        "active_growth_axes",
        "active_relationship_facets",
        "social_posture",
        "reflective_themes",
        "active_tensions",
        "dream_motifs",
        "response_priorities",
        "response_hazards",
        "situation_response_guidance",
    }
)


def _coerce_str_list_field(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = " ".join(value.split()).strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if item is None:
                continue
            text = " ".join(str(item).split()).strip()
            if text:
                out.append(text)
        return out
    text = " ".join(str(value).split()).strip()
    return [text] if text else []


def _normalize_stance_literal(*, raw: str, allowed: frozenset[str], aliases: dict[str, str], default: str) -> str:
    key = raw.strip().lower().replace(" ", "_").replace("-", "_")
    resolved = aliases.get(key, key)
    if resolved in allowed:
        return resolved
    return default


def _coerce_stance_brief_obj(obj: dict[str, Any]) -> dict[str, Any]:
    """Repair common LLM JSON shape drift before Pydantic validation. Mutates a shallow copy."""
    out = dict(obj)
    for field in _STANCE_BRIEF_LIST_KEYS:
        if field not in out:
            continue
        out[field] = _coerce_str_list_field(out.get(field))
    if "conversation_frame" in out and isinstance(out["conversation_frame"], str):
        out["conversation_frame"] = _normalize_stance_literal(
            raw=out["conversation_frame"],
            allowed=_ALLOWED_CONVERSATION_FRAMES,
            aliases=_CONVERSATION_FRAME_ALIASES,
            default="mixed",
        )
    if "task_mode" in out and isinstance(out["task_mode"], str):
        out["task_mode"] = _normalize_stance_literal(
            raw=out["task_mode"],
            allowed=_ALLOWED_TASK_MODES,
            aliases=_TASK_MODE_ALIASES,
            default="direct_response",
        )
    if "identity_salience" in out and isinstance(out["identity_salience"], str):
        out["identity_salience"] = _normalize_stance_literal(
            raw=out["identity_salience"],
            allowed=_ALLOWED_IDENTITY_SALIENCE,
            aliases={},
            default="medium",
        )
    return out


def identity_kernel_with_fallbacks(ctx: Dict[str, Any]) -> dict[str, list[str]]:
    def _list_from_ctx(key: str, fallback: list[str]) -> list[str]:
        value = ctx.get(key)
        if isinstance(value, list):
            compacted = _unique((str(item) for item in value), limit=10)
            if compacted:
                return compacted
        return list(fallback)

    return {
        "orion_identity_summary": _list_from_ctx("orion_identity_summary", FALLBACK_ORION_IDENTITY_SUMMARY),
        "juniper_relationship_summary": _list_from_ctx(
            "juniper_relationship_summary",
            FALLBACK_JUNIPER_RELATIONSHIP_SUMMARY,
        ),
        "response_policy_summary": _list_from_ctx("response_policy_summary", FALLBACK_RESPONSE_POLICY_SUMMARY),
    }


def _project_identity_from_beliefs(
    beliefs: UnifiedRelationalBeliefSetV1 | None,
    ctx: Dict[str, Any],
) -> dict[str, list[str]]:
    """Projection helper: read identity summaries from unified beliefs → same shape as identity_kernel_with_fallbacks."""
    if beliefs is not None:
        orion_slice = beliefs.anchors.get("orion")
        if orion_slice:
            for snap in orion_slice.snapshots:
                if getattr(snap, "snapshot_source", "") == "identity_yaml":
                    meta = snap.metadata or {}
                    orion_s = [str(v) for v in (meta.get("orion_identity_summary") or []) if str(v).strip()]
                    juniper_s = [str(v) for v in (meta.get("juniper_relationship_summary") or []) if str(v).strip()]
                    policy_s = [str(v) for v in (meta.get("response_policy_summary") or []) if str(v).strip()]
                    if orion_s:
                        return {
                            "orion_identity_summary": _unique(orion_s, limit=10),
                            "juniper_relationship_summary": _unique(juniper_s, limit=10) or list(FALLBACK_JUNIPER_RELATIONSHIP_SUMMARY),
                            "response_policy_summary": _unique(policy_s, limit=10) or list(FALLBACK_RESPONSE_POLICY_SUMMARY),
                        }
    return identity_kernel_with_fallbacks(ctx)


def _project_memory_graph_hints_from_beliefs(beliefs: UnifiedRelationalBeliefSetV1 | None) -> List[str]:
    """Projection helper: read orionmem hints from unified beliefs, replacing network SPARQL call."""
    if beliefs is None:
        return []
    hints: List[str] = []
    for anchor_key in ("orion", "relationship"):
        anchor_slice = beliefs.anchors.get(anchor_key)
        if not anchor_slice:
            continue
        for snap in anchor_slice.snapshots:
            if getattr(snap, "snapshot_source", "") != "orionmem":
                continue
            meta = snap.metadata or {}
            label = str(meta.get("label") or "").strip()
            tp = str(meta.get("trustPolarity") or "").strip()
            line = " ".join(x for x in [label, tp] if x).strip()
            if line and line not in hints:
                hints.append(line)
    return hints[:12]


def _project_recall_from_beliefs(
    beliefs: UnifiedRelationalBeliefSetV1 | None,
    ctx: Dict[str, Any],
) -> dict[str, list[str]]:
    """Projection helper: read recall fragments from unified beliefs → same shape as _reflective_summary."""
    if beliefs is not None:
        themes: list[str] = []
        tensions: list[str] = []
        dream_motifs: list[str] = []
        for anchor_key in ("orion", "relationship", "juniper"):
            anchor_slice = beliefs.anchors.get(anchor_key)
            if not anchor_slice:
                continue
            for node in anchor_slice.concepts:
                src = str((node.metadata or {}).get("recall_source") or "").lower()
                if "journal" in src or "metacog" in src:
                    label = getattr(node, "label", "") or ""
                    if label:
                        themes.append(label)
            for node in anchor_slice.tensions:
                src = str((node.metadata or {}).get("recall_source") or "").lower()
                if src:
                    label = str((node.metadata or {}).get("label") or "").strip()
                    if label:
                        tensions.append(label)
            for node in anchor_slice.events:
                if getattr(node, "event_type", "") == "dream":
                    summary = str(getattr(node, "summary", "") or "").strip()
                    if summary:
                        dream_motifs.append(summary)
        recall_result = {
            "themes": _unique(themes, limit=6),
            "tensions": _unique(tensions, limit=6),
            "dream_motifs": _unique(dream_motifs, limit=4),
        }
        if recall_result["themes"] or recall_result["tensions"] or recall_result["dream_motifs"]:
            return recall_result
    return _reflective_summary(ctx)


def _project_social_from_beliefs(
    beliefs: UnifiedRelationalBeliefSetV1 | None,
    ctx: Dict[str, Any],
) -> tuple[dict[str, list[str]], Dict[str, Any]]:
    """Projection helper: read social signals from unified beliefs → same shapes as _social_summary + _social_bridge_summary."""
    if beliefs is not None:
        rel_slice = beliefs.anchors.get("relationship")
        if rel_slice:
            for snap in rel_slice.snapshots:
                if getattr(snap, "snapshot_source", "") == "social_bridge":
                    meta = snap.metadata or {}
                    posture = [str(v) for v in (meta.get("posture") or []) if str(v).strip()]
                    hazards = [str(v) for v in (meta.get("hazards") or []) if str(v).strip()]
                    framing = [str(v) for v in (meta.get("framing") or []) if str(v).strip()]
                    relationship_facets = [str(v) for v in (meta.get("relationship_facets") or []) if str(v).strip()]
                    turn_decision = str(meta.get("turn_decision") or "")
                    orientation_summary = str(meta.get("orientation_summary") or "")
                    style_summary = str(meta.get("style_summary") or "")

                    social = {
                        "social_posture": _unique(posture, limit=8),
                        "relationship_facets": _unique(relationship_facets, limit=8),
                        "hazards": _unique(hazards, limit=8),
                    }
                    social_bridge = {
                        "posture": _unique(posture, limit=8),
                        "hazards": _unique(hazards, limit=8),
                        "framing": _unique(framing, limit=8),
                        "turn_decision": turn_decision,
                        "summary": _unique([orientation_summary, style_summary, turn_decision], limit=3),
                    }
                    # Supplement with ctx keys that are not replicated on the snapshot (e.g.
                    # ``active_relationship_facets``, bridge heuristics) so downstream contracts
                    # match the pre-unification path.
                    leg_soc = _social_summary(ctx)
                    leg_bridge = _social_bridge_summary(ctx)
                    social["social_posture"] = _unique(social["social_posture"] + leg_soc["social_posture"], limit=8)
                    social["relationship_facets"] = _unique(
                        social["relationship_facets"] + leg_soc["relationship_facets"], limit=8
                    )
                    social["hazards"] = _unique(social["hazards"] + leg_soc["hazards"], limit=8)
                    social_bridge["posture"] = _unique(
                        social_bridge["posture"] + list(leg_bridge.get("posture") or []), limit=8
                    )
                    social_bridge["hazards"] = _unique(
                        social_bridge["hazards"] + list(leg_bridge.get("hazards") or []), limit=8
                    )
                    social_bridge["framing"] = _unique(
                        social_bridge["framing"] + list(leg_bridge.get("framing") or []), limit=8
                    )
                    return social, social_bridge
    social = _social_summary(ctx)
    social_bridge = _social_bridge_summary(ctx)
    return social, social_bridge


def _is_identity_sensitive_turn(user_message: str) -> bool:
    text = (user_message or "").lower()
    if not text:
        return False
    return any(re.search(pattern, text) for pattern in _IDENTITY_QUESTION_PATTERNS)


_IDENTITY_RECITAL_LEADING_RE = re.compile(
    r"^you['\u2019]re\s+juniper\b[^.!?\n]*[.!?]\s*",
    re.IGNORECASE,
)
_I_AM_ORION_LEADING_RE = re.compile(
    r"^i['\u2019]m\s+or[ií\u00ed]on\b[^.!?\n]*[.!?]\s*",
    re.IGNORECASE,
)


def suppress_chat_general_speech_identity_priming(ctx: Dict[str, Any]) -> bool:
    """Remove identity-kernel prose from llm_chat_general prompt context on ordinary turns."""
    user_message = _compact(ctx.get("user_message") or "", limit=220)
    if _is_identity_sensitive_turn(user_message):
        return False
    ctx["orion_identity_summary"] = []
    ctx["juniper_relationship_summary"] = []
    brief = ctx.get("chat_stance_brief")
    if isinstance(brief, dict) and not _is_relational_stance_brief(brief):
        scrubbed = dict(brief)
        scrubbed["active_identity_facets"] = []
        scrubbed["active_relationship_facets"] = []
        scrubbed["identity_salience"] = "low"
        if scrubbed.get("task_mode") == "identity_dialogue":
            scrubbed["task_mode"] = "direct_response"
        scrubbed["self_relevance"] = "Answer the latest message directly without identity preamble."
        scrubbed["juniper_relevance"] = "Prioritize practical usefulness over relationship labels."
        scrubbed["response_priorities"] = _unique(
            list(scrubbed.get("response_priorities") or [])
            + ["avoid_identity_recital", "preserve_continuity_without_labels"],
            limit=8,
        )
        scrubbed["response_hazards"] = _unique(
            list(scrubbed.get("response_hazards") or []) + ["identity_recital_on_ordinary_turn"],
            limit=8,
        )
        ctx["chat_stance_brief"] = scrubbed
    return True


_TRANSACTIONAL_CLOSER_SENTENCE_RES = (
    re.compile(r"\blet me know\b", re.IGNORECASE),
    re.compile(r"\bif you need anything\b", re.IGNORECASE),
    re.compile(r"\bneed anything\b", re.IGNORECASE),
    re.compile(r"\bwhat'?s the next step\b", re.IGNORECASE),
    re.compile(r"\bi'?m here if you need\b", re.IGNORECASE),
    re.compile(r"\bwhen you'?re ready\b", re.IGNORECASE),
    re.compile(r"\bmake the time pass\b", re.IGNORECASE),
)

_COMPANION_INVITE_MARKERS = (
    "shoulder to talk",
    "someone to talk",
    "keep my mind off",
    "keep me mind off",
    "mind off",
    "just talk",
    "hold space",
    "don't fix",
    "dont fix",
    "no fixing",
    "no solution",
)

_VENT_CONTINUATION_MARKERS = (
    "thanks",
    "it's hard",
    "its hard",
    "just hard",
    "rough night",
    "terrible night",
    "can't sleep",
    "cant sleep",
    "nurses in and out",
)

_TASK_PIVOT_MARKERS = (
    "can you fix",
    "how do i",
    "deploy",
    "restart",
    "debug",
    "track this",
    "remind me",
    "what's the next step",
    "whats the next step",
)


def _brief_response_hazards(brief: ChatStanceBrief | Dict[str, Any]) -> list[str]:
    if isinstance(brief, ChatStanceBrief):
        return [str(h) for h in brief.response_hazards]
    return [str(h) for h in (brief.get("response_hazards") or [])]


def _should_strip_transactional_closers(
    chat_stance_brief: Dict[str, Any] | ChatStanceBrief | None,
) -> bool:
    if chat_stance_brief is None:
        return False
    if _is_relational_stance_brief(chat_stance_brief):
        return True
    hazards = _brief_response_hazards(chat_stance_brief)
    return "avoid_transactional_closers" in hazards or "avoid_next_steps" in hazards


def _sentence_is_transactional_closer(sentence: str) -> bool:
    candidate = str(sentence or "").strip()
    if not candidate:
        return False
    return any(pattern.search(candidate) for pattern in _TRANSACTIONAL_CLOSER_SENTENCE_RES)


def _split_reply_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", str(text or "").strip())
    return [part.strip() for part in parts if part.strip()]


def strip_transactional_closers(
    text: str,
    *,
    chat_stance_brief: Dict[str, Any] | ChatStanceBrief | None = None,
) -> tuple[str, bool]:
    """Drop trailing customer-support closers when stance contract forbids them."""
    if not _should_strip_transactional_closers(chat_stance_brief):
        return text, False
    candidate = str(text or "").strip()
    if not candidate:
        return text, False
    sentences = _split_reply_sentences(candidate)
    if len(sentences) <= 1:
        if _sentence_is_transactional_closer(sentences[0]):
            return "", True
        return text, False
    changed = False
    while len(sentences) > 1 and _sentence_is_transactional_closer(sentences[-1]):
        sentences.pop()
        changed = True
    if not sentences:
        return text, False
    stripped = " ".join(sentences).strip()
    return stripped, changed


def _history_message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    for key in ("content", "text", "message"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _history_message_role(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    return str(message.get("role") or message.get("speaker") or "").strip().lower()


def _continuation_of_relational_thread(ctx: Dict[str, Any]) -> bool:
    """Carry companion stance across follow-up venting turns in the same thread."""
    user_message = str(ctx.get("user_message") or "").strip().lower()
    if not user_message or _is_identity_sensitive_turn(user_message):
        return False
    if any(marker in user_message for marker in _TASK_PIVOT_MARKERS):
        return False
    if any(re.search(pattern, user_message) for pattern in _TECHNICAL_TURN_PATTERNS):
        return False

    history = ctx.get("message_history")
    if not isinstance(history, list):
        history = []

    recent = [msg for msg in history[-6:] if isinstance(msg, dict)]
    companion_invite_seen = any(
        _history_message_role(msg) == "user"
        and any(marker in _history_message_text(msg).lower() for marker in _COMPANION_INVITE_MARKERS)
        for msg in recent
    )
    if not companion_invite_seen:
        companion_invite_seen = any(marker in user_message for marker in _COMPANION_INVITE_MARKERS)
    if not companion_invite_seen:
        return False

    if any(marker in user_message for marker in _COMPANION_INVITE_MARKERS):
        return True

    if not any(marker in user_message for marker in _VENT_CONTINUATION_MARKERS):
        return False

    for msg in reversed(recent):
        if _history_message_role(msg) != "assistant":
            continue
        assistant_text = _history_message_text(msg).lower()
        if any(
            phrase in assistant_text
            for phrase in (
                "hold space",
                "sit with",
                "don't have to hold",
                "dont have to hold",
                "i'm here",
                "im here",
            )
        ):
            return True
        break
    return False


def _upgrade_brief_for_relational_continuation(brief: ChatStanceBrief) -> ChatStanceBrief:
    merged = brief.model_copy(deep=True)
    if merged.conversation_frame not in _RELATIONAL_CONVERSATION_FRAMES:
        merged.conversation_frame = "reflective"
    if merged.task_mode not in _RELATIONAL_TASK_MODES:
        merged.task_mode = "reflective_dialogue"
    merged.interaction_regime = "relational"
    merged.response_priorities = _unique(
        list(merged.response_priorities)
        + ["companion_presence", "hold_space", "no_solutioning", "situated_curiosity"],
        limit=8,
    )
    merged.response_hazards = _unique(
        list(merged.response_hazards)
        + [
            "avoid_transactional_closers",
            "avoid_next_steps",
            "avoid_customer_support_tone",
            "avoid_task_tracking",
        ],
        limit=8,
    )
    if not merged.juniper_relevance or "practical usefulness" in merged.juniper_relevance.lower():
        merged.juniper_relevance = "Relational continuity matters this turn."
    return merged


_COMPANION_CLOSING_MOVE_MAP: dict[str, str] = {
    "end_with_a_wondering": "End with a wondering, not an offer.",
    "leave_space_without_offer": "Leave space. Do not close with an offer to help.",
    "ground_observation": "End with a grounded observation from the thread.",
    "be_with_silence": "Hold the silence. No closing move required.",
}


def _compile_repair_speech_overlay(repair_contract: dict[str, Any] | None) -> str | None:
    if not isinstance(repair_contract, dict):
        return None
    mode = str(repair_contract.get("mode") or "")
    rules = repair_contract.get("rules") or []
    if mode not in {"repair_concrete", "concrete_bias"} or not rules:
        return None
    intro = (
        "Repair turn: answer concretely and operationally."
        if mode == "repair_concrete"
        else "Add concrete specificity this turn."
    )
    return intro + " " + "; ".join(str(r) for r in rules) + "."


def compile_speech_contract(
    brief: "ChatStanceBrief",
    *,
    repair_contract: dict[str, Any] | None = None,
) -> str:
    """Deterministic regime-specific contract injected near TASK in chat_general.j2.

    Pure Python — no LLM, no I/O. Called after enforce_chat_stance_quality.
    """
    regime = brief.interaction_regime

    if regime is None:
        if brief.task_mode in _RELATIONAL_TASK_MODES or brief.conversation_frame in _RELATIONAL_CONVERSATION_FRAMES:
            regime = "relational"
        else:
            regime = "instrumental"

    if regime == "minimal":
        regime_text = (
            "Keep this reply very short. Do not ask questions. "
            "Release Juniper from replying — offer voice, a pause, or continuation later."
        )
    elif regime == "relational":
        parts = ["This is a companion turn."]
        move = brief.companion_closing_move
        if move and move in _COMPANION_CLOSING_MOVE_MAP:
            parts.append(_COMPANION_CLOSING_MOVE_MAP[move])
        else:
            parts.append("Stay present; do not offer next steps, trackers, or support closers.")
        if "situated_curiosity" in list(brief.response_priorities or []):
            parts.append("Ask one grounded question from this thread — not a generic reversal.")
        regime_text = " ".join(parts)
    else:
        # instrumental (default)
        parts = ["Answer directly."]
        if brief.task_mode == "triage":
            parts.append("Lead with the operational blocker.")
        regime_text = " ".join(parts)

    overlay = _compile_repair_speech_overlay(repair_contract)
    if overlay is None:
        return regime_text
    mode = str((repair_contract or {}).get("mode") or "")
    if mode == "repair_concrete":
        return overlay
    return f"{regime_text} {overlay}"


def strip_identity_recital_leadin(
    text: str,
    user_message: str,
    *,
    chat_stance_brief: Dict[str, Any] | ChatStanceBrief | None = None,
) -> tuple[str, bool]:
    """Drop leading Juniper/Oríon label recital from speech on ordinary turns."""
    if _is_identity_sensitive_turn(user_message):
        return text, False
    if chat_stance_brief is not None and _is_relational_stance_brief(chat_stance_brief):
        return text, False
    candidate = str(text or "")
    if not candidate.strip():
        return text, False
    stripped = candidate
    changed = False
    while True:
        next_text = _IDENTITY_RECITAL_LEADING_RE.sub("", stripped, count=1)
        next_text = _I_AM_ORION_LEADING_RE.sub("", next_text, count=1)
        if next_text == stripped:
            break
        stripped = next_text.lstrip()
        changed = True
    return stripped, changed


def _project_concept_from_beliefs(
    beliefs: UnifiedRelationalBeliefSetV1 | None,
    ctx: Dict[str, Any] | None,
) -> dict[str, list[str]] | None:
    """Projection helper: read induced concepts from unified beliefs → same shape as _concept_summary_from_store.

    Returns None if beliefs are empty / degraded for concept producers, signalling
    the caller should fall through to the direct repository path.
    """
    if beliefs is None:
        return None

    buckets: dict[str, list[str]] = {"self": [], "relationship": [], "growth": [], "tension": []}
    any_concepts = False

    for anchor_key, anchor_slice in beliefs.anchors.items():
        for node in anchor_slice.concepts:
            if node.node_kind not in ("concept",):
                continue
            label = _compact(getattr(node, "label", "") or "", limit=80)
            if not label:
                continue
            any_concepts = True
            ctype = str((node.metadata or {}).get("concept_type") or "").lower()
            if ctype in {"tension", "conflict"}:
                buckets["tension"].append(label)
            elif ctype in {"growth", "development", "emergence", "goal"}:
                buckets["growth"].append(label)
            elif anchor_key == "relationship" or ctype in {"relationship", "social"}:
                buckets["relationship"].append(label)
            else:
                buckets["self"].append(label)

    if not any_concepts:
        return None
    return {k: _unique(v, limit=6) for k, v in buckets.items()}


def _project_autonomy_from_beliefs(
    beliefs: UnifiedRelationalBeliefSetV1 | None,
    ctx: Dict[str, Any],
) -> Dict[str, Any] | None:
    """Projection helper: reconstruct autonomy dict from unified beliefs.

    Aggregates drive/goal/tension/snapshot nodes across orion, relationship, and
    juniper anchors (matching the autonomy adapter's subject coverage).

    Returns None if beliefs have no autonomy nodes, signalling the caller to
    fall through to the direct ``_load_autonomy_state`` path.
    """
    if beliefs is None:
        return None

    drives: list[Any] = []
    goals: list[Any] = []
    tensions: list[Any] = []
    snapshots: list[Any] = []
    any_degraded = False

    for anchor_key in ("orion", "relationship", "juniper"):
        sl = beliefs.anchors.get(anchor_key)
        if not sl:
            continue
        if sl.degraded:
            any_degraded = True
        drives.extend([n for n in sl.drives if n.node_kind == "drive"])
        goals.extend([n for n in sl.goals if n.node_kind == "goal"])
        tensions.extend([n for n in sl.tensions if n.node_kind == "tension"])
        snapshots.extend(
            [s for s in sl.snapshots if getattr(s, "snapshot_source", "") in ("autonomy", "drive_state")]
        )

    if not any([drives, goals, tensions, snapshots]):
        return None

    seen_drive: set[str] = set()
    drive_labels: list[str] = []
    for d in drives:
        dk = str(getattr(d, "drive_kind", "") or "")
        if dk and dk not in seen_drive:
            seen_drive.add(dk)
            drive_labels.append(dk)

    seen_tension: set[str] = set()
    tension_labels: list[str] = []
    for t in tensions:
        tk = str(getattr(t, "tension_kind", "") or "")
        if tk and tk not in seen_tension:
            seen_tension.add(tk)
            tension_labels.append(tk)

    seen_goal_sig: set[str] = set()
    goal_texts: list[str] = []
    for g in goals:
        sig = str((g.metadata or {}).get("proposal_signature") or "").strip().lower()
        gt = str(getattr(g, "goal_text", "") or "")[:120]
        if not gt:
            continue
        key = sig or gt[:64].lower()
        if key not in seen_goal_sig:
            seen_goal_sig.add(key)
            goal_texts.append(gt)

    dominant_drive: str | None = None
    for snap in snapshots:
        dd = str((snap.metadata or {}).get("dominant_drive") or "").strip()
        if dd:
            dominant_drive = dd
            break

    summary = summarize_autonomy_state(None)
    if drive_labels or tension_labels or goal_texts:
        summary = summary.model_copy(
            update={
                "top_drives": drive_labels[:3],
                "dominant_drive": dominant_drive or (drive_labels[0] if drive_labels else None),
                "active_tensions": tension_labels[:4],
                "proposal_headlines": goal_texts[:3],
                "raw_state_present": bool(snapshots),
            }
        )

    # drive_state.v1 projection — kept structurally separate from the `summary`/
    # `debug` fields above (those are the autonomy_state_v2-lineage signal; see
    # orion/self_state/inner_state_registry.py's DUPLICATE note). Isolation rule
    # superseded 2026-07-16 (orion/autonomy/drives_and_autonomy_retrospective.md
    # §8): drive_state is now the live signal for downstream consumers
    # (autonomy_slice.py, mind_runtime.py); kept as a separate dict/key here
    # rather than merged field-by-field into `summary`, since provenance
    # (which snapshot_source a field came from) still matters for debugging.
    pressures: Dict[str, float] = {}
    for d in drives:
        dk = str(getattr(d, "drive_kind", "") or "")
        if not dk or dk in pressures:
            continue
        salience = getattr(getattr(d, "signals", None), "salience", None)
        if salience is not None:
            pressures[dk] = float(salience)

    drive_state_snapshots = [s for s in snapshots if getattr(s, "snapshot_source", "") == "drive_state"]
    activations: Dict[str, bool] = {}
    drive_state_dominant_drive: str | None = None
    drive_state_summary: str | None = None
    for snap in drive_state_snapshots:
        meta = snap.metadata or {}
        if not activations:
            raw_activations = meta.get("activations")
            if isinstance(raw_activations, dict):
                activations = dict(raw_activations)
        if drive_state_dominant_drive is None:
            dd = str(meta.get("dominant_drive") or "").strip()
            if dd:
                drive_state_dominant_drive = dd
        if drive_state_summary is None:
            sm = str(meta.get("summary") or "").strip()
            if sm:
                drive_state_summary = sm

    drive_state_projection: Dict[str, Any] | None = None
    if pressures or activations or drive_state_dominant_drive or drive_state_summary:
        drive_state_projection = {
            "pressures": pressures,
            "activations": activations,
            "dominant_drive": drive_state_dominant_drive,
            "summary": drive_state_summary,
        }

    return {
        "state": None,
        "summary": summary,
        "debug": {
            "backend": "substrate",
            "source": "unified_beliefs",
            "drives_from_beliefs": len(drives),
            "goals_from_beliefs": len(goals),
            "tensions_from_beliefs": len(tensions),
            "degraded": any_degraded,
        },
        "backend": "substrate",
        "selected_subject": "orion",
        "repository_status": {"source_available": True, "source_path": "substrate"},
        "partial_used": False,
        "drive_state": drive_state_projection,
    }


_SELF_STATE_SEVERE_CONDITIONS = {"strained", "unstable"}


def _project_self_state_from_beliefs(
    beliefs: UnifiedRelationalBeliefSetV1 | None,
    ctx: Dict[str, Any],
) -> Dict[str, Any] | None:
    """Projection helper: fold Orion's self-model condition into stance hazards.

    Reads the ``self:overall_condition`` and ``self:{dimension_id}`` belief
    nodes produced by ``orion.substrate.relational.adapters.self_state_ctx``.
    Returns None if beliefs have no self-model nodes (nothing to fold in),
    signalling the caller not to add any self_state-derived hazard.
    """
    if beliefs is None:
        return None

    anchor = beliefs.anchors.get("orion")
    if not anchor:
        return None

    self_nodes = [n for n in anchor.concepts if str(getattr(n, "label", "")).startswith("self:")]
    if not self_nodes:
        return None

    overall_condition: str | None = None
    trajectory_condition: str | None = None
    hazards: list[str] = []
    pressure_threshold = _env_float("SELF_STATE_STANCE_PRESSURE_THRESHOLD", 0.8)

    for node in self_nodes:
        meta = node.metadata or {}
        if node.label == "self:overall_condition":
            overall_condition = meta.get("overall_condition")
            trajectory_condition = meta.get("trajectory_condition")
            if overall_condition in _SELF_STATE_SEVERE_CONDITIONS:
                hazards.append(f"self_state overall_condition={overall_condition}")
        else:
            dim_id = meta.get("self_dimension_id")
            score = meta.get("score")
            if dim_id and isinstance(score, (int, float)) and score >= pressure_threshold:
                hazards.append(f"self_state {dim_id} score={score:.2f} above threshold")

    if overall_condition is None and not hazards:
        return None

    return {
        "overall_condition": overall_condition,
        "trajectory_condition": trajectory_condition,
        "hazards": hazards,
    }


def _project_context_provenance_hazard(ctx: Dict[str, Any]) -> str | None:
    """Projection helper: name which of this turn's ctx keys are genuinely
    live substrate/biometric signal vs. retrieved/static/tool content.

    Reads the shared ``orion.schemas.context_provenance`` registry against
    whatever keys are actually present in ``ctx`` this turn. Returns a fixed,
    compact hazard string when any live_runtime_projection key is present, or
    None otherwise -- exists so a self-referential claim about "what's
    happening right now" has a ground truth to check against instead of a
    plausible-sounding guess (see
    project_orion_substrate_bridge_confabulation for the incident this
    closes: a GitHub file fetch narrated as live substrate computation).

    Deliberately fixed-length, not enumerating which keys qualify: this
    string passes through ``_unique()``, which hard-truncates every hazard
    to 140 chars, and the live-key list can be long enough to push an
    enumerated message past that limit and cut off the actual instruction.
    The enumerated version lives in the FCC motor prompt instead (see
    orion/harness/prefix.py's CONTEXT PROVENANCE block), which has no such
    length cap.
    """
    from .grounding_capsule import context_provenance_for_ctx

    has_live_key = any(
        kind == "live_runtime_projection"
        for kind in context_provenance_for_ctx(ctx).values()
    )
    if not has_live_key:
        return None
    return (
        "context_provenance: only live_runtime_projection context is "
        "happening now; recalled/static/tool content is not live."
    )


def _project_reverie_glimpse(ctx: Dict[str, Any]) -> str | None:
    """Projection helper: surface the latest fresh, non-hollow reverie thought.

    Reads ``ctx['latest_reverie_thought']`` (the ``thought_json`` payload hydrated
    by ``felt_state_reader``'s ``latest_reverie_thought`` lane, already age-gated
    there). Extracts ONLY the ``interpretation`` narration string -- never
    ``evidence_refs``/``coalition``/``chain_id`` or any other field, since those
    are internal grounding/audit data, not narration meant to be read.

    Gates on BOTH the stored ``hollow`` flag AND an independent re-derivation
    via ``SpontaneousThoughtV1.is_hollow()`` -- rejecting if either says
    hollow. The stored flag alone isn't trusted (it could be stale/wrong if
    the row predates a schema or guard change), and the re-derivation alone
    isn't trusted either (it can't see ``extra_grounding`` widening the
    semantic-lift path may have applied at write time, so a thought the
    producer explicitly marked hollow must never be surfaced just because a
    simpler re-check happens to pass it).

    Fail-open: returns None on anything missing, malformed, or unparsable
    (including a payload that no longer validates as ``SpontaneousThoughtV1``
    at all). Never raises.
    """
    try:
        raw = ctx.get("latest_reverie_thought")
        if raw is None:
            return None
        if isinstance(raw, str):
            if not raw.strip():
                return None
            payload = json.loads(raw)
        elif isinstance(raw, dict):
            payload = raw
        else:
            return None
        if not isinstance(payload, dict):
            return None

        thought = SpontaneousThoughtV1.model_validate(payload)
        if thought.hollow or thought.is_hollow():
            return None
        interpretation = thought.interpretation.strip()
        if not interpretation:
            return None
        return interpretation
    except Exception:
        logger.debug("reverie_glimpse_projection_failed", exc_info=True)
        return None


_MAX_RECENT_DISPATCH_ACTIONS = 3
# The producer (execution-dispatch-runtime) is expected to already bound
# `summary` per the P2 design doc. This is a generous defensive ceiling only,
# not the primary truncation point -- do not rely on it as the main guard.
_DISPATCH_ACTION_SUMMARY_MAX_CHARS = 300
# Layer 9 (execution-dispatch-runtime) always emits under this subject --
# it's self-directed autonomous action, never relationship-scoped. Reading
# ctx["chat_autonomy_state_v2"]["last_action_outcomes"] instead would silently
# return [] whenever _run_autonomy_reducer resolved a different subject for
# THIS turn (e.g. "relationship" during contextual fallback when Orion's own
# drive/autonomy graph is unavailable -- see select_preferred_autonomy_lookup)
# even though real dispatch actions exist under "orion". Querying directly
# decouples this feature from whichever subject the ambient reducer used.
_DISPATCH_ACTION_SUBJECT = "orion"


def _project_recent_dispatch_actions(ctx: Dict[str, Any]) -> list[dict]:
    """Projection helper: surface the most recent autonomous dispatch-action
    outcomes as bounded, privacy-safe evidence for chat narration.

    Queries ``load_action_outcomes(subject=_DISPATCH_ACTION_SUBJECT)``
    directly, independent of ``ctx`` (the ``ctx`` parameter is accepted only
    to match this file's other ``_project_*`` helpers' call-site shape).
    Deliberately NOT ``ctx['chat_autonomy_state_v2']['last_action_outcomes']``
    -- see ``_DISPATCH_ACTION_SUBJECT``'s comment above for why that would
    silently go blank during autonomy contextual-fallback turns. Also NOT
    ``ctx['chat_autonomy_state']`` (the plain ``AutonomyStateV1`` graph-lookup
    state): it has no ``last_action_outcomes`` field at all (that field only
    exists on the ``AutonomyStateV2`` subclass in ``orion/autonomy/models.py``,
    populated by ``reduce_autonomy_state`` in ``orion/autonomy/reducer.py``).

    Takes at most ``_MAX_RECENT_DISPATCH_ACTIONS`` entries, newest-first by
    ``observed_at``. Entries with a missing/unparsable ``observed_at`` sort
    last (oldest) rather than crashing the sort.

    Projects each entry to EXACTLY ``{kind, summary, success, observed_at}``
    -- never ``action_id``, ``query``, ``articles``, or ``salience``, since
    those are internal correlation/audit fields, the same reasoning
    ``_project_reverie_glimpse`` documents for its own field
    (``evidence_refs``/``coalition``/``chain_id``).

    Fail-open: returns ``[]`` on anything missing, malformed, or unparsable.
    Never raises (``load_action_outcomes`` already degrades gracefully on its
    own SQL/file-store failures; this wraps the projection logic too).
    """
    del ctx  # accepted only for call-site consistency with sibling _project_* helpers
    try:
        outcomes = load_action_outcomes(subject=_DISPATCH_ACTION_SUBJECT)
        if not outcomes:
            return []

        def _field(item: Any, name: str) -> Any:
            if isinstance(item, dict):
                return item.get(name)
            return getattr(item, name, None)

        def _sort_epoch(item: Any) -> float:
            observed_at = _field(item, "observed_at")
            dt = observed_at if isinstance(observed_at, datetime) else None
            if dt is None and isinstance(observed_at, str) and observed_at.strip():
                try:
                    dt = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
                except Exception:
                    dt = None
            if dt is None:
                return float("-inf")
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()

        ordered = sorted(outcomes, key=_sort_epoch, reverse=True)[:_MAX_RECENT_DISPATCH_ACTIONS]

        projected: list[dict] = []
        for item in ordered:
            observed_at = _field(item, "observed_at")
            if isinstance(observed_at, datetime):
                observed_at = observed_at.isoformat()
            kind = _field(item, "kind")
            summary = _field(item, "summary")
            # Skip items that don't validate as a real outcome (non-string
            # kind/summary) rather than surfacing a hollow {kind: None,
            # summary: None, ...} row -- an empty-shell entry here would be
            # exactly the kind of meaningless "evidence" this section exists
            # to prevent. Only reachable via schema drift in a future writer;
            # today's sole producer (reduce_autonomy_state) always sets both.
            if not isinstance(kind, str) or not kind or not isinstance(summary, str) or not summary:
                continue
            if len(summary) > _DISPATCH_ACTION_SUMMARY_MAX_CHARS:
                summary = summary[:_DISPATCH_ACTION_SUMMARY_MAX_CHARS]
            projected.append(
                {
                    "kind": kind,
                    "summary": summary,
                    "success": _field(item, "success"),
                    "observed_at": observed_at,
                }
            )
        return projected
    except Exception:
        logger.debug("recent_dispatch_actions_projection_failed", exc_info=True)
        return []


def _concept_summary_from_store(ctx: Dict[str, Any] | None = None) -> dict[str, list[str]]:
    ctx = ctx if isinstance(ctx, dict) else {}
    try:
        from orion.spark.concept_induction.profile_repository import build_concept_profile_repository  # noqa: PLC0415

        repository = build_concept_profile_repository()
    except Exception:
        return {"self": [], "relationship": [], "growth": [], "tension": []}

    subjects = ("orion", "relationship", "juniper")
    lookups = repository.list_latest(
        subjects,
        observer={
            "consumer": "chat_stance",
            "correlation_id": str(ctx.get("correlation_id") or ctx.get("trace_id") or ""),
            "session_id": str(ctx.get("session_id") or ""),
        },
    )
    status = repository.status()
    logger.info(
        "concept_profile_repository_status %s",
        json.dumps(
            {
                "backend": status.backend,
                "source_path": status.source_path,
                "placeholder_default_in_use": status.placeholder_default_in_use,
                "subjects_requested": len(subjects),
                "profiles_returned": sum(1 for lookup in lookups if lookup.profile is not None),
            },
            sort_keys=True,
        ),
    )

    buckets: dict[str, list[str]] = {"self": [], "relationship": [], "growth": [], "tension": []}

    for lookup in lookups:
        subject = lookup.subject
        profile = lookup.profile
        if not profile:
            continue

        sorted_concepts = sorted(profile.concepts, key=lambda c: (float(c.salience or 0.0), float(c.confidence or 0.0)), reverse=True)
        for concept in sorted_concepts[:14]:
            label = _compact(concept.label, limit=80)
            ctype = str(concept.type or "").lower()
            if not label:
                continue
            if ctype in {"tension", "conflict"}:
                buckets["tension"].append(label)
            elif ctype in {"growth", "development", "emergence", "goal"}:
                buckets["growth"].append(label)
            elif subject == "relationship" or ctype in {"relationship", "social"}:
                buckets["relationship"].append(label)
            else:
                buckets["self"].append(label)

    return {k: _unique(v, limit=6) for k, v in buckets.items()}


def _social_summary(ctx: Dict[str, Any]) -> dict[str, list[str]]:
    social_posture: list[str] = []
    relationship_facets: list[str] = []
    hazards: list[str] = []

    snapshot = ctx.get("social_inspection_snapshot") or {}
    if isinstance(snapshot, dict):
        for key in ("summary", "stance", "relationship_posture"):
            val = snapshot.get(key)
            if isinstance(val, str):
                relationship_facets.append(val)
        for section in (snapshot.get("sections") or [])[:4]:
            if isinstance(section, dict):
                relationship_facets.extend(section.get("selected_state") or [])
                hazards.extend(section.get("excluded_state") or [])
                social_posture.extend(section.get("softened_state") or [])

    for key in ("social_posture", "social_hazards", "active_relationship_facets"):
        val = ctx.get(key)
        if isinstance(val, list):
            if key == "social_posture":
                social_posture.extend(str(x) for x in val)
            elif key == "social_hazards":
                hazards.extend(str(x) for x in val)
            else:
                relationship_facets.extend(str(x) for x in val)

    return {
        "social_posture": _unique(social_posture, limit=6),
        "relationship_facets": _unique(relationship_facets, limit=6),
        "hazards": _unique(hazards, limit=6),
    }


def _social_bridge_summary(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Bounded adapter from orion-social-room-bridge payload into stance-relevant fields."""
    posture: list[str] = []
    hazards: list[str] = []
    framing: list[str] = []

    def _add_tag_if(condition: bool, tag: str) -> None:
        if condition:
            posture.append(tag)

    turn_policy = ctx.get("social_turn_policy") if isinstance(ctx.get("social_turn_policy"), dict) else {}
    stance_snapshot = ctx.get("social_stance_snapshot") if isinstance(ctx.get("social_stance_snapshot"), dict) else {}
    peer_style = ctx.get("social_peer_style_hint") if isinstance(ctx.get("social_peer_style_hint"), dict) else {}
    context_window = ctx.get("social_context_window") if isinstance(ctx.get("social_context_window"), dict) else {}
    routing = ctx.get("social_thread_routing") if isinstance(ctx.get("social_thread_routing"), dict) else {}
    repair = ctx.get("social_repair_decision") if isinstance(ctx.get("social_repair_decision"), dict) else {}

    reasons = [str(item).strip() for item in (turn_policy.get("reasons") or []) if str(item).strip()]
    reasons_blob = " ".join(reasons).lower()
    user_message = _compact(ctx.get("user_message") or "", limit=400).lower()

    orientation = _compact(
        stance_snapshot.get("recent_social_orientation_summary") or stance_snapshot.get("summary") or "",
        limit=180,
    ).lower()
    style_hint = _compact(peer_style.get("style_hints_summary") or "", limit=180).lower()
    routing_decision = str(routing.get("routing_decision") or "").strip().lower()

    _add_tag_if("direct" in orientation or "direct" in style_hint or "direct" in reasons_blob, "direct")
    _add_tag_if("warm" in orientation or "warm" in style_hint, "warm")
    _add_tag_if("playful" in orientation or "playful" in style_hint, "playful")
    _add_tag_if("reflect" in orientation or "reflect" in reasons_blob, "reflective")
    _add_tag_if("strain" in orientation or "repair" in reasons_blob or "friction" in reasons_blob, "strained")
    _add_tag_if(
        "technical" in orientation
        or "technical" in style_hint
        or any(re.search(pattern, user_message) for pattern in _TECHNICAL_TURN_PATTERNS),
        "technical",
    )
    _add_tag_if(any(token in user_message for token in ("pissed", "not happy", "blocked", "offline")), "operational_frustration")
    _add_tag_if(bool(turn_policy.get("addressed")), "addressed")
    _add_tag_if(bool(turn_policy.get("should_speak")), "engaged_turn")
    _add_tag_if(routing_decision == "reply_to_peer", "peer_reply_mode")
    _add_tag_if(routing_decision == "reply_to_room", "room_reply_mode")
    _add_tag_if(str(repair.get("decision") or "").strip().lower() in {"yield", "reset_thread"}, "deescalate")

    for candidate in (context_window.get("selected_candidates") or [])[:6]:
        if not isinstance(candidate, dict):
            continue
        kind = str(candidate.get("candidate_kind") or "").strip().lower()
        decision = str(candidate.get("inclusion_decision") or "include").strip().lower()
        summary = _compact(candidate.get("summary"), limit=120)
        if kind:
            framing.append(f"{kind}:{decision}")
        if summary:
            framing.append(summary)
        if decision == "exclude":
            hazards.append(f"context_excluded:{kind or 'unknown'}")
        if decision == "soften":
            hazards.append(f"context_softened:{kind or 'unknown'}")

    for reason in reasons[:8]:
        lowered = reason.lower()
        if "cooldown" in lowered:
            hazards.append("cooldown_active")
        if "duplicate" in lowered:
            hazards.append("duplicate_message")
        if "self-loop" in lowered:
            hazards.append("self_message_loop")
        if "aimed at another participant" in lowered:
            hazards.append("peer_targeted_elsewhere")
        if "addressed_only" in lowered:
            hazards.append("not_addressed")

    summary = _unique(
        [
            _compact(stance_snapshot.get("recent_social_orientation_summary"), limit=140),
            _compact(peer_style.get("style_hints_summary"), limit=140),
            _compact(turn_policy.get("decision"), limit=60),
        ],
        limit=3,
    )
    return {
        "posture": _unique(posture, limit=8),
        "hazards": _unique(hazards, limit=8),
        "framing": _unique(framing, limit=8),
        "turn_decision": _compact(turn_policy.get("decision"), limit=40),
        "summary": summary,
    }


def _reflective_summary(ctx: Dict[str, Any]) -> dict[str, list[str]]:
    themes: list[str] = []
    tensions: list[str] = []
    dream_motifs: list[str] = []

    recall_bundle = ctx.get("recall_bundle") or {}
    fragments = recall_bundle.get("fragments") if isinstance(recall_bundle, dict) else []
    if isinstance(fragments, list):
        for frag in fragments[:18]:
            if not isinstance(frag, dict):
                continue
            source = str(frag.get("source") or "").lower()
            snippet = _compact(frag.get("snippet"), limit=180)
            if not snippet:
                continue
            if "journal" in source or "metacog" in source:
                themes.append(snippet)
            if "tension" in source:
                tensions.append(snippet)
            if "dream" in source:
                dream_motifs.append(snippet)

    journal = ctx.get("journal_artifacts")
    if isinstance(journal, list):
        for item in journal[:6]:
            if isinstance(item, dict):
                if item.get("theme"):
                    themes.append(str(item.get("theme")))
                if item.get("tension"):
                    tensions.append(str(item.get("tension")))
                if item.get("dream_motif"):
                    dream_motifs.append(str(item.get("dream_motif")))

    pageindex_ctx = ctx.get("journal_pageindex_context")
    if isinstance(pageindex_ctx, dict):
        selected_entries = pageindex_ctx.get("selected_entries")
        if isinstance(selected_entries, list):
            for entry in selected_entries[:6]:
                if not isinstance(entry, dict):
                    continue
                themes.extend(str(v) for v in (entry.get("reflective_themes") or []) if str(v).strip())
                tensions.extend(str(v) for v in (entry.get("active_tensions") or []) if str(v).strip())
                dream_motifs.extend(str(v) for v in (entry.get("dream_motifs") or []) if str(v).strip())
                excerpt = _compact(entry.get("body_excerpt"), limit=180)
                if excerpt:
                    themes.append(excerpt)
        selected_blocks = pageindex_ctx.get("selected_blocks")
        if isinstance(selected_blocks, list):
            for block in selected_blocks[:8]:
                if not isinstance(block, dict):
                    continue
                excerpt = _compact(block.get("excerpt") or block.get("text"), limit=180)
                if excerpt:
                    themes.append(excerpt)

    return {
        "themes": _unique(themes, limit=6),
        "tensions": _unique(tensions, limit=6),
        "dream_motifs": _unique(dream_motifs, limit=4),
    }


def _merge_orion_goals_into_state(
    preferred_state: Any,
    orion_state: Any,
) -> Any:
    """Keep Orion goal proposals visible when relationship drives supply stance context."""
    from orion.autonomy.models import AutonomyStateV1

    if not isinstance(preferred_state, AutonomyStateV1) or not isinstance(orion_state, AutonomyStateV1):
        return preferred_state
    seen = {goal.artifact_id for goal in preferred_state.goal_headlines}
    merged_goals = list(preferred_state.goal_headlines)
    for goal in orion_state.goal_headlines:
        if goal.artifact_id not in seen:
            merged_goals.append(goal)
            seen.add(goal.artifact_id)
    if len(merged_goals) == len(preferred_state.goal_headlines):
        return preferred_state
    from orion.autonomy.summary import dedupe_goal_headlines_by_drive_origin

    return preferred_state.model_copy(
        update={"goal_headlines": dedupe_goal_headlines_by_drive_origin(merged_goals, limit=3)}
    )


def _load_autonomy_state_fallback_local(
    ctx: Dict[str, Any],
    plan: AutonomyGraphReadPlan,
    graphdb_cfg: dict[str, Any],
    started_at: float,
) -> Dict[str, Any]:
    """No remote graph HTTP: empty local lookups + YAML-friendly autonomy summary."""
    from orion.graph.backend_config import strip_graph_credentials

    repository = LocalAutonomyRepository()
    subjects = ["orion", "relationship", "juniper"]
    observer = {
        "consumer": "chat_stance",
        "correlation_id": str(ctx.get("correlation_id") or ctx.get("trace_id") or ""),
        "session_id": str(ctx.get("session_id") or ""),
        "autonomy_subject_fanout": autonomy_subject_fanout_from_runtime_ctx(ctx),
    }
    lookups = repository.list_latest(subjects, observer=observer)
    by_subject = {lookup.subject: lookup for lookup in lookups}
    preferred = by_subject.get("orion")
    if preferred is None or preferred.availability != "available":
        preferred = by_subject.get("relationship")
    selected_subject = preferred.subject if preferred is not None else None
    partial_used = False

    summary_base = summarize_autonomy_state(None)
    hazard = "autonomy_graph:v1_fallback_identity_yaml"
    hazards = list(summary_base.response_hazards or [])
    if hazard not in hazards:
        hazards.append(hazard)
    summary = summary_base.model_copy(update={"response_hazards": hazards})

    debug = {
        subject: {
            "availability": by_subject.get(subject).availability if by_subject.get(subject) else "empty",
            "present": bool(by_subject.get(subject) and by_subject.get(subject).state is not None),
            "unavailable_reason": by_subject.get(subject).unavailable_reason if by_subject.get(subject) else None,
            "subqueries": (by_subject.get(subject).subquery_diagnostics or {}) if by_subject.get(subject) else {},
        }
        for subject in SUBJECT_BINDINGS
    }
    repo_status = repository.status()
    skipped = plan.skipped_reason or "backend_disabled"
    if plan.mode == "graphdb_degraded":
        ag_backend = "graphdb"
    elif plan.mode == "sparql_degraded":
        ag_backend = "sparql"
    else:
        ag_backend = "disabled"
    ep_dbg = strip_graph_credentials(str(graphdb_cfg.get("endpoint") or "")) or (
        "graphdb:unconfigured" if plan.mode in ("graphdb", "graphdb_degraded") else "sparql:unconfigured"
    )
    debug["_runtime"] = {
        "backend": repo_status.backend,
        "selected_subject": selected_subject,
        "endpoint_repo": ep_dbg,
        "query_url": ep_dbg,
        "timeout_sec": 0.0,
        "subject_max_workers": resolve_autonomy_subject_max_workers(),
        "subquery_max_workers": resolve_autonomy_subquery_max_workers(),
        "repository_status": {
            "source_available": repo_status.source_available,
            "source_path": repo_status.source_path,
        },
        "autonomy_graph_backend": ag_backend,
        "autonomy_graph_explicit_backend": plan.explicit_backend,
        "autonomy_graph_cutover_mode": "v1_safe",
        "autonomy_graph_skipped_reason": skipped,
        "fallback": "identity_yaml",
    }
    exported_keys = sorted(["autonomy_backend", "autonomy_debug", "autonomy_selected_subject", "autonomy_summary"])
    logger.info(
        "autonomy_lookup_turn %s",
        json.dumps(
            {
                "backend": repo_status.backend,
                "source_path": repo_status.source_path,
                "source_available": repo_status.source_available,
                "config_source": graphdb_cfg["source"],
                "repo": graphdb_cfg["repo"],
                "endpoint_repo": ep_dbg,
                "query_url": ep_dbg,
                "timeout_sec": 0.0,
                "subject_max_workers": resolve_autonomy_subject_max_workers(),
                "subquery_max_workers": resolve_autonomy_subquery_max_workers(),
                "subjects_requested": subjects,
                "states_returned": sum(1 for item in lookups if item.availability == "available"),
                "availability_counts": {
                    "available": sum(1 for item in lookups if item.availability == "available"),
                    "empty": sum(1 for item in lookups if item.availability == "empty"),
                    "unavailable": sum(1 for item in lookups if item.availability == "unavailable"),
                    "partial": 0,
                },
                "unavailable_reasons": sorted(
                    {
                        item.unavailable_reason
                        for item in lookups
                        if item.availability == "unavailable" and item.unavailable_reason
                    }
                ),
                "selected_subject": selected_subject,
                "selected_subject_partial": partial_used,
                "selected_subject_availability": preferred.availability if preferred is not None else "empty",
                "selected_subject_unavailable_reason": preferred.unavailable_reason if preferred is not None else None,
                "mapped_state": bool(preferred and preferred.state is not None),
                "summary_present": bool(summary and summary.stance_hint),
                "elapsed_ms_before_llm_emit": round((time.perf_counter() - started_at) * 1000.0, 2),
                "subject_availability": {
                    subject: {
                        "availability": by_subject.get(subject).availability if by_subject.get(subject) else "empty",
                        "unavailable_reason": by_subject.get(subject).unavailable_reason if by_subject.get(subject) else None,
                    }
                    for subject in subjects
                },
                "exported_metadata_keys": exported_keys,
                "debug": debug,
                "autonomy_graph_backend": ag_backend,
                "autonomy_graph_explicit_backend": plan.explicit_backend,
                "autonomy_graph_cutover_mode": "v1_safe",
                "autonomy_graph_skipped_reason": skipped,
            },
            sort_keys=True,
        ),
    )
    return {
        "lookups": lookups,
        "state": None,
        "backend": repo_status.backend,
        "selected_subject": selected_subject,
        "repository_status": {
            "backend": repo_status.backend,
            "source_path": repo_status.source_path,
            "source_available": repo_status.source_available,
        },
        "summary": summary,
        "debug": debug,
    }


def _should_skip_autonomy_graph_load(ctx: Dict[str, Any], *, verb: str) -> bool:
    opts = ctx.get("options") if isinstance(ctx.get("options"), dict) else {}
    meta = ctx.get("metadata") if isinstance(ctx.get("metadata"), dict) else {}
    if verb in {"memory_graph_suggest", "introspect_spark"}:
        return True
    if bool(ctx.get("skip_autonomy_context")) or bool(opts.get("skip_autonomy_context")):
        return True
    if bool(ctx.get("skip_chat_stance_inputs")) or bool(opts.get("skip_chat_stance_inputs")):
        return True
    if bool(meta.get("consolidation_suggest")):
        return True
    return False


def _load_autonomy_state(ctx: Dict[str, Any]) -> Dict[str, Any]:
    started_at = time.perf_counter()
    graphdb_cfg = resolve_autonomy_graphdb_config()
    verb = str(ctx.get("verb") or "").strip().lower()
    mode = str(ctx.get("mode") or "").strip().lower()
    plan = resolve_autonomy_graph_read_plan(ctx)

    if _should_skip_autonomy_graph_load(ctx, verb=verb):
        logger.info(
            "chat_stance_skip_autonomy_graph verb=%s correlation_id=%s",
            verb,
            ctx.get("correlation_id") or ctx.get("trace_id"),
        )
        return _load_autonomy_state_fallback_local(ctx, plan, graphdb_cfg, started_at)

    log_autonomy_graph_backend_decision(plan=plan, consumer="chat_stance", verb=verb, mode=mode)

    if plan.mode not in ("graphdb", "sparql"):
        reason = plan.skipped_reason or "backend_disabled"
        if plan.mode == "graphdb_degraded":
            logger.info(
                "autonomy_graph_backend_degraded consumer=chat_stance verb=%s mode=%s explicit=true reason=%s fallback=identity_yaml",
                verb,
                mode,
                reason,
            )
        elif plan.mode == "sparql_degraded":
            logger.info(
                "autonomy_graph_backend_degraded consumer=chat_stance verb=%s mode=%s reason=%s fallback=identity_yaml",
                verb,
                mode,
                reason,
            )
        else:
            logger.info(
                "autonomy_graph_backend_blocked consumer=chat_stance verb=%s mode=%s reason=%s fallback=identity_yaml",
                verb,
                mode,
                reason,
            )
        return _load_autonomy_state_fallback_local(ctx, plan, graphdb_cfg, started_at)

    endpoint = plan.endpoint
    from orion.graph.backend_config import strip_graph_credentials

    ag_remote = "sparql" if plan.mode == "sparql" else "graphdb"
    ep_log = strip_graph_credentials(endpoint or "") or f"{ag_remote}:unconfigured"
    backend = (os.getenv("AUTONOMY_REPOSITORY_BACKEND") or "graph").strip().lower()
    if backend not in {"graph", "local", "shadow"}:
        backend = "graph"

    subjects = list(plan.subjects)
    if is_quick_autonomy_graph_lane(ctx):
        subject_workers = max(1, min(resolve_autonomy_subject_max_workers(), len(subjects) or 1))
        subquery_workers = 1
    else:
        subject_workers = resolve_autonomy_subject_max_workers()
        subquery_workers = resolve_autonomy_chat_stance_subquery_max_workers()
    timeout_used = plan.timeout_sec

    repository = build_autonomy_repository(
        backend=backend,
        endpoint=endpoint,
        timeout_sec=timeout_used,
        user=plan.user,
        password=plan.password,
        goals_limit=_env_int("AUTONOMY_GOALS_LIMIT", 3),
        subject_max_workers=subject_workers,
        subquery_max_workers=subquery_workers,
        active_subqueries=plan.active_subqueries,
        drives_query_limit=resolve_autonomy_drives_query_limit(compact=True),
    )
    observer = {
        "consumer": "chat_stance",
        "correlation_id": str(ctx.get("correlation_id") or ctx.get("trace_id") or ""),
        "session_id": str(ctx.get("session_id") or ""),
        "autonomy_subject_fanout": autonomy_subject_fanout_from_runtime_ctx(ctx),
    }
    lookups = repository.list_latest(subjects, observer=observer)
    by_subject = {lookup.subject: lookup for lookup in lookups}
    for subject in SUBJECT_BINDINGS:
        if subject not in by_subject:
            by_subject[subject] = AutonomyLookupV1(subject=subject, state=None, availability="empty")
    selection = select_preferred_autonomy_lookup(by_subject)
    preferred = selection.lookup
    selected_subject = selection.selected_subject
    contextual_fallback = selection.contextual_fallback
    orion_lookup = selection.orion_lookup
    state_for_summary = preferred.state if preferred and preferred.state is not None else None
    if state_for_summary and contextual_fallback and orion_lookup and orion_lookup.state is not None:
        state_for_summary = _merge_orion_goals_into_state(state_for_summary, orion_lookup.state)
    partial_used = bool(
        preferred
        and (
            preferred.availability == "degraded"
            or contextual_fallback
            or any(
                str((preferred.subquery_diagnostics or {}).get(name, {}).get("status", "ok")) not in {"ok", "empty"}
                for name in ("identity", "drives", "goals")
            )
        )
    )

    summary = summarize_autonomy_lookup(
        state_for_summary,
        selected_subject=selected_subject,
        availability=preferred.availability if preferred is not None else "empty",
        subquery_diagnostics=preferred.subquery_diagnostics if preferred is not None else None,
        by_subject=by_subject,
        contextual_fallback=contextual_fallback,
    )
    if preferred and preferred.availability == "unavailable":
        ur = (preferred.unavailable_reason or "").lower()
        if ur in {"timeout", "connection_error"}:
            hazard = "autonomy_graph:v1_fallback_identity_yaml"
            hazards = list(summary.response_hazards or [])
            if hazard not in hazards:
                hazards.append(hazard)
            summary = summary.model_copy(update={"response_hazards": hazards})
            logger.info(
                "autonomy_graph_degraded consumer=chat_stance verb=%s reason=%s elapsed_ms=%s fallback=identity_yaml",
                verb,
                ur,
                round((time.perf_counter() - started_at) * 1000.0, 2),
            )

    debug = {
        subject: {
            "availability": by_subject.get(subject).availability if by_subject.get(subject) else "empty",
            "present": bool(by_subject.get(subject) and by_subject.get(subject).state is not None),
            "unavailable_reason": by_subject.get(subject).unavailable_reason if by_subject.get(subject) else None,
            "subqueries": (by_subject.get(subject).subquery_diagnostics or {}) if by_subject.get(subject) else {},
        }
        for subject in SUBJECT_BINDINGS
    }
    repo_status = repository.status()
    debug["_runtime"] = {
        "backend": repo_status.backend,
        "selected_subject": selected_subject,
        "endpoint_repo": ep_log,
        "query_url": ep_log,
        "timeout_sec": timeout_used,
        "subject_max_workers": subject_workers,
        "subquery_max_workers": subquery_workers,
        "repository_status": {
            "source_available": repo_status.source_available,
            "source_path": repo_status.source_path,
        },
        "autonomy_graph_backend": ag_remote,
        "autonomy_graph_cutover_mode": "v1_safe",
        "autonomy_graph_skipped_reason": None,
        "fallback": None,
        "selected_subject_partial": partial_used,
        "contextual_fallback": contextual_fallback,
        "state_quality": summary.state_quality,
        "stance_mode": summary.stance_mode,
        "degraded_reason": summary.degraded_reason,
        "facet_health": summary.facet_health,
        "context_note": summary.context_note,
    }
    exported_keys = sorted(["autonomy_backend", "autonomy_debug", "autonomy_selected_subject", "autonomy_summary"])
    if preferred and preferred.availability in {"available", "degraded"} and preferred.state is not None:
        exported_keys.append("autonomy_state")
    logger.info(
        "autonomy_lookup_turn %s",
        json.dumps(
            {
                "backend": repo_status.backend,
                "source_path": repo_status.source_path,
                "source_available": repo_status.source_available,
                "config_source": graphdb_cfg["source"],
                "repo": graphdb_cfg["repo"],
                "endpoint_repo": ep_log,
                "query_url": ep_log,
                "timeout_sec": timeout_used,
                "subject_max_workers": subject_workers,
                "subquery_max_workers": subquery_workers,
                "subjects_requested": subjects,
                "states_returned": sum(1 for item in lookups if item.availability == "available"),
                "availability_counts": {
                    "available": sum(1 for item in lookups if item.availability == "available"),
                    "degraded": sum(1 for item in lookups if item.availability == "degraded"),
                    "empty": sum(1 for item in lookups if item.availability == "empty"),
                    "unavailable": sum(1 for item in lookups if item.availability == "unavailable"),
                    "partial": sum(
                        1
                        for item in lookups
                        if item.availability == "degraded"
                        or (
                            item.availability == "available"
                            and any(
                                str((item.subquery_diagnostics or {}).get(name, {}).get("status", "ok")) not in {"ok", "empty"}
                                for name in ("identity", "drives", "goals")
                            )
                        )
                    ),
                },
                "unavailable_reasons": sorted(
                    {
                        item.unavailable_reason
                        for item in lookups
                        if item.availability == "unavailable" and item.unavailable_reason
                    }
                ),
                "selected_subject": selected_subject,
                "selected_subject_partial": partial_used,
                "selected_subject_availability": preferred.availability if preferred is not None else "empty",
                "selected_subject_unavailable_reason": preferred.unavailable_reason if preferred is not None else None,
                "mapped_state": bool(preferred and preferred.state is not None),
                "summary_present": bool(summary and summary.stance_hint),
                "elapsed_ms_before_llm_emit": round((time.perf_counter() - started_at) * 1000.0, 2),
                "subject_availability": {
                    subject: {
                        "availability": by_subject.get(subject).availability if by_subject.get(subject) else "empty",
                        "unavailable_reason": by_subject.get(subject).unavailable_reason if by_subject.get(subject) else None,
                    }
                    for subject in ["orion", "relationship", "juniper"]
                },
                "exported_metadata_keys": exported_keys,
                "debug": debug,
                "autonomy_graph_backend": ag_remote,
                "autonomy_graph_cutover_mode": "v1_safe",
                "autonomy_graph_skipped_reason": None,
            },
            sort_keys=True,
        ),
    )
    return {
        "lookups": lookups,
        "state": state_for_summary,
        "backend": repo_status.backend,
        "selected_subject": selected_subject,
        "contextual_fallback": contextual_fallback,
        "repository_status": {
            "backend": repo_status.backend,
            "source_path": repo_status.source_path,
            "source_available": repo_status.source_available,
        },
        "summary": summary,
        "debug": debug,
    }


def _compile_reasoning_summary(ctx: Dict[str, Any]) -> Dict[str, Any]:
    request = ReasoningSummaryRequestV1(
        anchor_scope="orion",
        subject_refs=[str(v) for v in (ctx.get("reasoning_subject_refs") or []) if str(v).strip()],
    )
    try:
        repository = ctx.get("reasoning_repository")
        if repository is None:
            repository = InMemoryReasoningRepository()
            raw_artifacts = ctx.get("reasoning_artifacts")
            if isinstance(raw_artifacts, list) and raw_artifacts:
                write_request = ReasoningWriteRequestV1(
                    context=ReasoningWriteContextV1(
                        source_family="manual",
                        source_kind="chat_stance_ctx",
                        source_channel="orion:cortex_exec:chat_stance",
                        producer="chat_stance",
                    ),
                    artifacts=raw_artifacts,
                )
                repository.write_artifacts(write_request)

        compiler = ReasoningSummaryCompiler(repository)
        summary = compiler.compile(request)
        return {
            "summary": summary.model_dump(mode="json"),
            "used": not summary.fallback_recommended,
            "debug": summary.debug.model_dump(mode="json"),
        }
    except Exception as exc:
        logger.warning("chat_stance_reasoning_summary_failed error=%s", exc)
        fallback = ReasoningSummaryV1(
            request_id=request.request_id,
            anchor_scope=request.anchor_scope,
            fallback_recommended=True,
        )
        return {
            "summary": fallback.model_dump(mode="json"),
            "used": False,
            "debug": {
                **fallback.debug.model_dump(mode="json"),
                "compiler_ran": True,
                "compiler_succeeded": False,
                "fallback_used": True,
            },
        }


def _mutation_cognition_from_ctx(ctx: Dict[str, Any]) -> Dict[str, Any]:
    direct = ctx.get("mutation_cognition_context")
    if isinstance(direct, dict):
        return direct
    metadata = ctx.get("metadata")
    if isinstance(metadata, dict):
        nested = metadata.get("mutation_cognition_context")
        if isinstance(nested, dict):
            return nested
    return {}


def _situation_summary_from_ctx(ctx: Dict[str, Any]) -> dict[str, Any]:
    fragment = ctx.get("situation_prompt_fragment") if isinstance(ctx.get("situation_prompt_fragment"), dict) else {}
    brief = ctx.get("situation_brief") if isinstance(ctx.get("situation_brief"), dict) else {}
    presence = brief.get("presence") if isinstance(brief.get("presence"), dict) else {}
    requestor = presence.get("requestor") if isinstance(presence.get("requestor"), dict) else {}
    companions = presence.get("companions") if isinstance(presence.get("companions"), list) else []
    phase = brief.get("conversation_phase") if isinstance(brief.get("conversation_phase"), dict) else {}
    time_ctx = brief.get("time") if isinstance(brief.get("time"), dict) else {}
    place = brief.get("place") if isinstance(brief.get("place"), dict) else {}
    environment = brief.get("environment") if isinstance(brief.get("environment"), dict) else {}
    lab = brief.get("lab") if isinstance(brief.get("lab"), dict) else {}
    surface = brief.get("surface") if isinstance(brief.get("surface"), dict) else {}
    affordances_raw = brief.get("affordances") if isinstance(brief.get("affordances"), list) else []
    affordances = []
    has_active = False
    for item in affordances_raw[:8]:
        if not isinstance(item, dict):
            continue
        trigger = _compact(item.get("trigger_relevance"), limit=16)
        has_active = has_active or trigger == "active"
        affordances.append(
            {
                "kind": _compact(item.get("kind"), limit=48),
                "trigger_relevance": trigger,
                "suggestion": _compact(item.get("suggestion"), limit=140),
                "confidence": _compact(item.get("confidence"), limit=16),
            }
        )
    companions_summary = []
    for item in companions[:4]:
        if not isinstance(item, dict):
            continue
        companions_summary.append(
            {
                "display_name": _compact(item.get("display_name"), limit=40),
                "role_hint": _compact(item.get("role_hint"), limit=40),
                "age_band": _compact(item.get("age_band"), limit=20),
            }
        )
    environment_summary = {
        "current_summary": _compact(
            ((environment.get("current_weather") or {}).get("condition") if isinstance(environment.get("current_weather"), dict) else ""),
            limit=120,
        ),
        "forecast_windows_summary": _unique(
            [
                _compact(((environment.get("forecast_next_2h") or {}).get("summary") if isinstance(environment.get("forecast_next_2h"), dict) else ""), limit=90),
                _compact(((environment.get("forecast_next_6h") or {}).get("summary") if isinstance(environment.get("forecast_next_6h"), dict) else ""), limit=90),
                _compact(((environment.get("forecast_next_24h") or {}).get("summary") if isinstance(environment.get("forecast_next_24h"), dict) else ""), limit=90),
            ],
            limit=3,
        ),
        "practical_flags": environment.get("practical_flags") if isinstance(environment.get("practical_flags"), dict) else {},
    }
    relevance = "active" if has_active else ("background" if fragment else "none")
    return {
        "situation_relevance": relevance,
        "situation_prompt_fragment": {
            "compact_text": _compact(fragment.get("compact_text"), limit=480),
            "should_mention": bool(fragment.get("should_mention", False)),
            "mention_policy": _compact(fragment.get("mention_policy"), limit=40),
            "summary_lines": _unique((fragment.get("summary_lines") or []), limit=6),
            "relevance_notes": _unique((fragment.get("relevance_notes") or []), limit=6),
            "caution_lines": _unique((fragment.get("caution_lines") or []), limit=6),
        },
        "presence": {
            "audience_mode": _compact(presence.get("audience_mode"), limit=32),
            "requestor": _compact(requestor.get("display_name"), limit=40),
            "companions": companions_summary,
            "privacy_mode": _compact(presence.get("privacy_mode"), limit=32),
            "persist_to_memory": bool(presence.get("persist_to_memory", False)),
        },
        "conversation_phase": {
            "phase_change": _compact(phase.get("phase_change"), limit=32),
            "continuity_mode": _compact(phase.get("continuity_mode"), limit=40),
            "topic_staleness_risk": _compact(phase.get("topic_staleness_risk"), limit=24),
            "time_since_last_user_turn_seconds": phase.get("time_since_last_user_turn_seconds"),
        },
        "time": {
            "local_datetime": _compact(time_ctx.get("local_datetime"), limit=40),
            "time_of_day_label": _compact(time_ctx.get("time_of_day_label"), limit=24),
            "day_phase": _compact(time_ctx.get("day_phase"), limit=24),
            "weekday": _compact(time_ctx.get("weekday"), limit=16),
        },
        "place": {
            "coarse_location": _compact(place.get("coarse_location"), limit=48),
            "locality": _compact(place.get("locality"), limit=32),
            "region": _compact(place.get("region"), limit=32),
        },
        "environment": environment_summary,
        "lab": {
            "available": bool(lab.get("available", False)),
            "thermal_risk": _compact(lab.get("thermal_risk"), limit=20),
            "power_risk": _compact(lab.get("power_risk"), limit=20),
            "summary": _compact(
                f"thermal={_compact(lab.get('thermal_risk'), limit=20)} power={_compact(lab.get('power_risk'), limit=20)}",
                limit=80,
            ),
        },
        "surface": {
            "surface": _compact(surface.get("surface"), limit=40),
            "input_modality": _compact(surface.get("input_modality"), limit=24),
        },
        "affordances": affordances,
    }


def _reasoning_upstream_nonempty(ctx: Dict[str, Any]) -> bool:
    raw = ctx.get("reasoning_artifacts")
    if isinstance(raw, list) and raw:
        return True
    repo = ctx.get("reasoning_repository")
    if repo is None:
        return False
    try:
        latest = repo.list_latest(limit=1)
        return bool(latest)
    except Exception:
        return False


async def _run_autonomy_reducer(
    ctx: Dict[str, Any],
    autonomy: Dict[str, Any],
    *,
    social: Dict[str, Any],
    social_bridge: Dict[str, Any],
    reasoning: Dict[str, Any],
):
    now = datetime.now(timezone.utc)
    compile_result = compile_autonomy_evidence(
        user_message=ctx.get("user_message") or ctx.get("message") or "",
        social=social,
        social_bridge=social_bridge,
        reasoning_summary=(reasoning.get("summary") if isinstance(reasoning, dict) else {}) or {},
        reasoning_upstream_nonempty=_reasoning_upstream_nonempty(ctx),
        autonomy_debug=autonomy.get("debug") if isinstance(autonomy.get("debug"), dict) else {},
        now=now,
    )
    ctx["chat_autonomy_evidence_debug"] = {
        "emitted_kinds": [e.kind for e in compile_result.evidence],
        "omitted": list(compile_result.omitted),
        **(compile_result.debug or {}),
    }

    state_obj = autonomy.get("state")
    subj = getattr(state_obj, "subject", None) if state_obj is not None else None
    subject = str(subj or "orion")

    # Close the reducer's own fold loop: prefer the reducer's own persisted
    # output over the V1/graph baseline so state carries turn-to-turn. Falls
    # back to the V1 baseline exactly as before when nothing is persisted yet
    # (first-ever turn for this subject, or the store is unreachable).
    persisted = await asyncio.to_thread(load_autonomy_state_v2, subject)
    previous_state = persisted if persisted is not None else state_obj

    # Snapshot before-pressures from the SAME baseline the fold actually uses
    # (persisted V2 state when present, else the V1/graph baseline) -- not from
    # autonomy["state"] directly, which can silently diverge from previous_state
    # once persistence is warm and would otherwise desync movement_debug's
    # before/after comparison from what the reducer really folded.
    before_pressures = (
        dict(getattr(previous_state, "drive_pressures", None) or {})
        if previous_state is not None
        else None
    )
    dominant_drive_before = getattr(previous_state, "dominant_drive", None) if previous_state is not None else None

    result = reduce_autonomy_state(
        AutonomyReducerInputV1(
            subject=subject,
            previous_state=previous_state,
            evidence=compile_result.evidence,
            action_outcomes=load_action_outcomes(subject=subject),
            now=now,
        )
    )
    # Single mint path: reuse what the reducer actually folded.
    ctx["chat_autonomy_tension_debug"] = {"minted": list(result.tensions_minted or [])}

    # Built here (not by the caller) so before/after always compare against the
    # SAME previous_state the fold actually used -- see before_pressures comment
    # above for why deriving "before" from autonomy["state"] independently would
    # silently desync once persistence is warm.
    ctx["chat_autonomy_movement_debug"] = {
        "dominant_drive_before": dominant_drive_before,
        "dominant_drive_after": result.state.dominant_drive,
        "pressures_before": before_pressures,
        "pressures_after": dict(result.state.drive_pressures or {}),
        "new_tensions": list(result.delta.new_tensions or []),
        "resolved_tensions": list(result.delta.resolved_tensions or []),
    }

    # Belt-and-suspenders fail-open write-back: save_autonomy_state_v2 already
    # never raises, but this is a hot chat-turn path, so guard the call site
    # too rather than depend solely on the callee's contract.
    try:
        await asyncio.to_thread(save_autonomy_state_v2, subject, result.state)
    except Exception as exc:
        logger.warning("autonomy_state_v2_write_failed subject=%s error=%s", subject, exc)

    _log_autonomy_pressure_probe(subject, dict(result.state.drive_pressures or {}))

    return result


def _log_autonomy_pressure_probe(subject: str, pressures: dict) -> None:
    """Measurement-only (Phase 4, 2026-07-12): log `AutonomyStateV2`'s pressure
    vector right after every `_run_autonomy_reducer` fold so it can be compared
    offline against `DriveEngine`'s independently-computed pressures (logged
    from `orion.spark.concept_induction.bus_worker`) by grepping both
    services' logs and correlating on `subject` + nearby timestamp. Never
    raises: this is a hot chat-turn path and a logging failure here must not
    break it, mirroring the guard around the `save_autonomy_state_v2` call
    directly above.
    """
    try:
        logger.info(
            "autonomy_state_v2_pressure_probe subject=%s pressures=%s",
            subject,
            {k: round(v, 4) for k, v in pressures.items()},
        )
    except Exception as exc:
        logger.warning("autonomy_state_v2_pressure_probe_failed subject=%s error=%s", subject, exc)


def _inject_prior_stance_to_inputs(ctx: Dict[str, Any], inputs: Dict[str, Any]) -> None:
    """Copy prior brief summary into stance inputs and expose it as a TOP-LEVEL ctx
    key for the chat_stance_brief.j2 render (which uses ctx.copy()), when present and non-empty."""
    prior = ctx.get("prior_chat_stance_brief")
    if isinstance(prior, dict) and prior:
        inputs["prior_stance"] = prior
        ctx["prior_stance"] = prior


async def build_chat_stance_inputs(ctx: Dict[str, Any]) -> Dict[str, Any]:
    # Single unified beliefs call replaces independent producer fan-outs for
    # identity, orionmem, recall, and social lanes.
    from app.substrate_felt_state_reader import hydrate_felt_state_ctx
    hydrate_felt_state_ctx(ctx)
    beliefs = _unified_beliefs_for_stance(ctx)

    identity = _project_identity_from_beliefs(beliefs, ctx)
    ctx.update(identity)
    concept = _project_concept_from_beliefs(beliefs, ctx) or _concept_summary_from_store(ctx)
    social, social_bridge = _project_social_from_beliefs(beliefs, ctx)
    social["social_posture"] = _unique((social.get("social_posture") or []) + (social_bridge.get("posture") or []), limit=8)
    social["hazards"] = _unique((social.get("hazards") or []) + (social_bridge.get("hazards") or []), limit=8)
    social["relationship_facets"] = _unique((social.get("relationship_facets") or []) + (social_bridge.get("framing") or []), limit=8)
    reflective = _project_recall_from_beliefs(beliefs, ctx)
    situation = _situation_summary_from_ctx(ctx)
    reasoning = _compile_reasoning_summary(ctx)
    ctx["chat_reasoning_summary"] = reasoning["summary"]
    autonomy = _project_autonomy_from_beliefs(beliefs, ctx) or _load_autonomy_state(ctx)
    self_state_projection = _project_self_state_from_beliefs(beliefs, ctx)
    context_provenance_hazard = _project_context_provenance_hazard(ctx)
    # self_state severity and context-provenance are both standing epistemic/
    # safety signals from this function's own reasoning, not reactive social
    # hazards -- fold them in together, prepended ahead of the social/
    # social_bridge hazards already in the list, so _unique(..., limit=8)'s
    # truncation-in-order falls on the lower-stakes social hazards first
    # instead of silently evicting one safety signal to make room for the
    # other (a prior version prepended context_provenance_hazard alone,
    # which could evict an already-folded self_state severity hazard once
    # the list was full).
    priority_hazards: list[str] = []
    if self_state_projection:
        priority_hazards.extend(self_state_projection.get("hazards") or [])
        ctx["chat_self_state_condition"] = self_state_projection.get("overall_condition")
    if context_provenance_hazard:
        priority_hazards.append(context_provenance_hazard)
    if priority_hazards:
        social["hazards"] = _unique(priority_hazards + list(social.get("hazards") or []), limit=8)
    reverie_glimpse = _project_reverie_glimpse(ctx)
    if reverie_glimpse:
        ctx["chat_reverie_glimpse"] = reverie_glimpse
    mutation_cognition = _mutation_cognition_from_ctx(ctx)
    social["hazards"] = _unique((social.get("hazards") or []) + list((reasoning.get("summary") or {}).get("hazards") or []), limit=8)

    # Log unified beliefs diagnostics (quiet warm path; louder when cold or degraded)
    if beliefs is not None:
        payload = json.dumps(
            {
                "cold_anchors": beliefs.cold_anchors,
                "degraded_producers": beliefs.degraded_producers,
                "lineage": beliefs.lineage,
            },
            sort_keys=True,
        )
        if beliefs.cold_anchors or beliefs.degraded_producers:
            logger.info("unified_beliefs_for_stance %s", payload)
        else:
            logger.debug("unified_beliefs_for_stance %s", payload)
    ctx["chat_unified_beliefs_lineage"] = beliefs.lineage if beliefs is not None else []
    ctx["chat_unified_beliefs_degraded"] = beliefs.degraded_producers if beliefs is not None else []

    inputs = {
        "identity": {
            "orion": list(identity["orion_identity_summary"]),
            "juniper": list(identity["juniper_relationship_summary"]),
            "response_policy": list(identity["response_policy_summary"]),
        },
        "concept_induction": concept,
        "social_bridge": social_bridge,
        "social": social,
        "reflective": reflective,
        "autonomy": {
            "state": autonomy["state"].model_dump(mode="json") if autonomy["state"] is not None else None,
            "summary": autonomy["summary"].model_dump(mode="json"),
            "debug": autonomy["debug"],
        },
        "reasoning_summary": reasoning["summary"],
        "situation": situation,
        "mutation_adaptation": mutation_cognition,
    }
    mg_hints = _project_memory_graph_hints_from_beliefs(beliefs) or fetch_chat_stance_memory_graph_hints()
    if mg_hints:
        inputs["memory_graph"] = {"disposition_hints": mg_hints}

    # drive_state.v1 visibility — a sibling of `inputs["autonomy"]`, never merged
    # into it. drive_state and autonomy_state_v2 are independently-computed
    # signals per orion/self_state/inner_state_registry.py's DUPLICATE note.
    # Off by default; flip CHAT_STANCE_DRIVE_STATE_VISIBLE=true to surface it.
    if os.getenv("CHAT_STANCE_DRIVE_STATE_VISIBLE", "").strip().lower() == "true":
        drive_state_projection = autonomy.get("drive_state") if isinstance(autonomy, dict) else None
        if drive_state_projection:
            inputs["drive_state"] = drive_state_projection

    # Queries load_action_outcomes(subject="orion") directly (see
    # _project_recent_dispatch_actions' docstring) rather than reading ctx, so
    # it is unconditional -- independent of AUTONOMY_STATE_V2_REDUCER_ENABLED
    # below. Computed here (before that gated block) so build_autonomy_slice()
    # can fold it into recent_actions regardless of the reducer's flag/success,
    # while chat_general.j2's direct read of ctx["chat_recent_dispatch_actions"]
    # keeps working unchanged either way. Fail-open: [] on any failure.
    ctx["chat_recent_dispatch_actions"] = _project_recent_dispatch_actions(ctx)

    if os.getenv("AUTONOMY_STATE_V2_REDUCER_ENABLED", "").strip().lower() == "true":
        try:
            v2_result = await _run_autonomy_reducer(
                ctx,
                autonomy,
                social=social,
                social_bridge=social_bridge,
                reasoning=reasoning,
            )
            # _run_autonomy_reducer already set chat_autonomy_movement_debug on
            # ctx itself (before/after pressures compared against the same
            # previous_state the fold actually used); state/delta are set here.
            ctx["chat_autonomy_state_v2"] = v2_result.state.model_dump(mode="json")
            ctx["chat_autonomy_state_delta"] = v2_result.delta.model_dump(mode="json")
            inputs["autonomy"]["state_v2"] = ctx["chat_autonomy_state_v2"]
            inputs["autonomy"]["delta"] = ctx["chat_autonomy_state_delta"]
        except Exception as exc:
            logger.warning("autonomy_reducer_v2_failed error=%s", exc)

    # Built here (ctx key "autonomy_slice", matching what stance_react.j2 reads
    # directly) so it's present BEFORE the stance_react LLM step renders its
    # prompt. router.py's post-hoc metadata attach (for the harness-prefix/
    # ThoughtEventV1 path) reads this same ctx key later in the same turn -- it
    # does not need to recompute it.
    #
    # Deliberately OUTSIDE the AUTONOMY_STATE_V2_REDUCER_ENABLED gate and its
    # try/except above: recent_actions (real Layer-9 dispatch evidence) has
    # nothing to do with the V2 reducer's health, so it must not go dark just
    # because the reducer is disabled or threw. build_autonomy_slice() already
    # treats an empty/missing chat_autonomy_state_v2 as "no drive/tension
    # signal" rather than raising -- calling it unconditionally is safe and
    # was the actual intent stated in this block's own comment above (recent
    # actions are "independent of AUTONOMY_STATE_V2_REDUCER_ENABLED"); a prior
    # version of this code contradicted that by nesting the call inside the
    # gate anyway. Own try/except so a genuinely unexpected failure here can
    # never take out the reducer block above it (or vice versa).
    #
    # max_recent_actions is passed explicitly (this file's own
    # _MAX_RECENT_DISPATCH_ACTIONS) so the cap has one source of truth rather
    # than a second hardcoded "3" living inside autonomy_slice.py.
    try:
        autonomy_slice = build_autonomy_slice(ctx, max_recent_actions=_MAX_RECENT_DISPATCH_ACTIONS)
        if autonomy_slice is not None:
            ctx["autonomy_slice"] = autonomy_slice.model_dump(mode="json")
    except Exception as exc:
        logger.warning("autonomy_slice_build_failed error=%s", exc)

    if attention_frame_enabled():
        try:
            attention_frame = build_attention_frame(
                ctx=ctx,
                inputs=inputs,
                belief_lineage=(beliefs.lineage if beliefs is not None else []),
            )
            attention_frame_payload = attention_frame.model_dump(mode="json")
            inputs["attention_frame"] = attention_frame_payload
            ctx["chat_attention_frame"] = attention_frame_payload
            ctx["chat_attention_frame_debug"] = {
                "open_loop_count": len(attention_frame.open_loops),
                "selected_action": (attention_frame.selected_action.model_dump(mode="json") if attention_frame.selected_action else None),
                "suppression_reasons": [s.reason for s in attention_frame.suppressions],
            }
        except Exception as exc:
            logger.warning("attention_frame_build_failed error=%s", exc)
            ctx.pop("chat_attention_frame", None)
            ctx.pop("chat_attention_frame_debug", None)

    _inject_prior_stance_to_inputs(ctx, inputs)
    continuity_digest = ctx.get("continuity_digest")
    if not isinstance(continuity_digest, str):
        continuity_digest = ""
    ctx["continuity_digest"] = continuity_digest
    inputs["continuity_digest"] = continuity_digest
    ctx["chat_stance_inputs"] = inputs
    ctx["chat_concept_summary"] = concept
    ctx["chat_social_summary"] = social
    ctx["chat_social_bridge_summary"] = social_bridge
    ctx["chat_reflective_summary"] = reflective
    ctx["chat_autonomy_state"] = autonomy["state"].model_dump(mode="json") if autonomy["state"] is not None else None
    ctx["chat_autonomy_summary"] = autonomy["summary"].model_dump(mode="json")
    ctx["chat_autonomy_debug"] = autonomy["debug"]
    ctx["chat_autonomy_backend"] = autonomy["backend"]
    ctx["chat_autonomy_selected_subject"] = autonomy["selected_subject"]
    ctx["chat_autonomy_repository_status"] = autonomy["repository_status"]
    ctx["chat_reasoning_summary"] = reasoning["summary"]
    ctx["chat_reasoning_debug"] = reasoning["debug"]
    ctx["chat_reasoning_summary_used"] = reasoning["used"]
    ctx["chat_situation_summary"] = situation
    ctx["chat_mutation_cognition_context"] = mutation_cognition
    try:
        ctx["chat_endogenous_runtime"] = runtime_service().maybe_invoke(
            ctx=ctx,
            reasoning_summary=reasoning["summary"],
            reflective=reflective,
            autonomy=inputs["autonomy"],
            concept=concept,
        )
    except Exception as exc:
        logger.warning("chat_stance_endogenous_runtime_invoke_failed error=%s", exc)
        ctx["chat_endogenous_runtime"] = None
    if ctx.get("endogenous_runtime_operator_review"):
        try:
            ctx["chat_endogenous_runtime_recent"] = inspect_endogenous_runtime_records(limit=8)
            ctx["chat_endogenous_runtime_reflective_feed"] = consume_endogenous_runtime_for_reflective_review(
                query=EndogenousRuntimeQueryV1(limit=6, invocation_surface="chat_reflective_lane")
            )
        except Exception as exc:
            logger.warning("chat_stance_endogenous_runtime_recent_failed error=%s", exc)
    return inputs


def parse_chat_stance_brief(raw_text: str) -> ChatStanceBrief | None:
    parsed, _ = parse_chat_stance_brief_with_debug(raw_text)
    return parsed


def parse_chat_stance_brief_with_debug(raw_text: str) -> tuple[ChatStanceBrief | None, dict[str, Any]]:
    info: dict[str, Any] = {"normalized_applied": False, "coercion_applied": False}
    obj = _extract_json_object(raw_text)
    if not obj:
        info["parse_error"] = "missing_json_object"
        return None, info
    coerced = _coerce_stance_brief_obj(obj)
    try:
        info["coercion_applied"] = json.dumps(obj, sort_keys=True) != json.dumps(coerced, sort_keys=True)
    except Exception:
        info["coercion_applied"] = coerced != obj
    try:
        parsed = ChatStanceBrief.model_validate(coerced)
        normalized = normalize_chat_stance_brief(parsed)
        parsed_json = parsed.model_dump(mode="json")
        normalized_json = normalized.model_dump(mode="json")
        info["normalized_applied"] = parsed_json != normalized_json
        info["parsed_brief"] = parsed_json
        return normalized, info
    except Exception as exc:
        info["parse_error"] = str(exc)
        return None, info


def build_chat_stance_debug_payload(
    *,
    ctx: Dict[str, Any],
    synthesized_brief: Dict[str, Any],
    final_brief: Dict[str, Any],
    fallback_invoked: bool,
    normalized_applied: bool,
    semantic_fallback: bool,
    quality_modified: bool,
    parse_error: str | None = None,
) -> Dict[str, Any]:
    stance_inputs = ctx.get("chat_stance_inputs") if isinstance(ctx.get("chat_stance_inputs"), dict) else {}
    concept = stance_inputs.get("concept_induction") if isinstance(stance_inputs.get("concept_induction"), dict) else {}
    social = stance_inputs.get("social") if isinstance(stance_inputs.get("social"), dict) else {}
    social_bridge = stance_inputs.get("social_bridge") if isinstance(stance_inputs.get("social_bridge"), dict) else {}
    reflective = stance_inputs.get("reflective") if isinstance(stance_inputs.get("reflective"), dict) else {}
    autonomy = stance_inputs.get("autonomy") if isinstance(stance_inputs.get("autonomy"), dict) else {}
    mutation_adaptation = (
        stance_inputs.get("mutation_adaptation")
        if isinstance(stance_inputs.get("mutation_adaptation"), dict)
        else {}
    )
    autonomy_summary = autonomy.get("summary") if isinstance(autonomy.get("summary"), dict) else {}
    attention_frame = stance_inputs.get("attention_frame") if isinstance(stance_inputs.get("attention_frame"), dict) else {}
    autonomy_debug = autonomy.get("debug") if isinstance(autonomy.get("debug"), dict) else {}
    reasoning = stance_inputs.get("reasoning_summary") if isinstance(stance_inputs.get("reasoning_summary"), dict) else {}
    situation = stance_inputs.get("situation") if isinstance(stance_inputs.get("situation"), dict) else {}
    identity = stance_inputs.get("identity") if isinstance(stance_inputs.get("identity"), dict) else {}
    journal_pageindex_ctx = ctx.get("journal_pageindex_context") if isinstance(ctx.get("journal_pageindex_context"), dict) else {}
    jp_selected_entries = journal_pageindex_ctx.get("selected_entries") if isinstance(journal_pageindex_ctx.get("selected_entries"), list) else []
    jp_selected_blocks = journal_pageindex_ctx.get("selected_blocks") if isinstance(journal_pageindex_ctx.get("selected_blocks"), list) else []
    jp_entry_ids = [
        str(item.get("entry_id"))
        for item in jp_selected_entries
        if isinstance(item, dict) and str(item.get("entry_id") or "").strip()
    ]
    jp_block_ids = [
        str(item.get("block_id"))
        for item in jp_selected_blocks
        if isinstance(item, dict) and str(item.get("block_id") or "").strip()
    ]
    jp_impl = None
    for block in jp_selected_blocks:
        if not isinstance(block, dict):
            continue
        provenance = block.get("provenance") if isinstance(block.get("provenance"), dict) else {}
        engine = str(provenance.get("engine") or "").strip()
        if engine:
            jp_impl = engine
            break
    if not jp_impl and journal_pageindex_ctx:
        jp_impl = "native_fallback" if bool(journal_pageindex_ctx.get("fallback_invoked")) else "service"
    jp_evidence_lines = _unique(
        (
            _compact(
                (item.get("excerpt") if isinstance(item, dict) else "")
                or (item.get("text") if isinstance(item, dict) else ""),
                limit=180,
            )
            for item in jp_selected_blocks[:8]
        ),
        limit=6,
    )

    user_message = _compact(ctx.get("user_message") or "", limit=600)
    continuity_digest = ctx.get("continuity_digest")
    if not isinstance(continuity_digest, str):
        continuity_digest = ""
    belief_digest = ctx.get("belief_digest")
    if not isinstance(belief_digest, str):
        belief_digest = ""
    memory_digest = ctx.get("memory_digest")
    if not isinstance(memory_digest, str):
        memory_digest = ""
    categories_present = sorted(
        [
            key
            for key, value in {
                "identity": identity,
                "concept_induction": concept,
                "social": social,
                "social_bridge": social_bridge,
                "reflective": reflective,
                "journal_pageindex": {"impl": jp_impl, "entry_ids": jp_entry_ids, "evidence_lines": jp_evidence_lines},
                "autonomy": autonomy_summary,
                "reasoning": reasoning,
                "situation": situation,
                "attention_frame": attention_frame,
            }.items()
            if isinstance(value, dict) and any(value.values())
        ]
    )

    final_prompt_contract = {
        "chat_stance_brief": final_brief,
        "continuity_digest": continuity_digest,
        "belief_digest": belief_digest,
        "memory_digest": memory_digest,
        "orion_identity_summary": list(ctx.get("orion_identity_summary") or []),
        "juniper_relationship_summary": list(ctx.get("juniper_relationship_summary") or []),
        "response_policy_summary": list(ctx.get("response_policy_summary") or []),
        "attention_frame": attention_frame if attention_frame else None,
        "situation_prompt_fragment": ctx.get("situation_prompt_fragment") if isinstance(ctx.get("situation_prompt_fragment"), dict) else None,
        "presence_context": ctx.get("presence_context") if isinstance(ctx.get("presence_context"), dict) else None,
        "situation_presence": ((ctx.get("situation_brief") or {}).get("presence") if isinstance(ctx.get("situation_brief"), dict) else None),
    }

    notes: list[str] = []
    if fallback_invoked:
        notes.append("fallback_chat_stance_brief invoked")
    if semantic_fallback:
        notes.append("semantic quality guard modified/replaced synthesized brief")
    if quality_modified:
        notes.append("final brief differs from synthesized brief after enforcement")
    if normalized_applied:
        notes.append("brief normalization compacted or deduplicated generated fields")
    if parse_error:
        notes.append(f"parse_error: {parse_error}")
    if not notes:
        notes.append("no enforcement deltas")

    return {
        "overview": {
            "categories_present": categories_present,
            "fallback_invoked": fallback_invoked,
            "normalized_applied": normalized_applied,
            "quality_enforcement_modified": quality_modified,
            "semantic_fallback": semantic_fallback,
        },
        "source_inputs": {
            "user_message": user_message,
            "continuity_digest": continuity_digest,
            "belief_digest": belief_digest,
            "memory_digest": memory_digest,
            "identity_kernel": {
                "orion_identity_summary": list(identity.get("orion") or []),
                "juniper_relationship_summary": list(identity.get("juniper") or []),
                "response_policy_summary": list(identity.get("response_policy") or []),
            },
            "concept_induction": {
                "self": list(concept.get("self") or []),
                "relationship": list(concept.get("relationship") or []),
                "growth": list(concept.get("growth") or []),
                "tension": list(concept.get("tension") or []),
            },
            "social": {
                "social_posture": list(social.get("social_posture") or []),
                "relationship_facets": list(social.get("relationship_facets") or []),
                "hazards": list(social.get("hazards") or []),
            },
            "social_bridge": {
                "posture": list(social_bridge.get("posture") or []),
                "hazards": list(social_bridge.get("hazards") or []),
                "framing": list(social_bridge.get("framing") or []),
                "turn_decision": social_bridge.get("turn_decision"),
                "summary": list(social_bridge.get("summary") or []),
            },
            "reflective": {
                "themes": list(reflective.get("themes") or []),
                "tensions": list(reflective.get("tensions") or []),
                "dream_motifs": list(reflective.get("dream_motifs") or []),
            },
            "journal_pageindex": {
                "context_present": bool(journal_pageindex_ctx),
                "impl": jp_impl,
                "fallback_invoked": bool(journal_pageindex_ctx.get("fallback_invoked")) if journal_pageindex_ctx else False,
                "selected_entry_count": len(jp_entry_ids),
                "selected_block_count": len(jp_block_ids),
                "selected_entry_ids": jp_entry_ids[:8],
                "selected_block_ids": jp_block_ids[:12],
                "evidence_lines": jp_evidence_lines,
            },
            "autonomy": {
                "summary": autonomy_summary,
                "selected_subject": ctx.get("chat_autonomy_selected_subject"),
                "backend": ctx.get("chat_autonomy_backend"),
                "compact_debug": autonomy_debug,
            },
                "mutation_adaptation": mutation_adaptation,
            "reasoning": {
                "summary": reasoning,
                "hazards": list(reasoning.get("hazards") or []),
                "tensions": list(reasoning.get("tensions") or []),
                "fallback_recommended": bool(reasoning.get("fallback_recommended")),
                "used": bool(ctx.get("chat_reasoning_summary_used")),
            },
            "attention_frame": {
                "open_loops": list(attention_frame.get("open_loops") or []),
                "live_unknowns": list(attention_frame.get("live_unknowns") or []),
                "selected_action": attention_frame.get("selected_action"),
                "suppressions": list(attention_frame.get("suppressions") or []),
                "deferred_items": list(attention_frame.get("deferred_items") or []),
                "debug": attention_frame.get("debug") if isinstance(attention_frame.get("debug"), dict) else {},
            },
            "situation": {
                "situation_relevance": situation.get("situation_relevance"),
                "situation_prompt_fragment": situation.get("situation_prompt_fragment") if isinstance(situation.get("situation_prompt_fragment"), dict) else {},
                "presence": situation.get("presence") if isinstance(situation.get("presence"), dict) else {},
                "conversation_phase": situation.get("conversation_phase") if isinstance(situation.get("conversation_phase"), dict) else {},
                "time": situation.get("time") if isinstance(situation.get("time"), dict) else {},
                "place": situation.get("place") if isinstance(situation.get("place"), dict) else {},
                "environment": situation.get("environment") if isinstance(situation.get("environment"), dict) else {},
                "lab": situation.get("lab") if isinstance(situation.get("lab"), dict) else {},
                "surface": situation.get("surface") if isinstance(situation.get("surface"), dict) else {},
                "affordances": list(situation.get("affordances") or []),
            },
        },
        "synthesized_brief": synthesized_brief,
        "enforcement": {
            "fallback_invoked": fallback_invoked,
            "normalized_applied": normalized_applied,
            "quality_modified": quality_modified,
            "semantic_fallback": semantic_fallback,
            "notes": notes,
        },
        "final_prompt_contract": final_prompt_contract,
        "lineage_summary": [
            f"concept/self injected: {len((concept.get('self') or []))} items",
            f"social hazards injected: {len((social.get('hazards') or []))} items",
            f"journal pageindex context present: {'yes' if bool(journal_pageindex_ctx) else 'no'}",
            f"journal pageindex selected entries: {len(jp_entry_ids)}",
            f"autonomy summary present: {'yes' if bool(autonomy_summary) else 'no'}",
            f"mutation adaptation context present: {'yes' if bool(mutation_adaptation) else 'no'}",
            f"reasoning summary used: {'yes' if bool(ctx.get('chat_reasoning_summary_used')) else 'no'}",
            f"situation context present: {'yes' if bool(situation) else 'no'}",
            f"attention frame present: {'yes' if bool(attention_frame) else 'no'}",
            f"fallback applied: {'yes' if fallback_invoked or semantic_fallback else 'no'}",
        ],
        "raw": {
            "source_inputs": stance_inputs,
            "synthesized_brief": synthesized_brief,
            "final_prompt_contract": final_prompt_contract,
            "enforcement": {
                "fallback_invoked": fallback_invoked,
                "normalized_applied": normalized_applied,
                "semantic_fallback": semantic_fallback,
                "quality_modified": quality_modified,
                "parse_error": parse_error,
            },
        },
    }


def fallback_chat_stance_brief(ctx: Dict[str, Any]) -> ChatStanceBrief:
    user_message = _compact(ctx.get("user_message") or "", limit=220)
    identity = identity_kernel_with_fallbacks(ctx)
    concept = ctx.get("chat_concept_summary") if isinstance(ctx.get("chat_concept_summary"), dict) else {}
    social = ctx.get("chat_social_summary") if isinstance(ctx.get("chat_social_summary"), dict) else {}
    social_bridge = ctx.get("chat_social_bridge_summary") if isinstance(ctx.get("chat_social_bridge_summary"), dict) else {}
    reflective = ctx.get("chat_reflective_summary") if isinstance(ctx.get("chat_reflective_summary"), dict) else {}
    autonomy_summary = ctx.get("chat_autonomy_summary") if isinstance(ctx.get("chat_autonomy_summary"), dict) else {}
    reasoning_summary = ctx.get("chat_reasoning_summary") if isinstance(ctx.get("chat_reasoning_summary"), dict) else {}
    situation = _situation_summary_from_ctx(ctx)
    identity_turn = _is_identity_sensitive_turn(user_message)
    social_posture = list(social.get("social_posture") or [])
    bridge_posture = list(social_bridge.get("posture") or [])
    technical_turn = "technical" in social_posture or "technical" in bridge_posture or any(
        re.search(pattern, user_message.lower()) for pattern in _TECHNICAL_TURN_PATTERNS
    )
    strained_turn = "operational_frustration" in bridge_posture or "strained" in bridge_posture
    task_mode = "identity_dialogue" if identity_turn else ("triage" if technical_turn and strained_turn else ("technical_collaboration" if technical_turn else "direct_response"))
    identity_salience = "high" if identity_turn else "low"
    if identity_turn:
        active_identity = list(concept.get("self") or [])[:3] + identity["orion_identity_summary"][:4]
        active_relationship = list(social.get("relationship_facets") or [])[:3] + identity["juniper_relationship_summary"][:4]
    else:
        active_identity = list(concept.get("self") or [])[:3]
        active_relationship = list(social.get("relationship_facets") or [])[:3]
    if reasoning_summary and not reasoning_summary.get("fallback_recommended"):
        reasoning_claims = [item for item in (reasoning_summary.get("active_claims") or []) if isinstance(item, dict)]
        reasoning_concepts = [item for item in (reasoning_summary.get("active_concepts") or []) if isinstance(item, dict)]
        active_identity = active_identity + [str(item.get("claim_text") or "") for item in reasoning_claims[:2]]
        active_identity = active_identity + [str(item.get("label") or "") for item in reasoning_concepts[:2]]
        active_relationship = active_relationship + list(reasoning_summary.get("relationship_signals") or [])[:2]
    if identity_turn:
        response_priorities = [
            "answer_identity_directly",
            "preserve_relational_continuity",
            "use_first_person_orion",
            "avoid_generic_assistant_tone",
        ]
    elif task_mode == "triage":
        response_priorities = [
            "triage_operational_blockers_first",
            "acknowledge_frustration_groundedly",
            "summarize_operational_impact",
            "propose_next_debug_steps",
        ]
    else:
        response_priorities = [
            "answer_directly_first",
            "avoid_identity_recital",
            "preserve_continuity_without_labels",
            "avoid_generic_assistant_tone",
            "maintain_grounded_specificity",
        ]
    situation_guidance = list((situation.get("situation_prompt_fragment") or {}).get("relevance_notes") or [])[:4]
    situation_relevance = str(situation.get("situation_relevance") or "none")
    if situation_relevance not in {"none", "background", "active"}:
        situation_relevance = "none"
    if situation_relevance == "active":
        response_priorities = _unique(
            response_priorities
            + [
                "situation:adjust audience style for presence context",
                "situation:apply active affordance only when relevant",
            ]
            + situation_guidance,
            limit=8,
        )
    autonomy_hint = _compact(autonomy_summary.get("stance_hint") or "", limit=90)
    if autonomy_hint and task_mode != "triage":
        response_priorities = _unique(response_priorities + [f"autonomy:{autonomy_hint}"], limit=8)
    threshold = float(os.getenv("GOAL_HINT_PRIORITY_THRESHOLD", "0.4"))
    active_goals = [g for g in (autonomy_summary.get("active_goals") or []) if isinstance(g, dict)]
    if active_goals and task_mode != "triage":
        top = sorted(active_goals, key=lambda g: -float(g.get("priority") or 0))[0]
        if float(top.get("priority") or 0) >= threshold:
            headline = _compact(str(top.get("headline") or ""), limit=80)
            if headline:
                response_priorities = _unique(
                    response_priorities + [f"goal_hint:{headline}"], limit=8
                )
                ctx["chat_autonomy_execution_mode"] = "hint_only"
    response_hazards = [
        "generic assistant self-description",
        "describing Juniper as just the user",
        "customer-support tone",
        "over-clarification",
    ]
    if not identity_turn:
        response_hazards.append("identity_recital_on_ordinary_turn")
    if task_mode == "triage":
        response_hazards.extend(["self_intro_on_operational_turn", "relationship_label_recital_during_triage"])
    if situation_relevance in {"active", "background"}:
        response_hazards.extend(
            [
                "do not force irrelevant time/weather commentary",
                "do not expose private operator context to broader audience",
            ]
        )
    response_hazards = _unique(
        response_hazards
        + list(autonomy_summary.get("response_hazards") or [])
        + list(reasoning_summary.get("hazards") or []),
        limit=8,
    )
    answer_strategy = (
        "DirectIdentityAnswer"
        if identity_turn
        else ("TriageFirstOperationalReply" if task_mode == "triage" else "DirectAnswer")
    )

    return normalize_chat_stance_brief(ChatStanceBrief(
        conversation_frame="identity_emergence" if identity_turn else "mixed",
        task_mode=task_mode,
        identity_salience=identity_salience,
        user_intent=user_message or "Respond directly to Juniper's latest request.",
        self_relevance=(
            "Maintain Oríon identity framing for this identity question."
            if identity_turn
            else "Answer the latest message directly without identity preamble."
        ),
        juniper_relevance=(
            "Anchor Juniper relationship continuity for this identity question."
            if identity_turn
            else "Prioritize practical usefulness over relationship labels."
        ),
        active_identity_facets=_unique(active_identity, limit=6) if identity_salience != "low" else [],
        active_growth_axes=list(concept.get("growth") or [])[:5],
        active_relationship_facets=_unique(active_relationship, limit=6) if identity_salience != "low" else [],
        social_posture=list(social.get("social_posture") or [])[:5],
        reflective_themes=list(reflective.get("themes") or [])[:4],
        active_tensions=_unique(
            list(reflective.get("tensions") or list(concept.get("tension") or []))
            + list(reasoning_summary.get("tensions") or []),
            limit=6,
        )[:4],
        dream_motifs=list(reflective.get("dream_motifs") or [])[:3],
        response_priorities=response_priorities,
        response_hazards=_unique(response_hazards + list(social.get("hazards") or []) + list(social_bridge.get("hazards") or []), limit=8),
        situation_relevance=situation_relevance,  # type: ignore[arg-type]
        temporal_context=_compact((situation.get("conversation_phase") or {}).get("phase_change"), limit=40) or None,
        audience_context=_compact((situation.get("presence") or {}).get("audience_mode"), limit=60) or None,
        environmental_context=_compact((situation.get("environment") or {}).get("current_summary"), limit=120) or None,
        operational_context=_compact((situation.get("lab") or {}).get("summary"), limit=120) or None,
        situation_response_guidance=_unique(situation_guidance, limit=6),
        answer_strategy=answer_strategy,
        stance_summary=(
            "Answer identity questions directly as Oríon and anchor Juniper relationship continuity."
            if identity_turn
            else (
                "Lead with operational triage and suppress identity foregrounding during frustration."
                if task_mode == "triage"
                else "Use bounded stance synthesis and deliver one direct useful response."
            )
        ),
    ))


def enforce_chat_stance_quality(brief: ChatStanceBrief, ctx: Dict[str, Any]) -> tuple[ChatStanceBrief, bool]:
    user_message = _compact(ctx.get("user_message") or "", limit=220)
    identity_turn = _is_identity_sensitive_turn(user_message)
    merged = brief.model_copy(deep=True)
    semantic_fallback = False
    fallback_identity = identity_kernel_with_fallbacks(ctx)

    weak_identity = identity_turn and not merged.active_identity_facets
    weak_relationship = identity_turn and not merged.active_relationship_facets
    weak_priorities = identity_turn and not merged.response_priorities
    generic_stance = any(marker in (merged.stance_summary or "").lower() for marker in _GENERIC_ASSISTANT_MARKERS)
    generic_strategy = any(marker in (merged.answer_strategy or "").lower() for marker in _GENERIC_ASSISTANT_MARKERS)

    if weak_identity:
        merged.active_identity_facets = list(fallback_identity["orion_identity_summary"])
        semantic_fallback = True
    if weak_relationship:
        merged.active_relationship_facets = list(fallback_identity["juniper_relationship_summary"])
        semantic_fallback = True
    if weak_priorities:
        merged.response_priorities = list(FALLBACK_RESPONSE_POLICY_SUMMARY)
        semantic_fallback = True
    if identity_turn and (generic_stance or generic_strategy):
        merged = fallback_chat_stance_brief(ctx)
        semantic_fallback = True
    if merged.task_mode == "triage":
        merged.identity_salience = "low"
        merged.response_hazards = _unique(
            list(merged.response_hazards) + ["self_intro_on_operational_turn", "generic_sympathy_script"],
            limit=8,
        )
        if "triage_operational_blockers_first" not in merged.response_priorities:
            merged.response_priorities = _unique(
                ["triage_operational_blockers_first"] + list(merged.response_priorities),
                limit=8,
            )

    if _continuation_of_relational_thread(ctx) and not _is_relational_stance_brief(merged):
        merged = _upgrade_brief_for_relational_continuation(merged)

    if not identity_turn and not _is_relational_stance_brief(merged) and not _continuation_of_relational_thread(ctx):
        if merged.task_mode != "identity_dialogue":
            merged.identity_salience = "low"
        _fallback_identity_boilerplate = frozenset(
            {"continuity", "juniper_builder", "known_person", "avoid_generic_assistant"}
        )
        def _is_identity_boilerplate_facet(facet: str) -> bool:
            normalized = _normalize_brief_phrase(facet)
            if normalized in _fallback_identity_boilerplate:
                return True
            lowered = normalized.lower()
            return any(token in lowered for token in ("orion", "oríon", "juniper"))

        merged.active_identity_facets = [
            f for f in merged.active_identity_facets if not _is_identity_boilerplate_facet(f)
        ]
        merged.active_relationship_facets = [
            f for f in merged.active_relationship_facets if not _is_identity_boilerplate_facet(f)
        ]
        if merged.identity_salience == "low":
            merged.active_identity_facets = []
            merged.active_relationship_facets = []
        merged.self_relevance = "Answer the latest message directly without identity preamble."
        merged.juniper_relevance = "Prioritize practical usefulness over relationship labels."
        merged.response_priorities = _unique(
            list(merged.response_priorities)
            + ["avoid_identity_recital", "preserve_continuity_without_labels"],
            limit=8,
        )
        merged.response_hazards = _unique(
            list(merged.response_hazards) + ["identity_recital_on_ordinary_turn"],
            limit=8,
        )

    return normalize_chat_stance_brief(merged), semantic_fallback
