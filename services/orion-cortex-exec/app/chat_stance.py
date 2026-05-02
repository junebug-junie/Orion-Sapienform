from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Dict, Iterable, List

from orion.autonomy.models import AutonomyEvidenceRefV1
from orion.autonomy.reducer import AutonomyReducerInputV1, reduce_autonomy_state
from orion.autonomy.summary import summarize_autonomy_state
from orion.autonomy.repository import SUBJECT_BINDINGS, build_autonomy_repository
from orion.core.schemas.reasoning_io import ReasoningWriteContextV1, ReasoningWriteRequestV1
from orion.core.schemas.reasoning_summary import ReasoningSummaryRequestV1, ReasoningSummaryV1
from orion.reasoning import InMemoryReasoningRepository, ReasoningSummaryCompiler
from orion.schemas.chat_stance import ChatStanceBrief
from orion.spark.concept_induction.profile_repository import build_concept_profile_repository

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


def _is_identity_sensitive_turn(user_message: str) -> bool:
    text = (user_message or "").lower()
    if not text:
        return False
    return any(re.search(pattern, text) for pattern in _IDENTITY_QUESTION_PATTERNS)


def _concept_summary_from_store(ctx: Dict[str, Any] | None = None) -> dict[str, list[str]]:
    ctx = ctx if isinstance(ctx, dict) else {}
    try:
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


def _load_autonomy_state(ctx: Dict[str, Any]) -> Dict[str, Any]:
    started_at = time.perf_counter()
    graphdb_cfg = resolve_autonomy_graphdb_config()
    endpoint = graphdb_cfg["endpoint"]

    backend = (os.getenv("AUTONOMY_REPOSITORY_BACKEND") or "graph").strip().lower()
    if backend not in {"graph", "local", "shadow"}:
        backend = "graph"

    repository = build_autonomy_repository(
        backend=backend,
        endpoint=endpoint,
        timeout_sec=resolve_autonomy_graph_timeout_sec(),
        user=graphdb_cfg["user"],
        password=graphdb_cfg["password"],
        goals_limit=_env_int("AUTONOMY_GOALS_LIMIT", 3),
        subject_max_workers=resolve_autonomy_subject_max_workers(),
        subquery_max_workers=resolve_autonomy_subquery_max_workers(),
    )
    subjects = ["orion", "relationship", "juniper"]
    observer = {
        "consumer": "chat_stance",
        "correlation_id": str(ctx.get("correlation_id") or ctx.get("trace_id") or ""),
        "session_id": str(ctx.get("session_id") or ""),
    }
    lookups = repository.list_latest(subjects, observer=observer)
    by_subject = {lookup.subject: lookup for lookup in lookups}
    preferred = by_subject.get("orion")
    if preferred is None or preferred.availability != "available":
        preferred = by_subject.get("relationship")
    selected_subject = preferred.subject if preferred is not None else None
    partial_used = bool(
        preferred
        and preferred.availability == "available"
        and any(
            str((preferred.subquery_diagnostics or {}).get(name, {}).get("status", "ok")) not in {"ok", "empty"}
            for name in ("identity", "drives", "goals")
        )
    )

    summary = summarize_autonomy_state(preferred.state if preferred and preferred.availability == "available" else None)
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
            "endpoint_repo": endpoint or "graphdb:unconfigured",
            "timeout_sec": resolve_autonomy_graph_timeout_sec(),
            "subject_max_workers": resolve_autonomy_subject_max_workers(),
            "subquery_max_workers": resolve_autonomy_subquery_max_workers(),
            "repository_status": {
                "source_available": repo_status.source_available,
                "source_path": repo_status.source_path,
        },
    }
    exported_keys = sorted(["autonomy_backend", "autonomy_debug", "autonomy_selected_subject", "autonomy_summary"])
    if preferred and preferred.availability == "available":
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
                "endpoint_repo": endpoint or "graphdb:unconfigured",
                "timeout_sec": resolve_autonomy_graph_timeout_sec(),
                "subject_max_workers": resolve_autonomy_subject_max_workers(),
                "subquery_max_workers": resolve_autonomy_subquery_max_workers(),
                "subjects_requested": subjects,
                "states_returned": sum(1 for item in lookups if item.availability == "available"),
                "availability_counts": {
                    "available": sum(1 for item in lookups if item.availability == "available"),
                    "empty": sum(1 for item in lookups if item.availability == "empty"),
                    "unavailable": sum(1 for item in lookups if item.availability == "unavailable"),
                    "partial": sum(
                        1
                        for item in lookups
                        if item.availability == "available"
                        and any(
                            str((item.subquery_diagnostics or {}).get(name, {}).get("status", "ok")) not in {"ok", "empty"}
                            for name in ("identity", "drives", "goals")
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
                    for subject in subjects
                },
                "exported_metadata_keys": exported_keys,
                "debug": debug,
            },
            sort_keys=True,
        ),
    )
    return {
        "lookups": lookups,
        "state": preferred.state if preferred and preferred.availability == "available" else None,
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


def _build_autonomy_reducer_evidence(ctx: Dict[str, Any], autonomy: Dict[str, Any]) -> List[AutonomyEvidenceRefV1]:
    evidence: List[AutonomyEvidenceRefV1] = []
    msg = ctx.get("user_message") or ctx.get("message") or ""
    if msg:
        digest = hashlib.sha256(str(msg)[:200].encode()).hexdigest()[:16]
        evidence.append(
            AutonomyEvidenceRefV1(
                evidence_id=f"user_turn:{digest}",
                source="user_message",
                kind="user_turn",
                summary=str(msg)[:200],
                confidence=0.9,
            )
        )
    debug = autonomy.get("debug") if isinstance(autonomy.get("debug"), dict) else {}
    orion_dbg = debug.get("orion") if isinstance(debug.get("orion"), dict) else {}
    avail = str(orion_dbg.get("availability") or "").strip()
    if avail:
        evidence.append(
            AutonomyEvidenceRefV1(
                evidence_id=f"infra_health:autonomy_graph:{avail}",
                source="infra",
                kind="infra_health",
                summary=f"autonomy graph availability={avail}",
                confidence=1.0,
            )
        )
    rs = ctx.get("chat_reasoning_summary") if isinstance(ctx.get("chat_reasoning_summary"), dict) else {}
    if rs.get("fallback_recommended"):
        evidence.append(
            AutonomyEvidenceRefV1(
                evidence_id="reasoning:fallback_recommended",
                source="reasoning",
                kind="reasoning_quality",
                summary="reasoning fallback recommended",
                confidence=0.6,
            )
        )
    social = ctx.get("chat_social_bridge_summary") if isinstance(ctx.get("chat_social_bridge_summary"), dict) else {}
    for hazard in social.get("hazards") or []:
        hid = hashlib.sha256(str(hazard)[:80].encode()).hexdigest()[:12]
        evidence.append(
            AutonomyEvidenceRefV1(
                evidence_id=f"social_bridge:{hid}",
                source="social_bridge",
                kind="relational_signal",
                summary=str(hazard)[:200],
                confidence=0.6,
            )
        )
    return evidence


def _run_autonomy_reducer(ctx: Dict[str, Any], autonomy: Dict[str, Any]):
    evidence = _build_autonomy_reducer_evidence(ctx, autonomy)
    state_obj = autonomy.get("state")
    subj = getattr(state_obj, "subject", None) if state_obj is not None else None
    subject = str(subj or "orion")
    return reduce_autonomy_state(
        AutonomyReducerInputV1(
            subject=subject,
            previous_state=state_obj,
            evidence=evidence,
            action_outcomes=[],
        )
    )


def build_chat_stance_inputs(ctx: Dict[str, Any]) -> Dict[str, Any]:
    identity = identity_kernel_with_fallbacks(ctx)
    ctx.update(identity)
    concept = _concept_summary_from_store(ctx)
    social = _social_summary(ctx)
    social_bridge = _social_bridge_summary(ctx)
    social["social_posture"] = _unique((social.get("social_posture") or []) + (social_bridge.get("posture") or []), limit=8)
    social["hazards"] = _unique((social.get("hazards") or []) + (social_bridge.get("hazards") or []), limit=8)
    social["relationship_facets"] = _unique((social.get("relationship_facets") or []) + (social_bridge.get("framing") or []), limit=8)
    reflective = _reflective_summary(ctx)
    situation = _situation_summary_from_ctx(ctx)
    reasoning = _compile_reasoning_summary(ctx)
    ctx["chat_reasoning_summary"] = reasoning["summary"]
    autonomy = _load_autonomy_state(ctx)
    mutation_cognition = _mutation_cognition_from_ctx(ctx)
    social["hazards"] = _unique((social.get("hazards") or []) + list((reasoning.get("summary") or {}).get("hazards") or []), limit=8)

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
    mg_hints = fetch_chat_stance_memory_graph_hints()
    if mg_hints:
        inputs["memory_graph"] = {"disposition_hints": mg_hints}

    if os.getenv("AUTONOMY_STATE_V2_REDUCER_ENABLED", "").strip().lower() == "true":
        try:
            v2_result = _run_autonomy_reducer(ctx, autonomy)
            ctx["chat_autonomy_state_v2"] = v2_result.state.model_dump(mode="json")
            ctx["chat_autonomy_state_delta"] = v2_result.delta.model_dump(mode="json")
            inputs["autonomy"]["state_v2"] = ctx["chat_autonomy_state_v2"]
            inputs["autonomy"]["delta"] = ctx["chat_autonomy_state_delta"]
        except Exception as exc:
            logger.warning("autonomy_reducer_v2_failed error=%s", exc)

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
            }.items()
            if isinstance(value, dict) and any(value.values())
        ]
    )

    final_prompt_contract = {
        "chat_stance_brief": final_brief,
        "memory_digest": memory_digest,
        "orion_identity_summary": list(ctx.get("orion_identity_summary") or []),
        "juniper_relationship_summary": list(ctx.get("juniper_relationship_summary") or []),
        "response_policy_summary": list(ctx.get("response_policy_summary") or []),
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
    identity_salience = "high" if identity_turn else ("low" if technical_turn else "medium")
    active_identity = list(concept.get("self") or [])[:3] + identity["orion_identity_summary"][:4]
    active_relationship = list(social.get("relationship_facets") or [])[:3] + identity["juniper_relationship_summary"][:4]
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
            "preserve_orion_juniper_continuity",
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
    response_hazards = [
        "generic assistant self-description",
        "describing Juniper as just the user",
        "customer-support tone",
        "over-clarification",
    ]
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
        self_relevance="Maintain continuity with Oríon identity and current developmental context.",
        juniper_relevance="Maintain relational continuity with Juniper while prioritizing usefulness.",
        active_identity_facets=_unique(active_identity, limit=6) if identity_salience != "low" else [],
        active_growth_axes=list(concept.get("growth") or [])[:5],
        active_relationship_facets=_unique(active_relationship, limit=6),
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

    return normalize_chat_stance_brief(merged), semantic_fallback
