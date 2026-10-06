from __future__ import annotations

import asyncio
import functools
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

try:
    import psycopg2  # type: ignore
except Exception:  # pragma: no cover - optional dependency in test env
    psycopg2 = None
from pydantic import ValidationError
from orion.core.schemas.substrate_mutation import MutationPressureEvidenceV1

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.contracts.recall import (
    MemoryBundleV1,
    RecallDecisionV1,
    RecallQueryV1,
)
from orion.memory.crystallization.repository import count_eligible_active

try:
    from .fusion import fuse_candidates, pcr_fuse_belief_candidates, render_continuity_bundle
    from .pcr_collectors import apply_collector_plan, collectors_for_intent
    from .collectors.active_packet import fetch_active_packet_fragments
    from .collectors.concept_region import fetch_concept_region_fragment_and_reinforce
    from .substrate_store import get_substrate_store
    from .profiles import get_profile
    from .settings import settings
    from .source_policy import build_vector_policy
    from .storage.rdf_adapter import (
        fetch_rdf_fragments,
        fetch_rdf_chatturn_fragments,
        fetch_rdf_chatturn_exact_matches,
    )
    from .storage.falkor_chat_adapter import fetch_falkor_chatturn_fragments
    from .storage.falkor_neighborhood_adapter import fetch_falkor_neighborhood_fragments
    from .storage.falkor_bus_synaptic_adapter import fetch_bus_synaptic_anomaly_fragments
    from .storage.falkor_entity_relatedness import (
        fetch_related_entities,
        fetch_entity_matches_for_turns,
        fetch_turns_mentioning_entities,
        fetch_entity_degrees,
    )
    from .sql_timeline import fetch_recent_fragments, fetch_related_by_entities, fetch_exact_fragments
    from .sql_chat import fetch_chat_history_pairs, fetch_chat_messages, fetch_chat_turn_timestamps, fetch_chat_turns_by_id, _to_epoch
    from .chat_source_tagging import chat_source_tags, render_quoted_chat_text
    from .cards_adapter import fetch_card_fragments_guarded
    try:
        from .storage.graph_compression_adapter import fetch_graph_compression_fragments
    except ImportError:
        fetch_graph_compression_fragments = None  # type: ignore

except ImportError as _e:  # pragma: no cover - fallback for runtime pathing
    _IMPORT_ERROR = _e
    try:
        # Container/package-safe absolute imports
        from app.fusion import fuse_candidates, pcr_fuse_belief_candidates, render_continuity_bundle  # type: ignore
        from app.pcr_collectors import apply_collector_plan, collectors_for_intent  # type: ignore
        from app.collectors.active_packet import fetch_active_packet_fragments  # type: ignore
        from app.collectors.concept_region import fetch_concept_region_fragment_and_reinforce  # type: ignore
        from app.substrate_store import get_substrate_store  # type: ignore
        from app.profiles import get_profile  # type: ignore
        from app.settings import settings  # type: ignore
        from app.source_policy import build_vector_policy  # type: ignore
        from app.storage.rdf_adapter import (  # type: ignore
            fetch_rdf_fragments,
            fetch_rdf_chatturn_fragments,
            fetch_rdf_chatturn_exact_matches,
        )
        from app.storage.falkor_chat_adapter import fetch_falkor_chatturn_fragments  # type: ignore
        from app.storage.falkor_neighborhood_adapter import fetch_falkor_neighborhood_fragments  # type: ignore
        from app.storage.falkor_entity_relatedness import (  # type: ignore
            fetch_related_entities,
            fetch_entity_matches_for_turns,
            fetch_turns_mentioning_entities,
            fetch_entity_degrees,
        )
        from app.sql_timeline import fetch_recent_fragments, fetch_related_by_entities, fetch_exact_fragments  # type: ignore
        from app.sql_chat import fetch_chat_history_pairs, fetch_chat_messages, fetch_chat_turn_timestamps, fetch_chat_turns_by_id, _to_epoch  # type: ignore
        from app.chat_source_tagging import chat_source_tags, render_quoted_chat_text  # type: ignore
        from app.cards_adapter import fetch_card_fragments_guarded  # type: ignore
        try:
            from app.storage.graph_compression_adapter import fetch_graph_compression_fragments  # type: ignore
        except ImportError:
            fetch_graph_compression_fragments = None  # type: ignore
    except ImportError:
        # IMPORTANT: raise the real root cause, not the fallback failure
        raise _IMPORT_ERROR

logger = logging.getLogger("orion-recall.worker")

try:
    import asyncpg  # type: ignore
except Exception:  # pragma: no cover
    asyncpg = None

_recall_pg_pool: Any = None


def set_recall_pg_pool(pool: Any) -> None:
    global _recall_pg_pool
    _recall_pg_pool = pool


RECALL_REQUEST_KIND = "recall.query.v1"
RECALL_REPLY_KIND = "recall.reply.v1"
RECALL_TELEMETRY_KIND = "recall.decision.v1"
def _build_recall_pressure_events(
    *,
    q: RecallQueryV1,
    decision: RecallDecisionV1,
    bundle: MemoryBundleV1,
    compare_summary: Dict[str, Any] | None = None,
    anchor_plan: Dict[str, Any] | None = None,
    selected_evidence_cards: list[Dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    events: list[MutationPressureEvidenceV1] = []
    snippets = [item.snippet.lower() for item in bundle.items]
    query_text = (q.fragment or "").lower()
    selected_any = bool(bundle.items)
    exact_anchor_tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\d+\b|\b[A-Fa-f0-9]{7,40}\b", str(q.fragment or ""))[:8]
    exact_anchor_hit = any(token.lower() in " ".join(snippets) for token in exact_anchor_tokens) if exact_anchor_tokens else True
    top_source = bundle.items[0].source if bundle.items else ""
    stale_selected = False
    if bundle.items and bundle.items[0].ts is not None:
        try:
            age_hours = max(0.0, (time.time() - float(bundle.items[0].ts)) / 3600.0)
            stale_selected = age_hours > 24.0 * 30.0
        except Exception:
            stale_selected = False

    shared_metadata: Dict[str, Any] = {"recall_evidence_kind": "live_shadow"}
    if isinstance(compare_summary, dict) and compare_summary:
        shared_metadata["v1_v2_compare"] = compare_summary
    if isinstance(anchor_plan, dict) and anchor_plan:
        shared_metadata["anchor_plan"] = anchor_plan
    if isinstance(selected_evidence_cards, list) and selected_evidence_cards:
        shared_metadata["selected_evidence_cards"] = selected_evidence_cards[:8]

    if not selected_any:
        events.append(
            MutationPressureEvidenceV1(
                source_service=settings.SERVICE_NAME,
                source_event_id=decision.corr_id,
                pressure_category="recall_miss_or_dissatisfaction",
                confidence=0.9,
                evidence_refs=[f"recall_decision:{decision.id}", f"query:{q.fragment[:120]}"],
                metadata={"reason": "no_selected_items", "profile": decision.profile, **shared_metadata},
            )
        )
    if selected_any and query_text and not any(tok in " ".join(snippets) for tok in _extract_keywords(q.fragment)):
        events.append(
            MutationPressureEvidenceV1(
                source_service=settings.SERVICE_NAME,
                source_event_id=decision.corr_id,
                pressure_category="unsupported_memory_claim",
                confidence=0.72,
                evidence_refs=[f"recall_decision:{decision.id}", f"selected_ids:{','.join(decision.selected_ids[:6])}"],
                metadata={"reason": "selected_without_query_support", **shared_metadata},
            )
        )
    if selected_any and top_source == "vector" and any("vector" in str(tag).lower() for item in bundle.items for tag in item.tags):
        events.append(
            MutationPressureEvidenceV1(
                source_service=settings.SERVICE_NAME,
                source_event_id=decision.corr_id,
                pressure_category="irrelevant_semantic_neighbor",
                confidence=0.55,
                evidence_refs=[f"recall_decision:{decision.id}", f"top_source:{top_source}"],
                metadata={"reason": "vector_top_hit_requires_anchor_validation", **shared_metadata},
            )
        )
    if not exact_anchor_hit:
        events.append(
            MutationPressureEvidenceV1(
                source_service=settings.SERVICE_NAME,
                source_event_id=decision.corr_id,
                pressure_category="missing_exact_anchor",
                confidence=0.81,
                evidence_refs=[f"recall_decision:{decision.id}", f"anchor_tokens:{','.join(exact_anchor_tokens[:6])}"],
                metadata={"reason": "exact_anchor_not_in_selected", **shared_metadata},
            )
        )
    if stale_selected:
        events.append(
            MutationPressureEvidenceV1(
                source_service=settings.SERVICE_NAME,
                source_event_id=decision.corr_id,
                pressure_category="stale_memory_selected",
                confidence=0.76,
                evidence_refs=[f"recall_decision:{decision.id}", f"top_id:{bundle.items[0].id}"],
                metadata={"reason": "top_item_stale", "source": bundle.items[0].source, **shared_metadata},
            )
        )
    # keep bounded and first-class serialized
    return [item.model_dump(mode="json") for item in events[:5]]



def _source() -> ServiceRef:
    return ServiceRef(
        name=settings.SERVICE_NAME,
        version=settings.SERVICE_VERSION,
        node=settings.NODE_NAME,
    )


def _extract_entities(text: str) -> List[str]:
    """Regression, found in code review (2026-07-19): the two patterns below
    used to be r"...(?:\\s+...)*" and r"...\\.[..\\.]+" -- a literal
    double-backslash inside an r-string matches a literal backslash
    character, not whitespace/a dot. That silently broke this function's
    entire multi-word-span and dotted-identifier purpose: "New York" could
    only ever come back as "New"/"York" separately, and "settings.py"/
    "example.com" never matched at all (confirmed live).

    Two things caught in the SAME review pass and fixed before this shipped
    (not separately, since both change the same line):
    - [a-z]+ (one-or-more) widened to [a-zA-Z]+ (still one-or-more, not
      zero-or-more) so all-caps acronyms match ("NVIDIA", not just
      "Nvidia") -- but keeping the "at least 2 chars" floor. An earlier
      version of this fix used [a-zA-Z]* (zero-or-more), which also matched
      bare single capital letters like "I" -- a real regression on the
      already-live (non-dark) sql_timeline.py::fetch_related_by_entities
      call site, which builds an unbounded ILIKE '%I%' from it.
    - The multi-word merge group capped at exactly one extra word (trailing
      `?`, not `*`), matching the sibling extractor's own convention
      already in this codebase (app/recall_v2.py::_extract_entities). `*`
      would let a long Title-Case sentence collapse into one giant string
      that then matches nothing real in Falkor or Postgres.

    Still doesn't filter stopword-like capitalized sentence-starters
    ("Tell" in "Tell me about...") -- a separate, disclosed, deliberately
    deferred concern (services/orion-recall/README.md's entity-relatedness-
    boost section), not something this fix attempts to solve."""
    # Deterministic order (2026-09-29): results come back in order of first
    # appearance in the text. This used to be a set(), so every caller that
    # sliced it ("first 3 entities") got an arbitrary, restart-dependent pick.
    first_pos: Dict[str, int] = {}
    for pattern, flags in (
        (r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?", 0),
        (r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", re.I),
        (r"[A-Za-z0-9_]+\.[A-Za-z0-9_.]+", 0),
    ):
        for m in re.finditer(pattern, text or "", flags=flags):
            ent = m.group(0).strip()
            if ent and (ent not in first_pos or m.start() < first_pos[ent]):
                first_pos[ent] = m.start()
    return sorted(first_pos, key=lambda e: (first_pos[e], e))


def _extract_keywords(text: str, *, max_keywords: int = 6) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9_]{3,}", (text or "").lower())
    seen = set()
    keywords: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= max_keywords:
            break
    return keywords


def _anchor_tokens(text: str, *, max_tokens: int = 3) -> List[str]:
    if not text:
        return []
    # Was r"\\b...\\d+\\b" (a literal backslash inside an r-string), so it
    # could only match text containing backslashes: the anchor rail was dead
    # (_anchor_tokens("p4 v100 gpu1") == []). Fixed 2026-09-29.
    # UUIDs first: \b treats their hyphens as word boundaries, so a run id
    # like 1765808d-3a64-4be2-be03-... would otherwise yield "be03" as an
    # "anchor" (seen on 12 of 80 live recall_telemetry queries once this
    # regex was fixed). Pure-hex ids of 8+ chars (trace ids) are dropped too.
    text = re.sub(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", " ", text)
    tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9]*\d+\b", text)
    seen = set()
    anchors: List[str] = []
    for token in tokens:
        if token in seen or (len(token) >= 8 and re.fullmatch(r"[0-9a-fA-F]+", token)):
            continue
        seen.add(token)
        anchors.append(token)
        if len(anchors) >= max_tokens:
            break
    return anchors


# The browse shortcut is for a short, direct request ("show recent
# memories"). Its regex was dead until 2026-09-29 (same literal-backslash bug
# as _anchor_tokens). Once live, an unbounded check would send any long
# prompt that merely mentions "recall ... context" down the recent-only
# browse path and skip retrieval entirely, so it only applies to short text.
_MEMORY_BROWSE_MAX_CHARS = 160


def _is_memory_browse(text: str) -> bool:
    """Recent-only browse shortcut. Off by default
    (RECALL_BROWSE_SHORTCUT_ENABLED=false keeps main's pre-2026-09-29
    behavior, when this regex could never match). When on, the object of the
    verb must be memory/memories within a few words ("show recent memories",
    "list my memories"): ordinary questions like "show me the context around
    the gpu1 crash" or "list the recent errors" must keep full retrieval
    (code review, PR #2416)."""
    if not bool(getattr(settings, "RECALL_BROWSE_SHORTCUT_ENABLED", False)):
        return False
    if not text or len(text) > _MEMORY_BROWSE_MAX_CHARS:
        return False
    lowered = text.lower()
    return bool(
        re.search(r"\b(fetch|show|list|browse|recall)\b(?:\s+[a-z']+){0,3}?\s+(memory|memories)\b", lowered)
    )


_SOCIAL_TOKENS = {
    "hi",
    "hey",
    "hello",
    "thanks",
    "thank",
    "good",
    "great",
    "cool",
    "nice",
    "fine",
    "well",
    "friend",
    "orion",
    "juniper",
    "morning",
    "afternoon",
    "evening",
    "awesome",
    "okay",
    "ok",
}

_STOPWORDS = {
    "the",
    "and",
    "that",
    "this",
    "with",
    "from",
    "have",
    "been",
    "just",
    "your",
    "about",
    "into",
    "over",
    "mostly",
    "very",
    "really",
    "still",
    "will",
    "would",
    "should",
    "could",
}


def _strip_recall_instruction_tail(text: str) -> Tuple[str, bool]:
    raw = str(text or "").strip()
    if not raw:
        return "", False
    parts = [part.strip(" -:\t") for part in re.split(r"[\n;]+", raw) if part.strip()]
    if len(parts) < 2:
        return raw, False
    tail = parts[-1].lower()
    instruction_markers = (
        "use recall",
        "based on recall",
        "remain based on recall",
        "stay based on recall",
        "process injection",
    )
    if any(marker in tail for marker in instruction_markers):
        return " ".join(parts[:-1]).strip(), True
    return raw, False


def _social_clause(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        return True
    if re.match(r"^(hi|hey|hello|yo|how are you|how's it going)[!.? ]*$", normalized):
        return True
    tokens = re.findall(r"[a-z']{2,}", normalized)
    if len(tokens) <= 7 and tokens and all(token in _SOCIAL_TOKENS for token in tokens):
        return True
    return False


def _informative_score(text: str) -> int:
    tokens = re.findall(r"[a-z0-9']{3,}", text.lower())
    informative = [tok for tok in tokens if tok not in _STOPWORDS]
    long_terms = [tok for tok in informative if len(tok) >= 6 or any(ch.isdigit() for ch in tok)]
    return len(informative) + len(long_terms)


def _derive_chat_general_query(fragment: str, *, verb: str | None, profile_name: str) -> Dict[str, Any]:
    raw = str(fragment or "").strip()
    applies = str(verb or "") == "chat_general" or profile_name.startswith("chat.general")
    if not applies:
        return {
            "query_fragment": raw,
            "tail_stripped": False,
            "query_changed": False,
            "turn_type": "default",
            "dropped_clauses": 0,
        }
    trimmed, tail_stripped = _strip_recall_instruction_tail(raw)
    clauses = [c.strip() for c in re.split(r"[.!?\n]+", trimmed) if c.strip()]
    substantive = [c for c in clauses if not _social_clause(c)]
    dropped = max(0, len(clauses) - len(substantive))
    ranked = sorted(substantive, key=_informative_score, reverse=True)
    query_fragment = ". ".join(ranked[:2]).strip() if ranked else trimmed
    if not query_fragment:
        query_fragment = raw
    turn_type = "substantive" if substantive else "social"
    return {
        "query_fragment": query_fragment,
        "tail_stripped": tail_stripped,
        "query_changed": query_fragment != raw,
        "turn_type": turn_type,
        "dropped_clauses": dropped,
    }


def _max_sub_queries() -> int:
    """RECALL_MAX_SUB_QUERIES, where 0 means uncapped. Deliberately not
    `x or DEFAULT`: that would turn a configured 0 (the rollback lever) back
    into the default cap."""
    raw = getattr(settings, "RECALL_MAX_SUB_QUERIES", None)
    if raw is None:
        return 4
    try:
        return max(0, int(raw))
    except Exception:
        return 4


def _max_query_chars() -> int:
    raw = getattr(settings, "RECALL_MAX_QUERY_CHARS", None)
    if raw is None:
        return 600
    try:
        return int(raw)
    except Exception:
        return 600


def _condense_query(text: str, *, max_chars: int) -> str:
    """Deterministic condensation of an over-long fragment, for every verb.

    Generalizes _derive_chat_general_query's clause scoring (which only runs
    for chat_general): strip a trailing recall instruction, split into
    clauses, drop social filler, rank by _informative_score (ties broken by
    position), then take whole clauses best-first until ``max_chars`` is
    full. A single clause longer than the budget is hard-cut. No LLM, same
    input -> same output.
    """
    trimmed, _tail = _strip_recall_instruction_tail(str(text or ""))
    # Keep each clause's terminator so questions can be told apart.
    raw_clauses = [c.strip() for c in re.findall(r"[^.!?\n]+[.!?]?", trimmed) if c.strip(" .!?\t")]
    is_question = [c.endswith("?") for c in raw_clauses]
    clauses = [c.rstrip(".!?").strip() for c in raw_clauses]
    substantive = [(i, c) for i, c in enumerate(clauses) if not _social_clause(c)] or list(enumerate(clauses))
    # Question clauses first: in a self-initiated prompt the question is
    # what the turn is about (the 30,663-char self-inquiry prompt's standing
    # question, not its most word-dense instruction). Then informative
    # score, then position. Prompts with no questions rank exactly as
    # _derive_chat_general_query's scoring would.
    ranked = sorted(substantive, key=lambda ic: (not is_question[ic[0]], -_informative_score(ic[1]), ic[0]))
    picked: List[str] = []
    used = 0
    for _i, clause in ranked:
        extra = len(clause) + (2 if picked else 0)
        if used + extra > max_chars:
            if not picked:
                cut = clause[:max_chars].strip()
                if cut:
                    picked.append(cut)
                    used = len(cut)
            continue
        picked.append(clause)
        used += extra
    condensed = ". ".join(picked).strip()
    # Never empty for non-empty input: fall back to the raw head.
    return condensed[:max_chars].strip() or str(text or "")[:max_chars].strip() or str(text or "")[:max_chars]


def _intake_query(q: RecallQueryV1, *, profile_name: str) -> Dict[str, Any]:
    """Pick the text recall actually searches (design step 1, intake guard).

    - ``caller``: RecallQueryV1.retrieval_query is set -> search it. The
      fragment stays the turn text (self-hit exclusion, provenance).
    - ``condensed``: no retrieval_query and the (chat_general-targeted)
      fragment exceeds RECALL_MAX_QUERY_CHARS -> _condense_query.
    - ``fragment``: otherwise, the fragment as today.
    """
    retrieval_query = str(getattr(q, "retrieval_query", None) or "").strip()
    if retrieval_query:
        # Classification only (turn_type feeds fusion's substantive-query
        # salience); the caller's text is searched as given.
        targeting = _derive_chat_general_query(retrieval_query, verb=q.verb, profile_name=profile_name)
        return {"search_text": retrieval_query, "source": "caller", "query_targeting": targeting}

    targeting = _derive_chat_general_query(q.fragment, verb=q.verb, profile_name=profile_name)
    text = str(targeting.get("query_fragment") or q.fragment or "")
    max_chars = _max_query_chars()
    if not text.strip():
        # Whitespace only: nothing to search for, and nothing to condense.
        return {"search_text": "", "source": "fragment", "query_targeting": targeting}
    if max_chars > 0 and len(text) > max_chars:
        condensed = _condense_query(text, max_chars=max_chars)
        targeting = {**targeting, "query_fragment": condensed, "query_changed": True, "condensed_from_chars": len(text)}
        return {"search_text": condensed, "source": "condensed", "query_targeting": targeting}
    return {"search_text": text, "source": "fragment", "query_targeting": targeting}


# Capitalized words that start sentences or address the reader, not things a
# memory is "about". Small and explicit on purpose (tested in
# tests/test_recall_bounded_retrieval.py): the 30,663-char live prompt turned
# "No", "It", "The", "You" into sub-queries. Not a taxonomy -- anything not
# listed here still has to win on specificity to survive the cap.
_EXPANSION_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "then", "so", "because", "also",
        "no", "not", "yes", "ok", "okay",
        "i", "me", "my", "we", "our", "us", "you", "your", "he", "she", "it", "its", "they", "them", "their",
        "this", "that", "these", "those", "there", "here",
        "what", "when", "where", "why", "how", "who", "which", "whether",
        "is", "are", "was", "were", "be", "been", "do", "does", "did", "have", "has", "had",
        "can", "could", "will", "would", "should", "may", "might", "must",
        "in", "on", "at", "to", "of", "for", "from", "with", "by", "as", "about", "into", "after", "before",
        "each", "every", "all", "any", "some", "one", "only", "just", "now", "never", "always",
        "please", "note", "use", "return", "write", "keep", "make", "tell", "let", "give",
        "hi", "hey", "hello", "thanks",
    }
)


def _clean_entity(ent: str) -> str:
    """Drop leading stopword words ("The Orion" -> "Orion"); '' if nothing is left."""
    words = ent.split()
    while words and words[0].lower() in _EXPANSION_STOPWORDS:
        words = words[1:]
    if not words:
        return ""
    if len(words) == 1 and words[0].lower() in _EXPANSION_STOPWORDS:
        return ""
    return " ".join(words)


def _entity_specificity_key(ent: str, first_pos: int) -> Tuple[int, int, int]:
    """Sort key, most specific first. Identifier-shaped (uuid, dotted name,
    contains a digit) beats a multi-word proper name beats a single word;
    then longer beats shorter; then earlier in the text. Fully deterministic."""
    if re.fullmatch(r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", ent, flags=re.I):
        kind = 3
    elif "." in ent or any(ch.isdigit() for ch in ent):
        kind = 2
    elif " " in ent:
        kind = 1
    else:
        kind = 0
    return (-kind, -len(ent), first_pos)


def _ranked_entities(text: str, *, limit: int) -> List[str]:
    """Bounded, stopword-filtered, specificity-ranked entities.

    ``limit`` 0 = the pre-2026-09-29 behavior: every _extract_entities hit,
    unfiltered, in first-appearance order (rollback lever)."""
    raw = _extract_entities(text)
    if limit <= 0:
        return raw
    seen: Dict[str, int] = {}
    cleaned: List[Tuple[str, int]] = []
    for pos, ent in enumerate(raw):
        c = _clean_entity(ent)
        if not c or len(c) < 2 or c.lower() in seen:
            continue
        seen[c.lower()] = pos
        cleaned.append((c, pos))
    cleaned.sort(key=lambda cp: _entity_specificity_key(cp[0], cp[1]))
    return [c for c, _ in cleaned[:limit]]


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return " ".join(text.split())


def _parse_exclusion(q: RecallQueryV1) -> Dict[str, Any]:
    raw = q.exclude if isinstance(q.exclude, dict) else {}
    active_turn_text = str(raw.get("active_turn_text") or q.fragment or "").strip()
    active_turn_ids: List[str] = []
    for value in raw.get("active_turn_ids", []):
        text = str(value or "").strip()
        if text and text not in active_turn_ids:
            active_turn_ids.append(text)
    try:
        active_turn_ts = float(raw.get("active_turn_ts")) if raw.get("active_turn_ts") is not None else None
    except Exception:
        active_turn_ts = None
    return {
        "active_turn_text": active_turn_text,
        "active_turn_ids": active_turn_ids,
        "active_turn_ts": active_turn_ts,
    }


def _suppress_self_hits(
    candidates: List[Dict[str, Any]],
    *,
    active_turn_text: str,
    active_turn_ids: List[str],
    active_turn_ts: float | None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if not candidates:
        return [], {}
    normalized_active = _normalize_text(active_turn_text)
    id_set = {str(v).strip() for v in active_turn_ids if str(v).strip()}
    suppression_counts: Dict[str, int] = {}
    filtered: List[Dict[str, Any]] = []
    for cand in candidates:
        source = str(cand.get("source") or "unknown")
        cand_id = str(cand.get("id") or "").strip()
        cand_text = _normalize_text(cand.get("text") or cand.get("snippet") or "")
        remove = False
        if cand_id and cand_id in id_set:
            remove = True
        elif normalized_active and normalized_active in cand_text:
            age_ok = True
            if active_turn_ts is not None and cand.get("ts") is not None:
                try:
                    age_ok = abs(float(cand.get("ts")) - active_turn_ts) <= 180.0
                except Exception:
                    age_ok = True
            if age_ok:
                remove = True
        if remove:
            suppression_counts[source] = suppression_counts.get(source, 0) + 1
            continue
        filtered.append(cand)
    return filtered, suppression_counts

def _anchor_overlap(text: str, anchors: List[str]) -> int:
    if not text or not anchors:
        return 0
    lowered = text.lower()
    return sum(1 for term in anchors if term and term.lower() in lowered)


def _artifact_density(text: str) -> int:
    if not text:
        return 0
    patterns = [
        r"/[A-Za-z0-9_\-./]+",
        r"orion-[a-z0-9\-]+",
        r"orion:[a-z0-9:._-]+",
        r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}",
    ]
    return sum(len(re.findall(pattern, text)) for pattern in patterns)


def _anchor_from_uri(uri: str) -> str:
    if not uri:
        return ""
    tail = uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
    return tail.replace("_", " ").strip()


def _extract_anchor_terms(rdf_items: List[Dict[str, Any]], *, max_items: int = 12) -> List[str]:
    terms: List[str] = []
    seen = set()

    def _add(term: str) -> None:
        cleaned = (term or "").strip()
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        terms.append(cleaned)

    for item in rdf_items:
        meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        for candidate in (meta.get("subject"), item.get("uri"), item.get("id")):
            if isinstance(candidate, str):
                _add(_anchor_from_uri(candidate))
        text = item.get("text")
        if isinstance(text, str):
            for token in re.findall(r"[A-Za-z0-9_]{3,}", text):
                _add(token)
        if len(terms) >= max_items:
            break

    return terms[:max_items]


def _expand_query(
    fragment: str,
    *,
    verb: str | None,
    intent: str | None,
    enable: bool,
    max_sub_queries: int | None = None,
    entities: List[str] | None = None,
) -> List[str]:
    """Sub-queries the retrievers run over: the search text, the verb/intent
    hints, then extracted entities.

    Bounded (2026-09-29): with K = RECALL_MAX_SUB_QUERIES > 0 the result has
    at most K + 2 entries, entities are stopword-filtered and ranked most
    specific first, and duplicates (case-insensitive) collapse. A 30,663-char
    prompt used to yield 268 sub-queries here. K = 0 restores the old,
    uncapped fan-out."""
    if not enable:
        return [fragment]
    k = _max_sub_queries() if max_sub_queries is None else max(0, int(max_sub_queries))
    hints = [h for h in (verb, intent) if h]
    if entities is None:
        entities = _ranked_entities(fragment, limit=k)
    if k <= 0:
        return [s for s in [fragment, *hints, *entities] if s]
    signals: List[str] = []
    seen: set[str] = set()
    for sig in [fragment, *hints, *entities]:
        key = str(sig or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        signals.append(sig)
    return signals[: k + 2]


_ENTITY_RELATEDNESS_MAX_QUERY_ENTITIES = 3
_ENTITY_RELATEDNESS_MAX_RELATED_PER_ENTITY = 15
_ENTITY_RELATEDNESS_MAX_INJECTED_TURNS = 6
# IDF-style discount constant: an entity at exactly this degree keeps full
# (1.0x) weight; above it, weight falls off as K/degree. Calibrated against
# live data: nvidia(23)->0.65x, atlas(17)->0.88x, tesla(7)/p4(13)->1.0x
# (capped) stay near full weight (genuinely specific entities); orion(282)
# ->0.05x, juniper(260)->0.06x are correctly suppressed (near-universal,
# near-zero discriminative value).
_ENTITY_RELATEDNESS_DEGREE_DISCOUNT_K = 15
# An entity whose discounted score falls below this floor can still
# contribute to BOOSTING an already-present candidate (weakly), but can no
# longer single-handedly drive a new turn INTO the pool -- see the
# injection_target_names comment below for why the score discount alone
# wasn't sufficient. For a DIRECT query-entity match (base score 1.0), K=15
# excludes anything with degree greater than ~100 (orion/juniper at
# 260-282 land at 0.05-0.06, well below); genuinely specific entities
# (nvidia=23 -> 0.65, atlas=17 -> 0.88) clear it easily. For a Jaccard-
# RELATED entity (base score already <1.0 before this discount), the
# effective degree cutoff is proportionally lower -- e.g. jaccard=0.3 hits
# the floor around degree=30, not degree=100 -- this is intentional
# (double-discounting a low-jaccard, moderate-frequency related entity is
# correct), not a bug, but the "~100" figure above only describes the
# direct-match case.
_ENTITY_RELATEDNESS_MIN_INJECTION_SCORE = 0.15
# Fallback discount for an entity fetch_entity_degrees has no data for --
# see the discount loop's own comment for why this must stay conservative
# (below the injection floor), not "no discount" (score * 1.0).
_ENTITY_RELATEDNESS_UNKNOWN_DEGREE_DISCOUNT = 0.1


def _boost_query_entities(query_text: str) -> List[str]:
    """Entities the boost targets: the specificity-ranked list, max(K, 3)
    long (the boost fans out on its first 3), or the raw list when K=0.
    Both process_recall and the standalone fallback call this, so they
    always agree on the count."""
    k = _max_sub_queries()
    return _ranked_entities(query_text, limit=(max(k, _ENTITY_RELATEDNESS_MAX_QUERY_ENTITIES) if k > 0 else 0))


async def _compute_entity_relatedness_boost_map(
    *,
    query_text: str,
    candidates: List[Dict[str, Any]],
    query_entities: List[str] | None = None,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Phase 2 of entity-graph-reasoning (docs/superpowers/specs/
    2026-07-19-recall-entity-graph-reasoning-arc.md). Best-effort: any
    failure (no client, no query entities, Falkor error) returns ({}, []),
    which is a true no-op for both the boost and the injection paths -- this
    must never be a hard dependency for recall to return results.

    1. Extract entities from the query (same _extract_entities heuristic
       used elsewhere in this file), lowercased to match Falkor's always-
       lowercase Entity.name convention (filter_noise() normalizes at write
       time -- see services/orion-meta-tags/app/falkor_recall_writer.py).
       Deduped via dict.fromkeys (not set()) to preserve first-appearance
       order before capping -- _extract_entities itself builds its result
       from a set(), so slicing that directly is non-deterministic across
       process restarts (confirmed live in code review); this re-orders it
       deterministically before the [:N] cap below is applied.
    2. For up to the first 3 query entities (by appearance order), fetch
       Jaccard-related entities (fetch_related_entities) IN PARALLEL --
       these are independent lookups keyed on different names, no reason to
       serialize 3 Falkor round trips in a hot path.
    3. Build a target->score map: the query entities THEMSELVES score 1.0
       (a candidate turn directly mentioning what the query is about is
       stronger evidence than "related to" it), related entities score
       their own Jaccard value. Then apply a document-frequency discount
       (fetch_entity_degrees) to EVERY target entity, direct or related --
       see the constant's own comment for why this is load-bearing, not
       cosmetic (a near-universal entity like "orion" must not score a
       flat 1.0 just because it was directly mentioned).
    4. Batch-lookup (one query) which EXISTING falkor_chat candidate turns
       mention any target entity, for boosting.
    5. ALSO fetch actual ChatTurn ids that mention any target entity,
       independent of what's already in the pool (fetch_turns_mentioning_
       entities), hydrate their text via Postgres, and return them as NEW
       candidates to inject -- excluding any turn_id already present.

    Step 5 exists because live evidence (6 real queries across 3 profiles)
    showed step 4 alone never changes a single ranking: falkor_chat's own
    fetch (falkor_chat_adapter.py) is deliberately recency-windowed with no
    query filter (Phase 4's own design), so an entity from a turn older
    than that window never enters the pool for step 4 to boost in the first
    place. A fusion-weight boost can only re-rank what's already fetched;
    step 5 is what actually gets query-relevant turns into the pool at all.

    The whole body below the cheap guard checks is wrapped in try/except:
    this function's entire contract is "never a hard dependency for recall
    to return results" -- the two Falkor calls were already individually
    guarded, but code review correctly found the surrounding logic (turn_id
    collection, dict comprehensions) was not, so a future edit there could
    still crash process_recall's default (non-PCR) path with no fallback.
    """
    if not settings.RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED or not query_text:
        return {}, []

    try:
        # 2026-09-29: the entities come ranked most-specific-first
        # (_ranked_entities) and bounded by RECALL_MAX_SUB_QUERIES, so the
        # "first 3" below are the 3 most specific, deterministically -- not
        # whatever order a set() happened to iterate in, and not all 266
        # capitalized words of a 30k-char prompt as degree/mention targets.
        if query_entities is None:
            query_entities = _boost_query_entities(query_text)
        query_entities = list(dict.fromkeys(str(e).lower() for e in query_entities if str(e).strip()))
        if not query_entities:
            return {}, []

        target_scores: Dict[str, float] = {e: 1.0 for e in query_entities}
        capped_entities = query_entities[:_ENTITY_RELATEDNESS_MAX_QUERY_ENTITIES]
        related_results = await asyncio.gather(
            *(
                fetch_related_entities(name=entity, max_results=_ENTITY_RELATEDNESS_MAX_RELATED_PER_ENTITY)
                for entity in capped_entities
            ),
            return_exceptions=True,
        )
        for entity, related in zip(capped_entities, related_results):
            if isinstance(related, BaseException):
                logger.debug(f"entity relatedness boost: fetch_related_entities skipped for {entity!r}: {related}")
                continue
            for r in related:
                name = str(r.get("name") or "")
                score = float(r.get("jaccard") or 0.0)
                if name and score > target_scores.get(name, 0.0):
                    target_scores[name] = score

        # Document-frequency discount, applied uniformly to every target
        # entity (both direct query matches and Jaccard-related ones).
        # Live-confirmed bug this closes: a mundane message that simply
        # addresses the assistant by name ("Orion, what do you think...")
        # extracted "orion" as a query entity and scored it a flat 1.0 --
        # "orion" is one of the two most frequent nodes in the whole graph
        # (282 mentions, confirmed live), so this injected generic filler
        # turns ("thanks, appreciated.") at full boost strength purely
        # because they happened to mention Orion by name. Jaccard-related
        # scores already have partial frequency awareness (their own
        # degree2 sits in the denominator), but the direct-match path had
        # none at all. K/degree is a real IDF instance, not a stoplist --
        # self-correcting as the graph grows, no hardcoded entity names.
        degrees = await fetch_entity_degrees(names=list(target_scores.keys()))
        for name in list(target_scores.keys()):
            degree = degrees.get(name)
            if degree and degree > 0:
                discount = min(1.0, _ENTITY_RELATEDNESS_DEGREE_DISCOUNT_K / degree)
            else:
                # A name absent from `degrees` is indistinguishable, from
                # here, between "genuinely brand-new entity, no live
                # mentions yet" and "the degree lookup call itself failed"
                # -- _safe_graph_query swallows Falkor errors into an empty
                # result, never a raised exception (found in code review).
                # Treating "unknown" as "no discount" would reopen this
                # patch's own bug on any transient Falkor hiccup on this one
                # call, indistinguishable in the logs from "working as
                # designed". Biasing conservative here is deliberate: a
                # missed injection opportunity for a genuinely rare entity
                # is a far smaller cost than silently reintroducing generic-
                # filler injection. Set below the injection floor so an
                # unverified entity can still weakly boost an existing
                # candidate but can never single-handedly drive injection.
                discount = _ENTITY_RELATEDNESS_UNKNOWN_DEGREE_DISCOUNT
            target_scores[name] *= discount

        existing_turn_ids = {
            str(c.get("uri") or c.get("id") or "").strip()
            for c in candidates
            if str(c.get("source") or "") == "falkor_chat" and (c.get("uri") or c.get("id"))
        } - {""}

        # Discounting the SCORE alone isn't enough to stop low-value
        # injection: an injected fragment still carries the same fixed
        # base_score (0.50) as any genuinely recency-fetched falkor_chat
        # candidate, so even a heavily-discounted entity boost doesn't stop
        # it from competing on base_score+recency alone once it's already
        # in the pool (live-confirmed: a "thanks, appreciated." turn still
        # placed top-4 with only a ~0.01 discounted boost). The real fix is
        # upstream of injection: entities whose discounted score falls
        # below this floor never drive a fetch_turns_mentioning_entities
        # call at all, so a near-universal entity like "orion" can't inject
        # ANY turn on its own -- it can still contribute to boosting a
        # turn that ALSO independently earned its way into the pool via a
        # real entity, since that pathway isn't gated here.
        injection_target_names = [
            name for name, score in target_scores.items() if score >= _ENTITY_RELATEDNESS_MIN_INJECTION_SCORE
        ]
        mentioning = (
            await fetch_turns_mentioning_entities(
                target_names=injection_target_names, max_results=_ENTITY_RELATEDNESS_MAX_INJECTED_TURNS
            )
            if injection_target_names
            else []
        )
        new_turn_ids = [
            t
            for t in dict.fromkeys(str(m.get("turn_id") or "").strip() for m in mentioning)
            if t and t not in existing_turn_ids
        ]

        injected: List[Dict[str, Any]] = []
        if new_turn_ids:
            text_map = await fetch_chat_turns_by_id(new_turn_ids)
            ts_by_turn = {str(m.get("turn_id")): m.get("ts") for m in mentioning}
            for turn_id in new_turn_ids:
                if turn_id not in text_map:
                    continue
                prompt, response, client_meta = text_map[turn_id]
                text = render_quoted_chat_text(prompt, response, client_meta)
                injected.append(
                    {
                        "id": turn_id,
                        "source": "falkor_chat",
                        "source_ref": "falkordb",
                        "uri": turn_id,
                        "text": text[:1800],
                        "ts": _to_epoch(ts_by_turn.get(turn_id)),
                        "tags": chat_source_tags(
                            client_meta, ["falkor", "chat", "chatturn", "entity_relatedness_injected"]
                        ),
                        "score": 0.50,
                        "meta": {},
                    }
                )

        # One batched call covering BOTH the recency-fetched candidates and
        # the newly-injected ones -- same precise per-turn scoring for both,
        # no hardcoded score for injected turns.
        all_turn_ids = sorted(existing_turn_ids | {f["uri"] for f in injected})
        boost_map: Dict[str, float] = {}
        if all_turn_ids:
            matches = await fetch_entity_matches_for_turns(
                turn_ids=all_turn_ids, target_names=list(target_scores.keys())
            )
            boost_map = {
                turn_id: max(target_scores.get(name, 0.0) for name in matched_names)
                for turn_id, matched_names in matches.items()
                if matched_names
            }

        return boost_map, injected
    except Exception as exc:
        logger.debug(f"entity relatedness boost: skipped: {exc}")
        return {}, []


def _rdf_enabled(profile: Dict[str, Any]) -> bool:
    profile_enable_rdf = bool(profile.get("enable_rdf", False))
    return (
        profile_enable_rdf or settings.RECALL_ENABLE_RDF
    ) and int(profile.get("rdf_top_k", 0)) > 0


def _sql_timeline_enabled_for_profile(profile: Dict[str, Any]) -> bool:
    if not settings.RECALL_ENABLE_SQL_TIMELINE:
        return False
    return bool(profile.get("enable_sql_timeline", True))


def _sql_chat_enabled_for_profile(profile: Dict[str, Any]) -> bool:
    if not settings.RECALL_ENABLE_SQL_CHAT:
        return False
    profile_name = str(profile.get("profile") or "")
    default_enabled = not profile_name.startswith("chat.general")
    return bool(profile.get("enable_sql_chat", default_enabled))


_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def _is_rdf_chatturn(frag: Dict[str, Any]) -> bool:
    # falkor_chat fragments also carry a "chatturn" tag (same recall-lane
    # semantics, different backend) but are windowed by
    # _window_sql_chat_candidates/_LOCAL_TS_ONLY_SOURCES instead, which
    # already has their accurate ts and doesn't need
    # _window_rdf_chatturn_candidates's Postgres round-trip. Excluded
    # explicitly rather than relying on _chatturn_id_from_fragment's
    # "/chatTurn/" uri-shape check to coincidentally not match falkor_chat's
    # plain-turn_id uri -- that was correct today but not guaranteed to stay
    # that way if either fragment shape changes independently later.
    if str(frag.get("source") or "") == "falkor_chat":
        return False
    if "chatturn" in {str(t).lower() for t in (frag.get("tags") or [])}:
        return True
    ref = str(frag.get("uri") or frag.get("id") or "")
    return "/chatTurn/" in ref


def _chatturn_id_from_fragment(frag: Dict[str, Any]) -> Optional[str]:
    """Recover the chat_history_log id from an RDF chat-turn fragment.

    RDF turn IRIs look like ``.../chatTurn/<uuid-with-underscores>`` because the writer
    sanitizes ``-`` to ``_``. Reverse that and validate the UUID shape so we only join
    ids we can trust.
    """
    ref = str(frag.get("uri") or frag.get("id") or "")
    if "/chatTurn/" not in ref:
        return None
    tail = ref.rsplit("/chatTurn/", 1)[-1].strip()
    if not tail:
        return None
    candidate = tail.replace("_", "-")
    return candidate if _UUID_RE.match(candidate) else None


async def _window_rdf_chatturn_candidates(
    candidates: List[Dict[str, Any]],
    *,
    since_minutes: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Drop RDF chat-turn candidates older than ``since_minutes``, stamping real timestamps.

    As of 2026-07-14, `storage/rdf_adapter.py`'s chat-turn fetch does select and order by a
    real `ORION.timestamp` literal, which feeds `scoring._compute_recency_factor`'s soft,
    continuous decay -- it no longer orders by an arbitrary UUID string sort. That fix does
    not replace this function's job, though: recency *scoring* down-weights old turns, it
    does not *exclude* them, and profiles that need a hard `since_minutes` cutoff (not just a
    lower score) still need turns outside the window dropped entirely. We resolve each turn's
    created_at from chat_history_log (the durable SQL source of truth) and keep only those
    inside the window. Non chat-turn candidates (SQL, cards, RDF claims, etc.) are returned
    untouched. Memory cards are never touched here.
    """
    if since_minutes <= 0:
        return candidates, 0
    turn_ids: Dict[int, str] = {}
    for idx, frag in enumerate(candidates):
        if not _is_rdf_chatturn(frag):
            continue
        cid = _chatturn_id_from_fragment(frag)
        if cid is not None:
            turn_ids[idx] = cid
    if not turn_ids:
        return candidates, 0

    try:
        ts_map = await fetch_chat_turn_timestamps(list(set(turn_ids.values())), since_minutes)
    except Exception as exc:  # pragma: no cover - defensive; never fail recall on this
        logger.debug("rdf chat-turn windowing skipped: %s", exc)
        return candidates, 0

    kept: List[Dict[str, Any]] = []
    dropped = 0
    for idx, frag in enumerate(candidates):
        if idx not in turn_ids:
            kept.append(frag)
            continue
        ts = ts_map.get(turn_ids[idx])
        if ts is None:
            # Outside the window or not resolvable to a chat row → drop from reflective recall.
            dropped += 1
            continue
        frag = dict(frag)
        frag["ts"] = ts
        kept.append(frag)
    return kept, dropped


_SQL_CHAT_SOURCES = frozenset({"sql_timeline", "sql_chat", "falkor_chat"})
# Sources within _SQL_CHAT_SOURCES that skip the Postgres row-id resolution
# below and go straight to the local-ts-cutoff fallback -- their ts is
# already accurate, so re-resolving it via fetch_chat_turn_timestamps would
# just be a second, redundant chat_history_log round-trip for the same ids.
_LOCAL_TS_ONLY_SOURCES = frozenset({"falkor_chat"})


def _sql_chat_row_id(candidate: Dict[str, Any]) -> Optional[str]:
    raw = str(candidate.get("id") or "").strip()
    if not raw:
        return None
    if _UUID_RE.match(raw):
        return raw
    if raw.startswith("chat_"):
        return None
    # chat_history_log rows sometimes use correlation_id-shaped ids
    if _UUID_RE.match(raw.replace("_", "-")):
        return raw.replace("_", "-")
    return None


async def _window_sql_chat_candidates(
    candidates: List[Dict[str, Any]],
    *,
    since_minutes: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Drop SQL chat/timeline (and Falkor chatturn) candidates outside ``since_minutes``.

    Mirrors ``_window_rdf_chatturn_candidates``: anchor-exact SQL rails used
    ``fetch_exact_fragments`` without a temporal filter, so old turns could surface
    into journal/metacog recall when expansion tokens matched generic words.
    Memory cards and non-SQL/non-falkor_chat sources are untouched.

    ``falkor_chat`` fragments (fetch_falkor_chatturn_fragments) carry a real
    ``ts`` already -- unlike RDF chatturn fragments, which need
    ``_window_rdf_chatturn_candidates``'s Postgres round-trip because the RDF
    graph has no usable timestamp, and unlike ``sql_chat``/``sql_timeline``
    fragments (deliberately excluded from the row-id resolution below via
    ``_LOCAL_TS_ONLY_SOURCES``): fetch_falkor_chatturn_fragments already did
    its own Postgres join for text and stamped an accurate ts from Falkor's
    ``ChatTurn.ts`` -- re-resolving it here via ``fetch_chat_turn_timestamps``
    would be a second, redundant chat_history_log round-trip for the same
    turn_ids on every windowed recall call. Falls straight to the
    local-``ts``-cutoff branch below instead, which is correct on its own
    since that ``ts`` is already trustworthy.
    """
    if since_minutes <= 0:
        return candidates, 0

    cutoff = time.time() - (int(since_minutes) * 60)
    sql_indices: Dict[int, str] = {}
    for idx, frag in enumerate(candidates):
        source = str(frag.get("source") or "")
        if source not in _SQL_CHAT_SOURCES or source in _LOCAL_TS_ONLY_SOURCES:
            continue
        row_id = _sql_chat_row_id(frag)
        if row_id is not None:
            sql_indices[idx] = row_id

    ts_map: Dict[str, float] = {}
    if sql_indices:
        try:
            ts_map = await fetch_chat_turn_timestamps(list(set(sql_indices.values())), since_minutes)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("sql chat windowing skipped: %s", exc)
            return candidates, 0

    kept: List[Dict[str, Any]] = []
    dropped = 0
    for idx, frag in enumerate(candidates):
        source = str(frag.get("source") or "")
        if source not in _SQL_CHAT_SOURCES:
            kept.append(frag)
            continue

        row_id = sql_indices.get(idx)
        if row_id is not None:
            ts = ts_map.get(row_id)
            if ts is None:
                dropped += 1
                continue
            frag = dict(frag)
            frag["ts"] = ts
            kept.append(frag)
            continue

        ts_val = frag.get("ts")
        try:
            ts_float = float(ts_val) if ts_val is not None else 0.0
        except Exception:
            ts_float = 0.0
        if ts_float <= 0.0 or ts_float < cutoff:
            dropped += 1
            continue
        kept.append(frag)

    return kept, dropped


async def _fetch_anchor_candidates(
    *,
    query_text: str,
    session_id: str | None,
    node_id: str | None,
    profile: Dict[str, Any],
    diagnostic: bool = False,
    exclusion: Dict[str, Any] | None = None,
    sink: List[Dict[str, Any]] | None = None,
    sink_key: Any = "anchor",
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Exact-token anchor rail. Each sub-fetch appends its result to ``sink``
    as it lands (same contract as _query_backends' units), so a deadline
    cancel keeps the SQL half even if the RDF half is still running."""
    tokens = _anchor_tokens(query_text)
    if not tokens:
        return [], {}

    def _to_sink(idx: int, name: str, cands: List[Dict[str, Any]], unit_counts: Dict[str, int], started: float) -> None:
        if sink is not None:
            sink.append(
                {
                    "key": sink_key,
                    "idx": idx,
                    "name": name,
                    "kind": _UNIT_RETRIEVER,
                    "candidates": list(cands),
                    "counts": dict(unit_counts),
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                }
            )

    candidates: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    limit = max(3, min(10, int(profile.get("sql_top_k", settings.RECALL_SQL_TOP_K))))
    exclusion = exclusion or {}
    since_minutes = int(profile.get("sql_since_minutes", settings.RECALL_SQL_SINCE_MINUTES))

    sql_started = time.perf_counter()
    try:
        sql_items = await fetch_exact_fragments(
            tokens=tokens,
            session_id=session_id,
            node_id=node_id,
            limit=limit,
            since_minutes=since_minutes,
            exclude_ids=exclusion.get("active_turn_ids"),
            exclude_text=exclusion.get("active_turn_text"),
        )
        counts["sql_timeline_anchor"] = len(sql_items)
        for item in sql_items:
            tags = list(item.tags or [])
            tags.append("anchor_exact")
            candidates.append(
                {
                    "id": item.id,
                    "source": "sql_timeline",
                    "source_ref": item.source_ref,
                    "text": item.text,
                    "ts": item.ts,
                    "tags": tags,
                    "score": 0.95,
                }
            )
    except Exception as exc:
        logger.debug(f"sql anchor fetch skipped: {exc}")
    counts["vector_anchor"] = 0
    _to_sink(0, "sql_timeline_anchor", candidates, counts, sql_started)

    if _rdf_enabled(profile) and settings.RECALL_RDF_ENDPOINT_URL:
        rdf_started = time.perf_counter()
        rdf_cands: List[Dict[str, Any]] = []
        try:
            # Synchronous requests.post (up to 5s): off the event loop, or it
            # stalls every concurrent backend unit, the deadline timer and
            # every other in-flight recall (code review, PR #2416).
            rdf = await asyncio.to_thread(
                fetch_rdf_chatturn_exact_matches,
                tokens=tokens,
                session_id=session_id,
                max_items=limit,
            )
            counts["rdf_chat_anchor"] = len(rdf)
            for item in rdf:
                item = dict(item)
                item["tags"] = list(item.get("tags") or []) + ["anchor_exact"]
                item["score"] = max(0.9, float(item.get("score") or 0.0))
                rdf_cands.append(item)
        except Exception as exc:
            logger.debug(f"rdf anchor fetch skipped: {exc}")
        candidates.extend(rdf_cands)
        _to_sink(1, "rdf_chat_anchor", rdf_cands, {"rdf_chat_anchor": len(rdf_cands)}, rdf_started)

    if diagnostic:
        logger.info(
            "anchor rail tokens=%s counts=%s",
            tokens,
            counts,
        )

    return candidates, counts


def _cards_fetch_enabled(profile: Dict[str, Any]) -> bool:
    if not bool(getattr(settings, "RECALL_ENABLE_CARDS", False)):
        return False
    w = profile.get("backend_weights")
    if not isinstance(w, dict):
        w = {}
    rel = profile.get("relevance")
    if isinstance(rel, dict) and isinstance(rel.get("backend_weights"), dict):
        w = {**w, **rel["backend_weights"]}
    try:
        wt = float(w.get("cards", 0.0) or 0.0)
    except Exception:
        wt = 0.0
    topk = int(profile.get("cards_top_k", 0) or 0)
    return topk > 0 or wt > 0.0


# Per-recall concurrency bound for backend units and the anchor rail (all
# sub-queries share it). Several units each open their own Postgres
# connection, so this is also the per-recall connection ceiling for the
# fetch stage. RECALL_FETCH_CONCURRENCY, default 4.
_FETCH_CONCURRENCY_DEFAULT = 4


def _fetch_concurrency() -> int:
    raw = getattr(settings, "RECALL_FETCH_CONCURRENCY", None)
    if raw is None:
        return _FETCH_CONCURRENCY_DEFAULT
    try:
        return max(1, int(raw))
    except Exception:
        return _FETCH_CONCURRENCY_DEFAULT

# Unit kinds. A "feed" answers "what is going on" and ignores the query text,
# so process_recall runs it once per recall; a "retriever" answers "what
# matches X" and runs once per sub-query.
_UNIT_FEED = "feed"
_UNIT_RETRIEVER = "retriever"


async def _query_backends(
    fragment: str,
    profile: Dict[str, Any],
    *,
    session_id: str | None,
    node_id: str | None,
    entities: List[str],
    diagnostic: bool = False,
    exclusion: Dict[str, Any] | None = None,
    lane: str | None = None,
    include_cards: bool = False,
    include_feeds: bool = True,
    include_retrievers: bool = True,
    falkor_chat_since_minutes: int | None = None,
    allow_empty_query_feeds: bool = False,
    semaphore: asyncio.Semaphore | None = None,
    sink: List[Dict[str, Any]] | None = None,
    sink_key: Any = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Fetch candidates from every enabled backend for one query text.

    Split 2026-09-29 (bounded-retrieval design) into ordered units, each
    tagged feed or retriever:

    - feeds (query-independent): bus_synaptic_anomaly, falkor_chat (recency
      only), sql_chat pairs/msgs, sql_timeline recent + related_by_entities
      (``entities`` is the recall-wide bounded list, not this sub-query).
    - retrievers (use ``fragment``): falkor_neighborhood, rdf_chat, rdf,
      cards (only with ``include_cards``), graph_compression.

    ``include_feeds``/``include_retrievers`` let process_recall run the feeds
    once per recall and the retrievers once per sub-query. With both True
    (the default) this is the old single-signal behavior: same gates, same
    backend order, same candidate order. Units run concurrently under
    ``semaphore``; each unit fails open on its own, and each completed unit
    is appended to ``sink`` (tagged ``sink_key``) as it lands, so a caller
    that cancels this coroutine at a deadline still keeps what finished.
    """
    exclusion = exclusion or {}
    units = _backend_units(
        fragment,
        profile,
        session_id=session_id,
        node_id=node_id,
        entities=entities,
        diagnostic=diagnostic,
        exclusion=exclusion,
        lane=lane,
        include_cards=include_cards,
        include_feeds=include_feeds,
        include_retrievers=include_retrievers,
        falkor_chat_since_minutes=falkor_chat_since_minutes,
        allow_empty_query_feeds=allow_empty_query_feeds,
    )
    sem = semaphore or asyncio.Semaphore(_fetch_concurrency())
    results: List[Tuple[List[Dict[str, Any]], Dict[str, int]] | None] = [None] * len(units)

    async def _run(idx: int, name: str, kind: str, factory) -> None:
        async with sem:
            started = time.perf_counter()
            try:
                cands, counts = await factory()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # each unit already fails open; belt and braces
                logger.debug("recall backend unit %s skipped: %s", name, exc)
                cands, counts = [], {}
            results[idx] = (cands, counts)
            if sink is not None:
                sink.append(
                    {
                        "key": sink_key,
                        "idx": idx,
                        "name": name,
                        "kind": kind,
                        "candidates": cands,
                        "counts": counts,
                        "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    }
                )

    await asyncio.gather(*(_run(i, n, k, f) for i, (n, k, f) in enumerate(units)))
    return _merge_unit_results(results, include_retrievers=include_retrievers, profile=profile)


def _merge_unit_results(
    results: List[Tuple[List[Dict[str, Any]], Dict[str, int]] | None],
    *,
    include_retrievers: bool,
    profile: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    candidates: List[Dict[str, Any]] = []
    backend_counts: Dict[str, int] = {}
    for res in results:
        if res is None:
            continue
        cands, counts = res
        candidates.extend(cands)
        for k, v in counts.items():
            backend_counts[k] = backend_counts.get(k, 0) + v
    if include_retrievers:
        backend_counts.setdefault("vector", 0)
        backend_counts.setdefault("graph_compression", 0)
    return candidates, backend_counts


def _backend_units(
    fragment: str,
    profile: Dict[str, Any],
    *,
    session_id: str | None,
    node_id: str | None,
    entities: List[str],
    diagnostic: bool,
    exclusion: Dict[str, Any],
    lane: str | None,
    include_cards: bool,
    include_feeds: bool,
    include_retrievers: bool,
    falkor_chat_since_minutes: int | None,
    allow_empty_query_feeds: bool,
) -> List[Tuple[str, str, Any]]:
    """Ordered (name, kind, coroutine-factory) list. Order matches the
    pre-split sequential code, so merged candidate order is unchanged."""
    units: List[Tuple[str, str, Any]] = []

    rdf_enabled = _rdf_enabled(profile) and bool(settings.RECALL_RDF_ENDPOINT_URL)
    rdf_top_k = int(profile.get("rdf_top_k", 0))
    # Phase 4 chatturn swap: independent of rdf_enabled/_rdf_enabled(profile)
    # (see settings.py's RECALL_FALKOR_IN_CHAT comment) -- Falkor chatturn
    # fetch doesn't need "RDF" enabled as a concept, and rdf_top_k still
    # bounds max_items since it's standing in for the same fetch. It DOES
    # still respect profile.get("enable_falkor_chat") though:
    # pcr_collectors.py::apply_collector_plan sets this False for PCR
    # intents (e.g. "procedural", "contradiction") that deliberately
    # suppress chat-turn content via plan.get("rdf_chat") -- without this
    # check, those intents would get chat-turn content back the moment
    # RECALL_FALKOR_IN_CHAT is on, defeating the suppression they were
    # designed to enforce regardless of which backend serves it.
    falkor_chat_enabled = bool(settings.RECALL_FALKOR_IN_CHAT) and bool(
        profile.get("enable_falkor_chat", True)
    )
    # Last live Fuseki read path in this service (verified live, 2026-07-22
    # -- see storage/falkor_neighborhood_adapter.py's docstring). Same
    # swap-not-additive convention as falkor_chat_enabled above -- and for
    # the same reason, deliberately NOT gated on rdf_enabled/RECALL_RDF_
    # ENDPOINT_URL: the whole point of this flag is to let Fuseki's endpoint
    # be removed from config entirely without silently killing this backend
    # too (review-caught: nesting it inside `if rdf_enabled:` made the swap
    # permanently inert the moment RECALL_RDF_ENDPOINT_URL was unset, which
    # is exactly the end-state this migration is working toward). Still
    # respects rdf_top_k>0 as the per-profile "wants graph-neighborhood
    # candidates at all" signal -- unlike falkor_chat, this isn't a
    # standalone concept from "rdf", it's a direct swap for that exact knob.
    falkor_neighborhood_enabled = (
        bool(settings.RECALL_FALKOR_NEIGHBORHOOD_IN_CHAT)
        and bool(profile.get("enable_falkor_neighborhood", True))
        and rdf_top_k > 0
    )

    if include_retrievers and falkor_neighborhood_enabled:

        async def _falkor_neighborhood():
            try:
                items = await fetch_falkor_neighborhood_fragments(query_text=fragment, max_items=rdf_top_k)
            except Exception as exc:
                logger.debug(f"falkor neighborhood fetch skipped: {exc}")
                items = []
            return list(items), {"falkor_neighborhood": len(items)}

        units.append(("falkor_neighborhood", _UNIT_RETRIEVER, _falkor_neighborhood))

    # Idea 4 of the bus synaptic graph arc (docs/superpowers/specs/2026-07-24-
    # bus-synaptic-graph-reasoning-consumer-design.md). Deliberately NOT gated
    # on query_text/fragment relevance like falkor_neighborhood above -- this
    # checks Orion's own live transport-layer state, not something "about"
    # what the user said, so it runs unconditionally whenever the flag and
    # profile allow it. A context feed: once per recall, not per sub-query
    # (it used to run 268 times for one 30k-char query).
    bus_synaptic_anomaly_enabled = bool(settings.RECALL_BUS_SYNAPTIC_ANOMALY_IN_CHAT) and bool(
        profile.get("enable_bus_synaptic_anomaly", True)
    )
    if include_feeds and bus_synaptic_anomaly_enabled:

        async def _bus_synaptic_anomaly():
            try:
                items = await fetch_bus_synaptic_anomaly_fragments(
                    max_edge_age_sec=float(settings.RECALL_BUS_SYNAPTIC_ANOMALY_MAX_AGE_SEC),
                )
            except Exception as exc:
                logger.debug(f"bus synaptic anomaly fetch skipped: {exc}")
                items = []
            return list(items), {"bus_synaptic_anomaly": len(items)}

        units.append(("bus_synaptic_anomaly", _UNIT_FEED, _bus_synaptic_anomaly))

    if include_feeds and falkor_chat_enabled:
        # Swap, not additive: this replaces the RDF chatturn fetch below,
        # not a merge -- running both would double up the same turns in
        # fusion's candidate list (see settings.py's RECALL_FALKOR_IN_CHAT
        # comment for why this differs from RECALL_GRAPHITI_IN_CHAT's
        # additive pattern). Recency-only (no query filter), so a feed.

        async def _falkor_chat():
            try:
                kwargs: Dict[str, Any] = {
                    "query_text": fragment,
                    "session_id": session_id,
                    "max_items": max(rdf_top_k, 6),
                }
                if falkor_chat_since_minutes is not None:
                    kwargs["since_minutes"] = falkor_chat_since_minutes
                if allow_empty_query_feeds:
                    kwargs["allow_empty_query"] = True
                items = await fetch_falkor_chatturn_fragments(**kwargs)
            except Exception as exc:
                logger.debug(f"falkor chat fetch skipped: {exc}")
                items = []
            return list(items), {"falkor_chat": len(items)}

        units.append(("falkor_chat", _UNIT_FEED, _falkor_chat))

    if include_retrievers and rdf_enabled:
        # The RDF adapters are synchronous (blocking HTTP); off the event
        # loop so the recall deadline can still fire while they run.
        if not falkor_chat_enabled:
            # 0) Pull raw ChatTurns (prompt/response) from GRAPH <orion:chat>.
            # Skipped when Falkor already covered this above (swap, not
            # additive). Keyword-filtered in SPARQL, so a retriever.

            async def _rdf_chat():
                try:
                    items = await asyncio.to_thread(
                        fetch_rdf_chatturn_fragments,
                        query_text=fragment,
                        session_id=session_id,
                        max_items=max(rdf_top_k, 6),
                    )
                except Exception as exc:
                    logger.debug(f"rdf backend skipped: {exc}")
                    return [], {}
                return list(items), {"rdf_chat": len(items)}

            units.append(("rdf_chat", _UNIT_RETRIEVER, _rdf_chat))

        if not falkor_neighborhood_enabled:

            async def _rdf():
                try:
                    items = await asyncio.to_thread(fetch_rdf_fragments, query_text=fragment, max_items=rdf_top_k)
                except Exception as exc:
                    logger.debug(f"rdf backend skipped: {exc}")
                    return [], {}
                return list(items), {"rdf": len(items)}

            units.append(("rdf", _UNIT_RETRIEVER, _rdf))

    if diagnostic and include_retrievers:
        logger.info(
            "recall rdf_enabled=%s rdf_top_k=%s",
            rdf_enabled,
            rdf_top_k,
        )

    if include_feeds and _sql_chat_enabled_for_profile(profile):

        async def _sql_chat():
            cands: List[Dict[str, Any]] = []
            counts: Dict[str, int] = {}
            try:
                chat_pairs = await fetch_chat_history_pairs(
                    limit=int(profile.get("sql_chat_top_k", settings.RECALL_SQL_TOP_K)),
                    since_minutes=int(profile.get("sql_since_minutes", settings.RECALL_SQL_SINCE_MINUTES)),
                    exclude_text=exclusion.get("active_turn_text"),
                    exclude_ids=exclusion.get("active_turn_ids"),
                )
                counts["sql_chat_pairs"] = len(chat_pairs)
                for item in chat_pairs:
                    cands.append(
                        {
                            "id": item.id,
                            "source": "sql_chat",
                            "source_ref": item.source_ref,
                            "text": item.text,
                            "ts": item.ts,
                            "tags": ["sql", "chat", "pairs"],
                            "score": 0.75,
                        }
                    )

                chat_msgs = await fetch_chat_messages(
                    limit=int(profile.get("sql_chat_top_k", settings.RECALL_SQL_TOP_K)),
                    since_minutes=int(profile.get("sql_since_minutes", settings.RECALL_SQL_SINCE_MINUTES)),
                    exclude_text=exclusion.get("active_turn_text"),
                    exclude_ids=exclusion.get("active_turn_ids"),
                )
                counts["sql_chat_msgs"] = len(chat_msgs)
                for item in chat_msgs:
                    cands.append(
                        {
                            "id": item.id,
                            "source": "sql_chat",
                            "source_ref": item.source_ref,
                            "text": item.text,
                            "ts": item.ts,
                            "tags": ["sql", "chat", "messages"],
                            "score": 0.75,
                        }
                    )
            except Exception as exc:
                logger.debug(f"sql chat backend skipped: {exc}")
            return cands, counts

        units.append(("sql_chat", _UNIT_FEED, _sql_chat))
    elif include_feeds and diagnostic:
        logger.info(
            "recall sql_chat skipped profile=%s enable_sql_chat=%s global_sql_chat_enabled=%s",
            profile.get("profile"),
            profile.get("enable_sql_chat"),
            settings.RECALL_ENABLE_SQL_CHAT,
        )

    if include_feeds and _sql_timeline_enabled_for_profile(profile):

        async def _sql_timeline():
            cands: List[Dict[str, Any]] = []
            counts: Dict[str, int] = {}
            try:
                since_minutes_effective = int(profile.get("sql_since_minutes", settings.RECALL_SQL_SINCE_MINUTES))
                since_hours_effective = int(profile.get("sql_since_hours", max(1, since_minutes_effective // 60)))
                sql_top_k = int(profile.get("sql_top_k", settings.RECALL_SQL_TOP_K))

                recent_items = await fetch_recent_fragments(
                    session_id,
                    node_id,
                    since_minutes_effective,
                    sql_top_k,
                    exclude_ids=exclusion.get("active_turn_ids"),
                    exclude_text=exclusion.get("active_turn_text"),
                )
                # ``entities`` is the recall-wide bounded list (at most
                # RECALL_MAX_SUB_QUERIES ILIKE patterns), computed once --
                # it used to be every capitalized word of the full query,
                # recomputed and re-queried once per sub-query.
                related_items = await fetch_related_by_entities(
                    entities,
                    since_hours_effective,
                    sql_top_k,
                    session_id=session_id,
                    exclude_ids=exclusion.get("active_turn_ids"),
                    exclude_text=exclusion.get("active_turn_text"),
                )

                all_items = list(recent_items) + list(related_items)
                counts["sql_timeline"] = len(all_items)
                for item in all_items:
                    cands.append(
                        {
                            "id": item.id,
                            "source": "sql_timeline",
                            "source_ref": item.source_ref,
                            "text": item.text,
                            "ts": item.ts,
                            "session_id": item.session_id,
                            "tags": item.tags,
                            "turn_effect_delta": item.turn_effect_delta,
                            "score": 0.7,
                        }
                    )
            except Exception as exc:
                logger.debug(f"sql timeline backend skipped: {exc}")
            return cands, counts

        units.append(("sql_timeline", _UNIT_FEED, _sql_timeline))
    elif include_feeds and diagnostic:
        logger.info(
            "recall sql_timeline skipped profile=%s enable_sql_timeline=%s global_sql_timeline_enabled=%s",
            profile.get("profile"),
            profile.get("enable_sql_timeline"),
            settings.RECALL_ENABLE_SQL_TIMELINE,
        )

    if include_retrievers and include_cards and _cards_fetch_enabled(profile):
        pool = _recall_pg_pool
        if pool is not None and asyncpg is not None:

            async def _cards():
                try:
                    card_frags = await fetch_card_fragments_guarded(
                        pool,
                        fragment,
                        profile,
                        lane=lane,
                        timeout_sec=float(getattr(settings, "RECALL_CARDS_TIMEOUT_SEC", 0.25) or 0.25),
                        max_neighbors=int(getattr(settings, "RECALL_CARDS_MAX_NEIGHBORS", 6) or 6),
                    )
                except Exception as exc:
                    logger.warning("cards fetch skipped: %s", exc)
                    return [], {}
                return list(card_frags), {"cards": len(card_frags)}

            units.append(("cards", _UNIT_RETRIEVER, _cards))
        elif diagnostic:
            logger.info("recall cards skipped pool_asyncpg_available=%s", pool is not None)

    # ── Graph Compression backend ─────────────────────────────────────────────
    compression_enabled = (
        bool(profile.get("enable_graph_compression"))
        and bool(getattr(settings, "RECALL_COMPRESSION_ENABLED", False))
        and bool(getattr(settings, "RECALL_COMPRESSION_PG_DSN", None))
        and fetch_graph_compression_fragments is not None
    )
    if include_retrievers and compression_enabled:

        async def _graph_compression():
            try:
                # Run the blocking Postgres + Fuseki I/O off the event loop so it does
                # not stall the recall hot path (mirrors the memory_graph_sparql path).
                compression_frags = await asyncio.to_thread(
                    fetch_graph_compression_fragments,
                    query_text=fragment,
                    mode=str(profile.get("compression_mode") or "unified"),
                    max_global=int(profile.get("compression_global_top_k") or 5),
                    max_local=int(profile.get("compression_local_top_k") or 5),
                    # self_study dropped from the default 2026-07-23: orion-graph-compression
                    # retired the scope entirely (live-verified zero communities/artifacts,
                    # ever -- its three source Fuseki graphs have always been empty).
                    scopes=list(profile.get("compression_scopes") or ["episodic", "substrate"]),
                    pg_dsn=settings.RECALL_COMPRESSION_PG_DSN,
                    rdf_query_url=getattr(settings, "RECALL_COMPRESSION_RDF_QUERY_URL", None),
                    rdf_user=getattr(settings, "RECALL_COMPRESSION_RDF_USER", "admin"),
                    rdf_pass=getattr(settings, "RECALL_COMPRESSION_RDF_PASS", "orion"),
                    timeout_sec=float(getattr(settings, "RECALL_COMPRESSION_TIMEOUT_SEC", 3.0)),
                )
            except Exception as exc:
                logger.debug("graph_compression_backend_skipped reason=%s", exc)
                return [], {"graph_compression": 0}
            return list(compression_frags), {"graph_compression": len(compression_frags)}

        units.append(("graph_compression", _UNIT_RETRIEVER, _graph_compression))

    return units


_telemetry_table_ready = False
_telemetry_failure_warned = False
# Names of the bounded-retrieval columns this process has confirmed exist
# (found in information_schema, or added by our own ALTER). The insert only
# writes these, so a failed/timed-out DDL degrades to the pre-2026-09-29 row.
_telemetry_present_columns: set = set()

_TELEMETRY_BOUNDED_RETRIEVAL_COLUMNS = (
    "query_chars integer",
    "retrieval_query_source text",
    "sub_query_count integer",
    "candidates_fetched integer",
    "candidates_kept integer",
    "deadline_hit boolean",
    "timings_ms jsonb",
)


_TELEMETRY_BOUNDED_RETRIEVAL_COLUMN_NAMES = tuple(c.split()[0] for c in _TELEMETRY_BOUNDED_RETRIEVAL_COLUMNS)


def _ensure_telemetry_schema(cur: Any) -> None:
    """CREATE TABLE if missing, then ALTER only the columns that are missing.

    Code review, PR #2416: `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` takes
    ACCESS EXCLUSIVE even when the column already exists, so running it on
    every boot queues behind a backup's lock and every later INSERT queues
    behind the ALTER (the 09-xx boot-hang pattern). So: read
    information_schema first (no table lock), ALTER only what is absent, and
    bound the DDL with lock_timeout/statement_timeout. Any failure is logged
    once and swallowed; the caller's insert then writes only confirmed
    columns.
    """
    global _telemetry_present_columns
    try:
        cur.execute("SET lock_timeout = '2s'")
        cur.execute("SET statement_timeout = '5s'")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recall_telemetry (
                id uuid primary key,
                corr_id text,
                session_id text,
                node_id text,
                verb text,
                profile text,
                query text,
                selected_ids jsonb,
                backend_counts jsonb,
                latency_ms integer,
                created_at timestamptz default now()
            )
            """
        )
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = 'recall_telemetry'"
        )
        existing = {str(r[0]) for r in (cur.fetchall() or [])}
        present = {c for c in _TELEMETRY_BOUNDED_RETRIEVAL_COLUMN_NAMES if c in existing}
        _telemetry_present_columns = set(present)
        # Bounded-retrieval columns (2026-09-29). Nullable and additive.
        # Kept in sync with sql/recall_telemetry.sql.
        for column_ddl in _TELEMETRY_BOUNDED_RETRIEVAL_COLUMNS:
            name = column_ddl.split()[0]
            if name in present:
                continue
            cur.execute(f"ALTER TABLE recall_telemetry ADD COLUMN IF NOT EXISTS {column_ddl}")
            _telemetry_present_columns.add(name)
    except Exception as exc:
        logger.warning(
            "recall_telemetry_schema_ddl_failed (not retried this process; inserting confirmed columns only: %s): %s",
            sorted(_telemetry_present_columns),
            exc,
        )
    # The timeouts stay set for this (per-call) connection, so the insert
    # that follows is bounded the same way.


def _persist_decision(decision: RecallDecisionV1) -> None:
    """
    Durable log to Postgres if available. Best-effort, blocking (psycopg2):
    callers on the event loop go through ``persist_decision_async``.

    jsonb columns take ``psycopg2.extras.Json``: psycopg2 cannot adapt a raw
    dict, so before 2026-09-29 every insert raised "can't adapt type 'dict'"
    and the failure was logged at debug level -- recall_telemetry had zero
    rows ever. The first failure per process is now a warning.
    """
    global _telemetry_table_ready, _telemetry_failure_warned
    dsn = settings.RECALL_PG_DSN
    if not dsn:
        return
    if psycopg2 is None:
        return
    from psycopg2.extras import Json  # type: ignore

    conn = None
    try:
        # Bounded: the bus handler awaits this before replying, so a hung
        # connect must not hold a recall reply hostage.
        conn = psycopg2.connect(dsn, connect_timeout=3)
        conn.autocommit = True
        with conn.cursor() as cur:
            if not _telemetry_table_ready:
                # Once per process, success or failure: a failed DDL is never
                # retried per request, and never blocks the insert below.
                _telemetry_table_ready = True
                _ensure_telemetry_schema(cur)
            new_cols = [c for c in _TELEMETRY_BOUNDED_RETRIEVAL_COLUMN_NAMES if c in _telemetry_present_columns]
            values_by_col = {
                "query_chars": decision.query_chars,
                "retrieval_query_source": decision.retrieval_query_source,
                "sub_query_count": decision.sub_query_count,
                "candidates_fetched": decision.candidates_fetched,
                "candidates_kept": decision.candidates_kept,
                "deadline_hit": decision.deadline_hit,
                "timings_ms": Json(dict(decision.timings_ms or {})),
            }
            base_cols = [
                "id", "corr_id", "session_id", "node_id", "verb", "profile", "query",
                "selected_ids", "backend_counts", "latency_ms",
            ]
            params = [
                decision.id,
                decision.corr_id,
                decision.session_id,
                decision.node_id,
                decision.verb,
                decision.profile,
                decision.query,
                Json(decision.selected_ids),
                Json(decision.backend_counts),
                decision.latency_ms,
            ] + [values_by_col[c] for c in new_cols]
            cols = base_cols + new_cols
            cur.execute(
                f"""
                INSERT INTO recall_telemetry
                ({", ".join(cols)})
                VALUES ({",".join(["%s"] * len(cols))})
                ON CONFLICT (id) DO NOTHING
                """,
                tuple(params),
            )
    except Exception as exc:
        if not _telemetry_failure_warned:
            _telemetry_failure_warned = True
            logger.warning("recall_telemetry_persist_failed (further failures at debug): %s", exc)
        else:
            logger.debug("recall_telemetry_persist_failed: %s", exc)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


async def persist_decision_async(decision: RecallDecisionV1) -> None:
    """Run the blocking psycopg2 write off the event loop, so one recall's
    telemetry connect+insert never stalls every other in-flight request."""
    await asyncio.to_thread(_persist_decision, decision)


def _log_debug_dump(
    *,
    corr_id: str,
    profile: Dict[str, Any],
    backend_counts: Dict[str, int],
    items: List[Any],
) -> None:
    top_n = int(getattr(settings, "RECALL_DEBUG_DUMP_TOP_N", 0) or 0)
    if top_n <= 0:
        return
    logger.info(
        "REC_TAPE RECALL corr_id=%s profile=%s backend_counts=%s selected_count=%s",
        corr_id,
        profile.get("profile"),
        backend_counts,
        len(items),
    )
    for idx, item in enumerate(items[:top_n]):
        source = getattr(item, "source", None) or (item.get("source") if isinstance(item, dict) else None)
        item_id = getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None)
        score = (
            getattr(item, "score", None)
            if hasattr(item, "score")
            else (item.get("score") if isinstance(item, dict) else None)
        )
        source_ref = getattr(item, "source_ref", None) or (item.get("source_ref") if isinstance(item, dict) else None)
        snippet = getattr(item, "snippet", None) or (item.get("text") if isinstance(item, dict) else None)
        snippet_head = str(snippet or "")[:160].replace("\n", " ")
        logger.info(
            "REC_TAPE RECALL item idx=%s source=%s id=%s score=%s source_ref=%s snippet_head=%r",
            idx,
            source,
            item_id,
            score,
            source_ref,
            snippet_head,
        )


def _bounded_selected_summary(items: List[Any], *, limit: int = 8) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for item in items[: max(1, limit)]:
        source = getattr(item, "source", None) or (item.get("source") if isinstance(item, dict) else None)
        item_id = getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None)
        score = getattr(item, "score", None) if hasattr(item, "score") else (item.get("score") if isinstance(item, dict) else None)
        source_ref = getattr(item, "source_ref", None) or (item.get("source_ref") if isinstance(item, dict) else None)
        summary.append(
            {
                "id": str(item_id or ""),
                "source": str(source or "unknown"),
                "score": float(score or 0.0),
                "source_ref": str(source_ref or "")[:120] or None,
            }
        )
    return summary


def _recall_deadline_budget_ms(q: RecallQueryV1) -> int:
    """Fetch budget in ms: 80% of the caller's deadline_ms when given, else
    RECALL_DEADLINE_MS_DEFAULT. <= 0 means no deadline."""
    caller = getattr(q, "deadline_ms", None)
    if caller is not None and int(caller) > 0:
        return max(1, int(int(caller) * 0.8))
    raw = getattr(settings, "RECALL_DEADLINE_MS_DEFAULT", None)
    if raw is None:
        return 60000
    try:
        return int(raw)
    except Exception:
        return 60000


async def _run_pcr_collectors(
    q: RecallQueryV1,
    *,
    pcr_backend_plan: Dict[str, bool],
    remaining_s: float | None,
    corr_id: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int], bool]:
    """Run the purposeful-recall PCR collectors under the recall deadline.

    Returns (candidates, backend_counts, elapsed_ms_per_collector,
    deadline_hit). Both collectors run concurrently and share whatever is left
    of the overall recall deadline (``remaining_s``; None = no deadline). A
    collector still running at the deadline is cancelled and its result
    dropped; the recall never fails because of it. concept_region's work runs
    in a thread (asyncio.to_thread) that cannot be killed: cancelling only
    stops the recall from waiting on it, the thread finishes in the
    background.

    Before 2026-09-30 this block ran after the deadline-bounded fetch, with
    no deadline and no timing: a cold get_substrate_store() (first-call
    Falkor hydration, measured 6.25s live) made the first belief recall after
    a restart take 9.5s with ~9.3s missing from timings_ms. Since 2026-10-06
    recall never hydrates: concept_region issues bounded direct Falkor reads
    (see app/substrate_store.py).
    """
    units: List[Tuple[str, Any]] = []
    # Set when the recall stops waiting for concept_region (deadline or
    # cancellation). The thread cannot be killed, so the collector checks this
    # right before its reinforcement write: dropped fragments must not be
    # reinforced as if they had been surfaced.
    cr_abandoned = threading.Event()
    on_cancel: Dict[str, threading.Event] = {"concept_region": cr_abandoned}
    if pcr_backend_plan.get("active_packet") and settings.RECALL_ACTIVE_PACKET_ENABLED:
        units.append(
            (
                "active_packet",
                lambda: fetch_active_packet_fragments(q, pool=_recall_pg_pool, settings=settings),
            )
        )
    if pcr_backend_plan.get("concept_region") and settings.RECALL_CONCEPT_REGION_ENABLED:
        # fetch_concept_region_fragment_and_reinforce's Falkor reads/write
        # are blocking network calls (bounded by the socket timeouts in
        # app/substrate_store.py) -- they must run inside the offloaded
        # thread, and so does get_substrate_store() for symmetry. The inner
        # lambda defers both calls into the thread (an argument expression
        # would be evaluated on the event loop). The collector also writes a
        # small activation bump for whatever it matched (see
        # collectors/CONCEPT_REINFORCEMENT_DESIGN.md), on the same thread.
        units.append(
            (
                "concept_region",
                lambda: asyncio.to_thread(
                    lambda: fetch_concept_region_fragment_and_reinforce(
                        q, store=get_substrate_store(), abandoned=cr_abandoned
                    )
                ),
            )
        )
    if not units:
        return [], {}, {}, False
    if remaining_s is not None and remaining_s <= 0:
        logger.warning(
            "recall_deadline_hit corr_id=%s stage=pcr_collectors skipped=%s",
            corr_id,
            [name for name, _f in units],
        )
        return [], {}, {name: 0 for name, _f in units}, True

    elapsed_ms: Dict[str, int] = {}

    async def _timed(name: str, factory: Any) -> Any:
        started = time.perf_counter()
        try:
            return await factory()
        except asyncio.CancelledError:
            ev = on_cancel.get(name)
            if ev is not None:
                ev.set()
            raise
        finally:
            elapsed_ms[name] = int((time.perf_counter() - started) * 1000)

    tasks: List[Tuple[str, asyncio.Future]] = []
    for name, factory in units:
        try:
            tasks.append((name, asyncio.ensure_future(_timed(name, factory))))
        except Exception as exc:  # one broken collector must not strand the other
            logger.debug("%s collector could not start: %s", name, exc)
    deadline_hit = False
    if tasks:
        _done, pending = await asyncio.wait(
            [t for _n, t in tasks],
            timeout=(max(0.0, remaining_s) if remaining_s is not None else None),
        )
        if pending:
            deadline_hit = True
            for n, t in tasks:
                if t in pending and n in on_cancel:
                    # Before cancel(), so the flag is up even if the
                    # thread returns before the cancellation is delivered.
                    on_cancel[n].set()
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            logger.warning(
                "recall_deadline_hit corr_id=%s stage=pcr_collectors pending=%s",
                corr_id,
                [n for n, t in tasks if t in pending],
            )

    candidates: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for name, task in tasks:
        if task.cancelled():
            continue
        exc = task.exception()
        if exc is not None:
            logger.debug("%s collector skipped: %s", name, exc)
            continue
        frags = list(task.result() or [])
        candidates.extend(frags)
        counts[name] = len(frags)
    return candidates, counts, elapsed_ms, deadline_hit


async def process_recall(
    q: RecallQueryV1,
    *,
    corr_id: str,
    diagnostic: bool = False,
) -> Tuple[MemoryBundleV1, RecallDecisionV1]:
    # Whole-recall clock: the deadline and latency_ms both run from here
    # (latency_ms used to stop before the entity boost and fusion).
    t0 = time.time()
    recall_started = time.perf_counter()
    timings_ms: Dict[str, int] = {}
    recall_phase = getattr(q, "recall_phase", None)
    selected_profile = q.profile
    intent_payload: Dict[str, Any] | None = None
    if bool(getattr(settings, "RECALL_INTENT_ROUTING_ENABLED", True)):
        try:
            from .intent import classify_intent_v1, intent_telemetry_payload, resolve_profile_for_intent

            profile_explicit = bool(getattr(q, "profile_explicit", False)) or recall_phase in {
                "continuity",
                "purposeful",
            }
            ic = classify_intent_v1(str(q.fragment or ""))
            if profile_explicit:
                selected_profile = q.profile
            else:
                selected_profile = resolve_profile_for_intent(ic.intent, fallback_profile=q.profile)
            intent_payload = intent_telemetry_payload(
                query_text=str(q.fragment or ""),
                intent=ic.intent,
                profile=selected_profile,
                override=profile_explicit,
            )
        except Exception as exc:
            logger.debug("intent routing skipped: %s", exc)
            selected_profile = q.profile
            intent_payload = None

    profile = get_profile(selected_profile)
    profile_name = str(profile.get("profile") or "")
    retrieval_intent = getattr(q, "retrieval_intent", None) or "semantic"
    pcr_backend_plan: dict[str, bool] = {}
    if settings.RECALL_PCR_ENABLED and recall_phase == "purposeful":
        pcr_backend_plan = collectors_for_intent(retrieval_intent)
        profile = apply_collector_plan(profile, pcr_backend_plan)
        profile_name = str(profile.get("profile") or profile_name)
    intake = _intake_query(q, profile_name=profile_name)
    query_targeting = intake["query_targeting"]
    query_fragment = str(intake["search_text"] or "")
    retrieval_query_source = str(intake["source"])
    context_only = str(getattr(q, "mode", "retrieve") or "retrieve") == "context_only"
    if diagnostic and query_targeting.get("query_changed"):
        logger.info(
            "recall query_targeting adjusted profile=%s verb=%s raw=%r targeted=%r turn_type=%s tail_stripped=%s",
            profile_name,
            q.verb,
            (q.fragment or "")[:220],
            query_fragment[:220],
            query_targeting.get("turn_type"),
            query_targeting.get("tail_stripped"),
        )
    enable_qe = bool(profile.get("enable_query_expansion", True))
    max_sub_queries = _max_sub_queries()
    # One bounded entity list per recall: sub-queries, sql_timeline's
    # related_by_entities patterns and the entity boost all read it (the
    # related_by_entities list used to be re-extracted from the full query on
    # every sub-query iteration).
    bounded_entities: List[str] = [] if context_only else _ranked_entities(query_fragment, limit=max_sub_queries)
    if context_only:
        signals: List[str] = []
    else:
        signals = _expand_query(
            query_fragment,
            verb=q.verb,
            intent=q.intent,
            enable=enable_qe,
            max_sub_queries=max_sub_queries,
            entities=bounded_entities,
        )
    timings_ms["intake"] = int((time.perf_counter() - recall_started) * 1000)
    ignored_session_id = q.session_id
    effective_session_id: str | None = None
    exclusion = _parse_exclusion(q)
    source_gating: Dict[str, str] = {}
    vector_policy = build_vector_policy(profile, settings)
    source_gating["vector"] = "removed_from_orion_recall"
    source_gating["sql_timeline"] = "enabled" if _sql_timeline_enabled_for_profile(profile) else "disabled_by_profile_or_global"
    source_gating["sql_chat"] = "enabled" if _sql_chat_enabled_for_profile(profile) else "disabled_by_profile_or_global"
    source_gating["rdf"] = "enabled" if _rdf_enabled(profile) else "disabled_by_profile_or_global"
    source_gating["falkor_chat"] = (
        "enabled"
        if bool(settings.RECALL_FALKOR_IN_CHAT) and bool(profile.get("enable_falkor_chat", True))
        else "disabled_by_profile_or_global"
    )

    timing_breakdown_ms: Dict[str, int] = {}
    candidates: List[Dict[str, Any]] = []
    backend_counts_total: Dict[str, int] = {}

    if not context_only and _is_memory_browse(query_fragment):
        since_minutes_effective = int(profile.get("sql_since_minutes", settings.RECALL_SQL_SINCE_MINUTES))
        browse_limit = max(10, min(20, int(profile.get("max_total_items", 12))))
        if _sql_timeline_enabled_for_profile(profile):
            source_gating["sql_timeline"] = "enabled"
            try:
                recent_items = await fetch_recent_fragments(
                    effective_session_id,
                    q.node_id,
                    since_minutes_effective,
                    browse_limit,
                    exclude_ids=exclusion.get("active_turn_ids"),
                    exclude_text=exclusion.get("active_turn_text"),
                )
            except Exception as exc:
                logger.debug(f"browse timeline fetch skipped: {exc}")
                recent_items = []
        else:
            source_gating["sql_timeline"] = "disabled_by_profile_or_global"
            logger.info(
                "browse sql_timeline skipped profile=%s enable_sql_timeline=%s global_sql_timeline_enabled=%s",
                profile.get("profile"),
                profile.get("enable_sql_timeline"),
                settings.RECALL_ENABLE_SQL_TIMELINE,
            )
            recent_items = []
        backend_counts_total["sql_timeline"] = len(recent_items)
        for item in recent_items:
            candidates.append(
                {
                    "id": item.id,
                    "source": "sql_timeline",
                    "source_ref": item.source_ref,
                    "text": item.text,
                    "ts": item.ts,
                    "tags": list(item.tags or []) + ["memory_browse"],
                    "score": 0.6,
                }
            )
        latency_ms = int((time.time() - t0) * 1000)
        bundle, ranking_debug = fuse_candidates(
            candidates=candidates,
            profile=profile,
            latency_ms=latency_ms,
            query_text=None,
            session_id=None,
            diagnostic=diagnostic,
            browse_mode=True,
        )
        _log_debug_dump(
            corr_id=corr_id,
            profile=profile,
            backend_counts=backend_counts_total or bundle.stats.backend_counts,
            items=list(bundle.items),
        )
        decision = RecallDecisionV1(
            corr_id=corr_id or str(uuid4()),
            session_id=ignored_session_id,
            node_id=q.node_id,
            verb=q.verb,
            profile=str(profile.get("profile") or q.profile),
            query=q.fragment,
            selected_ids=[i.id for i in bundle.items],
            backend_counts=backend_counts_total or bundle.stats.backend_counts,
            latency_ms=latency_ms,
            query_chars=len(query_fragment),
            retrieval_query_source=retrieval_query_source,
            sub_query_count=0,
            candidates_fetched=len(candidates),
            candidates_kept=len(candidates),
            deadline_hit=False,
            timings_ms={**timings_ms, "total": latency_ms},
            dropped=dict((bundle.stats.diagnostic or {}).get("drop_counts") or {}),
            ranking_debug=ranking_debug if diagnostic else [],
            recall_debug=(
                {
                    "profile_selected": str(profile.get("profile") or q.profile),
                    "profile_requested": q.profile,
                    **({"recall_intent": intent_payload} if intent_payload else {}),
                    "query_expansion_enabled": enable_qe,
                    "query_targeting": {
                        **query_targeting,
                        "raw_fragment": q.fragment,
                    },
                    "source_gating": source_gating,
                    "vector_policy": vector_policy,
                    "active_turn": {
                        "ids_count": len(list(exclusion.get("active_turn_ids") or [])),
                        "text_present": bool(str(exclusion.get("active_turn_text") or "").strip()),
                        "ts_present": exclusion.get("active_turn_ts") is not None,
                        "self_hit_suppressed": 0,
                    },
                    "fusion": bundle.stats.diagnostic or {},
                    "latency_breakdown_ms": {"total": latency_ms},
                    "selected_summary": _bounded_selected_summary(list(bundle.items)),
                }
                if diagnostic
                else {}
            ),
        )
        pressure_events = _build_recall_pressure_events(q=q, decision=decision, bundle=bundle)
        if pressure_events:
            merged_debug = dict(decision.recall_debug or {})
            merged_debug["pressure_events"] = pressure_events
            decision = decision.model_copy(update={"recall_debug": merged_debug})
        _log_debug_dump(
            corr_id=decision.corr_id,
            profile=profile,
            backend_counts=decision.backend_counts or {},
            items=list(bundle.items),
        )
        return bundle, decision

    # ── Fetch: feeds once, retrievers per sub-query, concurrently, one deadline ──
    sql_chat_window_min = int(
        profile.get("sql_chat_since_minutes")
        or profile.get("sql_since_minutes")
        or settings.RECALL_SQL_SINCE_MINUTES
    )
    deadline_budget_ms = _recall_deadline_budget_ms(q)
    deadline_at = recall_started + deadline_budget_ms / 1000.0 if deadline_budget_ms > 0 else None

    def _remaining_s() -> float | None:
        return None if deadline_at is None else deadline_at - time.perf_counter()

    fetch_started = time.perf_counter()
    semaphore = asyncio.Semaphore(_fetch_concurrency())
    sink: List[Dict[str, Any]] = []
    jobs: List[Tuple[Any, Any]] = []
    anchor_elapsed_ms: List[int] = []

    if not context_only and bool(profile.get("enable_anchor_candidates", True)):

        async def _anchor_job():
            # Same semaphore as the backend units: it opens Postgres (and
            # possibly RDF) connections too.
            async with semaphore:
                started = time.perf_counter()
                try:
                    return await _fetch_anchor_candidates(
                        query_text=query_fragment,
                        session_id=effective_session_id,
                        node_id=q.node_id,
                        profile=profile,
                        diagnostic=diagnostic,
                        exclusion=exclusion,
                        sink=sink,
                        sink_key="anchor",
                    )
                finally:
                    anchor_elapsed_ms.append(int((time.perf_counter() - started) * 1000))

        jobs.append(("anchor", _anchor_job))
    elif diagnostic:
        logger.info(
            "recall anchor rail skipped profile=%s enable_anchor_candidates=%s context_only=%s",
            profile.get("profile"),
            profile.get("enable_anchor_candidates"),
            context_only,
        )

    common_kwargs: Dict[str, Any] = {
        "session_id": effective_session_id,
        "node_id": q.node_id,
        "entities": bounded_entities,
        "diagnostic": diagnostic,
        "exclusion": exclusion,
        "lane": getattr(q, "lane", None),
        "falkor_chat_since_minutes": sql_chat_window_min,
        "semaphore": semaphore,
        "sink": sink,
    }
    if signals:
        for sig_i, sig in enumerate(signals):
            jobs.append(
                (
                    sig_i,
                    functools.partial(
                        _query_backends,
                        sig,
                        profile,
                        include_cards=(sig_i == 0),
                        include_feeds=(sig_i == 0),
                        include_retrievers=True,
                        sink_key=sig_i,
                        **common_kwargs,
                    ),
                )
            )
    else:
        # context_only (or nothing to search for): feeds only, once.
        jobs.append(
            (
                0,
                functools.partial(
                    _query_backends,
                    query_fragment,
                    profile,
                    include_cards=False,
                    include_feeds=True,
                    include_retrievers=False,
                    allow_empty_query_feeds=context_only,
                    sink_key=0,
                    **common_kwargs,
                ),
            )
        )

    job_tasks: List[Tuple[Any, asyncio.Future]] = []
    for key, factory in jobs:
        try:
            job_tasks.append((key, asyncio.ensure_future(factory())))
        except Exception as exc:  # one broken job must not strand the others
            logger.warning("recall fetch job %s could not start: %s", key, exc)
    remaining = _remaining_s()
    deadline_hit = False
    if job_tasks:
        _done, pending = await asyncio.wait(
            [t for _k, t in job_tasks],
            timeout=(max(0.0, remaining) if remaining is not None else None),
        )
        if pending:
            deadline_hit = True
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            logger.warning(
                "recall_deadline_hit corr_id=%s budget_ms=%s pending_jobs=%s completed_units=%s",
                corr_id,
                deadline_budget_ms,
                len(pending),
                len(sink),
            )

    for key, task in job_tasks:
        result = None
        if task.done() and not task.cancelled() and task.exception() is None:
            result = task.result()
        elif task.done() and not task.cancelled() and task.exception() is not None:
            logger.debug("recall fetch job %s failed: %s", key, task.exception())
        if key == "anchor":
            if result is None:
                partial = sorted((e for e in sink if e.get("key") == "anchor"), key=lambda e: e["idx"])
                result = _merge_unit_results(
                    [(e["candidates"], e["counts"]) for e in partial], include_retrievers=False, profile=profile
                )
            anchor_candidates, anchor_counts = result
            candidates.extend(anchor_candidates)
            for ck, cv in anchor_counts.items():
                backend_counts_total[ck] = cv
            continue
        if result is None:
            # Cancelled at the deadline (or crashed): keep the units that
            # completed before that, in their normal order.
            partial = sorted((e for e in sink if e.get("key") == key), key=lambda e: e["idx"])
            result = _merge_unit_results(
                [(e["candidates"], e["counts"]) for e in partial], include_retrievers=False, profile=profile
            )
        cand, counts = result
        candidates.extend(cand)
        for ck, cv in counts.items():
            backend_counts_total[ck] = backend_counts_total.get(ck, 0) + cv

    fetch_ms = int((time.perf_counter() - fetch_started) * 1000)
    feed_elapsed = [e["elapsed_ms"] for e in sink if e.get("kind") == _UNIT_FEED]
    retriever_elapsed = [
        e["elapsed_ms"] for e in sink if e.get("kind") == _UNIT_RETRIEVER and e.get("key") != "anchor"
    ] + anchor_elapsed_ms
    # Units overlap, so these are each group's critical path (slowest unit),
    # not sums; "fetch" is the wall time of the whole concurrent stage.
    timings_ms["feeds"] = max(feed_elapsed) if feed_elapsed else 0
    timings_ms["retrievers"] = max(retriever_elapsed) if retriever_elapsed else 0
    timings_ms["fetch"] = fetch_ms
    timing_breakdown_ms["anchor_fetch"] = max(anchor_elapsed_ms) if anchor_elapsed_ms else 0
    timing_breakdown_ms["backend_fetch"] = fetch_ms
    candidates_fetched = len(candidates)

    # memory_graph_sparql augment removed 2026-07-22: RECALL_MEMORY_GRAPH_SPARQL_ENABLED
    # was already false live (dead in production), and the Fuseki content it
    # would have read (orionmem AffectiveDisposition records) turned out to be
    # test-fixture pollution, not real approved memory -- see
    # orion/memory_graph/approve.py's docstring for the full trace. Purged
    # from Fuseki 2026-07-22.

    windowing_started = time.perf_counter()
    if settings.RECALL_RDF_CHAT_WINDOW_ENABLED:
        rdf_chat_window_min = int(
            profile.get("rdf_chat_since_minutes")
            or profile.get("sql_since_minutes")
            or settings.RECALL_SQL_SINCE_MINUTES
        )
        candidates, rdf_chat_dropped = await _window_rdf_chatturn_candidates(
            candidates, since_minutes=rdf_chat_window_min
        )
        if rdf_chat_dropped:
            backend_counts_total["rdf_chat_out_of_window_dropped"] = rdf_chat_dropped
            logger.info(
                "rdf chat-turn windowing profile=%s window_min=%s dropped=%s",
                profile.get("profile"),
                rdf_chat_window_min,
                rdf_chat_dropped,
            )

    candidates, sql_chat_dropped = await _window_sql_chat_candidates(
        candidates, since_minutes=sql_chat_window_min
    )
    if sql_chat_dropped:
        backend_counts_total["sql_chat_out_of_window_dropped"] = sql_chat_dropped
        logger.info(
            "sql chat windowing profile=%s window_min=%s dropped=%s",
            profile.get("profile"),
            sql_chat_window_min,
            sql_chat_dropped,
        )

    timings_ms["windowing"] = int((time.perf_counter() - windowing_started) * 1000)

    suppression_start = time.time()
    candidates, suppressed = _suppress_self_hits(
        candidates,
        active_turn_text=str(exclusion.get("active_turn_text") or ""),
        active_turn_ids=list(exclusion.get("active_turn_ids") or []),
        active_turn_ts=exclusion.get("active_turn_ts"),
    )
    timing_breakdown_ms["self_hit_suppression"] = int((time.time() - suppression_start) * 1000)
    timings_ms["suppression"] = timing_breakdown_ms["self_hit_suppression"]
    candidates_kept = len(candidates)
    if suppressed:
        logger.info(
            "recall self-hit suppression active_turn_ids=%s suppressed=%s",
            exclusion.get("active_turn_ids"),
            suppressed,
        )

    pcr_started = time.perf_counter()
    if settings.RECALL_PCR_ENABLED and recall_phase == "purposeful":
        pcr_cands, pcr_counts, pcr_elapsed, pcr_deadline_hit = await _run_pcr_collectors(
            q,
            pcr_backend_plan=pcr_backend_plan,
            remaining_s=_remaining_s(),
            corr_id=corr_id,
        )
        candidates.extend(pcr_cands)
        backend_counts_total.update(pcr_counts)
        for name, ms in pcr_elapsed.items():
            timings_ms[f"pcr_{name}"] = ms
        if pcr_deadline_hit:
            deadline_hit = True
    timings_ms["pcr_collectors"] = int((time.perf_counter() - pcr_started) * 1000)

    # Provisional: fusion stamps this into bundle.stats; both are overwritten
    # with the true end-to-end figure once everything below has run.
    latency_ms = int((time.time() - t0) * 1000)
    fuse_started = time.time()
    timings_ms["boost"] = 0
    if settings.RECALL_PCR_ENABLED and recall_phase == "continuity":
        profile["sql_since_minutes"] = settings.RECALL_CONTINUITY_SQL_MINUTES
        profile["render_budget_tokens"] = settings.RECALL_CONTINUITY_RENDER_BUDGET
        bundle, ranking_debug = render_continuity_bundle(
            candidates=candidates,
            profile=profile,
            query_text=query_fragment,
            latency_ms=latency_ms,
            session_id=effective_session_id,
        )
    elif settings.RECALL_PCR_ENABLED and recall_phase == "purposeful":
        belief_budget = q.belief_digest_max_tokens or settings.RECALL_BELIEF_RENDER_BUDGET
        profile["render_budget_tokens"] = belief_budget
        bundle, ranking_debug = pcr_fuse_belief_candidates(
            candidates=candidates,
            profile=profile,
            retrieval_intent=str(retrieval_intent),
            query_text=query_fragment,
            latency_ms=latency_ms,
        )
    else:
        entity_boost_started = time.time()
        entity_boost_map: Dict[str, float] = {}
        entity_injected_candidates: List[Dict[str, Any]] = []
        boost_remaining = _remaining_s()
        if context_only:
            pass  # the boost injects entity-matched turns: retrieval, not a feed
        elif boost_remaining is not None and boost_remaining <= 0:
            deadline_hit = True
        else:
            try:
                entity_boost_map, entity_injected_candidates = await asyncio.wait_for(
                    _compute_entity_relatedness_boost_map(
                        query_text=query_fragment,
                        candidates=candidates,
                        query_entities=_boost_query_entities(query_fragment),
                    ),
                    timeout=boost_remaining,
                )
            except asyncio.TimeoutError:
                deadline_hit = True
                logger.warning("recall_deadline_hit corr_id=%s stage=entity_boost", corr_id)
        if entity_injected_candidates:
            # Live evidence (6 real queries, 3 profiles) showed the boost
            # alone never fires: falkor_chat's own fetch is recency-windowed
            # with no query filter, so an entity from an older turn never
            # enters the pool for the boost to act on. These are turns
            # fetched specifically because they mention the query's own
            # entities (or Jaccard-related ones) -- added to the same pool
            # fuse_candidates already dedupes/ranks, not a separate path.
            candidates = candidates + entity_injected_candidates
            backend_counts_total["falkor_chat_entity_injected"] = len(entity_injected_candidates)
        # Kept separate from timing_breakdown_ms["fusion"] below: this is
        # Falkor I/O (up to 4 round trips when the flag is on), not
        # fuse_candidates' own in-process ranking work -- folding it into
        # "fusion" would mislabel new network latency as ranking-logic cost,
        # making a future latency regression here invisible in the existing
        # telemetry surface (found in code review).
        timing_breakdown_ms["entity_relatedness_boost"] = int((time.time() - entity_boost_started) * 1000)
        timings_ms["boost"] = timing_breakdown_ms["entity_relatedness_boost"]
        fuse_started = time.time()
        bundle, ranking_debug = fuse_candidates(
            candidates=candidates,
            profile=profile,
            latency_ms=latency_ms,
            query_text=query_fragment,
            session_id=effective_session_id,
            diagnostic=diagnostic,
            substantive_query=str(query_targeting.get("turn_type")) == "substantive",
            entity_boost_map=entity_boost_map,
        )
    timing_breakdown_ms["fusion"] = int((time.time() - fuse_started) * 1000)
    timings_ms["fusion"] = timing_breakdown_ms["fusion"]
    timing_breakdown_ms["total"] = latency_ms
    eligible_belief_count = 0
    eligible_started = time.perf_counter()
    if settings.RECALL_PCR_ENABLED and recall_phase in {"continuity", "purposeful"} and _recall_pg_pool is not None:
        # Debug-only count: bounded by what is left of the deadline and
        # skipped once it has passed, like the shadow compare below. Skipping
        # it does not cut any results, so it does not set deadline_hit.
        eligible_remaining = _remaining_s()
        if eligible_remaining is not None and eligible_remaining <= 0:
            logger.debug("eligible_belief_count skipped: deadline passed")
        else:
            try:
                eligible_belief_count = await asyncio.wait_for(
                    count_eligible_active(_recall_pg_pool), timeout=eligible_remaining
                )
            except asyncio.TimeoutError:
                logger.debug("eligible_belief_count skipped: deadline passed")
            except Exception as exc:
                logger.debug("eligible_belief_count skipped: %s", exc)
    timings_ms["eligible_count"] = int((time.perf_counter() - eligible_started) * 1000)
    pcr_debug: Dict[str, Any] | None = None
    if settings.RECALL_PCR_ENABLED and recall_phase in {"continuity", "purposeful"}:
        continuity_count = sum(1 for i in bundle.items if recall_phase == "continuity")
        belief_count = sum(1 for i in bundle.items if recall_phase == "purposeful")
        active_refs = [i.id for i in bundle.items if str(i.source) == "active_packet"]
        pcr_debug = {
            "enabled": settings.RECALL_PCR_ENABLED,
            "phase": recall_phase,
            "retrieval_intent": retrieval_intent,
            "intent_rule_id": (q.task_hints or {}).get("rule_id") if isinstance(q.task_hints, dict) else None,
            "skip_reasons": list((q.task_hints or {}).get("skip_reasons") or []) if isinstance(q.task_hints, dict) else [],
            "backend_plan": list(pcr_backend_plan.keys()) if pcr_backend_plan else [],
            "continuity_item_count": continuity_count if recall_phase == "continuity" else 0,
            "belief_item_count": belief_count if recall_phase == "purposeful" else len(bundle.items),
            "active_packet_refs": active_refs,
            "render_budget": {
                "continuity": settings.RECALL_CONTINUITY_RENDER_BUDGET if recall_phase == "continuity" else 0,
                "belief": (
                    q.belief_digest_max_tokens
                    or settings.RECALL_BELIEF_RENDER_BUDGET
                    if recall_phase == "purposeful"
                    else 0
                ),
            },
        }
    decision = RecallDecisionV1(
        corr_id=corr_id or str(uuid4()),
        session_id=ignored_session_id,
        node_id=q.node_id,
        verb=q.verb,
        profile=str(profile.get("profile") or q.profile),
        query=query_fragment,
        selected_ids=[i.id for i in bundle.items],
        backend_counts=backend_counts_total or bundle.stats.backend_counts,
        latency_ms=latency_ms,
        query_chars=len(query_fragment),
        retrieval_query_source=retrieval_query_source,
        sub_query_count=len(signals),
        candidates_fetched=candidates_fetched,
        candidates_kept=candidates_kept,
        deadline_hit=deadline_hit,
        timings_ms=dict(timings_ms),
        dropped=dict((bundle.stats.diagnostic or {}).get("drop_counts") or {}),
        ranking_debug=ranking_debug if diagnostic else [],
        recall_debug=(
            {
                "latency_breakdown_ms": timing_breakdown_ms,
                "profile_selected": str(profile.get("profile") or q.profile),
                "profile_requested": q.profile,
                **({"recall_intent": intent_payload} if intent_payload else {}),
                "query_expansion_enabled": enable_qe,
                "query_targeting": {
                    **query_targeting,
                    "raw_fragment": q.fragment,
                },
                "source_gating": source_gating,
                "vector_policy": vector_policy,
                "active_turn": {
                    "ids_count": len(list(exclusion.get("active_turn_ids") or [])),
                    "text_present": bool(str(exclusion.get("active_turn_text") or "").strip()),
                    "ts_present": exclusion.get("active_turn_ts") is not None,
                    "self_hit_suppressed": suppressed,
                },
                "fusion": bundle.stats.diagnostic or {},
                "selected_summary": _bounded_selected_summary(list(bundle.items)),
                **({"eligible_belief_count": eligible_belief_count} if settings.RECALL_PCR_ENABLED and recall_phase in {"continuity", "purposeful"} else {}),
                **({"pcr": pcr_debug} if pcr_debug else {}),
            }
            if diagnostic
            else (
                {
                    "latency_breakdown_ms": timing_breakdown_ms,
                    **({"eligible_belief_count": eligible_belief_count} if settings.RECALL_PCR_ENABLED and recall_phase in {"continuity", "purposeful"} else {}),
                    **({"pcr": pcr_debug} if pcr_debug else {}),
                }
            )
        ),
    )
    compare_summary: Dict[str, Any] = {}
    anchor_plan_summary: Dict[str, Any] = {}
    selected_cards: list[Dict[str, Any]] = []
    # Main's effective triggers only: empty bundle or vector-topped. Main
    # also listed "query has anchor tokens", but _anchor_tokens was dead
    # there, so that branch never fired; fixing the regex must not start
    # running this inline diagnostic (extra Postgres + RDF round trips before
    # the reply) on every anchor-bearing query (code review, PR #2416).
    should_shadow_compare = bool(
        not context_only
        and (
            not bundle.items
            or any(str(item.source or "") == "vector" for item in bundle.items[:2])
        )
    )
    shadow_started = time.perf_counter()
    shadow_remaining = _remaining_s()
    if should_shadow_compare and shadow_remaining is not None and shadow_remaining <= 0:
        # Diagnostic only: never spend time past the deadline on it.
        should_shadow_compare = False
    if should_shadow_compare:
        try:
            from .recall_v2 import run_recall_v2_shadow

            shadow_q = q if query_fragment == q.fragment else q.model_copy(update={"fragment": query_fragment})
            shadow_bundle, shadow_debug = await asyncio.wait_for(
                run_recall_v2_shadow(shadow_q, profile=profile), timeout=shadow_remaining
            )
            compare_summary = {
                "v1_latency_ms": decision.latency_ms,
                "v2_latency_ms": int(shadow_debug.get("latency_ms") or 0),
                "selected_count_delta": len(shadow_bundle.items) - len(bundle.items),
                "v1_selected_count": len(bundle.items),
                "v2_selected_count": len(shadow_bundle.items),
            }
            anchor_plan_summary = dict(shadow_debug.get("plan") or {})
            selected_cards = list(shadow_debug.get("ranked_cards") or [])[:6]
        except Exception as exc:
            logger.debug(f"recall shadow compare skipped: {exc}")
    timings_ms["shadow_compare"] = int((time.perf_counter() - shadow_started) * 1000)

    pressure_events = _build_recall_pressure_events(
        q=q,
        decision=decision,
        bundle=bundle,
        compare_summary=compare_summary,
        anchor_plan=anchor_plan_summary,
        selected_evidence_cards=selected_cards,
    )
    if pressure_events:
        merged_debug = dict(decision.recall_debug or {})
        merged_debug["pressure_events"] = pressure_events
        if compare_summary:
            merged_debug["compare_summary"] = compare_summary
        if anchor_plan_summary:
            merged_debug["anchor_plan_summary"] = anchor_plan_summary
        if selected_cards:
            merged_debug["selected_evidence_cards"] = selected_cards
        decision = decision.model_copy(update={"recall_debug": merged_debug})
    if diagnostic:
        logger.info(
            "recall_diagnostic_summary corr_id=%s profile=%s requested_profile=%s gating=%s drop_counts=%s selected_counts=%s suppressed=%s latency_breakdown_ms=%s selected=%s",
            decision.corr_id,
            profile.get("profile"),
            q.profile,
            source_gating,
            decision.dropped,
            (bundle.stats.diagnostic or {}).get("source_selected_counts", {}),
            suppressed,
            timing_breakdown_ms,
            decision.selected_ids[:8],
        )
    # True end-to-end latency, after boost, fusion and the shadow compare.
    latency_ms = int((time.time() - t0) * 1000)
    timings_ms["total"] = latency_ms
    timing_breakdown_ms["total"] = latency_ms
    bundle.stats.latency_ms = latency_ms
    decision = decision.model_copy(
        update={"latency_ms": latency_ms, "timings_ms": dict(timings_ms), "deadline_hit": deadline_hit}
    )
    _log_debug_dump(
        corr_id=decision.corr_id,
        profile=profile,
        backend_counts=decision.backend_counts or {},
        items=list(bundle.items),
    )
    return bundle, decision


def build_reply_envelope(bundle: MemoryBundleV1, env: BaseEnvelope, *, debug: Dict[str, Any] | None = None) -> BaseEnvelope:
    payload: Dict[str, Any] = {"bundle": bundle.model_dump(mode="json")}
    if debug:
        payload["debug"] = debug
    return BaseEnvelope(
        kind=RECALL_REPLY_KIND,
        source=_source(),
        correlation_id=env.correlation_id,
        causality_chain=env.causality_chain,
        payload=payload,
        reply_to=None,
    )


def telemetry_envelope(decision: RecallDecisionV1, env: BaseEnvelope) -> BaseEnvelope:
    return BaseEnvelope(
        kind=RECALL_TELEMETRY_KIND,
        source=_source(),
        correlation_id=env.correlation_id,
        causality_chain=env.causality_chain,
        payload=decision.model_dump(mode="json"),
    )


async def handle_recall(env: BaseEnvelope, *, bus) -> BaseEnvelope:
    if env.kind not in {RECALL_REQUEST_KIND, "recall.query.request"}:
        return BaseEnvelope(
            kind=RECALL_REPLY_KIND,
            source=_source(),
            correlation_id=env.correlation_id,
            payload={"error": f"unsupported_kind:{env.kind}"},
        )

    raw_payload: Dict[str, Any] = env.payload if isinstance(env.payload, dict) else {}
    diagnostic = bool((raw_payload.get("options") or {}).get("diagnostic"))
    payload_obj = dict(raw_payload)
    payload_obj.pop("options", None)
    try:
        q = RecallQueryV1.model_validate(payload_obj)
    except ValidationError as ve:
        return BaseEnvelope(
            kind=RECALL_REPLY_KIND,
            source=_source(),
            correlation_id=env.correlation_id,
            payload={"error": "validation_failed", "details": ve.errors()},
        )

    corr = str(env.correlation_id)
    logger.info(
        "recall_bus_request_begin corr_id=%s verb=%s profile=%s session_id=%s node_id=%s",
        corr,
        q.verb,
        q.profile,
        q.session_id,
        q.node_id,
    )
    wall_t0 = time.perf_counter()
    bundle, decision = await process_recall(q, corr_id=corr, diagnostic=diagnostic)
    wall_ms = int((time.perf_counter() - wall_t0) * 1000)
    logger.info(
        "recall_bus_request_complete corr_id=%s wall_ms=%s process_latency_ms=%s profile=%s verb=%s backend_counts=%s",
        corr,
        wall_ms,
        decision.latency_ms,
        decision.profile,
        decision.verb,
        decision.backend_counts,
    )
    if wall_ms >= 30000:
        logger.warning(
            "recall_bus_request_slow corr_id=%s wall_ms=%s latency_breakdown_ms=%s",
            corr,
            wall_ms,
            (decision.recall_debug or {}).get("latency_breakdown_ms"),
        )

    # emit telemetry (fire and forget)
    try:
        await bus.publish(settings.RECALL_BUS_TELEMETRY, telemetry_envelope(decision, env))
    except Exception as exc:
        logger.debug(f"telemetry publish failed: {exc}")

    await persist_decision_async(decision)

    debug_payload: Dict[str, Any] | None = None
    if diagnostic:
        debug_payload = {
            "decision": {
                "corr_id": decision.corr_id,
                "profile": decision.profile,
                "selected_ids": decision.selected_ids[:8],
                "backend_counts": decision.backend_counts,
                "dropped": decision.dropped,
                "recall_debug": decision.recall_debug,
            }
        }
    return build_reply_envelope(bundle, env, debug=debug_payload)
