from __future__ import annotations

import logging
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef
from orion.core.llm_json import parse_json_object
from orion.memory.crystallization.bus_emit import emit_crystallization_lifecycle
from orion.memory.crystallization.candidate_retrieval import fetch_similar_candidates
from orion.memory.crystallization.dynamics import reinforce
from orion.memory.crystallization.repository import insert_concept_relation_decision, update_crystallization
from orion.memory.crystallization.schemas import (
    CrystallizationLinkV1,
    MemoryCrystallizationV1,
    _utc_now,
)

logger = logging.getLogger(__name__)

ConceptRelation = Literal["same", "refines", "contradicts", "unrelated"]

_SUMMARY_TRUNC_CHARS = 300
_SUBJECT_TRUNC_CHARS = 200


class ConceptRelationDecision(BaseModel):
    """Local to this seam -- not a bus-published event, no registry entry needed
    (same precedent as ConsolidationGateResult / ValidationResult elsewhere in this codebase)."""

    # extra="ignore", not "forbid": small instruct models frequently add a stray field
    # (e.g. "reasoning") to structured output despite prompt instructions to emit only
    # the bare object. Rejecting the whole decision over an extra key would make this
    # already-conservative feature degrade to "unrelated" far more often than the
    # relation/confidence/target fields themselves warrant.
    model_config = ConfigDict(extra="ignore")

    relation: ConceptRelation
    target_crystallization_id: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _target_required_unless_unrelated(self) -> "ConceptRelationDecision":
        if self.relation != "unrelated" and not self.target_crystallization_id:
            raise ValueError("target_crystallization_id required unless relation is unrelated")
        return self


def merge_new_evidence(
    target: MemoryCrystallizationV1,
    candidate: MemoryCrystallizationV1,
) -> MemoryCrystallizationV1:
    """Moved here from intake_pipeline.py so both the same-window Jaccard reinforce
    path and this module's cross-window 'same' path share one implementation
    (concept_relation.py is imported BY intake_pipeline.py, so the helper had to
    move to break the circular import rather than staying private in the caller)."""
    updated = target.model_copy(deep=True)
    seen = {(ev.source_kind, ev.source_id) for ev in updated.evidence}
    for ev in candidate.evidence:
        key = (ev.source_kind, ev.source_id)
        if key in seen:
            continue
        updated.evidence.append(ev)
        seen.add(key)
    return updated


def _truncate(text: str, *, limit: int) -> str:
    s = (text or "").strip()
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def _build_relation_prompt(candidate: MemoryCrystallizationV1, similar_existing: list[MemoryCrystallizationV1]) -> str:
    candidate_block = (
        "NEW CANDIDATE:\n"
        f"subject: {_truncate(candidate.subject, limit=_SUBJECT_TRUNC_CHARS)}\n"
        f"summary: {_truncate(candidate.summary, limit=_SUMMARY_TRUNC_CHARS)}\n"
    )

    existing_lines = []
    for item in similar_existing:
        existing_lines.append(
            f"- id: {item.crystallization_id}\n"
            f"  status: {item.status}\n"
            f"  subject: {_truncate(item.subject, limit=_SUBJECT_TRUNC_CHARS)}\n"
            f"  summary: {_truncate(item.summary, limit=_SUMMARY_TRUNC_CHARS)}\n"
        )
    existing_block = "EXISTING MEMORY ITEMS:\n" + "\n".join(existing_lines)

    instructions = (
        "Judge the relation between the NEW CANDIDATE above and the EXISTING MEMORY ITEMS.\n"
        "Choose exactly one relation:\n"
        '  "same"        = restates the same belief/fact as one existing item, just worded differently\n'
        '  "refines"     = a more accurate/updated/narrower version of one existing item (the old one\n'
        "                  should eventually be considered superseded by a human reviewer)\n"
        '  "contradicts" = conflicts with one existing item (both may still be worth keeping on record)\n'
        '  "unrelated"   = none of the above; default when uncertain\n'
        "\n"
        "Respond with ONLY a single JSON object, no prose, no markdown fences, shaped exactly like:\n"
        '{"relation": "same"|"refines"|"contradicts"|"unrelated", '
        '"target_crystallization_id": "<id of the existing item this relates to, or null if unrelated>", '
        '"confidence": <float between 0.0 and 1.0>}\n'
    )

    return f"{candidate_block}\n{existing_block}\n\n{instructions}"


async def resolve_concept_relation(
    bus: OrionBusAsync,
    *,
    candidate: MemoryCrystallizationV1,
    similar_existing: list[MemoryCrystallizationV1],
    settings: Any,
) -> ConceptRelationDecision:
    """Bounded structured-output LLM call. NEVER raises -- degrades to
    ConceptRelationDecision(relation="unrelated", confidence=0.0) on any failure
    (timeout, malformed bus reply, invalid JSON, schema validation failure)."""
    if not similar_existing:
        return ConceptRelationDecision(relation="unrelated", confidence=0.0)
    try:
        prompt = _build_relation_prompt(candidate, similar_existing)
        rpc_corr = str(uuid4())
        reply_channel = f"orion:exec:result:LLMGatewayService:{rpc_corr}"
        route = str(getattr(settings, "TURN_CHANGE_CLASSIFY_ROUTE", "metacog") or "metacog")
        payload = ChatRequestPayload(
            messages=[LLMMessage(role="user", content=prompt)],
            route=route,
            options={
                "return_logprobs": False,
                "max_tokens": 200,
                "llm_route": route,
                "purpose": "classify",
                "skip_spark_candidate_publish": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        env = BaseEnvelope(
            kind="llm.chat.request",
            source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION, node=settings.NODE_NAME),
            correlation_id=rpc_corr,
            reply_to=reply_channel,
            payload=payload.model_dump(mode="json"),
        )
        # Plain getattr default (no `or` fallback): these are numeric fields where an
        # explicit 0 is a meaningful operator choice, not "unset" -- `0 or default`
        # would silently clobber it back to the default.
        timeout_sec = float(getattr(settings, "CONCEPT_RELATION_TIMEOUT_SEC", 8.0))
        msg = await bus.rpc_request(
            settings.CHANNEL_LLM_INTAKE, env, reply_channel=reply_channel, timeout_sec=timeout_sec,
        )
        decoded = bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RuntimeError(decoded.error)
        content = str(decoded.envelope.payload.get("content") or decoded.envelope.payload.get("text") or "")
        # Models sometimes wrap the JSON object in prose or markdown fences despite
        # instructions -- reuse the shared, already-battle-tested LLM JSON extraction
        # (handles ```json fences, trailing commas, Python True/False/None literals)
        # instead of a narrower hand-rolled parse. Any failure falls through to the
        # except block below.
        obj = parse_json_object(content)
        return ConceptRelationDecision.model_validate(obj)
    except Exception as exc:
        logger.warning("concept_relation_resolve_failed error=%s", exc)
        return ConceptRelationDecision(relation="unrelated", confidence=0.0)


async def maybe_resolve_concept_relation(
    pool,
    bus: OrionBusAsync | None,
    *,
    candidate: MemoryCrystallizationV1,
    settings: Any,
    emit_kw: dict[str, str],
) -> tuple[str, MemoryCrystallizationV1, str] | None:
    """Returns a (crystallization_id, row, outcome) tuple ONLY when it took a decisive
    action (relation == "same"). Returns None in every other case (no candidates, LLM
    degraded/unrelated, confidence below floor, or relation in {"refines","contradicts"})
    so the caller falls through to its existing, unchanged insert/formation_policy logic.

    Deliberately conservative scope: "refines" and "contradicts" only ATTACH a typed link
    to `candidate.links` (which the existing insert_crystallization() already persists to
    memory_crystallization_links -- first-ever rows in that table) and otherwise change
    NOTHING about how the candidate is formed. They do NOT call governor.supersede() on the
    existing target. Auto-superseding an already-active belief on the strength of an LLM
    judgment about a brand-new, not-yet-approved proposal would be applying that proposal's
    consequence onto canonical state automatically -- out of scope for this patch under the
    "no auto-apply of proposals" constraint. A human reviewing the new link via the existing
    /api/memory/crystallizations/{id}/links endpoint can supersede manually. This is a
    deliberate, documented scope reduction from the original design doc, not an oversight.
    """
    # embed_host_url/chroma_host emptiness is not re-checked here: fetch_similar_candidates
    # already degrades to [] on the same condition, and empty candidates is handled next.
    candidates = await fetch_similar_candidates(
        candidate,
        pool=pool,
        embed_host_url=getattr(settings, "CRYSTALLIZER_EMBED_HOST_URL", "") or "",
        chroma_host=getattr(settings, "CHROMA_HOST", "") or "",
        chroma_port=int(getattr(settings, "CHROMA_PORT", 8000)),
        chroma_collection=getattr(settings, "CRYSTALLIZER_VECTOR_COLLECTION", "orion_memory_crystallizations") or "orion_memory_crystallizations",
        # Plain getattr default (no `or` fallback) for the same reason as timeout_sec
        # above: CONCEPT_RELATION_CANDIDATE_LIMIT=0 is a legitimate "throttle to zero"
        # operator setting, not a signal to fall back to the default of 5.
        limit=int(getattr(settings, "CONCEPT_RELATION_CANDIDATE_LIMIT", 5)),
        embed_timeout_ms=int(getattr(settings, "CRYSTALLIZER_EMBED_TIMEOUT_MS", 8000)),
    )
    if not candidates:
        return None

    # Cross-window candidate retrieval is deliberately scope-free (that's the whole
    # point of this feature -- same-window Jaccard dedup can never match across
    # windows because scope is unique per window). But it must still respect the same
    # kind boundary detect_duplicates() enforces (detection.py: `candidate.kind ==
    # other.kind`) -- a "stance" should never be judged "same as" a "semantic" fact
    # just because the text embeds nearby. Filter before spending an LLM call on it.
    candidates = [c for c in candidates if c.kind == candidate.kind]
    if not candidates:
        return None

    decision = await resolve_concept_relation(bus, candidate=candidate, similar_existing=candidates, settings=settings)

    # Plain getattr default (no `or` fallback): CONCEPT_RELATION_CONFIDENCE_FLOOR=0.0
    # is a legitimate "accept every LLM decision" operator setting.
    confidence_floor = float(getattr(settings, "CONCEPT_RELATION_CONFIDENCE_FLOOR", 0.6))
    floor_cleared = decision.confidence >= confidence_floor

    # Record every real decision here, before the filter below discards anything --
    # previously only the decisive outcome reached a log line (see logger.info below),
    # so every "unrelated" decision and every sub-floor "contradicts"/"refines" decision
    # vanished silently. scripts/concept_relation_digest.py reads this table to surface
    # those near-misses (threshold-tuning report) and to produce belief-revision
    # "reflection" crystallizations for the ones that did clear the floor.
    #
    # This write is purely observational -- unlike resolve_concept_relation() above
    # (which is documented to NEVER raise), a DB error here must not propagate and take
    # down the whole formation flow. Before this patch nothing between here and the
    # caller's insert_crystallization() could fail this early; a swallowed exception on
    # a diagnostic log write is a strictly better failure mode than losing an entire
    # consolidation window over it (see services/orion-memory-consolidation's
    # mark_failed(window_id) on any uncaught exception).
    try:
        await insert_concept_relation_decision(
            pool,
            candidate_crystallization_id=candidate.crystallization_id,
            target_crystallization_id=decision.target_crystallization_id,
            relation=decision.relation,
            confidence=decision.confidence,
            floor_cleared=floor_cleared,
        )
    except Exception as exc:
        logger.warning("concept_relation_decision_log_failed error=%s", exc)

    if decision.relation == "unrelated" or not floor_cleared:
        return None

    target = next((c for c in candidates if c.crystallization_id == decision.target_crystallization_id), None)
    if target is None:
        # LLM referenced an id outside the candidate set it was given -- never trust an
        # unseen id, fall through to normal handling.
        return None

    # Audit trail for the LLM judgment itself, independent of which branch below acts on
    # it -- the refines/contradicts branch already records this on the link's note/confidence,
    # but the "same" branch's only other effect (reinforce()) is indistinguishable after the
    # fact from an ordinary same-window Jaccard reinforce without this.
    logger.info(
        "concept_relation_decided candidate_id=%s relation=%s target_id=%s confidence=%.2f",
        candidate.crystallization_id, decision.relation, target.crystallization_id, decision.confidence,
    )
    relation_provenance = {
        "concept_relation": {
            "relation": decision.relation,
            "target_crystallization_id": target.crystallization_id,
            "confidence": decision.confidence,
        }
    }

    if decision.relation == "same":
        merged = merge_new_evidence(target, candidate)
        updated = reinforce(merged, now=_utc_now())
        updated.provenance = {**(updated.provenance or {}), **relation_provenance}
        await update_crystallization(pool, updated)
        await emit_crystallization_lifecycle(bus, lifecycle="reinforced", crystallization=updated, **emit_kw)
        return target.crystallization_id, updated, "reinforced_by_relation"

    if decision.relation in ("refines", "contradicts"):
        candidate.links.append(
            CrystallizationLinkV1(
                target_crystallization_id=target.crystallization_id,
                relation="supersedes" if decision.relation == "refines" else "contradicts",
                confidence=decision.confidence,
                note="concept_relation_llm",
            )
        )
        candidate.provenance = {**(candidate.provenance or {}), **relation_provenance}
        return None

    return None
