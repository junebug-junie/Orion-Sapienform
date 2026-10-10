from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import logging

from pydantic import BaseModel, ConfigDict, Field, field_validator

from orion.schemas.reading import ReadingRequestedV1, SourceFetchEvidenceV1

_log = logging.getLogger(__name__)


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorldPulseReadSeedV1(_Base):
    seed_id: str = Field(min_length=1)
    kind: Literal["finding", "digest_item", "reading"]
    run_id: str = Field(min_length=1)
    url: str = Field(min_length=1)
    title: str = ""
    section: str = ""
    item_id: str | None = None  # digest_item only
    request: ReadingRequestedV1 | None = None


class WorldPulseReadConceptCandidateV1(_Base):
    label: str = Field(min_length=1)
    definition: str | None = None
    link_hints: list[str] = Field(default_factory=list)


class WorldPulseReadPriorCandidateV1(_Base):
    claim: str = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


def _coerce_prior_item(item: Any) -> Any:
    if isinstance(item, WorldPulseReadPriorCandidateV1):
        return item
    if isinstance(item, str):
        claim = item.strip()
        if not claim:
            return None
        return {"claim": claim, "confidence": 0.5}
    if isinstance(item, dict):
        claim = item.get("claim") or item.get("text") or item.get("prior") or item.get("statement")
        if claim is None and len(item) == 1:
            claim = next(iter(item.values()))
        if not isinstance(claim, str) or not claim.strip():
            return None
        conf = item.get("confidence", 0.5)
        try:
            conf_f = float(conf)
        except (TypeError, ValueError):
            conf_f = 0.5
        return {"claim": claim.strip(), "confidence": conf_f}
    return None


def _coerce_concept_item(item: Any) -> Any:
    if isinstance(item, WorldPulseReadConceptCandidateV1):
        return item
    if isinstance(item, str):
        label = item.strip()
        if not label:
            return None
        return {"label": label}
    if isinstance(item, dict):
        label = item.get("label") or item.get("name") or item.get("concept")
        if not isinstance(label, str) or not label.strip():
            return None
        out: dict[str, Any] = {"label": label.strip()}
        if item.get("definition") is not None:
            out["definition"] = item.get("definition")
        hints = item.get("link_hints")
        if isinstance(hints, list):
            out["link_hints"] = [str(h) for h in hints if str(h).strip()]
        return out
    return None


def _coerce_prior_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        item = _coerce_prior_item(value)
        return [item] if item is not None else []
    if not isinstance(value, list):
        return []
    out: list[Any] = []
    for item in value:
        coerced = _coerce_prior_item(item)
        if coerced is not None:
            out.append(coerced)
    return out


def _coerce_concept_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        item = _coerce_concept_item(value)
        return [item] if item is not None else []
    if not isinstance(value, list):
        return []
    out: list[Any] = []
    for item in value:
        coerced = _coerce_concept_item(item)
        if coerced is not None:
            out.append(coerced)
    return out


def _coerce_thread_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, dict):
            text = item.get("thread") or item.get("text") or item.get("question")
            if isinstance(text, str) and text.strip():
                out.append(text.strip())
    return out


PriorTestVerdict = Literal["supported", "revised", "refuted", "untested"]
_PRIOR_TEST_VERDICTS: tuple[str, ...] = ("supported", "revised", "refuted", "untested")


class WorldPulseReadPriorTestV1(_Base):
    """Stage 2's verdict on one Stage 1 prior. ``claim_ref`` is the prior's
    claim text (or a short handle for it); ``verdict`` is what the second
    pass concluded after testing it against the handoff and any hops."""

    claim_ref: str = Field(min_length=1)
    verdict: PriorTestVerdict = "untested"
    why: str = ""


def _coerce_prior_test_item(item: Any) -> Any:
    if isinstance(item, WorldPulseReadPriorTestV1):
        return item
    if isinstance(item, str):
        ref = item.strip()
        return {"claim_ref": ref} if ref else None
    if not isinstance(item, dict):
        return None
    ref = item.get("claim_ref") or item.get("claim") or item.get("prior") or item.get("ref")
    if not isinstance(ref, str) or not ref.strip():
        return None
    verdict_raw = item.get("verdict") or item.get("status") or item.get("result") or "untested"
    verdict = str(verdict_raw).strip().lower()
    if verdict not in _PRIOR_TEST_VERDICTS:
        verdict = "untested"
    why = item.get("why") or item.get("reason") or item.get("evidence") or item.get("note") or ""
    return {"claim_ref": ref.strip(), "verdict": verdict, "why": str(why).strip()}


def _coerce_prior_test_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, dict)):
        item = _coerce_prior_test_item(value)
        return [item] if item is not None else []
    if not isinstance(value, list):
        return []
    out: list[Any] = []
    for item in value:
        coerced = _coerce_prior_test_item(item)
        if coerced is not None:
            out.append(coerced)
    return out


class WorldPulseReadHandoffV1(_Base):
    """Stage 1 → Stage 2 (and Concept Atlas) artifact."""

    seed_ref: WorldPulseReadSeedV1
    what_i_learned: str = Field(min_length=1)
    candidate_priors: list[WorldPulseReadPriorCandidateV1] = Field(default_factory=list)
    concept_candidates: list[WorldPulseReadConceptCandidateV1] = Field(default_factory=list)
    open_threads: list[str] = Field(default_factory=list)
    trace_id: str = Field(min_length=1)
    created_at: datetime
    producer_hint: Literal["world_pulse_read_pipeline"] = "world_pulse_read_pipeline"
    # Tool-trace proof the Stage 1 turn fetched this seed's source
    # (orion/world_pulse_read/read_evidence.py). Set server-side, never by the
    # model. Empty on rows written before 2026-09-25; Stage 1 refuses to mark a
    # seed `done` without it and Stage 2 skips a handoff that lacks it.
    read_evidence: list[SourceFetchEvidenceV1] = Field(default_factory=list)

    @field_validator("what_i_learned")
    @classmethod
    def nonempty_learning(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("empty_learning")
        return value.strip()

    @field_validator("candidate_priors", mode="before")
    @classmethod
    def _priors_before(cls, value: Any) -> list[Any]:
        return _coerce_prior_list(value)

    @field_validator("concept_candidates", mode="before")
    @classmethod
    def _concepts_before(cls, value: Any) -> list[Any]:
        return _coerce_concept_list(value)

    @field_validator("open_threads", mode="before")
    @classmethod
    def _threads_before(cls, value: Any) -> list[str]:
        return _coerce_thread_list(value)


# #2497 reader allowlist: the existing semantic predicates a reading may claim.
# Operational predicates (activates, suppresses, seeks...) are never emitted by a reader.
READING_CLAIM_PREDICATES: tuple[str, ...] = (
    "subtype_of", "part_of", "refines", "associated_with", "causes", "co_occurs_with",
)
ReadingClaimOutcomeV1 = Literal["accepted_provisional", "proposed", "rejected"]


class WorldPulseReadClaimReceiptV1(_Base):
    """What deterministic code did with one relationship claim. Always server-set
    (orion/world_pulse_read/assertions.py); anything the model writes here is discarded."""

    outcome: ReadingClaimOutcomeV1
    # accepted_provisional: journalled proposal + accepting decision. proposed: journalled
    # proposal only, no link. rejected: refused by validation or not journalled at all.
    # reason, a short machine label: accepted | quote_not_found | quote_too_short |
    # quote_not_about_endpoints |
    # domain_rule:<predicate> | already_decided | decision_journal_failed |
    # predicate_not_allowed | unknown_subject | unknown_object | same_endpoint | duplicate |
    # journal_unavailable
    reason: str = Field(min_length=1)
    proposal_id: str | None = None
    assertion_id: str | None = None
    decision_id: str | None = None
    # The retained text the quote was found in, and where (half-open UTF-8 byte range).
    content_sha256: str | None = None
    representation: Literal["source_text", "tool_digest"] | None = None
    span_start: int | None = Field(default=None, ge=0)
    span_end: int | None = Field(default=None, ge=0)


class WorldPulseReadRelationshipClaimV1(_Base):
    """Stage 2's proposed relationship between a concept this read produced
    (``subject_id``) and an existing atlas concept (``object_id``), both chosen
    from id lists Hub put in the prompt. The model proposes; code decides."""

    subject_id: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object_id: str = Field(min_length=1)
    statement_text: str = Field(min_length=1)
    quote: str = ""
    receipt: WorldPulseReadClaimReceiptV1 | None = None


_CLAIM_FIELDS = frozenset(WorldPulseReadRelationshipClaimV1.model_fields)


def _coerce_claim_list(value: Any) -> list[Any]:
    """Claims are optional: one malformed claim (an empty statement, an invented key)
    must not throw away the whole Stage 2 read. Each claim is validated on its own,
    unknown keys dropped, invalid claims dropped with a warning. Receipts are kept here
    (a stored row carries real ones); the Stage 2 loop strips any from raw model output."""
    if not isinstance(value, list):
        return []
    out: list[Any] = []
    for item in value:
        if isinstance(item, WorldPulseReadRelationshipClaimV1):
            out.append(item)
            continue
        if not isinstance(item, dict):
            _log.warning("world_pulse_read_claim_dropped reason=not_object")
            continue
        unknown = sorted(k for k in item if k not in _CLAIM_FIELDS)
        try:
            out.append(WorldPulseReadRelationshipClaimV1.model_validate(
                {k: v for k, v in item.items() if k in _CLAIM_FIELDS}))
        except ValueError as exc:
            _log.warning("world_pulse_read_claim_dropped reason=invalid errors=%d", len(getattr(exc, "errors", lambda: [])()))
            continue
        if unknown:
            _log.warning("world_pulse_read_claim_keys_dropped keys=%s", ",".join(unknown))
    return out


def strip_model_claim_receipts(parsed: dict[str, Any]) -> dict[str, Any]:
    """Raw model output only: a model cannot author a receipt (it is code's verdict)."""
    claims = parsed.get("relationship_claims")
    if isinstance(claims, list):
        parsed["relationship_claims"] = [
            {k: v for k, v in item.items() if k != "receipt"} if isinstance(item, dict) else item
            for item in claims
        ]
    return parsed


class WorldPulseReadStage2ResultV1(_Base):
    """Stage 2 FCC result. ``need_stage1_urls`` may trigger Stage 1 re-entry.

    The prompt asks the second pass to form/test priors and note hops, so the
    schema carries exactly that work (every field defaulted -- an older
    ``stage2_result_json`` row with only ``summary`` still validates). Any
    other top-level key the model invents is dropped with a logged warning by
    the Stage 2 loop before validation; ``extra="forbid"`` stays on so an
    unlogged drift cannot slip through the model itself.
    """

    summary: str = Field(min_length=1)
    need_stage1_urls: list[str] = Field(default_factory=list)
    candidate_priors: list[WorldPulseReadPriorCandidateV1] = Field(default_factory=list)
    priors_tested: list[WorldPulseReadPriorTestV1] = Field(default_factory=list)
    concept_candidates: list[WorldPulseReadConceptCandidateV1] = Field(default_factory=list)
    open_threads: list[str] = Field(default_factory=list)
    hops: list[str] = Field(default_factory=list)
    # Model-proposed relationship claims; each gets a server-set receipt before the
    # result is stored. Older rows have none.
    relationship_claims: list[WorldPulseReadRelationshipClaimV1] = Field(default_factory=list)
    round_trips: int = Field(default=0, ge=0)
    trace_id: str = Field(min_length=1)
    created_at: datetime
    seed_id: str = ""
    request: ReadingRequestedV1 | None = None
    producer_hint: Literal["world_pulse_read_stage2"] = "world_pulse_read_stage2"

    @field_validator("summary")
    @classmethod
    def nonempty_summary(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("empty_summary")
        return value.strip()

    @field_validator("candidate_priors", mode="before")
    @classmethod
    def _priors_before(cls, value: Any) -> list[Any]:
        return _coerce_prior_list(value)

    @field_validator("priors_tested", mode="before")
    @classmethod
    def _priors_tested_before(cls, value: Any) -> list[Any]:
        return _coerce_prior_test_list(value)

    @field_validator("concept_candidates", mode="before")
    @classmethod
    def _concepts_before(cls, value: Any) -> list[Any]:
        return _coerce_concept_list(value)

    @field_validator("open_threads", "hops", mode="before")
    @classmethod
    def _string_lists_before(cls, value: Any) -> list[str]:
        return _coerce_thread_list(value)

    @field_validator("relationship_claims", mode="before")
    @classmethod
    def _claims_before(cls, value: Any) -> list[Any]:
        return _coerce_claim_list(value)
