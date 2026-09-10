from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from orion.schemas.reading import ReadingRequestedV1


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


class WorldPulseReadStage2ResultV1(_Base):
    """Stage 2 FCC result. ``need_stage1_urls`` may trigger Stage 1 re-entry."""

    summary: str = Field(min_length=1)
    need_stage1_urls: list[str] = Field(default_factory=list)
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
