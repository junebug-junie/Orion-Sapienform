from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class EdgeDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    s: str
    p: str
    o: str
    confidence: Optional[float] = None


class ParticipantDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    entity_id: str
    role: str


class SituationDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    utterance_ids: List[str]
    label: Optional[str] = None
    stimulus_entity_id: Optional[str] = None
    about_entity_ids: List[str] = Field(default_factory=list)
    target_of_negative_affect_ids: List[str] = Field(default_factory=list)
    affectLabel: Optional[str] = Field(None, alias="affectLabel")
    timeQualitative: Optional[str] = Field(None, alias="timeQualitative")
    occurredAt: Optional[str] = None
    participants: List[ParticipantDraft] = Field(default_factory=list)


class EntityDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    label: str
    entityKind: str = Field(alias="entityKind")
    generalizes_to: Optional[str] = Field(None, alias="generalizes_to")


class DispositionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    holder_id: str = Field(alias="holder_id")
    target_id: str = Field(alias="target_id")
    trustPolarity: str = Field(alias="trustPolarity")
    description: Optional[str] = None


class SuggestDraftV1(BaseModel):
    """Appendix C draft JSON (+ optional fields used by the Joey/cats exemplar)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    ontology_version: str
    utterance_ids: List[str]
    entities: List[EntityDraft] = Field(default_factory=list)
    situations: List[SituationDraft] = Field(default_factory=list)
    edges: List[EdgeDraft] = Field(default_factory=list)
    dispositions: List[DispositionDraft] = Field(default_factory=list)
    utterance_text_by_id: Dict[str, str] = Field(default_factory=dict)


class MemoryGraphSubschemaV1(BaseModel):
    """Appendix D projection embedded under card.subschema.memory_graph."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    ontology_version: str
    named_graphs: List[str]
    situation_id: Optional[str] = None
    utterance_ids: List[str] = Field(default_factory=list)
    facts: List[Dict[str, Any]] = Field(default_factory=list)
    entity_refs: Dict[str, str] = Field(default_factory=dict)
