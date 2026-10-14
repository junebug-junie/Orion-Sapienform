"""Person model contracts (P1/P2).

Spec: docs/superpowers/specs/2026-10-14-person-model-and-identity-card-design.md
(rev 3). Only the contracts with a producer in this patch are defined here:

- ``PersonRefV1``: who a person is to the module (node id, name, stated pronouns).
  Producer: ``orion.person_model.person.load_people`` (config/person_model/people.yaml).
- ``PersonMentionV1``: one extractor span. Producer: ``orion.person_model.extract``.
- ``PersonLinkageDecisionV1``: one scored pair and its outcome. Producer:
  ``orion.person_model.linkage.decide``.
- ``EntityMergeLogEntryV1``: one merge-log row with what applying it would change,
  so it can be undone. Producer: ``orion.person_model.merge_log``.

The attribute, study-run, situation-person and identity-card contracts land with
their producers (P3-P6), not before. None of these is published on the bus yet:
P2 is a dry run that writes files only.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

PersonRole = Literal["interlocutor", "self", "other"]
LinkageRelation = Literal["same_as", "facet_of", "junk", "not_home", "distinct", "no_match"]
LinkageGroup = Literal["auto_merge", "needs_your_call", "auto_reject"]


class PronounEvidenceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_kind: str
    source_ref: str
    quote: str = Field(min_length=1)


class PersonRefV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["person.ref.v1"] = "person.ref.v1"
    node_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    role: PersonRole = "other"
    # Stated pronouns only ("she/her"). None means: use the name, never guess.
    pronouns: Optional[str] = None
    pronouns_evidence: List[PronounEvidenceV1] = Field(default_factory=list)


class PersonMentionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["person.mention.v1"] = "person.mention.v1"
    mention_id: str  # uuid5(source_kind|source_id|start|end|extractor_version)
    source_kind: str
    source_id: str
    span_start: int = Field(ge=0)
    span_end: int = Field(ge=0)
    text: str
    label: str
    score: float
    extractor_version: str


class LinkageFeatureV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    level: str  # e.g. "agree", "weak", "disagree", "missing"
    weight: float  # log2(m/u) contribution to the score
    detail: Optional[str] = None


class PersonLinkageDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["person.linkage.decision.v1"] = "person.linkage.decision.v1"
    decision_id: str  # uuid5(anchor|candidate|linkage_version): a rerun is identical
    linkage_version: str
    anchor_id: str
    anchor_label: str
    candidate_id: str
    candidate_label: str
    candidate_kind: str  # substrate_node | referent_alias | mention_cluster
    relation: LinkageRelation
    group: LinkageGroup
    score: Optional[float] = None  # None for rule outcomes (facet/junk/not_home)
    upper_threshold: float
    lower_threshold: float
    rule: Optional[str] = None  # which pattern rule or guard decided it, if any
    features: List[LinkageFeatureV1] = Field(default_factory=list)
    evidence: Dict[str, object] = Field(default_factory=dict)


class EntityMergeLogEntryV1(BaseModel):
    """One merge-log row. ``ops`` lists exactly what applying would write, so
    ``orion.person_model.merge_log.unmerge_ops`` can invert it without reading
    anything else."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["person.entity_merge_log.v1"] = "person.entity_merge_log.v1"
    merge_id: str  # uuid5(survivor|absorbed|linkage_version)
    survivor_node_id: str
    absorbed_node_id: str
    relation: Literal["same_as", "facet_of", "junk", "not_home"]
    score: Optional[float] = None
    features_json: List[LinkageFeatureV1] = Field(default_factory=list)
    linkage_version: str
    decision: Literal["auto", "asked", "confirmed", "rejected"]
    decided_by: str
    ops: List[Dict[str, object]] = Field(default_factory=list)
    applied_at: Optional[str] = None
    reverted_at: Optional[str] = None
