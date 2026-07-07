from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from orion.schemas.recall_pcr import RecallPhaseV1, RetrievalIntentV1


class MemoryItemV1(BaseModel):
    """A single retrieved memory item."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(..., description="Stable identifier for the item")
    source: str = Field(..., description="Backend source (vector/rdf/sql/etc)")
    source_ref: Optional[str] = Field(None, description="Collection/index/table reference")
    uri: Optional[str] = Field(None, description="Canonical URI if available")
    score: float = Field(0.0, description="Normalized [0,1] relevance score")
    ts: Optional[float] = Field(None, description="Epoch seconds for recency ordering")
    title: Optional[str] = None
    snippet: str = Field("", description="Short text fragment")
    tags: List[str] = Field(default_factory=list)


class MemoryBundleStatsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend_counts: Dict[str, int] = Field(default_factory=dict)
    latency_ms: int = 0
    profile: Optional[str] = None
    diagnostic: Optional[Dict[str, Any]] = None


class MemoryBundleV1(BaseModel):
    """Prompt-ready bundle produced by recall."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    rendered: str = Field("", description="Concise prompt-ready text")
    items: List[MemoryItemV1] = Field(default_factory=list)
    stats: MemoryBundleStatsV1 = Field(default_factory=MemoryBundleStatsV1)


class RecallQueryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    fragment: str = Field(..., description="User fragment/query text")
    verb: Optional[str] = None
    intent: Optional[str] = None
    session_id: Optional[str] = None
    node_id: Optional[str] = None
    profile: str = Field("reflect.v1", description="Recall profile name")
    lane: Optional[str] = Field(
        default=None,
        description="Visibility lane (e.g. chat, social); set by orchestrator for memory cards scope.",
    )
    profile_explicit: bool = Field(
        default=False,
        description="When True, intent router must not override the requested recall profile.",
    )
    exclude: Optional[Dict[str, object]] = Field(
        default=None,
        description="Optional current-turn exclusion hints (ids/text/timestamps).",
    )
    reply_to: Optional[str] = None
    recall_phase: Optional[RecallPhaseV1] = Field(
        default=None,
        description="PCR phase: skip, continuity, or purposeful.",
    )
    retrieval_intent: Optional[RetrievalIntentV1] = Field(
        default=None,
        description="Structural retrieval intent derived after stance (PCR).",
    )
    task_hints: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional stance/appraisal hints: task_mode, conversation_frame, "
            "shift_kind, novelty_score, conversation_phase, open_loop_ids, hub_chat_lane."
        ),
    )
    seed_crystallization_id: Optional[str] = Field(
        default=None,
        description="Optional crystallization seed for contradiction or open-loop recall.",
    )
    continuity_digest_max_tokens: Optional[int] = Field(
        default=None,
        description="Render budget override for phase-1 continuity digest.",
    )
    belief_digest_max_tokens: Optional[int] = Field(
        default=None,
        description="Render budget override for phase-3 belief digest.",
    )


class RecallReplyV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    bundle: MemoryBundleV1
    correlation_id: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None


RecallVectorPolicyReasonV1 = Literal[
    "enabled",
    "disabled_global",
    "disabled_profile_vector_top_k_zero",
    "disabled_profile_enable_vector_false",
]


class RecallVectorPolicyPathV1(BaseModel):
    """Per-fetch-path vector gating outcome (see ``source_policy.recall_vector_allowed``)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    allowed: bool
    reason: RecallVectorPolicyReasonV1
    profile: str = ""
    vector_top_k: int = 0
    recall_enable_vector: bool = Field(
        default=True,
        validation_alias=AliasChoices("RECALL_ENABLE_VECTOR", "recall_enable_vector"),
        serialization_alias="RECALL_ENABLE_VECTOR",
    )
    path: str = ""


class RecallVectorPolicyV1(BaseModel):
    """Path-keyed vector policy diagnostics attached under ``recall_debug.vector_policy``."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    main: Optional[RecallVectorPolicyPathV1] = None
    anchor: Optional[RecallVectorPolicyPathV1] = None
    graphtri: Optional[RecallVectorPolicyPathV1] = None
    collectors: Optional[RecallVectorPolicyPathV1] = None
    v2_shadow_exact: Optional[RecallVectorPolicyPathV1] = None
    v2_shadow_semantic: Optional[RecallVectorPolicyPathV1] = None


class RecallSourceGatingV1(BaseModel):
    """Coarse source enablement labels under ``recall_debug.source_gating``."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    vector: Optional[str] = None
    sql_timeline: Optional[str] = None
    sql_chat: Optional[str] = None
    rdf: Optional[str] = None


class RecallAdapterDiagnosticsV1(BaseModel):
    """Substrate recall adapter map/drop counters (``metadata.recall_adapter`` on first mapped node)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    recall_fragments_seen: int = 0
    recall_fragments_mapped: int = 0
    recall_fragments_dropped: int = 0
    dropped_counts_by_reason: Dict[str, int] = Field(default_factory=dict)
    mapped_counts_by_source: Dict[str, int] = Field(default_factory=dict)
    mapped_counts_by_node_type: Dict[str, int] = Field(default_factory=dict)
    original_sources_seen: List[str] = Field(default_factory=list)


class RecallDebugV1(BaseModel):
    """Documented keys for ``RecallDecisionV1.recall_debug`` / v2 shadow debug payloads."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    vector_policy: Optional[RecallVectorPolicyV1] = None
    source_gating: Optional[RecallSourceGatingV1] = None
    compare_summary: Optional[Dict[str, Any]] = None
    selected_evidence_cards: Optional[List[Dict[str, Any]]] = None
    pressure_events: Optional[List[Dict[str, Any]]] = None


class RecallDecisionV1(BaseModel):
    """Telemetry for recall decisions."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    corr_id: str
    session_id: Optional[str] = None
    node_id: Optional[str] = None
    verb: Optional[str] = None
    profile: Optional[str] = None
    query: str
    selected_ids: List[str] = Field(default_factory=list)
    dropped: Dict[str, int] = Field(default_factory=dict)
    backend_counts: Dict[str, int] = Field(default_factory=dict)
    latency_ms: int = 0
    recall_debug: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Bounded diagnostics for recall explainability. Documented shape: RecallDebugV1 "
            "(vector_policy, source_gating, pressure_events, selected_evidence_cards)."
        ),
    )
    ranking_debug: List[Dict[str, Optional[float | int | str | bool]]] = Field(
        default_factory=list,
        description="Optional relevance diagnostics for ranked candidates.",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
