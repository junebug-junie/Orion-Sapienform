from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _WPBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


RegionScope = Literal[
    "global",
    "us",
    "state",
    "local",
    "technology",
    "science",
    "climate",
    "energy",
    "healthcare",
    "security",
    "hardware",
    "general_world",
]

WorldPulseSourceType = Literal["rss", "api", "html", "manual", "connector"]
WorldPulseSourceStrategy = Literal["rss", "atom", "sitemap", "html_section", "manual_urls", "api"]
Factuality = Literal["high", "medium", "low", "unknown"]
ClaimPromotionStatus = Literal[
    "observed",
    "candidate",
    "corroborated",
    "accepted_working_claim",
    "disputed",
    "rejected",
    "expired",
]
RunStatus = Literal["pending", "running", "completed", "failed", "partial"]
RunRequester = Literal["scheduler", "manual", "test", "hub"]
CoverageStatus = Literal["complete", "partial", "sparse", "empty"]
SectionCoverageState = Literal["covered", "missing", "source_unavailable", "no_articles"]


class WorldPulseAllowedUsesV1(_WPBase):
    digest: bool = True
    claim_extraction: bool = True
    graph_write: bool = False
    stance_capsule: bool = False
    prior_update_candidate: bool = False


class WorldPulseSourceV1(_WPBase):
    source_id: str
    name: str
    type: WorldPulseSourceType
    strategy: WorldPulseSourceStrategy | None = None
    url: str | None = None
    urls: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    allowed_path_prefixes: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    region_scope: RegionScope = "general_world"
    trust_tier: int = Field(ge=1, le=5)
    factuality: Factuality = "unknown"
    bias_profile: str | None = None
    enabled: bool = True
    approved: bool = False
    required: bool = False
    allowed_uses: WorldPulseAllowedUsesV1 = Field(default_factory=WorldPulseAllowedUsesV1)
    politics_allowed: bool = True
    requires_corroboration: bool = False
    max_articles_per_day: int = Field(default=10, ge=0)
    fetch_timeout_seconds: int | None = Field(default=None, ge=1)
    user_agent: str | None = None
    sitemap_max_urls: int = Field(default=100, ge=1, le=500)
    sitemap_max_child_sitemaps: int = Field(default=3, ge=1, le=20)
    html_link_limit: int = Field(default=200, ge=1, le=1000)
    notes: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.strategy:
            return
        inferred = {
            "rss": "rss",
            "api": "api",
            "html": "html_section",
            "manual": "manual_urls",
            "connector": "api",
        }.get(self.type, "rss")
        object.__setattr__(self, "strategy", inferred)


class SourceRegistryV1(_WPBase):
    version: str = "v1"
    sources: list[WorldPulseSourceV1] = Field(default_factory=list)
    required_sections: list[str] = Field(default_factory=list)
    recommended_sections: list[str] = Field(default_factory=list)
    digest_policy: dict[str, Any] = Field(default_factory=dict)
    situation_policy: dict[str, Any] = Field(default_factory=dict)
    ranking_policy: dict[str, Any] = Field(default_factory=dict)
    clustering_policy: dict[str, Any] = Field(default_factory=dict)
    default_limits: dict[str, Any] = Field(default_factory=dict)
    trust_policy: dict[str, Any] = Field(default_factory=dict)
    politics_policy: dict[str, Any] = Field(default_factory=dict)
    stance_policy: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class SourceTrustAssessmentV1(_WPBase):
    source_id: str
    trust_tier: int = Field(ge=1, le=5)
    allowed_uses: WorldPulseAllowedUsesV1
    requires_corroboration: bool
    politics_allowed: bool
    factuality: Factuality = "unknown"
    bias_profile: str | None = None
    confidence_weight: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    assessed_at: datetime


class ArticleRecordV1(_WPBase):
    article_id: str
    run_id: str
    source_id: str
    source_name: str
    url: str
    canonical_url: str | None = None
    title: str
    subtitle: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    fetched_at: datetime
    text_excerpt: str | None = None
    normalized_text_hash: str
    content_hash: str
    language: str | None = "en"
    categories: list[str] = Field(default_factory=list)
    region_scope: RegionScope = "general_world"
    source_trust_tier: int = Field(ge=1, le=5)
    allowed_uses: WorldPulseAllowedUsesV1
    dedupe_key: str
    cluster_id: str | None = None
    extraction_status: str = "pending"
    provenance: dict[str, Any] = Field(default_factory=dict)
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class ArticleClusterV1(_WPBase):
    cluster_id: str
    run_id: str
    title: str
    summary: str
    article_ids: list[str] = Field(default_factory=list)
    representative_article_id: str | None = None
    topic_ids: list[str] = Field(default_factory=list)
    category: str
    categories: list[str] = Field(default_factory=list)
    region_scope: RegionScope = "general_world"
    topic_terms: list[str] = Field(default_factory=list)
    article_count: int = 0
    source_ids: list[str] = Field(default_factory=list)
    source_count: int = 0
    source_tiers_present: list[int] = Field(default_factory=list)
    source_agreement: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: datetime


class ClaimRecordV1(_WPBase):
    claim_id: str
    run_id: str
    article_id: str
    cluster_id: str | None = None
    topic_id: str | None = None
    claim_text: str
    claim_type: str = "factual"
    subject_entities: list[str] = Field(default_factory=list)
    object_entities: list[str] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    region_scope: RegionScope = "general_world"
    temporal_scope: str | None = None
    valid_as_of: datetime | None = None
    expires_at: datetime | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_trust_tier: int = Field(ge=1, le=5)
    corroboration_status: str = "uncorroborated"
    promotion_status: ClaimPromotionStatus = "observed"
    controversy_level: str = "low"
    source_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    extracted_at: datetime


class EntityRecordV1(_WPBase):
    entity_id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    entity_type: str
    description: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    first_seen_at: datetime
    last_seen_at: datetime
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    external_refs: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


class EventRecordV1(_WPBase):
    event_id: str
    run_id: str
    title: str
    event_type: str
    summary: str
    occurred_at: datetime | None = None
    detected_at: datetime
    location_entities: list[str] = Field(default_factory=list)
    involved_entities: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    article_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    volatility: str = "low"
    status: str = "developing"
    provenance: dict[str, Any] = Field(default_factory=dict)


class TopicRecordV1(_WPBase):
    topic_id: str
    title: str
    normalized_key: str
    category: str
    region_scope: RegionScope = "general_world"
    relevance_tags: list[str] = Field(default_factory=list)
    description: str = ""
    created_at: datetime
    last_seen_at: datetime
    active: bool = True
    tracking_status: str = "candidate"


class SituationEvidenceV1(_WPBase):
    evidence_id: str
    run_id: str
    source_id: str
    article_id: str
    claim_id: str | None = None
    event_id: str | None = None
    quote_excerpt: str | None = None
    evidence_summary: str
    trust_tier: int = Field(ge=1, le=5)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    url: str | None = None
    canonical_url: str | None = None
    published_at: datetime | None = None
    captured_at: datetime


class SituationObservationV1(_WPBase):
    observation_id: str
    topic_id: str
    run_id: str
    observation_summary: str
    supporting_claim_ids: list[str] = Field(default_factory=list)
    supporting_event_ids: list[str] = Field(default_factory=list)
    supporting_article_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    observed_at: datetime
    expires_at: datetime | None = None
    status: str = "active"


class SituationPriorUpdateCandidateV1(_WPBase):
    candidate_id: str
    topic_id: str
    run_id: str
    existing_prior: str
    new_evidence: str
    proposed_update: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    requires_review: bool = True
    affected_orion_contexts: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    created_at: datetime
    status: str = "candidate"


class SituationRelevanceV1(_WPBase):
    juniper: float = Field(default=0.0, ge=0.0, le=1.0)
    orion_lab: float = Field(default=0.0, ge=0.0, le=1.0)
    local_utah: float = Field(default=0.0, ge=0.0, le=1.0)
    general_world: float = Field(default=0.0, ge=0.0, le=1.0)


class SituationChangeV1(_WPBase):
    change_id: str
    topic_id: str
    run_id: str
    change_type: str
    change_summary: str
    previous_state: str | None = None
    new_state: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    importance: float = Field(default=0.0, ge=0.0, le=1.0)
    relevance: SituationRelevanceV1 = Field(default_factory=SituationRelevanceV1)
    requires_followup: bool = False
    expires_or_recheck_after: datetime | None = None
    created_at: datetime


class TopicSituationBriefV1(_WPBase):
    topic_id: str
    title: str
    scope: str
    category: str
    region_scope: RegionScope = "general_world"
    current_assessment: str
    previous_assessment: str | None = None
    last_updated: datetime
    first_seen_at: datetime
    status: str = "developing"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    volatility: str = "low"
    controversy: str = "low"
    source_agreement: str = "unknown"
    relevance: SituationRelevanceV1 = Field(default_factory=SituationRelevanceV1)
    known_facts: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    recent_changes: list[str] = Field(default_factory=list)
    prior_assumptions: list[str] = Field(default_factory=list)
    prior_update_candidates: list[str] = Field(default_factory=list)
    source_mix: dict[str, int] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    article_ids: list[str] = Field(default_factory=list)
    watch_conditions: list[str] = Field(default_factory=list)
    next_recheck_at: datetime | None = None
    stance_eligible: bool = False
    epistemic_posture: dict[str, Any] = Field(default_factory=dict)
    tracking_status: str = "candidate"
    created_at: datetime
    updated_at: datetime


class WorldLearningDeltaV1(_WPBase):
    learning_id: str
    run_id: str
    topic_id: str
    category: str
    summary: str
    why_it_matters: str
    entities: list[str] = Field(default_factory=list)
    claims: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_count: int = 0
    expires_at: datetime | None = None
    relevance_tags: list[str] = Field(default_factory=list)
    stance_eligible: bool = False
    graph_eligible: bool = False
    prior_update_candidate_ids: list[str] = Field(default_factory=list)
    created_at: datetime


class WorthReadingItemV1(_WPBase):
    reading_id: str
    title: str
    source_id: str
    article_id: str | None = None
    url: str | None = None
    reason_selected: str
    reading_type: str
    trust_tier: int = Field(ge=1, le=5)
    category: str
    topic_ids: list[str] = Field(default_factory=list)
    priority: int = 1
    created_at: datetime


class WorthWatchingItemV1(_WPBase):
    watch_id: str
    topic_id: str
    title: str
    reason: str
    watch_condition: str
    recheck_after: datetime | None = None
    category: str
    region_scope: RegionScope = "general_world"
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    volatility: str = "low"
    priority: int = 1
    created_at: datetime


class DailyWorldPulseItemV1(_WPBase):
    item_id: str
    run_id: str
    title: str
    category: str
    region_scope: RegionScope = "general_world"
    summary: str
    why_it_matters: str
    what_changed: str
    context_bullets: list[str] = Field(default_factory=list)
    by_the_numbers: list[str] = Field(default_factory=list)
    what_theyre_saying: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    orion_read: str
    what_to_watch: list[str] = Field(default_factory=list)
    worth_reading: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    article_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    topic_ids: list[str] = Field(default_factory=list)
    situation_change_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    volatility: str = "low"
    source_agreement: str = "unknown"
    stance_eligible: bool = False
    created_at: datetime


class DailyWorldPulseSectionsV1(_WPBase):
    us_politics: list[str] = Field(default_factory=list)
    global_politics: list[str] = Field(default_factory=list)
    local_politics: list[str] = Field(default_factory=list)
    ai_technology: list[str] = Field(default_factory=list)
    science_climate_energy: list[str] = Field(default_factory=list)
    healthcare_mental_health: list[str] = Field(default_factory=list)
    security_infrastructure_software: list[str] = Field(default_factory=list)
    hardware_compute_gpu: list[str] = Field(default_factory=list)
    local_conditions: list[str] = Field(default_factory=list)


class SectionCoverageV1(_WPBase):
    sources_enabled: int = 0
    sources_fetched: int = 0
    articles_accepted: int = 0
    digest_items: int = 0
    status: SectionCoverageState = "missing"


class SectionRollupV1(_WPBase):
    section: str
    status: SectionCoverageState = "missing"
    article_count: int = 0
    cluster_count: int = 0
    digest_item_count: int = 0
    top_topic_ids: list[str] = Field(default_factory=list)
    summary: str = ""
    source_notes: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class DailyWorldPulseV1(_WPBase):
    run_id: str
    date: str
    generated_at: datetime
    title: str
    executive_summary: str
    sections: DailyWorldPulseSectionsV1
    items: list[DailyWorldPulseItemV1] = Field(default_factory=list)
    things_worth_reading: list[WorthReadingItemV1] = Field(default_factory=list)
    things_worth_watching: list[WorthWatchingItemV1] = Field(default_factory=list)
    orion_analysis_layer: str
    source_notes: list[str] = Field(default_factory=list)
    confidence_summary: str = ""
    coverage_status: CoverageStatus = "empty"
    section_coverage: dict[str, SectionCoverageV1] = Field(default_factory=dict)
    section_rollups: list[SectionRollupV1] = Field(default_factory=list)
    accepted_article_count: int = 0
    article_cluster_count: int = 0
    max_digest_items_total: int = 0
    source_ids: list[str] = Field(default_factory=list)
    article_count: int = 0
    claim_count: int = 0
    event_count: int = 0
    situation_change_count: int = 0
    stance_capsule_id: str | None = None
    graph_delta_id: str | None = None
    hub_message_id: str | None = None
    email_status: str = "not_requested"
    created_at: datetime


class WorldContextTopicV1(_WPBase):
    topic_id: str
    topic: str
    summary: str
    relevance_tags: list[str] = Field(default_factory=list)
    expires_at: datetime | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    use_when: list[str] = Field(default_factory=list)
    do_not_volunteer: bool = True


class WorldContextCapsuleV1(_WPBase):
    capsule_id: str
    run_id: str
    date: str
    locality: str = "Utah"
    generated_at: datetime
    salient_topics: list[WorldContextTopicV1] = Field(default_factory=list)
    politics_context: dict[str, Any] = Field(default_factory=dict)
    local_conditions: list[str] = Field(default_factory=list)
    security_advisories: list[str] = Field(default_factory=list)
    orion_lab_relevant_items: list[str] = Field(default_factory=list)
    use_policy: dict[str, Any] = Field(default_factory=dict)
    expires_at: datetime | None = None
    stance_eligible_item_ids: list[str] = Field(default_factory=list)
    created_at: datetime


class GraphDeltaPlanV1(_WPBase):
    graph_delta_id: str
    run_id: str
    graph_name: str
    triples: str
    summary: str
    triple_count: int = 0
    policy_stamp: dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = True
    created_at: datetime


class HubWorldPulseMessageV1(_WPBase):
    message_id: str
    run_id: str
    title: str
    date: str
    executive_summary: str
    cards: list[DailyWorldPulseItemV1] = Field(default_factory=list)
    worth_reading: list[WorthReadingItemV1] = Field(default_factory=list)
    worth_watching: list[WorthWatchingItemV1] = Field(default_factory=list)
    rendered_markdown: str = ""
    structured_payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class EmailWorldPulseRenderV1(_WPBase):
    run_id: str
    subject: str
    opening: str
    plaintext_body: str
    html_body: str | None = None
    to: list[str] = Field(default_factory=list)
    from_email: str | None = None
    dry_run: bool = True
    created_at: datetime


class WorldPulseRunV1(_WPBase):
    run_id: str
    date: str
    started_at: datetime
    completed_at: datetime | None = None
    status: RunStatus = "pending"
    requested_by: RunRequester = "manual"
    dry_run: bool = True
    sources_considered: int = 0
    sources_fetched: int = 0
    sources_failed: int = 0
    sources_skipped: int = 0
    articles_fetched: int = 0
    articles_accepted: int = 0
    claims_extracted: int = 0
    events_extracted: int = 0
    entities_extracted: int = 0
    situation_briefs_updated: int = 0
    situation_changes_created: int = 0
    digest_created: bool = False
    sql_emit_status: str = "pending"
    graph_emit_status: str = "pending"
    email_status: str = "pending"
    hub_publish_status: str = "pending"
    stance_capsule_status: str = "pending"
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class WorldPulseRunResultV1(_WPBase):
    run: WorldPulseRunV1
    digest: DailyWorldPulseV1 | None = None
    capsule: WorldContextCapsuleV1 | None = None
    graph_delta_plan: GraphDeltaPlanV1 | None = None
    publish_status: dict[str, Any] = Field(default_factory=dict)
