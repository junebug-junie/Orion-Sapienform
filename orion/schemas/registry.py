from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type

from pydantic import BaseModel


@dataclass(frozen=True)
class SchemaRegistration:
    model: Type[BaseModel]
    kind: str

from orion.core.bus.bus_schemas import (
    ChatRequestPayload,
    ChatResultPayload,
    RecallRequestPayload,
    RecallResultPayload,
)
from orion.core.contracts.memory_cards import MemoryCardV1
from orion.core.contracts.substrate_read import (
    SubstrateReadQueryV1,
    SubstrateReadReplyV1,
)
from orion.core.contracts.recall import (
    RecallAdapterDiagnosticsV1,
    RecallDebugV1,
    RecallDecisionV1,
    RecallQueryV1,
    RecallReplyV1,
    RecallSourceGatingV1,
    RecallVectorPolicyPathV1,
    RecallVectorPolicyV1,
)
from orion.core.verbs.models import VerbEffectV1, VerbRequestV1, VerbResultV1
from orion.schemas.actions.daily import DailyMetacogV1, DailyPulseV1
from orion.schemas.affectgpt import (
    AffectGptAssessRequestPayload,
    AffectGptAssessResultPayload,
    JuniperMultimodalAffectV1,
)
from orion.schemas.actions.mesh_ops import (
    DiskHealthDeviceV1,
    DiskHealthSnapshotV1,
    DockerPruneResultV1,
    DockerPruneSnapshotV1,
    MeshNodeStatusV1,
    MeshOpsRoundResultV1,
    MeshStatusSnapshotV1,
    OpsMeshRoundJournalEntryV1,
    RepoPullRequestDigestItemV1,
    RepoRecentChangesDigestV1,
)
from orion.journaler.schemas import JournalEntryDraftV1, JournalEntryIndexV1, JournalEntryWriteV1, JournalTriggerV1
from orion.schemas.collapse_mirror import CollapseMirrorEntryV2, CollapseMirrorStoredV1
from orion.schemas.cortex.contracts import (
    AgentTraceStepV1,
    AgentTraceSummaryV1,
    AgentTraceToolStatV1,
    CortexClientRequest,
    CortexClientResult,
    CortexChatRequest,
    CortexChatResult,
    RecallDirective,
    AutoRouteDecisionV1,
    AutoDepthDecisionV1,
)
from orion.schemas.cortex.exec import CortexExecRequestPayload, CortexExecResultPayload
from orion.schemas.pre_turn_appraisal import (
    PreTurnAppraisalRequestV1,
    TurnAppraisalBundleV1,
    TurnAppraisalParadigmSliceV1,
    TurnWindowMessageV1,
)
from orion.schemas.cortex.schemas import PlanExecutionRequest, PlanExecutionResult
from orion.schemas.mind.artifact import MindRunArtifactV1
from orion.schemas.biometrics_projection import (
    ActiveNodePressureProjectionV1,
    NodeBiometricsProjectionV1,
)
from orion.schemas.execution_projection import (
    ExecutionRunStateV1,
    ExecutionTrajectoryProjectionV1,
)
from orion.schemas.grammar import GrammarEventV1
from orion.schemas.graph_write_intent import GraphWriteIntentV1
from orion.schemas.memory_consolidation import (
    ChatHistorySparkMetaPatchV1,
    MemoryConsolidationWindowV1,
    MemoryGraphSuggestDraftRecordV1,
    MemoryTurnPersistedV1,
)
from orion.schemas.memory_crystallization import ActiveMemoryPacketV1, MemoryCrystallizationV1
from orion.schemas.context_exec import (
    BeliefProvenanceReportV1,
    ContextExecBudgetV1,
    ContextExecFindingV1,
    ContextExecOperatorSummaryV1,
    ContextExecPermissionV1,
    ContextExecRequestV1,
    ContextExecRunV1,
    ContextExecSafetySummaryV1,
    ContextExecVerbStepV1,
    EvidenceBundle,
    InvestigationReportV2,
    InvestigationSectionV2,
    MemoryCorrectionProposalV1,
    PatchProposalV1,
    ProposalEnvelopeV1,
    RepoImpactAnalysisReportV1,
    SourceResult,
    TraceAutopsyReportV1,
)
from orion.schemas.proposal_ledger import (
    ProposalExecutionEligibilityV1,
    ProposalExecutionReceiptV1,
    ProposalLedgerRecordV1,
    ProposalReviewDecisionV1,
    ProposalTriageDecisionV1,
)
from orion.schemas.self_experiments import (
    SelfExperimentCreateRequestV1,
    SelfExperimentCreateResponseV1,
    SelfExperimentDispatchRequestV1,
    SelfExperimentDispatchResponseV1,
    SelfExperimentListResponseV1,
    SelfExperimentRecordV1,
    SelfExperimentSpecV1,
)
from orion.schemas.affective_state import JuniperAffectiveStateV1
from orion.schemas.doc_semantic_drift import DocSemanticDriftV1
from orion.schemas.dev_economics import DevEconomicsLedgerV1
from orion.schemas.power import PowerIntentSettledV1, PowerIntentV1
from orion.schemas.codebase_delta import CodebaseDeltaV1
from orion.schemas.organ_emission import OrganEmissionV1
from orion.schemas.reduction_receipt import ProjectionUpdateV1, ReductionReceiptV1
from orion.schemas.state_delta import StateDeltaV1
from orion.schemas.substrate_telemetry import SubstrateTierOutcomesPayloadV1
from orion.schemas.transport_projection import TransportBusProjectionV1, TransportBusStateV1
from orion.schemas.agents.bound_capability import (
    BoundCapabilityExecutionFailureV1,
    BoundCapabilityExecutionRequestV1,
    BoundCapabilityExecutionResultV1,
    CapabilityRecoveryDecisionV1,
    CapabilityRecoveryReasonV1,
)
from orion.schemas.platform import CoreEventV1, GenericPayloadV1, SystemErrorV1
from orion.schemas.chat_history import ChatHistoryMessageV1, ChatHistoryTurnV1  # includes memory policy fields
from orion.schemas.chat_response_feedback import ChatResponseFeedbackV1
from orion.schemas.metacognitive_trace import MetacognitiveTraceV1
from orion.schemas.chat_gpt_log import (
    ChatGptConversationV1,
    ChatGptDerivedExampleV1,
    ChatGptImportRunV1,
    ChatGptLogTurnV1,
    ChatGptMessageV1,
)
from orion.schemas.chat_response_feedback import ChatResponseFeedbackV1
from orion.schemas.social_chat import (
    SocialConceptEvidenceV1,
    SocialGroundingStateV1,
    SocialRedactionScoreV1,
    SocialRoomTurnStoredV1,
    SocialRoomTurnV1,
    TownContinuityReadV1,
    TownContinuityTurnV1,
)
from orion.schemas.room_claude import (
    ExternalRoomResponderV1,
    RoomClaudeRequestV1,
    RoomClaudeUtteranceV1,
    RoomTranscriptEntryV1,
)
from orion.schemas.social_bridge import (
    CallSyneRoomMessageV1,
    ExternalRoomMessageV1,
    ExternalRoomParticipantV1,
    ExternalRoomPostRequestV1,
    ExternalRoomPostResultV1,
    ExternalRoomTurnSkippedV1,
)
from orion.schemas.social_commitment import SocialCommitmentResolutionV1, SocialCommitmentV1
from orion.schemas.social_calibration import SocialCalibrationSignalV1, SocialPeerCalibrationV1, SocialTrustBoundaryV1
from orion.schemas.social_context import SocialContextCandidateV1, SocialContextSelectionDecisionV1, SocialContextWindowV1
from orion.schemas.social_inspection import (
    SocialInspectionDecisionTraceV1,
    SocialInspectionSectionV1,
    SocialInspectionSnapshotV1,
)
from orion.schemas.social_freshness import SocialDecaySignalV1, SocialMemoryFreshnessV1, SocialRegroundingDecisionV1
from orion.schemas.social_deliberation import (
    SocialBridgeSummaryV1,
    SocialClarifyingQuestionV1,
    SocialDeliberationDecisionV1,
)
from orion.schemas.social_floor import (
    SocialClosureSignalV1,
    SocialFloorDecisionV1,
    SocialTurnHandoffV1,
)
from orion.schemas.social_claim import (
    SocialClaimAttributionV1,
    SocialClaimRevisionV1,
    SocialClaimStanceV1,
    SocialClaimV1,
    SocialConsensusStateV1,
    SocialDivergenceSignalV1,
)
from orion.schemas.social_memory import (
    SocialParticipantContinuityV1,
    SocialRelationalMemoryUpdateV1,
    SocialRoomContinuityV1,
    SocialStanceSnapshotV1,
)
from orion.schemas.social_epistemic import SocialEpistemicDecisionV1, SocialEpistemicSignalV1
from orion.schemas.social_repair import SocialRepairDecisionV1, SocialRepairSignalV1
from orion.schemas.social_artifact import SocialArtifactProposalV1, SocialArtifactRevisionV1, SocialArtifactConfirmationV1
from orion.schemas.social_autonomy import SocialOpenThreadV1, SocialTurnPolicyDecisionV1
from orion.schemas.social_style import SocialPeerStyleHintV1, SocialRoomRitualSummaryV1, SocialStyleAdaptationSnapshotV1
from orion.schemas.social_skills import SocialSkillRequestV1, SocialSkillResultV1, SocialSkillSelectionV1
from orion.schemas.social_thread import SocialHandoffSignalV1, SocialThreadRoutingDecisionV1, SocialThreadStateV1
from orion.schemas.social_scenario import (
    SocialScenarioEvaluationResultV1,
    SocialScenarioExpectationV1,
    SocialScenarioFixtureV1,
)
from orion.schemas.social_gif import (
    SocialGifIntentV1,
    SocialGifInterpretationV1,
    SocialGifObservedSignalV1,
    SocialGifPolicyDecisionV1,
    SocialGifProxyContextV1,
    SocialGifUsageStateV1,
)
from orion.schemas.social_shakedown import SocialShakedownFixV1, SocialShakedownIssueV1
from orion.schemas.vector.schemas import (
    EmbeddingGenerateV1,
    EmbeddingResultV1,
    VectorDocumentUpsertV1,
    VectorUpsertV1,
    VectorWriteRequest,
)
from orion.core.schemas.concept_induction import ConceptProfile, ConceptProfileDelta
from orion.core.schemas.reasoning import (
    ClaimV1,
    ConceptV1,
    RelationV1,
    ContradictionV1,
    MentorProposalV1,
    PromotionDecisionV1,
    VerbEvaluationV1,
    SparkStateSnapshotV1 as ReasoningSparkStateSnapshotV1,
    ReasoningArtifactBaseV1,
    ReasoningProvenanceV1,
    ReasoningEdgeV1,
)
from orion.core.schemas.mentor import (
    MentorConstraintsV1,
    MentorContextSliceV1,
    MentorGatewayResultV1,
    MentorProposalItemV1,
    MentorRequestV1,
    MentorResponseV1,
)
from orion.core.schemas.spark_canonical import SparkSourceSnapshotV1
from orion.core.schemas.reasoning_io import (
    ReasoningWriteContextV1,
    ReasoningWriteRequestV1,
    ReasoningWriteResultV1,
)
from orion.core.schemas.reasoning_policy import (
    ContradictionFindingV1,
    EntityLifecycleEvaluationRequestV1,
    EntityLifecycleEvaluationResultV1,
    PromotionEvaluationItemV1,
    PromotionEvaluationRequestV1,
    PromotionEvaluationResultV1,
)
from orion.core.schemas.reasoning_summary import (
    ReasoningAutonomySummaryV1,
    ReasoningClaimDigestV1,
    ReasoningConceptDigestV1,
    ReasoningSparkSummaryV1,
    ReasoningSummaryDebugV1,
    ReasoningSummaryRequestV1,
    ReasoningSummaryV1,
)
from orion.core.schemas.endogenous import (
    EndogenousTriggerRequestV1,
    EndogenousTriggerSignalV1,
    EndogenousTriggerDecisionV1,
    EndogenousTriggerDebugV1,
    EndogenousWorkflowActionV1,
    EndogenousWorkflowPlanV1,
    EndogenousWorkflowExecutionResultV1,
    EndogenousHistoryEntryV1,
)
from orion.core.schemas.endogenous_runtime import (
    EndogenousRuntimeAuditV1,
    EndogenousRuntimeConsumptionItemV1,
    EndogenousRuntimeExecutionRecordV1,
    EndogenousRuntimeQueryV1,
    EndogenousRuntimeResultV1,
    EndogenousRuntimeSignalDigestV1,
)
from orion.core.schemas.endogenous_eval import (
    EndogenousCalibrationProfileV1,
    EndogenousCalibrationRecommendationV1,
    EndogenousEvaluationRequestV1,
    EndogenousEvaluationResultV1,
    EndogenousMetricSummaryV1,
    PromotionCalibrationSummaryV1,
    ReasoningSummaryCalibrationSummaryV1,
)
from orion.core.schemas.calibration_adoption import (
    CalibrationAdoptionRequestV1,
    CalibrationAdoptionResultV1,
    CalibrationProfileAuditV1,
    CalibrationProfileResolutionV1,
    CalibrationProfileV1,
    CalibrationRollbackRequestV1,
    CalibrationRollbackResultV1,
    CalibrationRolloutScopeV1,
)
from orion.core.schemas.cognitive_substrate import (
    NodeRefV1,
    EdgeRefV1,
    SubjectRefV1,
    EvidenceRefV1,
    SubstrateActivationV1,
    SubstrateTemporalWindowV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
    BaseSubstrateNodeV1,
    EntityNodeV1,
    ConceptNodeV1,
    EventNodeV1,
    EvidenceNodeV1,
    ContradictionNodeV1,
    TensionNodeV1,
    DriveNodeV1,
    GoalNodeV1,
    StateSnapshotNodeV1,
    HypothesisNodeV1,
    OntologyBranchNodeV1,
    SubstrateEdgeV1,
    SubstrateGraphRecordV1,
)
from orion.core.schemas.frontier_expansion import (
    FrontierContextRefsV1,
    FrontierGraphRegionRefV1,
    FrontierDeltaItemV1,
    FrontierExpansionRequestV1,
    FrontierExpansionResponseV1,
    FrontierGraphDeltaBundleV1,
    FrontierSourceProvenanceV1,
)
from orion.core.schemas.frontier_landing import (
    FrontierLandingRequestV1,
    FrontierDeltaLandingDecisionV1,
    FrontierLandingResultV1,
)
from orion.core.schemas.frontier_curiosity import (
    FrontierInvocationSignalV1,
    FrontierInvocationDecisionV1,
    FrontierInvocationPlanV1,
    FrontierInvocationRunResultV1,
)
from orion.core.schemas.substrate_consolidation import (
    GraphConsolidationRequestV1,
    GraphConsolidationDecisionV1,
    GraphConsolidationResultV1,
    GraphReviewCycleRecordV1,
    GraphStateDeltaDigestV1,
)
from orion.core.schemas.substrate_review_queue import (
    GraphReviewCyclePolicyV1,
    GraphReviewCycleBudgetV1,
    GraphReviewQueueItemV1,
    GraphReviewScheduleDecisionV1,
    GraphReviewQueueSnapshotV1,
)
from orion.core.schemas.substrate_review_runtime import (
    GraphReviewRuntimeRequestV1,
    GraphReviewRuntimeResultV1,
)
from orion.core.schemas.substrate_review_telemetry import (
    GraphReviewTelemetryRecordV1,
    GraphReviewTelemetryQueryV1,
    GraphReviewTelemetrySummaryV1,
    GraphReviewCalibrationRequestV1,
    GraphReviewCalibrationRecommendationV1,
)
from orion.core.schemas.substrate_policy_adoption import (
    SubstratePolicyAdoptionRequestV1,
    SubstratePolicyAdoptionResultV1,
    SubstratePolicyAuditEventV1,
    SubstratePolicyComparisonV1,
    SubstratePolicyInspectionV1,
    SubstratePolicyOverridesV1,
    SubstratePolicyProfileV1,
    SubstratePolicyResolutionV1,
    SubstratePolicyRollbackRequestV1,
    SubstratePolicyRollbackResultV1,
    SubstratePolicyRolloutScopeV1,
)
from orion.core.schemas.substrate_policy_comparison import (
    SubstratePolicyComparisonRequestV1,
    SubstratePolicyEffectivenessReportV1,
    SubstratePolicyMetricDeltaV1,
)
from orion.core.schemas.drives import (
    AutonomyGoalPlannedV1,
    DriveAuditV1,
    DriveStateV1,
    GoalProposalV1,
    IdentitySnapshotV1,
    TensionEventV1,
    TurnDossierV1,
)
from orion.autonomy.models import ActionOutcomeEmitV1
from orion.schemas.telemetry.biometrics import (
    BiometricsPayload,
    BiometricsSampleV1,
    BiometricsSummaryV1,
    BiometricsInductionV1,
    BiometricsClusterV1,
)
from orion.schemas.telemetry.cabinet_ambient_spike import CabinetAmbientSpikeV1
from orion.schemas.telemetry.dream import (
    DreamInternalTriggerV1,
    DreamRequest,
    DreamResultV1,
    DreamTriggerPayload,
)
from orion.schemas.rdf import RdfBuildRequest, RdfWriteRequest, RdfWriteResult
from orion.schemas.spark_concept_graph import SparkConceptProfileGraphMaterializationV1
from orion.schemas.self_study_analysis import (
    AnalysisFindingV1,
    AnalysisMetricV1,
    SelfStudyAnalysisResultV1,
)
from orion.schemas.self_study import (
    SelfConceptEvidenceRefV1,
    SelfConceptInduceResultV1,
    SelfConceptRefV1,
    SelfConceptReflectResultV1,
    SelfStudyHarnessResultV1,
    SelfStudyHarnessScenarioResultV1,
    SelfStudyHarnessSoakResultV1,
    SelfStudyHarnessSummaryV1,
    SelfInducedConceptV1,
    SelfKnowledgeItemV1,
    SelfKnowledgeSectionCountsV1,
    SelfStudyRetrievalBackendStatusV1,
    SelfStudyRetrievalCountsV1,
    SelfStudyRetrievalGroupV1,
    SelfStudyRetrievedRecordV1,
    SelfStudyConsumerContextV1,
    SelfStudyConsumerPolicyDecisionV1,
    SelfStudyRetrieveFiltersV1,
    SelfStudyRetrieveRequestV1,
    SelfStudyRetrieveResultV1,
    SelfReflectiveFindingV1,
    SelfRepoInspectResultV1,
    SelfSnapshotV1,
    SelfWritebackStatusV1,
)
from orion.schemas.telemetry.inner_state import InnerFeatureV1, InnerStateFeaturesV1
from orion.schemas.telemetry.mood_arc import MoodArcCorpusRowV1, MoodArcEncoderManifestV1
from orion.schemas.telemetry.phi_encoder import PhiEncoderManifestV1
from orion.schemas.telemetry.reasoning import ReasoningActivityV1, ReasoningCallV1
from orion.schemas.telemetry.spark import SparkStateSnapshotV1, SparkTelemetryPayload
from orion.schemas.telemetry.spark_ack import SparkStateSnapshotAckV1
from orion.schemas.telemetry.spark_candidate import SparkCandidateV1
from orion.schemas.telemetry.spark_signal import SparkSignalV1
from orion.signals.models import OrionSignalV1
from orion.schemas.telemetry.system_health import (
    EquilibriumSnapshotV1,
    EquilibriumServiceTransitionV1,
    SystemHealthV1,
    BusConsumerReadinessV1,
    ServiceLivenessV1,
)
from orion.schemas.telemetry.rpc_health import RpcHealthSnapshotV1
from orion.schemas.telemetry.cognition_trace import CognitionTracePayload
from orion.schemas.telemetry.metacognition import MetacognitionTickV1
from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1
from orion.schemas.telemetry.meta_tags import MetaTagsPayload, MetaTagsRequestV1, MetaTagsResultV1
from orion.schemas.metacog_patches import MetacogDraftTextPatchV1
from orion.schemas.metacog_entry import MetacogEntryV1, MetacogRepairPressure
from orion.schemas.repair_pressure_appraisal import RepairPressureAppraisalV1
from orion.schemas.telemetry.field_channel_anomaly_score import FieldChannelAnomalyScoreV1
from orion.schemas.state.contracts import StateGetLatestRequest, StateLatestReply
from orion.schemas.world_model import (
    WorldModelFeatureGroupV1,
    WorldModelPredictionPayload,
    WorldModelTaskRequestPayload,
    WorldModelTrajectoryStepV1,
)
from orion.schemas.vision import (
    VisionArtifactPayload,
    VisionSceneInventoryV1,
    VisionCouncilRequestPayload,
    VisionCouncilResultPayload,
    VisionEdgeActivityPayload,
    VisionEdgeArtifact,
    VisionEdgeError,
    VisionEdgeHealth,
    VisionEventBundleItem,
    VisionEventCandidateV1,
    VisionEventPayload,
    VisionFramePointerPayload,
    RetinaClipCaptureRequestPayload,
    RetinaClipCaptureResultPayload,
    VisionGrammarProjectionCandidateV1,
    VisionGuardAlert,
    VisionGuardSignal,
    VisionMemoryDeltaCandidateV1,
    VisionSalientObservationV1,
    VisionSceneEntityV1,
    VisionSceneInterpretationV1,
    VisionSceneRelationV1,
    VisionScribeAckPayload,
    VisionScribeRequestPayload,
    VisionScribeResultPayload,
    VisionTaskRelevanceV1,
    VisionTaskRequestPayload,
    VisionTaskResultPayload,
    VisionUncertaintyV1,
    VisionWindowPayload,
    VisionWindowRequestPayload,
    VisionWindowResultPayload,
)
from orion.schemas.tts import (
    TTSRequestPayload,
    TTSResultPayload,
    STTRequestPayload,
    STTResultPayload,
)
from orion.schemas.notify import (
    ChatAttentionAck,
    ChatAttentionRequest,
    ChatAttentionState,
    ChatMessageNotification,
    ChatMessageReceipt,
    ChatMessageState,
    DeliveryAttempt,
    HubNotificationEvent,
    NotificationAccepted,
    NotificationRecord,
    NotificationReceiptEvent,
    NotificationRequest,
    NotificationPreference,
    NotificationPreferencesUpdate,
    PreferenceResolutionRequest,
    PreferenceResolutionResponse,
    RecipientProfile,
    RecipientProfileUpdate,
)
from orion.schemas.harness_finalize import (
    FinalizeReflectionV1,
    GrammarReceiptV1,
    HarnessDraftMoleculeV1,
    HarnessPostTurnClosureV1,
    HarnessRepairOverlayV1,
    HarnessRunCancelV1,
    HarnessRunRequestV1,
    HarnessRunStepV1,
    HarnessRunV1,
    HarnessTurnOutcomeMoleculeV1,
    HarnessVerdictMoleculeV1,
    SubstrateFinalizeAppraisalV1,
)
from orion.schemas.compaction import MemoryCompactionDeltaV1
from orion.schemas.reverie import (
    CompactionRequestV1,
    ResonanceAlertV1,
    ReverieChainV1,
    ReverieRefractoryEntry,
    SpontaneousThoughtV1,
)
from orion.schemas.reverie_visual import ReverieVisualArtifactV1, ReverieVisualChainV1
from orion.schemas.thought import (
    CoalitionSnapshotV1,
    GroundingCapsuleV1,
    HubAssociationBundleV1,
    StanceHarnessSliceV1,
    StanceReactRequestV1,
    ThoughtDecisionRecordV1,
    ThoughtEventV1,
)
from orion.schemas.attention_frame import (
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
    AttentionSignalV1,
    SalienceFeaturesV1,
    VoluntaryOverrideV1,
)
from orion.schemas.attention_self_model import AttentionSelfModelV1
from orion.schemas.attention_salience import (
    AttentionLoopOutcomeV1,
    AttentionSalienceTraceV1,
    PendingAttentionCardV1,
)
from orion.schemas.chat_stance import ChatStanceBrief
from orion.schemas.situation import (
    AffectContextV1,
    AgendaContextV1,
    CabinetContextV1,
    ConversationPhaseContextV1,
    CuriosityPriorContextV1,
    CuriosityPriorSummaryV1,
    EnvironmentContextV1,
    LabContextV1,
    PerceptionContextV1,
    PlaceContextV1,
    PresenceCompanionV1,
    PresenceContextV1,
    RequestorContextV1,
    ReverieContextV1,
    ReverieSnippetV1,
    SituationAffordanceV1,
    SituationBriefV1,
    SituationDiagnosticsV1,
    SituationPolicyV1,
    SituationPromptFragmentV1,
    SurfaceContextV1,
    TimeContextV1,
    WeatherAlertV1,
    WeatherCurrentV1,
    WeatherForecastWindowV1,
    WeatherPracticalFlagsV1,
)
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1
from orion.schemas.field_goal import DominanceStreakTickV1, FieldGoalProvenanceV1
from orion.schemas.field_state import FieldEdgeV1, FieldStateV1
from orion.schemas.causal_geometry import (
    CausalGeometryDivergenceEntryV1,
    CausalGeometryEdgeV1,
    CausalGeometrySnapshotV1,
)
from orion.schemas.execution_dispatch_frame import (
    ExecutionDispatchCandidateV1,
    ExecutionDispatchFrameV1,
)
from orion.schemas.consolidation_frame import (
    ConsolidationFrameV1,
    ExpectationV1,
    MotifObservationV1,
    SchemaCandidateV1,
    SparseTensorSliceV1,
)
from orion.schemas.feedback_frame import FeedbackFrameV1, OutcomeObservationV1
from orion.schemas.brain_frame import (
    BrainEdgeSampleV1,
    BrainNodeSampleV1,
    BrainRegionV1,
    BrainSpotlightV1,
    SubstrateBrainFrameV1,
)
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1, PolicyDecisionV1
from orion.schemas.proposal_frame import ProposalCandidateV1, ProposalFrameV1
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1
from orion.schemas.evidence_index import (
    EvidenceQueryResultItemV1,
    EvidenceQueryV1,
    EvidenceUnitV1,
    MarkdownSpecIngestV1,
    ParsedDocumentBlockV1,
    ParsedDocumentIngestV1,
    ParsedDocumentSectionV1,
)
from orion.schemas.graph_compression import (
    CompressionRegionV1,
    CompressionStalenessMarkV1,
    GraphCompressionRegionMaterializedV1,
)
from orion.schemas.topic_foundry import (
    TopicFoundryDriftAlertV1,
    TopicFoundryEnrichCompleteV1,
    TopicFoundryRunCompleteV1,
)
from orion.schemas.workflow_execution import (
    WorkflowDispatchRequestV1,
    WorkflowExecutionPolicyV1,
    WorkflowScheduleAnalyticsV1,
    WorkflowScheduleEventRecordV1,
    WorkflowScheduleManageRequestV1,
    WorkflowScheduleManageResponseV1,
    WorkflowScheduleRecordV1,
    WorkflowScheduleRunRecordV1,
    WorkflowScheduleSpecV1,
)
from orion.schemas.world_pulse import (
    ArticleClusterV1,
    ArticleRecordV1,
    ClaimRecordV1,
    DailyWorldPulseItemV1,
    DailyWorldPulseV1,
    EmailWorldPulseRenderV1,
    EntityRecordV1,
    EventRecordV1,
    GraphDeltaPlanV1,
    HubWorldPulseMessageV1,
    SituationChangeV1,
    SituationEvidenceV1,
    SituationObservationV1,
    SituationPriorUpdateCandidateV1,
    SourceRegistryV1,
    SourceTrustAssessmentV1,
    TopicRecordV1,
    TopicSituationBriefV1,
    WorldContextCapsuleV1,
    WorldLearningDeltaV1,
    WorldPulseRunResultV1,
    WorldPulseRunV1,
    WorldPulseSourceV1,
    WorthReadingItemV1,
    WorthWatchingItemV1,
)
from orion.schemas.embodiment import (
    EmbodimentIntentV1,
    EmbodimentOutcomeV1,
    OrionTownPersonaV1,
    WorldPerceptionV1,
)
from orion.schemas.self_study_enrichment import SelfStudyEnrichmentRequestV1

_REGISTRY: Dict[str, Type[BaseModel]] = {
    "GenericPayloadV1": GenericPayloadV1,
    "CoreEventV1": CoreEventV1,
    "SystemErrorV1": SystemErrorV1,
    "VerbRequestV1": VerbRequestV1,
    "VerbResultV1": VerbResultV1,
    "VerbEffectV1": VerbEffectV1,
    "ChatRequestPayload": ChatRequestPayload,
    "ChatResultPayload": ChatResultPayload,
    "RecallRequestPayload": RecallRequestPayload,
    "RecallResultPayload": RecallResultPayload,
    "RecallDecisionV1": RecallDecisionV1,
    "RecallReplyV1": RecallReplyV1,
    "RecallQueryV1": RecallQueryV1,
    # SubstrateReadService (2026-08-12, design doc
    # 2026-08-12-substrate-action-perception-design.md option B(2)): lets a
    # verb ask the substrate about the real host it runs on. Registered as a
    # pair -- an unregistered event shape must never be published
    # (CLAUDE.md section 6).
    "SubstrateReadQueryV1": SubstrateReadQueryV1,
    "SubstrateReadReplyV1": SubstrateReadReplyV1,
    "RecallDebugV1": RecallDebugV1,
    "RecallVectorPolicyV1": RecallVectorPolicyV1,
    "RecallVectorPolicyPathV1": RecallVectorPolicyPathV1,
    "RecallSourceGatingV1": RecallSourceGatingV1,
    "RecallAdapterDiagnosticsV1": RecallAdapterDiagnosticsV1,
    "MindRunArtifactV1": MindRunArtifactV1,
    "SubstrateTierOutcomesPayloadV1": SubstrateTierOutcomesPayloadV1,
    "GrammarEventV1": GrammarEventV1,
    "OrganEmissionV1": OrganEmissionV1,
    "ReductionReceiptV1": ReductionReceiptV1,
    "StateDeltaV1": StateDeltaV1,
    "ProjectionUpdateV1": ProjectionUpdateV1,
    "NodeBiometricsProjectionV1": NodeBiometricsProjectionV1,
    "ActiveNodePressureProjectionV1": ActiveNodePressureProjectionV1,
    "ExecutionRunStateV1": ExecutionRunStateV1,
    "ExecutionTrajectoryProjectionV1": ExecutionTrajectoryProjectionV1,
    "TransportBusStateV1": TransportBusStateV1,
    "TransportBusProjectionV1": TransportBusProjectionV1,
    "CodebaseDeltaV1": CodebaseDeltaV1,
    "JuniperAffectiveStateV1": JuniperAffectiveStateV1,
    "DocSemanticDriftV1": DocSemanticDriftV1,
    "DevEconomicsLedgerV1": DevEconomicsLedgerV1,
    "PowerIntentV1": PowerIntentV1,
    "PowerIntentSettledV1": PowerIntentSettledV1,
    "CortexClientRequest": CortexClientRequest,
    "CortexClientResult": CortexClientResult,
    "AgentTraceToolStatV1": AgentTraceToolStatV1,
    "AgentTraceStepV1": AgentTraceStepV1,
    "AgentTraceSummaryV1": AgentTraceSummaryV1,
    "CortexExecRequestPayload": CortexExecRequestPayload,
    "CortexExecResultPayload": CortexExecResultPayload,
    "PreTurnAppraisalRequestV1": PreTurnAppraisalRequestV1,
    "TurnAppraisalBundleV1": TurnAppraisalBundleV1,
    "TurnAppraisalParadigmSliceV1": TurnAppraisalParadigmSliceV1,
    "TurnWindowMessageV1": TurnWindowMessageV1,
    "PlanExecutionRequest": PlanExecutionRequest,
    "PlanExecutionResult": PlanExecutionResult,
    "CollapseMirrorEntryV2": CollapseMirrorEntryV2,  # change_type dict coercion support
    "CollapseMirrorStoredV1": CollapseMirrorStoredV1,
    "CognitionTracePayload": CognitionTracePayload,
    "MetacognitionTickV1": MetacognitionTickV1,
    "MetacognitiveTraceV1": MetacognitiveTraceV1,
    "MetacogTriggerV1": MetacogTriggerV1,
    "MetacogDraftTextPatchV1": MetacogDraftTextPatchV1,
    "MetacogEntryV1": MetacogEntryV1,
    "RepairPressureAppraisalV1": RepairPressureAppraisalV1,
    "FieldChannelAnomalyScoreV1": FieldChannelAnomalyScoreV1,
    "MetacogRepairPressure": MetacogRepairPressure,
    "InnerStateFeaturesV1": InnerStateFeaturesV1,
    "InnerFeatureV1": InnerFeatureV1,
    "MoodArcCorpusRowV1": MoodArcCorpusRowV1,
    "MoodArcEncoderManifestV1": MoodArcEncoderManifestV1,
    "PhiEncoderManifestV1": PhiEncoderManifestV1,
    "ReasoningCallV1": ReasoningCallV1,
    "ReasoningActivityV1": ReasoningActivityV1,
    "SparkCandidateV1": SparkCandidateV1,
    "SparkSignalV1": SparkSignalV1,
    "SparkStateSnapshotAckV1": SparkStateSnapshotAckV1,
    "SparkStateSnapshotV1": SparkStateSnapshotV1,
    "SparkTelemetryPayload": SparkTelemetryPayload,
    "SystemHealthV1": SystemHealthV1,
    "RpcHealthSnapshotV1": RpcHealthSnapshotV1,
    "BusConsumerReadinessV1": BusConsumerReadinessV1,
    "ServiceLivenessV1": ServiceLivenessV1,
    "EquilibriumSnapshotV1": EquilibriumSnapshotV1,
    "EquilibriumServiceTransitionV1": EquilibriumServiceTransitionV1,
    "RdfWriteRequest": RdfWriteRequest,
    "RdfWriteResult": RdfWriteResult,
    "RdfBuildRequest": RdfBuildRequest,
    "SparkConceptProfileGraphMaterializationV1": SparkConceptProfileGraphMaterializationV1,
    "SelfKnowledgeItemV1": SelfKnowledgeItemV1,
    "SelfKnowledgeSectionCountsV1": SelfKnowledgeSectionCountsV1,
    "SelfSnapshotV1": SelfSnapshotV1,
    "SelfWritebackStatusV1": SelfWritebackStatusV1,
    "SelfRepoInspectResultV1": SelfRepoInspectResultV1,
    "SelfConceptEvidenceRefV1": SelfConceptEvidenceRefV1,
    "SelfInducedConceptV1": SelfInducedConceptV1,
    "SelfConceptInduceResultV1": SelfConceptInduceResultV1,
    "SelfConceptRefV1": SelfConceptRefV1,
    "SelfReflectiveFindingV1": SelfReflectiveFindingV1,
    "SelfConceptReflectResultV1": SelfConceptReflectResultV1,
    "AnalysisMetricV1": AnalysisMetricV1,
    "AnalysisFindingV1": AnalysisFindingV1,
    "SelfStudyAnalysisResultV1": SelfStudyAnalysisResultV1,
    "SelfStudyRetrieveFiltersV1": SelfStudyRetrieveFiltersV1,
    "SelfStudyRetrieveRequestV1": SelfStudyRetrieveRequestV1,
    "SelfStudyRetrievedRecordV1": SelfStudyRetrievedRecordV1,
    "SelfStudyRetrievalGroupV1": SelfStudyRetrievalGroupV1,
    "SelfStudyRetrievalCountsV1": SelfStudyRetrievalCountsV1,
    "SelfStudyRetrievalBackendStatusV1": SelfStudyRetrievalBackendStatusV1,
    "SelfStudyRetrieveResultV1": SelfStudyRetrieveResultV1,
    "SelfStudyConsumerPolicyDecisionV1": SelfStudyConsumerPolicyDecisionV1,
    "SelfStudyConsumerContextV1": SelfStudyConsumerContextV1,
    "SelfStudyHarnessScenarioResultV1": SelfStudyHarnessScenarioResultV1,
    "SelfStudyHarnessSoakResultV1": SelfStudyHarnessSoakResultV1,
    "SelfStudyHarnessSummaryV1": SelfStudyHarnessSummaryV1,
    "SelfStudyHarnessResultV1": SelfStudyHarnessResultV1,
    "VisionFramePointerPayload": VisionFramePointerPayload,
    "AffectGptAssessRequestPayload": AffectGptAssessRequestPayload,
    "AffectGptAssessResultPayload": AffectGptAssessResultPayload,
    "JuniperMultimodalAffectV1": JuniperMultimodalAffectV1,
    "RetinaClipCaptureRequestPayload": RetinaClipCaptureRequestPayload,
    "RetinaClipCaptureResultPayload": RetinaClipCaptureResultPayload,
    "VisionTaskRequestPayload": VisionTaskRequestPayload,
    "VisionTaskResultPayload": VisionTaskResultPayload,
    "VisionArtifactPayload": VisionArtifactPayload,
    "VisionEdgeArtifact": VisionEdgeArtifact,
    "VisionEdgeActivityPayload": VisionEdgeActivityPayload,
    "VisionSceneInventoryV1": VisionSceneInventoryV1,
    # Pre-existing gap surfaced by test_registry_and_schema_registry_agree:
    # registered in SCHEMA_REGISTRY, carried on a real channel
    # (orion/bus/channels.yaml:2790) and published by
    # scripts/self_study_enrichment_hook.py, but absent here -- so any
    # consumer calling resolve() on it raised "Unknown schema_id".
    "SelfStudyEnrichmentRequestV1": SelfStudyEnrichmentRequestV1,
    "VisionEdgeHealth": VisionEdgeHealth,
    "VisionEdgeError": VisionEdgeError,
    "VisionEventPayload": VisionEventPayload,
    "VisionEventBundleItem": VisionEventBundleItem,
    "VisionSceneEntityV1": VisionSceneEntityV1,
    "VisionSceneRelationV1": VisionSceneRelationV1,
    "VisionSalientObservationV1": VisionSalientObservationV1,
    "VisionUncertaintyV1": VisionUncertaintyV1,
    "VisionTaskRelevanceV1": VisionTaskRelevanceV1,
    "VisionMemoryDeltaCandidateV1": VisionMemoryDeltaCandidateV1,
    "VisionEventCandidateV1": VisionEventCandidateV1,
    "VisionGrammarProjectionCandidateV1": VisionGrammarProjectionCandidateV1,
    "VisionSceneInterpretationV1": VisionSceneInterpretationV1,
    "VisionWindowPayload": VisionWindowPayload,
    "VisionWindowRequestPayload": VisionWindowRequestPayload,
    "VisionWindowResultPayload": VisionWindowResultPayload,
    "VisionCouncilRequestPayload": VisionCouncilRequestPayload,
    "VisionCouncilResultPayload": VisionCouncilResultPayload,
    "VisionScribeAckPayload": VisionScribeAckPayload,
    "VisionScribeRequestPayload": VisionScribeRequestPayload,
    "VisionScribeResultPayload": VisionScribeResultPayload,
    "VisionGuardSignal": VisionGuardSignal,
    "VisionGuardAlert": VisionGuardAlert,
    "WorldModelFeatureGroupV1": WorldModelFeatureGroupV1,
    "WorldModelTrajectoryStepV1": WorldModelTrajectoryStepV1,
    "WorldModelTaskRequestPayload": WorldModelTaskRequestPayload,
    "WorldModelPredictionPayload": WorldModelPredictionPayload,
    "CortexChatRequest": CortexChatRequest,
    "CortexChatResult": CortexChatResult,
    "RecallDirective": RecallDirective,
    "AutoRouteDecisionV1": AutoRouteDecisionV1,
    "AutoDepthDecisionV1": AutoDepthDecisionV1,
    "AttentionFrameV1": AttentionFrameV1,
    "AttentionSignalV1": AttentionSignalV1,
    "SalienceFeaturesV1": SalienceFeaturesV1,
    "AttentionBroadcastProjectionV1": AttentionBroadcastProjectionV1,
    "VoluntaryOverrideV1": VoluntaryOverrideV1,
    "AttentionSelfModelV1": AttentionSelfModelV1,
    "AttentionSalienceTraceV1": AttentionSalienceTraceV1,
    "AttentionLoopOutcomeV1": AttentionLoopOutcomeV1,
    "PendingAttentionCardV1": PendingAttentionCardV1,
    "ChatStanceBrief": ChatStanceBrief,
    "RequestorContextV1": RequestorContextV1,
    "PresenceCompanionV1": PresenceCompanionV1,
    "PresenceContextV1": PresenceContextV1,
    "TimeContextV1": TimeContextV1,
    "ConversationPhaseContextV1": ConversationPhaseContextV1,
    "PlaceContextV1": PlaceContextV1,
    "WeatherCurrentV1": WeatherCurrentV1,
    "WeatherForecastWindowV1": WeatherForecastWindowV1,
    "WeatherAlertV1": WeatherAlertV1,
    "WeatherPracticalFlagsV1": WeatherPracticalFlagsV1,
    "EnvironmentContextV1": EnvironmentContextV1,
    "AgendaContextV1": AgendaContextV1,
    "LabContextV1": LabContextV1,
    "CabinetContextV1": CabinetContextV1,
    "SurfaceContextV1": SurfaceContextV1,
    "SituationAffordanceV1": SituationAffordanceV1,
    "SituationPolicyV1": SituationPolicyV1,
    "SituationDiagnosticsV1": SituationDiagnosticsV1,
    "PerceptionContextV1": PerceptionContextV1,
    "AffectContextV1": AffectContextV1,
    "CuriosityPriorSummaryV1": CuriosityPriorSummaryV1,
    "CuriosityPriorContextV1": CuriosityPriorContextV1,
    "ReverieSnippetV1": ReverieSnippetV1,
    "ReverieContextV1": ReverieContextV1,
    "SituationBriefV1": SituationBriefV1,
    "SituationPromptFragmentV1": SituationPromptFragmentV1,
    "ChatHistoryMessageV1": ChatHistoryMessageV1,  # includes memory policy + client_meta fields
    "ChatHistoryTurnV1": ChatHistoryTurnV1,  # includes memory policy + client_meta fields
    "ChatResponseFeedbackV1": ChatResponseFeedbackV1,
    "ChatGptLogTurnV1": ChatGptLogTurnV1,
    "ChatGptMessageV1": ChatGptMessageV1,
    "ChatResponseFeedbackV1": ChatResponseFeedbackV1,
    "SocialConceptEvidenceV1": SocialConceptEvidenceV1,
    "SocialGroundingStateV1": SocialGroundingStateV1,
    "SocialRedactionScoreV1": SocialRedactionScoreV1,
    "SocialRoomTurnV1": SocialRoomTurnV1,
    "SocialRoomTurnStoredV1": SocialRoomTurnStoredV1,
    "TownContinuityTurnV1": TownContinuityTurnV1,
    "TownContinuityReadV1": TownContinuityReadV1,
    "RoomClaudeRequestV1": RoomClaudeRequestV1,
    "RoomClaudeUtteranceV1": RoomClaudeUtteranceV1,
    "RoomTranscriptEntryV1": RoomTranscriptEntryV1,
    "ExternalRoomResponderV1": ExternalRoomResponderV1,
    "SocialCommitmentV1": SocialCommitmentV1,
    "SocialCommitmentResolutionV1": SocialCommitmentResolutionV1,
    "SocialBridgeSummaryV1": SocialBridgeSummaryV1,
    "SocialClarifyingQuestionV1": SocialClarifyingQuestionV1,
    "SocialDeliberationDecisionV1": SocialDeliberationDecisionV1,
    "SocialTurnHandoffV1": SocialTurnHandoffV1,
    "SocialClosureSignalV1": SocialClosureSignalV1,
    "SocialFloorDecisionV1": SocialFloorDecisionV1,
    "SocialClaimV1": SocialClaimV1,
    "SocialClaimRevisionV1": SocialClaimRevisionV1,
    "SocialClaimStanceV1": SocialClaimStanceV1,
    "SocialClaimAttributionV1": SocialClaimAttributionV1,
    "SocialConsensusStateV1": SocialConsensusStateV1,
    "SocialDivergenceSignalV1": SocialDivergenceSignalV1,
    "SocialEpistemicSignalV1": SocialEpistemicSignalV1,
    "SocialEpistemicDecisionV1": SocialEpistemicDecisionV1,
    "SocialRepairSignalV1": SocialRepairSignalV1,
    "SocialRepairDecisionV1": SocialRepairDecisionV1,
    "CallSyneRoomMessageV1": CallSyneRoomMessageV1,
    "ExternalRoomParticipantV1": ExternalRoomParticipantV1,
    "ExternalRoomMessageV1": ExternalRoomMessageV1,
    "ExternalRoomPostRequestV1": ExternalRoomPostRequestV1,
    "ExternalRoomPostResultV1": ExternalRoomPostResultV1,
    "ExternalRoomTurnSkippedV1": ExternalRoomTurnSkippedV1,
    "SocialParticipantContinuityV1": SocialParticipantContinuityV1,
    "SocialRoomContinuityV1": SocialRoomContinuityV1,
    "SocialCalibrationSignalV1": SocialCalibrationSignalV1,
    "SocialPeerCalibrationV1": SocialPeerCalibrationV1,
    "SocialTrustBoundaryV1": SocialTrustBoundaryV1,
    "SocialContextCandidateV1": SocialContextCandidateV1,
    "SocialContextSelectionDecisionV1": SocialContextSelectionDecisionV1,
    "SocialContextWindowV1": SocialContextWindowV1,
    "SocialInspectionSnapshotV1": SocialInspectionSnapshotV1,
    "SocialInspectionSectionV1": SocialInspectionSectionV1,
    "SocialInspectionDecisionTraceV1": SocialInspectionDecisionTraceV1,
    "SocialDecaySignalV1": SocialDecaySignalV1,
    "SocialRegroundingDecisionV1": SocialRegroundingDecisionV1,
    "SocialMemoryFreshnessV1": SocialMemoryFreshnessV1,
    "SocialStanceSnapshotV1": SocialStanceSnapshotV1,
    "SocialRelationalMemoryUpdateV1": SocialRelationalMemoryUpdateV1,
    "SocialArtifactProposalV1": SocialArtifactProposalV1,
    "SocialArtifactRevisionV1": SocialArtifactRevisionV1,
    "SocialArtifactConfirmationV1": SocialArtifactConfirmationV1,
    "SocialOpenThreadV1": SocialOpenThreadV1,
    "SocialTurnPolicyDecisionV1": SocialTurnPolicyDecisionV1,
    "SocialPeerStyleHintV1": SocialPeerStyleHintV1,
    "SocialRoomRitualSummaryV1": SocialRoomRitualSummaryV1,
    "SocialStyleAdaptationSnapshotV1": SocialStyleAdaptationSnapshotV1,
    "SocialScenarioFixtureV1": SocialScenarioFixtureV1,
    "SocialScenarioExpectationV1": SocialScenarioExpectationV1,
    "SocialScenarioEvaluationResultV1": SocialScenarioEvaluationResultV1,
    "SocialGifPolicyDecisionV1": SocialGifPolicyDecisionV1,
    "SocialGifIntentV1": SocialGifIntentV1,
    "SocialGifUsageStateV1": SocialGifUsageStateV1,
    "SocialGifObservedSignalV1": SocialGifObservedSignalV1,
    "SocialGifProxyContextV1": SocialGifProxyContextV1,
    "SocialGifInterpretationV1": SocialGifInterpretationV1,
    "SocialShakedownIssueV1": SocialShakedownIssueV1,
    "SocialShakedownFixV1": SocialShakedownFixV1,
    "SocialSkillRequestV1": SocialSkillRequestV1,
    "SocialSkillResultV1": SocialSkillResultV1,
    "SocialSkillSelectionV1": SocialSkillSelectionV1,
    "SocialThreadStateV1": SocialThreadStateV1,
    "SocialThreadRoutingDecisionV1": SocialThreadRoutingDecisionV1,
    "SocialHandoffSignalV1": SocialHandoffSignalV1,
    "VectorWriteRequest": VectorWriteRequest,
    "VectorDocumentUpsertV1": VectorDocumentUpsertV1,
    "VectorUpsertV1": VectorUpsertV1,
    "EmbeddingGenerateV1": EmbeddingGenerateV1,
    "EmbeddingResultV1": EmbeddingResultV1,
    "ConceptProfile": ConceptProfile,
    "ConceptProfileDelta": ConceptProfileDelta,
    "ReasoningArtifactBaseV1": ReasoningArtifactBaseV1,
    "ReasoningProvenanceV1": ReasoningProvenanceV1,
    "ReasoningEdgeV1": ReasoningEdgeV1,
    "ClaimV1": ClaimV1,
    "ConceptV1": ConceptV1,
    "RelationV1": RelationV1,
    "ContradictionV1": ContradictionV1,
    "MentorProposalV1": MentorProposalV1,
    "PromotionDecisionV1": PromotionDecisionV1,
    "VerbEvaluationV1": VerbEvaluationV1,
    "ReasoningSparkStateSnapshotV1": ReasoningSparkStateSnapshotV1,
    "ReasoningWriteContextV1": ReasoningWriteContextV1,
    "ReasoningWriteRequestV1": ReasoningWriteRequestV1,
    "ReasoningWriteResultV1": ReasoningWriteResultV1,
    "ContradictionFindingV1": ContradictionFindingV1,
    "EntityLifecycleEvaluationRequestV1": EntityLifecycleEvaluationRequestV1,
    "EntityLifecycleEvaluationResultV1": EntityLifecycleEvaluationResultV1,
    "PromotionEvaluationItemV1": PromotionEvaluationItemV1,
    "PromotionEvaluationRequestV1": PromotionEvaluationRequestV1,
    "PromotionEvaluationResultV1": PromotionEvaluationResultV1,
    "ReasoningAutonomySummaryV1": ReasoningAutonomySummaryV1,
    "ReasoningClaimDigestV1": ReasoningClaimDigestV1,
    "ReasoningConceptDigestV1": ReasoningConceptDigestV1,
    "ReasoningSparkSummaryV1": ReasoningSparkSummaryV1,
    "ReasoningSummaryDebugV1": ReasoningSummaryDebugV1,
    "ReasoningSummaryRequestV1": ReasoningSummaryRequestV1,
    "ReasoningSummaryV1": ReasoningSummaryV1,
    "EndogenousTriggerRequestV1": EndogenousTriggerRequestV1,
    "EndogenousTriggerSignalV1": EndogenousTriggerSignalV1,
    "EndogenousTriggerDecisionV1": EndogenousTriggerDecisionV1,
    "EndogenousTriggerDebugV1": EndogenousTriggerDebugV1,
    "EndogenousWorkflowActionV1": EndogenousWorkflowActionV1,
    "EndogenousWorkflowPlanV1": EndogenousWorkflowPlanV1,
    "EndogenousWorkflowExecutionResultV1": EndogenousWorkflowExecutionResultV1,
    "EndogenousHistoryEntryV1": EndogenousHistoryEntryV1,
    "EndogenousRuntimeAuditV1": EndogenousRuntimeAuditV1,
    "EndogenousRuntimeConsumptionItemV1": EndogenousRuntimeConsumptionItemV1,
    "EndogenousRuntimeExecutionRecordV1": EndogenousRuntimeExecutionRecordV1,
    "EndogenousRuntimeQueryV1": EndogenousRuntimeQueryV1,
    "EndogenousRuntimeResultV1": EndogenousRuntimeResultV1,
    "EndogenousRuntimeSignalDigestV1": EndogenousRuntimeSignalDigestV1,
    "EndogenousCalibrationProfileV1": EndogenousCalibrationProfileV1,
    "EndogenousCalibrationRecommendationV1": EndogenousCalibrationRecommendationV1,
    "EndogenousEvaluationRequestV1": EndogenousEvaluationRequestV1,
    "EndogenousEvaluationResultV1": EndogenousEvaluationResultV1,
    "EndogenousMetricSummaryV1": EndogenousMetricSummaryV1,
    "PromotionCalibrationSummaryV1": PromotionCalibrationSummaryV1,
    "ReasoningSummaryCalibrationSummaryV1": ReasoningSummaryCalibrationSummaryV1,
    "CalibrationRolloutScopeV1": CalibrationRolloutScopeV1,
    "CalibrationProfileV1": CalibrationProfileV1,
    "CalibrationAdoptionRequestV1": CalibrationAdoptionRequestV1,
    "CalibrationAdoptionResultV1": CalibrationAdoptionResultV1,
    "CalibrationRollbackRequestV1": CalibrationRollbackRequestV1,
    "CalibrationRollbackResultV1": CalibrationRollbackResultV1,
    "CalibrationProfileAuditV1": CalibrationProfileAuditV1,
    "CalibrationProfileResolutionV1": CalibrationProfileResolutionV1,
    "NodeRefV1": NodeRefV1,
    "EdgeRefV1": EdgeRefV1,
    "SubjectRefV1": SubjectRefV1,
    "EvidenceRefV1": EvidenceRefV1,
    "SubstrateActivationV1": SubstrateActivationV1,
    "SubstrateTemporalWindowV1": SubstrateTemporalWindowV1,
    "SubstrateProvenanceV1": SubstrateProvenanceV1,
    "SubstrateSignalBundleV1": SubstrateSignalBundleV1,
    "BaseSubstrateNodeV1": BaseSubstrateNodeV1,
    "EntityNodeV1": EntityNodeV1,
    "ConceptNodeV1": ConceptNodeV1,
    "EventNodeV1": EventNodeV1,
    "EvidenceNodeV1": EvidenceNodeV1,
    "ContradictionNodeV1": ContradictionNodeV1,
    "TensionNodeV1": TensionNodeV1,
    "DriveNodeV1": DriveNodeV1,
    "GoalNodeV1": GoalNodeV1,
    "StateSnapshotNodeV1": StateSnapshotNodeV1,
    "HypothesisNodeV1": HypothesisNodeV1,
    "OntologyBranchNodeV1": OntologyBranchNodeV1,
    "SubstrateEdgeV1": SubstrateEdgeV1,
    "SubstrateGraphRecordV1": SubstrateGraphRecordV1,
    "FrontierContextRefsV1": FrontierContextRefsV1,
    "FrontierGraphRegionRefV1": FrontierGraphRegionRefV1,
    "FrontierDeltaItemV1": FrontierDeltaItemV1,
    "FrontierExpansionRequestV1": FrontierExpansionRequestV1,
    "FrontierExpansionResponseV1": FrontierExpansionResponseV1,
    "FrontierGraphDeltaBundleV1": FrontierGraphDeltaBundleV1,
    "FrontierSourceProvenanceV1": FrontierSourceProvenanceV1,
    "FrontierLandingRequestV1": FrontierLandingRequestV1,
    "FrontierDeltaLandingDecisionV1": FrontierDeltaLandingDecisionV1,
    "FrontierLandingResultV1": FrontierLandingResultV1,
    "FrontierInvocationSignalV1": FrontierInvocationSignalV1,
    "FrontierInvocationDecisionV1": FrontierInvocationDecisionV1,
    "FrontierInvocationPlanV1": FrontierInvocationPlanV1,
    "FrontierInvocationRunResultV1": FrontierInvocationRunResultV1,
    "GraphConsolidationRequestV1": GraphConsolidationRequestV1,
    "GraphConsolidationDecisionV1": GraphConsolidationDecisionV1,
    "GraphConsolidationResultV1": GraphConsolidationResultV1,
    "GraphReviewCycleRecordV1": GraphReviewCycleRecordV1,
    "GraphStateDeltaDigestV1": GraphStateDeltaDigestV1,
    "GraphReviewCyclePolicyV1": GraphReviewCyclePolicyV1,
    "GraphReviewCycleBudgetV1": GraphReviewCycleBudgetV1,
    "GraphReviewQueueItemV1": GraphReviewQueueItemV1,
    "GraphReviewScheduleDecisionV1": GraphReviewScheduleDecisionV1,
    "GraphReviewQueueSnapshotV1": GraphReviewQueueSnapshotV1,
    "GraphReviewRuntimeRequestV1": GraphReviewRuntimeRequestV1,
    "GraphReviewRuntimeResultV1": GraphReviewRuntimeResultV1,
    "GraphReviewTelemetryRecordV1": GraphReviewTelemetryRecordV1,
    "GraphReviewTelemetryQueryV1": GraphReviewTelemetryQueryV1,
    "GraphReviewTelemetrySummaryV1": GraphReviewTelemetrySummaryV1,
    "GraphReviewCalibrationRequestV1": GraphReviewCalibrationRequestV1,
    "GraphReviewCalibrationRecommendationV1": GraphReviewCalibrationRecommendationV1,
    "GraphWriteIntentV1": GraphWriteIntentV1,
    "SubstratePolicyRolloutScopeV1": SubstratePolicyRolloutScopeV1,
    "SubstratePolicyOverridesV1": SubstratePolicyOverridesV1,
    "SubstratePolicyProfileV1": SubstratePolicyProfileV1,
    "SubstratePolicyAdoptionRequestV1": SubstratePolicyAdoptionRequestV1,
    "SubstratePolicyAdoptionResultV1": SubstratePolicyAdoptionResultV1,
    "SubstratePolicyRollbackRequestV1": SubstratePolicyRollbackRequestV1,
    "SubstratePolicyRollbackResultV1": SubstratePolicyRollbackResultV1,
    "SubstratePolicyAuditEventV1": SubstratePolicyAuditEventV1,
    "SubstratePolicyResolutionV1": SubstratePolicyResolutionV1,
    "SubstratePolicyInspectionV1": SubstratePolicyInspectionV1,
    "SubstratePolicyComparisonV1": SubstratePolicyComparisonV1,
    "SubstratePolicyComparisonRequestV1": SubstratePolicyComparisonRequestV1,
    "SubstratePolicyMetricDeltaV1": SubstratePolicyMetricDeltaV1,
    "SubstratePolicyEffectivenessReportV1": SubstratePolicyEffectivenessReportV1,
    "SparkSourceSnapshotV1": SparkSourceSnapshotV1,
    "MentorConstraintsV1": MentorConstraintsV1,
    "MentorContextSliceV1": MentorContextSliceV1,
    "MentorRequestV1": MentorRequestV1,
    "MentorProposalItemV1": MentorProposalItemV1,
    "MentorResponseV1": MentorResponseV1,
    "MentorGatewayResultV1": MentorGatewayResultV1,
    "DriveStateV1": DriveStateV1,
    "DriveAuditV1": DriveAuditV1,
    "IdentitySnapshotV1": IdentitySnapshotV1,
    "GoalProposalV1": GoalProposalV1,
    "AutonomyGoalPlannedV1": AutonomyGoalPlannedV1,
    "ActionOutcomeEmitV1": ActionOutcomeEmitV1,
    "TensionEventV1": TensionEventV1,
    "TurnDossierV1": TurnDossierV1,
    "BiometricsPayload": BiometricsPayload,
    "BiometricsSampleV1": BiometricsSampleV1,
    "BiometricsSummaryV1": BiometricsSummaryV1,
    "BiometricsInductionV1": BiometricsInductionV1,
    "BiometricsClusterV1": BiometricsClusterV1,
    "CabinetAmbientSpikeV1": CabinetAmbientSpikeV1,
    "DreamRequest": DreamRequest,
    "DreamTriggerPayload": DreamTriggerPayload,
    "DreamInternalTriggerV1": DreamInternalTriggerV1,
    "DreamResultV1": DreamResultV1,
    "MetaTagsRequestV1": MetaTagsRequestV1,
    "MetaTagsResultV1": MetaTagsResultV1,
    "MetaTagsPayload": MetaTagsPayload,
    "StateGetLatestRequest": StateGetLatestRequest,
    "StateLatestReply": StateLatestReply,
    "TTSRequestPayload": TTSRequestPayload,
    "TTSResultPayload": TTSResultPayload,
    "STTRequestPayload": STTRequestPayload,
    "STTResultPayload": STTResultPayload,
    "NotificationRequest": NotificationRequest,
    "NotificationAccepted": NotificationAccepted,
    "NotificationRecord": NotificationRecord,
    "NotificationReceiptEvent": NotificationReceiptEvent,
    "DeliveryAttempt": DeliveryAttempt,
    "HubNotificationEvent": HubNotificationEvent,
    "ChatAttentionRequest": ChatAttentionRequest,
    "ChatAttentionAck": ChatAttentionAck,
    "ChatAttentionState": ChatAttentionState,
    "ChatMessageNotification": ChatMessageNotification,
    "ChatMessageReceipt": ChatMessageReceipt,
    "ChatMessageState": ChatMessageState,
    "RecipientProfile": RecipientProfile,
    "RecipientProfileUpdate": RecipientProfileUpdate,
    "NotificationPreference": NotificationPreference,
    "NotificationPreferencesUpdate": NotificationPreferencesUpdate,
    "PreferenceResolutionRequest": PreferenceResolutionRequest,
    "PreferenceResolutionResponse": PreferenceResolutionResponse,
    "DailyPulseV1": DailyPulseV1,
    "JournalTriggerV1": JournalTriggerV1,
    "JournalEntryDraftV1": JournalEntryDraftV1,
    "JournalEntryIndexV1": JournalEntryIndexV1,
    "JournalEntryWriteV1": JournalEntryWriteV1,
    "DailyMetacogV1": DailyMetacogV1,
    "MeshNodeStatusV1": MeshNodeStatusV1,
    "MeshStatusSnapshotV1": MeshStatusSnapshotV1,
    "DiskHealthDeviceV1": DiskHealthDeviceV1,
    "DiskHealthSnapshotV1": DiskHealthSnapshotV1,
    "RepoPullRequestDigestItemV1": RepoPullRequestDigestItemV1,
    "RepoRecentChangesDigestV1": RepoRecentChangesDigestV1,
    "DockerPruneResultV1": DockerPruneResultV1,
    "DockerPruneSnapshotV1": DockerPruneSnapshotV1,
    "MeshOpsRoundResultV1": MeshOpsRoundResultV1,
    "OpsMeshRoundJournalEntryV1": OpsMeshRoundJournalEntryV1,
    "OrionSignalV1": OrionSignalV1,
    "TopicFoundryRunCompleteV1": TopicFoundryRunCompleteV1,
    "TopicFoundryEnrichCompleteV1": TopicFoundryEnrichCompleteV1,
    "TopicFoundryDriftAlertV1": TopicFoundryDriftAlertV1,
    "WorkflowScheduleSpecV1": WorkflowScheduleSpecV1,
    "WorkflowExecutionPolicyV1": WorkflowExecutionPolicyV1,
    "WorkflowDispatchRequestV1": WorkflowDispatchRequestV1,
    "WorkflowScheduleRecordV1": WorkflowScheduleRecordV1,
    "WorkflowScheduleAnalyticsV1": WorkflowScheduleAnalyticsV1,
    "WorkflowScheduleEventRecordV1": WorkflowScheduleEventRecordV1,
    "WorkflowScheduleRunRecordV1": WorkflowScheduleRunRecordV1,
    "WorkflowScheduleManageRequestV1": WorkflowScheduleManageRequestV1,
    "WorkflowScheduleManageResponseV1": WorkflowScheduleManageResponseV1,
    "CapabilityRecoveryReasonV1": CapabilityRecoveryReasonV1,
    "CapabilityRecoveryDecisionV1": CapabilityRecoveryDecisionV1,
    "BoundCapabilityExecutionRequestV1": BoundCapabilityExecutionRequestV1,
    "BoundCapabilityExecutionResultV1": BoundCapabilityExecutionResultV1,
    "BoundCapabilityExecutionFailureV1": BoundCapabilityExecutionFailureV1,
    "FieldEdgeV1": FieldEdgeV1,
    "FieldStateV1": FieldStateV1,
    "FieldAttentionTargetV1": FieldAttentionTargetV1,
    "FieldAttentionFrameV1": FieldAttentionFrameV1,
    "FieldGoalProvenanceV1": FieldGoalProvenanceV1,
    "DominanceStreakTickV1": DominanceStreakTickV1,
    "SelfStateDimensionV1": SelfStateDimensionV1,
    "SelfStateV1": SelfStateV1,
    "PolicyDecisionV1": PolicyDecisionV1,
    "PolicyDecisionFrameV1": PolicyDecisionFrameV1,
    "ProposalCandidateV1": ProposalCandidateV1,
    "ProposalFrameV1": ProposalFrameV1,
    "ExecutionDispatchCandidateV1": ExecutionDispatchCandidateV1,
    "ExecutionDispatchFrameV1": ExecutionDispatchFrameV1,
    "ConsolidationFrameV1": ConsolidationFrameV1,
    "MotifObservationV1": MotifObservationV1,
    "ExpectationV1": ExpectationV1,
    "SparseTensorSliceV1": SparseTensorSliceV1,
    "SchemaCandidateV1": SchemaCandidateV1,
    "FeedbackFrameV1": FeedbackFrameV1,
    "OutcomeObservationV1": OutcomeObservationV1,
    "SubstrateBrainFrameV1": SubstrateBrainFrameV1,
    "CausalGeometryEdgeV1": CausalGeometryEdgeV1,
    "CausalGeometryDivergenceEntryV1": CausalGeometryDivergenceEntryV1,
    "CausalGeometrySnapshotV1": CausalGeometrySnapshotV1,
    "BrainRegionV1": BrainRegionV1,
    "BrainSpotlightV1": BrainSpotlightV1,
    "BrainNodeSampleV1": BrainNodeSampleV1,
    "BrainEdgeSampleV1": BrainEdgeSampleV1,
    "EvidenceUnitV1": EvidenceUnitV1,
    "EvidenceQueryV1": EvidenceQueryV1,
    "EvidenceQueryResultItemV1": EvidenceQueryResultItemV1,
    "MarkdownSpecIngestV1": MarkdownSpecIngestV1,
    "ParsedDocumentIngestV1": ParsedDocumentIngestV1,
    "ParsedDocumentSectionV1": ParsedDocumentSectionV1,
    "ParsedDocumentBlockV1": ParsedDocumentBlockV1,
    "ChatGptImportRunV1": ChatGptImportRunV1,
    "ChatGptConversationV1": ChatGptConversationV1,
    "ChatGptDerivedExampleV1": ChatGptDerivedExampleV1,
    "SourceRegistryV1": SourceRegistryV1,
    "WorldPulseSourceV1": WorldPulseSourceV1,
    "SourceTrustAssessmentV1": SourceTrustAssessmentV1,
    "ArticleRecordV1": ArticleRecordV1,
    "ArticleClusterV1": ArticleClusterV1,
    "ClaimRecordV1": ClaimRecordV1,
    "EntityRecordV1": EntityRecordV1,
    "EventRecordV1": EventRecordV1,
    "TopicRecordV1": TopicRecordV1,
    "SituationEvidenceV1": SituationEvidenceV1,
    "SituationObservationV1": SituationObservationV1,
    "SituationChangeV1": SituationChangeV1,
    "TopicSituationBriefV1": TopicSituationBriefV1,
    "SituationPriorUpdateCandidateV1": SituationPriorUpdateCandidateV1,
    "WorldLearningDeltaV1": WorldLearningDeltaV1,
    "DailyWorldPulseItemV1": DailyWorldPulseItemV1,
    "WorthReadingItemV1": WorthReadingItemV1,
    "WorthWatchingItemV1": WorthWatchingItemV1,
    "DailyWorldPulseV1": DailyWorldPulseV1,
    "WorldContextCapsuleV1": WorldContextCapsuleV1,
    "WorldPulseRunV1": WorldPulseRunV1,
    "WorldPulseRunResultV1": WorldPulseRunResultV1,
    "GraphDeltaPlanV1": GraphDeltaPlanV1,
    "HubWorldPulseMessageV1": HubWorldPulseMessageV1,
    "EmailWorldPulseRenderV1": EmailWorldPulseRenderV1,
    "CompressionRegionV1": CompressionRegionV1,
    "CompressionStalenessMarkV1": CompressionStalenessMarkV1,
    "GraphCompressionRegionMaterializedV1": GraphCompressionRegionMaterializedV1,
    "MemoryCardV1": MemoryCardV1,
    "MemoryCrystallizationV1": MemoryCrystallizationV1,
    "ActiveMemoryPacketV1": ActiveMemoryPacketV1,
    "MemoryTurnPersistedV1": MemoryTurnPersistedV1,
    "ChatHistorySparkMetaPatchV1": ChatHistorySparkMetaPatchV1,
    "MemoryConsolidationWindowV1": MemoryConsolidationWindowV1,
    "MemoryGraphSuggestDraftRecordV1": MemoryGraphSuggestDraftRecordV1,
    "ContextExecRequestV1": ContextExecRequestV1,
    "ContextExecRunV1": ContextExecRunV1,
    "ContextExecOperatorSummaryV1": ContextExecOperatorSummaryV1,
    "ContextExecSafetySummaryV1": ContextExecSafetySummaryV1,
    "ContextExecPermissionV1": ContextExecPermissionV1,
    "ContextExecBudgetV1": ContextExecBudgetV1,
    "ContextExecFindingV1": ContextExecFindingV1,
    "ContextExecVerbStepV1": ContextExecVerbStepV1,
    "BeliefProvenanceReportV1": BeliefProvenanceReportV1,
    "TraceAutopsyReportV1": TraceAutopsyReportV1,
    "RepoImpactAnalysisReportV1": RepoImpactAnalysisReportV1,
    "InvestigationReportV2": InvestigationReportV2,
    "InvestigationSectionV2": InvestigationSectionV2,
    "EvidenceBundle": EvidenceBundle,
    "SourceResult": SourceResult,
    "PatchProposalV1": PatchProposalV1,
    "MemoryCorrectionProposalV1": MemoryCorrectionProposalV1,
    "ProposalEnvelopeV1": ProposalEnvelopeV1,
    "ProposalLedgerRecordV1": ProposalLedgerRecordV1,
    "ProposalTriageDecisionV1": ProposalTriageDecisionV1,
    "ProposalReviewDecisionV1": ProposalReviewDecisionV1,
    "ProposalExecutionEligibilityV1": ProposalExecutionEligibilityV1,
    "ProposalExecutionReceiptV1": ProposalExecutionReceiptV1,
    "SelfExperimentSpecV1": SelfExperimentSpecV1,
    "SelfExperimentCreateRequestV1": SelfExperimentCreateRequestV1,
    "SelfExperimentCreateResponseV1": SelfExperimentCreateResponseV1,
    "SelfExperimentRecordV1": SelfExperimentRecordV1,
    "SelfExperimentDispatchRequestV1": SelfExperimentDispatchRequestV1,
    "SelfExperimentDispatchResponseV1": SelfExperimentDispatchResponseV1,
    "SelfExperimentListResponseV1": SelfExperimentListResponseV1,
    "CoalitionSnapshotV1": CoalitionSnapshotV1,
    "StanceHarnessSliceV1": StanceHarnessSliceV1,
    "HubAssociationBundleV1": HubAssociationBundleV1,
    "ThoughtEventV1": ThoughtEventV1,
    "ThoughtDecisionRecordV1": ThoughtDecisionRecordV1,
    "GroundingCapsuleV1": GroundingCapsuleV1,
    "SpontaneousThoughtV1": SpontaneousThoughtV1,
    "ReverieChainV1": ReverieChainV1,
    "ReverieRefractoryEntry": ReverieRefractoryEntry,
    "CompactionRequestV1": CompactionRequestV1,
    "MemoryCompactionDeltaV1": MemoryCompactionDeltaV1,
    "ResonanceAlertV1": ResonanceAlertV1,
    "ReverieVisualChainV1": ReverieVisualChainV1,
    "ReverieVisualArtifactV1": ReverieVisualArtifactV1,
    "StanceReactRequestV1": StanceReactRequestV1,
    "GrammarReceiptV1": GrammarReceiptV1,
    "HarnessDraftMoleculeV1": HarnessDraftMoleculeV1,
    "SubstrateFinalizeAppraisalV1": SubstrateFinalizeAppraisalV1,
    "FinalizeReflectionV1": FinalizeReflectionV1,
    "HarnessVerdictMoleculeV1": HarnessVerdictMoleculeV1,
    "HarnessTurnOutcomeMoleculeV1": HarnessTurnOutcomeMoleculeV1,
    "HarnessPostTurnClosureV1": HarnessPostTurnClosureV1,
    "HarnessRepairOverlayV1": HarnessRepairOverlayV1,
    "HarnessRunRequestV1": HarnessRunRequestV1,
    "HarnessRunCancelV1": HarnessRunCancelV1,
    "HarnessRunStepV1": HarnessRunStepV1,
    "HarnessRunV1": HarnessRunV1,
    "EmbodimentIntentV1": EmbodimentIntentV1,
    "EmbodimentOutcomeV1": EmbodimentOutcomeV1,
    "WorldPerceptionV1": WorldPerceptionV1,
    "OrionTownPersonaV1": OrionTownPersonaV1,

}

# Incremental kind lookup for new schemas; runtime validation still uses resolve() / _REGISTRY.
SCHEMA_REGISTRY: Dict[str, SchemaRegistration] = {
    # orion-affectgpt-worker / orion-juniper-affective-state (2026-08-22).
    # AffectGptAssessRequestPayload is not registered here -- its envelope
    # kind is set literally by the producer, same as VisionTaskRequestPayload
    # above. Registered in BOTH this dict and `_REGISTRY` -- see the note
    # immediately below.
    # Power intent and settlement (2026-08-28, design doc
    # 2026-08-28-consequential-action-space-and-power-budget-design.md stage 2).
    # Registered in BOTH this dict and `_REGISTRY` above -- they are separate maps
    # and a schema present in only one is half-registered.
    "PowerIntentV1": SchemaRegistration(
        model=PowerIntentV1,
        kind="power.intent.v1",
    ),
    "PowerIntentSettledV1": SchemaRegistration(
        model=PowerIntentSettledV1,
        kind="power.intent.settled.v1",
    ),
    "CabinetAmbientSpikeV1": SchemaRegistration(
        model=CabinetAmbientSpikeV1,
        kind="cabinet.ambient.spike.v1",
    ),
    "AffectGptAssessResultPayload": SchemaRegistration(
        model=AffectGptAssessResultPayload,
        kind="affectgpt.assess.result",
    ),
    "JuniperMultimodalAffectV1": SchemaRegistration(
        model=JuniperMultimodalAffectV1,
        kind="affectgpt.juniper_multimodal_affect.v1",
    ),
    # orion-vision-retina bus-reachable clip capture (2026-08-22). Request
    # kind is set literally by the producer (orion-juniper-affective-state),
    # same pattern as AffectGptAssessRequestPayload above.
    "RetinaClipCaptureResultPayload": SchemaRegistration(
        model=RetinaClipCaptureResultPayload,
        kind="retina.clip_capture.result",
    ),
    # SubstrateReadService (2026-08-12, design doc
    # 2026-08-12-substrate-action-perception-design.md option B(2)). Registered
    # in BOTH this dict and `_REGISTRY` above -- they are separate maps and a
    # schema present in only one is half-registered.
    # Claude as a third social-room participant (2026-08-14, design doc
    # hub-social-room-claude-companion.md). Only these two carry a message
    # kind; RoomTranscriptEntryV1 and ExternalRoomResponderV1 are nested
    # payload models, so they live in `_REGISTRY` alone by design.
    "RoomClaudeRequestV1": SchemaRegistration(
        model=RoomClaudeRequestV1,
        kind="room.claude.request.v1",
    ),
    "RoomClaudeUtteranceV1": SchemaRegistration(
        model=RoomClaudeUtteranceV1,
        kind="room.claude.utterance.v1",
    ),
    "SubstrateReadQueryV1": SchemaRegistration(
        model=SubstrateReadQueryV1,
        kind="substrate_read.query.v1",
    ),
    "SubstrateReadReplyV1": SchemaRegistration(
        model=SubstrateReadReplyV1,
        kind="substrate_read.reply.v1",
    ),
    "InnerStateFeaturesV1": SchemaRegistration(
        model=InnerStateFeaturesV1,
        kind="self.inner_features.v1",
    ),
    "MoodArcCorpusRowV1": SchemaRegistration(
        model=MoodArcCorpusRowV1,
        kind="self.mood_arc_corpus.v1",
    ),
    "MoodArcEncoderManifestV1": SchemaRegistration(
        model=MoodArcEncoderManifestV1,
        kind="self.mood_arc_encoder.manifest.v1",
    ),
    "PhiEncoderManifestV1": SchemaRegistration(
        model=PhiEncoderManifestV1,
        kind="self.phi_encoder.manifest.v1",
    ),
    "ReasoningCallV1": SchemaRegistration(
        model=ReasoningCallV1,
        kind="cognition.reasoning_call.v1",
    ),
    "ReasoningActivityV1": SchemaRegistration(
        model=ReasoningActivityV1,
        kind="cognition.reasoning_activity.v1",
    ),
    "VisionEdgeActivityPayload": SchemaRegistration(
        model=VisionEdgeActivityPayload,
        kind="vision.edge.activity.v1",
    ),
    "VisionSceneInventoryV1": SchemaRegistration(
        model=VisionSceneInventoryV1,
        kind="vision.scene.inventory.v1",
    ),
    "CoalitionSnapshotV1": SchemaRegistration(
        model=CoalitionSnapshotV1,
        kind="coalition.snapshot.v1",
    ),
    "StanceHarnessSliceV1": SchemaRegistration(
        model=StanceHarnessSliceV1,
        kind="stance.harness.slice.v1",
    ),
    "HubAssociationBundleV1": SchemaRegistration(
        model=HubAssociationBundleV1,
        kind="hub.association.bundle.v1",
    ),
    "ThoughtEventV1": SchemaRegistration(
        model=ThoughtEventV1,
        kind="thought.event.v1",
    ),
    "ThoughtDecisionRecordV1": SchemaRegistration(
        model=ThoughtDecisionRecordV1,
        kind="thought.decision.record.v1",
    ),
    "GroundingCapsuleV1": SchemaRegistration(
        model=GroundingCapsuleV1,
        kind="grounding.capsule.v1",
    ),
    "SpontaneousThoughtV1": SchemaRegistration(
        model=SpontaneousThoughtV1,
        kind="reverie.thought.v1",
    ),
    "ReverieChainV1": SchemaRegistration(
        model=ReverieChainV1,
        kind="reverie.chain.v1",
    ),
    "ReverieRefractoryEntry": SchemaRegistration(
        model=ReverieRefractoryEntry,
        kind="reverie.refractory.entry.v1",
    ),
    "CompactionRequestV1": SchemaRegistration(
        model=CompactionRequestV1,
        kind="dream.compaction.request.v1",
    ),
    "MemoryCompactionDeltaV1": SchemaRegistration(
        model=MemoryCompactionDeltaV1,
        kind="dream.compaction.delta.v1",
    ),
    "ReverieVisualChainV1": SchemaRegistration(
        model=ReverieVisualChainV1,
        kind="reverie.visual.chain.v1",
    ),
    "ReverieVisualArtifactV1": SchemaRegistration(
        model=ReverieVisualArtifactV1,
        kind="reverie.visual.artifact.v1",
    ),
    "ResonanceAlertV1": SchemaRegistration(
        model=ResonanceAlertV1,
        kind="reverie.resonance.alert.v1",
    ),
    "StanceReactRequestV1": SchemaRegistration(
        model=StanceReactRequestV1,
        kind="stance.react.request.v1",
    ),
    "GrammarReceiptV1": SchemaRegistration(
        model=GrammarReceiptV1,
        kind="grammar.receipt.v1",
    ),
    "HarnessDraftMoleculeV1": SchemaRegistration(
        model=HarnessDraftMoleculeV1,
        kind="harness.draft.molecule.v1",
    ),
    "SubstrateFinalizeAppraisalV1": SchemaRegistration(
        model=SubstrateFinalizeAppraisalV1,
        kind="substrate.finalize.appraisal.v1",
    ),
    "FinalizeReflectionV1": SchemaRegistration(
        model=FinalizeReflectionV1,
        kind="finalize.reflection.v1",
    ),
    "HarnessVerdictMoleculeV1": SchemaRegistration(
        model=HarnessVerdictMoleculeV1,
        kind="harness.verdict.molecule.v1",
    ),
    "HarnessTurnOutcomeMoleculeV1": SchemaRegistration(
        model=HarnessTurnOutcomeMoleculeV1,
        kind="harness.turn.outcome.v1",
    ),
    "HarnessPostTurnClosureV1": SchemaRegistration(
        model=HarnessPostTurnClosureV1,
        kind="harness.post_turn.closure.v1",
    ),
    "HarnessRepairOverlayV1": SchemaRegistration(
        model=HarnessRepairOverlayV1,
        kind="harness.repair.overlay.v1",
    ),
    "HarnessRunRequestV1": SchemaRegistration(
        model=HarnessRunRequestV1,
        kind="harness.run.request.v1",
    ),
    "HarnessRunCancelV1": SchemaRegistration(
        model=HarnessRunCancelV1,
        kind="harness.run.cancel.v1",
    ),
    "HarnessRunStepV1": SchemaRegistration(
        model=HarnessRunStepV1,
        kind="harness.run.step.v1",
    ),
    "HarnessRunV1": SchemaRegistration(
        model=HarnessRunV1,
        kind="harness.run.v1",
    ),
    "ActionOutcomeEmitV1": SchemaRegistration(
        model=ActionOutcomeEmitV1,
        kind="action.outcome.emit.v1",
    ),
    "EmbodimentIntentV1": SchemaRegistration(
        model=EmbodimentIntentV1, kind="embodiment.intent.v1"
    ),
    "EmbodimentOutcomeV1": SchemaRegistration(
        model=EmbodimentOutcomeV1, kind="embodiment.outcome.v1"
    ),
    "WorldPerceptionV1": SchemaRegistration(
        model=WorldPerceptionV1, kind="embodiment.perception.v1"
    ),
    "OrionTownPersonaV1": SchemaRegistration(
        model=OrionTownPersonaV1, kind="embodiment.persona.v1"
    ),
    "AttentionSalienceTraceV1": SchemaRegistration(
        model=AttentionSalienceTraceV1,
        kind="attention.salience.trace.v1",
    ),
    "AttentionLoopOutcomeV1": SchemaRegistration(
        model=AttentionLoopOutcomeV1,
        kind="attention.loop.outcome.v1",
    ),
    "PendingAttentionCardV1": SchemaRegistration(
        model=PendingAttentionCardV1,
        kind="attention.pending.card.v1",
    ),
    "SubstrateBrainFrameV1": SchemaRegistration(
        model=SubstrateBrainFrameV1,
        kind="substrate.brain_frame.v1",
    ),
    "CausalGeometrySnapshotV1": SchemaRegistration(
        model=CausalGeometrySnapshotV1,
        kind="causal.geometry.snapshot.v1",
    ),
    "GraphWriteIntentV1": SchemaRegistration(
        model=GraphWriteIntentV1,
        kind="graph.write_intent.v1",
    ),
    "AttentionBroadcastProjectionV1": SchemaRegistration(
        model=AttentionBroadcastProjectionV1,
        kind="attention.broadcast.projection.v1",
    ),
    "EquilibriumServiceTransitionV1": SchemaRegistration(
        model=EquilibriumServiceTransitionV1,
        kind="equilibrium.service.transition.v1",
    ),
    # AST/HOT reducer output (Phase 1, docs/superpowers/specs/
    # 2026-07-18-objective-3-consciousness-scaffolded-roadmap-design.md).
    # Registered per CLAUDE.md sec 6 even though this artifact is not yet
    # published to any bus channel -- read-only measurement instrument only,
    # not wired into a live consumer/producer path. See
    # orion/substrate/attention_self_model.py and
    # scripts/analysis/measure_ast_hot_reducer.py.
    "AttentionSelfModelV1": SchemaRegistration(
        model=AttentionSelfModelV1,
        kind="attention.self_model.v1",
    ),
    "SelfStudyEnrichmentRequestV1": SchemaRegistration(
        model=SelfStudyEnrichmentRequestV1,
        kind="self_study.enrichment.request.v1",
    ),
    # orion-world-model (2026-08-20, scaffold patch -- see module docstring in
    # orion/schemas/world_model.py). Only the request/prediction pair carries
    # a message kind; WorldModelFeatureGroupV1/WorldModelTrajectoryStepV1 are
    # nested payload models, so they live in `_REGISTRY` alone by design
    # (same split as VisionArtifactOutputs/VisionObject etc. in vision.py).
    "WorldModelTaskRequestPayload": SchemaRegistration(
        model=WorldModelTaskRequestPayload,
        kind="world_model.task.request",
    ),
    "WorldModelPredictionPayload": SchemaRegistration(
        model=WorldModelPredictionPayload,
        kind="world_model.prediction",
    ),
}


def resolve(schema_id: str) -> Type[BaseModel]:
    try:
        return _REGISTRY[schema_id]
    except KeyError as exc:
        raise ValueError(f"Unknown schema_id: {schema_id}") from exc
