from __future__ import annotations

from typing import Dict, Type

from pydantic import BaseModel

from orion.core.bus.bus_schemas import (
    ChatRequestPayload,
    ChatResultPayload,
    RecallRequestPayload,
    RecallResultPayload,
)
from orion.core.contracts.recall import RecallDecisionV1, RecallReplyV1, RecallQueryV1
from orion.core.verbs.models import VerbEffectV1, VerbRequestV1, VerbResultV1
from orion.schemas.actions.daily import DailyMetacogV1, DailyPulseV1
from orion.journaler.schemas import JournalEntryDraftV1, JournalEntryWriteV1, JournalTriggerV1
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
from orion.schemas.cortex.schemas import PlanExecutionRequest, PlanExecutionResult
from orion.schemas.platform import CoreEventV1, GenericPayloadV1, SystemErrorV1
from orion.schemas.chat_history import ChatHistoryMessageV1, ChatHistoryTurnV1  # includes memory policy fields
from orion.schemas.metacognitive_trace import MetacognitiveTraceV1
from orion.schemas.chat_gpt_log import ChatGptLogTurnV1, ChatGptMessageV1
from orion.schemas.social_chat import (
    SocialConceptEvidenceV1,
    SocialGroundingStateV1,
    SocialRedactionScoreV1,
    SocialRoomTurnStoredV1,
    SocialRoomTurnV1,
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
from orion.core.schemas.drives import (
    DriveAuditV1,
    DriveStateV1,
    GoalProposalV1,
    IdentitySnapshotV1,
    TensionEventV1,
    TurnDossierV1,
)
from orion.schemas.telemetry.biometrics import (
    BiometricsPayload,
    BiometricsSampleV1,
    BiometricsSummaryV1,
    BiometricsInductionV1,
    BiometricsClusterV1,
)
from orion.schemas.telemetry.dream import (
    DreamInternalTriggerV1,
    DreamRequest,
    DreamResultV1,
    DreamTriggerPayload,
)
from orion.schemas.pad.v1 import PadEventV1, StateFrameV1, PadRpcRequestV1, PadRpcResponseV1
from orion.schemas.rdf import RdfBuildRequest, RdfWriteRequest, RdfWriteResult
from orion.schemas.spark_concept_graph import SparkConceptProfileGraphMaterializationV1
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
from orion.schemas.telemetry.spark import SparkStateSnapshotV1, SparkTelemetryPayload
from orion.schemas.telemetry.spark_ack import SparkStateSnapshotAckV1
from orion.schemas.telemetry.spark_candidate import SparkCandidateV1
from orion.schemas.telemetry.spark_signal import SparkSignalV1
from orion.schemas.telemetry.system_health import EquilibriumSnapshotV1, SystemHealthV1
from orion.schemas.telemetry.cognition_trace import CognitionTracePayload
from orion.schemas.telemetry.metacognition import MetacognitionTickV1
from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1
from orion.schemas.telemetry.meta_tags import MetaTagsPayload, MetaTagsRequestV1, MetaTagsResultV1
from orion.schemas.metacog_patches import MetacogDraftTextPatchV1, MetacogEnrichScorePatchV1
from orion.schemas.state.contracts import StateGetLatestRequest, StateLatestReply
from orion.schemas.vision import (
    VisionArtifactPayload,
    VisionCouncilRequestPayload,
    VisionCouncilResultPayload,
    VisionEdgeArtifact,
    VisionEdgeError,
    VisionEdgeHealth,
    VisionEventPayload,
    VisionFramePointerPayload,
    VisionGuardAlert,
    VisionGuardSignal,
    VisionScribeRequestPayload,
    VisionScribeResultPayload,
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
from orion.schemas.topic import TopicSummaryEventV1, TopicShiftEventV1, TopicRailAssignedV1
from orion.schemas.chat_stance import ChatStanceBrief
from orion.schemas.topic_foundry import (
    KgEdgeIngestV1,
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
    "CortexClientRequest": CortexClientRequest,
    "CortexClientResult": CortexClientResult,
    "AgentTraceToolStatV1": AgentTraceToolStatV1,
    "AgentTraceStepV1": AgentTraceStepV1,
    "AgentTraceSummaryV1": AgentTraceSummaryV1,
    "CortexExecRequestPayload": CortexExecRequestPayload,
    "CortexExecResultPayload": CortexExecResultPayload,
    "PlanExecutionRequest": PlanExecutionRequest,
    "PlanExecutionResult": PlanExecutionResult,
    "CollapseMirrorEntryV2": CollapseMirrorEntryV2,  # change_type dict coercion support
    "CollapseMirrorStoredV1": CollapseMirrorStoredV1,
    "CognitionTracePayload": CognitionTracePayload,
    "MetacognitionTickV1": MetacognitionTickV1,
    "MetacognitiveTraceV1": MetacognitiveTraceV1,
    "MetacogTriggerV1": MetacogTriggerV1,
    "MetacogDraftTextPatchV1": MetacogDraftTextPatchV1,
    "MetacogEnrichScorePatchV1": MetacogEnrichScorePatchV1,
    "SparkCandidateV1": SparkCandidateV1,
    "SparkSignalV1": SparkSignalV1,
    "SparkStateSnapshotAckV1": SparkStateSnapshotAckV1,
    "SparkStateSnapshotV1": SparkStateSnapshotV1,
    "SparkTelemetryPayload": SparkTelemetryPayload,
    "SystemHealthV1": SystemHealthV1,
    "EquilibriumSnapshotV1": EquilibriumSnapshotV1,
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
    "VisionArtifactPayload": VisionArtifactPayload,
    "VisionEdgeArtifact": VisionEdgeArtifact,
    "VisionEdgeHealth": VisionEdgeHealth,
    "VisionEdgeError": VisionEdgeError,
    "VisionEventPayload": VisionEventPayload,
    "VisionWindowPayload": VisionWindowPayload,
    "VisionWindowRequestPayload": VisionWindowRequestPayload,
    "VisionWindowResultPayload": VisionWindowResultPayload,
    "VisionCouncilRequestPayload": VisionCouncilRequestPayload,
    "VisionCouncilResultPayload": VisionCouncilResultPayload,
    "VisionScribeRequestPayload": VisionScribeRequestPayload,
    "VisionScribeResultPayload": VisionScribeResultPayload,
    "VisionGuardSignal": VisionGuardSignal,
    "VisionGuardAlert": VisionGuardAlert,
    "CortexChatRequest": CortexChatRequest,
    "CortexChatResult": CortexChatResult,
    "RecallDirective": RecallDirective,
    "AutoRouteDecisionV1": AutoRouteDecisionV1,
    "AutoDepthDecisionV1": AutoDepthDecisionV1,
    "ChatStanceBrief": ChatStanceBrief,
    "ChatHistoryMessageV1": ChatHistoryMessageV1,  # includes memory policy + client_meta fields
    "ChatHistoryTurnV1": ChatHistoryTurnV1,  # includes memory policy + client_meta fields
    "ChatGptLogTurnV1": ChatGptLogTurnV1,
    "ChatGptMessageV1": ChatGptMessageV1,
    "SocialConceptEvidenceV1": SocialConceptEvidenceV1,
    "SocialGroundingStateV1": SocialGroundingStateV1,
    "SocialRedactionScoreV1": SocialRedactionScoreV1,
    "SocialRoomTurnV1": SocialRoomTurnV1,
    "SocialRoomTurnStoredV1": SocialRoomTurnStoredV1,
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
    "DriveStateV1": DriveStateV1,
    "DriveAuditV1": DriveAuditV1,
    "IdentitySnapshotV1": IdentitySnapshotV1,
    "GoalProposalV1": GoalProposalV1,
    "TensionEventV1": TensionEventV1,
    "TurnDossierV1": TurnDossierV1,
    "BiometricsPayload": BiometricsPayload,
    "BiometricsSampleV1": BiometricsSampleV1,
    "BiometricsSummaryV1": BiometricsSummaryV1,
    "BiometricsInductionV1": BiometricsInductionV1,
    "BiometricsClusterV1": BiometricsClusterV1,
    "DreamRequest": DreamRequest,
    "DreamTriggerPayload": DreamTriggerPayload,
    "DreamInternalTriggerV1": DreamInternalTriggerV1,
    "DreamResultV1": DreamResultV1,
    "PadEventV1": PadEventV1,
    "StateFrameV1": StateFrameV1,
    "PadRpcRequestV1": PadRpcRequestV1,
    "PadRpcResponseV1": PadRpcResponseV1,
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
    "JournalEntryWriteV1": JournalEntryWriteV1,
    "DailyMetacogV1": DailyMetacogV1,
    "TopicSummaryEventV1": TopicSummaryEventV1,
    "TopicShiftEventV1": TopicShiftEventV1,
    "TopicRailAssignedV1": TopicRailAssignedV1,
    "TopicFoundryRunCompleteV1": TopicFoundryRunCompleteV1,
    "TopicFoundryEnrichCompleteV1": TopicFoundryEnrichCompleteV1,
    "TopicFoundryDriftAlertV1": TopicFoundryDriftAlertV1,
    "KgEdgeIngestV1": KgEdgeIngestV1,
    "WorkflowScheduleSpecV1": WorkflowScheduleSpecV1,
    "WorkflowExecutionPolicyV1": WorkflowExecutionPolicyV1,
    "WorkflowDispatchRequestV1": WorkflowDispatchRequestV1,
    "WorkflowScheduleRecordV1": WorkflowScheduleRecordV1,
    "WorkflowScheduleAnalyticsV1": WorkflowScheduleAnalyticsV1,
    "WorkflowScheduleEventRecordV1": WorkflowScheduleEventRecordV1,
    "WorkflowScheduleRunRecordV1": WorkflowScheduleRunRecordV1,
    "WorkflowScheduleManageRequestV1": WorkflowScheduleManageRequestV1,
    "WorkflowScheduleManageResponseV1": WorkflowScheduleManageResponseV1,

}


def resolve(schema_id: str) -> Type[BaseModel]:
    try:
        return _REGISTRY[schema_id]
    except KeyError as exc:
        raise ValueError(f"Unknown schema_id: {schema_id}") from exc
