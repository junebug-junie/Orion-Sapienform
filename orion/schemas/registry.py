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
from orion.schemas.collapse_mirror import CollapseMirrorEntryV2
from orion.schemas.cortex.contracts import (
    CortexClientRequest,
    CortexClientResult,
    CortexChatRequest,
    CortexChatResult,
    RecallDirective,
)
from orion.schemas.cortex.exec import CortexExecRequestPayload, CortexExecResultPayload
from orion.schemas.cortex.schemas import PlanExecutionRequest, PlanExecutionResult
from orion.schemas.platform import CoreEventV1, GenericPayloadV1, SystemErrorV1
from orion.schemas.chat_history import ChatHistoryMessageV1, ChatHistoryTurnV1  # includes memory policy fields
from orion.schemas.vector.schemas import (
    EmbeddingGenerateV1,
    EmbeddingResultV1,
    VectorDocumentUpsertV1,
    VectorUpsertV1,
    VectorWriteRequest,
)
from orion.core.schemas.concept_induction import ConceptProfile, ConceptProfileDelta
from orion.schemas.telemetry.biometrics import (
    BiometricsPayload,
    BiometricsSampleV1,
    BiometricsSummaryV1,
    BiometricsInductionV1,
    BiometricsClusterV1,
)
from orion.schemas.pad.v1 import PadEventV1, StateFrameV1, PadRpcRequestV1, PadRpcResponseV1
from orion.schemas.rdf import RdfBuildRequest, RdfWriteRequest, RdfWriteResult
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
from orion.schemas.topic_foundry import (
    KgEdgeIngestV1,
    TopicFoundryDriftAlertV1,
    TopicFoundryEnrichCompleteV1,
    TopicFoundryRunCompleteV1,
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
    "CortexExecRequestPayload": CortexExecRequestPayload,
    "CortexExecResultPayload": CortexExecResultPayload,
    "PlanExecutionRequest": PlanExecutionRequest,
    "PlanExecutionResult": PlanExecutionResult,
    "CollapseMirrorEntryV2": CollapseMirrorEntryV2,  # change_type dict coercion support
    "CognitionTracePayload": CognitionTracePayload,
    "MetacognitionTickV1": MetacognitionTickV1,
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
    "ChatHistoryMessageV1": ChatHistoryMessageV1,  # includes memory policy + client_meta fields
    "ChatHistoryTurnV1": ChatHistoryTurnV1,  # includes memory policy + client_meta fields
    "VectorWriteRequest": VectorWriteRequest,
    "VectorDocumentUpsertV1": VectorDocumentUpsertV1,
    "VectorUpsertV1": VectorUpsertV1,
    "EmbeddingGenerateV1": EmbeddingGenerateV1,
    "EmbeddingResultV1": EmbeddingResultV1,
    "ConceptProfile": ConceptProfile,
    "ConceptProfileDelta": ConceptProfileDelta,
    "BiometricsPayload": BiometricsPayload,
    "BiometricsSampleV1": BiometricsSampleV1,
    "BiometricsSummaryV1": BiometricsSummaryV1,
    "BiometricsInductionV1": BiometricsInductionV1,
    "BiometricsClusterV1": BiometricsClusterV1,
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
    "DailyMetacogV1": DailyMetacogV1,
    "TopicSummaryEventV1": TopicSummaryEventV1,
    "TopicShiftEventV1": TopicShiftEventV1,
    "TopicRailAssignedV1": TopicRailAssignedV1,
    "TopicFoundryRunCompleteV1": TopicFoundryRunCompleteV1,
    "TopicFoundryEnrichCompleteV1": TopicFoundryEnrichCompleteV1,
    "TopicFoundryDriftAlertV1": TopicFoundryDriftAlertV1,
    "KgEdgeIngestV1": KgEdgeIngestV1,

}


def resolve(schema_id: str) -> Type[BaseModel]:
    try:
        return _REGISTRY[schema_id]
    except KeyError as exc:
        raise ValueError(f"Unknown schema_id: {schema_id}") from exc
