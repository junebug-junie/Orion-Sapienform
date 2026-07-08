"""
All signal adapters, exported from one place.
``ADAPTERS`` is the live registry used by the gateway.
"""
from .cognition_trace import CognitionTraceAdapter
from .cortex_gateway import CortexGatewayAdapter
from .cortex_orch import CortexOrchAdapter
from .hub import HubAdapter
from .persistence_writers import RdfWriterAdapter, SqlWriterAdapter, VectorWriterAdapter
from .biometrics import BiometricsAdapter
from .collapse_mirror import CollapseMirrorAdapter
from .equilibrium import EquilibriumAdapter
from .recall import RecallAdapter
from .spark import SparkAdapter
from .autonomy import AutonomyAdapter
from .world_pulse import WorldPulseAdapter
from .social_memory import SocialMemoryAdapter
from .social_room_bridge import SocialRoomBridgeAdapter
from .vision import VisionAdapter
from .dream import DreamAdapter
from .state_journaler import StateJournalerAdapter
from .topic_foundry import TopicFoundryAdapter
from .concept_induction import ConceptInductionAdapter
from .graph_cognition import GraphCognitionAdapter
from .chat_stance import ChatStanceAdapter
from .journaler import JournalerAdapter
from .power_guard import PowerGuardAdapter
from .security_watcher import SecurityWatcherAdapter

from orion.signals.adapters.base import OrionSignalAdapter
from typing import List

ADAPTERS: List[OrionSignalAdapter] = [
    CognitionTraceAdapter(),
    CortexGatewayAdapter(),
    CortexOrchAdapter(),
    HubAdapter(),
    BiometricsAdapter(),
    CollapseMirrorAdapter(),
    EquilibriumAdapter(),
    RecallAdapter(),
    SparkAdapter(),
    AutonomyAdapter(),
    WorldPulseAdapter(),
    SocialMemoryAdapter(),
    SocialRoomBridgeAdapter(),
    VisionAdapter(),
    DreamAdapter(),
    StateJournalerAdapter(),
    TopicFoundryAdapter(),
    ConceptInductionAdapter(),
    GraphCognitionAdapter(),
    ChatStanceAdapter(),
    JournalerAdapter(),
    PowerGuardAdapter(),
    SecurityWatcherAdapter(),
    SqlWriterAdapter(),
    RdfWriterAdapter(),
    VectorWriterAdapter(),
]

__all__ = [
    "CognitionTraceAdapter",
    "CortexGatewayAdapter",
    "CortexOrchAdapter",
    "HubAdapter",
    "SqlWriterAdapter",
    "RdfWriterAdapter",
    "VectorWriterAdapter",
    "BiometricsAdapter",
    "CollapseMirrorAdapter",
    "EquilibriumAdapter",
    "RecallAdapter",
    "SparkAdapter",
    "AutonomyAdapter",
    "WorldPulseAdapter",
    "SocialMemoryAdapter",
    "SocialRoomBridgeAdapter",
    "VisionAdapter",
    "DreamAdapter",
    "StateJournalerAdapter",
    "TopicFoundryAdapter",
    "ConceptInductionAdapter",
    "GraphCognitionAdapter",
    "ChatStanceAdapter",
    "JournalerAdapter",
    "PowerGuardAdapter",
    "SecurityWatcherAdapter",
    "ADAPTERS",
]
