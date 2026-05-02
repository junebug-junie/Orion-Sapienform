"""
All signal adapters, exported from one place.
``ADAPTERS`` is the live registry used by the gateway.
"""
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
from .agent_chain import AgentChainAdapter
from .planner import PlannerAdapter
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
    AgentChainAdapter(),
    PlannerAdapter(),
    DreamAdapter(),
    StateJournalerAdapter(),
    TopicFoundryAdapter(),
    ConceptInductionAdapter(),
    GraphCognitionAdapter(),
    ChatStanceAdapter(),
    JournalerAdapter(),
    PowerGuardAdapter(),
    SecurityWatcherAdapter(),
]

__all__ = [
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
    "AgentChainAdapter",
    "PlannerAdapter",
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
