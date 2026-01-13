# services/orion-cortex-exec/app/service_registry.py

SERVICE_BINDINGS = {
    # --- Semantic Aliases ---
    "llm.brain":           "LLMGatewayService",
    "memory.vector":       "VectorMemoryService",
    "memory.sql":          "SqlMemoryService",
    "memory.collapse":     "CollapseMirrorService",

    # --- Vision Services ---
    "vision.host":         "VisionHostService",
    "vision.window":       "VisionWindowService",
    "vision.council":      "VisionCouncilService",
    "vision.scribe":       "VisionScribeService",

    # --- Metacognition (Identity Mappings) ---
    # Required because log_orion_metacognition.yaml refers to them by name
    "MetacogContextService": "MetacogContextService",
    "MetacogDraftService":   "MetacogDraftService",
    "MetacogEnrichService":  "MetacogEnrichService",
    "MetacogPublishService": "MetacogPublishService",
    "MetaTagsService":       "MetaTagsService",

    # --- Core Execution Services (Identity Mappings) ---
    # Required for other verbs that use direct service names
    "RecallService":         "RecallService",
    "AgentChainService":     "AgentChainService",
    "PlannerReactService":   "PlannerReactService",
    "CouncilService":        "CouncilService",
    "LLMGatewayService":     "LLMGatewayService",
    "VerbRequestService":    "VerbRequestService",
}


def resolve_service(alias: str) -> str:
    """
    Map a semantic alias in the ontology (llm.brain, memory.vector, etc.)
    to the concrete service suffix used on the bus.
    """
    try:
        return SERVICE_BINDINGS[alias]
    except KeyError:
        # Fallback: If the alias looks like a real service name (PascalCase), allow it.
        # This prevents crashing on new services not yet in the map.
        if alias.endswith("Service"):
            return alias
        raise ValueError(f"Unknown service alias: {alias!r}")
