# services/orion-cortex-exec/app/service_registry.py

SERVICE_BINDINGS = {
    # semantic alias       # bus suffix / service name
    "llm.brain":           "LLMGatewayService",
    "memory.vector":       "VectorMemoryService",
    "memory.sql":          "SqlMemoryService",
    # etc…
}


def resolve_service(alias: str) -> str:
    """
    Map a semantic alias in the ontology (llm.brain, memory.vector, etc.)
    to the concrete service suffix used on the bus.
    """
    try:
        return SERVICE_BINDINGS[alias]
    except KeyError:
        raise ValueError(f"Unknown service alias: {alias!r}")
