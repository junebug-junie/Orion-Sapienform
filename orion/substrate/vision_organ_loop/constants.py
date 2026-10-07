from orion.schemas.vision_organ_projection import (
    VISION_ORGAN_SOURCE_SERVICE,
    VISION_ORGAN_TRACE_PREFIX,
)

VISION_ORGAN_PROJECTION_ID = "active_vision_organ_projection"
VISION_ORGAN_GRAMMAR_CURSOR_NAME = "vision_organ_grammar_reducer"
VISION_ORGAN_REDUCER_ID = "vision_organ_reducer"
VISION_ORGAN_TARGET_KIND = "vision_organ"
# Off-lattice field node the organ reading lands on (same shape as
# node:substrate.rpc_delivery). Successor of node:substrate.vision, which the
# substrate runtime's artifact-listener tick wrote until this lane replaced it.
VISION_ORGAN_NODE_ID = "node:substrate.vision_organ"

__all__ = [
    "VISION_ORGAN_GRAMMAR_CURSOR_NAME",
    "VISION_ORGAN_NODE_ID",
    "VISION_ORGAN_PROJECTION_ID",
    "VISION_ORGAN_REDUCER_ID",
    "VISION_ORGAN_SOURCE_SERVICE",
    "VISION_ORGAN_TARGET_KIND",
    "VISION_ORGAN_TRACE_PREFIX",
]
