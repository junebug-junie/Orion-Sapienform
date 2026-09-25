from orion.schemas.llm_inference_projection import (
    LLM_INFERENCE_SOURCE_SERVICE,
    LLM_INFERENCE_TRACE_PREFIX,
)

LLM_INFERENCE_PROJECTION_ID = "active_llm_inference_projection"
LLM_INFERENCE_GRAMMAR_CURSOR_NAME = "llm_inference_grammar_reducer"
LLM_INFERENCE_REDUCER_ID = "llm_inference_reducer"
LLM_INFERENCE_TARGET_KIND = "llm_inference_node"

# Field-topology node ids a served_by label may resolve to. Same convention and
# same set as services/orion-cortex-exec/app/executor.py::_KNOWN_FIELD_NODES
# (served_by labels are "{node}-worker[-lane][-N]"). The field digester creates
# whatever node id it is handed, so an unknown label must stay unattributed
# rather than mint a phantom node:circe-worker-2.
KNOWN_FIELD_NODES = frozenset({"athena", "circe", "prometheus"})

__all__ = [
    "KNOWN_FIELD_NODES",
    "LLM_INFERENCE_GRAMMAR_CURSOR_NAME",
    "LLM_INFERENCE_PROJECTION_ID",
    "LLM_INFERENCE_REDUCER_ID",
    "LLM_INFERENCE_SOURCE_SERVICE",
    "LLM_INFERENCE_TARGET_KIND",
    "LLM_INFERENCE_TRACE_PREFIX",
]
