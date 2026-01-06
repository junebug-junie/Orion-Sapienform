from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field


class GenerateBody(BaseModel):
    # Model can be omitted; we fall back to settings.default_model
    model: Optional[str] = None
    prompt: str
    options: Optional[dict] = None
    stream: Optional[bool] = False
    return_json: Optional[bool] = False
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source: Optional[str] = None

    # Semantic routing support
    verb: Optional[str] = None
    profile_name: Optional[str] = None


class ChatBody(BaseModel):
    # Model can be omitted; we fall back to settings.default_model
    model: Optional[str] = None

    # Messages can start empty; gateway can adapt prompt → messages
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    raw_user_text: Optional[str] = Field(
        default=None,
        description="Canonical raw user text supplied by upstream services (pre-scaffold).",
    )

    options: Optional[dict] = None
    stream: Optional[bool] = False
    return_json: Optional[bool] = False
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source: Optional[str] = None

    # Semantic routing support
    verb: Optional[str] = None
    profile_name: Optional[str] = None


class ExecStepPayload(BaseModel):
    """
    Payload for a Cortex exec_step routed through the LLM Gateway.

    This mirrors what the Cortex Orchestrator sends today, plus the
    fully-built prompt.
    """
    verb: str
    step: str
    order: int
    service: str
    origin_node: str

    prompt: Optional[str] = None
    prompt_template: Optional[str] = None

    context: Dict[str, Any] = {}
    args: Dict[str, Any] = {}
    prior_step_results: List[Dict[str, Any]] = []
    raw_user_text: Optional[str] = Field(
        default=None,
        description="Canonical raw user message associated with this exec step.",
    )

    requires_gpu: bool = False
    requires_memory: bool = False


class EmbeddingsBody(BaseModel):
    """
    Request body for embeddings generation via LLM Gateway.
    """
    model: Optional[str] = None
    input: List[str]
    options: Optional[dict] = None
    trace_id: Optional[str] = None
    source: Optional[str] = None

    # Optional semantic routing (in case we want profile-based embeddings)
    verb: Optional[str] = None
    profile_name: Optional[str] = None


class ExecutionEnvelope(BaseModel):
    """
    Standard message shape on the bus for LLM gateway.

    event: "chat" | "generate" | "exec_step" | "embeddings" | ...
    service: "LLMGatewayService"
    correlation_id: correlation / trace id
    reply_channel: bus channel to send result to
    payload: event-specific body (ChatBody / GenerateBody / ExecStepPayload / EmbeddingsBody)
    """
    event: str
    service: str
    correlation_id: str
    reply_channel: str
    payload: Dict[str, Any]
