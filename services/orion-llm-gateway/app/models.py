from typing import Any, Dict, Optional, List, Literal
from pydantic import BaseModel, Field, ConfigDict


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class GenerateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Model can be omitted; we fall back to settings.default_model
    model: Optional[str] = None
    prompt: str
    options: Dict[str, Any] = Field(default_factory=dict)
    stream: bool = False
    return_json: bool = False
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source: Optional[str] = None

    # Semantic routing support
    verb: Optional[str] = None
    profile_name: Optional[str] = None


class ChatBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Model can be omitted; we fall back to settings.default_model
    model: Optional[str] = None

    # Messages can start empty; gateway can adapt prompt → messages
    messages: List[ChatMessage] = Field(default_factory=list)
    raw_user_text: Optional[str] = Field(
        default=None,
        description="Canonical raw user text supplied by upstream services (pre-scaffold).",
    )

    options: Dict[str, Any] = Field(default_factory=dict)
    stream: bool = False
    return_json: bool = False
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source: Optional[str] = None

    # Semantic routing support
    verb: Optional[str] = None
    profile_name: Optional[str] = None


class ExecStepPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
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

    context: Dict[str, Any] = Field(default_factory=dict)
    args: Dict[str, Any] = Field(default_factory=dict)
    prior_step_results: List[Dict[str, Any]] = Field(default_factory=list)
    raw_user_text: Optional[str] = Field(
        default=None,
        description="Canonical raw user message associated with this exec step.",
    )

    requires_gpu: bool = False
    requires_memory: bool = False
    profile_name: Optional[str] = None


class EmbeddingsBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    """
    Request body for embeddings generation via LLM Gateway.
    """
    model: Optional[str] = None
    input: List[str]
    options: Dict[str, Any] = Field(default_factory=dict)
    trace_id: Optional[str] = None
    source: Optional[str] = None

    # Optional semantic routing (in case we want profile-based embeddings)
    verb: Optional[str] = None
    profile_name: Optional[str] = None


class ExecutionEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")
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
