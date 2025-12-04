# services/orion-llm-gateway/app/models.py

from typing import Any, Dict, Optional, List
from pydantic import BaseModel


class GenerateBody(BaseModel):
    model: str
    prompt: str
    options: Optional[dict] = None
    stream: Optional[bool] = False
    return_json: Optional[bool] = False
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source: Optional[str] = None


class ChatBody(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    options: Optional[dict] = None
    stream: Optional[bool] = False
    return_json: Optional[bool] = False
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source: Optional[str] = None


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

    requires_gpu: bool = False
    requires_memory: bool = False


class ExecutionEnvelope(BaseModel):
    """
    Standard message shape on the bus for LLM gateway.

    event: "chat" | "generate" | "exec_step" | ...
    service: "LLMGatewayService"
    correlation_id: correlation / trace id
    reply_channel: bus channel to send result to
    payload: event-specific body (ChatBody / GenerateBody / ExecStepPayload)
    """
    event: str
    service: str
    correlation_id: str
    reply_channel: str
    payload: Dict[str, Any]
