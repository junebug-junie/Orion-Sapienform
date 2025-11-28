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


class ExecutionEnvelope(BaseModel):
    """
    Standard message shape on the bus for LLM gateway.

    This is the same structure Brain uses, just with service="LLMGatewayService".
    """
    event: str              # "chat" | "generate" | ...
    service: str            # "LLMGatewayService"
    correlation_id: str
    reply_channel: str
    payload: Dict[str, Any]  # expects {"body": {...}}
