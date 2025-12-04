# In services/orion-brain/app/models.py
from typing import Any, List, Dict, Optional
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
    messages: List[Dict]
    options: Optional[dict] = None
    stream: Optional[bool] = False
    return_json: Optional[bool] = False
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source: Optional[str] = None

class CortexExecRequest(BaseModel):
    event: str
    verb: str
    step: str
    order: int
    requires_gpu: bool
    requires_memory: bool
    prompt_template: Optional[str]
    args: Dict[str, Any]
    context: Dict[str, Any]
    correlation_id: str
    reply_channel: str
    origin_node: str
    service: str


# ============================================================
# Execution Envelope (Cortex ↔ Brain standard message shape)
# ============================================================
from pydantic import BaseModel
from typing import Dict, Any, Optional

class ExecutionEnvelope(BaseModel):
    event: str                       # e.g. "exec_step", "chat", "tts"
    service: str                     # e.g. "LLMGatewayService"
    correlation_id: str
    reply_channel: str
    payload: Dict[str, Any]          # the verb/step/args/history/etc
