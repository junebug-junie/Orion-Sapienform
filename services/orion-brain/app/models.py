from typing import List, Dict, Optional
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

class ChatBody(BaseModel):
    model: str
    messages: List[Dict]
    options: Optional[dict] = None
    stream: Optional[bool] = False
    return_json: Optional[bool] = False
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
