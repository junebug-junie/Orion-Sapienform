from datetime import datetime
from typing import Optional
from pydantic import BaseModel

class ChatHistoryInput(BaseModel):
    id: Optional[str] = None
    trace_id: str
    source: str
    prompt: Optional[str] = None
    response: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    created_at: Optional[datetime] = None

    def normalize(self) -> "ChatHistoryInput":
        self.id = self.trace_id
        self.created_at = datetime.utcnow()
        return self
