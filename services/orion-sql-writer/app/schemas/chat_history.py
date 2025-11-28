from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel

class ChatHistoryInput(BaseModel):
    id: Optional[str] = None
    trace_id: str
    source: str
    prompt: Optional[str] = None
    response: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    spark_meta: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

    def normalize(self) -> "ChatHistoryInput":

        if self.id is None:
            self.id = self.trace_id
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        return self
