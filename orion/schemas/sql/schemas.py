from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any
from datetime import datetime, timezone

class SqlWriteRequest(BaseModel):
    """
    Standard request to write data to the SQL database.
    """
    model_config = ConfigDict(extra="ignore")

    table_name: str
    data: Dict[str, Any]
    id: Optional[str] = None # PK if known
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
