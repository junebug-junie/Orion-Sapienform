from datetime import datetime, date
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class DreamFragmentMeta(BaseModel):
    id: str
    kind: str
    salience: float
    tags: List[str] = []

class DreamMetrics(BaseModel):
    gpu_w: Optional[float] = None
    util: Optional[float] = None
    mem_mb: Optional[float] = None
    cpu_c: Optional[float] = None

class DreamInput(BaseModel):
    tldr: Optional[str] = None
    themes: Optional[List[str]] = []
    symbols: Optional[Dict[str, str]] = {}
    narrative: Optional[str] = None
    fragments: Optional[List[DreamFragmentMeta]] = []
    metrics: Optional[DreamMetrics] = Field(default=None)
    dream_date: Optional[date] = None
    created_at: Optional[datetime] = None

    def normalize(self) -> "DreamInput":
        if not self.dream_date:
            self.dream_date = date.today()
        if not self.created_at:
            self.created_at = datetime.utcnow()
        return self
