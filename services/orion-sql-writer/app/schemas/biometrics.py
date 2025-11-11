from pydantic import BaseModel
from typing import Optional, Dict, Any

class BiometricsInput(BaseModel):
    timestamp: str
    gpu: Optional[Dict[str, Any]] = None
    cpu: Optional[Dict[str, Any]] = None
    node: Optional[str] = None
    service_name: Optional[str] = None
    service_version: Optional[str] = None
