from __future__ import annotations

import datetime
import uuid
from typing import Any, Dict, List, Optional, TypeVar, Generic
from pydantic import BaseModel, Field, ConfigDict

T = TypeVar("T")

class CausalityNode(BaseModel):
    """
    A single node in the causality chain, tracking the path of the request.
    """
    service: str
    event: str
    timestamp: float
    trace_id: Optional[str] = None

class BaseEnvelope(BaseModel, Generic[T]):
    """
    The Titanium Contract: A strict envelope for all Orion bus messages.
    """
    model_config = ConfigDict(extra="ignore")

    # Routing
    event: str = Field(..., description="The semantic event name (e.g., 'llm.chat').")
    # RELAXED: source is now optional to support legacy callers (e.g. Cortex Orch)
    source: str = Field(default="unknown", description="The service name originating this envelope.")
    reply_channel: Optional[str] = Field(None, description="Where to send the reply (if RPC).")

    # Tracking
    correlation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Global ID for the request chain."
    )
    causality_chain: List[CausalityNode] = Field(
        default_factory=list,
        description="Ordered list of services that touched this request."
    )
    timestamp: float = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).timestamp(),
        description="UTC timestamp of when this envelope was created."
    )

    # Payload
    payload: T = Field(..., description="The typed payload.")

    def add_causality(self, service: str, event: str) -> None:
        """Appends the current service to the causality chain."""
        self.causality_chain.append(
            CausalityNode(
                service=service,
                event=event,
                timestamp=datetime.datetime.now(datetime.timezone.utc).timestamp(),
                trace_id=self.correlation_id,
            )
        )
