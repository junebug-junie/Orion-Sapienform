from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class PowerStatus(BaseModel):
    """
    Parsed snapshot of UPS state from SNMP.
    """

    raw_status: str = Field(..., description="High-level status, e.g. ONLINE or ONBATT")
    on_battery: bool = Field(..., description="True if UPS reports on-battery state")
    line_voltage: Optional[float] = None
    battery_charge_pct: Optional[float] = None
    time_left_min: Optional[float] = None
    raw: Dict[str, Any] = Field(default_factory=dict)


class PowerEvent(BaseModel):
    """
    Payload published to the Orion bus when power state changes.
    """

    kind: str = Field(..., description="e.g. power.guard.on_battery, power.guard.grace_elapsed, power.guard.restored")
    node: str
    ups_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: PowerStatus
    details: Dict[str, Any] = Field(default_factory=dict)
