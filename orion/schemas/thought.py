from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1
from orion.schemas.pre_turn_appraisal import TurnAppraisalBundleV1


class HubAssociationBundleV1(BaseModel):
    schema_version: Literal["hub.association.bundle.v1"] = "hub.association.bundle.v1"
    correlation_id: str
    broadcast: AttentionBroadcastProjectionV1 | None
    broadcast_stale: bool
    execution_trajectory_slice: dict[str, Any] | None = None
    repair_bundle: TurnAppraisalBundleV1 | None = None
    read_source: Literal["felt_state_reader", "hub_sql_fallback"]
