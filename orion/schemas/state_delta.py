from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class StateDeltaV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["substrate.state_delta.v1"] = "substrate.state_delta.v1"
    delta_id: str
    target_projection: str
    target_kind: str
    target_id: str
    operation: Literal[
        "create",
        "update",
        "reinforce",
        "decay",
        "merge",
        "suppress",
        "noop",
    ]
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    caused_by_event_ids: list[str]
    reducer_id: str
    explanation: str | None = None
