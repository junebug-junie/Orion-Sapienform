"""Fixed GPU slot HTTP contracts. No caller-controlled Docker parameters."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class GpuSlotRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slot: Literal["circe-gpu1", "circe-gpu2"]
    target: Literal["affect", "agent", "diffusion", "agent-burst"]
    operation_id: str = Field(min_length=1, max_length=128)
    generation: int = Field(ge=1)

    @model_validator(mode="after")
    def known_pair(self):
        if self.target not in {"circe-gpu1": {"affect", "agent"},
                               "circe-gpu2": {"diffusion", "agent-burst"}}[self.slot]:
            raise ValueError("cross_slot_target")
        return self
