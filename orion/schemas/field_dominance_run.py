"""Completed, observed goal-provenance focus runs; SQL-only, no decision consumer."""
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class FieldDominanceRunV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    target_id: str
    target_kind: str
    started_at: AwareDatetime
    ended_at: AwareDatetime
    tick_count: int = Field(ge=1)
    min_streak_at_run: int = Field(ge=1)
    first_source_attention_frame_id: str
    last_source_attention_frame_id: str
    # The first observed run may have begun before this recorder was installed.
    left_censored: bool = False

    @model_validator(mode="after")
    def ordered_times(self) -> "FieldDominanceRunV1":
        if self.ended_at < self.started_at:
            raise ValueError("ended_at precedes started_at")
        return self
