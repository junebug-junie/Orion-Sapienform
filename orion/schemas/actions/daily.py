from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DailyPulseV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str = Field(min_length=10, max_length=10)
    timezone: str = Field(min_length=1, max_length=80)
    yesterday_theme: str = Field(min_length=1, max_length=200)
    today_focus: str = Field(min_length=1, max_length=200)
    gentle_challenge: str = Field(min_length=1, max_length=240)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("date")
    @classmethod
    def _date_format(cls, value: str) -> str:
        if len(value) != 10 or value[4] != "-" or value[7] != "-":
            raise ValueError("date must be YYYY-MM-DD")
        return value


class DailyMetacogV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str = Field(min_length=10, max_length=10)
    timezone: str = Field(min_length=1, max_length=80)
    thinking_patterns: list[str] = Field(min_length=1, max_length=5)
    blindspots: list[str] = Field(default_factory=list, max_length=5)
    course_correction: str = Field(min_length=1, max_length=300)
    tomorrow_experiment: str = Field(min_length=1, max_length=240)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("date")
    @classmethod
    def _date_format(cls, value: str) -> str:
        if len(value) != 10 or value[4] != "-" or value[7] != "-":
            raise ValueError("date must be YYYY-MM-DD")
        return value

    @field_validator("thinking_patterns", "blindspots")
    @classmethod
    def _trim_items(cls, value: list[str]) -> list[str]:
        return [str(v).strip() for v in value if str(v).strip()]
