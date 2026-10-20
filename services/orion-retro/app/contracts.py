from typing import Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Start(Strict):
    game_id: str = Field(pattern=r'^[a-z0-9][a-z0-9-]{0,63}$')
    idempotency_key: str = Field(min_length=1, max_length=64)


class Control(Strict):
    connection_id: str = Field(min_length=36, max_length=36)
    port: int = Field(ge=0, le=1)
    action: Literal['acquire', 'release', 'takeover'] = 'acquire'


class Input(Strict):
    version: Literal[1]
    session_id: str = Field(min_length=36, max_length=36)
    generation: str = Field(min_length=36, max_length=36)
    lease: str = Field(min_length=32, max_length=128)
    epoch: int = Field(ge=1)
    seq: int = Field(ge=0, le=2**53-1)
    port: int = Field(ge=0, le=1)
    mask: int = Field(ge=0, le=65535)


class ScriptControl(Strict):
    cognition: bool = False
    connection_id: str = Field(min_length=36, max_length=36)
    generation: str = Field(min_length=36, max_length=36)
    port: int = Field(ge=0, le=1)
    action: Literal['start', 'stop']
    duration_seconds: int = Field(default=60, ge=5, le=1800)
    expected_epoch: int = Field(ge=1)


class Goal(Strict):
    connection_id: str = Field(min_length=36, max_length=36)
    generation: str = Field(min_length=36, max_length=36)
    port: int = Field(ge=0, le=1)
    expected_epoch: int = Field(ge=1)
    attempt_id: str = Field(min_length=1, max_length=80)
    goal: Literal['wait', 'walk_left', 'walk_right', 'jump_left', 'jump_right']
    rationale: str = Field(min_length=1, max_length=512)
    planner_id: str = Field(min_length=1, max_length=128)
