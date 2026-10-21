"""Simulation-only contracts. Never route these payloads to personal memory."""
from __future__ import annotations
from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class TownEffect(Strict):
    op: Literal['stock_delta','need_delta','fact_set','trust_delta','place_flag','belief_set']
    target: str = Field(min_length=1,max_length=160)
    value: bool | float | str
    source: str | None = None

class TownCondition(Strict):
    target: str = Field(pattern=r'^(stock|fact):[a-zA-Z0-9_.:|-]+$')
    cmp: Literal['eq','gte']
    value: bool | float | str

class TownQuest(Strict):
    id: str = Field(min_length=1,max_length=160)
    title: str = Field(min_length=1,max_length=200)
    participants: list[str] = Field(min_length=1,max_length=8)
    goal: list[TownCondition] = Field(min_length=1,max_length=8)
    steps: list[list[TownCondition]] = Field(default_factory=list,max_length=8)
    deadline: int = Field(ge=0)
    status: Literal['open'] = 'open'
    reward: list[TownEffect] = Field(default_factory=list,max_length=16)
    penalty: list[TownEffect] = Field(default_factory=list,max_length=16)

class TownSourceRef(Strict):
    kind: Literal['digest','weather']
    id: str = Field(min_length=1,max_length=200)

class TownSimEventRequestV1(Strict):
    correlation_id: str = Field(min_length=1,max_length=100)
    world_id: str = Field(min_length=1,max_length=100)
    source: Literal['juniper','orion','mirror']
    kind: Literal['rumor','delivery','weather_spell','price_shift','late_shipment','quest']
    title: str = Field(min_length=1,max_length=200)
    effects: list[TownEffect] = Field(default_factory=list,max_length=16)
    quest: TownQuest | None = None
    source_ref: TownSourceRef | None = None
    expires_at: int | None = Field(None,ge=0)
    requested_by: str = Field(min_length=1,max_length=100)

    @model_validator(mode='after')
    def source_contract(self):
        if self.source == 'mirror' and not self.source_ref:
            raise ValueError('mirror source_ref required')
        if self.kind == 'quest' and not self.quest:
            raise ValueError('quest goal required')
        if self.quest and self.kind != 'quest':
            raise ValueError('quest requires quest kind')
        return self

class TownSimEventResultV1(Strict):
    correlation_id: str
    world_id: str
    accepted: bool
    reason: str
    event_id: str | None = None

class TownSimSceneResolvedV1(Strict):
    id: str
    world_id: str
    scene_id: str
    participants: list[str]
    purpose: str
    legal_results: list[dict[str,Any]]
    chosen_results: list[str]
    effects_applied: list[TownEffect]
    seed: int
    status: Literal['resolved','stale']
    transcript: list[dict[str,str]]

class TownSimPredictionScoreV1(Strict):
    id: str
    world_id: str
    target_id: str
    purpose: str
    probabilities: dict[str,Annotated[float,Field(ge=0,le=1)]]
    outcome: list[str]
    brier: float = Field(ge=0,le=1)
    brier_resample: float | None = Field(None,ge=0,le=1)
    brier_base_rate: float = Field(ge=0,le=1)
    baseline_count: int = Field(ge=0)
    resample_count: int = Field(ge=0,le=3)
    status: str
    signal_status: Literal['unverified'] = 'unverified'

class TownSimStoredV1(Strict):
    world_id: str
    event_id: str
    record_kind: Literal['town.sim.scene.resolved.v1','town.sim.prediction.scored.v1']
