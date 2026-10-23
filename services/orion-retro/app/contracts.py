from typing import Annotated, Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Start(Strict):
    game_id: str = Field(pattern=r'^[a-z0-9][a-z0-9-]{0,63}$')
    idempotency_key: str = Field(min_length=1, max_length=64)
    save_id: str | None = Field(default=None, pattern=r'^[0-9a-f]{32}$')


class LoadSave(Strict):
    save_id: str = Field(pattern=r'^[0-9a-f]{32}$')


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


# ---- Zelda GoalV2 (Nova keeps Goal above) -------------------------------------------
class Sponsored(Strict):
    """Exactly one of a live browser connection or a server-side leisure grant."""
    connection_id: str | None = Field(default=None, min_length=36, max_length=36)
    leisure_id: str | None = Field(default=None, min_length=1, max_length=64,
                                   pattern=r'^[A-Za-z0-9-]{1,64}$')

    @model_validator(mode='after')
    def _one_sponsor(self):
        if (self.connection_id is None) == (self.leisure_id is None):
            raise ValueError('exactly_one_sponsor')
        return self

    def sponsor(self):
        """The arbiter connection key. A leisure grant never shares a browser's key."""
        return self.connection_id if self.connection_id else 'leisure:'+self.leisure_id


class LeisureControl(Sponsored):
    port: int = Field(ge=0, le=1)
    action: Literal['acquire', 'release'] = 'acquire'


class LeisureScriptControl(Sponsored):
    cognition: Literal[True] = True
    generation: str = Field(min_length=36, max_length=36)
    port: int = Field(ge=0, le=1)
    action: Literal['start', 'stop']
    duration_seconds: int = Field(default=60, ge=5, le=1800)
    expected_epoch: int = Field(ge=1)


class Tile(Strict):
    x: int = Field(ge=0, le=31)
    y: int = Field(ge=0, le=21)


class ExitGoal(Strict):
    kind: Literal['exit']
    dir: Literal['N', 'S', 'E', 'W']


class ExploreGoal(Strict):
    kind: Literal['explore']
    dir: Literal['N', 'S', 'E', 'W']


class EnterGoal(Strict):
    kind: Literal['enter']
    tile: Tile


class CollectGoal(Strict):
    kind: Literal['collect']
    tile: Tile


class FightGoal(Strict):
    kind: Literal['fight']
    target: Literal['nearest'] | int = Field(default='nearest')

    @model_validator(mode='after')
    def _slot(self):
        if self.target != 'nearest' and not (isinstance(self.target, int)
                                             and not isinstance(self.target, bool) and 1 <= self.target <= 11):
            raise ValueError('target_slot_out_of_range')
        return self


class RetreatGoal(Strict):
    kind: Literal['retreat']


class WaitGoal(Strict):
    kind: Literal['wait']


class Prediction(Strict):
    kind: Literal['reach_room', 'enter_cave', 'item_gained', 'enemies_cleared', 'survive', 'unknown']
    value: int | None = Field(default=None, ge=0, le=255)

    @model_validator(mode='after')
    def _value(self):
        # Only reach_room carries a value (the room id); every other kind has none.
        if (self.kind == 'reach_room') != (self.value is not None):
            raise ValueError('prediction_value_mismatch')
        return self


class GoalV2(Sponsored):
    generation: str = Field(min_length=36, max_length=36)
    port: int = Field(ge=0, le=1)
    expected_epoch: int = Field(ge=1)
    decision_id: str = Field(min_length=1, max_length=80)  # echoes the worker's offered slot
    goal: Annotated[ExitGoal | ExploreGoal | EnterGoal | CollectGoal | FightGoal | RetreatGoal | WaitGoal,
                    Field(discriminator='kind')]
    prediction: Prediction
    rationale: str = Field(min_length=1, max_length=512)
    planner_id: str = Field(min_length=1, max_length=128)
    budget_seconds: int = Field(ge=3, le=30)
