"""Optional asynchronous tactical choice using Orion's existing LLM gateway.

Only bounded game facts are sent. No chat/recall, frame images, tools or memory writes.
Never runs in the emulator or input relay task.
"""
import asyncio
import json
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field
from typing import Annotated, Any, Literal, Optional, Union


class Choice(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    goal: Literal['wait', 'walk_left', 'walk_right', 'jump_left', 'jump_right']
    rationale: str = Field(min_length=1, max_length=512)


async def choose(config, attempt):
    from orion.core.bus.async_service import OrionBusAsync
    from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef
    from orion.core.llm_json import parse_json_object
    bus = OrionBusAsync(str(config.ORION_BUS_URL))
    async def request():
        await bus.connect()
        correlation = str(uuid4())
        reply = f'orion:exec:result:{config.LLM_GATEWAY_SERVICE_NAME}:{correlation}'
        prompt = (
            'Choose one supported five-second game motor goal from this observed attempt. '
            'This is Nova the Squirrel only when observation.status is observed. '
            'If unknown, choose wait; do not invent game state. '
            'No horizontal progress after walking can justify testing a jump in that direction. '
            'Position change is not proof of success or reaching an exit. '
            'The motor pulses jump while moving for jump_left/jump_right. '
            'Reply only JSON with goal and a brief evidence-grounded rationale. Facts: '
            + json.dumps(attempt, separators=(',', ':')))
        payload = ChatRequestPayload(messages=[LLMMessage(role='user', content=prompt)],
            route='metacog_background', options={'max_tokens': 256, 'purpose': 'retro_goal',
                'skip_spark_candidate_publish': True,
                'chat_template_kwargs': {'enable_thinking': False}})
        envelope = BaseEnvelope(kind='llm.chat.request',
            source=ServiceRef(name='orion-hub-retro', version='1', node=config.NODE_NAME),
            correlation_id=correlation, reply_to=reply, payload=payload.model_dump(mode='json'))
        msg = await bus.rpc_request(config.CHANNEL_LLM_INTAKE, envelope,
                                    reply_channel=reply, timeout_sec=20)
        decoded = bus.codec.decode(msg.get('data'))
        if not decoded.ok: raise ValueError('planner_invalid_envelope')
        data = decoded.envelope.payload
        if not (data.get('content') or data.get('text')):
            raise ValueError('planner_empty_result:'+str(data.get('raw', {}).get('error', 'no_content'))[:200])
        choice = Choice.model_validate(parse_json_object(data.get('content') or data.get('text') or ''))
        if attempt['observation'].get('status') != 'observed' and choice.goal != 'wait':
            raise ValueError('planner_unobserved_action')
        return choice.model_dump() | {'planner_id': 'llm-gateway:metacog_background:'+correlation}
    try:
        return await asyncio.wait_for(request(), 22)
    finally:
        await bus.close()


# --- Zelda planner v2 (docs/superpowers/specs/2026-10-22-orion-plays-zelda-design.md section D) ---
#
# One decision per Retro offer. Retro (Lane A) offers a decision on room entry, goal finished,
# death or stuck; the Hub plans each decision_id at most once at a time, with at most one
# request in flight per process. The planner uses the local GPU route ``metacog_background``
# only. A heat shed, a parse failure, an unsupported goal or an unknown observation produces
# NO goal: Retro's motor holds ``wait``. There is no fallback planner and no fallback goal.
#
# Retro contract this reads (session status ``scripted[].cognition``, Lane A):
#   decision: {decision_id, trigger, observation (zelda-1-ram-v1), supported_goals}
#   goal:     the accepted GoalV2 (or None)
#   history:  goal_finished records [{decision_id, goal, prediction, outcome, verdict, ...}]
# and GET /sessions/{id}/atlas (zelda-atlas-v1). Unknown extra fields are ignored.

from orion.gpu_pool.shed_backoff import ShedBackoff, gateway_error_reason, gateway_shed_refusal

ZELDA_OBSERVATION_SCHEMA = 'zelda-1-ram-v1'
ZELDA_ROUTE = 'metacog_background'
ZELDA_GOALS = ('exit', 'enter', 'fight', 'retreat', 'collect', 'explore', 'wait')
PREDICTION_KINDS = ('reach_room', 'enter_cave', 'item_gained', 'enemies_cleared', 'survive', 'unknown')
HISTORY_LIMIT = 5
# Overworld and dungeon rooms are both laid out 16 wide in RAM ($EB = row*16 + col).
ROOM_COLS, OVERWORLD_ROWS = 16, 8
MAP_ROWS, MAP_COLS, HUD_PX, TILE_PX = 22, 32, 64, 8

# One shared backoff per process for this call path (orion/gpu_pool/shed_backoff.py).
SHED_BACKOFF = ShedBackoff(name='hub-retro-planner')
_in_flight = False


class PlannerError(Exception):
    """A decision that produced no goal. ``reason`` is the stable code shown to the operator."""

    def __init__(self, reason: str, **details: Any):
        super().__init__(reason)
        self.reason = reason
        self.details = details

    def public(self) -> dict:
        return {'detail': self.reason, **self.details}


class PlannerBusy(PlannerError):
    pass


class PlannerRejected(PlannerError):
    pass


class PlannerShed(PlannerError):
    """The GPU pool refused (or our backoff knows it would refuse) because it is shedding."""


class PlannerUnavailable(PlannerError):
    pass


Direction = Literal['N', 'S', 'E', 'W']


class _Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Tile(_Strict):
    x: int = Field(ge=0, le=MAP_COLS - 1)
    y: int = Field(ge=0, le=MAP_ROWS - 1)


class ExitGoal(_Strict):
    kind: Literal['exit']
    dir: Direction


class EnterGoal(_Strict):
    kind: Literal['enter']
    tile: Tile


class FightGoal(_Strict):
    kind: Literal['fight']
    target: Union[Literal['nearest'], Annotated[int, Field(ge=1, le=11)]]


class RetreatGoal(_Strict):
    kind: Literal['retreat']


class CollectGoal(_Strict):
    kind: Literal['collect']
    tile: Tile


class ExploreGoal(_Strict):
    kind: Literal['explore']
    dir: Direction


class WaitGoal(_Strict):
    kind: Literal['wait']


GoalSpec = Annotated[Union[ExitGoal, EnterGoal, FightGoal, RetreatGoal, CollectGoal, ExploreGoal, WaitGoal],
                     Field(discriminator='kind')]


class Prediction(_Strict):
    kind: Literal['reach_room', 'enter_cave', 'item_gained', 'enemies_cleared', 'survive', 'unknown']
    value: Optional[Annotated[int, Field(ge=0, le=255)]] = None


class ZeldaChoice(_Strict):
    """What the model must return. Anything else (extra keys, other goals) is rejected."""
    goal: GoalSpec
    prediction: Prediction
    rationale: str = Field(min_length=1, max_length=512)
    budget_seconds: int = Field(ge=3, le=30)


def _grid(observation: dict) -> Optional[list]:
    walls = observation.get('walls')
    grid = walls.get('grid') if isinstance(walls, dict) else walls
    if (isinstance(grid, list) and len(grid) == MAP_ROWS
            and all(isinstance(r, list) and len(r) == MAP_COLS for r in grid)):
        return grid
    return None


def _tile_xy(tile: Any) -> Optional[tuple[int, int]]:
    if isinstance(tile, dict) and isinstance(tile.get('x'), int) and isinstance(tile.get('y'), int):
        return tile['x'], tile['y']
    if isinstance(tile, (list, tuple)) and len(tile) == 2 and all(isinstance(v, int) for v in tile):
        return tile[0], tile[1]
    return None


def _slot_char(slot: int) -> str:
    return '0123456789ab'[slot] if 0 <= slot < 12 else '?'


def render_room_map(observation: dict, room: Optional[dict] = None) -> str:
    """The room as 22 text rows of 32 tiles (8 px each): # wall, . floor, L Link, 1-9/a/b enemy
    slot, C known cave. ``unavailable`` when the reader sent no wall grid (scroll, menu, death)."""
    grid = _grid(observation)
    if grid is None:
        return 'room map unavailable in this mode'
    rows = [['#' if cell else '.' for cell in row] for row in grid]
    for cave in (room or {}).get('caves') or []:
        xy = _tile_xy(cave.get('tile') if isinstance(cave, dict) else None)
        if xy and 0 <= xy[0] < MAP_COLS and 0 <= xy[1] < MAP_ROWS:
            rows[xy[1]][xy[0]] = 'C'
    def mark(x, y, ch):
        col, row = int(x) // TILE_PX, (int(y) - HUD_PX) // TILE_PX
        if 0 <= col < MAP_COLS and 0 <= row < MAP_ROWS:
            rows[row][col] = ch
    for enemy in observation.get('enemies') or []:
        if isinstance(enemy, dict) and isinstance(enemy.get('x'), int) and isinstance(enemy.get('y'), int):
            mark(enemy['x'], enemy['y'], _slot_char(int(enemy.get('slot', -1))))
    link = observation.get('link') or {}
    if isinstance(link.get('x'), int) and isinstance(link.get('y'), int):
        mark(link['x'], link['y'], 'L')
    header = '   ' + ''.join(str(c % 10) for c in range(MAP_COLS))
    return '\n'.join([header] + [f'{r:2d} ' + ''.join(row) for r, row in enumerate(rows)])


def room_key(level: Any, room: Any) -> str:
    return f'{level}:{room}'


def neighbours(level: int, room: int) -> dict[str, int]:
    """Adjacent room ids by direction. Rooms are 16 wide; the overworld is 8 rows tall."""
    row, col = divmod(room, ROOM_COLS)
    out = {}
    if row > 0: out['N'] = room - ROOM_COLS
    if row < OVERWORLD_ROWS - 1: out['S'] = room + ROOM_COLS
    if col < ROOM_COLS - 1: out['E'] = room + 1
    if col > 0: out['W'] = room - 1
    return out


def atlas_neighbourhood(atlas: Optional[dict], level: Any, room: Any) -> str:
    """This room, its neighbours, unexplored exits, known caves and death sites, from the atlas only."""
    if not isinstance(atlas, dict) or not isinstance(level, int) or not isinstance(room, int):
        return 'atlas unavailable'
    rooms = atlas.get('rooms') if isinstance(atlas.get('rooms'), dict) else {}
    here = rooms.get(room_key(level, room))
    lines = []
    if not here:
        lines.append(f'this room {room_key(level, room)}: first visit (not in atlas yet)')
    else:
        exits = here.get('exits_seen') or {}
        seen = ' '.join(f'{d}={exits[d]}' for d in 'NSEW' if exits.get(d) is not None) or 'none'
        unexplored = ','.join(d for d in 'NSEW' if exits.get(d) is None) or 'none'
        caves = []
        for cave in here.get('caves') or []:
            xy = _tile_xy(cave.get('tile')) if isinstance(cave, dict) else None
            if xy:
                caves.append(f'({xy[0]},{xy[1]}) {"entered" if cave.get("entered") else "not entered"}')
        lines.append(f'this room {room_key(level, room)}: visits {here.get("visits", 0)}, deaths {here.get("deaths", 0)}, '
                     f'exits seen {seen}, unexplored exits {unexplored}, caves {"; ".join(caves) or "none"}, '
                     f'learned-blocked tiles {len(here.get("learned_blocked") or [])}')
    for d, other in neighbours(level, room).items():
        r = rooms.get(room_key(level, other))
        lines.append(f'{d} {room_key(level, other)}: ' + (
            f'visited {r.get("visits", 0)}x, deaths {r.get("deaths", 0)}' if r else 'never visited'))
    deaths = sorted(((k, v.get('deaths', 0)) for k, v in rooms.items() if isinstance(v, dict) and v.get('deaths')),
                    key=lambda kv: -kv[1])[:8]
    lines.append('death sites: ' + (', '.join(f'{k} ({n})' for k, n in deaths) or 'none'))
    items = [f'{i.get("item")} at {i.get("room")}' for i in (atlas.get('items_gained') or []) if isinstance(i, dict)]
    lines.append('items gained: ' + (', '.join(items[-8:]) or 'none'))
    lines.append(f'rooms known: {len(rooms)}')
    return '\n'.join(lines)


def _goal_text(goal: Any) -> str:
    if not isinstance(goal, dict):
        return str(goal)
    kind = goal.get('kind', '?')
    arg = goal.get('dir') or goal.get('target') or _tile_xy(goal.get('tile'))
    return f'{kind} {arg}' if arg is not None else str(kind)


def history_lines(history: Any) -> str:
    rows = [h for h in (history or []) if isinstance(h, dict)][-HISTORY_LIMIT:]
    if not rows:
        return 'none yet'
    out = []
    for h in rows:
        p = h.get('prediction') or {}
        out.append(f'{_goal_text(h.get("goal"))}; predicted {p.get("kind", "?")}'
                   f'{" " + str(p["value"]) if p.get("value") is not None else ""}; '
                   f'outcome {h.get("outcome", "unknown")}; verdict {h.get("verdict", "unknown")}')
    return '\n'.join(out)


def observation_facts(observation: dict) -> dict:
    """The observation without the wall grid (the map carries it)."""
    return {k: v for k, v in observation.items() if k not in ('walls',)}


def build_zelda_prompt(decision: dict, atlas: Optional[dict], history: Any) -> str:
    obs = decision['observation']
    here = None
    if isinstance(atlas, dict) and isinstance(atlas.get('rooms'), dict):
        here = atlas['rooms'].get(room_key(obs.get('level'), obs.get('room')))
    return '\n'.join([
        'You are Orion, playing The Legend of Zelda (NES) through a navigator that turns one goal into input.',
        'Choose ONE goal for the next 3-30 seconds from the observed facts only. Do not use outside knowledge of the game map.',
        f'Trigger for this decision: {decision.get("trigger", "unknown")}.',
        'Room map (22 rows x 32 tiles, 8 px tiles; # wall, . floor, L Link, 1-9/a/b enemy slot, C known cave):',
        render_room_map(obs, here),
        'Observation (zelda-1-ram-v1): ' + json.dumps(observation_facts(obs), separators=(',', ':')),
        'Atlas (only what you observed before):',
        atlas_neighbourhood(atlas, obs.get('level'), obs.get('room')),
        'Last decisions and verdicts (oldest first):',
        history_lines(history),
        'Supported goals: ' + ', '.join(decision['supported_goals']) + '.',
        'Goal JSON shapes: {"kind":"exit","dir":"N|S|E|W"} {"kind":"enter","tile":{"x":0-31,"y":0-21}} '
        '{"kind":"fight","target":"nearest" or enemy slot 1-11} {"kind":"retreat"} '
        '{"kind":"collect","tile":{"x":0-31,"y":0-21}} {"kind":"explore","dir":"N|S|E|W"} {"kind":"wait"}.',
        'Prediction kinds: reach_room (value = room id), enter_cave, item_gained, enemies_cleared, survive, unknown.',
        'Make the prediction checkable: it will be graded met/refuted from RAM.',
        'Reply only JSON: {"goal":{...},"prediction":{"kind":"...","value":int or null},'
        '"rationale":"brief, grounded in the facts above","budget_seconds":3-30}',
    ])


def validate_zelda_choice(raw: Any, decision: dict) -> ZeldaChoice:
    """Parse and check the model's answer against what this decision allows. Raises PlannerRejected."""
    from pydantic import ValidationError
    try:
        choice = ZeldaChoice.model_validate(raw)
    except ValidationError as exc:
        raise PlannerRejected('planner_invalid_goal', error=str(exc.errors()[0].get('msg', ''))[:200])
    if choice.goal.kind not in decision['supported_goals']:
        raise PlannerRejected('planner_unsupported_goal', goal=choice.goal.kind)
    if isinstance(choice.goal, FightGoal) and choice.goal.target != 'nearest':
        alive = {e.get('slot') for e in decision['observation'].get('enemies') or [] if isinstance(e, dict)}
        if choice.goal.target not in alive:
            raise PlannerRejected('planner_unknown_target', target=choice.goal.target)
    if choice.prediction.kind == 'reach_room' and choice.prediction.value is None:
        raise PlannerRejected('planner_unchecked_prediction')
    return choice


def check_decision(decision: Any) -> dict:
    """A decision the planner may answer, or PlannerRejected. Unknown observations get no goal."""
    if not isinstance(decision, dict) or not isinstance(decision.get('decision_id'), str):
        raise PlannerRejected('decision_pending')
    obs = decision.get('observation')
    if not isinstance(obs, dict) or obs.get('status') != 'observed':
        raise PlannerRejected('observation_unknown',
                              observation_reason=str((obs or {}).get('reason', 'unknown'))[:120])
    goals = decision.get('supported_goals')
    if not isinstance(goals, list) or not goals or any(g not in ZELDA_GOALS for g in goals):
        raise PlannerRejected('decision_supported_goals_invalid')
    return decision


async def _gateway_chat(config, prompt: str, *, max_tokens: int) -> tuple[dict, str]:
    """One llm.chat.request on the local GPU route. Returns (result payload, correlation id)."""
    from orion.core.bus.async_service import OrionBusAsync
    from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef
    bus = OrionBusAsync(str(config.ORION_BUS_URL))
    correlation = str(uuid4())
    async def request():
        await bus.connect()
        reply = f'orion:exec:result:{config.LLM_GATEWAY_SERVICE_NAME}:{correlation}'
        payload = ChatRequestPayload(messages=[LLMMessage(role='user', content=prompt)],
            route=ZELDA_ROUTE, options={'max_tokens': max_tokens, 'purpose': 'retro_goal_v2',
                'skip_spark_candidate_publish': True,
                'chat_template_kwargs': {'enable_thinking': False}})
        envelope = BaseEnvelope(kind='llm.chat.request',
            source=ServiceRef(name='orion-hub-retro', version='2', node=config.NODE_NAME),
            correlation_id=correlation, reply_to=reply, payload=payload.model_dump(mode='json'))
        msg = await bus.rpc_request(config.CHANNEL_LLM_INTAKE, envelope, reply_channel=reply, timeout_sec=30)
        decoded = bus.codec.decode(msg.get('data'))
        if not decoded.ok:
            raise PlannerUnavailable('planner_invalid_envelope')
        return decoded.envelope.payload
    try:
        return await asyncio.wait_for(request(), 32), correlation
    finally:
        await bus.close()


async def plan_zelda(config, decision: Any, atlas: Optional[dict], history: Any, *, chat=None) -> dict:
    """Answer one Retro decision with a GoalV2 body (without sponsor/generation/port/epoch).

    Raises PlannerBusy (another request in flight), PlannerRejected (nothing to answer, or the
    answer broke the contract), PlannerShed (GPU heat shed; ``details`` carries the pool's
    ``reason`` and ``retry_after_sec``) or PlannerUnavailable. Never returns a goal it did not
    get from the model."""
    global _in_flight
    decision = check_decision(decision)
    if _in_flight:
        raise PlannerBusy('planner_busy')
    key = ZELDA_ROUTE
    wait, token = SHED_BACKOFF.admit(key)
    if wait is not None:
        raise PlannerShed('gpu_pool_unavailable', reason=SHED_BACKOFF.reason(key) or 'shed:unknown',
                          retry_after_sec=round(wait, 1), skipped_locally=True)
    _in_flight = True
    outcome: Optional[str] = 'cancelled'
    try:
        prompt = build_zelda_prompt(decision, atlas, history)
        try:
            data, correlation = await (chat or _gateway_chat)(config, prompt, max_tokens=320)
        except PlannerError:
            outcome = 'pool_unreachable'
            raise
        except Exception as exc:
            outcome = 'pool_unreachable'
            raise PlannerUnavailable('planner_unavailable', error=type(exc).__name__)
        shed = gateway_shed_refusal(data)
        if shed:
            outcome = shed[0]
            SHED_BACKOFF.resolve(key, shed[0], token=token, hint_sec=shed[1])
            token = None
            retry = SHED_BACKOFF.remaining(key)
            raise PlannerShed('gpu_pool_unavailable', reason=shed[0],
                              retry_after_sec=round(retry if retry is not None else (shed[1] or 30.0), 1))
        refusal = gateway_error_reason(data)
        if refusal:
            outcome = refusal
            raise PlannerUnavailable('gpu_pool_unavailable', reason=refusal)
        outcome = None  # the pool answered: a shed episode (if any) ends
        text = data.get('content') or data.get('text') or ''
        if not text:
            raise PlannerUnavailable('planner_empty_result',
                                     error=str((data.get('raw') or {}).get('error', 'no_content'))[:200])
        from orion.core.llm_json import parse_json_object
        try:
            raw = parse_json_object(text)
        except ValueError:
            raise PlannerRejected('planner_unparseable')
        choice = validate_zelda_choice(raw, decision)
        return choice.model_dump(mode='json') | {
            'decision_id': decision['decision_id'],
            'planner_id': 'llm-gateway:metacog_background:' + correlation}
    finally:
        _in_flight = False
        if token is not None:
            SHED_BACKOFF.resolve(key, outcome, token=token)
