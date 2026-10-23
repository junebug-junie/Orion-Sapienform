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
# One decision per Retro offer. Retro's navigator (services/orion-retro/app/zelda/navigate.py,
# Lane A) offers a decision slot whenever Link is controllable and no goal is active: at
# session start, on room entry and after every goal_finished (met, refuted, hit, stuck,
# budget, death). The Hub plans each decision_id at most once at a time, with at most one
# request in flight per process. Local GPU route ``metacog_background`` only. A heat shed, a
# parse failure, an unsupported goal or an unknown observation produces NO goal: Retro's
# motor holds still. There is no fallback planner and no fallback goal.
#
# Retro contract read here (session status ``scripted[].cognition``, policy_id
# ``zelda-navigator-v1``):
#   attempt:       {decision_id, port, expected_epoch, observation (zelda-1-ram-v1 without the
#                   wall grid unless Retro includes it), supported_goals, supported_predictions,
#                   open_sides (the screen edges Link can walk to, or None outside play),
#                   previous (the last goal_finished row)}
#   goal:          the accepted GoalV2, or None
#   last_finished: the newest goal_finished row {decision_id, goal, prediction, outcome, verdict}
# GET /sessions/{id}/atlas: the neighbourhood view {lineage_id, level, room, current,
#   adjacent{N..W: {room, known, seen_exit}} (off-grid sides absent), rooms_known, rooms_with_unvisited_sides,
#   caves[{room, tile, entered}], deaths[{room, deaths}], map{"level:room": {visits, exits,
#   caves, deaths}}}. Callers keep the last-5 history themselves (``remember_finished``).
# Unknown extra fields are ignored everywhere.

from collections import deque

from orion.gpu_pool.shed_backoff import ShedBackoff, gateway_error_reason, gateway_shed_refusal

ZELDA_POLICY_ID = 'zelda-navigator-v1'
ZELDA_ROUTE = 'metacog_background'
ZELDA_GOALS = ('exit', 'enter', 'fight', 'retreat', 'collect', 'explore', 'wait')
PREDICTION_KINDS = ('reach_room', 'enter_cave', 'item_gained', 'enemies_cleared', 'survive', 'unknown')
HISTORY_LIMIT = 5
MAP_ROWS, MAP_COLS, HUD_PX, TILE_PX = 22, 32, 64, 8

# One shared backoff per process for this call path (orion/gpu_pool/shed_backoff.py).
SHED_BACKOFF = ShedBackoff(name='hub-retro-planner')
_in_flight = False


class PlannerError(Exception):
    """A decision that produced no goal. ``reason`` is the stable code shown to the operator."""

    def __init__(self, code: str, /, **details: Any):
        super().__init__(code)
        self.reason = code          # the stable code; ``details['reason']`` is the pool's own reason
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


def is_zelda_cognition(cognition: Any) -> bool:
    return isinstance(cognition, dict) and cognition.get('policy_id') == ZELDA_POLICY_ID


def remember_finished(history: deque, cognition: Any) -> deque:
    """Append Retro's newest goal_finished row once (by decision_id). Retro only shows the last one."""
    row = cognition.get('last_finished') if isinstance(cognition, dict) else None
    if isinstance(row, dict) and row.get('decision_id') and not any(
            h.get('decision_id') == row['decision_id'] for h in history):
        history.append(row)
    return history


def _grid(observation: dict) -> Optional[list]:
    walls = observation.get('walls')
    grid = walls.get('grid') if isinstance(walls, dict) else walls
    if (isinstance(grid, list) and len(grid) == MAP_ROWS
            and all(isinstance(r, list) and len(r) == MAP_COLS for r in grid)):
        return grid
    return None


def _terrain(observation: dict) -> Optional[list]:
    t = observation.get('terrain')
    if (isinstance(t, list) and len(t) == MAP_ROWS
            and all(isinstance(r, str) and len(r) == MAP_COLS and set(r) <= set('#.xC') for r in t)):
        return t
    return None


def _tile_xy(tile: Any) -> Optional[tuple[int, int]]:
    if isinstance(tile, dict) and isinstance(tile.get('x'), int) and isinstance(tile.get('y'), int):
        return tile['x'], tile['y']
    if isinstance(tile, (list, tuple)) and len(tile) == 2 and all(isinstance(v, int) for v in tile):
        return tile[0], tile[1]
    return None


def tile_of(thing: dict) -> Optional[tuple[int, int]]:
    """Retro's own ``tile`` when present, else its convention: the game's collision hotspot
    (column (x+4)//8, row (y+11-64+4)//8; services/orion-retro/app/zelda/observe.py)."""
    xy = _tile_xy(thing.get('tile'))
    if xy:
        return xy
    if isinstance(thing.get('x'), int) and isinstance(thing.get('y'), int):
        return (thing['x'] + 4) // TILE_PX, (thing['y'] + 11 - HUD_PX + 4) // TILE_PX
    return None


def _slot_char(slot: Any) -> str:
    return '0123456789ab'[slot] if isinstance(slot, int) and 0 <= slot < 12 else '?'


def _room_caves(atlas: Optional[dict], level: Any, room: Any) -> list:
    if not isinstance(atlas, dict):
        return []
    key = f'{level}:{room}'
    return [c for c in atlas.get('caves') or [] if isinstance(c, dict) and c.get('room') == key]


def render_room_map(observation: dict, atlas: Optional[dict] = None) -> str:
    """The room as 22 text rows of 32 tiles (8 px): # wall, . floor, ? terrain not sent,
    x learned-blocked, C known cave, d item drop, 1-9/a/b enemy slot, L Link."""
    terrain = _terrain(observation)
    grid = None if terrain else _grid(observation)
    if terrain is None and grid is None and observation.get('mode') not in ('play', 'cave'):
        return f'room map unavailable in mode {observation.get("mode", "unknown")}'
    # Retro's own terrain (#/./x/C, navigate.py terrain()) wins; a raw wall grid is drawn here.
    rows = ([list(r) for r in terrain] if terrain else
            [['#' if cell else '.' for cell in row] for row in grid] if grid
            else [['?'] * MAP_COLS for _ in range(MAP_ROWS)])
    def put(xy, ch):
        if xy and 0 <= xy[0] < MAP_COLS and 0 <= xy[1] < MAP_ROWS:
            rows[xy[1]][xy[0]] = ch
    current = (atlas or {}).get('current') if isinstance(atlas, dict) else None
    same_room = isinstance(atlas, dict) and atlas.get('room') == observation.get('room')
    if not terrain and same_room and isinstance(current, dict):
        for t in current.get('learned_blocked') or []:
            put(_tile_xy(t), 'x')
    for cave in ([] if terrain else _room_caves(atlas, observation.get('level'), observation.get('room'))):
        put(_tile_xy(cave.get('tile')), 'C')
    for drop in observation.get('drops') or []:
        if isinstance(drop, dict): put(tile_of(drop), 'd')
    for enemy in observation.get('enemies') or []:
        if isinstance(enemy, dict): put(tile_of(enemy), _slot_char(enemy.get('slot')))
    if isinstance(observation.get('link'), dict):
        put(tile_of(observation['link']), 'L')
    header = '   ' + ''.join(str(c % 10) for c in range(MAP_COLS))
    note = [] if (terrain or grid) else ['(walls not included in this observation; ? = terrain unknown to the planner)']
    return '\n'.join(note + [header] + [f'{r:2d} ' + ''.join(row) for r, row in enumerate(rows)])


def atlas_neighbourhood(atlas: Optional[dict]) -> str:
    """This room, its neighbours, unexplored exits, caves and deaths, from Orion's atlas only."""
    if not isinstance(atlas, dict):
        return 'atlas unavailable'
    level, room = atlas.get('level'), atlas.get('room')
    here = atlas.get('current')
    lines = []
    if not isinstance(here, dict):
        lines.append(f'this room {level}:{room}: first visit (not in the atlas yet)')
    else:
        exits = here.get('exits_seen') or {}
        seen = ' '.join(f'{d}={exits[d]}' for d in 'NSEW' if exits.get(d) is not None) or 'none'
        unvisited = ','.join(d for d in 'NSEW' if exits.get(d) is None) or 'none'
        caves = [f'({xy[0]},{xy[1]}) {"entered" if c.get("entered") else "not entered"}'
                 for c in here.get('caves') or [] if isinstance(c, dict) and (xy := _tile_xy(c.get('tile')))]
        lines.append(f'this room {level}:{room}: visits {here.get("visits", 0)}, deaths {here.get("deaths", 0)}, '
                     f'sides crossed {seen}, sides never crossed (may be walls) {unvisited}, '
                     f'caves {"; ".join(caves) or "none"}, '
                     f'learned-blocked tiles {len(here.get("learned_blocked") or [])}')
    for d in 'NSEW':
        a = (atlas.get('adjacent') or {}).get(d)
        if isinstance(a, dict):
            lines.append(f'{d} {level}:{a.get("room")}: ' + ('known' if a.get('known') else 'never visited')
                         + (f', reached from here before' if a.get('seen_exit') is not None else ''))
    deaths = [f'{x.get("room")} ({x.get("deaths")})' for x in atlas.get('deaths') or [] if isinstance(x, dict)]
    lines.append('death sites: ' + (', '.join(deaths[:8]) or 'none'))
    frontier = [k for k in atlas.get('rooms_with_unvisited_sides') or [] if isinstance(k, str)]
    lines.append(f'rooms known on this level: {atlas.get("rooms_known", 0)}; with sides never crossed: '
                 + (', '.join(frontier[:12]) or 'none'))
    return '\n'.join(lines)


def _goal_text(goal: Any) -> str:
    if not isinstance(goal, dict):
        return str(goal)
    kind = goal.get('kind', '?')
    arg = goal.get('dir') or goal.get('target') or _tile_xy(goal.get('tile'))
    return f'{kind} {arg}' if arg is not None else str(kind)


def sides_line(decision: dict, atlas: Optional[dict] = None) -> str:
    sides = decision.get('open_sides')
    if not isinstance(sides, list):
        return 'Sides Link can walk off: not known here.'
    adjacent = (atlas or {}).get('adjacent') if isinstance(atlas, dict) else None
    def label(d):
        a = (adjacent or {}).get(d)
        if not isinstance(a, dict): return d
        return f'{d} (room {a.get("room")}, {"visited" if a.get("known") else "never visited"})'
    closed = [d for d in 'NSEW' if d not in sides]
    return (f'Sides Link can walk off from here (Retro searched the map; an exit can only use these): '
            f'{", ".join(label(d) for d in sides) or "none"}. No way out: {", ".join(closed) or "none"}.')


def history_lines(history: Any) -> str:
    rows = [h for h in (history or []) if isinstance(h, dict)][-HISTORY_LIMIT:]
    if not rows:
        return 'none yet'
    out = []
    for h in rows:
        p = h.get('prediction') or {}
        out.append(f'{_goal_text(h.get("goal"))}; predicted {p.get("kind", "?")}'
                   f'{" " + str(p["value"]) if p.get("value") is not None else ""}; '
                   f'outcome {h.get("outcome", "unknown")}; verdict {h.get("verdict", "unknown")}'
                   + ('; Link did not move' if h.get('applied_frames') == 0 else ''))
    return '\n'.join(out)


def observation_facts(observation: dict) -> dict:
    """The observation without the wall grid (the map carries it)."""
    return {k: v for k, v in observation.items() if k not in ('walls', 'terrain')}


def build_zelda_prompt(decision: dict, atlas: Optional[dict], history: Any) -> str:
    obs = decision['observation']
    previous = decision.get('previous')
    why = (f'the previous goal ended: {_goal_text(previous.get("goal"))} -> {previous.get("outcome", "unknown")}'
           if isinstance(previous, dict) else 'first decision of this session')
    if isinstance(previous, dict) and previous.get('outcome') == 'death':
        # Retro recovers through the Continue screen itself; death is only why we decide again.
        why += ' (Link died; Retro already chose Continue and play has resumed)'
    return '\n'.join([
        'You are Orion, playing The Legend of Zelda (NES) through a navigator that turns one goal into input.',
        'Choose ONE goal for the next 3-30 seconds from the observed facts only. Do not use outside knowledge of the game map.',
        f'Why a decision now: {why}.',
        'Room map (22 rows x 32 tiles of 8 px, row 0 = top of the play area; tile x = column, y = row): '
        '# blocked, . walkable, x blocked (Link bumped into it here before), C a cave you entered, '
        '? terrain not known, L Link, 1-9/a/b enemy slot, d item drop.',
        render_room_map(obs, atlas),
        sides_line(decision, atlas),
        'Observation (zelda-1-ram-v1): ' + json.dumps(observation_facts(obs), separators=(',', ':')),
        'Atlas (only what you observed before):',
        atlas_neighbourhood(atlas),
        'Last decisions and verdicts (oldest first):',
        history_lines(history),
        'Supported goals: ' + ', '.join(decision['supported_goals']) + '.',
        'Goal JSON shapes: {"kind":"exit","dir":"N|S|E|W"} {"kind":"enter","tile":{"x":0-31,"y":0-21}} '
        '{"kind":"fight","target":"nearest" or enemy slot 1-11} {"kind":"retreat"} '
        '{"kind":"collect","tile":{"x":0-31,"y":0-21}} {"kind":"explore","dir":"N|S|E|W"} {"kind":"wait"}.',
        'Prediction kinds: reach_room (value = the room id you expect to reach; required), '
        'enter_cave, item_gained, enemies_cleared, survive, unknown (no value for these).',
        'Make the prediction checkable: Retro grades it met/refuted from RAM.',
        'Items inside caves are not in the observation: to collect one, give the tile yourself.',
        'fight needs at least one enemy in the observation.',
        'Reply only JSON: {"goal":{...},"prediction":{"kind":"..."[, "value":int]},'
        '"rationale":"brief, grounded in the facts above","budget_seconds":3-30}',
    ])


def validate_zelda_choice(raw: Any, decision: dict) -> ZeldaChoice:
    """Parse and check the model's answer against what this decision allows. Raises PlannerRejected.
    Mirrors Retro's own GoalV2 checks so a doomed goal is never posted."""
    from pydantic import ValidationError
    try:
        choice = ZeldaChoice.model_validate(raw)
    except ValidationError as exc:
        raise PlannerRejected('planner_invalid_goal', error=str(exc.errors()[0].get('msg', ''))[:200])
    if choice.goal.kind not in decision['supported_goals']:
        raise PlannerRejected('planner_unsupported_goal', goal=choice.goal.kind)
    predictions = decision.get('supported_predictions') or PREDICTION_KINDS
    if choice.prediction.kind not in predictions:
        raise PlannerRejected('planner_unsupported_prediction', prediction=choice.prediction.kind)
    if (choice.prediction.kind == 'reach_room') != (choice.prediction.value is not None):
        raise PlannerRejected('planner_prediction_value_mismatch')
    obs = decision['observation']
    if choice.goal.kind != 'wait' and not (obs.get('link') or {}).get('controllable', True):
        raise PlannerRejected('observation_unknown', observation_reason='link_not_controllable')
    sides = decision.get('open_sides')
    direction = getattr(choice.goal, 'dir', None)
    if isinstance(choice.goal, ExitGoal) and isinstance(sides, list) and direction not in sides:
        raise PlannerRejected('planner_side_closed', dir=direction)   # Retro would push into a wall
    previous = decision.get('previous')
    if isinstance(previous, dict) and previous.get('goal') == choice.goal.model_dump(mode='json') and (
            previous.get('applied_frames') == 0
            or (isinstance(choice.goal, ExploreGoal) and isinstance(sides, list) and direction not in sides)):
        # Live 2026-10-23: "explore S" eight times against a solid south wall, 0 frames moved.
        raise PlannerRejected('planner_repeats_no_effect', goal=_goal_text(previous['goal']))
    if isinstance(choice.goal, FightGoal):
        alive = {e.get('slot') for e in obs.get('enemies') or [] if isinstance(e, dict)}
        if not alive:
            raise PlannerRejected('planner_no_enemies')   # Retro: no_enemies_present
        if choice.goal.target != 'nearest' and choice.goal.target not in alive:
            raise PlannerRejected('planner_unknown_target', target=choice.goal.target)
    return choice


def refusal_note(exc: 'PlannerRejected') -> str:
    """One plain line telling the model why its answer was refused, for the single retry."""
    d = exc.details
    text = {'planner_side_closed': f'exit {d.get("dir")}: Link cannot reach that edge from here',
            'planner_repeats_no_effect': f'{d.get("goal")}: the same goal just left Link where he was',
            'planner_unparseable': 'the reply was not one JSON object'}.get(exc.reason)
    return text or exc.reason + (f' ({d["error"]})' if d.get('error') else '')


def check_decision(decision: Any) -> dict:
    """A decision the planner may answer, or PlannerRejected. Unknown observations get no goal."""
    if not isinstance(decision, dict) or not isinstance(decision.get('decision_id'), str):
        raise PlannerRejected('decision_pending')
    obs = decision.get('observation')
    if not isinstance(obs, dict) or obs.get('status') != 'observed':
        raise PlannerRejected('observation_unknown',
                              observation_reason=str((obs or {}).get('reason', 'unknown'))[:120])
    if obs.get('menu') or obs.get('mode') in ('death', 'menu'):
        # The Continue screen (observation.menu) belongs to Retro's own death recovery. The
        # planner never steers through a menu; a new decision is offered once play resumes.
        raise PlannerRejected('observation_unknown', observation_reason='menu')
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
        # A refused answer gets one retry that names the refusal; the second refusal is final.
        for attempt in (1, 2):
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
                try:
                    raw = parse_json_object(text)
                except ValueError:
                    raise PlannerRejected('planner_unparseable')
                choice = validate_zelda_choice(raw, decision)
                break
            except PlannerRejected as exc:
                if attempt == 2 or exc.reason == 'observation_unknown':
                    raise   # Link is not controllable: asking again cannot help
                prompt += f'\nYour previous answer was refused: {refusal_note(exc)}. Choose a different goal.'
        return choice.model_dump(mode='json') | {
            'decision_id': decision['decision_id'],
            'planner_id': 'llm-gateway:metacog_background:' + correlation}
    finally:
        _in_flight = False
        if token is not None:
            SHED_BACKOFF.resolve(key, outcome, token=token)
