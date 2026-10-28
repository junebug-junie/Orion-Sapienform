"""Zelda body adapter for action bets: options, quotes, the planner's menu, place and settle.

Spec: docs/superpowers/specs/2026-10-23-embodied-experience-learning-design.md
("Hub: the Zelda adapter and the menu", the Zelda settle table, "Speaking").

Every planned Zelda goal is a bet in orion-prediction ("this action, in this situation, at this
place, works with probability p"). This module owns the Zelda meaning of that bet:

- ``build_options``: the moves worth asking about for one Retro decision, each mapped onto the
  game-free vocabulary in orion/embodiment/skills.py (skill + categorical context) and, for a
  traverse on the overworld, a place key ``zelda:<level>:<room>:<dir>``.
- ``ZeldaExperience.prepare``: settles the previous goal's bet, asks for one quote over the
  options, and returns the top 5 as a menu for ``retro_planner.plan_zelda``. Any failure -> None
  and the log line ``action_quote_unavailable``: the planner then runs the plain #2870 prompt and
  no bet is placed. This module never invents a probability; ``p`` comes only from the service.
- ``ZeldaExperience.placed``: after Retro accepted the goal (POST /goal 200), places the bet.
- ``settle_body``: the settle table, from Retro's ``goal_finished`` row plus the next decision's
  observation. ``stopped:*`` is always void, never a failure.
- ``ZeldaExperience.end_session``: settles any bet still open when a session ends.

Value (``VALUE``) is the body's preference, hand-set for v1 and labelled as such; it is not a
prediction and is never sent to the service. Kill switch: HUB_ACTION_BETS_ENABLED (shipped on);
off -> no menu, no place, no settle (exactly the #2870 behaviour).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from orion.embodiment.skills import context as skill_context, count_band, health_band, threat_band
from orion.embodiment.transition import FEATURE_VERSION, zelda_facts, zelda_features
from orion.schemas.prediction import (ACTION_BET_MAX_OPTIONS, ActionBetOptionV1, ActionBetPlaceV1,
                                      ActionBetQuoteRequestV1, ActionBetQuoteV1, ActionBetSettleV1)

logger = logging.getLogger('orion-hub.retro.experience')

DOMAIN = 'retro:zelda'
BODY = 'retro'
NEAR_PX = 48            # services/orion-retro/app/zelda/navigate.py RETREAT_DISTANCE (Manhattan px)
MENU_SIZE = 5
TIMEOUT_SEC = 2.0
CARD_TIMEOUT_SEC = 10.0
OPEN_CAP = 200          # in-process open bets; the service expires anything dropped (never_settled)
EIG_WEIGHT = 2.0
DIRS = 'NSEW'
SKILL_OF = {'exit': 'traverse', 'explore': 'traverse', 'fight': 'engage', 'retreat': 'withdraw',
            'collect': 'acquire', 'enter': 'enter', 'wait': 'hold'}
SKILL_WORDS = {'traverse': 'moving', 'engage': 'fighting', 'withdraw': 'retreating',
               'acquire': 'picking up', 'enter': 'entering', 'hold': 'waiting'}

# Hand-set v1 value table (spec "Scoring"). Learning it is out of scope for v1. visited_room is
# divided by the room's visit count (room_value).
VALUE = {'new_room': 1.0, 'visited_room': 0.3, 'drop': 0.7, 'fight': 0.6,
         'withdraw_hurt': 0.4, 'withdraw': 0.1, 'cave': 0.5, 'hold': 0.05}

# Options the planner model sees. Live 2026-10-27: shown the full menu, the 8B model took the
# highest chance 10 of 10 times (a 115 <-> 99 round trip ranked 3rd); shown the top 2, it left by a
# less-seen side 10 of 10 times.
PLANNER_SHORTLIST = 2

# Prediction and budget for a menu pick by number (traverse is computed from the side's room).
DEFAULT_PLANS = {'engage': ({'kind': 'enemies_cleared'}, 20), 'acquire': ({'kind': 'item_gained'}, 10),
                 'enter': ({'kind': 'enter_cave'}, 10), 'withdraw': ({'kind': 'survive'}, 8),
                 'hold': ({'kind': 'survive'}, 5)}

Http = Callable[..., Awaitable[tuple[int, Any]]]


# --- options ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Option:
    option_id: str
    goal: dict
    skill: str
    context: dict
    place: Optional[str]
    value: float
    text: str
    target_room: Optional[int] = None   # traverse: the room on that side, when the atlas knows it

    @property
    def default_plan(self) -> tuple[dict, int]:
        """(prediction, budget_seconds) used when the model answers with just this option's number:
        the checkable claim this move makes, and a time box that fits it."""
        if self.skill == 'traverse':
            return ({'kind': 'reach_room', 'value': self.target_room} if isinstance(self.target_room, int)
                    else {'kind': 'unknown'}), 15
        if self.skill == 'engage' and self.context.get('threat') != 'one':
            # Retro ends a fight-nearest goal when that one enemy dies; "all cleared" would be
            # refuted almost every time with 2+ enemies (review 2026-10-24).
            return {'kind': 'survive'}, 20
        return DEFAULT_PLANS.get(self.skill, ({'kind': 'unknown'}, 10))

    def wire(self) -> ActionBetOptionV1:
        return ActionBetOptionV1(option_id=self.option_id, skill=self.skill, context=self.context,
                                 place=self.place)


def _xy(thing: Any) -> Optional[tuple[int, int]]:
    if isinstance(thing, dict) and all(isinstance(thing.get(k), int) and not isinstance(thing.get(k), bool)
                                       for k in ('x', 'y')):
        return thing['x'], thing['y']
    return None


def _enemy_distances(obs: dict) -> list[int]:
    link = _xy(obs.get('link'))
    if link is None:
        return []
    return [abs(e[0] - link[0]) + abs(e[1] - link[1])
            for e in (_xy(x) for x in obs.get('enemies') or [] if isinstance(x, dict)) if e]


def _enemies(obs: dict) -> list[dict]:
    return [e for e in obs.get('enemies') or [] if isinstance(e, dict)]


def _health(obs: dict) -> str:
    h = obs.get('hearts') if isinstance(obs.get('hearts'), dict) else {}
    return health_band(h.get('current'), h.get('containers'))


def _here(atlas: Optional[dict], obs: dict) -> Optional[dict]:
    """The atlas, only when it describes the room Link is standing in (never inside a cave: Retro
    reports the overworld room id there, but the room's sides are not the cave's)."""
    if (isinstance(atlas, dict) and obs.get('mode') == 'play' and atlas.get('room') == obs.get('room')
            and atlas.get('level') == obs.get('level')):
        return atlas
    return None


def side_state(decision: dict, atlas: Optional[dict], d: str) -> str:
    """open_unvisited | open_visited | closed | unknown for one screen edge.

    ``open_sides`` (Retro searched the map, #2870) decides open vs closed. Without it a side is
    ``unknown``, except an edge the atlas leaves out of ``adjacent`` (off the overworld grid),
    which is ``closed``."""
    obs = decision.get('observation') or {}
    here = _here(atlas, obs)
    adjacent = here.get('adjacent') if here and isinstance(here.get('adjacent'), dict) else None
    sides = decision.get('open_sides')
    a = (adjacent or {}).get(d)
    if isinstance(sides, list):
        if d not in sides:
            return 'closed'
        if not isinstance(a, dict):
            return 'unknown'
        return 'open_visited' if a.get('known') else 'open_unvisited'
    if adjacent is not None and not isinstance(a, dict):
        return 'closed'
    return 'unknown'


def _traverse_text(kind: str, d: str, side: str, a: Optional[dict], visits: Optional[int] = None) -> str:
    text = f'{kind} {d}'
    if isinstance(a, dict) and a.get('room') is not None:
        seen = 'never visited' if not a.get('known') else (
            f'visited {visits} time{"" if visits == 1 else "s"}' if isinstance(visits, int) else 'visited')
        text += f' -> room {a.get("room")} ({seen})'
    if side == 'closed':
        text += ' (no way out found on this side)'
    return text


def room_value(atlas: Optional[dict], level: Any, room: Optional[int]) -> float:
    """A known room's worth falls as 1/visits (count-based novelty; visits from the atlas map).
    Live 2026-10-27: with every known room worth the same 0.3, the round trip 115 <-> 99 (28 of 28
    met each way) outranked every other side, and Link crossed it 30 times."""
    visits = room_visits(atlas, level, room)
    return VALUE['visited_room'] / max(1, visits) if isinstance(visits, int) else VALUE['visited_room']


def room_visits(atlas: Optional[dict], level: Any, room: Optional[int]) -> Optional[int]:
    entry = ((atlas or {}).get('map') or {}).get(f'{level}:{room}') if room is not None else None
    visits = entry.get('visits') if isinstance(entry, dict) else None
    return visits if isinstance(visits, int) else None


def transition_features(decision: dict, goal: Any) -> dict:
    """The world-model inputs for this bet (orion/embodiment/transition.py), scored in orion-prediction's
    shadow. Empty when the observation is not a read play state: no features, no shadow, never a guess."""
    obs = decision.get('observation') or {}
    if obs.get('status') != 'observed' or not isinstance(goal, dict) or not (obs.get('link') or {}).get('tile'):
        return {}
    return {'features': zelda_features(zelda_facts(obs), goal, decision.get('open_sides')),
            'feature_version': FEATURE_VERSION}


def option_for_goal(goal: dict, decision: dict, atlas: Optional[dict]) -> Optional[Option]:
    """Skill, context, place and value for one GoalV2 ``goal`` in this decision. Used for every
    menu option and for an off-menu choice. None for a goal kind the body has no skill for."""
    if not isinstance(goal, dict) or goal.get('kind') not in SKILL_OF:
        return None
    obs = decision.get('observation') or {}
    kind, skill = goal['kind'], SKILL_OF[goal['kind']]
    health = _health(obs)
    threat = threat_band(_enemy_distances(obs), NEAR_PX)
    here = _here(atlas, obs)
    if skill == 'traverse':
        d = goal.get('dir')
        side = side_state(decision, atlas, d)
        a = ((here or {}).get('adjacent') or {}).get(d) if here else None
        level, room = obs.get('level'), obs.get('room')
        # Overworld only: dungeon levels decode as 'unknown' (one place key would mix every dungeon).
        place = (f'zelda:{level}:{room}:{d}' if level == 0 and isinstance(room, int) and obs.get('mode') == 'play'
                 else None)   # overworld rooms only: never a cave, which reports the room above it
        room = a.get('room') if isinstance(a, dict) and isinstance(a.get('room'), int) else None
        value = (room_value(atlas, level, room) if isinstance(a, dict) and a.get('known')
                 else VALUE['new_room'])
        return Option(f'{kind}:{d}', dict(goal), skill, skill_context(side=side, threat=threat, health=health),
                      place, value, _traverse_text(kind, d, side, a, room_visits(atlas, level, room)), room)
    if skill == 'engage':
        n = len(_enemies(obs))
        target = goal.get('target')
        return Option(f'fight:{target}', dict(goal), skill,
                      skill_context(threat=count_band(n) if n else 'none', health=health), None, VALUE['fight'],
                      f'fight {"the nearest enemy" if target == "nearest" else "enemy " + str(target)}'
                      f' ({n} in the room)')
    if skill == 'withdraw':
        return Option('retreat', dict(goal), skill, skill_context(threat=threat, health=health), None,
                      VALUE['withdraw_hurt'] if health in ('hurt', 'low') else VALUE['withdraw'],
                      'retreat from the nearest enemy')
    tile = goal.get('tile') if isinstance(goal.get('tile'), dict) else {}
    at = f'({tile.get("x")},{tile.get("y")})'
    if skill == 'acquire':
        return Option(f'collect:{tile.get("x")},{tile.get("y")}', dict(goal), skill,
                      skill_context(threat=threat, health=health), None, VALUE['drop'], f'collect the drop at {at}')
    if skill == 'enter':
        return Option(f'enter:{tile.get("x")},{tile.get("y")}', dict(goal), skill, skill_context(threat=threat),
                      None, VALUE['cave'], f'enter the cave at {at}')
    return Option('wait', dict(goal), skill, skill_context(threat=threat, health=health), None, VALUE['hold'], 'wait')


def build_options(decision: dict, atlas: Optional[dict]) -> list[Option]:
    """The spec's option table for one Retro decision (at most ACTION_BET_MAX_OPTIONS)."""
    from scripts.retro_planner import _room_caves, _tile_xy, tile_of
    obs = decision.get('observation') or {}
    allowed = set(decision.get('supported_goals') or SKILL_OF)
    sides = decision.get('open_sides')
    dirs = [d for d in DIRS if d in sides] if isinstance(sides, list) else list(DIRS)
    goals: list[dict] = []
    # One traverse option per side (exit and explore toward one side are the same bet): exit
    # where Retro found an opening, explore where nothing is known about the sides.
    for d in dirs:
        kind = next((k for k in (('exit', 'explore') if isinstance(sides, list) else ('explore', 'exit'))
                     if k in allowed), None)
        if kind:
            goals.append({'kind': kind, 'dir': d})
    if _enemies(obs) and 'fight' in allowed:
        goals.append({'kind': 'fight', 'target': 'nearest'})
    if 'collect' in allowed:
        for drop in obs.get('drops') or []:
            xy = tile_of(drop) if isinstance(drop, dict) else None
            if xy and {'kind': 'collect', 'tile': {'x': xy[0], 'y': xy[1]}} not in goals:
                goals.append({'kind': 'collect', 'tile': {'x': xy[0], 'y': xy[1]}})
    if 'enter' in allowed:
        for cave in _room_caves(_here(atlas, obs), obs.get('level'), obs.get('room')):
            xy = _tile_xy(cave.get('tile'))
            if xy:
                goals.append({'kind': 'enter', 'tile': {'x': xy[0], 'y': xy[1]}})
    if 'retreat' in allowed and threat_band(_enemy_distances(obs), NEAR_PX) == 'near':
        goals.append({'kind': 'retreat'})
    # Wait only when an enemy is close, Link cannot move, or nothing else is possible. Live
    # 2026-10-27: an empty room, sides unknown; "wait ... Predicts survive" read as the safe pick,
    # was met every time, and its chance climbed to 37 of 37 while Link stood still for five
    # minutes. A far enemy is no reason either: it keeps the wait met without ending the loop.
    stuck = not (obs.get('link') or {}).get('controllable', True)
    if 'wait' in allowed and (threat_band(_enemy_distances(obs), NEAR_PX) == 'near' or stuck or not goals):
        goals.append({'kind': 'wait'})
    out = [o for o in (option_for_goal(g, decision, atlas) for g in goals) if o is not None]
    if len(out) > ACTION_BET_MAX_OPTIONS:   # keep wait (always last when offered) inside the cap
        keep_last = out[-1].goal == {'kind': 'wait'}
        out = out[:ACTION_BET_MAX_OPTIONS - 1] + [out[-1]] if keep_last else out[:ACTION_BET_MAX_OPTIONS]
    return out


# --- scoring and the menu -------------------------------------------------------------------------

@dataclass(frozen=True)
class Entry:
    rank: int
    option: Option
    quote: ActionBetQuoteV1
    score: float

    @property
    def line(self) -> str:
        return menu_line(self.rank, self.option, self.quote)


@dataclass(frozen=True)
class Menu:
    entries: tuple

    def for_planner(self, limit: Optional[int] = PLANNER_SHORTLIST) -> list[dict]:
        """What retro_planner.plan_zelda takes: rank, goal, the prompt line, and the prediction and
        budget a pick by number uses. Only a shortlist of ``limit`` (None = all): the score decides
        what is in the running, the model picks between them and can still refuse them on facts.
        The shortlist keeps the best traverse and the best other move, so a way out is always shown
        (review: fight and retreat, never discounted by visits, could fill both slots)."""
        entries = list(self.entries)
        if limit is not None:
            moves = [e for e in entries if e.option.skill == 'traverse'][:1]
            others = [e for e in entries if e.option.skill != 'traverse'][:1]
            keep = moves + others if moves and others else entries[:limit]
            entries = sorted(keep + [e for e in entries if e not in keep][:max(0, limit - len(keep))],
                             key=lambda e: e.rank)
        out = []
        for e in entries:
            prediction, budget = e.option.default_plan
            out.append({'rank': e.rank, 'goal': e.option.goal, 'line': e.line,
                        'prediction': prediction, 'budget_seconds': budget})
        return out

    def rank_of(self, goal: Any) -> Optional[int]:
        return next((e.rank for e in self.entries if e.option.goal == goal), None)


def score(option: Option, quote: ActionBetQuoteV1) -> float:
    """value x (p + 2 x EIG): learning about an action is worth only as much as the action. An
    untried low-value action (wait) no longer outranks a likely exit on curiosity alone."""
    return option.value * (quote.p + EIG_WEIGHT * quote.eig_nats)


def build_menu(options: list[Option], quotes: dict[str, ActionBetQuoteV1], size: int = MENU_SIZE) -> Menu:
    scored = [(score(o, quotes[o.option_id]), i, o) for i, o in enumerate(options) if o.option_id in quotes]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return Menu(tuple(Entry(rank, o, quotes[o.option_id], s) for rank, (s, _, o) in enumerate(scored[:size], 1)))


def _n(x: float) -> str:
    """Counts above the place level weigh each place as one unit, so they can be fractional."""
    return f'{round(float(x), 1):g}'


def _counts(quote: ActionBetQuoteV1) -> dict[str, tuple[str, str]]:
    # A level with less than one unit of evidence ("0.5 of 0.5") reads as noise to the model: omit it.
    return {c.level: (_n(c.successes), _n(c.successes + c.failures)) for c in quote.counts
            if c.successes + c.failures >= 1}


def counts_phrase(skill: str, quote: ActionBetQuoteV1) -> str:
    c = _counts(quote)
    overall = c.get('skill')
    word = SKILL_WORDS.get(skill, skill)
    tail = f'{word} overall {overall[0]} of {overall[1]}' if overall else None
    if quote.level == 'place' and 'place' in c:
        return f'{c["place"][0]} of {c["place"][1]} here' + (f'; {tail}' if tail else '')
    if quote.level == 'situation' and 'situation' in c:
        return f'{c["situation"][0]} of {c["situation"][1]} in this situation' + (f'; {tail}' if tail else '')
    if tail:
        return tail
    return 'no tries yet'


def learning_word(eig_nats: float) -> str:
    # 0.193 nats at Beta(1,1) (nothing known); < 0.03 once a level has ~20 settled bets.
    return 'high' if eig_nats >= 0.1 else 'medium' if eig_nats >= 0.03 else 'low'


def menu_line(rank: int, option: Option, quote: ActionBetQuoteV1) -> str:
    prediction, budget = option.default_plan
    claim = prediction['kind'] + (f' {prediction["value"]}' if prediction.get('value') is not None else '')
    return (f'{rank}. {option.text}. Chance it works {quote.p:.2f} ({counts_phrase(option.skill, quote)}). '
            f'Learning value {learning_word(quote.eig_nats)}. Predicts {claim}, {budget} s.')


# --- settle rules ------------------------------------------------------------------------------------

def start_facts(observation: Any) -> dict:
    """What a settle rule compares against: where Link was and what he had when the goal was planned."""
    obs = observation if isinstance(observation, dict) else {}
    hearts = obs.get('hearts') if isinstance(obs.get('hearts'), dict) else {}
    items = obs.get('items') if isinstance(obs.get('items'), dict) else {}
    return {'level': obs.get('level'), 'room': obs.get('room'), 'mode': obs.get('mode'),
            'hearts': hearts.get('current'), 'containers': hearts.get('containers'),
            'items': {k: v for k, v in items.items() if isinstance(v, int) and not isinstance(v, bool)}}


def _observed(obs: Any) -> bool:
    return isinstance(obs, dict) and obs.get('status') == 'observed'


def _place_changed(start: dict, obs: dict) -> bool:
    # navigate.py changed_place(): room or level differs, or the goal started outside play.
    return (obs.get('room') != start.get('room') or obs.get('level') != start.get('level')
            or (start.get('mode') not in (None, 'play') and obs.get('mode') == 'play'))


def _gained(start: dict, obs: dict) -> bool:
    after = start_facts(obs)
    return (any(after['items'].get(k, 0) > v for k, v in start['items'].items())
            or any(k not in start['items'] and v > 0 for k, v in after['items'].items())
            or (isinstance(after['containers'], (int, float)) and isinstance(start.get('containers'), (int, float))
                and after['containers'] > start['containers'])
            # A heart drop raises only hearts.current (drops are 'contents unknown' to Retro).
            or (isinstance(after['hearts'], (int, float)) and isinstance(start.get('hearts'), (int, float))
                and after['hearts'] > start['hearts']))


def settle_body(skill: str, start: dict, row: Any, next_observation: Any) -> Optional[ActionBetSettleV1]:
    """The Zelda settle table. ``row`` is Retro's goal_finished; ``next_observation`` is the next
    decision's observation (None when there is none). None = cannot tell from these facts."""
    if not isinstance(row, dict) or not isinstance(row.get('outcome'), str) or not row['outcome']:
        return None
    outcome = row['outcome']
    prediction = row.get('prediction') if isinstance(row.get('prediction'), dict) else {}
    detail = {k: row.get(k) for k in ('decision_id', 'verdict', 'applied_frames', 'bumps') if k in row}
    def out(result, rule):
        return ActionBetSettleV1(result=result, outcome=outcome[:64], detail=detail | {'rule': rule})
    if outcome.startswith('stopped:'):
        return out('void', 'stopped_is_void')   # takeover, pause, lost sponsor: never a failure
    nxt = next_observation if _observed(next_observation) else None
    if skill == 'hold':
        if outcome == 'death':
            return out('failure', 'death')
        if nxt is None or not isinstance(nxt.get('hearts'), dict) or not isinstance(start.get('hearts'), (int, float)):
            return None
        lost = isinstance(nxt['hearts'].get('current'), (int, float)) and nxt['hearts']['current'] < start['hearts']
        return out('failure', 'heart_lost') if lost else out('success', 'no_heart_lost')
    if outcome == 'death':
        return out('failure', 'death')
    if skill == 'traverse':
        if outcome.startswith('interrupted:'):
            return out('failure', 'interrupted')
        if nxt is None:
            return None
        return out('success', 'room_changed') if _place_changed(start, nxt) else out('failure', 'same_room')
    if skill == 'engage':
        if outcome == 'completed' or (outcome == 'prediction_met' and prediction.get('kind') == 'enemies_cleared'):
            return out('success', 'completed')
        if outcome in ('budget_expired', 'interrupted:stuck'):
            return out('failure', outcome)
        return out('void', 'outcome_not_in_settle_table')
    if skill == 'withdraw':
        if outcome == 'completed':
            return out('success', 'completed')
        if outcome.startswith('interrupted:'):
            return out('failure', 'interrupted')
        return out('void', 'outcome_not_in_settle_table')
    if skill == 'acquire':
        if prediction.get('kind') == 'item_gained' and row.get('verdict') == 'met':
            return out('success', 'item_gained_met')
        if nxt is None:
            return None
        return out('success', 'item_count_up') if _gained(start, nxt) else out('failure', 'no_item')
    if skill == 'enter':
        # Retro grades enter_cave before 'completed', so a met enter_cave is the same fact.
        if outcome == 'completed' or (outcome == 'prediction_met' and prediction.get('kind') == 'enter_cave'):
            return out('success', 'completed')
        return out('failure', 'not_entered')
    return None


# --- the client ---------------------------------------------------------------------------------------

def bet_id_for(retro_session_id: str, decision_id: str) -> str:
    return f'{DOMAIN}:{retro_session_id}:{decision_id}'


@dataclass
class OpenBet:
    bet_id: str
    session_id: str
    decision_id: str
    skill: str
    start: dict
    pending: Optional[ActionBetSettleV1] = None   # computed but not yet accepted by the service


@dataclass
class ZeldaExperience:
    """One per co-play router or leisure run. Never raises out of a public coroutine."""
    enabled: bool = False
    base_url: str = 'http://127.0.0.1:8132'
    http: Optional[Http] = None            # tests: async (method, path, json=None, params=None) -> (status, data)
    timeout_sec: float = TIMEOUT_SEC
    open: dict = field(default_factory=dict)   # bet_id -> OpenBet

    @classmethod
    def from_config(cls, config) -> 'ZeldaExperience':
        # A config object without the key (test namespaces, older callers) is off; Settings ships it on.
        return cls(enabled=bool(getattr(config, 'HUB_ACTION_BETS_ENABLED', False)),
                   base_url=str(getattr(config, 'HUB_PREDICTION_URL', '') or 'http://127.0.0.1:8132'))

    async def _call(self, method: str, path: str, *, json: Any = None, params: Optional[dict] = None,
                    timeout: Optional[float] = None) -> tuple[int, Any]:
        limit = timeout or self.timeout_sec
        if self.http is not None:
            return await asyncio.wait_for(self.http(method, path, json=json, params=params), limit)
        import aiohttp
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=limit)) as s:
            async with s.request(method, self.base_url.rstrip('/') + path, json=json, params=params) as r:
                try:
                    data = await r.json(content_type=None)
                except Exception:  # noqa: BLE001
                    data = None
                return r.status, data

    async def get(self, path: str, params: Optional[dict] = None) -> tuple[int, Any]:
        """Read-only GET for the memory card's learned lines (session end, off the plan path, so it
        may take longer than a quote: /action-bets/live grades every settled bet). Raises on failure."""
        return await self._call('GET', path, params=params, timeout=CARD_TIMEOUT_SEC)

    # quote / menu ------------------------------------------------------------------------------------
    async def quote(self, options: list[Option]) -> Optional[dict[str, ActionBetQuoteV1]]:
        """option_id -> quote, or None (logged ``action_quote_unavailable``) on ANY failure."""
        if not options:
            return None
        try:
            body = ActionBetQuoteRequestV1(domain=DOMAIN, body=BODY, options=[o.wire() for o in options])
            status, data = await self._call('POST', '/action-bets/quote', json=body.model_dump(mode='json'))
            if status != 200:
                raise ValueError(f'status_{status}')
            rows = data.get('quotes') if isinstance(data, dict) else data
            if not isinstance(rows, list):
                raise ValueError('quotes_not_a_list')
            quotes = {}
            for row in rows:
                q = ActionBetQuoteV1.model_validate(row)
                quotes[q.option_id] = q
            missing = [o.option_id for o in options if o.option_id not in quotes]
            if missing:
                raise ValueError('missing_quotes:' + ','.join(missing[:3]))
            return quotes
        except Exception as exc:  # noqa: BLE001 -- the planner runs the plain prompt; no p is invented
            logger.warning('action_quote_unavailable reason=%s', f'{type(exc).__name__}:{exc}'[:200])
            return None

    async def prepare(self, retro_session_id: str, decision: Any, atlas: Optional[dict]) -> Optional[Menu]:
        """Settle the previous goal's bet, then quote this decision's options. None = no menu."""
        if not self.enabled or not isinstance(decision, dict) or not isinstance(decision.get('observation'), dict):
            return None
        try:
            await self.settle_previous(retro_session_id, decision)
            if decision['observation'].get('status') != 'observed':
                return None
            options = build_options(decision, atlas)
            quotes = await self.quote(options)
            return build_menu(options, quotes) if quotes else None
        except Exception:  # noqa: BLE001
            logger.warning('action_quote_unavailable reason=adapter_error', exc_info=True)
            return None

    # place ---------------------------------------------------------------------------------------------
    async def placed(self, retro_session_id: str, decision: dict, atlas: Optional[dict], goal: dict,
                     menu: Optional[Menu]) -> Optional[dict]:
        """Retro accepted ``goal`` (POST /goal 200): place its bet. Only when this decision was quoted
        (``menu`` set): an unreachable service means no menu and no bet."""
        if not self.enabled or menu is None:
            return None
        try:
            option = option_for_goal(goal.get('goal'), decision, atlas)
            if option is None:
                return None
            decision_id = str(goal.get('decision_id') or decision.get('decision_id'))
            # The planner ranks against the shortlist it saw; a pick outside it still has a full-menu rank.
            rank = goal.get('chosen_rank')
            if rank is None:
                rank = menu.rank_of(goal.get('goal'))
            bet = ActionBetPlaceV1(bet_id=bet_id_for(retro_session_id, decision_id), domain=DOMAIN, body=BODY,
                                   skill=option.skill, context=option.context, place=option.place,
                                   action={'goal': goal.get('goal'), 'prediction': goal.get('prediction'),
                                           'decision_id': decision_id, 'planner_id': goal.get('planner_id')},
                                   chosen_rank=rank, session_id=retro_session_id,
                                   **transition_features(decision, goal.get('goal')))
            status, data = await self._call('POST', '/action-bets', json=bet.model_dump(mode='json'))
            if status != 200:
                logger.warning('action_bet_place_failed bet_id=%s status=%s', bet.bet_id, status)
                return None
            while len(self.open) >= OPEN_CAP:          # e.g. a bet whose previous was never seen
                self.open.pop(next(iter(self.open)))
            self.open[bet.bet_id] = OpenBet(bet.bet_id, retro_session_id, decision_id, option.skill,
                                            start_facts(decision.get('observation')))
            logger.info('action_bet_placed bet_id=%s skill=%s place=%s chosen_rank=%s p=%s', bet.bet_id,
                        option.skill, option.place, rank, (data or {}).get('p') if isinstance(data, dict) else None)
            return data if isinstance(data, dict) else {}
        except Exception as exc:  # noqa: BLE001
            logger.warning('action_bet_place_failed reason=%s', f'{type(exc).__name__}:{exc}'[:200])
            return None

    # settle --------------------------------------------------------------------------------------------
    async def settle(self, bet_id: str, body: ActionBetSettleV1) -> bool:
        """True when the service holds a result for this bet (settled now, same result, or a conflict)."""
        try:
            status, _ = await self._call('POST', f'/action-bets/{bet_id}/settle', json=body.model_dump(mode='json'))
        except Exception as exc:  # noqa: BLE001
            logger.warning('action_bet_settle_failed bet_id=%s reason=%s', bet_id, type(exc).__name__)
            return False
        if status == 200:
            logger.info('action_bet_settled bet_id=%s result=%s outcome=%s', bet_id, body.result, body.outcome)
            return True
        if status in (404, 409):
            # 409: already settled differently (or expired never_settled); 404: unknown. Retrying won't help.
            logger.warning('action_bet_settle_refused bet_id=%s status=%s result=%s', bet_id, status, body.result)
            return True
        logger.warning('action_bet_settle_failed bet_id=%s status=%s', bet_id, status)
        return False

    async def _flush(self, record: OpenBet) -> None:
        if record.pending is not None and await self.settle(record.bet_id, record.pending):
            self.open.pop(record.bet_id, None)

    async def settle_previous(self, retro_session_id: str, decision: dict) -> None:
        """Retry missed settles of this session, then settle the bet for ``decision.previous``."""
        if not self.enabled:
            return
        previous = decision.get('previous') if isinstance(decision.get('previous'), dict) else None
        for record in [r for r in self.open.values() if r.session_id == retro_session_id]:
            if record.pending is None and previous and previous.get('decision_id') == record.decision_id:
                record.pending = settle_body(record.skill, record.start, previous, decision.get('observation'))
            await self._flush(record)

    async def end_session(self, retro_session_id: str, finished: Any, last_decision: Any = None) -> None:
        """Settle every bet of this session still open: from its goal_finished row when Retro showed
        one (the next observation only when ``last_decision`` follows that row), else void
        ``session_ended``. Never raises."""
        if not self.enabled:
            return
        try:
            rows = {r.get('decision_id'): r for r in (finished or []) if isinstance(r, dict)}
            last = last_decision if isinstance(last_decision, dict) else {}
            prev_id = (last.get('previous') or {}).get('decision_id') if isinstance(last.get('previous'), dict) else None
            for record in [r for r in self.open.values() if r.session_id == retro_session_id]:
                if record.pending is None:
                    row = rows.get(record.decision_id)
                    nxt = last.get('observation') if prev_id == record.decision_id else None
                    record.pending = (settle_body(record.skill, record.start, row, nxt) if row else None) or \
                        ActionBetSettleV1(result='void', outcome='session_ended',
                                          detail={'decision_id': record.decision_id,
                                                  'rule': 'finished_row_seen' if row else 'no_finished_row'})
                await self._flush(record)
                self.open.pop(record.bet_id, None)   # session over: the service expires anything left
        except Exception:  # noqa: BLE001
            logger.warning('action_bet_session_end_failed session_id=%s', retro_session_id, exc_info=True)

    async def learned_lines(self, retro_session_id: str, since: Optional[float] = None) -> list[str]:
        """The memory card's learned lines (scripts.retro_memory_export.fetch_learned_lines)."""
        if not self.enabled:
            return []
        from scripts.retro_memory_export import fetch_learned_lines
        return await fetch_learned_lines(retro_session_id, get=self.get, since=since)


__all__ = ['BODY', 'DOMAIN', 'Entry', 'Menu', 'NEAR_PX', 'OpenBet', 'Option', 'VALUE', 'ZeldaExperience',
           'bet_id_for', 'build_menu', 'build_options', 'counts_phrase', 'menu_line', 'option_for_goal',
           'score', 'settle_body', 'side_state', 'start_facts']
