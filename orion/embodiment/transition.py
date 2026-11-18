"""Body-level transition features: "if the body does this here, what happens?" (shared contract).

One definition, used by every side so they cannot drift:
- the dojo (services/orion-retro/evals/zelda_dojo.py) writes facts with ``zelda_facts``;
- the trainer (services/orion-world-model) turns them into inputs and labels;
- the Hub sends ``zelda_features`` with each action bet;
- orion-prediction runs the exported weights on those numbers (``TransitionWeights``, numpy only).

Inputs are body facts, never game RAM: skill (orion/embodiment/skills.py), direction, open sides,
health, threats, items, cave / dungeon. Outputs (HEADS) are what happened, labelled with the same
settle rules the Hub's action bets use (start_facts, place_changed, gained live here and the Hub
imports them). ``success_p`` maps them onto each skill's success; skills whose success the heads do
not describe get None, never a guess. numpy is imported only to run weights (the dojo has none).
"""
from __future__ import annotations

import math
from typing import Any

from orion.embodiment.skills import SKILLS   # traverse, engage, withdraw, acquire, enter, hold

FEATURE_VERSION = 'zelda-body-v1'
SKILL_OF = {'exit': 'traverse', 'explore': 'traverse', 'fight': 'engage', 'collect': 'acquire',
            'retreat': 'withdraw', 'wait': 'hold', 'enter': 'enter'}
DIRS = 'NSWE'
HEADS = ('place_changed', 'health_lost', 'item_gained', 'died')
N_FEATURES = len(SKILLS) + len(DIRS) + len(DIRS) + 8


def zelda_facts(obs: dict) -> dict:
    """Retro's zelda-1-ram-v1 observation (or its offered view) -> the facts the features read."""
    return dict(mode=obs.get('mode'), level=obs.get('level'), room=obs.get('room'),
                tile=(obs.get('link') or {}).get('tile'), hearts=obs.get('hearts') or {}, items=obs.get('items') or {},
                enemies=[dict(type=e.get('type'), tile=e.get('tile')) for e in obs.get('enemies') or []],
                drops=[d.get('tile') for d in obs.get('drops') or []])


def zelda_features(before: dict, goal: dict, open_sides: Any) -> list[float]:
    """Zelda adapter: facts + goal + the offer's open sides -> the body-level input vector."""
    skill = SKILL_OF.get(goal.get('kind'), 'hold')
    d = goal.get('dir')
    hearts = before.get('hearts') or {}
    health = (hearts.get('current') or 0) / max(1, hearts.get('containers') or 1)
    tile = before.get('tile') or [15, 11]
    enemies = before.get('enemies') or []
    dists = [abs(e['tile'][0] - tile[0]) + abs(e['tile'][1] - tile[1]) for e in enemies if e.get('tile')]
    items = sum(v for v in (before.get('items') or {}).values() if isinstance(v, int))
    sides = open_sides if isinstance(open_sides, list) else []
    return ([float(skill == s) for s in SKILLS] + [float(d == x) for x in DIRS]
            + [float(x in sides) for x in DIRS]
            + [health, float(health < 0.5), min(len(enemies), 6) / 6,
               (min(dists) / 40 if dists else 1.0), math.log1p(items) / 5, float(open_sides is None),
               float(before.get('mode') == 'cave'), float(before.get('level') not in (0, None))])


# --- the Hub's settle rules (services/orion-hub/scripts/retro_experience.py settle_body uses these) ---
def start_facts(obs: Any) -> dict:
    """What a settle rule compares against: where Link was and what he had. Takes a Retro observation
    or zelda_facts (same keys)."""
    obs = obs if isinstance(obs, dict) else {}
    hearts = obs.get('hearts') if isinstance(obs.get('hearts'), dict) else {}
    items = obs.get('items') if isinstance(obs.get('items'), dict) else {}
    return {'level': obs.get('level'), 'room': obs.get('room'), 'mode': obs.get('mode'),
            'hearts': hearts.get('current'), 'containers': hearts.get('containers'),
            'items': {k: v for k, v in items.items() if isinstance(v, int) and not isinstance(v, bool)}}


def place_changed(start: dict, obs: dict) -> bool:
    """navigate.py changed_place(): room or level differs, or the goal started outside play."""
    return (obs.get('room') != start.get('room') or obs.get('level') != start.get('level')
            or (start.get('mode') not in (None, 'play') and obs.get('mode') == 'play'))


def gained(start: dict, obs: dict) -> bool:
    """An item count went up, a new item appeared, a heart container was added, or hearts rose
    (a heart drop raises only hearts.current)."""
    after = start_facts(obs)
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    return (any(after['items'].get(k, 0) > v for k, v in start['items'].items())
            or any(k not in start['items'] and v > 0 for k, v in after['items'].items())
            or (num(after['containers']) and num(start.get('containers')) and after['containers'] > start['containers'])
            or (num(after['hearts']) and num(start.get('hearts')) and after['hearts'] > start['hearts']))


def zelda_targets(row: dict) -> list[float] | None:
    """What happened, from one dojo row, labelled with the Hub's settle rules so a head means what a
    bet settles on. None when the row says nothing (no after state)."""
    before, after = row.get('before') or {}, row.get('after')
    if not after:
        return None
    start, outcome = start_facts(before), str(row.get('outcome') or '')
    died = bool(row.get('died')) or outcome == 'death'
    hb, ha = start['hearts'], start_facts(after)['hearts']
    lost = died or (isinstance(ha, (int, float)) and isinstance(hb, (int, float)) and ha < hb)
    # traverse settles failure on death and on an interruption, whatever the room did
    moved = place_changed(start, after) and not died and not outcome.startswith('interrupted:')
    return [float(moved), float(lost), float(gained(start, after)), float(died),
            math.log1p(row.get('frames') or 0) / 8]


def success_p(skill: str, heads: dict[str, float]) -> float | None:
    """The chance the skill's action bet settles success (heads are labelled with settle_body's rules):
    traverse when the place changes (not by dying, not interrupted); acquire when an item is gained;
    hold when no heart is lost (a death loses them all). engage, withdraw and enter settle on the
    motor's own 'completed', which no head describes: None."""
    if skill == 'traverse':
        return heads['place_changed']
    if skill == 'acquire':
        return heads['item_gained']
    if skill == 'hold':
        return 1 - heads['health_lost']
    return None


def _gelu(x):
    import numpy as np
    return 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


class TransitionWeights:
    """The trained TransitionModel's MLP, exported to numpy (services/orion-world-model/scripts/
    train_transition.py writes the .npz): body Linear-GELU-Linear-GELU, then an events Linear."""

    def __init__(self, arrays: dict):
        import numpy as np   # only weight loading needs numpy: the dojo's retro image has none
        self.w = {k: np.asarray(v, dtype=np.float64) for k, v in arrays.items() if k.startswith(('body', 'events'))}
        self.version = str(arrays.get('feature_version', ''))
        self.tag = str(arrays.get('tag', ''))
        if self.w['body.0.weight'].shape[1] != N_FEATURES:
            raise ValueError(f"weights expect {self.w['body.0.weight'].shape[1]} features, contract has {N_FEATURES}")

    @classmethod
    def load(cls, path) -> 'TransitionWeights':
        import numpy as np
        with np.load(path, allow_pickle=False) as z:
            return cls({k: z[k] for k in z.files})

    def heads(self, x: list[float]) -> dict[str, float]:
        import numpy as np
        h = np.asarray(x, dtype=np.float64)
        h = _gelu(self.w['body.0.weight'] @ h + self.w['body.0.bias'])
        h = _gelu(self.w['body.2.weight'] @ h + self.w['body.2.bias'])
        logits = self.w['events.weight'] @ h + self.w['events.bias']
        p = 1 / (1 + np.exp(-logits))
        return {name: float(p[i]) for i, name in enumerate(HEADS)}
