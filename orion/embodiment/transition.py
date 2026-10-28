"""Body-level transition features: "if the body does this here, what happens?" (shared contract).

One definition, used by every side so they cannot drift:
- the dojo (services/orion-retro/evals/zelda_dojo.py) writes facts with ``zelda_facts``;
- the trainer (services/orion-world-model) turns them into inputs and labels;
- the Hub sends ``zelda_features`` with each action bet;
- orion-prediction runs the exported weights on those numbers (``TransitionWeights``, numpy only).

Inputs are body facts, never game RAM: skill (orion/embodiment/skills.py), direction, open sides,
health, threats, items, cave / dungeon. Outputs (HEADS) are what happened. ``success_p`` maps them onto
the success each skill's action bet settles on; skills whose success the heads do not describe get
None, never a guess.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

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


def zelda_targets(row: dict) -> list[float] | None:
    """What happened, from one dojo row; None when the row says nothing (no after state)."""
    before, after = row.get('before') or {}, row.get('after')
    if not after:
        return None
    hb, ha = (before.get('hearts') or {}).get('current', 0), (after.get('hearts') or {}).get('current', 0)
    ib = {k: v for k, v in (before.get('items') or {}).items() if isinstance(v, int)}
    ia = {k: v for k, v in (after.get('items') or {}).items() if isinstance(v, int)}
    gained = any(ia[k] > ib[k] for k in ib.keys() & ia.keys())     # only items read as numbers both times
    place = lambda f: (f.get('mode'), f.get('level'), f.get('room'))   # leaving a cave keeps the room id
    return [float(place(after) != place(before)),
            float(ha < hb), float(gained), float(bool(row.get('died'))),
            math.log1p(row.get('frames') or 0) / 8]


def success_p(skill: str, heads: dict[str, float]) -> float | None:
    """The chance the skill's action bet settles success, from the heads (Hub settle_body rules):
    traverse/enter succeed when the place changes; acquire when an item is gained; hold when no heart
    is lost and the body lives. engage/withdraw settle on the motor's own 'completed', which no head
    describes: None."""
    if skill in ('traverse', 'enter'):
        return heads['place_changed']
    if skill == 'acquire':
        return heads['item_gained']
    if skill == 'hold':
        return (1 - heads['health_lost']) * (1 - heads['died'])
    return None


_erf = np.vectorize(math.erf)


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + _erf(x / math.sqrt(2)))


class TransitionWeights:
    """The trained TransitionModel's MLP, exported to numpy (services/orion-world-model/scripts/
    train_transition.py writes the .npz): body Linear-GELU-Linear-GELU, then an events Linear."""

    def __init__(self, arrays: dict[str, np.ndarray]):
        self.w = {k: np.asarray(v, dtype=np.float64) for k, v in arrays.items() if k.startswith(('body', 'events'))}
        self.version = str(arrays.get('feature_version', ''))
        self.tag = str(arrays.get('tag', ''))
        if self.w['body.0.weight'].shape[1] != N_FEATURES:
            raise ValueError(f"weights expect {self.w['body.0.weight'].shape[1]} features, contract has {N_FEATURES}")

    @classmethod
    def load(cls, path) -> 'TransitionWeights':
        with np.load(path, allow_pickle=False) as z:
            return cls({k: z[k] for k in z.files})

    def heads(self, x: list[float]) -> dict[str, float]:
        h = np.asarray(x, dtype=np.float64)
        h = _gelu(self.w['body.0.weight'] @ h + self.w['body.0.bias'])
        h = _gelu(self.w['body.2.weight'] @ h + self.w['body.2.bias'])
        logits = self.w['events.weight'] @ h + self.w['events.bias']
        p = 1 / (1 + np.exp(-logits))
        return {name: float(p[i]) for i, name in enumerate(HEADS)}
