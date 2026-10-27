"""Action-conditioned transition model: "if the body does this here, what happens?"

The parked WorldModel (app/model.py) has no action input, so it cannot answer that question.
This small model can. It is trained on dojo transitions (services/orion-retro/evals/zelda_dojo.py).

Inputs are body-level facts, never game RAM: the skill (orion/embodiment/skills.py vocabulary),
a direction, the open sides, health, threat count and nearest-threat distance, items held.
Outputs are what happened, each a probability: the place changed, health was lost, an item was
gained, the body died; plus log duration. A game adapter (here: zelda_features) maps one game's
facts into these inputs; a second game or a real motor needs only its own adapter.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from orion.embodiment.skills import SKILLS   # traverse, engage, withdraw, acquire, enter, hold

SKILL_OF = {'exit': 'traverse', 'explore': 'traverse', 'fight': 'engage', 'collect': 'acquire',
            'retreat': 'withdraw', 'wait': 'hold', 'enter': 'enter'}
DIRS = 'NSWE'
HEADS = ('place_changed', 'health_lost', 'item_gained', 'died')
N_FEATURES = len(SKILLS) + len(DIRS) + len(DIRS) + 8


def zelda_features(before: dict, goal: dict, open_sides: Any) -> list[float]:
    """Zelda adapter: dojo/Retro facts -> the body-level input vector."""
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


class TransitionModel(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.events = nn.Linear(hidden, len(HEADS))      # logits, one per HEADS entry
        self.duration = nn.Linear(hidden, 1)             # log1p(frames) / 8

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.events(h), self.duration(h).squeeze(-1)

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits, duration = self(x)
        p = torch.sigmoid(logits)
        return {**{name: p[:, i] for i, name in enumerate(HEADS)},
                'frames': torch.expm1(duration.clamp(min=0) * 8)}


def loss(model: TransitionModel, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits, duration = model(x)
    return (nn.functional.binary_cross_entropy_with_logits(logits, y[:, :len(HEADS)])
            + nn.functional.mse_loss(duration, y[:, len(HEADS)]))
