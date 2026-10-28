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

import torch
from torch import nn

from orion.embodiment.transition import (  # noqa: F401  (re-exported for the trainer and tests)
    DIRS, FEATURE_VERSION, HEADS, N_FEATURES, SKILL_OF, zelda_facts, zelda_features, zelda_targets)


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
