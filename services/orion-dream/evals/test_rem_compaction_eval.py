"""Phase F eval — REM compaction *behavior* over a corpus, not just unit paths.

Unit tests prove individual seams. This eval asserts the load-bearing safety
property holds in aggregate over a mixed corpus of requests: no matter what the
awake reverie asks, a Phase-F delta is always a proposal, never fabricates the
dangerous ops (downscale/prune), and reports honest deterministic metrics.
"""
from __future__ import annotations

import random

from app.rem_compaction import build_compaction_delta


def _corpus(n: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    hints = ["consolidate", "downscale", "prune", "consolidate", "consolidate"]
    out = []
    for i in range(n):
        out.append(
            {
                "request_id": f"r-{i}",
                "theme": f"loop:ol-{rng.randint(0, 20)}",
                "op_hint": rng.choice(hints),
                "evidence_refs": [f"ol-{rng.randint(0, 20)}" for _ in range(rng.randint(0, 6))],
            }
        )
    return out


def test_delta_is_always_a_proposal_over_corpus():
    for seed in range(20):
        delta = build_compaction_delta(_corpus(40, seed=seed))
        assert delta.proposal_marked is True
        # Phase F never fabricates the applier's dangerous ops from the awake path.
        assert delta.downscale == []
        assert delta.prune == []
        # Metrics are honest and deterministic: cards_out == consolidate entries.
        assert delta.metrics.cards_out == len(delta.consolidate)
        assert delta.metrics.edges_downscaled == 0
        assert delta.metrics.rows_pruned == 0


def test_only_consolidate_hints_become_cards():
    corpus = _corpus(60, seed=3)
    expected = sum(1 for r in corpus if r["op_hint"] == "consolidate" and r["theme"])
    delta = build_compaction_delta(corpus)
    assert delta.metrics.cards_out == expected


def test_every_card_is_non_empty_cognition():
    """No empty-shell gist cards (§0A) — every proposed card carries real text."""
    for seed in range(10):
        delta = build_compaction_delta(_corpus(30, seed=seed))
        for entry in delta.consolidate:
            assert entry.gist_card.strip(), "gist card must never be empty"
