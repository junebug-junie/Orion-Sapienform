"""Pure parts of run_clothing_clue_eval.py, and the committed text side is current."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_clothing_clue_eval as ev  # noqa: E402


def test_auc_counts_ties_half_and_handles_empty_sides() -> None:
    assert ev.auc([2.0, 3.0], [1.0]) == 1.0
    assert ev.auc([1.0], [1.0]) == 0.5
    assert ev.auc([0.0], [1.0, 2.0]) == 0.0
    assert ev.auc([], [1.0]) is None


def test_passage_scores_average_frames_of_one_walk_past_against_the_neutral_prompt() -> None:
    neutral = np.array([1.0, 0.0])
    phrases = {"x": np.array([0.0, 1.0])}
    got = ev.passage_scores([("p1", np.array([0.0, 1.0])), ("p1", np.array([1.0, 0.0])),
                             ("p2", np.array([1.0, 0.0]))], neutral, phrases)
    assert sorted(got["x"]) == [-1.0, 0.0]          # p1 mean of (+1, -1); p2 -1


def test_committed_text_vectors_match_the_phrase_list() -> None:
    model_id, neutral, phrases = ev.load_text()
    assert model_id == "google/siglip-so400m-patch14-384" and neutral.shape == (1152,)
    assert set(phrases) == set(__import__("json").loads(ev.PHRASES.read_text())["phrases"])
