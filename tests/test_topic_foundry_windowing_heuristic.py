import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "services" / "orion-topic-foundry"))

from app.services.windowing import _heuristic_gate_score


def test_heuristic_gate_score_bounds():
    assert _heuristic_gate_score("") == 0.0
    score = _heuristic_gate_score("This is a test window with varied tokens and some repeated text.")
    assert 0.0 <= score <= 1.0


def test_heuristic_gate_score_prefers_longer_text():
    short = _heuristic_gate_score("Short text.")
    longer = _heuristic_gate_score(
        "This is a longer segment of text with multiple words and more variety to improve heuristic scoring."
    )
    assert longer >= short
