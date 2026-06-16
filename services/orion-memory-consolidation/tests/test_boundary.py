import importlib.util
import sys
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load(rel_path: str, name: str):
    for key in list(sys.modules):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    sys.path.insert(0, str(SERVICE_ROOT))
    path = SERVICE_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


boundary = _load("app/boundary.py", "memory_consolidation_boundary")
should_close_window = boundary.should_close_window
scores_from_llm_result = boundary.scores_from_llm_result

from orion.schemas.memory_consolidation import MemoryTurnPersistedV1


class _Settings:
    MEMORY_BOUNDARY_SCORE_THRESHOLD = 0.70
    MEMORY_BOUNDARY_LLM_ONLY_THRESHOLD = 0.85
    MEMORY_BOUNDARY_OVERRIDE_THRESHOLD = 0.92


@pytest.mark.parametrize(
    "phase,bnd,expected",
    [
        ("long_gap", 0.75, True),
        ("next_day", 0.70, True),
        ("stale_thread", 0.80, True),
        ("long_gap", 0.50, False),
        ("unknown", 0.90, True),
        ("unknown", 0.80, False),
        ("same_breath", 0.95, True),
        ("short_pause", 0.92, True),
        ("resumed_thread", 0.91, False),
    ],
)
def test_should_close_window_matrix(phase, bnd, expected):
    turn = MemoryTurnPersistedV1(
        correlation_id="c1",
        prompt="p",
        response="r",
        spark_meta={"conversation_phase": {"phase_change": phase}},
    )
    scores = {"conversation_boundary_score": bnd}
    assert should_close_window(turn, scores, _Settings()) is expected


def test_scores_from_llm_result_uses_logprobs():
    content = "MEMORY: YES\nBOUNDARY: NO\n"
    raw = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {"token": "MEMORY:", "logprob": -0.1},
                        {"token": "YES", "logprob": -0.2, "top_logprobs": [{"token": "YES", "logprob": -0.2}, {"token": "NO", "logprob": -2.0}]},
                        {"token": "BOUNDARY:", "logprob": -0.1},
                        {"token": "NO", "logprob": -0.3, "top_logprobs": [{"token": "NO", "logprob": -0.3}, {"token": "YES", "logprob": -1.5}]},
                    ]
                }
            }
        ]
    }
    mem, bnd = scores_from_llm_result(content, raw)
    assert mem is not None
    assert bnd is not None
    assert mem > 0.5
    assert bnd < 0.5
