"""evals/run_current_turn_disclosure_live_eval.py calls the model only through an operator lane
(orion/evals/leased_llm.py): no model server URL, and a recall stops the run (exit 3, no result)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parents[1] / "evals" / "run_current_turn_disclosure_live_eval.py"


def _load():
    spec = importlib.util.spec_from_file_location("disclosure_live_eval", EVAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Lane:
    lane = "chat"

    def __init__(self, error=None):
        self.error, self.bodies = error, []

    def chat(self, body, timeout_sec=None):
        self.bodies.append(body)
        if self.error:
            raise self.error
        return {"choices": [{"message": {"content": '{"signals": [], "wants_direct_answer": true}'}}]}


def test_the_eval_names_no_model_server_url():
    src = EVAL.read_text()
    assert "urllib.request" not in src and "--url" not in src


def test_a_call_goes_through_the_lane_with_thinking_off():
    mod = _load()
    lane = _Lane()
    raw, _ = mod._call(lane, "prompt")
    assert raw.startswith("{") and lane.bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "model" not in lane.bodies[0]   # the lane names its operator route itself


def test_a_recall_stops_the_run_instead_of_counting_a_failure():
    from orion.evals.leased_llm import LaneRecalled

    mod = _load()
    with pytest.raises(LaneRecalled):
        mod._run(_Lane(error=LaneRecalled("seat taken")), 1)
