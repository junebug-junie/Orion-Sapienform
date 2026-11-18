"""The replay evals call a model only through an operator lane (orion/evals/leased_llm.py): a failed
call is a row error, a recall stops the eval. No eval here names a model server's address."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from orion.evals.leased_llm import LaneCallError, LaneRecalled
from orion.harness.evals import judge_crash_replay_eval as crash
from orion.harness.evals import judge_obedience_replay_eval as obedience
from orion.harness.evals import perception_claims_replay_eval as perception
from orion.harness.evals import promise_ledger_replay_eval as ledger

HERE = Path(__file__).resolve().parent


class FakeLane:
    def __init__(self, *, content="{}", error=None):
        self.bodies: list[dict] = []
        self.content, self.error = content, error

    def chat(self, body, timeout_sec=None):
        self.bodies.append(body)
        if self.error is not None:
            raise self.error
        return {"choices": [{"message": {"content": self.content}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 3}}


def test_a_failed_call_is_a_row_error_not_a_crash():
    lane = FakeLane(error=LaneCallError("http 502 schema refused", 502))
    assert crash._call(lane, {"messages": []})["failure"].startswith("engine:http 502")
    payload, error, _ = ledger._call(lane, {"messages": []})
    assert payload is None and error.startswith("http 502")


def test_a_recall_propagates_so_the_eval_stops():
    lane = FakeLane(error=LaneRecalled("seat taken"))
    with pytest.raises(LaneRecalled):
        crash._call(lane, {"messages": []})
    with pytest.raises(LaneRecalled):
        ledger._call(lane, {"messages": []})
    with pytest.raises(LaneRecalled):
        obedience._chat(lane, "prompt")
    with pytest.raises(LaneRecalled):
        perception._chat(lane, "prompt", 10)


def test_a_good_call_is_parsed():
    lane = FakeLane(content=json.dumps({"promises_made": []}))
    payload, error, _ = ledger._call(lane, {"messages": []})
    assert payload == {"promises_made": []} and error is None
    assert obedience._chat(FakeLane(content="hi"), "p") == "hi"


def test_every_replay_eval_holds_its_seat_as_operator_work():
    for mod in (crash, obedience, perception, ledger):
        assert mod.HOLDER.startswith("operator:")


@pytest.mark.parametrize("name", ["judge_crash_replay_eval.py", "judge_obedience_replay_eval.py",
                                  "perception_claims_replay_eval.py", "promise_ledger_replay_eval.py"])
def test_no_replay_eval_names_a_model_server_or_posts_directly(name):
    src = (HERE / name).read_text()
    assert not re.search(r"https?://\d+\.\d+\.\d+\.\d+", src)
    assert "httpx.post" not in src and "urllib.request" not in src
