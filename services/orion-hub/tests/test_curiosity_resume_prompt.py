"""Hub prepends what an earlier attempt already wrote when durable-runs
retries a curiosity turn under the same run_id (`Hop.n` collision fix).

Exercises `CuriosityInvestigation._prompt_for_attempt` with a stub `self`:
the method touches nothing on the instance but `_reader`, and building a
real `CuriosityInvestigation` here would drag in the whole Hub loop for a
three-branch decision.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from orion.curiosity.worldview import WorldviewReader
from orion.schemas.durable_run import CuriosityTurnRequestV1
from scripts.curiosity_investigation import CuriosityInvestigation


class _Reader(WorldviewReader):
    def __init__(self, rows) -> None:
        super().__init__(host="x", port=1, graph_name="g", client=object())
        self.rows = rows
        self.queries: list[str] = []

    def query(self, cypher: str):
        self.queries.append(cypher)
        return self.rows


def _request(attempt: int) -> CuriosityTurnRequestV1:
    return CuriosityTurnRequestV1(
        run_id="abc123",
        correlation_id="00000000-0000-0000-0000-000000000001",
        prompt="FROZEN KICKOFF PROMPT",
        timeout_sec=10.0,
        attempt=attempt,
    )


def _prompt_for(attempt: int, reader) -> tuple[str, object]:
    stub = SimpleNamespace(_reader=reader)
    text = asyncio.run(CuriosityInvestigation._prompt_for_attempt(stub, _request(attempt)))
    return text, reader


def test_first_attempt_sends_the_frozen_prompt_and_reads_nothing():
    reader = _Reader(rows=[{"n": 1, "note": "would be a bug to see this"}])
    text, reader = _prompt_for(1, reader)
    assert text == "FROZEN KICKOFF PROMPT"
    assert reader.queries == []


def test_retry_with_no_prior_hops_sends_the_frozen_prompt():
    text, reader = _prompt_for(2, _Reader(rows=[]))
    assert text == "FROZEN KICKOFF PROMPT"
    assert len(reader.queries) == 1


def test_retry_with_prior_hops_prepends_them_and_the_next_n():
    reader = _Reader(
        rows=[
            {"n": 2, "note": "second stop", "written_at": 1_789_000_000_002},
            {"n": 1, "note": "first stop", "written_at": 1_789_000_000_001},
        ]
    )
    text, _ = _prompt_for(3, reader)
    assert text.startswith("RESUMED SITTING.")
    assert text.index("n=1: first stop") < text.index("n=2: second stop")
    assert "Number your next hop n=3" in text
    assert text.endswith("FROZEN KICKOFF PROMPT")


def test_no_reader_configured_sends_the_frozen_prompt():
    stub = SimpleNamespace(_reader=None)
    text = asyncio.run(CuriosityInvestigation._prompt_for_attempt(stub, _request(2)))
    assert text == "FROZEN KICKOFF PROMPT"
