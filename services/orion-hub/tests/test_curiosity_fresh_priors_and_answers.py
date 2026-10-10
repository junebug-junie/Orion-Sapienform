"""A brief frozen at admission is corrected at turn start, a backlog stops new
admissions, and investigations can revise lived answers.

Live 2026-10-09: seven investigation briefs waited 8-12 h for a GPU hold and
were shown a prior at 0.55 / never tested after it had moved six times; run
334d78 recorded its revision "from 0.55". Same day, a lived answer recorded
`revises` against a 10-01 answer while a 10-04 one existed.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from orion.curiosity.kickoff_prompt import (
    build_kickoff_prompt,
    build_prior_drift_preamble,
    offered_priors_in_prompt,
)
from orion.curiosity.self_inquiry import (
    LivedAnswer,
    OpenLivedQuestion,
    build_lived_answer_history_write,
    read_current_lived_answers,
)
from orion.curiosity.study_material import StudyMaterial
from orion.curiosity.value import PriorState
from orion.curiosity.worldview import Prior, WorldviewReader, WorldviewSnapshot
from orion.schemas.durable_run import CuriosityTurnRequestV1
from scripts.curiosity_investigation import (
    QUEUE_BACKLOG_BLOCK_REASON,
    CuriosityInvestigation,
)
from datetime import datetime, timezone

PRIOR_ID = "prior.curation_gate_rejects_most"
RUN = "abc123def456"


class _Reader(WorldviewReader):
    def __init__(self, rows) -> None:
        super().__init__(host="x", port=1, graph_name="g", client=object())
        self.rows = rows
        self.queries: list[str] = []

    def query(self, cypher: str):
        self.queries.append(cypher)
        return self.rows(cypher) if callable(self.rows) else self.rows


def _stale_prior() -> Prior:
    return Prior(
        prior_id=PRIOR_ID, claim="The curation gate rejects most candidates.",
        confidence=0.55, status="open", times_tested=0,
    )


def _brief_prompt() -> str:
    view = WorldviewSnapshot(live_priors=[_stale_prior()], live_total=1)
    return build_kickoff_prompt(
        StudyMaterial(generated_at=datetime(2026, 10, 9, tzinfo=timezone.utc)),
        view=view, run_id=RUN, graph_enabled=True,
    )


def test_offered_priors_round_trip_preview():
    prior = Prior(prior_id="p.x", claim="c", confidence=0.7, status="open", times_tested=3)
    never = Prior(prior_id="p.y", claim="d", confidence=None, status="open", times_tested=0)
    text = prior.preview() + "\n" + never.preview()
    assert offered_priors_in_prompt(text) == {"p.x": (0.7, 3), "p.y": (None, 0)}


def test_attempt_prompt_shows_the_current_prior_value_not_the_frozen_one():
    prompt = _brief_prompt()
    assert "[confidence=0.55, never tested]" in prompt
    live_row = {"prior_id": PRIOR_ID, "confidence": 0.82, "times_tested": 4, "status": "supported"}
    reader = _Reader(rows=[live_row])
    request = CuriosityTurnRequestV1(
        run_id=RUN, correlation_id="00000000-0000-0000-0000-000000000001",
        prompt=prompt, timeout_sec=10.0, attempt=1,
    )
    text = asyncio.run(
        CuriosityInvestigation._prompt_for_attempt(SimpleNamespace(_reader=reader), request)
    )
    head = text[: text.index(prompt)]
    assert "THE NUMBERS BELOW ARE OLDER THAN THIS SITTING" in head
    assert f"prior_id: {PRIOR_ID}" in head
    assert "confidence 0.55 -> 0.82, tested 0 -> 4" in head
    assert text.endswith(prompt)


def test_unmoved_prior_adds_nothing():
    prompt = _brief_prompt()
    same = {PRIOR_ID: PriorState(prior_id=PRIOR_ID, confidence=0.551, times_tested=0, status="open")}
    assert build_prior_drift_preamble(prompt, same) == ""


def test_closed_and_vanished_priors_are_named():
    prompt = _stale_prior().preview() + "\n" + Prior(
        prior_id="p.gone", claim="x", confidence=0.4, status="open", times_tested=1
    ).preview()
    states = {PRIOR_ID: PriorState(prior_id=PRIOR_ID, confidence=0.55, times_tested=0, status="refuted")}
    text = build_prior_drift_preamble(prompt, states)
    assert "status now refuted" in text
    assert "prior_id: p.gone" in text and "no longer in your graph" in text


def test_unreadable_prior_state_sends_the_frozen_prompt():
    prompt = _brief_prompt()

    def boom(_):
        from orion.curiosity.worldview import WorldviewUnavailable
        raise WorldviewUnavailable("down")

    request = CuriosityTurnRequestV1(
        run_id=RUN, correlation_id="00000000-0000-0000-0000-000000000001",
        prompt=prompt, timeout_sec=10.0, attempt=1,
    )
    text = asyncio.run(
        CuriosityInvestigation._prompt_for_attempt(SimpleNamespace(_reader=_Reader(rows=boom)), request)
    )
    assert text == prompt


# --- the queued-backlog cap --------------------------------------------------


class _Conn:
    def __init__(self, value):
        self.value = value

    async def fetchval(self, sql, *args):
        if isinstance(self.value, Exception):
            raise self.value
        assert "durable_admission_runs" in sql and "run.admitted" in sql
        return self.value


class _Pool:
    def __init__(self, value):
        self.conn = _Conn(value)

    def acquire(self):
        pool = self

        class _Ctx:
            async def __aenter__(self):
                return pool.conn

            async def __aexit__(self, *a):
                return False

        return _Ctx()


def _cap_stub(queued, cap=1, admitted=True):
    stub = SimpleNamespace(
        max_queued_investigations=cap, kickoff_via_cortex=True,
        durable_admission_enabled=admitted, _pool_provider=lambda: _Pool(queued),
    )
    stub._queued_investigations = lambda: CuriosityInvestigation._queued_investigations(stub)
    return stub


def _blocks(stub) -> bool:
    return asyncio.run(CuriosityInvestigation._queue_backlog_blocks(stub))


def test_backlog_at_cap_blocks_admission():
    assert _blocks(_cap_stub(1)) is True
    assert QUEUE_BACKLOG_BLOCK_REASON == "queue_backlog"


def test_backlog_under_cap_admits():
    assert _blocks(_cap_stub(0)) is False


def test_cap_zero_or_non_admitted_path_never_blocks():
    assert _blocks(_cap_stub(9, cap=0)) is False
    assert _blocks(_cap_stub(9, admitted=False)) is False


def test_unreadable_backlog_fails_open():
    assert _blocks(_cap_stub(RuntimeError("no table"))) is False


# --- lived answers ------------------------------------------------------------


def _row(run_id, written_at, text="t", qid="lived.her_team_unnamed"):
    return {"run_id": run_id, "question_id": qid, "family": "lived", "text": text,
            "evidence": [], "revises": "", "written_at": written_at}


def test_current_answer_is_the_newest_not_the_oldest():
    # The live 10-09 shape: 453e (10-01, mirrored) and e511 (10-04, draft).
    reader = _Reader(rows=[_row("453e5abdef75", 1790851851768), _row("e511d15b9534", 1791122694500)])
    current = read_current_lived_answers(reader, ["lived.her_team_unnamed"])
    assert current["lived.her_team_unnamed"].run_id == "e511d15b9534"


def test_current_answer_excludes_this_run():
    reader = _Reader(rows=[_row("e511d15b9534", 1791122694500), _row(RUN, 1791544310739)])
    current = read_current_lived_answers(reader, ["lived.her_team_unnamed"], exclude_run_id=RUN)
    assert current["lived.her_team_unnamed"].run_id == "e511d15b9534"


def test_previous_lived_reads_the_graph_newest_for_revises():
    reader = _Reader(rows=[_row("453e5abdef75", 1790851851768, "old"),
                           _row("e511d15b9534", 1791122694500, "newer draft")])
    stub = SimpleNamespace(_reader=reader)

    async def _mirror(_qid):
        raise AssertionError("graph answered; the mirror must not be consulted")

    stub._read_previous_lived_mirror = _mirror
    prev = asyncio.run(
        CuriosityInvestigation._read_previous_lived(stub, "lived.her_team_unnamed", run_id=RUN)
    )
    assert prev.run_id == "e511d15b9534" and prev.content == "newer draft"


def test_kickoff_offers_open_lived_questions_with_prefilled_revises():
    q = OpenLivedQuestion(
        question_id="lived.her_team_unnamed", text="Who is on her team?",
        current=LivedAnswer("e511d15b9534", "lived.her_team_unnamed", "lived", "Unnamed so far.",
                            [], "", 1791122694500),
    )
    text = build_kickoff_prompt(
        StudyMaterial(generated_at=datetime(2026, 10, 9, tzinfo=timezone.utc)),
        view=WorldviewSnapshot(), run_id=RUN, graph_enabled=True, open_lived_questions=[q],
    )
    assert "QUESTIONS YOU KEEP ABOUT YOUR OWN LIFE" in text
    assert "current answer (run e511d15b9534, 2026-10-04): Unnamed so far." in text
    assert (f'MERGE (a:LivedAnswer {{run_id: "{RUN}", question_id: "lived.her_team_unnamed"}})'
            in text)
    assert 'a.revises = "e511d15b9534"' in text


def test_kickoff_omits_lived_questions_when_graph_unwritable():
    q = OpenLivedQuestion(question_id="lived.x", text="Q?")
    text = build_kickoff_prompt(
        StudyMaterial(generated_at=datetime(2026, 10, 9, tzinfo=timezone.utc)),
        view=WorldviewSnapshot(unavailable_reason="down"), run_id=RUN, graph_enabled=True,
        open_lived_questions=[q],
    )
    assert "QUESTIONS YOU KEEP ABOUT YOUR OWN LIFE" not in text


def test_investigation_mirror_entry_ids_do_not_collide_across_questions():
    a = LivedAnswer(RUN, "lived.a", "lived", "x", ["journal:1"], "", 1)
    b = LivedAnswer(RUN, "lived.b", "lived", "y", ["journal:2"], "", 1)
    ids = {build_lived_answer_history_write(x, per_question=True).entry_id for x in (a, b)}
    assert ids == {f"self-lived:{RUN}:lived.a", f"self-lived:{RUN}:lived.b"}
    # A self-inquiry run keeps its historical id.
    assert build_lived_answer_history_write(a).entry_id == f"self-lived:{RUN}"
