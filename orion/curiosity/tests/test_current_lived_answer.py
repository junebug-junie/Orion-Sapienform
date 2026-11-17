"""#3396 Patch F: one rule for "Orion's current answer"."""
from __future__ import annotations

from orion.curiosity.self_inquiry import LivedAnswer, newest_answer_per_question, read_current_lived_answers
from orion.world_pulse_read import resonance_records as rr


def _a(run: str, at: int | None, *, ev: bool = True, q: str = "lived.q1") -> LivedAnswer:
    return LivedAnswer(run_id=run, question_id=q, family="lived", text=f"answer {run}",
                       evidence=["row:1"] if ev else [], written_at=at)


def test_newest_wins_and_history_keeps_both():
    refuted, corrected = _a("aaaaaaaaaaaa", 100), _a("bbbbbbbbbbbb", 200)
    assert newest_answer_per_question([refuted, corrected])["lived.q1"] is corrected
    assert newest_answer_per_question([corrected, refuted], require_evidence=True)["lived.q1"] is corrected


def test_equal_timestamps_break_on_run_id_every_time():
    x, y = _a("aaaaaaaaaaaa", 100), _a("bbbbbbbbbbbb", 100)
    for order in ([x, y], [y, x]):
        assert newest_answer_per_question(order)["lived.q1"] is y


def test_missing_written_at_sorts_as_oldest():
    undated, dated = _a("ffffffffffff", None), _a("aaaaaaaaaaaa", 1)
    assert newest_answer_per_question([undated, dated])["lived.q1"] is dated


def test_evidence_rule_skips_a_newer_draft_and_has_nothing_when_only_drafts():
    good, draft = _a("aaaaaaaaaaaa", 100), _a("bbbbbbbbbbbb", 200, ev=False)
    assert newest_answer_per_question([good, draft])["lived.q1"] is draft  # prompt rule
    assert newest_answer_per_question([good, draft], require_evidence=True)["lived.q1"] is good
    only = _a("cccccccccccc", 50, ev=False, q="lived.q2")
    assert "lived.q2" not in newest_answer_per_question([only], require_evidence=True)
    assert newest_answer_per_question([only])["lived.q2"] is only


class _Reader:
    def __init__(self, rows):
        self.rows = rows

    def query(self, cypher):
        return self.rows


def test_prompt_reader_keeps_newest_of_any_kind():
    rows = [
        {"run_id": "aaaaaaaaaaaa", "question_id": "lived.q1", "text": "old", "evidence": ["r"], "written_at": 100},
        {"run_id": "bbbbbbbbbbbb", "question_id": "lived.q1", "text": "draft", "evidence": [], "written_at": 200},
    ]
    assert read_current_lived_answers(_Reader(rows), ["lived.q1"])["lived.q1"].text == "draft"
    out = read_current_lived_answers(_Reader(rows), ["lived.q1"], exclude_run_id="bbbbbbbbbbbb")
    assert out["lived.q1"].text == "old"


class _Falkor:
    """Rows shaped like redis-cli GRAPH.RO_QUERY replies (lists come back as strings)."""

    def rows(self, graph, cypher, **params):
        if "MATCH (a:LivedAnswer)" in cypher:
            return [
                {"nid": 3, "run_id": "cccccccccccc", "qid": "lived.q1", "text": "draft", "revises": "",
                 "evidence": "[]", "at": "300"},
                {"nid": 2, "run_id": "bbbbbbbbbbbb", "qid": "lived.q1", "text": "corrected", "revises": "x",
                 "evidence": "[row:2, q:3]", "at": "200"},
                {"nid": 1, "run_id": "aaaaaaaaaaaa", "qid": "lived.q1", "text": "refuted", "revises": "",
                 "evidence": "[row:1]", "at": "100"},
            ]
        if "SelfQuestionMint" in cypher:
            return [{"qid": "lived.q1", "text": "what am I?"}]
        return []


def test_resonance_pool_offers_only_the_current_answer():
    docs, pairs, _ = rr.worldview_pool(_Falkor())
    lived = [d for d in docs if d.ref_kind == "lived_answer"]
    assert [d.ref_id for d in lived] == ["lived.q1@bbbbbbbbbbbb"]
    prev = [n for n in lived[0].neighborhood if n.relation == "previous_answer"]
    assert [n.ref_id for n in prev] == ["lived.q1@aaaaaaaaaaaa"]
    # History stays available to callers that want it.
    assert len(pairs["answers"]) == 3


class _FalkorNullFirst(_Falkor):
    """FalkorDB returns a null written_at FIRST under ORDER BY ... DESC."""

    def rows(self, graph, cypher, **params):
        rows = super().rows(graph, cypher, **params)
        if "MATCH (a:LivedAnswer)" in cypher:
            undated = {"nid": 9, "run_id": "dddddddddddd", "qid": "lived.q1", "text": "undated", "revises": "",
                       "evidence": "[row:9]", "at": None}
            return [undated] + rows
        return rows


def test_previous_answer_uses_the_helper_order_not_falkor_null_first():
    docs, _, _ = rr.worldview_pool(_FalkorNullFirst())
    lived = [d for d in docs if d.ref_kind == "lived_answer"]
    assert [d.ref_id for d in lived] == ["lived.q1@bbbbbbbbbbbb"]
    prev = [n.ref_id for n in lived[0].neighborhood if n.relation == "previous_answer"]
    assert prev == ["lived.q1@aaaaaaaaaaaa"]


def test_lived_question_neighbours_skip_a_newer_draft():
    docs, _, _ = rr.worldview_pool(_Falkor())
    q = next(d for d in docs if d.ref_kind == "lived_question")
    assert [n.ref_id for n in q.neighborhood] == ["lived.q1@bbbbbbbbbbbb", "lived.q1@aaaaaaaaaaaa"]


class _FalkorBlankQuestion(_Falkor):
    """Live shape on 2026-11-14 (#3585): a SelfQuestionMint with text '' whose LivedAnswer is current."""

    def rows(self, graph, cypher, **params):
        if "SelfQuestionMint" in cypher:
            return [{"qid": "lived.q1", "text": ""}, {"qid": "lived.q2", "text": "   "}]
        return super().rows(graph, cypher, **params)


def test_blank_question_text_does_not_halt_the_pool():
    docs, pairs, _ = rr.worldview_pool(_FalkorBlankQuestion())
    assert not [d for d in docs if d.ref_kind == "lived_question"]
    lived = [d for d in docs if d.ref_kind == "lived_answer"]
    assert [d.ref_id for d in lived] == ["lived.q1@bbbbbbbbbbbb"]
    assert not [n for n in lived[0].neighborhood if n.relation == "answers_question"]
    assert pairs["questions"] == []


class _FalkorQuestionKey(_Falkor):
    """Live shape on 2026-11-17 (#3592): words under `question`, not `text`."""

    def rows(self, graph, cypher, **params):
        if "SelfQuestionMint" in cypher:
            return [{"qid": "lived.q1", "text": None, "question": "which weights does it read?"},
                    {"qid": "lived.q2", "text": "", "question": None}]
        return super().rows(graph, cypher, **params)


def test_question_keyed_mint_is_read_and_blank_one_is_logged(caplog):
    with caplog.at_level("WARNING", logger="orion.world_pulse_read.resonance_records"):
        docs, pairs, _ = rr.worldview_pool(_FalkorQuestionKey())
    assert pairs["questions"] == [{"qid": "lived.q1", "text": "which weights does it read?"}]
    assert [d.ref_id for d in docs if d.ref_kind == "lived_question"] == ["lived.q1"]
    assert "resonance_self_question_mint_dropped count=1" in caplog.text and "lived.q2" in caplog.text
