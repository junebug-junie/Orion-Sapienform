from orion.curiosity.self_inquiry import LivedAnswer, build_lived_answer_history_write


def test_mirror_refuses_empty_evidence():
    row = LivedAnswer(
        run_id="abcd12",
        question_id="lived.care",
        family="lived",
        text="I care about Juniper.",
        evidence=[],
        revises="",
        written_at=1,
    )
    assert build_lived_answer_history_write(row) is None


def test_mirror_sets_concept_id_namespace():
    row = LivedAnswer(
        run_id="abcd12",
        question_id="lived.care",
        family="lived",
        text="I care about continuity with Juniper.",
        evidence=["journal_entries:1"],
        revises="",
        written_at=1,
    )
    write = build_lived_answer_history_write(row)
    assert write is not None
    assert write.concept_id == "self:lived:lived.care"
    assert write.produced_by == "curiosity_self_inquiry"
