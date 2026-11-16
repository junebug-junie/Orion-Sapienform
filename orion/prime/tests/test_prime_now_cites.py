"""``experience`` as a citable ref kind in Prime's answer (integration patch 3): parse, keep only shown, cap 3."""

from __future__ import annotations

from datetime import datetime, timezone

from orion.prime.state import PrimeStateV1
from orion.prime.thought import (
    FrameInputs,
    PrimeAnswerV1,
    apply_answer,
    build_frame,
    parse_refs,
    shown_cites,
    shown_experience_ids,
    summary_note,
    system_prompt,
)

T0 = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
SHOWN = ["reading_turn:r1", "curiosity_run:c1", "vision_percept:v1", "street_event:s1"]


def _apply(raw: dict, shown=SHOWN, **kw):
    return apply_answer(PrimeStateV1(), PrimeAnswerV1.model_validate(raw), now=T0, arc_ids=[], focus_pending=False,
                        shown_experience=shown, **kw)


def test_parse_refs_splits_experience_from_openable_kinds():
    refs, cites, problems = parse_refs([{"kind": "experience", "id": "reading_turn:r1"},
                                        {"kind": "experience", "id": "experience:curiosity_run:c1"},
                                        {"kind": "experience", "id": "reading_turn:r1"},
                                        {"kind": "act", "id": "act_0123456789abcdef"}])
    assert cites == ["reading_turn:r1", "curiosity_run:c1"]  # prefix stripped, duplicate once
    assert [(r.kind, r.id) for r in refs] == [("act", "act_0123456789abcdef")] and problems == []


def test_parse_refs_names_a_bad_experience_ref_and_other_bad_kinds():
    _refs, cites, problems = parse_refs([{"kind": "experience", "id": ""}, {"kind": "experience", "id": 7},
                                         {"kind": "experience", "id": "x", "extra": 1}, {"kind": "dream", "id": "d"}])
    assert cites == [] and len(problems) == 4
    assert parse_refs("nope") == ([], [], ["refs must be a list of {kind, id}"])
    assert parse_refs(None) == ([], [], [])


def test_only_shown_cites_are_kept_in_order_at_most_three():
    applied = _apply({"reading": "r", "refs": [{"kind": "experience", "id": e}
                                               for e in ["gpu_wait:not-shown", *SHOWN]]})
    assert applied.experience_cites == SHOWN[:3]
    assert any("not shown in this frame" in w and "gpu_wait:not-shown" in w for w in applied.warnings)
    assert any("only the first 3 are recorded" in w for w in applied.warnings)
    assert "cited 3 experience(s)" in summary_note(applied)


def test_no_now_read_means_no_cite_is_kept():
    applied = _apply({"reading": "r", "refs": [{"kind": "experience", "id": SHOWN[0]}]}, shown=None)
    assert applied.experience_cites == []
    assert any("the now was not read this thought" in w for w in applied.warnings)
    assert shown_cites([], None) == ([], [])


def test_a_bad_cite_never_fails_the_thought_or_refuses_the_move():
    raw = {"reading": "r", "refs": [{"kind": "experience"}, {"kind": "arc", "id": "8d87adeca2ce06a5"}],
           "move": {"kind": "curiosity_focus", "question": "what heats the closet?", "why": "heat",
                    "refs": [{"kind": "experience", "id": SHOWN[1]}]}}
    applied = _apply(raw)
    assert applied.move is not None and applied.refs == []  # the experience ref is not forwarded to curiosity
    assert applied.experience_cites == [SHOWN[1]]
    assert any(w.startswith("refs: ") for w in applied.warnings)
    assert any("1 arc/act ref(s) ignored" in w for w in applied.warnings)


def test_section_and_prompt_appear_only_when_the_now_was_read():
    now = {"shown_id": "s1", "items": [{"event_id": SHOWN[0], "organ": "reading", "source_kind": "reading_turn",
                                        "label": "read about heat", "occurred_at": T0.isoformat(), "salience": None,
                                        "related_refs": []}]}
    with_now = FrameInputs(now=T0, orion_now=now)
    frame = build_frame(PrimeStateV1(), with_now)
    assert "- [experience:reading_turn:r1] reading 0 min ago: read about heat (salience unknown)" in frame
    assert shown_experience_ids(with_now) == [SHOWN[0]]
    without = FrameInputs(now=T0)
    assert "WHAT YOUR ORGANS LIVED" not in build_frame(PrimeStateV1(), without)
    assert shown_experience_ids(without) is None
    assert '"kind": "experience"' in system_prompt(None, experiences=True)
    assert '"kind": "experience"' not in system_prompt(None)
