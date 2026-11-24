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
                                        {"kind": "act", "id": "act_000000000000000000000000"}])
    assert cites == ["reading_turn:r1", "curiosity_run:c1"]  # prefix stripped, duplicate once
    assert [(r.kind, r.id) for r in refs] == [("act", "act_000000000000000000000000")] and problems == []


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
    assert "proposed 3 experience cite(s)" in summary_note(applied)


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


def test_a_bad_experience_ref_inside_a_move_is_a_warning_not_a_refusal():
    raw = {"reading": "r", "move": {"kind": "curiosity_focus", "question": "what heats the closet?", "why": "heat",
                                    "refs": [{"kind": "experience"}]}}
    applied = _apply(raw)
    assert applied.move is not None and applied.move_refusal is None
    assert any(w.startswith("move refs: experience ref") for w in applied.warnings)
    # A bad arc/act ref still refuses the move, as before.
    bad_arc = _apply({"reading": "r", "move": {**raw["move"], "refs": [{"kind": "arc", "id": "nope"}]}})
    assert bad_arc.move is None and "arc" in (bad_arc.move_refusal or "")


def test_a_question_quoting_a_cited_experience_id_is_refused_with_words_that_say_why():
    eid = "curiosity_run:8580502e-e6ac-5b06-90c9-0f244b9d51ab"
    raw = {"reading": "r", "move": {"kind": "curiosity_focus", "why": "heat",
                                    "question": "what did run 8580502e-e6ac-5b06-90c9-0f244b9d51ab find?",
                                    "refs": [{"kind": "experience", "id": eid}]}}
    applied = _apply(raw, shown=[eid])
    assert applied.move is None
    assert "quotes the id of an experience it cites" in (applied.move_refusal or "")
    assert applied.experience_cites == [eid]  # the cite still counts


def _reply_template(prompt: str) -> str:
    """The JSON reply template: from "Reply with ONE JSON object" to the first blank line."""
    start = prompt.index("Reply with ONE JSON object")
    end = prompt.find("\n\n", start)
    return prompt[start:end if end != -1 else len(prompt)]


def test_the_reply_template_itself_carries_the_experience_cites_when_the_now_is_shown():
    """#3425: live 11-16..11-24 Prime cited in 1 of 70 thoughts that were shown a now. The cite instruction was a
    paragraph after the template, and the template's only "refs" sat inside the move (arc/act only), so the
    model followed the template. The top-level refs must be IN the template, in arc and event mode alike, with
    regulate and take-on on (their paragraphs sit between the template and the cite paragraph live)."""
    for events in (False, True):
        prompt = system_prompt({"curiosity": ("slow",)}, events=events, take_on=True, experiences=True, needs=True,
                               evidence_clock=True)
        template = _reply_template(prompt)
        assert '\n "refs": [] | [{"kind": "experience", "id": "<the id after experience:>"}],\n "move": null |' in template
        # Refs before the move, never last: as the last key the model dropped the closing brace (6 of 15 live).
        assert template.rstrip().endswith('"id": "..."}]}}')
        # The move's own refs are unchanged (arc/act, or act only in event mode).
        assert ('"refs": [{"kind": "act", "id": "..."}]}}' if events
                else '"refs": [{"kind": "arc" | "act", "id": "..."}]}}') in template
    # No now in the frame: the template offers no experience cite.
    assert "experience" not in _reply_template(system_prompt(None, events=True))


def test_an_answer_written_to_the_new_template_records_its_cites():
    """An answer shaped exactly like the template (top-level refs before the move) keeps its shown cites.
    Characterization, not the regression: the parser already accepted top-level refs before #3425."""
    raw = {"reading": "the closet ran hot after the reading", "pick": "none",
           "concerns": {"keep": [], "add": [], "drop": []},
           "refs": [{"kind": "experience", "id": SHOWN[0]}, {"kind": "experience", "id": f"experience:{SHOWN[2]}"}],
           "move": None}
    applied = _apply(raw)
    assert applied.experience_cites == [SHOWN[0], SHOWN[2]]
    assert not applied.warnings


def test_a_dropped_cite_names_its_id_before_the_drop_reasons_and_warnings():
    """2026-11-20 09:29Z live: a Prime cite was dropped as not shown, and the 500-char act note cut the warning before
    the id, so the mismatch could not be read back. The unshown id now comes right after the cite count."""
    state = PrimeStateV1(concerns=[])
    raw = {"reading": "r", "refs": [{"kind": "experience", "id": SHOWN[0]},
                                    {"kind": "experience", "id": "curiosity_run:b23f79951e7a"}]}
    applied = _apply(raw)
    assert applied.experience_cites == [SHOWN[0]]
    assert applied.experience_unshown == ["curiosity_run:b23f79951e7a"]
    applied.dropped = ["c1 (" + "x" * 400 + ")"]
    note = summary_note(applied, now_use="stored 1/1")
    assert "1 cite(s) not shown: curiosity_run:b23f79951e7a" in note[:200]
    assert note.index("not shown: curiosity_run") < note.index("dropped c1") < note.index("warnings:")
    # A long move refusal (no cap) comes after the id, so the 500-char cut cannot hide it.
    applied.move_refusal = "r" * 600
    note = summary_note(applied)
    assert note.index("not shown: curiosity_run") < note.index("move refused")
    assert "curiosity_run:b23f79951e7a" in note[:300]
    # No now read: every cite is unchecked and named as such.
    unread = _apply(raw, shown=None)
    assert unread.experience_unshown == [SHOWN[0], "curiosity_run:b23f79951e7a"] and unread.experience_unread
    assert "2 cite(s) unchecked, the now was not read: reading_turn:r1" in summary_note(unread)
