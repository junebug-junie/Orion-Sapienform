"""Orion Prime step 3, pure side: the regulate move, its refusals, the prompt and frame, and its grade."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.physiology import prime_charge as pc
from orion.prime import acts as pa
from orion.prime.state import PrimeRegulationOpenV1, PrimeStateV1
from orion.prime.thought import (
    SYSTEM_PROMPT,
    FrameInputs,
    PrimeAnswerV1,
    RegulateMoveV1,
    apply_answer,
    build_frame,
    parse_answer,
    summary_note,
    system_prompt,
)

T0 = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)
C = "durable.curiosity_investigate"
RECEPTORS = {C: ("slow", "quicken")}


def answer(**move):
    return PrimeAnswerV1.model_validate({"reading": "r", "move": {"kind": "regulate", "family": C, "direction": "slow",
                                                                  "why": "curiosity ran five times", **move}})


def reg(arm="treated", direction="slow", written=True, **kw):
    base = dict(act_id="act_r", signal_id="sig1", thought_id="t1", family=C, direction=direction, why="w", arm=arm,
                magnitude=0.5, opened_at=T0, expires_at=T0 + timedelta(hours=3), written=written)
    return PrimeRegulationOpenV1(**{**base, **kw})


def eff(want_id, effect, why="value 0.82 - lane charge 0.00 beat Prime's slow charge 0.54"):
    return pc.PrimeChargeEffectV1(want_id=want_id, effect=effect, last=effect, bids=1, first_at=T0, last_at=T0, why=why)


def test_a_regulate_move_to_a_family_with_the_receptor_goes_with_its_hours_clamped():
    for hours, want in ((None, 3.0), (0.2, 1.0), (12, 6.0), (2.5, 2.5)):
        a = apply_answer(PrimeStateV1(), answer(hours=hours), now=T0, arc_ids=[], focus_pending=False,
                         receptors=RECEPTORS)
        assert a.move_refusal is None and isinstance(a.move, RegulateMoveV1) and a.move.hours == want
    assert "move regulate slow durable.curiosity_investigate for 3 h" in summary_note(
        apply_answer(PrimeStateV1(), answer(), now=T0, arc_ids=[], focus_pending=False, receptors=RECEPTORS))


def test_an_unhandled_signal_is_refused_visibly_never_dropped():
    cases = [
        (answer(family="durable.self_inquiry"), RECEPTORS, "declares no receptors"),
        (answer(direction="faster"), RECEPTORS, "not a signal any organ handles"),
        (answer(direction="quicken"), {C: ("slow",)}, "handles slow, not quicken"),
        (answer(), None, "regulation is off"),
    ]
    for ans, receptors, why in cases:
        a = apply_answer(PrimeStateV1(), ans, now=T0, arc_ids=[], focus_pending=False, receptors=receptors)
        assert a.move is None and why in a.move_refusal, (why, a.move_refusal)
        assert "move refused" in summary_note(a)


def test_one_signal_per_organ_until_it_is_graded():
    a = apply_answer(PrimeStateV1(open_regulations=[reg()]), answer(direction="quicken"), now=T0, arc_ids=[],
                     focus_pending=False, receptors=RECEPTORS)
    assert a.move is None and "not graded yet" in a.move_refusal and "15:00Z" in a.move_refusal


def test_a_regulate_move_ignores_the_curiosity_question_gates():
    """An open question or a pending focus blocks a question, not a regulate signal."""
    from orion.prime.state import PrimeNominationV1

    nom = PrimeNominationV1(act_id="a", thought_id="t", focus_id="f", question="q", why="w", arm="treated",
                            opened_at=T0)
    a = apply_answer(PrimeStateV1(open_nominations=[nom]), answer(), now=T0, arc_ids=[], focus_pending=None,
                     receptors=RECEPTORS)
    assert a.move_refusal is None


def test_a_bad_direction_does_not_lose_the_thought():
    parsed, err = parse_answer({"reading": "r", "move": {"kind": "regulate", "family": "x", "direction": "?", "why": "w"}})
    assert err is None and isinstance(parsed.move, RegulateMoveV1)


def test_the_move_is_offered_only_when_regulation_is_on_and_someone_receives_it():
    assert system_prompt(None) == SYSTEM_PROMPT and system_prompt({}) == SYSTEM_PROMPT
    assert "regulate" not in SYSTEM_PROMPT
    on = system_prompt(RECEPTORS)
    assert '"kind": "regulate"' in on and "outbids it" in on


def test_the_frame_shows_receptors_and_what_each_open_signal_did_so_far():
    state = PrimeStateV1(open_regulations=[reg(), reg(arm="control", signal_id="sig2", family="x")])
    inputs = FrameInputs(now=T0, receptors=RECEPTORS,
                         regulation_effects={"sig1": [eff("r1", "held", "did not beat"), eff("r2", "outbid")],
                                             "sig2": None})
    frame = build_frame(state, inputs)
    assert "ORGANS YOU CAN SIGNAL" in frame and f"- {C}: slow, quicken" in frame
    assert "2 of its runs bid, 1 held by my charge, 1 outbid it" in frame and "beat Prime's slow charge 0.54" in frame
    assert "held back as a control, not applied" in frame and "the market's record was not read" in frame
    assert "ORGANS YOU CAN SIGNAL" not in build_frame(PrimeStateV1(), FrameInputs(now=T0))
    gap = build_frame(PrimeStateV1(), FrameInputs(now=T0, receptors_error="families config unreadable (missing)"))
    assert "unavailable: families config unreadable" in gap


def test_the_grade_is_the_markets_record_same_rule_both_arms():
    assert pa.settle_regulation(reg(), [eff("r1", "held"), eff("r2", "outbid")])[0] is True
    assert pa.settle_regulation(reg(), [eff("r2", "outbid")])[0] is False
    produced, note = pa.settle_regulation(reg(), [])
    assert not produced and "no bid from it came" in note
    produced, note = pa.settle_regulation(reg(arm="control"), [eff("r1", "held")])
    assert produced and "would have been held" in note
    assert pa.settle_regulation(reg(direction="quicken"), [eff("r1", "released")])[0] is True
    produced, note = pa.settle_regulation(reg(written=False, write_error="ConnectionError: x"), [eff("r1", "held")])
    assert not produced and "not delivered" in note


def test_the_regulate_act_is_the_existing_verb_with_a_horizon_past_the_signal():
    act = pa.regulate_act("sig1", family=C, arm="control", hours=3.0, thought_act_id="act_t")
    assert act.actor == "prime" and act.verb == "regulate" and act.expect.signal_id == "artifact_produced"
    assert act.expect.horizon_sec == 3 * 3600 + pc.GRACE_SEC and act.test_before == "holdback:control"
    assert act.object_ref == f"physiology_family:{C}" and act.parent_act_id == "act_t"
    assert act.act_id == pa.regulate_act("sig1", family=C, arm="control", hours=3.0, thought_act_id=None).act_id


def test_persisted_state_from_before_regulation_still_loads():
    old = '{"schema_version": "prime.state.v1", "concerns": [], "open_nominations": [], "thoughts_total": 3}'
    assert PrimeStateV1.model_validate_json(old).open_regulations == []


def test_an_organ_rests_from_prime_as_long_as_the_last_signal_lasted():
    state = PrimeStateV1(regulation_rest_until={C: T0 + timedelta(hours=2)})
    a = apply_answer(state, answer(), now=T0, arc_ids=[], focus_pending=False, receptors=RECEPTORS)
    assert a.move is None and "rests from my signals until 14:00Z" in a.move_refusal
    later = apply_answer(state, answer(), now=T0 + timedelta(hours=2), arc_ids=[], focus_pending=False,
                         receptors=RECEPTORS)
    assert later.move_refusal is None
    assert "rests from your signals until 14:00Z" in build_frame(state, FrameInputs(now=T0, receptors=RECEPTORS))
