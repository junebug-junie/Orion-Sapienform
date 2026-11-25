from datetime import datetime, timedelta, timezone
from random import Random

from orion.curiosity.self_question_pool import SelfQuestion, load_seed_questions, pick_question


def _q(**kwargs) -> SelfQuestion:
    base = dict(
        question_id="x",
        text="t",
        family="lived",
        pinned=False,
        minted_by="juniper",
        status="open",
        ask_count=0,
        last_asked_at=None,
    )
    base.update(kwargs)
    return SelfQuestion(**base)


def test_seed_includes_anatomy_and_lived_pins() -> None:
    pool = load_seed_questions()
    families = {q.family for q in pool}
    assert families == {"lived", "anatomy"}
    assert any(q.pinned and q.family == "lived" for q in pool)
    assert any(q.question_id == "anatomy.made_of" or "made of" in q.text.lower() for q in pool)


def test_pinned_floor_forces_lived_even_when_ratio_saturated() -> None:
    now = datetime(2026, 9, 18, tzinfo=timezone.utc)
    stale = _q(
        question_id="lived.who_matters",
        family="lived",
        pinned=True,
        text="Who matters?",
        last_asked_at=now - timedelta(days=30),
        ask_count=1,
    )
    anatomy = _q(question_id="anatomy.made_of", family="anatomy", pinned=True, text="What am I made of?")
    # recent_families already 100% lived — floor must still win
    picked = pick_question(
        pool=[stale, anatomy],
        recent_families=["lived"] * 12,
        lived_weight=0.75,
        pinned_floor_days=7.0,
        now=now,
        rng=Random(0),
    )
    assert picked.question_id == "lived.who_matters"


def test_draw_prefers_anatomy_when_lived_over_weight_and_no_floor() -> None:
    now = datetime(2026, 9, 18, tzinfo=timezone.utc)
    lived = _q(question_id="lived.a", family="lived", pinned=True, last_asked_at=now, ask_count=5)
    anatomy = _q(question_id="anatomy.made_of", family="anatomy", pinned=True, last_asked_at=now, ask_count=0)
    # Force anatomy by saturating lived in the rolling window and using rng that picks anatomy branch
    picks = [
        pick_question(
            pool=[lived, anatomy],
            recent_families=["lived"] * 20,
            lived_weight=0.75,
            pinned_floor_days=7.0,
            now=now,
            rng=Random(i),
        ).family
        for i in range(40)
    ]
    assert "anatomy" in picks


def test_pick_question_excludes_parked() -> None:
    now = datetime(2026, 9, 18, tzinfo=timezone.utc)
    open_q = _q(question_id="lived.open", status="open")
    parked = _q(question_id="lived.parked", status="parked")
    picked = pick_question(
        pool=[open_q, parked],
        recent_families=[],
        now=now,
        rng=Random(0),
    )
    assert picked.question_id == "lived.open"


# --- the door (docs/superpowers/specs/2026-11-25-orion-self-reflection-drift-design.md) ---------------

from orion.curiosity.self_question_pool import (  # noqa: E402
    INSERT_HELD_SQL,
    SELECT_ALL_SQL,
    UPSERT_MINT_SQL,
    merge_seed_with_rows,
    passes_door,
)
import pytest  # noqa: E402


@pytest.mark.parametrize("door", ["held", "routed", "demoted", "unjudged"])
def test_an_orion_question_the_door_has_not_admitted_is_never_picked(door) -> None:
    now = datetime(2026, 11, 25, tzinfo=timezone.utc)
    # Never asked, so the pick's own order would put it first; the door must still keep it out.
    blocked = _q(question_id="lived.orion.new", minted_by="orion", door=door)
    juniper = _q(question_id="lived.who_matters", pinned=True, last_asked_at=now - timedelta(days=1), ask_count=9)
    for seed in range(50):
        picked = pick_question(pool=[blocked, juniper], recent_families=[], now=now, rng=Random(seed))
        assert picked.question_id == "lived.who_matters"
    with pytest.raises(ValueError):
        pick_question(pool=[blocked], recent_families=[], now=now, rng=Random(0))


@pytest.mark.parametrize("door", ["admitted", None])
def test_an_admitted_or_pre_door_orion_question_is_picked_as_before(door) -> None:
    now = datetime(2026, 11, 25, tzinfo=timezone.utc)
    fresh = _q(question_id="lived.orion.new", minted_by="orion", door=door)
    juniper = _q(question_id="lived.who_matters", pinned=True, last_asked_at=now - timedelta(days=1), ask_count=9)
    assert pick_question(pool=[fresh, juniper], recent_families=["lived"], now=now, rng=Random(0)) == fresh


@pytest.mark.parametrize("door", ["held", "routed", "demoted", "unjudged", "admitted", None])
def test_juniper_questions_never_pass_the_door(door) -> None:
    # Even a Juniper row carrying a door value (it never should) is hers: always eligible.
    assert passes_door(_q(minted_by="juniper", door=door))


def test_with_the_door_off_every_orion_question_is_pickable_again() -> None:
    now = datetime(2026, 11, 25, tzinfo=timezone.utc)
    held = _q(question_id="lived.orion.held", minted_by="orion", door="held")
    assert pick_question(pool=[held], recent_families=[], now=now, rng=Random(0), door_enabled=False) == held
    assert not passes_door(held) and passes_door(held, door_enabled=False)


def test_the_door_column_is_read_and_merged() -> None:
    assert "door" in SELECT_ALL_SQL
    rows = [{"question_id": "lived.orion.x", "text": "Why?", "family": "lived", "pinned": False, "minted_by": "orion",
             "status": "open", "ask_count": 0, "last_asked_at": None, "door": "held"}]
    merged = merge_seed_with_rows(load_seed_questions(), rows)
    assert next(q for q in merged if q.question_id == "lived.orion.x").door == "held"
    assert all(q.door is None for q in merged if q.minted_by == "juniper")


def test_every_orion_write_is_held_and_never_rewrites_juniper() -> None:
    assert "'held'" in UPSERT_MINT_SQL and "'orion'" in UPSERT_MINT_SQL
    assert UPSERT_MINT_SQL.rstrip().endswith("WHERE curiosity_self_questions.minted_by = 'orion'")
    # A demoted question stays demoted on a re-mint; new text is held again.
    assert "THEN 'demoted'" in UPSERT_MINT_SQL and "IS DISTINCT FROM EXCLUDED.text THEN 'held'" in UPSERT_MINT_SQL
    assert "'held'" in INSERT_HELD_SQL and "'orion'" in INSERT_HELD_SQL and "DO NOTHING" in INSERT_HELD_SQL
