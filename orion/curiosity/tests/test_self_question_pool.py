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
