"""Memory confirmation loop: card wording, source monitoring, ids, and the recall predicate.

Pure tests (no database). The Postgres-backed loop is in test_confirmation_pg.py.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import get_args

import pytest

from orion.memory.episode import confirmation as c
from orion.memory.episode.validate import INTERNAL_CHANNELS, UNJUDGED_STAKES_LABEL
from orion.schemas.memory_episode import Channel, HIGH_STAKES_REASONS, Voice

AT = datetime(2026, 10, 3, 15, 0, tzinfo=timezone.utc)
# Phrases that present a memory as something Juniper said or the two of them worked out.
# ("not from anything you told me" is the disclaimer, not an attribution.)
JUNIPER_FRAMES = re.compile(r"\byou told me something\b|\byou (said|mentioned)\b|\bwe worked out\b", re.I)


def _q(**kw) -> str:
    base = dict(statement="Juniper said things with her sister have felt asymmetric lately.", voice="juniper_said",
                channel="chat", stakes_reason="family_relationships", occurred_at=AT)
    base.update(kw)
    return c.render_question(**base)


def test_example_card_reads_in_orions_voice():
    q = _q()
    assert q.startswith("You told me something on Oct 3, and I wrote it down like this:")
    assert "“Juniper said things with her sister have felt asymmetric lately.”" in q
    assert "family" in q
    assert q.endswith("Want me to remember that?")


def test_every_high_stakes_category_has_its_own_card_wording():
    """Each stakes category's consumer is its line on the card: no category may fall through to
    the generic wording, and no two categories may share a line (a category with no distinct
    behavior would be a label without a consumer)."""
    assert set(c.WHY_BY_REASON) == set(HIGH_STAKES_REASONS)
    assert len(set(c.WHY_BY_REASON.values())) == len(c.WHY_BY_REASON)
    for reason, why in c.WHY_BY_REASON.items():
        assert why in _q(stakes_reason=reason)
        assert c.WHY_UNJUDGED not in _q(stakes_reason=reason)


@pytest.mark.parametrize("reason", [UNJUDGED_STAKES_LABEL, None, "", "something_new"])
def test_uncategorized_high_stakes_says_so(reason):
    assert c.WHY_UNJUDGED in _q(stakes_reason=reason)


def test_direction_and_identity_cards_close_with_their_own_question():
    assert _q(stakes_reason="orion_asks_direction").endswith("Is that the right direction?")
    assert _q(stakes_reason="identity_conclusion_about_juniper").endswith("Is that fair, and should I keep it?")


@pytest.mark.parametrize("channel", sorted(INTERNAL_CHANNELS))
@pytest.mark.parametrize("voice", list(get_args(Voice)))
def test_reverie_derived_memory_is_never_asked_as_something_juniper_said(channel, voice):
    """Source monitoring: an internal-channel memory is Orion's own, whatever voice it carries."""
    q = _q(channel=channel, voice=voice)
    assert not JUNIPER_FRAMES.search(q.split("“")[0]), q
    assert "not from anything you told me" in q


@pytest.mark.parametrize("channel", list(get_args(Channel)))
@pytest.mark.parametrize("voice", list(get_args(Voice)))
def test_only_a_chat_juniper_voice_is_framed_as_hers(channel, voice):
    frame = _q(channel=channel, voice=voice).split("“")[0]
    hers = bool(JUNIPER_FRAMES.search(frame))
    assert hers == (channel == "chat" and voice in ("juniper_said", "worked_out_together")), frame


def test_orion_thought_from_chat_is_framed_as_orions_own_take():
    frame = _q(voice="orion_thought").split("“")[0]
    assert "my own take" in frame and not JUNIPER_FRAMES.search(frame)


def test_statement_is_quoted_whole_and_long_ones_are_cut():
    q = _q(statement="  spaced   out\nstatement  ")
    assert "“spaced out statement”" in q
    long_q = _q(statement="word " * 200)
    quoted = long_q.split("“")[1].split("”")[0]
    assert len(quoted) == c.MAX_STATEMENT_CHARS and quoted.endswith("…")


def test_date_falls_back_to_created_at_and_is_omitted_when_unknown():
    assert "on Oct 3" in _q(occurred_at=None, created_at=AT)
    assert _q(occurred_at=None, created_at=None).startswith("You told me something, and")


def test_ids_are_deterministic_and_round_trip():
    mid = str(uuid.uuid4())
    loop = c.loop_id_for(mid)
    assert loop == f"memory-confirm-{mid}"
    assert c.memory_id_from_loop(loop) == mid
    assert c.ask_id_for(loop) == c.ask_id_for(loop) != c.ask_id_for(c.loop_id_for(str(uuid.uuid4())))
    assert c.outcome_id_for("a") == c.outcome_id_for("a") != c.outcome_id_for("b")
    for bad in ("open-loop-1", "memory-confirm-not-a-uuid", "", None):
        assert c.memory_id_from_loop(bad) is None


def test_resolution_maps_to_verdict_and_card_status():
    assert c.RESOLUTION_VERDICT == {"confirmed": "resolved", "revised": "resolved", "rejected": "dismissed"}
    assert c.RESOLUTION_ASK_STATUS == {"confirmed": "answered", "revised": "answered", "rejected": "dismissed"}


def test_outcome_to_apply_reads_resolution_and_ask_from_features():
    row = {"outcome_id": "o", "loop_id": "memory-confirm-x", "verdict": "resolved", "actor": "juniper",
           "note": None, "features_at_close": '{"resolution": "Confirmed", "ask_id": "a"}'}
    o = c.OutcomeToApply.from_row(row)
    assert (o.resolution, o.ask_id, o.note) == ("confirmed", "a", "")
    assert c.OutcomeToApply(outcome_id="o", loop_id="l", verdict="resolved", note="",
                            features={"resolution": "maybe"}).resolution is None


def test_recall_predicate_excludes_rejected():
    assert "confirmation_state <> 'rejected'" in c.RECALLABLE_WHERE
    assert "status = 'active'" in c.RECALLABLE_WHERE


def test_daily_report_names_every_state_the_loop_writes():
    """Each confirmation state the loop writes has a reader that says it in words (the daily
    old-vs-new report), so no state is a label nothing shows."""
    import inspect

    from orion.memory.episode.report import CONFIRMATION_FLAG

    src = inspect.getsource(c)
    # Every state the module sets or filters on (SET and WHERE clauses alike).
    named = set(re.findall(r"confirmation_state = '([a-z_]+)'", src))
    assert named == {"pending_confirmation", "unconfirmed", "confirmed", "rejected", "corrected"}
    assert named <= set(CONFIRMATION_FLAG)
    assert "auto" not in CONFIRMATION_FLAG
