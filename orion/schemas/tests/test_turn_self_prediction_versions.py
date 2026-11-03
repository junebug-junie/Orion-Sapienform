"""Bet v2 rides beside v1: readers accept both before any producer writes v2 (consumer-first).

The models are extra="forbid", so a reader that only knows v1 rejects a v2 bet, and with it the whole
stance record. These checks must pass on every reader before the producer switches to v2.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.thought import (
    ThoughtDecisionRecordV1,
    TurnSelfPredictionV1,
    TurnSelfPredictionV2,
)

V1_BARE = {"asks_question_p": 0.2, "steps_lo": 3, "steps_hi": 5}
V2 = {"schema_version": "turn.self_prediction.v2", "asks_question_p": 0.2, "fetches_outside_p": 0.5,
      "more_steps_than_usual_p": 0.6, "longer_reply_than_usual_p": 0.4}


def _record(sp):
    return ThoughtDecisionRecordV1(event_id="e", correlation_id="c", session_id=None,
                                   created_at=datetime(2026, 11, 3, tzinfo=timezone.utc),
                                   disposition="proceed", self_prediction=sp)


def test_an_unversioned_bet_is_still_v1():
    sp = _record(V1_BARE).self_prediction
    assert isinstance(sp, TurnSelfPredictionV1) and sp.schema_version == "turn.self_prediction.v1"
    # Probabilities alone match both shapes; unversioned means v1, never a guess at v2.
    assert isinstance(_record({"asks_question_p": 0.2}).self_prediction, TurnSelfPredictionV1)


def test_a_v2_bet_is_read_as_v2_and_round_trips():
    rec = _record(V2)
    assert isinstance(rec.self_prediction, TurnSelfPredictionV2)
    again = ThoughtDecisionRecordV1.model_validate(rec.model_dump(mode="json"))
    assert again.self_prediction == rec.self_prediction


def test_a_stored_v1_bet_round_trips():
    rec = _record(V1_BARE)
    assert ThoughtDecisionRecordV1.model_validate(rec.model_dump(mode="json")).self_prediction == rec.self_prediction


@pytest.mark.parametrize("bad", [{"more_steps_than_usual_p": 0.6},                      # v2 key, no version
                                 dict(V2, steps_lo=3),                                   # range key on v2
                                 dict(V2, more_steps_than_usual_p=1.5)])
def test_mixed_or_out_of_range_bets_are_refused(bad):
    with pytest.raises(ValidationError):
        _record(bad)
