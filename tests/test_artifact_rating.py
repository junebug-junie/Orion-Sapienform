"""Scoring a human rating of an Orion-produced artifact.

The negative tests are the ones that matter. A rating is the first outcome in
this system that Orion cannot produce itself, so the ways it could quietly
become fake are new: a magnitude invented from a category count, a scalar
conjured from an unscoreable value, or an unrated artifact treated as a bad
one.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from orion.autonomy.prediction import EffectPosterior
from orion.autonomy.rating import (
    RATING_OBSERVATION_VARIANCE,
    RATING_PRIOR_VARIANCE,
    cold_rating_prior,
    rating_key,
    rating_scalar,
    score_rating,
)
from orion.schemas.chat_response_feedback import (
    ChatResponseFeedbackV1,
    build_artifact_ref,
    parse_artifact_ref,
)

NOW = datetime(2026, 8, 21, 21, 0, tzinfo=timezone.utc)


def _score(feedback_value="up", categories=("helpful_actionable",), prior=None, free_text=None):
    return score_rating(
        artifact_ref="artifact:journal:2026-08-21",
        dispatch_id="dispatch:abc",
        dispatch_kind="summarize",
        target_id="self:current",
        feedback_value=feedback_value,
        categories=list(categories),
        free_text=free_text,
        rated_at=NOW,
        prior=prior,
    )


class TestTheArithmetic:
    """Every expected value hand-computed in the comment beside it."""

    def test_first_up_rating_from_a_cold_prior(self):
        s = _score()
        # prior (mean 0, var 1), observation +1, obs var 1
        # precision = 1/1 + 1/1 = 2 -> var = 0.5
        # mean = 0.5 * (0/1 + 1/1) = 0.5
        assert s.posterior_mean == pytest.approx(0.5)
        assert s.posterior_variance == pytest.approx(0.5)
        assert s.posterior_n == 1
        assert s.rating == 1.0
        assert s.predicted_rating == 0.0
        assert s.prediction_error == pytest.approx(1.0)

    def test_second_up_rating(self):
        s = _score(prior=EffectPosterior(0.5, 0.5, 1))
        # precision = 1/0.5 + 1/1 = 3 -> var = 1/3
        # mean = (1/3) * (0.5/0.5 + 1/1) = 2/3
        assert s.posterior_mean == pytest.approx(2.0 / 3.0)
        assert s.posterior_variance == pytest.approx(1.0 / 3.0)
        assert s.posterior_n == 2

    def test_surprise_is_the_exact_kl(self):
        s = _score()
        # 0.5 * (ln(1.0/0.5) + (0.5 + 0.5^2)/1.0 - 1)
        #   = 0.5 * (0.6931472 + 0.75 - 1) = 0.2215736
        expected = 0.5 * (math.log(1.0 / 0.5) + (0.5 + 0.25) / 1.0 - 1.0)
        assert s.surprise_nats == pytest.approx(expected)
        assert s.surprise_nats == pytest.approx(0.2215736, abs=1e-6)

    def test_down_is_the_mirror_of_up(self):
        up = _score("up")
        down = _score("down", categories=("not_actionable",))
        assert down.rating == -1.0
        assert down.posterior_mean == pytest.approx(-up.posterior_mean)
        assert down.posterior_variance == pytest.approx(up.posterior_variance)
        # Same information either way -- a thumbs-down teaches exactly as much
        # as a thumbs-up, which is the point of scoring in nats.
        assert down.surprise_nats == pytest.approx(up.surprise_nats)

    def test_a_contradicting_rating_pulls_the_belief_back(self):
        confident_up = EffectPosterior(0.8, 0.2, 4)
        s = _score("down", categories=("not_actionable",), prior=confident_up)
        # precision = 1/0.2 + 1/1 = 6 -> var = 1/6
        # mean = (1/6) * (0.8/0.2 + -1/1) = (1/6) * 3 = 0.5
        assert s.posterior_mean == pytest.approx(0.5)
        assert s.posterior_variance == pytest.approx(1.0 / 6.0)


class TestMagnitudeIsNotInvented:
    """The defect this whole arc exists to delete, in its newest disguise."""

    def test_five_categories_score_exactly_the_same_as_one(self):
        one = _score("down", categories=("not_actionable",))
        five = _score(
            "down",
            categories=(
                "not_actionable",
                "too_abstract",
                "missed_relevant_context",
                "overconfident_false_certainty",
                "ignored_instructions",
            ),
        )
        assert five.rating == one.rating
        assert five.posterior_mean == one.posterior_mean
        assert five.surprise_nats == one.surprise_nats

    def test_categories_are_still_recorded(self):
        """Not scored is not the same as not kept. They say WHY."""
        s = _score("down", categories=("too_abstract", "not_actionable"))
        assert s.categories == ("too_abstract", "not_actionable")

    def test_free_text_is_carried_verbatim(self):
        s = _score(free_text="the middle section was the only useful part")
        assert s.free_text == "the middle section was the only useful part"

    def test_an_unscoreable_value_is_refused_not_coerced(self):
        for bad in ("neutral", "", "UP", "meh", "1"):
            with pytest.raises(ValueError):
                rating_scalar(bad)

    def test_only_up_and_down_exist(self):
        assert rating_scalar("up") == 1.0
        assert rating_scalar("down") == -1.0


class TestRedundancyStopsPaying:
    def test_repeated_identical_ratings_earn_less_each_time(self):
        prior = cold_rating_prior()
        nats = []
        for _ in range(6):
            s = _score(prior=prior)
            nats.append(s.surprise_nats)
            prior = EffectPosterior(s.posterior_mean, s.posterior_variance, s.posterior_n)
        assert nats == sorted(nats, reverse=True)
        # The sixth identical rating is worth under a twentieth of the first.
        assert nats[5] < nats[0] / 20

    def test_a_surprising_rating_earns_more_than_a_confirming_one(self):
        confident_up = EffectPosterior(0.8, 0.2, 4)
        confirming = _score("up", prior=confident_up)
        contradicting = _score("down", categories=("not_actionable",), prior=confident_up)
        assert contradicting.surprise_nats > confirming.surprise_nats


class TestTheColdPriorAssertsNothing:
    def test_cold_prior_is_neutral_and_maximally_uncertain(self):
        p = cold_rating_prior()
        assert p.mean == 0.0
        assert p.n == 0
        # For observations on {-1, +1} the largest achievable variance is 1.0.
        assert p.variance == RATING_PRIOR_VARIANCE == 1.0
        assert RATING_OBSERVATION_VARIANCE == 1.0

    def test_key_is_action_scoped_not_signal_scoped(self):
        """No signal component: the rating IS the signal. No bin, no arm --
        a rating does not mean-revert, so there is nothing to match on."""
        assert rating_key("summarize", "self:current") == ("summarize", "self:current")


DISPATCH = (
    "dispatch:proposal:prune_stopped_containers:tick_fc7585176059:none:"
    "execution_dispatch_policy.v1"
)
REF = build_artifact_ref("journal", DISPATCH)


def _artifact_feedback(**kwargs):
    base = dict(
        feedback_id="f1",
        target_artifact_ref=REF,
        feedback_value="up",
        categories=[],
        user_id="juniper",
    )
    base.update(kwargs)
    return ChatResponseFeedbackV1(**base)


class TestAnArtifactRefCarriesItsAction:
    """A rating that cannot be attributed to an action teaches nothing about
    any action, which is the whole purpose of the pipeline it feeds."""

    def test_round_trip(self):
        kind, dispatch_id = parse_artifact_ref(REF)
        assert kind == "journal"
        assert dispatch_id == DISPATCH

    def test_a_dispatch_id_survives_its_own_colons(self):
        """Real dispatch ids contain 5 colons. A naive split loses them."""
        assert DISPATCH.count(":") == 5
        assert parse_artifact_ref(REF)[1] == DISPATCH

    @pytest.mark.parametrize(
        "bad",
        [
            "artifact:journal:oops",          # no dispatch id
            "journal:" + DISPATCH,            # no artifact prefix
            "artifact::" + DISPATCH,          # no kind
            "artifact:journal",               # no dispatch component at all
            "",
        ],
    )
    def test_unattributable_refs_are_refused_not_stored(self, bad):
        with pytest.raises(ValueError):
            parse_artifact_ref(bad)

    def test_the_model_refuses_them_too(self):
        with pytest.raises(ValueError):
            _artifact_feedback(target_artifact_ref="artifact:journal:oops")

    def test_a_colon_in_the_kind_is_refused(self):
        with pytest.raises(ValueError):
            build_artifact_ref("jour:nal", DISPATCH)


class TestAttestation:
    def test_an_artifact_rating_requires_a_rater_on_record(self):
        """Not authentication -- nothing on this host can prove a human. But
        an unattributed rating cannot be told apart from Orion rating itself,
        so it is refused rather than stored ambiguously."""
        with pytest.raises(ValueError):
            _artifact_feedback(user_id=None)

    def test_chat_feedback_still_does_not_require_one(self):
        """Two live rows have user_id NULL. This must not break them."""
        f = ChatResponseFeedbackV1(
            feedback_id="f", target_turn_id="t1", feedback_value="up", categories=[]
        )
        assert f.user_id is None


class TestTheFeedbackContract:
    def test_an_artifact_is_a_valid_target(self):
        assert _artifact_feedback().target_artifact_ref == REF

    def test_a_chat_target_key_is_byte_identical_to_before_this_patch(self):
        """submission_fingerprint is a sha256 OF target_key. Reshaping every
        existing chat key would change every fingerprint."""
        f = ChatResponseFeedbackV1(
            feedback_id="f2",
            target_turn_id="t1",
            feedback_value="down",
            categories=["not_actionable"],
        )
        assert f.target_key == "t1||||"

    def test_the_artifact_component_is_prefixed(self):
        assert _artifact_feedback().target_key.endswith(f"|artifact={REF}")

    def test_a_target_is_still_required(self):
        with pytest.raises(ValueError):
            ChatResponseFeedbackV1(feedback_id="f4", feedback_value="up")

    def test_a_pipe_in_a_chat_field_cannot_forge_an_artifact_key(self):
        """The real collision case, which the first version of this test
        missed by comparing a trivial pair. The join separator is an
        unescaped "|", so before the artifact component was prefixed a chat
        row whose user_id contained "|artifact:z" produced a byte-identical
        target_key AND submission_fingerprint to a genuine artifact rating."""
        forged = ChatResponseFeedbackV1(
            feedback_id="a",
            target_turn_id="t",
            user_id="u|artifact:" + DISPATCH,
            feedback_value="up",
            categories=[],
        )
        genuine = ChatResponseFeedbackV1(
            feedback_id="b",
            target_turn_id="t",
            user_id="u",
            target_artifact_ref=build_artifact_ref("artifact", DISPATCH),
            feedback_value="up",
            categories=[],
        )
        assert forged.target_key != genuine.target_key
        assert forged.submission_fingerprint != genuine.submission_fingerprint

    def test_artifact_and_chat_feedback_do_not_collide_on_fingerprint(self):
        a = _artifact_feedback(feedback_id="a")
        b = ChatResponseFeedbackV1(
            feedback_id="b", target_turn_id=REF, feedback_value="up", categories=[]
        )
        assert a.submission_fingerprint != b.submission_fingerprint

    def test_two_identical_ratings_share_a_fingerprint(self):
        """The property the migration's partial unique index relies on: a
        rater unsure the first one landed runs it again, and both carry the
        same opinion under different feedback_ids."""
        a = _artifact_feedback(feedback_id="run-1")
        b = _artifact_feedback(feedback_id="run-2")
        assert a.feedback_id != b.feedback_id
        assert a.submission_fingerprint == b.submission_fingerprint


class TestTheScaleIsSharedAndTheNumbersAreNot:
    def test_a_zero_effect_pressure_observation_outscores_a_human_rating(self):
        """Measured, and recorded as a test so the claim cannot quietly
        become true-by-assertion later. An earlier docstring claimed the
        shared unit made the two ledgers comparable; it does not."""
        from orion.autonomy.prediction import EffectPosterior, score_observation
        from orion.autonomy.rating import cold_start_surprise_nats

        _, pressure_nats, _ = score_observation(EffectPosterior.cold(), 0.0)
        rating_nats = cold_start_surprise_nats()
        assert pressure_nats == pytest.approx(0.5595, abs=1e-3)
        assert rating_nats == pytest.approx(0.2216, abs=1e-3)
        assert pressure_nats > 2.5 * rating_nats

    def test_the_reference_matches_a_real_first_rating(self):
        from orion.autonomy.rating import cold_start_surprise_nats

        assert _score().surprise_nats == pytest.approx(cold_start_surprise_nats())
