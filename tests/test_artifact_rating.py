"""The artifact-ref half of the chat_response_feedback contract.

Until 2026-11-07 this file also tested scoring a human rating of an Orion-produced artifact
(orion/autonomy/rating.py, fed by scripts/rate_artifact.py and scripts/score_artifact_ratings.py).
Every artifact those scripts rated was a dispatch result, and they read the dispatch tables; the
whole rating path retired with the dispatch rung (ladder retirement PR 6). It had scored 0 rows
(substrate_action_ratings, live 2026-11-07). The schema's artifact-ref contract stays: it is part
of a bus payload, and its refusals still guard the stored column.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.schemas.chat_response_feedback import (
    ChatResponseFeedbackV1,
    build_artifact_ref,
    parse_artifact_ref,
)

NOW = datetime(2026, 8, 21, 21, 0, tzinfo=timezone.utc)


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
