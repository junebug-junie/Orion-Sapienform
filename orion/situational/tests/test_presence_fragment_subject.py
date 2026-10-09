"""presence_fragment names a matched person; callers that pass no subject
(endogenous_outreach) keep the exact old wording."""

from orion.situational.perception_reader import presence_fragment


def test_no_subject_keeps_legacy_wording():
    assert presence_fragment("present", 3 * 3600.0) == "Someone has been in view for about 3 hours."
    assert presence_fragment("recent", 600.0) == "Someone stepped out of view 10 minutes ago."


def test_named_subject_is_capitalised_and_attributed_to_the_face_match():
    assert presence_fragment("present", 420.0, subject="juniper") == (
        "Juniper has been in view for 7 minutes (matched by face)."
    )
    assert presence_fragment("recent", 420.0, subject="juniper") == (
        "Juniper stepped out of view 7 minutes ago (matched by face)."
    )


def test_placeholder_subjects_are_not_names():
    for s in (None, "", "unknown", "none", "NONE", "  "):
        assert presence_fragment("present", 60.0, subject=s).startswith("Someone")


def test_absent_or_undated_says_nothing():
    assert presence_fragment("absent", 60.0, subject="juniper") is None
    assert presence_fragment("present", None, subject="juniper") is None
