from orion.curiosity.access_refusals import count_access_refusals, ACCESS_REFUSAL_THRESHOLD
from orion.curiosity.role_teach_disclosure import format_access_refusal_progress


def test_count_two_permission_denied() -> None:
    notes = [
        "psql: permission denied for table durable_admission_runs",
        "second hop: Permission Denied reading schema",
        "ordinary bash ok",
    ]
    assert count_access_refusals(notes) >= 2


def test_oracle_and_non_acl_prose_are_not_access_refusals() -> None:
    """Bare ``acl`` must not false-fire inside words like oracle / spectacle."""
    notes = [
        "consulted the frontier oracle for architecture drift",
        "hop notes describe a spectacle of unrelated errors",
        "ordinary bash ok",
    ]
    assert count_access_refusals(notes) == 0


def test_real_acl_block_phrasing_still_counts() -> None:
    notes = [
        "ACL blocked atlas read",
        "graph acl: user lacks RO_QUERY",
        "insufficient_privilege on journal_entries",
    ]
    assert count_access_refusals(notes) == 3
    assert count_access_refusals(notes) >= ACCESS_REFUSAL_THRESHOLD


def test_progress_line_only_at_threshold() -> None:
    assert format_access_refusal_progress(1) == []
    lines = format_access_refusal_progress(2)
    assert lines
    assert "hand off" in "\n".join(lines).lower()
    assert "cursor" in "\n".join(lines).lower()
