from orion.curiosity.access_refusals import count_access_refusals, ACCESS_REFUSAL_THRESHOLD
from orion.curiosity.role_teach_disclosure import format_access_refusal_progress


def test_count_two_permission_denied() -> None:
    notes = [
        "psql: permission denied for table durable_admission_runs",
        "second hop: Permission Denied reading schema",
        "ordinary bash ok",
    ]
    assert count_access_refusals(notes) >= 2


def test_progress_line_only_at_threshold() -> None:
    assert format_access_refusal_progress(1) == []
    lines = format_access_refusal_progress(2)
    assert lines
    assert "hand off" in "\n".join(lines).lower()
    assert "cursor" in "\n".join(lines).lower()
