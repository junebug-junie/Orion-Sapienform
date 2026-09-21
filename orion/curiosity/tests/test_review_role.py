"""`:ReviewRole` -- Orion's optional, separate choice for who grades its own
hops later (self_review or hire_cursor_review). Same authorship rule as
`:InvestigationRole`: Python never MERGEs this node; a missing node means
self_review, the same "missing node is no decision" default the hire-choice
side already uses.
"""

from __future__ import annotations

from orion.curiosity.kickoff_prompt import _review_role_section, build_kickoff_prompt
from orion.curiosity.study_material import StudyMaterial
from orion.curiosity.worldview import (
    ALL_REVIEW_ROLES_CYPHER,
    ReviewRoleRecord,
    WorldviewReader,
    WorldviewUnavailable,
    build_review_role,
    latest_review_role_by_run,
    list_review_roles_for_run_cypher,
    read_all_review_roles,
)


class _FakeReader(WorldviewReader):
    def __init__(self, *, answers=None, raises=False) -> None:
        super().__init__(host="x", port=1, graph_name="g", client=object())
        self.answers = answers or {}
        self.raises = raises

    def query(self, cypher: str):
        if self.raises:
            raise WorldviewUnavailable("ConnectionError: nope")
        for needle, rows in self.answers.items():
            if needle in cypher:
                return rows
        return []


# --- cypher --------------------------------------------------------------


def test_list_review_roles_for_run_cypher_shape():
    text = list_review_roles_for_run_cypher("abc123")
    assert "ReviewRole" in text
    assert "r.run_id = 'abc123'" in text
    assert "ORDER BY r.written_at ASC" in text


def test_all_review_roles_cypher_has_no_run_id_filter():
    assert "ReviewRole" in ALL_REVIEW_ROLES_CYPHER
    assert "WHERE" not in ALL_REVIEW_ROLES_CYPHER


# --- row -> dataclass ------------------------------------------------------


def test_build_review_role_requires_run_id_and_choice():
    assert build_review_role({"run_id": "", "choice": "self_review"}) is None
    assert build_review_role({"run_id": "abc", "choice": ""}) is None
    rec = build_review_role(
        {"run_id": "abc", "choice": "hire_cursor_review", "why": "backlog is bad", "written_at": 100}
    )
    assert rec == ReviewRoleRecord(
        run_id="abc", choice="hire_cursor_review", why="backlog is bad", written_at=100
    )


# --- reads ------------------------------------------------------------------


def test_read_all_review_roles_drops_bad_rows_and_fails_open():
    rows = [
        {"run_id": "abc", "choice": "self_review", "why": "", "written_at": 100},
        {"run_id": "", "choice": "hire_cursor_review"},  # dropped: no run_id
        {"run_id": "def", "choice": "hire_cursor_review", "why": "queue is busy", "written_at": 200},
    ]
    reader = _FakeReader(answers={"MATCH (r:ReviewRole)": rows})
    out = read_all_review_roles(reader)
    assert [r.run_id for r in out] == ["abc", "def"]

    assert read_all_review_roles(_FakeReader(raises=True)) == []


def test_latest_review_role_by_run_picks_newest_written_at_per_run():
    records = [
        ReviewRoleRecord(run_id="r1", choice="self_review", why="", written_at=100),
        ReviewRoleRecord(run_id="r1", choice="hire_cursor_review", why="changed my mind", written_at=200),
        ReviewRoleRecord(run_id="r2", choice="hire_cursor_review", why="", written_at=50),
    ]
    latest = latest_review_role_by_run(records)
    assert latest["r1"].choice == "hire_cursor_review"
    assert latest["r1"].why == "changed my mind"
    assert latest["r2"].choice == "hire_cursor_review"
    assert "r3" not in latest


def test_latest_review_role_by_run_empty_input():
    assert latest_review_role_by_run([]) == {}


# --- kickoff prompt teach ---------------------------------------------------


def test_review_role_section_teaches_the_merge_template():
    lines = _review_role_section(run_id="abc123")
    text = "\n".join(lines)
    assert 'MERGE (r:ReviewRole {' in text
    assert 'run_id: "abc123"' in text
    assert '"self_review|hire_cursor_review"' in text
    assert "self_review" in text and "hire_cursor_review" in text
    # writing nothing must be explicitly framed as fine, same as InvestigationRole
    assert "self_review" in text.split("Writing nothing is fine")[-1]
    # Elevated local queue → offload grading; never "backed up ⇒ stay self_review".
    lower = text.lower()
    assert "prefer hire_cursor_review" in lower or "prefer hire_cursor_review (" in lower
    assert "prefer self_review, not hire_cursor_review" not in lower
    assert "reason to prefer self_review" not in lower


def _material() -> StudyMaterial:
    from datetime import datetime, timezone

    return StudyMaterial(generated_at=datetime(2026, 9, 20, tzinfo=timezone.utc))


def test_build_kickoff_prompt_includes_review_role_section_when_contractor_peer_enabled():
    text = build_kickoff_prompt(
        _material(), run_id="abc123", graph_enabled=True, contractor_peer_enabled=True
    )
    assert "WHO GRADES THIS SITTING'S HOPS" in text
    assert 'MERGE (r:ReviewRole {' in text
    # comes after the investigation-role teach, not instead of it
    assert text.index("YOUR ROLE FOR THIS SITTING") < text.index("WHO GRADES THIS SITTING'S HOPS")


def test_build_kickoff_prompt_omits_review_role_section_by_default():
    # contractor_peer_enabled defaults False -- same gate :InvestigationRole/
    # HelpRequest teach already uses; the review-role teach rides on it too.
    text = build_kickoff_prompt(_material(), run_id="abc123", graph_enabled=True)
    assert "WHO GRADES THIS SITTING'S HOPS" not in text
    assert "ReviewRole" not in text


def test_build_kickoff_prompt_omits_review_role_section_when_not_writable():
    text = build_kickoff_prompt(
        _material(), run_id="abc123", graph_enabled=False, contractor_peer_enabled=True
    )
    assert "WHO GRADES THIS SITTING'S HOPS" not in text
    assert "ReviewRole" not in text
