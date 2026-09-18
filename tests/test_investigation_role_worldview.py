"""RO read of Orion-authored :InvestigationRole — Hub never writes these nodes."""

from __future__ import annotations

import pytest

from orion.curiosity.worldview import (
    LABEL_INVESTIGATION_ROLE,
    WorldviewUnavailable,
    latest_investigation_role,
    list_investigation_roles_for_run_cypher,
    read_investigation_roles,
)


class _FakeReader:
    def __init__(self, *, rows=None, raises=False) -> None:
        self.rows = rows or []
        self.raises = raises
        self.queries: list[str] = []

    def query(self, cypher: str):
        self.queries.append(cypher)
        if self.raises:
            raise WorldviewUnavailable("ConnectionError: nope")
        return list(self.rows)


def test_latest_investigation_role_wins() -> None:
    reader = _FakeReader(
        rows=[
            {
                "run_id": "abcd1234abcd",
                "choice": "local_crawl",
                "why": "try the graph first",
                "written_at": 100,
            },
            {
                "run_id": "abcd1234abcd",
                "choice": "hire_cursor",
                "why": "need a repo look after a short crawl",
                "written_at": 200,
            },
        ]
    )
    roles = read_investigation_roles(reader, "abcd1234abcd")
    latest = latest_investigation_role(roles)
    assert latest is not None
    assert latest.choice == "hire_cursor"
    assert latest.why == "need a repo look after a short crawl"
    assert latest.run_id == "abcd1234abcd"
    assert latest.written_at == 200
    assert roles[0].choice == "local_crawl"


def test_investigation_role_cypher_is_read_only_and_parameterized() -> None:
    cypher = list_investigation_roles_for_run_cypher("abcd1234abcd")
    assert LABEL_INVESTIGATION_ROLE == "InvestigationRole"
    assert "MATCH (r:InvestigationRole)" in cypher
    assert "r.run_id = 'abcd1234abcd'" in cypher
    assert "ORDER BY r.written_at ASC" in cypher
    assert "MERGE" not in cypher
    assert "CREATE" not in cypher
    assert "SET " not in cypher
    assert "DELETE" not in cypher


@pytest.mark.parametrize("bad", ["'; MATCH (n) DETACH DELETE n //", "abc-123", "", "ABC123", None])
def test_investigation_role_cypher_refuses_non_hex_run_id(bad) -> None:
    with pytest.raises(ValueError):
        list_investigation_roles_for_run_cypher(bad)


def test_unreadable_investigation_roles_are_empty() -> None:
    assert read_investigation_roles(_FakeReader(raises=True), "abcd1234abcd") == []
    assert read_investigation_roles(_FakeReader(), "not-hex") == []
    assert latest_investigation_role([]) is None
