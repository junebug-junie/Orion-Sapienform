"""Dream lookups: both kinds, blind-experiment rules, empty vs counts, caps."""
import re
from datetime import date, datetime, timedelta, timezone

from app import introspect_dreams as dq

NOW = datetime(2026, 9, 29, 6, 0, tzinfo=timezone.utc)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class FakeConn:
    """Returns given rows (newest first) by table; honors ids and limit like the SQL does."""

    def __init__(self, narratives=(), hypotheses=()):
        self.tables = {"dreams": list(narratives), "dream_hypothesis": list(hypotheses)}
        self.calls = []

    def execute(self, clause, params=None):
        sql, params = str(clause), dict(params or {})
        self.calls.append((sql, params))
        table = "dream_hypothesis" if "FROM dream_hypothesis" in sql else "dreams" if "FROM dreams" in sql else None
        if table is None:
            return _Result([])
        rows = self.tables[table]
        if "ids" in params:
            key = "hypothesis_id" if table == "dream_hypothesis" else "id"
            rows = [r for r in rows if r[key] in params["ids"]]
        total = len(rows)
        return _Result([{**r, "total": total} for r in rows[: params.get("limit", len(rows))]])


def narrative(i, hours_ago, *, tldr="A dream.", story="It went on.", themes=("vision",)):
    return {
        "id": i, "dream_date": date(2026, 9, 28), "tldr": tldr, "narrative": story,
        "themes": list(themes), "occurred_at": NOW - timedelta(hours=hours_ago),
    }


def hypothesis(hid, hours_ago, *, expires_in_hours=24, why="Both mention RPC."):
    return {
        "hypothesis_id": hid, "cycle_id": "dc-1", "claim": f"claim {hid}", "why": why,
        "occurred_at": NOW - timedelta(hours=hours_ago), "expires_at": NOW + timedelta(hours=expires_in_hours),
    }


def test_recent_merges_both_kinds_newest_first_and_counts_the_window():
    conn = FakeConn([narrative(19, 5), narrative(18, 50)], [hypothesis("dh-aaa111", 1), hypothesis("dh-bbb222", 20)])
    result = dq.recent(conn, kind=None, since=None, limit=3, now=NOW)
    assert result.ok and result.operation == "dreams" and result.total_available == 4
    assert [i.id for i in result.items] == ["dh-aaa111", "dream:19", "dh-bbb222"]
    assert {i.kind for i in result.items} == {"dream_narrative", "dream_hypothesis"}
    assert all(i.epistemic_status == "unsettled" for i in result.items)


def test_recent_kind_filter_queries_one_table():
    conn = FakeConn([narrative(19, 5)], [hypothesis("dh-aaa111", 1)])
    result = dq.recent(conn, kind="narrative", since=None, limit=5, now=NOW)
    assert [i.id for i in result.items] == ["dream:19"]
    assert not any("dream_hypothesis" in sql for sql, _ in conn.calls)


def test_empty_window_is_ok_and_zero():
    result = dq.recent(FakeConn(), kind=None, since=NOW, limit=5, now=NOW)
    assert result.ok and result.items == [] and result.total_available == 0


def test_narrative_item_shape_and_utc():
    row = narrative(19, 5, themes=["x" * 200] + [f"t{i}" for i in range(12)])
    row["occurred_at"] = row["occurred_at"].replace(tzinfo=None)
    [item] = dq.recent(FakeConn([row]), kind=None, since=None, limit=5, now=NOW).items
    assert item.text == "A dream.\n\nIt went on." and item.occurred_at.tzinfo is not None
    assert item.extra["dream_date"] == "2026-09-28"
    assert len(item.extra["themes"]) == 8 and len(item.extra["themes"][0]) == 80


def test_hypothesis_item_has_no_arm_or_refs_and_flags_expiry():
    conn = FakeConn([], [hypothesis("dh-aaa111", 1, expires_in_hours=-1)])
    [item] = dq.recent(conn, kind=None, since=None, limit=5, now=NOW).items
    assert item.text == "claim dh-aaa111\nWhy: Both mention RPC."
    assert set(item.extra) == {"cycle_id", "expired"} and item.extra["expired"] is True


def test_every_hypothesis_statement_is_offered_only_and_never_selects_arm_or_refs():
    assert dq.HYPOTHESIS_SQL
    for sql in dq.HYPOTHESIS_SQL:
        assert "h.offered_at IS NOT NULL" in sql
        selected = re.search(r"SELECT(.*?)FROM", sql, re.S).group(1)
        for column in ("arm", "ref_a", "ref_b"):
            assert not re.search(rf"\b{column}\b", selected), (column, sql)


def test_no_hypothesis_statement_escapes_the_pin():
    module_sql = [v for v in vars(dq).values() if isinstance(v, str) and "FROM dream_hypothesis" in v]
    assert module_sql and all(sql in dq.HYPOTHESIS_SQL for sql in module_sql)


def test_by_ids_keeps_rank_order_attaches_similarity_and_drops_missing():
    conn = FakeConn([narrative(19, 5)], [hypothesis("dh-aaa111", 1)])
    scored = [("dh-aaa111", 0.81), ("dream:7", 0.8), ("dream:19", 0.7)]
    result = dq.by_ids(conn, scored, kind=None, since=None, limit=5, now=NOW)
    assert [(i.id, i.extra["similarity"]) for i in result.items] == [("dh-aaa111", 0.81), ("dream:19", 0.7)]
    assert result.total_available == 2


def test_by_ids_respects_kind_and_limit():
    conn = FakeConn([narrative(19, 5), narrative(18, 6)], [hypothesis("dh-aaa111", 1)])
    scored = [("dh-aaa111", 0.9), ("dream:19", 0.8), ("dream:18", 0.7)]
    result = dq.by_ids(conn, scored, kind="narrative", since=None, limit=1, now=NOW)
    assert [i.id for i in result.items] == ["dream:19"] and result.total_available == 2


def test_by_ids_skips_empty_id_lists():
    conn = FakeConn([narrative(19, 5)])
    dq.by_ids(conn, [("dream:19", 0.9)], kind=None, since=None, limit=5, now=NOW)
    assert not any("dream_hypothesis" in sql for sql, _ in conn.calls)


def test_one_returns_full_text_or_empty():
    long_story = "s" * 5000
    conn = FakeConn([narrative(19, 5, story=long_story)], [hypothesis("dh-aaa111", 1)])
    [item] = dq.one(conn, "dream:19", now=NOW).items
    assert len(item.text) == 4000 and item.truncated
    assert dq.one(conn, "dh-aaa111", now=NOW).items[0].id == "dh-aaa111"
    missing = dq.one(conn, "dh-fffffff", now=NOW)
    assert missing.ok and missing.items == [] and missing.total_available == 0


def test_index_rows_pairs_kinds_and_scans_offered_hypotheses_only():
    conn = FakeConn([narrative(19, 5)], [hypothesis("dh-aaa111", 1)])
    pairs = dq.index_rows(conn)
    assert [(k, dq.doc_id(k, r)) for k, r in pairs] == [("narrative", "dream:19"), ("hypothesis", "dh-aaa111")]
    assert dq.split_ids(["dream:3", "dh-abc123", "dream:x"]) == ([3], ["dh-abc123"])
