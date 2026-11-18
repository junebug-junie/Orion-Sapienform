"""Dream lookups: both kinds, blind-experiment rules, empty vs counts, caps."""
import inspect
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
    assert item.text == "A dream.\n\nIt went on." and item.occurred_at.utcoffset() == timedelta(0)
    assert item.occurred_at == (NOW - timedelta(hours=5))
    assert item.extra["dream_date"] == "2026-09-28"
    assert len(item.extra["themes"]) == 8 and len(item.extra["themes"][0]) == 80


def test_non_utc_aware_timestamps_normalize_to_utc():
    denver = timezone(timedelta(hours=-6))
    row = hypothesis("dh-aaa111", 1)
    row["occurred_at"] = datetime(2026, 9, 29, 0, 30, tzinfo=denver)
    [item] = dq.recent(FakeConn([], [row]), kind=None, since=None, limit=5, now=NOW).items
    assert item.occurred_at.utcoffset() == timedelta(0)
    assert item.occurred_at == datetime(2026, 9, 29, 6, 30, tzinfo=timezone.utc)


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


_READS_HYPOTHESES = re.compile(r"\b(?:from|join)\s+dream_hypothesis\b", re.IGNORECASE)


def test_no_hypothesis_statement_escapes_the_pin():
    module_sql = [v for v in vars(dq).values() if isinstance(v, str) and _READS_HYPOTHESES.search(v)]
    assert module_sql and all(sql in dq.HYPOTHESIS_SQL for sql in module_sql)
    assert len(_READS_HYPOTHESES.findall(inspect.getsource(dq))) == len(dq.HYPOTHESIS_SQL)


_N_SINCE = "(CAST(:since AS timestamptz) IS NULL OR (d.created_at AT TIME ZONE 'UTC') >= CAST(:since AS timestamptz))"
_H_SINCE = "(CAST(:since AS timestamptz) IS NULL OR h.offered_at >= CAST(:since AS timestamptz))"


def test_every_recent_and_by_ids_statement_filters_on_since():
    pins = {
        "NARRATIVE_RECENT_SQL": _N_SINCE, "NARRATIVE_BY_IDS_SQL": _N_SINCE,
        "HYPOTHESIS_RECENT_SQL": _H_SINCE, "HYPOTHESIS_BY_IDS_SQL": _H_SINCE,
    }
    for name, predicate in pins.items():
        assert predicate in getattr(dq, name), name


def test_recent_and_by_ids_pass_the_callers_since_to_every_query():
    since = NOW - timedelta(hours=12)
    conn = FakeConn([narrative(19, 5)], [hypothesis("dh-aaa111", 1)])
    dq.recent(conn, kind=None, since=since, limit=5, now=NOW)
    dq.by_ids(conn, [("dream:19", 0.9), ("dh-aaa111", 0.8)], kind=None, since=since, limit=5, now=NOW)
    assert len(conn.calls) == 4
    assert all(params["since"] == since for _, params in conn.calls)


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
    hypothesis_sql = [sql for sql, _ in conn.calls if _READS_HYPOTHESES.search(sql)]
    assert hypothesis_sql and all(sql in dq.HYPOTHESIS_SQL for sql in hypothesis_sql)
    assert all("h.offered_at IS NOT NULL" in sql for sql in hypothesis_sql)


def test_split_ids_ignores_malformed_and_non_ascii_digit_ids():
    assert dq.split_ids(["dream:3", "dh-abc123", "dream:x", "dream:²", "dream:"]) == ([3], ["dh-abc123"])


MCP_TOOL_RESULT_MAX_CHARS = 12000


def test_worst_case_dream_results_fit_the_mcp_tool_result_budget():
    import json

    accent = "é" * 10_000
    big_narratives = [narrative(i, i, tldr=accent, story=accent, themes=[accent] * 20) for i in range(1, 6)]
    big_hypotheses = [
        {**hypothesis(f"dh-{i:06x}", i), "claim": accent, "why": accent, "cycle_id": accent} for i in range(1, 6)
    ]
    conn = FakeConn(big_narratives, big_hypotheses)
    results = [
        dq.recent(conn, kind="narrative", since=None, limit=5, now=NOW),
        dq.recent(conn, kind="hypothesis", since=None, limit=5, now=NOW),
        dq.one(conn, "dream:1", now=NOW),
    ]
    for result in results:
        assert result.items
        assert len(json.dumps(result.model_dump(mode="json"), ensure_ascii=False)) < MCP_TOOL_RESULT_MAX_CHARS
