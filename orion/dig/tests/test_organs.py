"""`dig organs <date>`: the daily dig roundup (integration Part A.7). Every line is a count, a quote or a ref;
the footer is the coverage report's own number; breaks, reads and cites are told apart; failures are named."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.dig import __main__ as dig
from orion.dig import organs
from orion.integration.organs import load_organs
from orion.temporal_self.now import parse_organs

REPO = Path(__file__).resolve().parents[3]
DAY = date(2026, 11, 15)

# A small registry: three organs, one that reads the now, one exempt, one that emits nothing.
REG = parse_organs({"organs": {
    "reverie": {"temporal_sources": ["reverie_chain"], "families": ["thought.reverie"],
                "reads_now": "services/orion-thought/app/temporal_self_cue.py"},
    "dream": {"temporal_sources": ["dream_cycle", "dream_hypothesis"], "families": ["dream.cycle"],
              "reads_now": None, "exempt_read": "not yet migrated; integration patch 7"},
    "prime": {"temporal_sources": ["prime_thought"], "families": ["prime"],
              "reads_now": "services/orion-mind/app/prime_loop.py"},
    "personality": {"temporal_sources": [], "exempt_emit": "later", "reads_now": None, "exempt_read": "later"},
}})


def _history(kind: str, per_day: int, days: int = 14) -> list[dict]:
    return [{"source_kind": kind, "day": (DAY - timedelta(days=i)).isoformat(), "n": str(per_day)}
            for i in range(1, days + 1)]


def _rows() -> dict[str, list[dict]]:
    return {
        "kinds": [{"source_kind": "reverie_chain", "n": "5"}, {"source_kind": "dream_cycle", "n": "2"},
                  {"source_kind": "prime_thought", "n": "1"}],
        "uses": [("e-dream-1", "dream_cycle", "reverie")],
        "cites_made": [{"citing_organ": "reverie", "n": "1"}],
        "reads": [{"reader_organ": "reverie", "reads": "4", "reads_no_cite": "2", "reads_unclosed": "1"},
                  {"reader_organ": "prime", "reads": "2", "reads_no_cite": "2"}],
        "top": [
            {"event_id": "e-rev-lo", "source_kind": "reverie_chain", "source_table": "substrate_reverie_chain",
             "source_ref": "c1", "label": "open-loop-a", "verdict": "", "salience": "0.20",
             "occurred_at": "2026-11-15 10:00:00+00", "payload": "{}"},
            {"event_id": "e-rev-hi", "source_kind": "reverie_chain", "source_table": "substrate_reverie_chain",
             "source_ref": "c2", "label": "open-loop-b", "verdict": "", "salience": "0.90",
             "occurred_at": "2026-11-15 09:00:00+00", "payload": "{}"},
            {"event_id": "e-dream-1", "source_kind": "dream_cycle", "source_table": "dream_cycle",
             "source_ref": "dc-1", "label": "pressure", "verdict": "", "salience": "",
             "occurred_at": "2026-11-15 08:00:00+00", "payload": "{}"},
            {"event_id": "e-prime-1", "source_kind": "prime_thought", "source_table": "grammar_acts",
             "source_ref": "act_abc", "label": 'woke: quiet: nothing has "woken" me', "verdict": "",
             "salience": "", "occurred_at": "2026-11-15 12:00:00+00", "payload": "{}"},
        ],
        "history": _history("dream_hypothesis", 8) + _history("reverie_chain", 4)
        + [{"source_kind": "prime_thought", "day": (DAY - timedelta(days=1)).isoformat(), "n": "1"}],
        "acts": [{"actor": "dream.cycle", "verb": "make", "opened": "2", "closed": "2"},
                 {"actor": "prime", "verb": "infer", "opened": "3", "closed": "3"},
                 {"actor": "mystery.actor", "verb": "make", "opened": "4", "closed": "1"}],
        "prime_wakes": [{"act_id": "a1", "outcome_note": "woke: quiet: nothing has woken me for 6 hours | no move"},
                        {"act_id": "a2", "outcome_note": "woke: need:horizon concern:9f | moved"},
                        {"act_id": "a3", "outcome_note": "woke: a chat with Juniper turn made a promise"}],
        "cited": [{"event_id": "e-dream-1", "citing_organ": "reverie", "citing_ref": "thought:1",
                   "source_kind": "dream_cycle", "label": "pressure", "verdict": "", "payload": "{}"}],
        "notify": [{"severity": "critical", "status": "sent", "n": "4"},
                   {"severity": "info", "status": "no_email", "n": "6"}],
        "notify_titles": [{"severity": "critical", "title": "node_availability:circe", "n": "4"}],
    }


def _build(rows=None, unread=None):
    rows = _rows() if rows is None else rows
    rows = {**rows, "uses": [{"event_id": e, "source_kind": k, "citing_organ": c} for e, k, c in rows.get("uses", [])]}
    return organs.build(DAY, rows, REG, unread)


# --- the line rule -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("line", [
    "The camera was busy today.",
    "Orion had a reflective, integrated day across organs",
    "  write-only: nothing was cited by anyone",
])
def test_an_unquoted_summary_line_is_refused(line):
    assert organs.line_kind(line) is None
    with pytest.raises(ValueError, match="not a count, a quote or a ref"):
        organs.check_lines(["organs 2026-11-15: 3 organs", line])


@pytest.mark.parametrize("line,kind", [
    ("camera: 571 experiences", "count"),
    ('  did not read the now (registry: "not yet migrated")', "quote"),
    ("  see [experience:reverie_chain:abc]", "ref"),
    ("  the act [act:act_123]", "ref"),
])
def test_count_quote_and_ref_lines_pass(line, kind):
    assert organs.line_kind(line) == kind
    organs.check_lines([line, ""])  # blank lines are layout


def test_every_built_line_passes_the_rule_even_for_wordy_or_numberless_labels():
    rows = _rows()
    rows["top"][0]["label"] = "a quiet afternoon of thinking"  # no digit: still a quote with a ref
    r = _build(rows)
    organs.check_lines(r.lines)  # build() also runs it on its own output
    assert all(organs.line_kind(x) for x in r.lines)


# --- what each organ did --------------------------------------------------------------------------------------


def test_counts_top_items_by_salience_and_refs():
    r = _build()
    text = "\n".join(r.lines)
    assert "reverie: 5 experiences (reverie_chain 5)" in text
    rev = [x for x in r.lines if "open-loop" in x]
    assert "e-rev-hi" in rev[0] and "salience 0.90" in rev[0]  # highest salience first
    assert "e-rev-lo" in rev[1]
    assert "dream: 2 experiences (dream_cycle 2); 2 acts opened, 2 closed (dream.cycle make 2)" in text
    prime_item = next(x for x in r.lines if "e-prime-1" in x)
    assert "[act:act_abc]" in prime_item  # the source row is an act: `dig act` opens it
    assert "'woken'" in prime_item and prime_item.count('"') == 2  # inner quotes never break the quote
    assert {"experience:e-rev-hi", "experience:e-prime-1", "act:act_abc"} <= set(r.refs)


def test_read_and_cited_nothing_is_told_apart_from_did_not_read():
    lines = _build().lines
    # prime's row predates reads_unclosed (an old query): it reads as 0 never closed
    assert ("  read the now 2 times; 2 of those reads cited nothing; 0 never closed (no model call used them); "
            "0 valid cites") in lines  # prime
    assert ("  read the now 4 times; 2 of those reads cited nothing; 1 never closed (no model call used them); "
            "1 valid cites") in lines  # reverie
    assert '  did not read the now (registry: "not yet migrated; integration patch 7")' in lines  # dream


def test_what_it_cited_who_cited_it_and_closed_or_write_only():
    lines = _build().lines
    assert '  cited from dream: "dream cycle: pressure" [experience:e-dream-1]' in lines
    assert "  closed: 1 of its 2 experiences cited by another organ (today's cites by reverie 1)" in lines
    assert "  write-only: 0 of its 5 experiences cited by another organ" in lines  # reverie itself


def test_break_is_zero_where_the_organs_own_history_usually_has_some():
    lines = _build().lines
    assert "  break: 0 dream_hypothesis today; usual 8 a day (median of the 14 days before)" in lines
    # prime_thought had 1 day of 14: median 0, so 0 today is no break
    assert not any("break: 0 prime_thought" in x for x in lines)


def test_prime_thoughts_are_counted_by_what_woke_them_including_a_need():
    assert organs.wake_reason("woke: need:horizon concern:9f | moved") == "need:horizon"
    assert organs.wake_reason("woke: need:connection: chat abc") == "need:connection"
    assert organs.wake_reason("woke: quiet: nothing has woken me for 6 hours") == "quiet floor"
    assert organs.wake_reason("woke: my letter for 2026-11-14") == "a change"
    assert organs.wake_reason("") == "still open"
    line = next(x for x in _build().lines if x.startswith("  thoughts by what woke them"))
    assert "a change 1" in line and "need:horizon 1" in line and "quiet floor 1" in line


def test_quiet_organs_unclaimed_actors_and_unread_are_named_never_empty():
    r = _build(unread={"acts": "RuntimeError: needs GRANT SELECT ON grammar_acts TO orion_readonly"})
    text = "\n".join(r.lines)
    assert "quiet organs (0 experiences, 0 acts, 0 reads, 0 cites): 1 (personality)" in text
    assert "acts by actors no organ claims: mystery.actor 4" in text
    assert 'UNREAD acts: "RuntimeError: needs GRANT SELECT ON grammar_acts TO orion_readonly"' in text


def test_footer_is_the_coverage_reports_number_and_the_digest_counts():
    rows = _rows()
    r = _build(rows)
    spec = importlib.util.spec_from_file_location("rc", REPO / "scripts" / "report_integration_coverage.py")
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    lo, hi = (datetime.fromisoformat(x) for x in organs.day_bounds(DAY))
    rep = rc.build_report(kind_counts={"reverie_chain": 5, "dream_cycle": 2, "prime_thought": 1},
                          uses=rows["uses"], cites_made={"reverie": 1}, reads={}, lo=lo, hi=hi, registry=REG,
                          missing_tables=[])
    assert r.use_rate == rep["experience_use_rate"] and r.closed_organs == rep["closed_organs"]
    text = "\n".join(r.lines)
    assert (f"experience use rate: {rep['used_by_other']} / {rep['experiences']} = 12.50% (cited by another organ "
            "within 24 h); closed organs: 1 of 4 (dream)") in text
    assert ("notifications (the retired Daily Digest's counts): 10 requests; by severity info 6, critical 4; "
            "by status no_email 6, sent 4") in text
    assert '  critical x4: "node_availability:circe"' in text


# --- the SQL ------------------------------------------------------------------------------------------------


def test_every_query_is_one_read_only_select_over_the_half_open_denver_day():
    qs = organs.queries(DAY)
    assert set(qs) == {"kinds", "uses", "cites_made", "reads", "top", "history", "acts", "prime_wakes", "cited",
                       "notify", "notify_titles"}
    a, b = organs.day_bounds(DAY)
    assert (a, b) == ("2026-11-15T07:00:00+00:00", "2026-11-16T07:00:00+00:00")
    for name, sql in qs.items():
        s = " ".join(sql.split()).upper()
        assert s.startswith(("SELECT", "WITH")) and ";" not in sql, name
        for word in ("INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "GRANT "):
            assert word not in s, (name, word)
        assert "%(" not in sql, name  # the coverage SQL's params are bound
        if name != "history":
            assert a in sql and b in sql, name


def test_dst_day_is_one_local_day():
    a, b = organs.day_bounds(date(2026, 11, 1))  # fall back: a 25-hour day
    assert (datetime.fromisoformat(b) - datetime.fromisoformat(a)) == timedelta(hours=25)


def test_window_rules():
    organs.check_window(DAY, DAY)
    with pytest.raises(ValueError, match="at most 7"):
        organs.check_window(DAY - timedelta(days=7), DAY)
    with pytest.raises(ValueError, match="no Temporal Self record"):
        organs.check_window(date(2026, 9, 1), date(2026, 9, 2))


# --- the readers ------------------------------------------------------------------------------------------


def _serve(sql_to_rows: dict[str, list[dict]], fail: set[str] = frozenset()):
    by_sql = {sql: name for name, sql in organs.queries(DAY).items()}

    def fetch(sql):
        name = by_sql[sql]
        if name in fail:
            raise RuntimeError(f"needs GRANT SELECT ON {name}")
        return sql_to_rows.get(name, [])
    return fetch


def test_read_sync_names_a_failed_query_and_keeps_the_rest():
    rows = _rows()
    rows["uses"] = [{"event_id": e, "source_kind": k, "citing_organ": c} for e, k, c in rows["uses"]]
    r = organs.read_sync(DAY, _serve(rows, fail={"notify", "notify_titles"}), REG)
    assert set(r.unread) == {"notify", "notify_titles"}
    assert any(x.startswith("UNREAD notify:") for x in r.lines)
    assert any(x.startswith("reverie: 5 experiences") for x in r.lines)


def test_read_async_uses_the_same_queries():
    rows = _rows()
    rows["uses"] = [{"event_id": e, "source_kind": k, "citing_organ": c} for e, k, c in rows["uses"]]
    fetch = _serve(rows)

    class Conn:
        async def fetch(self, sql):
            return fetch(sql)

    r = asyncio.run(organs.read_async(Conn(), DAY, REG))
    assert r.lines == organs.read_sync(DAY, fetch, REG).lines and not r.unread


def test_the_real_registry_builds_a_roundup_for_every_organ():
    reg = load_organs()
    r = organs.build(DAY, {}, reg)
    organs.check_lines(r.lines)
    assert f"quiet organs (0 experiences, 0 acts, 0 reads, 0 cites): {len(reg.organs)}" in r.lines[1]


# --- the CLI ----------------------------------------------------------------------------------------------


class _Runner:
    """ReadOnlyPsql stand-in: CSV out, keyed by which query the SQL is."""

    def __init__(self, rows):
        self.by_sql = {}
        for d in (DAY, DAY - timedelta(days=1)):
            self.by_sql.update({sql: name for name, sql in organs.queries(d).items()})
        self.rows = rows
        self.calls = 0

    def run(self, sql, flags):
        import csv
        import io
        self.calls += 1
        rows = self.rows.get(self.by_sql[sql], [])
        if not rows:
            return "\n", 0
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
        return buf.getvalue(), 0


def test_cmd_organs_prints_the_capped_roundup_and_writes_every_line(tmp_path, monkeypatch):
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path))
    monkeypatch.setattr("orion.integration.organs.load_organs", lambda: REG)
    rows = _rows()
    rows["uses"] = [{"event_id": e, "source_kind": k, "citing_organ": c} for e, k, c in rows["uses"]]
    runner = _Runner(rows)
    text = dig.cmd_organs(DAY - timedelta(days=1), DAY, dig.Out("t"), runner=runner)
    assert runner.calls == 2 * len(organs.queries(DAY))
    assert text.splitlines()[0].startswith("organs 2026-11-14")
    assert "read 2 day(s) in" in text and "full roundup" in text
    path = Path(text.rsplit("full roundup (every organ, every line): ", 1)[1].split(";")[0])
    saved = json.loads(path.read_text())
    assert [d["day"] for d in saved] == ["2026-11-14", "2026-11-15"]
    assert len(text) <= dig.MAX_CHARS + 300


def test_cli_routes_organs_and_refuses_a_long_window(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path))
    assert "organs <date> [<end>]" in dig.HELP
    code = dig.main(["organs", "2026-11-01", "2026-11-12"])
    assert code == 1 and "at most 7 days" in capsys.readouterr().err


def test_a_dream_hypothesis_is_counted_and_quoted_but_its_id_never_printed():
    # The dream experiment's blind rule: a hypothesis id is the scorecard's join key.
    rows = _rows()
    hyp = {"event_id": "dream_hypothesis:dh-1a69e4d982ac", "source_kind": "dream_hypothesis",
           "source_table": "dream_hypothesis", "source_ref": "dh-1a69e4d982ac", "label": "",
           "verdict": "", "salience": "0.99", "occurred_at": "2026-11-15 11:00:00+00", "payload": "{}"}
    rows["top"].append(hyp)
    rows["kinds"].append({"source_kind": "dream_hypothesis", "n": "3"})
    rows["cited"].append({"event_id": hyp["event_id"], "citing_organ": "reverie", "citing_ref": "t:2",
                          "source_kind": "dream_hypothesis", "label": "", "verdict": "", "payload": "{}"})
    # and an id quoted inside another row's own label is withheld too
    rows["top"][0]["label"] = "thinking about dream_hypothesis:dh-1a69e4d982ac again"
    r = _build(rows)
    text = "\n".join(r.lines)
    assert "dh-1a69e4d982ac" not in text and not any("dream_hypothesis:dh" in x for x in r.refs)
    assert organs.BLIND_REF in text and "dream: 5 experiences (dream_hypothesis 3, dream_cycle 2)" in text
    organs.check_lines(r.lines)


def test_ties_order_the_same_from_csv_text_and_typed_rows():
    rows = _rows()
    for r in rows["top"]:
        if r["source_kind"] == "reverie_chain":
            r["salience"] = "0.5"
    typed = json.loads(json.dumps(rows))
    for r in typed["top"]:
        r["occurred_at"] = datetime.fromisoformat(r["occurred_at"].replace(" ", "T") + ":00")
        r["salience"] = float(r["salience"]) if r["salience"] else None
    assert _build(rows).lines == _build(typed).lines
    assert "e-rev-lo" in next(x for x in _build(rows).lines if "open-loop" in x)  # newer first on a tie
