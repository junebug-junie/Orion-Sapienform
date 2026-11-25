"""scripts/lane_door_report.py: the evidence Juniper reads before a lane's door may refuse (Phase 3)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import lane_door_report as rep  # noqa: E402


def _line(day: str, state: str = "leased", lane: str = "chat", lease: str | None = "L1", src: str = "100.92.216.81",
          path: str = "/v1/chat/completions") -> str:
    return json.dumps({"ts": f"{day}T12:00:00.000+00:00", "lane": lane, "src": src, "method": "POST", "path": path,
                       "lease": state,
                       "lease_id": lease if state == "leased" else ("invalid" if state == "invalid" else None), "status": 200, "ms": 3.0})


def _days(start: int, n: int) -> list[str]:
    return [f"2026-11-{d:02d}" for d in range(start, start + n)]


def _verdict(lines, **kw):
    report, _ = rep.aggregate(lines)
    return rep.verdict(report["chat"], streak_days=7, max_gap_pct=1.0, joined=kw.pop("joined", False),
                       today=kw.pop("today", "2026-11-22"))


def test_seven_clean_days_without_the_lease_join_are_not_ready():
    ok, why = _verdict([_line(d) for d in _days(15, 7)])
    assert not ok and "lease join was not run" in why


def test_one_unleased_call_in_the_window_is_not_ready_and_names_the_day():
    lines = [_line(d) for d in _days(15, 7)] + [_line("2026-11-19", state="unleased", src="100.92.216.81")]
    ok, why = _verdict(lines)
    assert not ok and why.startswith("2026-11-19") and "1 unleased" in why


def test_a_missing_day_is_unknown_not_zero():
    lines = [_line(d) for d in _days(15, 7) if d != "2026-11-18"]
    ok, why = _verdict(lines)
    assert not ok and "2026-11-18: no log" in why


def test_today_is_partial_and_never_counts():
    ok, _ = _verdict([_line(d) for d in _days(16, 7)], today="2026-11-22")   # 16..22, 22 is today
    assert not ok


def test_invalid_lease_calls_block_like_unleased_ones():
    lines = [_line(d) for d in _days(15, 7)] + [_line("2026-11-21", state="invalid")]
    assert not _verdict(lines)[0]


def _ok_streak_joined():
    report, _ = rep.aggregate([_line(d) for d in _days(15, 7)])
    for d in report["chat"].values():
        d.unmatched_leases, d.gateway_calls = 0, 1
    return report


def test_a_day_with_no_leased_traffic_proves_nothing():
    lines = [_line(d) for d in _days(15, 7) if d != "2026-11-20"] + [_line("2026-11-20", path="/x", lane="fast")]
    report, _ = rep.aggregate(lines)
    report["chat"]["2026-11-20"] = rep.Day()
    ok, why = rep.verdict(report["chat"], streak_days=7, max_gap_pct=1.0, joined=False, today="2026-11-22")
    assert not ok and "no leased calls" in why


def test_joined_gaps_and_unmatched_leases_block():
    report, _ = rep.aggregate([_line(d) for d in _days(15, 7)])
    for d in report["chat"].values():
        d.unmatched_leases, d.gateway_calls = 0, 1
    assert rep.verdict(report["chat"], streak_days=7, max_gap_pct=1.0, joined=True, today="2026-11-22")[0]
    report["chat"]["2026-11-17"].gateway_calls = 3
    ok, why = rep.verdict(report["chat"], streak_days=7, max_gap_pct=1.0, joined=True, today="2026-11-22")
    assert not ok and "gateway sent 3" in why
    report["chat"]["2026-11-17"].gateway_calls = 0
    ok, why = rep.verdict(report["chat"], streak_days=7, max_gap_pct=1.0, joined=True, today="2026-11-22")
    assert not ok and "recorded none" in why
    report["chat"]["2026-11-17"].gateway_calls = 1
    report["chat"]["2026-11-18"].unmatched_leases = 2
    ok, why = rep.verdict(report["chat"], streak_days=7, max_gap_pct=1.0, joined=True, today="2026-11-22")
    assert not ok and "not in gpu_pool_events" in why


def test_lease_ids_are_collected_and_rows_from_the_old_keyed_door_are_counted_apart():
    old = json.dumps({"ts": "2026-11-15T12:00:00.000+00:00", "lane": "chat", "key": "ok", "lease_id": "L9"})
    counts: dict = {}
    report, bad = rep.aggregate([_line("2026-11-15", lease="L1"), _line("2026-11-15", lease="L2"), old], counts)
    day = report["chat"]["2026-11-15"]
    assert day.lease_ids == {"L1", "L2"} and day.leased == 2 and bad == 0 and counts == {"pre_revision": 1}


def test_render_names_who_sent_unleased_calls(capsys, tmp_path):
    f = tmp_path / "chat-2026-11-15.jsonl"
    f.write_text("\n".join([_line("2026-11-15"), _line("2026-11-15", state="unleased", src="100.92.216.81",
                                                         path="/v1/chat/completions"), "not json"]) + "\n")
    assert rep.main([str(tmp_path), "--today", "2026-11-16"]) == 0
    out = capsys.readouterr().out
    assert "NOT READY" in out and "unleased x1: 100.92.216.81 POST /v1/chat/completions" in out
    assert "1 unreadable lines skipped" in out
    assert "== lane agent-gpu2: NOT READY" in out and "no log lines for this lane" in out


def test_calls_an_enforcing_door_already_refused_are_shown_but_do_not_block():
    lines = [_line(d) for d in _days(15, 7)]
    refused = json.loads(_line("2026-11-19", state="unleased"))
    refused.update(status=401, refused=True)
    report, _ = rep.aggregate(lines + [json.dumps(refused)])
    day = report["chat"]["2026-11-19"]
    assert (day.refused, day.unleased) == (1, 0)
    ok, why = rep.verdict(report["chat"], streak_days=7, max_gap_pct=1.0, joined=False, today="2026-11-22")
    assert "lease join was not run" in why            # the refused call did not block the streak


def test_a_call_sent_again_under_one_lease_does_not_open_a_gap():
    """#3919: the gateway counts one call per lease; a re-send under the same lease reaches the door twice."""
    lines = [_line(d, lease=f"L-{d}") for d in _days(15, 7)] + [_line("2026-11-17", lease="L-2026-11-17")]
    report, _ = rep.aggregate(lines)
    for d in report["chat"].values():
        d.unmatched_leases, d.gateway_calls = 0, 1
    assert report["chat"]["2026-11-17"].leased == 2
    ok, why = rep.verdict(report["chat"], streak_days=7, max_gap_pct=1.0, joined=True, today="2026-11-22")
    assert ok, why


def test_more_distinct_leases_than_gateway_rows_still_opens_a_gap():
    """An extra lease the pool never released on a context overflow: one gateway row, two leases."""
    lines = [_line(d, lease=f"L-{d}") for d in _days(15, 7)] + [_line("2026-11-17", lease="L-extra")]
    report, _ = rep.aggregate(lines)
    for d in report["chat"].values():
        d.unmatched_leases, d.gateway_calls = 0, 1
    ok, why = rep.verdict(report["chat"], streak_days=7, max_gap_pct=1.0, joined=True, today="2026-11-22")
    assert not ok and "2 distinct leases" in why and "gateway sent 1" in why


def _joined_with_overflow(extra_overflowed: int, overflow_rows: int, extra_leases: int = 1):
    """2026-11-17: one gateway row, 1 + extra_leases distinct leases; extra_overflowed of them released overflow."""
    lines = [_line(d, lease=f"L-{d}") for d in _days(15, 7)]
    lines += [_line("2026-11-17", lease=f"L-extra-{i}") for i in range(extra_leases)]
    report, _ = rep.aggregate(lines)
    for d in report["chat"].values():
        d.unmatched_leases, d.gateway_calls, d.overflow_leases, d.overflow_rows = 0, 1, 0, 0
    day = report["chat"]["2026-11-17"]
    day.overflow_leases, day.overflow_rows = extra_overflowed, overflow_rows
    return report


def test_an_overflow_re_lease_reads_a_zero_gap():
    """#3942: the first lease overflowed and was given back; the re-lease wrote the request's only row."""
    report = _joined_with_overflow(extra_overflowed=1, overflow_rows=0)
    day = report["chat"]["2026-11-17"]
    assert rep.overflow_extras(day) == 1 and rep.gap_pct(day) == 0.0
    ok, why = rep.verdict(report["chat"], streak_days=7, max_gap_pct=1.0, joined=True, today="2026-11-22")
    assert ok, why


def test_a_final_overflow_has_its_own_row_and_is_not_an_extra():
    """Live 2026-11-20 agent-gpu2: the overflowed lease ended its request, so the gateway wrote a
    context_overflow row for it. Netting it again would hide a real extra lease."""
    report = _joined_with_overflow(extra_overflowed=1, overflow_rows=1)
    day = report["chat"]["2026-11-17"]
    assert rep.overflow_extras(day) == 0
    ok, why = rep.verdict(report["chat"], streak_days=7, max_gap_pct=1.0, joined=True, today="2026-11-22")
    assert not ok and "gateway sent 1 (" in why


def test_overflow_netting_never_covers_a_lease_that_did_not_overflow():
    """Two extra leases, only one overflowed: the other still opens the gap, and the message names the netting."""
    report = _joined_with_overflow(extra_overflowed=1, overflow_rows=0, extra_leases=2)
    ok, why = rep.verdict(report["chat"], streak_days=7, max_gap_pct=1.0, joined=True, today="2026-11-22")
    assert not ok and "3 distinct leases" in why and "gateway sent 1 + 1 overflow re-leases" in why


def test_join_db_reads_overflow_releases_and_overflow_rows(monkeypatch):
    """The SQL wiring: overflowed lease ids come from gpu_pool_events, overflow rows from llm_call_usage."""
    import types

    class Cur:
        def __init__(self):
            self.last, self.params = "", None

        def execute(self, sql, params=None):
            self.last, self.params = sql, params

        def fetchall(self):
            assert "context_overflow" in self.last and "gpu_pool_events" in self.last
            return [(lid, lid == "L-over") for lid in self.params[0] if lid != "L-fake"]

        def fetchone(self):
            assert "llm_call_usage" in self.last and "FILTER (WHERE outcome = 'context_overflow')" in self.last
            return (1, 0)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class Conn:
        def cursor(self):
            return Cur()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setitem(sys.modules, "psycopg2", types.SimpleNamespace(connect=lambda dsn: Conn()))
    report, _ = rep.aggregate([_line("2026-11-17", lease="L-main"), _line("2026-11-17", lease="L-over")])
    rep.join_db(report, "postgresql://fake")
    day = report["chat"]["2026-11-17"]
    assert (day.unmatched_leases, day.gateway_calls, day.overflow_leases, day.overflow_rows) == (0, 1, 1, 0)
    assert rep.gap_pct(day) == 0.0
    out = rep.render(report, {"chat": (False, "x")}, joined=True)
    assert "overflow" in out.splitlines()[1]
    assert out.splitlines()[2].split()[-2:] == ["1", "0.0"]   # overflow column, then gap%

    report, _ = rep.aggregate([_line("2026-11-17", lease="L-main"), _line("2026-11-17", lease="L-fake")])
    rep.join_db(report, "postgresql://fake")
    day = report["chat"]["2026-11-17"]
    assert (day.unmatched_leases, day.overflow_leases) == (1, 0)   # a made-up id is never an overflow
