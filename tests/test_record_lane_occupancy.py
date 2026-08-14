"""Tests for scripts/analysis/record_lane_occupancy.py (ROADMAP A1).

Every numeric fixture here is hand-computed and shown in the assertion's comment, so a test
that passes for the wrong reason is visible. The occupancy histogram used throughout is the
real one observed on atlas `quick` (:8013) on 2026-08-13 -- {0:101, 1:7, 2:1, 3:3, 4:9} over
121 samples -- so the statistics tests double as a regression on the numbers the roadmap cites.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

_MOD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "analysis",
    "record_lane_occupancy.py",
)
_spec = importlib.util.spec_from_file_location("record_lane_occupancy", _MOD_PATH)
assert _spec and _spec.loader
rlo = importlib.util.module_from_spec(_spec)
sys.modules["record_lane_occupancy"] = rlo
_spec.loader.exec_module(rlo)


# The real atlas `quick` observation, 2026-08-13.
QUICK_HIST = {0: 101, 1: 7, 2: 1, 3: 3, 4: 9}
QUICK_N = 121


def _rows_from_hist(hist, url="http://atlas:8013", slots=4, start_ts=1000.0):
    rows, ts = [], start_ts
    for busy, count in sorted(hist.items()):
        for _ in range(count):
            rows.append(
                {"ts": ts, "url": url, "reachable": True, "slots_total": slots, "slots_busy": busy}
            )
            ts += 1.0
    return rows


# --------------------------------------------------------------------- erlang_b


def test_erlang_b_c1_is_a_over_1_plus_a():
    # c=1 has the closed form B = a/(1+a).  a=1.0 -> 1/2 = 0.5 exactly.
    assert rlo.erlang_b(1, 1.0) == pytest.approx(0.5, abs=1e-12)


def test_erlang_b_c1_equals_mean_occupancy_the_whole_point_of_rule_2():
    # At c=1, blocking probability and mean occupancy are the SAME number. This is why
    # "8% utilised" on a 1-slot lane and on a 4-slot lane are not comparable.
    a = 0.453
    b = rlo.erlang_b(1, a)
    carried = a * (1.0 - b)  # mean busy servers
    assert b == pytest.approx(carried, abs=1e-12)
    # a/(1+a) = 0.453/1.453 = 0.3117687...
    assert b == pytest.approx(0.3117687543, abs=1e-9)


def test_erlang_b_c2_hand_computed():
    # B(2,1) = (a^2/2) / (1 + a + a^2/2) = 0.5 / 2.5 = 0.2 exactly.
    assert rlo.erlang_b(2, 1.0) == pytest.approx(0.2, abs=1e-12)


def test_erlang_b_c4_at_measured_offered_load():
    # a=0.453, c=4.  Series terms a^k/k!:
    #   1, 0.453, 0.1026045, 0.0154932795, 0.001754613903   -> sum 1.572852393
    #   B = 0.001754613903 / 1.572852393 = 0.0011155617...   (the roadmap's "0.111%")
    # NB: an earlier hand-division here gave 0.001115605 and this test failed against the
    # implementation. The implementation was right; the arithmetic was mine. Left recorded
    # because it is the reason these fixtures are hand-computed rather than snapshotted.
    assert rlo.erlang_b(4, 0.453) == pytest.approx(0.0011155617, rel=1e-8)


def test_erlang_b_edges():
    assert rlo.erlang_b(4, 0.0) == 0.0  # no offered load -> never blocks
    assert rlo.erlang_b(0, 1.0) == 1.0  # no servers -> always blocks


def test_erlang_b_is_monotone_in_offered_load():
    vals = [rlo.erlang_b(4, a) for a in (0.1, 0.5, 1.0, 2.0, 4.0, 8.0)]
    assert vals == sorted(vals)


def test_erlang_b_decreases_with_more_servers():
    vals = [rlo.erlang_b(c, 1.0) for c in (1, 2, 4, 8)]
    assert vals == sorted(vals, reverse=True)


# ------------------------------------------------------- offered_load_from_carried


def test_offered_load_inverts_c1_closed_form():
    # c=1: carried = a/(1+a).  carried=0.5 <=> a=1.0 exactly.
    assert rlo.offered_load_from_carried(1, 0.5) == pytest.approx(1.0, rel=1e-6)


def test_offered_load_inverts_c2_hand_computed():
    # c=2, a=1.0: B=0.2 so carried = 1.0*0.8 = 0.8.  Inverting 0.8 must return 1.0.
    assert rlo.offered_load_from_carried(2, 0.8) == pytest.approx(1.0, rel=1e-6)


def test_offered_load_roundtrips_through_erlang_b():
    for c in (1, 2, 4, 8):
        for a in (0.05, 0.453, 1.0, 3.0):
            carried = a * (1.0 - rlo.erlang_b(c, a))
            assert rlo.offered_load_from_carried(c, carried) == pytest.approx(a, rel=1e-5)


def test_offered_load_none_when_carried_at_or_above_capacity():
    # carried >= c means the lane never releases a slot: not an attainable steady state.
    assert rlo.offered_load_from_carried(4, 4.0) is None
    assert rlo.offered_load_from_carried(4, 4.5) is None


def test_offered_load_zero_carried_is_zero_not_none():
    assert rlo.offered_load_from_carried(4, 0.0) == 0.0


# --------------------------------------------------------------- parse_slots_payload


def test_parse_slots_counts_only_processing():
    payload = [
        {"id": 0, "is_processing": True},
        {"id": 1, "is_processing": False},
        {"id": 2, "is_processing": True},
        {"id": 3},
    ]
    assert rlo.parse_slots_payload(payload) == (4, 2)


def test_parse_slots_empty_list_is_zero_zero():
    assert rlo.parse_slots_payload([]) == (0, 0)


def test_parse_slots_rejects_error_envelope_rather_than_reporting_idle():
    # A dict body means the endpoint errored. Returning (0,0) here would manufacture an
    # idle reading for a lane we actually know nothing about.
    with pytest.raises(ValueError):
        rlo.parse_slots_payload({"error": "slots disabled"})


def test_parse_slots_rejects_non_list():
    with pytest.raises(ValueError):
        rlo.parse_slots_payload("not json array")


# ------------------------------------------------------------------- accumulate


def test_accumulate_statistics_match_hand_computed_quick_lane():
    st = rlo.accumulate(_rows_from_hist(QUICK_HIST))["http://atlas:8013"]
    assert st.n_reachable == QUICK_N
    assert st.servers == 4
    # mean busy = (0*101 + 1*7 + 2*1 + 3*3 + 4*9) / 121 = 54/121 = 0.4462809...
    assert st.mean_busy == pytest.approx(54 / 121, abs=1e-12)
    # P(any busy) = (7+1+3+9)/121 = 20/121 = 0.1652892...
    assert st.p_any_busy == pytest.approx(20 / 121, abs=1e-12)
    # P(all busy) = 9/121 = 0.0743801...  -> the roadmap's 7.4%
    assert st.p_all_busy == pytest.approx(9 / 121, abs=1e-12)


def test_unreachable_samples_are_excluded_not_counted_as_idle():
    """The circe case. Counting an switched-off host as 0-busy manufactures idleness."""
    rows = _rows_from_hist({4: 10})  # 10 reachable samples, all completely full
    rows += [{"ts": 2000.0 + i, "url": "http://atlas:8013", "reachable": False, "error": "down"}
             for i in range(90)]
    st = rlo.accumulate(rows)["http://atlas:8013"]
    assert st.n_total == 100
    assert st.n_reachable == 10
    assert st.n_unreachable == 90
    # If the 90 down samples were counted as busy=0, P(all busy) would be 10/100 = 10%.
    assert st.p_all_busy == pytest.approx(1.0)
    assert st.mean_busy == pytest.approx(4.0)


def test_malformed_reachable_row_counts_as_unreachable_not_as_zero():
    rows = [{"ts": 1.0, "url": "u", "reachable": True, "slots_total": None, "slots_busy": None}]
    st = rlo.accumulate(rows)["u"]
    assert st.n_reachable == 0 and st.n_unreachable == 1
    assert st.p_all_busy is None


def test_servers_uses_modal_slot_count_across_a_restart():
    rows = _rows_from_hist({0: 5}, slots=4) + _rows_from_hist({0: 2}, slots=8, start_ts=5000.0)
    st = rlo.accumulate(rows)["http://atlas:8013"]
    assert st.servers == 4  # 5 samples at 4 slots beats 2 samples at 8


def test_window_is_span_not_count():
    rows = _rows_from_hist({0: 3})  # ts 1000, 1001, 1002
    st = rlo.accumulate(rows)["http://atlas:8013"]
    assert st.window_sec == pytest.approx(2.0)


def test_empty_lane_yields_none_rather_than_zero():
    st = rlo.LaneStats(url="u")
    assert st.mean_busy is None and st.p_all_busy is None and st.p_any_busy is None


# ------------------------------------------------------------ route table -> lanes


ROUTE_TABLE = {
    "chat": {"url": "http://circe:8011", "served_by": "circe-worker-1"},
    "agent": {"url": "http://circe:8011", "served_by": "circe-worker-1"},
    "metacog": {"url": "http://atlas:8012", "served_by": "atlas-worker-2"},
    "quick": {"url": "http://atlas:8013", "served_by": "atlas-worker-fast-1"},
    "quick_background": {"url": "http://atlas:8013", "served_by": "atlas-worker-fast-1",
                         "priority": "background", "reserved_free_slots": 2},
}


def test_routes_sharing_an_upstream_collapse_to_one_lane():
    lanes = rlo.lanes_from_route_table(ROUTE_TABLE)
    assert len(lanes) == 3  # 5 routes, 3 distinct URLs
    by_url = {lane.url: lane for lane in lanes}
    assert by_url["http://circe:8011"].routes == ("agent", "chat")
    assert by_url["http://atlas:8013"].routes == ("quick", "quick_background")


def test_lane_label_names_every_route_sharing_the_slots():
    lanes = {lane.url: lane for lane in rlo.lanes_from_route_table(ROUTE_TABLE)}
    assert lanes["http://atlas:8013"].label == "quick+quick_background @ atlas-worker-fast-1"


def test_route_without_url_is_skipped():
    assert rlo.lanes_from_route_table({"broken": {"served_by": "x"}, "ok": {"url": "u"}}) == [
        rlo.Lane(url="u", routes=("ok",), served_by="")
    ]


def test_read_route_table_prefers_environment(monkeypatch):
    monkeypatch.setenv(rlo.ROUTE_TABLE_KEY, json.dumps({"quick": {"url": "http://env:1"}}))
    assert rlo.read_route_table(("/nonexistent",))["quick"]["url"] == "http://env:1"


def test_read_route_table_strips_shell_quotes(monkeypatch, tmp_path):
    p = tmp_path / ".env"
    p.write_text(f"OTHER=1\n{rlo.ROUTE_TABLE_KEY}='{json.dumps({'q': {'url': 'u'}})}'\n")
    monkeypatch.delenv(rlo.ROUTE_TABLE_KEY, raising=False)
    assert rlo.read_route_table((str(p),)) == {"q": {"url": "u"}}


# ----------------------------------------------------------------- report gating


def test_report_refuses_short_windows_by_default():
    stats = rlo.accumulate(_rows_from_hist(QUICK_HIST))  # 121 samples
    text, ok = rlo.format_report(stats, {}, min_samples=600, allow_short=False)
    assert ok is False
    assert "REFUSING to report" in text
    # and it must not have leaked the numbers it refused to stand behind
    assert "P(all busy)" not in text


def test_allow_short_reports_but_shouts():
    stats = rlo.accumulate(_rows_from_hist(QUICK_HIST))
    text, ok = rlo.format_report(stats, {}, min_samples=600, allow_short=True)
    assert ok is True
    assert "SHORT WINDOW" in text
    assert "P(all busy)  7.44%" in text  # 9/121


def test_report_leads_with_ceiling_statistic_not_the_mean():
    stats = rlo.accumulate(_rows_from_hist(QUICK_HIST))
    text, _ = rlo.format_report(stats, {}, min_samples=1, allow_short=False)
    assert text.index("P(all busy)") < text.index("mean busy")
    assert "NOT the ceiling" in text


def test_report_omits_burstiness_when_nothing_ever_blocked():
    """A lane that never filled has no burstiness finding -- printing "0x" reads as one."""
    stats = rlo.accumulate(_rows_from_hist({0: 40, 1: 5}))
    text, _ = rlo.format_report(stats, {}, min_samples=1, allow_short=False)
    assert "P(all busy)  0.00%" in text
    assert "burstiness" not in text


def test_report_flags_burstiness_against_poisson():
    stats = rlo.accumulate(_rows_from_hist(QUICK_HIST))
    text, _ = rlo.format_report(stats, {}, min_samples=1, allow_short=False)
    # carried 0.446 at c=4 -> a ~= 0.4465; Erlang-B ~= 0.107%; observed 7.44% -> ~70x
    assert "burstiness" in text
    assert "more blocking than Poisson" in text


def test_report_names_a_fully_unreachable_lane_without_statistics():
    rows = [{"ts": float(i), "url": "http://circe:8011", "reachable": False, "error": "down"}
            for i in range(50)]
    text, ok = rlo.format_report(rlo.accumulate(rows), {}, min_samples=600, allow_short=False)
    assert ok is True  # an off host is not a short-window failure
    assert "UNREACHABLE for the whole window" in text
    assert "P(all busy)" not in text


# ------------------------------------------------------------------------- io


def test_read_samples_skips_blank_and_malformed_lines(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text('{"ts":1,"url":"u","reachable":true,"slots_total":4,"slots_busy":0}\n'
                 "\n"
                 "not json\n"
                 '{"ts":2,"url":"u","reachable":false,"error":"x"}\n')
    rows = rlo.read_samples(str(p))
    assert len(rows) == 2


def test_sample_to_json_omits_slot_fields_when_unreachable():
    s = rlo.Sample(ts=1.5, url="u", reachable=False, error="boom")
    d = json.loads(s.to_json())
    assert d == {"ts": 1.5, "url": "u", "reachable": False, "error": "boom"}
    assert "slots_busy" not in d


def test_poll_lane_records_unreachable_rather_than_raising(monkeypatch):
    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(rlo.urllib.request, "urlopen", boom)
    s = rlo.poll_lane(rlo.Lane(url="http://down:1", routes=("chat",), served_by="x"), now=42.0)
    assert s.reachable is False and s.ts == 42.0 and "connection refused" in (s.error or "")
