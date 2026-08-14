"""Tests for scripts/analysis/record_lane_occupancy.py (ROADMAP A1).

Every numeric fixture is hand-computed with the arithmetic shown in the assertion, so a test
that passes for the wrong reason is visible. The occupancy histogram used throughout is the
real one observed on atlas `quick` (:8013) on 2026-08-13 -- {0:101, 1:7, 2:1, 3:3, 4:9} over
121 samples -- so the statistics tests double as a regression on the numbers the roadmap cites.

Tests named `test_review_*` are regressions for findings from the code review of the first
commit; each names the wrong answer the old code produced.
"""
from __future__ import annotations

import http.client
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

URL = "http://atlas:8013"
QUICK_HIST = {0: 101, 1: 7, 2: 1, 3: 3, 4: 9}  # real atlas `quick`, 2026-08-13
QUICK_N = 121


def _rows(hist, url=URL, slots=4, start_ts=1000.0, step=1.0):
    rows, ts = [], start_ts
    for busy, count in sorted(hist.items()):
        for _ in range(count):
            rows.append(
                {"ts": ts, "url": url, "reachable": True, "slots_total": slots, "slots_busy": busy}
            )
            ts += step
    return rows


def _lane(url=URL, reserved=0, routes=("quick",)):
    return {url: rlo.Lane(url=url, routes=routes, served_by="w", reserved_free_slots=reserved)}


# --------------------------------------------------------------------- erlang_b


def test_erlang_b_c1_is_a_over_1_plus_a():
    # c=1 closed form B = a/(1+a); a=1.0 -> 0.5 exactly.
    assert rlo.erlang_b(1, 1.0) == pytest.approx(0.5, abs=1e-12)


def test_erlang_b_c1_equals_mean_occupancy_the_whole_point_of_rule_2():
    # At c=1 blocking probability and mean occupancy are the SAME number, which is why
    # "8% utilised" on a 1-slot lane and on a 4-slot lane are not comparable.
    a = 0.453
    b = rlo.erlang_b(1, a)
    assert b == pytest.approx(a * (1.0 - b), abs=1e-12)
    assert b == pytest.approx(0.3117687543, abs=1e-9)  # 0.453/1.453


def test_erlang_b_c2_hand_computed():
    # B(2,1) = (1/2) / (1 + 1 + 1/2) = 0.5/2.5 = 0.2 exactly.
    assert rlo.erlang_b(2, 1.0) == pytest.approx(0.2, abs=1e-12)


def test_erlang_b_c4_at_measured_offered_load():
    # a=0.453, c=4. Terms a^k/k!: 1, 0.453, 0.1026045, 0.0154932795, 0.001754613903
    #   sum 1.572852393; B = 0.001754613903/1.572852393 = 0.0011155617...
    # NB: an earlier hand-division here gave 0.001115605 and this test failed against the
    # implementation. The implementation was right; the arithmetic was mine. Left recorded
    # because it is the reason these fixtures are hand-computed rather than snapshotted.
    assert rlo.erlang_b(4, 0.453) == pytest.approx(0.0011155617, rel=1e-8)


def test_erlang_b_edges():
    assert rlo.erlang_b(4, 0.0) == 0.0
    assert rlo.erlang_b(0, 1.0) == 1.0


def test_erlang_b_monotone_in_load_and_decreasing_in_servers():
    assert [rlo.erlang_b(4, a) for a in (0.1, 0.5, 1.0, 2.0, 4.0)] == sorted(
        [rlo.erlang_b(4, a) for a in (0.1, 0.5, 1.0, 2.0, 4.0)]
    )
    vals = [rlo.erlang_b(c, 1.0) for c in (1, 2, 4, 8)]
    assert vals == sorted(vals, reverse=True)


# ------------------------------------------------------- offered_load_from_carried


def test_offered_load_inverts_c1_closed_form():
    # c=1: carried = a/(1+a); carried=0.5 <=> a=1.0 exactly.
    assert rlo.offered_load_from_carried(1, 0.5) == pytest.approx(1.0, rel=1e-6)


def test_offered_load_inverts_c2_hand_computed():
    # c=2, a=1.0: B=0.2 so carried = 0.8. Inverting 0.8 must return 1.0.
    assert rlo.offered_load_from_carried(2, 0.8) == pytest.approx(1.0, rel=1e-6)


def test_offered_load_roundtrips():
    for c in (1, 2, 4, 8):
        for a in (0.05, 0.453, 1.0, 3.0):
            carried = a * (1.0 - rlo.erlang_b(c, a))
            assert rlo.offered_load_from_carried(c, carried) == pytest.approx(a, rel=1e-5)


def test_offered_load_none_at_or_above_capacity():
    assert rlo.offered_load_from_carried(4, 4.0) is None
    assert rlo.offered_load_from_carried(4, 4.5) is None


def test_offered_load_zero_carried_is_zero_not_none():
    assert rlo.offered_load_from_carried(4, 0.0) == 0.0


def test_review_offered_load_rejects_nan_rather_than_returning_a_number():
    """Finding 10: NaN collapsed the bracket and returned ~1.24e-60 as an offered load."""
    assert rlo.offered_load_from_carried(4, float("nan")) is None


def test_review_offered_load_rejects_zero_servers():
    """Finding 10: c<=0 returned 0.0, asserting a load for a lane with no servers."""
    assert rlo.offered_load_from_carried(0, 0.0) is None
    assert rlo.offered_load_from_carried(-1, 0.0) is None


# --------------------------------------------------------------- parse_slots_payload


def test_parse_slots_counts_only_processing():
    payload = [
        {"id": 0, "is_processing": True},
        {"id": 1, "is_processing": False},
        {"id": 2, "is_processing": True},
        {"id": 3},
    ]
    assert rlo.parse_slots_payload(payload) == (4, 2)


def test_review_empty_slot_list_is_indeterminate_not_idle():
    """Finding 4: `[]` became (0,0) -> a reachable idle sample.

    llama.cpp serves `[]` while a model loads. The live gateway reads it as ZERO FREE slots
    and blocks background dispatch, so scoring it idle disagrees with production in the
    unsafe direction.
    """
    with pytest.raises(ValueError, match="empty slot list"):
        rlo.parse_slots_payload([])


def test_parse_slots_rejects_error_envelope():
    with pytest.raises(ValueError):
        rlo.parse_slots_payload({"error": "slots disabled"})


def test_parse_slots_rejects_non_list():
    with pytest.raises(ValueError):
        rlo.parse_slots_payload("not json array")


# ------------------------------------------------------------------- accumulate


def test_accumulate_matches_hand_computed_quick_lane():
    # Uniform 1 s spacing -> every weight is 1.0, so time-weighted == count-based here.
    st = rlo.accumulate(_rows(QUICK_HIST))[URL]
    assert st.n_reachable == QUICK_N
    assert st.servers == 4
    assert st.total_weight == pytest.approx(121.0)
    # mean busy = (0*101 + 1*7 + 2*1 + 3*3 + 4*9)/121 = 54/121
    assert st.mean_busy == pytest.approx(54 / 121, abs=1e-12)
    # P(any busy) = (7+1+3+9)/121 = 20/121
    assert st.p_any_busy == pytest.approx(20 / 121, abs=1e-12)
    # P(all busy) = 9/121 = 0.0743801...  -> the roadmap's 7.4%
    assert st.p_all_busy == pytest.approx(9 / 121, abs=1e-12)


def test_review_slot_count_change_does_not_manufacture_a_ceiling():
    """Finding 1 (CRITICAL): the review's exact input reported P(all busy) 45.45%.

    600 samples at 4 slots busy in {0,1}, then 500 at 8 slots busy in {4,5,6}. The lane is
    never full in either regime, so the true answer is 0.00%. The old code scored the wide
    regime against the modal count of 4.
    """
    rows = _rows({0: 500, 1: 100}, slots=4, start_ts=0.0)
    rows += _rows({4: 200, 5: 200, 6: 100}, slots=8, start_ts=600.0)
    st = rlo.accumulate(rows)[URL]
    assert st.servers == 4  # 600 > 500
    assert st.slot_count_changed is True
    assert st.n_off_modal == 500
    assert st.p_all_busy == pytest.approx(0.0)  # NOT 0.4545
    # mean must stay within the modal regime's capacity: 100/600 busy-slots
    assert st.mean_busy == pytest.approx(100 / 600, abs=1e-9)
    assert st.mean_busy <= st.servers


def test_review_slot_count_change_is_announced_in_the_report():
    rows = _rows({0: 10}, slots=4, start_ts=0.0) + _rows({0: 4}, slots=8, start_ts=100.0)
    text, _ = rlo.format_report(
        rlo.accumulate(rows), {}, min_samples=1, min_window_sec=0.0, allow_short=False
    )
    assert "SLOT COUNT CHANGED mid-window" in text
    assert "4 slots x10" in text and "8 slots x4" in text
    assert "off-modal EXCLUDED" in text


def test_unreachable_samples_are_excluded_not_counted_as_idle():
    """The circe case: an off host must not read as an idle lane."""
    rows = _rows({4: 10})
    rows += [{"ts": 2000.0 + i, "url": URL, "reachable": False, "error": "down"} for i in range(90)]
    st = rlo.accumulate(rows)[URL]
    assert st.n_total == 100 and st.n_reachable == 10 and st.n_indeterminate == 90
    assert st.p_all_busy == pytest.approx(1.0)  # not 10/100
    assert st.mean_busy == pytest.approx(4.0)


def test_zero_slot_total_is_indeterminate():
    rows = [{"ts": 1.0, "url": URL, "reachable": True, "slots_total": 0, "slots_busy": 0}]
    st = rlo.accumulate(rows)[URL]
    assert st.n_reachable == 0 and st.n_indeterminate == 1
    assert st.p_all_busy is None


def test_malformed_reachable_row_is_indeterminate():
    rows = [{"ts": 1.0, "url": URL, "reachable": True, "slots_total": None, "slots_busy": None}]
    st = rlo.accumulate(rows)[URL]
    assert st.n_reachable == 0 and st.n_indeterminate == 1


def test_empty_lane_yields_none_rather_than_zero():
    st = rlo.LaneStats(url="u")
    assert st.mean_busy is None and st.p_all_busy is None and st.p_any_busy is None


# ------------------------------------------------------------ coverage vs span


def test_review_coverage_excludes_the_hole_between_two_runs():
    """Finding 2 (HIGH): --out appends, so span claimed 12 h for 20 min of samples."""
    rows = _rows({0: 300}, start_ts=0.0)                 # 300 s of data
    rows += _rows({0: 290, 4: 10}, start_ts=43200.0)     # 300 s more, 12 h later
    st = rlo.accumulate(rows)[URL]
    assert st.span_sec == pytest.approx(43200.0 + 299.0)     # ~12.08 h span
    assert st.coverage_sec == pytest.approx(600.0, abs=2.0)  # ~600 s actually observed
    assert st.n_discontinuities == 1


def test_review_report_shows_span_and_coverage_separately():
    rows = _rows({0: 300}, start_ts=0.0) + _rows({0: 300}, start_ts=43200.0)
    text, _ = rlo.format_report(
        rlo.accumulate(rows), {}, min_samples=1, min_window_sec=0.0, allow_short=False
    )
    assert "span         12.08 h" in text
    assert "coverage 0.17 h" in text
    assert "1 gap(s) between runs discarded" in text


def test_contiguous_run_has_no_discontinuities():
    st = rlo.accumulate(_rows({0: 100}))[URL]
    assert st.n_discontinuities == 0
    assert st.coverage_sec == pytest.approx(100.0)


# ------------------------------------------------------------ time weighting


def test_review_statistics_are_time_weighted_not_sample_counted():
    """Finding 7 (MEDIUM): samples thin out when a worker is slow, i.e. when it is busy.

    Ten idle samples 1 s apart (ts 0..9), then two full samples 10 s apart (ts 19, 29).
    Weights are the gap to the next sample; the final sample gets the median gap (1.0):

        busy=0:  9 x 1.0  +  10.0 (ts 9 -> 19)          = 19.0 s
        busy=4:      10.0 (ts 19 -> 29)  +  1.0 (final) = 11.0 s
        total                                            = 30.0 s

    Time-weighted P(all busy) = 11/30 = 36.67%. Counting samples gives 2/12 = 16.67%, which
    understates the ceiling by more than 2x -- in the direction that matters, because the
    thinning happens precisely while the lane is saturated.
    """
    rows = [{"ts": float(i), "url": URL, "reachable": True, "slots_total": 4, "slots_busy": 0}
            for i in range(10)]
    rows += [{"ts": 9.0 + 10.0 * k, "url": URL, "reachable": True, "slots_total": 4, "slots_busy": 4}
             for k in (1, 2)]
    st = rlo.accumulate(rows)[URL]
    assert st.total_weight == pytest.approx(30.0)
    assert st.p_all_busy == pytest.approx(11 / 30, abs=1e-12)
    assert st.p_all_busy > 2 * (2 / 12)  # strictly worse than the sample-counted answer


def test_review_a_slow_tick_is_not_a_discontinuity():
    """The regression that DISCONTINUITY_FACTOR=5 introduced while fixing coverage.

    A 10 s gap against a 1 s median is an ordinary slow tick (4 lanes x 3 s timeout), not a
    hole between runs. Classifying it as one reverted every statistic to sample-counting.
    """
    rows = [{"ts": t, "url": URL, "reachable": True, "slots_total": 4, "slots_busy": 0}
            for t in (0.0, 1.0, 2.0, 12.0)]
    st = rlo.accumulate(rows)[URL]
    assert st.n_discontinuities == 0
    assert st.coverage_sec == pytest.approx(13.0)  # 1 + 1 + 10 + 1 (final = median 1.0)


# ------------------------------------------------- reserved_free_slots / admission


def test_review_background_admission_ceiling_is_reported():
    """Finding 5: the dispatcher blocks background at free < reserved, not at all-busy.

    On {0:101,1:7,2:1,3:3,4:9} with c=4 and reserved=2, background is refused whenever
    busy >= 3, i.e. (3+9)/121 = 12/121 = 9.92% -- not the 9/121 = 7.44% all-busy figure.
    """
    st = rlo.accumulate(_rows(QUICK_HIST))[URL]
    assert st.p_all_busy == pytest.approx(9 / 121, abs=1e-12)
    assert st.p_admission_blocked(2) == pytest.approx(12 / 121, abs=1e-12)


def test_admission_ceiling_reduces_to_all_busy_when_nothing_reserved():
    st = rlo.accumulate(_rows(QUICK_HIST))[URL]
    assert st.p_admission_blocked(0) == st.p_all_busy


def test_admission_ceiling_appears_in_report_with_the_threshold():
    text, _ = rlo.format_report(
        rlo.accumulate(_rows(QUICK_HIST)),
        _lane(reserved=2, routes=("quick", "quick_background")),
        min_samples=1,
        min_window_sec=0.0,
        allow_short=False,
    )
    assert "P(bg blocked)   9.92%" in text
    assert "blocks at busy>=3" in text


def test_reserved_free_slots_survives_route_table_parsing():
    lanes = {l.url: l for l in rlo.lanes_from_route_table(ROUTE_TABLE)}
    assert lanes["http://atlas:8013"].reserved_free_slots == 2  # from quick_background
    assert lanes["http://atlas:8012"].reserved_free_slots == 0


# ------------------------------------------------------------ route table -> lanes


ROUTE_TABLE = {
    "chat": {"url": "http://circe:8011", "served_by": "circe-worker-1"},
    "metacog": {"url": "http://atlas:8012", "served_by": "atlas-worker-2"},
    "quick": {"url": "http://atlas:8013", "served_by": "atlas-worker-fast-1"},
    "quick_background": {"url": "http://atlas:8013", "served_by": "atlas-worker-fast-1",
                         "priority": "background", "reserved_free_slots": 2},
    "agent": {"url": "http://atlas:8014", "served_by": "atlas-worker-agent-1"},
}


def test_routes_sharing_an_upstream_collapse_to_one_lane():
    lanes = rlo.lanes_from_route_table(ROUTE_TABLE)
    assert len(lanes) == 4  # 5 routes, 4 distinct URLs
    by_url = {lane.url: lane for lane in lanes}
    assert by_url["http://atlas:8013"].routes == ("quick", "quick_background")


def test_lane_label_names_every_route_sharing_the_slots():
    lanes = {lane.url: lane for lane in rlo.lanes_from_route_table(ROUTE_TABLE)}
    assert lanes["http://atlas:8013"].label == "quick+quick_background @ atlas-worker-fast-1"


def test_route_without_url_is_skipped():
    assert rlo.lanes_from_route_table({"broken": {"served_by": "x"}, "ok": {"url": "u"}}) == [
        rlo.Lane(url="u", routes=("ok",), served_by="", reserved_free_slots=0)
    ]


def test_read_route_table_prefers_environment(monkeypatch):
    monkeypatch.setenv(rlo.ROUTE_TABLE_KEY, json.dumps({"quick": {"url": "http://env:1"}}))
    assert rlo.read_route_table(("/nonexistent",))["quick"]["url"] == "http://env:1"


def test_read_route_table_strips_shell_quotes(monkeypatch, tmp_path):
    p = tmp_path / ".env"
    p.write_text(f"OTHER=1\n{rlo.ROUTE_TABLE_KEY}='{json.dumps({'q': {'url': 'u'}})}'\n")
    monkeypatch.delenv(rlo.ROUTE_TABLE_KEY, raising=False)
    assert rlo.read_route_table((str(p),)) == {"q": {"url": "u"}}


def test_review_bad_json_exits_cleanly_rather_than_raising_a_traceback(monkeypatch):
    """Finding 10: a raw JSONDecodeError traceback instead of a diagnosable message."""
    monkeypatch.setenv(rlo.ROUTE_TABLE_KEY, "{not json")
    with pytest.raises(SystemExit, match="not valid JSON"):
        rlo.read_route_table(("/nonexistent",))


# ----------------------------------------------------------------- report gating


def test_report_refuses_short_sample_count_by_default():
    text, ok = rlo.format_report(
        rlo.accumulate(_rows(QUICK_HIST)), {}, min_samples=600, min_window_sec=0.0, allow_short=False
    )
    assert ok is False
    assert "REFUSING to report" in text
    assert "P(all busy)" not in text  # must not leak numbers it refused to stand behind


def test_review_min_samples_alone_does_not_admit_a_sixty_second_window():
    """Finding 3 (HIGH): 600 samples at --interval 0.1 is 60 s, and passed the old gate."""
    rows = _rows({0: 600}, step=0.1)  # 600 samples spanning ~60 s
    stats = rlo.accumulate(rows)
    text, ok = rlo.format_report(
        stats, {}, min_samples=600, min_window_sec=3600.0, allow_short=False
    )
    assert ok is False
    assert "coverage < --min-window-sec" in text
    assert "P(all busy)" not in text


def test_gate_passes_when_both_sample_count_and_window_are_met():
    rows = _rows({0: 3600}, step=1.0)  # 3600 samples over 1 h
    text, ok = rlo.format_report(
        rlo.accumulate(rows), {}, min_samples=600, min_window_sec=3600.0, allow_short=False
    )
    assert ok is True
    assert "REFUSING" not in text and "SHORT" not in text


def test_allow_short_reports_but_shouts():
    text, ok = rlo.format_report(
        rlo.accumulate(_rows(QUICK_HIST)), {}, min_samples=600, min_window_sec=3600.0, allow_short=True
    )
    assert ok is True
    assert "SHORT" in text
    assert "P(all busy)  7.44%" in text  # 9/121


def test_report_leads_with_ceiling_statistic_not_the_mean():
    text, _ = rlo.format_report(
        rlo.accumulate(_rows(QUICK_HIST)), {}, min_samples=1, min_window_sec=0.0, allow_short=False
    )
    assert text.index("P(all busy)") < text.index("mean busy")
    assert "NOT the ceiling" in text
    assert "time-weighted" in text


def test_report_names_a_fully_unreachable_lane_without_statistics():
    rows = [{"ts": float(i), "url": "http://circe:8011", "reachable": False, "error": "down"}
            for i in range(50)]
    text, ok = rlo.format_report(
        rlo.accumulate(rows), {}, min_samples=600, min_window_sec=3600.0, allow_short=False
    )
    assert ok is True  # an off host is not a short-window failure
    assert "NO MEASUREMENTS in this window" in text
    assert "P(all busy)" not in text


# --------------------------------------------------------------------- burstiness


def test_review_burstiness_asserts_the_computed_ratio_not_a_substring():
    """Finding 9: the old test passed whether the ratio printed 1x or 10000x.

    For {0:101,1:7,2:1,3:3,4:9} at c=4: carried 54/121 = 0.446281, inverted to
    a = 0.447..., Erlang-B(4, a) ~= 0.00106, observed 9/121 = 0.074380 -> ~70x.
    """
    st = rlo.accumulate(_rows(QUICK_HIST))[URL]
    a = rlo.offered_load_from_carried(4, st.mean_busy)
    pred = rlo.erlang_b(4, a)
    ratio = st.p_all_busy / pred
    assert a == pytest.approx(0.44676, rel=1e-3)
    assert pred == pytest.approx(0.001063, rel=1e-2)
    assert ratio == pytest.approx(70.0, rel=0.05)
    text, _ = rlo.format_report(
        rlo.accumulate(_rows(QUICK_HIST)), {}, min_samples=1, min_window_sec=0.0, allow_short=False
    )
    assert f"{ratio:.0f}x MORE blocking than Poisson" in text


def test_review_smoother_than_poisson_is_reported_as_such_not_as_zero_x():
    """Finding 8: ratio < 1 rendered as "0x more blocking than Poisson"."""
    text, _ = rlo.format_report(
        rlo.accumulate(_rows({3: 90, 4: 10})), {}, min_samples=1, min_window_sec=0.0, allow_short=False
    )
    assert "LESS than Poisson" in text
    assert "0x" not in text


def test_burstiness_is_suppressed_on_a_single_slot_lane():
    """At c=1 the ratio is ~1.00 by construction -- a tautology, not a measurement.

    Seen live on the `chat` lane (circe, 1 slot): P(all busy) 0.27% against an Erlang-B of
    0.268%, i.e. 1.00x. That is arithmetic, not evidence about the arrival process.
    """
    text, _ = rlo.format_report(
        rlo.accumulate(_rows({0: 90, 1: 10}, slots=1)),
        {}, min_samples=1, min_window_sec=0.0, allow_short=False,
    )
    assert "n/a at 1 slot" in text
    assert "MORE blocking than Poisson" not in text


def test_report_omits_burstiness_when_nothing_ever_blocked():
    text, _ = rlo.format_report(
        rlo.accumulate(_rows({0: 40, 1: 5})), {}, min_samples=1, min_window_sec=0.0, allow_short=False
    )
    assert "P(all busy)  0.00%" in text
    assert "burstiness" not in text


# ------------------------------------------------------------------------- io


def test_read_samples_skips_blank_and_malformed_lines(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text('{"ts":1,"url":"u","reachable":true,"slots_total":4,"slots_busy":0}\n'
                 "\n"
                 "not json\n"
                 '{"ts":2,"url":"u","reachable":false,"error":"x"}\n')
    assert len(rlo.read_samples(str(p))) == 2


def test_sample_to_json_omits_slot_fields_when_unreachable():
    d = json.loads(rlo.Sample(ts=1.5, url="u", reachable=False, error="boom").to_json())
    assert d == {"ts": 1.5, "url": "u", "reachable": False, "error": "boom"}


def test_poll_lane_records_unreachable_rather_than_raising(monkeypatch):
    monkeypatch.setattr(rlo.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("refused")))
    s = rlo.poll_lane(rlo.Lane(url="http://down:1", routes=("chat",), served_by="x"), now=42.0)
    assert s.reachable is False and s.ts == 42.0 and "refused" in (s.error or "")


@pytest.mark.parametrize(
    "exc",
    [
        http.client.IncompleteRead(b"x"),
        http.client.BadStatusLine("garbage"),
        http.client.LineTooLong("header line"),
        RuntimeError("something nobody predicted"),
    ],
)
def test_review_http_client_and_unforeseen_exceptions_do_not_kill_a_24h_run(monkeypatch, exc):
    """Finding 6 (MEDIUM): http.client.HTTPException is not an OSError.

    IncompleteRead is what a host being powered off mid-body produces -- the module's own
    stated normal case -- and it aborted record() after 3 polls.
    """
    monkeypatch.setattr(rlo.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(exc))
    s = rlo.poll_lane(rlo.Lane(url="http://x:1", routes=("chat",), served_by="w"), now=1.0)
    assert s.reachable is False
    assert type(exc).__name__ in (s.error or "")


def test_review_record_rejects_nonpositive_interval_and_duration(tmp_path):
    """Finding 10: --interval 0 hammered /slots thousands of times a second."""
    lane = rlo.Lane(url="http://x:1", routes=("q",), served_by="w")
    with pytest.raises(SystemExit, match="interval"):
        rlo.record([lane], str(tmp_path / "o.jsonl"), interval=0.0, duration=10.0)
    with pytest.raises(SystemExit, match="duration"):
        rlo.record([lane], str(tmp_path / "o.jsonl"), interval=1.0, duration=-5.0)


# --------------------------------------------------- route-table refresh (mid-run)


def _fake_poll(lane, **kw):
    return rlo.Sample(ts=1000.0, url=lane.url, reachable=True, slots_total=4, slots_busy=0)


def test_review_record_rereads_the_route_table_mid_run(monkeypatch, tmp_path):
    """The defect this fixes: the lane set was snapshotted once at startup.

    `agent` was repointed from atlas to circe mid-run and the recorder kept polling the old
    address, reporting "0 measured over 3.67 h" for a lane that was live and serving turns.
    Over a 24 h gate window that is a fabricated finding, not a stale label.
    """
    tables = [
        {"agent": {"url": "http://old:8014"}},
        {"agent": {"url": "http://new:8014"}},
    ]
    calls = {"n": 0}

    def fake_read(env_files):
        calls["n"] += 1
        return tables[min(calls["n"] - 1, len(tables) - 1)]

    monkeypatch.setattr(rlo, "read_route_table", fake_read)
    monkeypatch.setattr(rlo, "poll_lane", _fake_poll)
    out = tmp_path / "s.jsonl"
    rlo.record(
        rlo.lanes_from_route_table(tables[0]), str(out),
        interval=0.01, duration=0.15, route_refresh_sec=0.02,
    )
    urls = {json.loads(l)["url"] for l in out.read_text().splitlines() if l.strip()}
    assert "http://old:8014" in urls  # polled before the change
    assert "http://new:8014" in urls  # and after -- the whole point


def test_route_refresh_zero_pins_the_lane_set(monkeypatch, tmp_path):
    def fake_read(env_files):
        raise AssertionError("must not re-read when pinned")

    monkeypatch.setattr(rlo, "read_route_table", fake_read)
    monkeypatch.setattr(rlo, "poll_lane", _fake_poll)
    out = tmp_path / "s.jsonl"
    rlo.record(
        [rlo.Lane(url="http://a:1", routes=("q",), served_by="w")], str(out),
        interval=0.01, duration=0.08, route_refresh_sec=0.0,
    )
    assert out.read_text().strip()


def test_a_failed_refresh_keeps_recording_rather_than_ending_the_run(monkeypatch, tmp_path):
    """A 24h run must not die because the env file was briefly unreadable."""
    def fake_read(env_files):
        raise SystemExit("env vanished")

    monkeypatch.setattr(rlo, "read_route_table", fake_read)
    monkeypatch.setattr(rlo, "poll_lane", _fake_poll)
    out = tmp_path / "s.jsonl"
    n = rlo.record(
        [rlo.Lane(url="http://a:1", routes=("q",), served_by="w")], str(out),
        interval=0.01, duration=0.1, route_refresh_sec=0.02,
    )
    assert n > 0


def test_negative_route_refresh_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="route-refresh-sec"):
        rlo.record(
            [rlo.Lane(url="http://a:1", routes=("q",), served_by="w")],
            str(tmp_path / "o.jsonl"), interval=1.0, duration=1.0, route_refresh_sec=-1.0,
        )
