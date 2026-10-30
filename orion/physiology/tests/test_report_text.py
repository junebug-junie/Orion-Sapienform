"""skills.physiology.state.v1 text: plain first-person words from a PhysiologyReportV1, every
empty section explained, sections chosen by the caller, the fetch failing in plain words."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from orion.physiology import report_text as rt
from orion.schemas.physiology_report import REPORT_SECTIONS, PhysiologyReportV1

FIXTURE = Path(__file__).resolve().parents[1] / "evals" / "fixtures" / "physiology_report_live_2026-10-29_24h.json"


def live() -> dict:
    return json.loads(FIXTURE.read_text())


def empty() -> dict:
    rep = PhysiologyReportV1(generated_at="2026-10-29T20:00:00Z",
                             window={"start": "2026-10-28T20:00:00Z", "end": "2026-10-29T20:00:00Z"},
                             missing=[{"section": s, "reason": f"why {s} is empty"} for s in REPORT_SECTIONS])
    return rep.model_dump(mode="json")


def test_fixture_is_a_valid_contract_report():
    PhysiologyReportV1.model_validate(live())


def test_sections_list_matches_the_contract():
    assert rt.SECTIONS == REPORT_SECTIONS


def test_overview_names_what_binds_in_its_own_words_and_fits_a_turn():
    text = rt.render_report_text(live())
    assert "Binding now: Energy money this month, Cabinet heat" in text
    assert "Orion's share 1.98 C" in text            # the Budget organ's own why, verbatim
    assert "$182.27 spent on electricity this month against a $100 envelope" in text
    assert "reaches $193.80" in text
    assert "hub.outreach (tier 2): 237 funded, 548 deferred" in text
    assert "ended without success" in text and "attempt_failed" in text
    assert "orion-kev/kev" in text
    assert len(text.splitlines()) <= 40 and len(text) <= 4000
    assert "None" not in text and "nan" not in text.lower()


def test_slack_lanes_fold_into_one_line_in_the_overview_but_not_in_full_limits():
    over = rt.render_report_text(live())
    assert "GPU lanes with slack:" in over and "GPU lane agent (" not in over
    full = rt.render_report_text(live(), ("limits",))
    assert "GPU lane agent (lease-seconds per hour): slack; 861 lease-s against a line of 3,600 lease-s" in full


def test_every_section_renders_or_says_why():
    for s in REPORT_SECTIONS:
        text = rt.render_report_text(empty(), (s,))
        assert f"{s}: no data. why {s} is empty" in text, s


def test_live_vram_and_value_sections_explain_their_gaps():
    text = rt.render_report_text(live(), ("vram", "value_per_cost", "unit_costs"))
    assert "vram: no data." in text and "gpuN_mem_used_mb" in text
    assert "hub.outreach: value 0.39 over 82 acts" in text
    assert "per 1k tokens: not measured" in text
    assert "100 W for 2 h on athena: $0.0251, 200 W·h, +0.64 °C cabinet" in text


def test_normalize_sections():
    assert rt.normalize_sections(None) == rt.OVERVIEW
    assert rt.normalize_sections("overview") == rt.OVERVIEW
    assert rt.normalize_sections("all") == rt.SECTIONS
    assert rt.normalize_sections("money, queue-wait") == ("money", "queue_wait")
    assert rt.normalize_sections(["decisions"]) == ("decisions",)
    with pytest.raises(ValueError, match="unknown section"):
        rt.normalize_sections("mood")


class _Handler(BaseHTTPRequestHandler):
    status = 200
    body = b"{}"
    seen: list = []

    def do_GET(self):  # noqa: N802
        type(self).seen.append(self.path)
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(type(self).body)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_fetch_passes_window_sections_family(server):
    _Handler.status, _Handler.body, _Handler.seen = 200, json.dumps(live()).encode(), []
    out = rt.fetch_report(base_url=server, window="7d", sections=("money", "decisions"), family="hub.outreach")
    assert out["money"]["mtd_usd"] == 182.27
    q = parse_qs(urlparse(_Handler.seen[0]).query)
    assert q == {"window": ["7d"], "sections": ["money,decisions"], "family": ["hub.outreach"]}


def test_fetch_failure_is_a_plain_reason(server):
    _Handler.status, _Handler.body = 503, json.dumps({"detail": "the physiology report is disabled"}).encode()
    with pytest.raises(rt.ReportUnavailable, match="HTTP 503 the physiology report is disabled"):
        rt.fetch_report(base_url=server)
    with pytest.raises(rt.ReportUnavailable, match="unreachable"):
        rt.fetch_report(base_url="http://127.0.0.1:9", timeout_sec=1.0)
    with pytest.raises(ValueError, match="window"):
        rt.fetch_report(base_url=server, window="1y")


def test_headline_never_counts_no_price_as_slack():
    rep = live()
    for lim in rep["limits"]:
        lim["now"]["bind_state"] = "slack" if lim["constraint"].startswith("lane:") else None
    text = rt.render_report_text(rep, ("limits",))
    assert "Nothing with a price binds right now." in text and "Unknown (no price or no reading): " in text
    assert "Nothing binds right now." not in text
    over = rt.render_report_text(rep)
    assert "Cabinet heat (my share against the soft line): no price" in over   # not folded into slack


def test_unknown_tokens_are_not_printed_as_zero():
    rep = live()
    rep["claude"]["window_tokens"] = {"input": None, "output": None, "cache_read": None, "cache_write": None}
    text = rt.render_report_text(rep, ("claude",))
    assert "token counts unknown" in text and "0 output tokens" not in text


def test_family_filter_is_named_in_the_header():
    rep = live()
    rep["query"] = {"family": "hub.outreach"}
    assert "family hub.outreach (narrows wants, decisions" in rt.render_report_text(rep).splitlines()[0]


def test_cap_text_keeps_replies_turn_sized():
    text = "\n".join(f"line {i} " + "x" * 150 for i in range(80))
    out = rt.cap_text(text)
    assert len(out.splitlines()) <= 40 and len(out) <= 4000 and "fewer sections" in out
    assert rt.cap_text("short") == "short"


def test_all_sections_give_one_line_each_and_fit_a_turn():
    text = rt.render_report_text(live(), rt.SECTIONS)
    assert len(text.splitlines()) <= 20 and len(text) <= 4000
    assert "processes: no data. per-container CPU watts are not reported yet" in text
    assert text.splitlines()[-1].startswith("(One line per section.")


def test_processes_render_largest_first():
    rep = live()
    rep["processes"] = [{"host": "circe", "name": "orion-circe-atlas-agent", "watts": 18.5, "cores": None},
                        {"host": "circe", "name": "(host, not in a container)", "watts": 2.0, "cores": None}]
    text = rt.render_report_text(rep, ("processes",))
    assert "- circe orion-circe-atlas-agent: 18.5 W" in text and "CPU watts above idle" in text


# --- hosts: what is spinning each machine (2026-10-30) ------------------------------------------

HOSTS_FIXTURE = Path(__file__).resolve().parents[1] / "evals" / "fixtures" / "physiology_report_live_2026-10-30_24h.json"


def hosts_live() -> dict:
    return json.loads(HOSTS_FIXTURE.read_text())


def test_hosts_fixture_is_a_valid_contract_report():
    rep = PhysiologyReportV1.model_validate(hosts_live())
    assert {h.host for h in rep.hosts} == {"circe", "athena"}


def test_overview_has_one_what_is_spinning_line_per_machine():
    rep = hosts_live()
    text = rt.render_report_text(rep)
    assert "hosts" in rt.OVERVIEW
    for h in rep["hosts"]:
        line = next(l for l in text.splitlines() if l.startswith(f"- {h['host']} draws "))
        assert f"{h['chassis_w_now']:.0f} W now vs {h['chassis_w_rest']:.0f} W at rest" in line
        assert "GPUs " in line and "CPU " in line and "other " in line
    assert "zelda_dojo (Juniper's)" in text            # sponsored work, apart from Orion's activity
    assert rt.cap_text(text) == text                  # still fits the dig cap


def test_hosts_full_view_says_measured_vs_estimated_and_names_holders():
    rep = hosts_live()
    text = rt.render_report_text(rep, ("hosts",))
    assert "measured: whole-machine watts" in text and "Estimated: rest (pinned" in text
    assert "- rest: pinned" in text and "2026-10-01..2026-10-13" in text
    assert "held by curiosity.investigate runs" in text or "held by reading.turn runs" in text
    assert "Juniper's sponsored work, not my activity: 32 containers" in text
    assert "None" not in text and "nan" not in text


def test_host_flag_narrows_only_the_machine_sections():
    rep = hosts_live()
    text = rt.render_report_text(rep, ("hosts", "processes"), host="circe")
    assert "machine circe (narrows hosts, processes, vram)" in text.splitlines()[0]
    assert "circe draws" in text and "athena draws" not in text
    assert "- athena " not in text
    assert all(r["host"] == "circe" for r in rt.narrow_to_host(rep, "circe")["processes"])
    gone = rt.render_report_text(rep, ("hosts",), host="hecate")
    assert "no fresh reading for hecate (machines with one: athena, circe)" in gone


def test_host_headline_words_and_drift():
    h = {"host": "circe", "chassis_w_now": 1150.0, "chassis_w_rest": 541.0, "above_rest_w": 609.0,
         "usd_per_month_at_this_rate": 55.8, "rest_usd_per_month": 49.6, "recent_floor_w": 600.0}
    assert rt.host_headline(h) == ("circe draws 1150 W now vs 541 W at rest (+609 W, about $56/month at this rate; "
                                   "my body at rest there costs about $50/month).")
    assert "59 W above the pinned rest: new always-on draw" in rt.floor_drift(h)
    assert rt.floor_drift({**h, "recent_floor_w": 550.0}) is None
    assert rt.host_headline({"host": "x", "chassis_w_now": None}) == "x: whole-machine draw unknown (no chassis reading)."
    assert rt.host_parts({"gpu_w": 72.0, "gpu_w_rest": 72.4, "cpu_w": 10.0, "cpu_w_rest": None, "other_w": None}) == \
        "GPUs 72 W (+0), CPU 10 W"


def test_energy_exogenous_bind_is_my_body_at_rest_not_not_my_load():
    """Juniper 2026-10-30: the always-on body is Orion's own; 'not from my load' was false for energy."""
    lim = {"constraint": "energy", "label": "Energy money this month", "unit": "usd", "line": 100.0,
           "now": {"level": 183.7, "bind_state": "exogenous_bind", "why": "my body at rest costs about $146/month"}}
    line = rt._limit_line(lim)
    assert "BINDING: my always-on body alone reaches the envelope" in line and "not from my load" not in line
    heat = {**lim, "constraint": "heat"}
    assert "not from my load" in rt._limit_line(heat)    # heat: the room really is outside Orion
    money = rt.render_money({"money": {"mtd_usd": 183.7, "envelope_usd": 100.0, "forecast_usd": 196.7, "floor_usd": 194.0}})
    assert "My always-on body alone (rest plus anything left running) reaches $194.00 this month" in money[0] and "switched on" not in money[0]


# --- overhead: every cost number on the whole-machine footing (live 2026-10-30T04:37Z) -------------

OVERHEAD_FIXTURE = Path(__file__).resolve().parents[1] / "evals" / "fixtures" / \
    "physiology_report_live_2026-10-30_overhead_24h.json"


def overhead_live() -> dict:
    return json.loads(OVERHEAD_FIXTURE.read_text())


def test_overhead_fixture_is_a_valid_contract_report_with_a_fit_per_host():
    rep = PhysiologyReportV1.model_validate(overhead_live())
    by = {h.host: h for h in rep.hosts}
    assert by["circe"].overhead_fitted and 2.0 < by["circe"].overhead_factor < 2.8
    assert by["athena"].overhead_fitted and 1.0 < by["athena"].overhead_factor < 1.6   # per host: not one constant
    for h in rep.hosts:
        for c in h.cards:
            if c.above_idle_w is not None:
                assert c.watts_at_wall_est == pytest.approx(c.above_idle_w * h.overhead_factor, abs=0.2)
    paint = next(u for u in rep.unit_costs if u.unit.startswith("one reverie_diffusion run (painting)"))
    assert paint.status == "derived" and paint.wh == pytest.approx(paint.wh_component * paint.overhead_factor, abs=1e-2)


def test_cards_containers_and_processes_show_both_numbers_and_say_estimate():
    rep = overhead_live()
    text = rt.render_report_text(rep, ("hosts", "processes"))
    circe = next(h for h in rep["hosts"] if h["host"] == "circe")
    gpu1 = next(c for c in circe["cards"] if c["card"] == "gpu1")
    assert f"- gpu1 {gpu1['watts']:.0f} W, rest {gpu1['idle_w']:.0f} W (≈{gpu1['watts_at_wall_est']:.0f} W at the wall above rest, est.)" in text
    assert f"- overhead: each watt of GPU/CPU load above idle draws about {circe['overhead_factor']:.2f} W at the wall above rest" in text
    assert "with an intercept:" in text                     # drift above the pinned rest is visible
    assert "at-the-wall figures are ESTIMATES" in text and "fitted on circe" in text and "fitted on athena" in text
    p = rep["processes"][0]
    assert f"- {p['host']} {p['name']}: {p['watts']:.1f} W CPU (≈{p['watts_at_wall_est']:.0f} W at the wall above rest, est.)" in text
    k = {h["host"]: h["overhead_factor"] for h in rep["hosts"]}
    in_procs = sorted({p_["host"] for p_ in rep["processes"]})            # the top 12 can all be one machine
    assert ", ".join(f"{h} x {k[h]:.2f} (fitted)" for h in in_procs) in text
    assert "every 'at the wall' figure" in text and "None" not in text and "nan" not in text
    overview = rt.render_report_text(rep)
    busiest = max(circe["cards"], key=lambda c: c.get("above_idle_w") or 0.0)   # the overview shows busy cards only
    if busiest["above_idle_w"] >= rt.CARD_BUSY_ABOVE_W:
        assert f"(≈{busiest['watts_at_wall_est']:.0f} W at the wall above rest, est.)" in overview
    assert overview.count("at-the-wall figures are ESTIMATES") == 1
    assert rt.cap_text(overview) == overview


def test_parts_estimates_far_from_the_measured_surplus_now_are_said():
    h = {"host": "circe", "above_rest_w": 637.0, "wall_est_parts_w": 1210.0, "overhead_now_ratio": 1.23,
         "overhead_factor": 2.34}
    words = rt.overhead_lag_words(h)
    assert "add to 1210 W against 637 W measured above rest (this sample's ratio 1.23, the fitted 2.34)" in words
    assert "average cost over hours, not a split of this second" in words
    assert rt.overhead_lag_words({**h, "wall_est_parts_w": 700.0}) is None      # within 25%: nothing to say
    assert rt.overhead_lag_words({**h, "wall_est_parts_w": None}) is None


def test_fallback_factor_is_named_in_the_overview():
    rep = overhead_live()
    for h in rep["hosts"]:
        h["overhead_basis"] = "PR #2992 fit on circe ...; not fitted on " + h["host"] + ": 3 loaded hours, need 48"
        h["overhead_factor"], h["overhead_fitted"] = 2.69, False
    line = rt.overhead_summary(rep["hosts"])
    assert "circe x 2.69 (fallback, not fitted here)" in line
    # processes rows carry no overhead_fitted: the basis decides
    assert "(fallback, not fitted here)" in rt.overhead_summary(
        [{"host": "x", "overhead_factor": 2.69, "overhead_basis": "PR #2992 fit ...; not fitted on x"}])


def test_old_reports_without_overhead_fields_render_as_before():
    rep = hosts_live()                        # 2026-10-30 fixture, before the overhead fields existed
    text = rt.render_report_text(rep, ("hosts", "processes"))
    assert "at the wall" not in text.replace("every 'at the wall' figure", "") and "ESTIMATES" not in text


def test_charged_to_no_act_names_its_hour():
    rep = overhead_live()
    circe = next(h for h in rep["hosts"] if h["host"] == "circe")
    text = rt.render_report_text(rep, ("hosts",), host="circe")
    line = next(l for l in text.splitlines() if l.startswith("- above rest but charged to no act:"))
    assert f"{circe['unattributed_above_rest_w']:.0f} W" in line and "on GPU leases whose turn has no act" in line
    assert f"average over {circe['attribution_hour_start'][11:16]}-" in line and "newest complete hour" in line
    assert "until the whole-machine per-act meter" not in text


def test_town_sessions_read_in_plain_words():
    rep = empty()
    rep["wants"]["town"] = {
        "local_date": "2026-10-30", "orion_seconds_today": 1800.0, "manual_seconds_today": 600.0,
        "daily_cap_seconds": 7200.0, "open_kind": "orion",
        "sessions": [{"session_id": "o1", "kind": "orion", "started_at": "2026-10-30T15:00:00Z",
                      "seconds": 1800.0, "start_reason": "want_funded", "stop_reason": "session_cap",
                      "utterances": 2}],
    }
    text = rt.render_report_text(rep, sections=("wants",))
    assert "AI Town today (2026-10-30): I spent 30 min of my 2.0 h cap" in text
    assert "Juniper opened it for 10 min (not counted)" in text
    assert "I chose it; ended: session_cap; 2 line(s) I spoke" in text
