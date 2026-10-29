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
