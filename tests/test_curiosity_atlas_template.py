"""The Curiosity page's render logic, executed rather than eyeballed.

The render functions run under node against fixture payloads, with
`document` stubbed, so the assertions are on the HTML the page would produce.
Layout and paint are still unverified by anything but looking at the page.

What this guards (design doc 2026-09-22-curiosity-tab-redesign, acceptance
checks 5 and 6): the page has no path that grows with the total number of
runs ever; the three lines are named in plain words; a run is addressable
by `#run=<id>`; and a poll that returns the same payload touches no DOM.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "services" / "orion-hub" / "templates" / "curiosity_atlas.html"
)
_HUB = Path(__file__).resolve().parents[1] / "services" / "orion-hub"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not available on this host"
)

_IDS = ["updated", "banners", "tiles", "strip", "strip-days", "strip-note", "legend",
        "story", "story-note", "self-current", "self-lived-wrap", "self-lived-summary",
        "self-lived", "self-history-wrap", "self-history-summary", "self-history",
        "priors", "priors-note", "toggle-closed", "prior-filter", "briefs-summary",
        "peer-briefs-table", "refresh", "run-now", "run-now-msg", "self-run-now",
        "self-run-now-msg"]

_HARNESS = """
const ELS = {};
const mk = (id) => ({ id, innerHTML: "", textContent: "", hidden: false, value: "",
  classList: { toggle(){}, add(){}, remove(){} },
  addEventListener(){}, setAttribute(){}, getAttribute(){ return "false"; } });
for (const id of __IDS__) ELS[id] = mk(id);
globalThis.document = {
  getElementById: (id) => ELS[id] || mk(id),
  addEventListener(){}, querySelectorAll(){ return []; }, hidden: false,
};
globalThis.window = globalThis;
globalThis.location = { hash: "" };
globalThis.history = { replaceState(){} };
globalThis.fetch = async () => { throw new Error("fetch is not available in the harness"); };
__SCRIPT__
const fx = JSON.parse(require("fs").readFileSync(process.argv[2], "utf8"));
const out = { renders: {} };
if (fx.runs) {
  out.renders.first = applyRuns(fx.runs, fx.now || Date.now());
  out.renders.second = applyRuns(JSON.parse(JSON.stringify(fx.runs)), fx.now || Date.now());
  if (fx.runs_changed) out.renders.third = applyRuns(fx.runs_changed, fx.now || Date.now());
}
if (fx.atlas) { applyAtlas(fx.atlas); if (fx.show_closed) { state.showClosed = true; renderPriors(fx.atlas); }
  if (fx.filter) { state.filter = fx.filter; renderPriors(fx.atlas); } }
if (fx.story) renderStory(fx.story);
for (const k of Object.keys(ELS)) out[k] = ELS[k].innerHTML || ELS[k].textContent;
console.log(JSON.stringify(out));
"""


def _script() -> str:
    html = TEMPLATE.read_text(encoding="utf-8")
    script = re.search(r"<script>\n(.*?)\n</script>", html, re.S)
    assert script, "no <script> block in the template"
    js = script.group(1)
    # Drop the bootstrap only: `boot()` wires listeners and starts the poll.
    js = re.sub(r"^boot\(\);\s*$", "", js, flags=re.M)
    assert "boot();" not in js
    return js


def _render(fixture: dict, tmp_path: Path) -> dict:
    hp = tmp_path / "harness.js"
    hp.write_text(_HARNESS.replace("__SCRIPT__", _script()).replace("__IDS__", json.dumps(_IDS)), encoding="utf-8")
    fp = tmp_path / "fixture.json"
    fp.write_text(json.dumps(fixture), encoding="utf-8")
    proc = subprocess.run(["node", str(hp), str(fp)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr[:3000]
    return json.loads(proc.stdout)


NOW_MS = int(time.time() * 1000)


def _run(**over) -> dict:
    base = {
        "run_id": "446ddd7165d5", "line": "investigate", "line_known": True,
        "plain_line_label": "World question", "started_at": NOW_MS - 3_600_000,
        "started_from": "admission", "finished_at": NOW_MS - 600_000, "duration_sec": 3000.0,
        "accepted_at": NOW_MS - 3_600_000, "admitted_at": NOW_MS - 2_000_000, "lane": "agent",
        "lane_wait_sec": 1600.0, "active_sec": 1400.0, "retries": 0, "anomalies": {},
        "status": "completed", "attempts": 1, "error": "", "hops": 2, "findings": 1, "revisions": 1,
        "prior_touched": {"prior_id": "p1", "claim": "who matters", "from": 0.6, "to": 0.68,
                          "from_status": "revised", "to_status": "supported"},
        "reach_out": {"wanted": False, "why": "", "decision": None, "gate": None, "decided_at": None,
                      "sent_at": None, "composed_text": "", "reply": None},
        "journal_entry_id": "j", "finding_text": "f", "self_written": None, "self_sense": None,
        "harness": None, "outcome_kind": "finished",
    }
    base.update(over)
    return base


def _runs(runs=None, **over) -> dict:
    base = {
        "available": True, "window_days": 14, "line": "all",
        "stores": {"postgres": "ok", "graph": "ok"},
        "runs": runs if runs is not None else [_run()],
        "reach_outs": {"wanted": 6, "sent": 0, "blocked_by": {"blocked:daily_cap": 4}, "top_block_reason": "blocked:daily_cap", "not_recorded": 2},
        "totals": {"investigate": 1, "self_inquiry": 0, "self_sense_eval": 0},
        "schedule": {"available": True, "local_date": "2026-09-22", "tz": "America/Denver",
                     "runs_seen_today": {"investigate": 2, "self_inquiry": 1, "self_sense_eval": 4},
                     "lines": {
                         "investigate": {"enabled": True, "daily_cap": 3, "cooldown_sec": 14400.0, "runs_today": 2,
                                         "last_at": None, "next_eligible_at": None},
                         "self_inquiry": {"enabled": True, "daily_cap": 3, "cooldown_sec": 7200.0, "runs_today": 1,
                                          "last_at": None, "next_eligible_at": None},
                         "self_sense_eval": {"enabled": True, "daily_cap": 7, "cooldown_sec": 10800.0, "runs_today": 4,
                                             "last_at": None, "next_eligible_at": None},
                     }},
    }
    base.update(over)
    return base


def _atlas(priors=None, **over) -> dict:
    base = {"available": True, "live_total": 1, "closed_total": 0, "pool_is_dead": False,
            "history_recorded": True, "priors": priors or [], "revisions": [], "peer_briefs": [],
            "self": {"available": True, "current": {"version": 3, "created_at": None, "content": "I am the loop.",
                                                    "evidence_refs": [], "produced_by": "curiosity_self_inquiry"},
                     "history": [], "lived_answers": [], "journal_entries": [], "latest_eval_run_id": None, "latest_eval": []}}
    base.update(over)
    return base


def _prior(**over) -> dict:
    base = {"prior_id": "p1", "claim": "a live claim", "confidence": 0.68, "status": "supported",
            "times_tested": 2, "is_closed": False, "created_by_run": "r1", "last_run_id": "446ddd7165d5",
            "formed_from": "", "last_tested_at": "", "last_tested_at_ms": NOW_MS - 1000, "why": "",
            "trajectory": [{"run_id": "", "confidence": 0.6, "status": "revised", "recorded": True, "written_at": None},
                           {"run_id": "446ddd7165d5", "confidence": 0.68, "status": "supported", "recorded": True, "written_at": NOW_MS - 1000}]}
    base.update(over)
    return base


# --- the page is bounded and names its parts in plain words -----------------


def test_the_three_lines_are_named_in_plain_words_and_the_old_unbounded_panels_are_gone() -> None:
    page = TEMPLATE.read_text(encoding="utf-8")
    for label in ("World question", "Self question", "Self-sense check"):
        assert label in page, label
    assert 'id="strip"' in page and 'id="story"' in page
    assert 'id="growth"' not in page and "What each run added" not in page
    assert 'id="priors-table"' not in page
    assert "<h2>Every prior</h2>" not in page and 'id="priors-table"' not in page
    assert "renderGrowth" not in page and "total_added" not in page
    assert "{{HUB_UI_ASSET_VERSION}}" in page


def test_the_page_fetches_the_two_run_endpoints_and_routes_by_hash() -> None:
    page = TEMPLATE.read_text(encoding="utf-8")
    assert "/curiosity/api/runs?days=" in page
    assert "/curiosity/api/run/" in page
    assert "/curiosity/api/atlas" in page
    assert "#run=" in page and "hashchange" in page and "location.hash" in page
    assert "payloadHash" in page and "state.runsHash" in page, "the hash-compare guard"


def test_a_poll_with_an_unchanged_payload_touches_no_dom(tmp_path) -> None:
    """Acceptance check 6. The previous page rewrote eight sections every
    60s and lost every open disclosure and scroll position."""
    changed = _runs(runs=[_run(), _run(run_id="second", line="self_inquiry", plain_line_label="Self question")])
    out = _render({"runs": _runs(), "runs_changed": changed}, tmp_path)
    assert out["renders"] == {"first": True, "second": False, "third": True}


def test_the_budget_tiles_say_n_of_cap_per_line_and_flag_a_traceless_run(tmp_path) -> None:
    out = _render({"runs": _runs()}, tmp_path)
    tiles = out["tiles"]
    assert "World questions" in tiles and "Self questions" in tiles and "Self-sense checks" in tiles
    assert re.search(r"2<span class=\"unit\">of 3 today", tiles)
    assert re.search(r"4<span class=\"unit\">of 7 today", tiles)
    assert "0<span class=\"unit\">of 6 sent" in tiles
    assert "most blocked by daily cap" in tiles
    assert "2 with no decision recorded" in tiles
    assert "left no trace" not in tiles, "counter 2, stores 2: nothing missing"


def test_a_counter_ahead_of_the_stores_is_named_on_the_tile(tmp_path) -> None:
    runs = _runs()
    runs["schedule"]["runs_seen_today"] = {"investigate": 1}
    out = _render({"runs": runs}, tmp_path)
    assert "counter says 2, the stores show 1: 1 left no trace" in out["tiles"]


def test_the_strip_has_fourteen_days_and_one_chip_per_run_by_line_and_outcome(tmp_path) -> None:
    runs = _runs(runs=[
        _run(),
        _run(run_id="dead", status="failed", outcome_kind="died", error="rpc:TimeoutError", retries=1),
        _run(run_id="ask", outcome_kind="reached_out_blocked",
             reach_out={"wanted": True, "why": "w", "decision": "blocked:daily_cap", "gate": "daily_cap",
                        "decided_at": None, "sent_at": None, "composed_text": "", "reply": None}),
        _run(run_id="sense", line="self_sense_eval", plain_line_label="Self-sense check"),
        _run(run_id="undated", started_at=None, finished_at=None, accepted_at=None, admitted_at=None),
    ])
    out = _render({"runs": runs, "now": NOW_MS}, tmp_path)
    strip = out["strip"]
    assert strip.count('class="day') == 14
    assert strip.count('data-run="') == 4, "an undated run is on no day"
    assert 'class="chip line-investigate glyph-finished' in strip
    assert 'class="chip line-investigate glyph-died' in strip and "has-retries" in strip
    assert 'glyph-reached_out_blocked' in strip and "reach-out: blocked:daily_cap" in strip
    assert 'chip line-self_sense_eval' in strip
    assert "1 with no clock, not shown" in out["strip-note"]
    assert "World question" in out["legend"] and "retried at least once" in out["legend"]


def test_the_story_prints_a_relative_clock_the_lane_wait_and_the_reach_out_decision(tmp_path) -> None:
    run = _run(reach_out={"wanted": True, "why": "the repair step recorded its own prompt",
                          "decision": "blocked:daily_cap", "gate": "daily_cap", "decided_at": NOW_MS,
                          "sent_at": None, "composed_text": "", "reply": None}, retries=1,
               anomalies={"run.checkpoint_resume_failed": 12})
    story = {
        "available": True, "found": True, "run": run, "readings_available": False, "harness": None,
        "journal_body": "the journal prose",
        "timeline": [
            {"at": run["accepted_at"], "offset_sec": 0.0, "kind": "lifecycle", "attempt": None, "status": "accepted", "node": ""},
            {"at": run["accepted_at"], "offset_sec": 0.1, "kind": "lifecycle", "attempt": None, "status": "waiting", "node": "", "lane": "agent"},
            {"at": run["admitted_at"], "offset_sec": 1600.0, "kind": "lifecycle", "attempt": None, "status": "admitted", "node": "", "lane": "agent", "wait_sec": 1600.0},
            {"at": run["admitted_at"] + 1000, "offset_sec": 1601.0, "kind": "role_choice", "attempt": None, "choice": "local_crawl", "why": "queue pressure was high"},
            {"at": run["admitted_at"] + 714000, "offset_sec": 2314.0, "kind": "hop", "attempt": 1, "n": 1, "note": "first note", "readings": []},
            {"at": run["admitted_at"] + 720000, "offset_sec": 2320.0, "kind": "lifecycle", "attempt": None, "status": "retrying", "node": "harness_turn", "error": "no_final_frame"},
            {"at": run["admitted_at"] + 730000, "offset_sec": 2330.0, "kind": "attempt", "attempt": 2, "n": 1},
            {"at": run["admitted_at"] + 731000, "offset_sec": 2331.0, "kind": "hop", "attempt": 2, "n": 1, "note": "again", "readings": []},
            {"at": None, "offset_sec": None, "kind": "finding", "attempt": None, "text": "the finding", "evidence": "psql"},
            {"at": run["admitted_at"] + 740000, "offset_sec": 2340.0, "kind": "revision", "attempt": None, "prior_id": "p1", "claim": "who matters", "from": 0.6, "to": 0.68, "from_status": "revised", "to_status": "supported"},
            {"at": run["finished_at"], "offset_sec": 3000.0, "kind": "outcome", "attempt": None, "continue_line": True, "continue_note": "read the 7 rows", "reach_out": True, "reach_out_why": "the repair step recorded its own prompt"},
            {"at": run["finished_at"], "offset_sec": 3000.0, "kind": "lifecycle", "attempt": None, "status": "completed", "node": "finish"},
            {"at": NOW_MS, "offset_sec": 3600.0, "kind": "outreach", "attempt": None, "decision": "blocked:daily_cap", "gate": "daily_cap", "composed_text": "", "sent_at": None},
        ],
    }
    out = _render({"story": story}, tmp_path)
    s = out["story"]
    assert "+0:00" in s and "+26:40" in s and "+38:34" in s and "+50:00" in s
    assert "waiting for the agent lane" in s
    assert "got the agent lane</span> after 27 min waiting" in s
    assert "waited 27 min for the agent lane" in s and "ran 23 min" in s
    assert "hop 1</span>" in s and "(attempt 2)" in s and "── attempt 2 ──" in s
    assert "turn died, retrying</span> at harness_turn" in s
    assert "1 retry" in s and "12 checkpoint resume failed" in s
    assert 'class="t unk"' in s, "an undated finding is printed with no offset, not a fake one"
    assert "prior 0.60 → 0.68</span>, revised → supported" in s
    assert "blocked: daily cap" in s
    assert "reach-out:</span> wanted — <span class=\"txt\">the repair step recorded its own prompt" in s
    assert "reply:</span> —" in s
    assert "No supervisor readings" in s
    assert "harness timing: not recorded" in s
    assert "Journal entry" in s and "the journal prose" in s


def test_a_sent_reach_out_with_a_reply_reads_as_such(tmp_path) -> None:
    run = _run(outcome_kind="reached_out_sent",
               reach_out={"wanted": True, "why": "w", "decision": "sent", "gate": None, "decided_at": NOW_MS,
                          "sent_at": NOW_MS, "composed_text": "Juniper, the repair step…",
                          "reply": {"at": NOW_MS + 60000, "text": "oh no, which step?"}},
               harness={"elapsed_sec": 712.4, "turn_correlation_id": "abc"})
    story = {"available": True, "found": True, "run": run, "readings_available": True, "journal_body": "",
             "timeline": [{"at": NOW_MS, "offset_sec": 3600.0, "kind": "outreach", "decision": "sent", "gate": None, "composed_text": "Juniper, the repair step…", "sent_at": NOW_MS},
                          {"at": NOW_MS + 60000, "offset_sec": 3660.0, "kind": "reply", "text": "oh no, which step?"}]}
    out = _render({"story": story}, tmp_path)
    s = out["story"]
    assert 'class="dec sent">sent' in s
    assert "Juniper replied</span>" in s and "next message in that session, within 12h" in s
    assert "oh no, which step?" in s
    assert "12 min in the harness" in s and ">abc<" in s
    assert "No supervisor readings" not in s


def test_a_not_recorded_decision_is_not_dressed_up_as_a_gate(tmp_path) -> None:
    run = _run(outcome_kind="reached_out_blocked",
               reach_out={"wanted": True, "why": "w", "decision": "not_recorded", "gate": None, "decided_at": None,
                          "sent_at": None, "composed_text": "", "reply": None})
    out = _render({"story": {"available": True, "found": True, "run": run, "readings_available": False, "journal_body": "", "timeline": []}}, tmp_path)
    assert "not recorded — nothing wrote a decision" in out["story"]
    assert "blocked:" not in out["story"]


def test_a_missing_run_and_an_unreadable_run_read_differently(tmp_path) -> None:
    missing = _render({"story": {"available": True, "found": False, "run_id": "nope"}}, tmp_path)
    broken = _render({"story": {"available": False, "reason": "ConnectionError: nope"}}, tmp_path)
    assert "No store knows run" in missing["story"]
    assert "Could not read this run: ConnectionError" in broken["story"]


def test_priors_hide_closed_by_default_and_the_toggle_and_filter_work(tmp_path) -> None:
    priors = [_prior(), _prior(prior_id="dead", claim="a refuted claim", status="refuted", is_closed=True, trajectory=[])]
    atlas = _atlas(priors=priors, live_total=1, closed_total=1)
    out = _render({"atlas": atlas}, tmp_path)
    assert "a live claim" in out["priors"] and "a refuted claim" not in out["priors"]
    assert "Show closed (1)" in out["toggle-closed"]
    assert "1 shown of 2 (1 live, 1 closed)" in out["priors-note"]
    assert "<svg" in out["priors"], "a recorded two-point history draws a time-axis sparkline"
    assert 'data-run="446ddd7165d5"' in out["priors"], "last tested by opens that run"

    shown = _render({"atlas": atlas, "show_closed": True}, tmp_path)
    assert "a refuted claim" in shown["priors"] and "Hide closed (1)" in shown["toggle-closed"]
    filtered = _render({"atlas": atlas, "show_closed": True, "filter": "refuted"}, tmp_path)
    assert "a refuted claim" in filtered["priors"] and "a live claim" not in filtered["priors"]


def test_an_unrecorded_trajectory_says_not_recorded_not_never_moved(tmp_path) -> None:
    prior = _prior(trajectory=[{"run_id": "r1", "confidence": 0.85, "status": "open", "recorded": False, "written_at": None}])
    out = _render({"atlas": _atlas(priors=[prior], history_recorded=False)}, tmp_path)
    assert "no revision recorded yet" in out["priors"]


def test_the_self_card_and_briefs_render_and_prose_is_escaped(tmp_path) -> None:
    atlas = _atlas(peer_briefs=[{"brief_id": "b", "help_id": "h", "run_id": "r9", "peer": "claude",
                                 "status": "refused_budget", "summary": "", "refusal_reason": "no budget"}])
    atlas["self"]["current"]["content"] = '<img src=x onerror="alert(1)"> & more'
    out = _render({"atlas": atlas}, tmp_path)
    assert "&lt;img" in out["self-current"] and "<img" not in out["self-current"]
    assert "&amp; more" in out["self-current"]
    assert "Contractor briefs (1)" in out["briefs-summary"]
    assert "refused: no budget" in out["peer-briefs-table"]


def test_orion_prose_in_the_story_is_escaped(tmp_path) -> None:
    run = _run()
    story = {"available": True, "found": True, "run": run, "readings_available": False, "journal_body": "",
             "timeline": [{"at": NOW_MS, "offset_sec": 1.0, "kind": "hop", "attempt": 1, "n": 1,
                           "note": '<img src=x onerror="alert(1)"> a & b < c', "readings": []}]}
    out = _render({"story": story}, tmp_path)
    assert "<img" not in out["story"] and "&lt;img" in out["story"] and "a &amp; b &lt; c" in out["story"]


def test_an_unreadable_store_and_an_unconfigured_graph_read_differently(tmp_path) -> None:
    broken = _render({"runs": _runs(available=False, reason="RuntimeError: pg down", runs=[])}, tmp_path)
    assert "Sittings could not be read" in broken["banners"]
    partial = _render({"runs": _runs(stores={"postgres": "ok", "graph": "ConnectionError: nope"})}, tmp_path)
    assert "graph read failed" in partial["banners"] and "partial" in partial["banners"]
    off = _render({"runs": _runs(stores={"postgres": "ok", "graph": "graph_not_configured"})}, tmp_path)
    assert "switched off" in off["banners"] and "failed" not in off["banners"]


# --- the Hub tab -----------------------------------------------------------
#
# CLAUDE.md section 9: a rendered template, a linked asset, and the changed
# interaction are three separate things and all three have to be checked.


def test_the_tab_button_and_its_panel_both_exist() -> None:
    index = (_HUB / "templates" / "index.html").read_text(encoding="utf-8")
    assert 'id="curiosityAtlasTabButton"' in index
    assert 'href="#curiosity-atlas"' in index
    assert 'id="curiosity-atlas" data-panel="curiosity-atlas"' in index
    assert 'src="/curiosity"' in index, "the iframe must point at the real route"


def test_every_hub_side_wire_the_tab_needs_is_present() -> None:
    app = (_HUB / "static" / "js" / "app.js").read_text(encoding="utf-8")
    for needle in (
        'document.getElementById("curiosityAtlasTabButton")',
        'document.getElementById("curiosity-atlas")',
        'document.getElementById("curiosityAtlasPanelFrame")',
        'document.getElementById("curiosityAtlasPanelRefresh")',
        'effectiveTab === "curiosity-atlas"',
        'setActiveTab("curiosity-atlas")',
        'h === "#curiosity-atlas"',
        "styleTabButton(curiosityAtlasTabButton, isCuriosityAtlas)",
    ):
        assert needle in app, needle


def test_the_page_stops_polling_when_its_panel_is_hidden() -> None:
    """The iframe keeps running behind a hidden tab. Without this contract the
    page is a FalkorDB read every 60s for a panel nobody is looking at."""
    page = TEMPLATE.read_text(encoding="utf-8")
    assert "window.OrionCuriosityAtlas" in page
    for fn in ("refresh", "activate", "deactivate"):
        assert fn in page, fn
    assert "clearInterval" in page, "deactivate must actually stop the timer"
    assert "visibilitychange" in page

    app = (_HUB / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert "OrionCuriosityAtlas" in app, "the host never calls the contract"
    assert 'ping("deactivate")' in app, "hiding the tab must call deactivate"


def test_the_standalone_link_and_the_iframe_agree_on_the_route() -> None:
    index = (_HUB / "templates" / "index.html").read_text(encoding="utf-8")
    block = index.split('id="curiosity-atlas"', 1)[1].split("</section>", 1)[0]
    assert block.count('"/curiosity"') == 2, block.count('"/curiosity"')
