"""The atlas page's render logic, executed rather than eyeballed.

Two of the eight findings in this branch's review were bugs in the template's
JavaScript, and the Python tests next door could not have caught either: the
growth panel filtered its segments to a hardcoded list of node kinds while the
total summed every kind, so a run writing `:PriorRevision` rendered a full-width
bar labelled "1" beside a total of "6"; and the "left no trace" banner counted a
run killed mid-write as having written nothing, contradicting its own body text
and the ledger pill that already reported it.

So the render functions run here, under node, against fixture payloads. The
harness stubs `document` rather than a browser: this asserts on the HTML the
functions produce, which is where both bugs lived. Layout and paint are still
unverified by anything but looking at the page.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "services" / "orion-hub" / "templates" / "curiosity_atlas.html"
)

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not available on this host"
)

_HARNESS = """
const ELS = {};
const mk = (id) => ({ id, innerHTML: "", textContent: "", hidden: false,
  addEventListener(){}, setAttribute(){}, getAttribute(){ return "false"; } });
for (const id of ["updated","banners","tiles","traj-note","trajectories",
                  "growth","growth-table","runs","priors-table","refresh",
                  "toggle-tables"]) ELS[id] = mk(id);
globalThis.document = {
  getElementById: (id) => ELS[id] || mk(id),
  addEventListener(){}, hidden: false,
};
globalThis.window = globalThis;
__SCRIPT__
const d = JSON.parse(require("fs").readFileSync(process.argv[2], "utf8"));
renderBanners(d); renderTiles(d); renderTrajectories(d);
renderGrowth(d); renderRuns(d); renderPriors(d);
const out = {};
for (const k of Object.keys(ELS)) out[k] = ELS[k].innerHTML || ELS[k].textContent;
console.log(JSON.stringify(out));
"""


def _render(payload: dict, tmp_path: Path) -> dict[str, str]:
    html = TEMPLATE.read_text(encoding="utf-8")
    script = re.search(r"<script>\n(.*?)\n</script>", html, re.S)
    assert script, "no <script> block in the template"
    js = script.group(1)
    # Drop the bootstrap: it wires listeners and starts a poll on a live page.
    # `activate()` is the entry point, so removing the bare call is what stops
    # the harness fetching; the functions themselves stay defined and asserted
    # on below.
    js = js.replace('$("refresh").addEventListener("click", load);', "")
    js = re.sub(r'\$\("toggle-tables"\)\.addEventListener[\s\S]*?\}\);\n', "", js)
    js = re.sub(r'^activate\(\);\s*$', "", js, flags=re.M)

    hp = tmp_path / "harness.js"
    hp.write_text(_HARNESS.replace("__SCRIPT__", js), encoding="utf-8")
    pp = tmp_path / "payload.json"
    pp.write_text(json.dumps(payload), encoding="utf-8")
    proc = subprocess.run(
        ["node", str(hp), str(pp)], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr[:2000]
    return json.loads(proc.stdout)


def _now_ms() -> int:
    """A timestamp that is "today" whenever the suite runs.

    A hardcoded epoch here is a test that passes until midnight: this file was
    written on 2026-08-27 with a literal 2026-08-27 stamp and started failing
    the moment the date rolled over, in the two tests whose whole subject is
    "did this run happen today".
    """
    import time

    return int(time.time() * 1000)


def _run(**over) -> dict:
    base = {
        "run_id": "r1", "written_at": _now_ms(), "hops": 1, "hop_notes": [],
        "findings": [], "added": {"Prior": 1}, "priors_created": [],
        "priors_touched": [], "continue_line": False, "continue_note": "",
        "reach_out": False, "reach_out_why": "", "total_added": 1,
    }
    base.update(over)
    return base


def _payload(**over) -> dict:
    base = {
        "available": True, "live_total": 1, "closed_total": 0,
        "pool_is_dead": False, "history_recorded": False,
        "priors": [], "runs": [], "revisions": [],
        "schedule": {"available": False, "runs_today": None,
                     "runs_wrote_today": None, "daily_cap": 3,
                     "next_eligible_at": None, "cooldown_sec": None,
                     "last_investigation_at": None},
    }
    base.update(over)
    return base


def _segments(growth_html: str) -> list[int]:
    return [int(m or 0) for m in re.findall(r'class="seg"[^>]*>(\d*)<', growth_html)]


def _series_vars(growth_html: str) -> list[str]:
    """Which palette slot each segment actually drew with.

    Read off the rendered style rather than from KIND_ORDER, so a kind that
    falls through to the last colour is indistinguishable from one that was
    assigned it -- which is the bug this exists to catch.
    """
    return re.findall(r"var\((--series-\d+)\)", growth_html)


def test_every_node_kind_gets_a_segment_even_one_nobody_hardcoded(tmp_path) -> None:
    """The growth query counts every labelled node and `total_added` sums all of
    them. A segment list filtered to a hardcoded set made the bar disagree with
    its own total — the review finding, in the case that produced it."""
    run = _run(added={"Prior": 1, "PriorRevision": 5, "Concept": 2, "Unheard": 3},
               total_added=11)
    out = _render(_payload(runs=[run]), tmp_path)
    assert sum(_segments(out["growth"])) == 11
    assert "PriorRevision" in out["growth"]
    assert "Unheard" in out["growth"], "an unknown kind must still be shown"


def test_the_segments_always_sum_to_the_total_column(tmp_path) -> None:
    run = _run(added={"Prior": 2, "Finding": 1, "Hop": 5, "TurnOutcome": 1,
                      "PriorRevision": 2}, total_added=11)
    out = _render(_payload(runs=[run]), tmp_path)
    total = int(re.search(r'class="total">(\d+)', out["growth"]).group(1))
    assert sum(_segments(out["growth"])) == total == 11


# The palette carries NINE slots: six node kinds plus the three edge kinds.
# Before the edge kinds were listed, `-> SUPPORTS`, `-> CONTRADICTS` and
# `-> ABOUT` all fell through to the final colour -- one hue for three
# different edges -- and a run writing six node labels plus a single edge
# crossed the fold threshold and pushed real node labels into grey.
_KNOWN_KINDS = ["Prior", "Finding", "Hop", "TurnOutcome", "PriorRevision",
                "Concept", "-> SUPPORTS", "-> CONTRADICTS", "-> ABOUT"]


def test_every_named_kind_gets_its_own_hue_with_nothing_folded(tmp_path) -> None:
    added = {k: 1 for k in _KNOWN_KINDS}
    out = _render(_payload(runs=[_run(added=added, total_added=9)]), tmp_path)
    assert "Other" not in out["growth"], "a named kind was folded away"
    assert sum(_segments(out["growth"])) == 9


def test_the_kind_after_the_last_slot_folds_rather_than_inventing_a_hue(tmp_path) -> None:
    added = {k: 1 for k in _KNOWN_KINDS + ["Extra1", "Extra2"]}
    out = _render(_payload(runs=[_run(added=added, total_added=11)]), tmp_path)
    assert "Other" in out["growth"]
    assert sum(_segments(out["growth"])) == 11, "folding must not lose a node"


def test_an_edge_kind_does_not_take_a_node_kinds_colour(tmp_path) -> None:
    """The failure this guards is silent and visual: three distinct edge types
    rendering as one another, in a legend that looks complete."""
    added = {k: 1 for k in _KNOWN_KINDS}
    out = _render(_payload(runs=[_run(added=added, total_added=9)]), tmp_path)
    # Each slot appears twice -- once in the bar, once in the legend swatch --
    # so the claim is that NINE DISTINCT slots are in play, not that no var
    # repeats. Before the edge kinds were named, the three edge types all drew
    # `--series-9` and this collapsed to seven.
    used = _series_vars(out["growth"])
    assert len(set(used)) == len(_KNOWN_KINDS), (
        f"{len(set(used))} distinct colours for {len(_KNOWN_KINDS)} kinds: "
        f"{sorted(set(used))}"
    )


def test_a_run_killed_mid_write_is_not_reported_as_having_written_nothing(tmp_path) -> None:
    """A run's only timestamp comes from its `:TurnOutcome`, so a turn killed
    mid-write has nodes but no date. Counting it as "left no trace" contradicted
    the banner's own body and double-reported a run the ledger already flags."""
    killed = _run(run_id="killed", written_at=None, hops=3,
                  added={"Hop": 3}, total_added=3)
    dated = _run(run_id="ok")
    out = _render(_payload(
        runs=[dated, killed],
        schedule={"available": True, "runs_today": 2, "runs_wrote_today": 2,
                  "daily_cap": 3, "next_eligible_at": None, "cooldown_sec": 1.0,
                  "last_investigation_at": None},
    ), tmp_path)
    assert "wrote nothing at all" not in out["banners"]
    assert "died before writing an outcome" in out["runs"]


def test_a_run_that_wrote_nothing_at_all_is_surfaced_from_the_counter_gap(tmp_path) -> None:
    """Its only evidence: Redis counted it, the graph has no node carrying its
    id, so it appears in no panel. That is a banner, not a silence."""
    out = _render(_payload(
        runs=[_run()],
        schedule={"available": True, "runs_today": 3, "runs_wrote_today": 1,
                  "daily_cap": 3, "next_eligible_at": None, "cooldown_sec": 1.0,
                  "last_investigation_at": None},
    ), tmp_path)
    assert "wrote nothing at all" in out["banners"]
    assert "2 runs today" in out["banners"]


def test_the_banner_is_silent_when_the_server_could_not_count(tmp_path) -> None:
    """`runs_wrote_today` is None when Redis or the zone is unavailable.
    Treating that as 0 would fire the alarm on every page load during an
    outage of something unrelated to Orion."""
    out = _render(_payload(
        runs=[_run()],
        schedule={"available": True, "runs_today": 3, "runs_wrote_today": None,
                  "daily_cap": 3, "next_eligible_at": None,
                  "cooldown_sec": 1.0, "last_investigation_at": None},
    ), tmp_path)
    assert "wrote nothing at all" not in out["banners"]


def test_an_unreadable_graph_and_an_unconfigured_one_read_differently(tmp_path) -> None:
    broken = _render(_payload(available=False, reason="ConnectionError: nope"), tmp_path)
    off = _render(_payload(available=False, reason="graph_not_configured"), tmp_path)
    assert "could not be read" in broken["banners"]
    assert "outage" in broken["banners"]
    assert "switched off" in off["banners"]
    assert "outage" not in off["banners"]


def test_an_empty_pool_and_a_dead_pool_read_differently(tmp_path) -> None:
    empty = _render(_payload(live_total=0, closed_total=0), tmp_path)
    dead = _render(_payload(live_total=0, closed_total=4, pool_is_dead=True), tmp_path)
    assert "has not written a prior yet" in empty["banners"]
    assert "No priors are still in play" in dead["banners"]
    assert "inherits nothing to test" in dead["banners"]


def test_an_unrecorded_trajectory_says_not_recorded_not_never_moved(tmp_path) -> None:
    prior = {"prior_id": "p1", "claim": "a claim", "confidence": 0.85,
             "status": "open", "times_tested": 0, "is_closed": False,
             "created_by_run": "r1", "last_run_id": "", "formed_from": "",
             "last_tested_at": "", "why": "",
             "trajectory": [{"run_id": "r1", "confidence": 0.85,
                             "status": "open", "recorded": False}]}
    out = _render(_payload(priors=[prior], history_recorded=False), tmp_path)
    assert "no revision recorded yet" in out["trajectories"]
    assert "No movement has been recorded yet" in out["traj-note"]


def test_confidence_going_down_is_drawn_and_labelled(tmp_path) -> None:
    """The loop's headline acceptance check has to be visible on the page, not
    merely representable in the payload."""
    prior = {"prior_id": "p1", "claim": "a claim", "confidence": 0.40,
             "status": "revised", "times_tested": 1, "is_closed": False,
             "created_by_run": "r1", "last_run_id": "r2", "formed_from": "",
             "last_tested_at": "", "why": "",
             "trajectory": [{"run_id": "", "confidence": 0.90, "status": "open",
                             "recorded": True},
                            {"run_id": "r2", "confidence": 0.40,
                             "status": "revised", "recorded": False}]}
    out = _render(_payload(priors=[prior], history_recorded=True), tmp_path)
    assert "<svg" in out["trajectories"], "a two-point history must draw a line"
    assert "-0.50 overall" in out["trajectories"]


def test_a_closed_prior_is_in_the_table_but_not_the_trajectory_panels(tmp_path) -> None:
    closed = {"prior_id": "dead", "claim": "a refuted claim", "confidence": 0.1,
              "status": "refuted", "times_tested": 3, "is_closed": True,
              "created_by_run": "r1", "last_run_id": "r2", "formed_from": "",
              "last_tested_at": "", "why": "", "trajectory": []}
    out = _render(_payload(priors=[closed], live_total=0, closed_total=1,
                           pool_is_dead=True), tmp_path)
    assert "a refuted claim" in out["priors-table"]
    assert "a refuted claim" not in out["trajectories"]


def test_orion_prose_is_escaped_rather_than_rendered_as_markup(tmp_path) -> None:
    """Orion writes these strings by hand inside a turn. They are data."""
    run = _run(reach_out_why='<img src=x onerror="alert(1)">',
               continue_note="a & b < c")
    out = _render(_payload(runs=[run]), tmp_path)
    assert "<img" not in out["runs"]
    assert "&lt;img" in out["runs"]
    assert "a &amp; b &lt; c" in out["runs"]


# --- the Hub tab -----------------------------------------------------------
#
# CLAUDE.md section 9: a rendered template, a linked asset, and the changed
# interaction are three separate things and all three have to be checked. The
# page existed at /curiosity for a whole review cycle before anything in the Hub
# UI pointed at it.

_HUB = Path(__file__).resolve().parents[1] / "services" / "orion-hub"


def test_the_tab_button_and_its_panel_both_exist() -> None:
    index = (_HUB / "templates" / "index.html").read_text(encoding="utf-8")
    assert 'id="curiosityAtlasTabButton"' in index
    assert 'href="#curiosity-atlas"' in index
    assert 'id="curiosity-atlas" data-panel="curiosity-atlas"' in index
    assert 'src="/curiosity"' in index, "the iframe must point at the real route"


def test_every_hub_side_wire_the_tab_needs_is_present() -> None:
    """A tab is not one edit. Miss the hash branch and a deep link silently
    lands on #hub; miss the styling and the button never looks selected."""
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
    """Two copies of a URL is how one of them goes stale."""
    index = (_HUB / "templates" / "index.html").read_text(encoding="utf-8")
    block = index.split('id="curiosity-atlas"', 1)[1].split("</section>", 1)[0]
    assert block.count('"/curiosity"') == 2, block.count('"/curiosity"')
