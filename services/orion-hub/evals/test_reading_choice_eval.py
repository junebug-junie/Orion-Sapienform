"""Offline eval: the reading choice point on a frozen live shape (2026-11-03 PBS -> PolitiFact hop).

The live replay is run_reading_choice_replay.py (read-only, per-day estimate). This file pins the
verdicts the replay depends on, so the estimate and the producer cannot drift apart.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

from orion.world_pulse_read.fetch_provenance import hop_allowlist
from orion.world_pulse_read.next_read_choice import (
    next_read_options,
    parse_next_read_section,
    reading_choices,
    render_next_read_section,
)



def _summarize():
    path = Path(__file__).with_name("run_reading_choice_replay.py")
    spec = importlib.util.spec_from_file_location("run_reading_choice_replay", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.summarize


AT = datetime(2026, 11, 3, 19, 30, tzinfo=timezone.utc)
PBS = "https://www.pbs.org/newshour/tag/politifact"
# The shape of the live PBS tag page digest: a list of fact checks with their links.
PBS_DIGEST = (
    "PolitiFact items on PBS NewsHour:\n"
    "1. [Fact-checking claims about tariffs and grocery prices](https://www.pbs.org/newshour/politics/tariffs-groceries)\n"
    "2. [Chris Murphy on stroke risk in younger adults](https://politifact.com/article/2026/nov/02/chris-murphy-stroke-younger-adults-risk-factors/)\n"
    "3. [What the data says about crime trends](https://www.pbs.org/newshour/nation/crime-trends)\n"
)


def test_orion_picking_the_politifact_link_records_rank_two_over_the_other_two():
    shown = next_read_options([(PBS, PBS_DIGEST)], exclude_urls=[PBS])
    bound = parse_next_read_section("handoff\n" + render_next_read_section(shown))
    rows, reason = reading_choices(
        seed_id="reading:19c74f98", shown=bound,
        picked_urls=["https://politifact.com/article/2026/nov/02/chris-murphy-stroke-younger-adults-risk-factors/"],
        allowlist=hop_allowlist([(PBS, PBS_DIGEST)]), occurred_at=AT)
    assert reason == "picked"
    assert rows[0].chosen.text == "Chris Murphy on stroke risk in younger adults (politifact.com)"
    assert rows[0].chosen_market_rank == 2
    assert [a.text for a in rows[0].alternatives] == [
        "Fact-checking claims about tariffs and grocery prices (pbs.org)",
        "What the data says about crime trends (pbs.org)",
    ]


def test_summary_counts_decisions_not_rows():
    table = _summarize()([("d1", "picked", 3), ("d1", "declined_all", 1), ("d1", "no_links_shown", 0),
                       ("d2", "only_unfetched_urls", 0)])
    assert table["d1"] == {"passes": 3, "picked": 1, "declined_all": 1, "no_links_shown": 1,
                           "rows": 4, "decisions": 2}
    assert table["d2"]["passes"] == 1 and "decisions" not in table["d2"]


# #3443, live 2026-11-15 09:35Z: a read of https://arxiv.org/pdf/2611.17316 showed six options, every
# one the paper itself, its licence, or an XMP namespace; Orion "picked" its own arXiv page.
ARXIV_PDF = "https://arxiv.org/pdf/2611.17316"
ARXIV_PDF_DIGEST = (
    "**3) URLs found in the metadata**\n\n"
    "- https://doi.org/10.48550/arXiv.2611.17316\n"
    "- https://arxiv.org/abs/2611.17316v1\n"
    "- http://creativecommons.org/licenses/by/4.0/\n\n"
    "(The XMP block also declares standard RDF/XML namespace URIs — e.g. "
    "`http://www.w3.org/1999/02/22-rdf-syntax-ns#`, `http://purl.org/dc/elements/1.1/`, "
    "`http://ns.adobe.com/xap/1.0/` — but these are metadata namespace declarations, not cited sources.)\n"
)


def test_live_self_doi_licence_and_namespaces_make_no_choice():
    shown = next_read_options([(ARXIV_PDF, ARXIV_PDF_DIGEST)], exclude_urls=[ARXIV_PDF])
    assert shown == []
    rows, reason = reading_choices(
        seed_id="reading:7ea0589d", shown=shown, picked_urls=["https://arxiv.org/abs/2611.17316v1"],
        allowlist=hop_allowlist([(ARXIV_PDF, ARXIV_PDF_DIGEST)]), occurred_at=AT)
    assert (rows, reason) == ([], "no_links_shown")
