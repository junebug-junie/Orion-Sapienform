"""Eval: every reverie theme in a real week reaches the dream as a readable title.

Fixture: the 11 distinct reverie themes (compaction requests + resonance alerts) of the
7 days before 2026-10-13, each with the latest attention_salience_trace description for
its theme_key -- what cycle_store's title join returns live. Before this change the dream
read every one of them as a bare `open-loop-<hex>` id (e.g. 4 of 12 items in the
2026-10-09 06:33 sleep)."""
from __future__ import annotations

import json
import re
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "reverie_loop_titles_2026-10-13.json"
BARE_ID = re.compile(r"open-loop-[0-9a-f]{6,}")


def test_a_real_week_of_reverie_themes_reads_as_titles():
    from app.replay import candidate_from_compaction_request, candidate_from_resonance

    titles = json.loads(FIXTURE.read_text())["titles"]
    assert len(titles) == 11
    texts = []
    for i, (loop_id, title) in enumerate(sorted(titles.items())):
        texts.append(candidate_from_compaction_request(
            {"request_id": f"r{i}", "theme": loop_id, "reason": "reverie_chain_max_steps", "title": title}).text)
        texts.append(candidate_from_resonance({"alert_id": f"a{i}", "theme_key": loop_id, "title": title}).text)
    print("\n".join(sorted(set(texts))[:6]))
    assert not [t for t in texts if BARE_ID.search(t)]
    assert any("Harness closure prediction error" in t for t in texts)
