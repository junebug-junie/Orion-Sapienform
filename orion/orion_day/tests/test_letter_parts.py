"""Letter parts: stable numbering, citation resolution, and the claim check."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from orion.orion_day.budget import build_llm_view, extract_refs, material_ref_records, material_refs
from orion.orion_day.gather import gather_orion_day
from orion.orion_day.letter_parts import (
    claim_check,
    extract_claim_tokens,
    find_part,
    resolve_citations,
    split_carry,
    split_note,
)
from orion.orion_day.tests import fixtures as fx

NOTE = """# September 29

The dream organ woke at 06:35:27Z after 260.6 hours of silence.

---

I re-tested `self:pool_heavy_composition_reverts` and it held at 0.72; see #2557.

```
a code block

with a blank line inside
```

- a list as its own paragraph
- second line
"""

CARRY = """Threads for tomorrow:

- **First thread.** Cites [curiosity:RUN] and wraps
  onto an indented line.
- **Second thread.** Cites [reading_journal:nope-not-here].

  Still the second thread after a blank line.
1. **Third thread**, numbered.
lazy continuation, as markdown reads it.

Trailing words after the list.
"""


def _material():
    return asyncio.run(gather_orion_day(fx.FakeConn(), fx.LETTER_DATE,
                                        now=datetime(2026, 9, 30, 14, 30, tzinfo=timezone.utc)))


def test_note_numbers_prose_blocks_and_skips_headings_and_rules():
    parts = split_note(NOTE)
    paragraphs = [p for p in parts if p.kind == "paragraph"]
    assert [p.index for p in paragraphs] == [1, 2, 3, 4]
    assert paragraphs[0].text.startswith("The dream organ woke")
    assert "with a blank line inside" in paragraphs[2].text  # a fence never splits
    assert paragraphs[3].text.startswith("- a list")
    assert [p.text for p in parts if p.kind == "other"] == ["# September 29", "---"]
    assert paragraphs[1].ref("2026-09-29") == "2026-09-29 ¶2"


def test_note_split_loses_no_text():
    joined = "".join(p.text for p in split_note(NOTE))
    assert "".join(NOTE.split()) == "".join(joined.split())


def test_carry_items_keep_their_continuations():
    parts = split_carry(CARRY)
    items = [p for p in parts if p.kind == "carry"]
    assert [p.index for p in items] == [1, 2, 3]
    assert "onto an indented line." in items[0].text
    assert "Still the second thread" in items[1].text
    assert items[2].text.startswith("1. **Third thread**") and "lazy continuation" in items[2].text
    assert [p.text for p in parts if p.kind == "other"] == ["Threads for tomorrow:", "Trailing words after the list."]
    assert find_part(parts, "carry", 2) is items[1]
    assert find_part(parts, "carry", 9) is None
    assert items[0].ref("2026-09-29") == "2026-09-29 carry 1"


def test_numbering_is_stable_across_calls():
    assert split_note(NOTE) == split_note(NOTE)
    assert split_carry(CARRY) == split_carry(CARRY)


def test_every_digest_ref_maps_to_a_record():
    material = _material()
    records = material_ref_records(material)
    assert set(records) == material_refs(material)
    view = build_llm_view(material)
    assert view.included_refs and all(ref in records for ref in view.included_refs)
    assert extract_refs(view.digest_md) == view.included_refs


def test_citations_resolve_or_report_unresolved():
    material = _material()
    run_id = material.curiosity_runs[0].run_id
    item = split_carry(CARRY.replace("RUN", run_id))[1]
    cites = resolve_citations(item.text, material)
    assert [c.ref for c in cites] == [f"curiosity:{run_id}"]
    assert cites[0].resolved and cites[0].record["run_id"] == run_id
    bad = resolve_citations(split_carry(CARRY)[2].text, material)
    assert [(c.ref, c.resolved) for c in bad] == [("reading_journal:nope-not-here", False)]


def test_claim_tokens_are_the_checkable_ones():
    tokens = extract_claim_tokens(
        'At 2026-10-09T06:35:27Z and 01:02:09Z, 260.6 hours, 4,059 rows, 122 rows, 7 of 9, '
        'on 10-07, `ledger_x`, prior self:pool_heavy_reverts, PR #2557, "a quote long enough here".'
    )
    assert tokens == [
        ("2026-10-09T06:35:27Z", "timestamp"),
        ("01:02:09Z", "timestamp"),
        ("260.6", "decimal"),
        ("4,059", "integer"),
        ("122", "integer"),
        ("ledger_x", "code"),
        ("self:pool_heavy_reverts", "prior_id"),
        ("2557", "pr"),
        ("a quote long enough here", "quote"),
    ]


def test_claim_check_finds_supported_tokens_and_flags_planted_ones():
    material = _material()
    run = material.curiosity_runs[0]
    planted = run.model_copy(update={"journal_body": "Silence of 260.6 hours, last row at 2026-09-29T06:35:27.123+00:00."})
    material = material.model_copy(update={"curiosity_runs": [planted, *material.curiosity_runs[1:]]})
    results = {t.token: t for t in claim_check("After 260.6 hours, at 06:35:27Z, then 999.4 hours.", material)}
    assert results["260.6"].found_in == (f"curiosity:{run.run_id}",)
    assert results["06:35:27Z"].found
    assert not results["999.4"].found


def test_number_match_respects_digit_boundaries():
    material = _material()
    run = material.curiosity_runs[0]
    planted = run.model_copy(update={"journal_body": "values 1260.65 and 4059"})
    material = material.model_copy(update={"curiosity_runs": [planted]})
    results = {t.token: t for t in claim_check("260.6 and 4,059", material)}
    assert not results["260.6"].found  # inside 1260.65, not the same number
    assert results["4,059"].found      # thousands separator normalised
