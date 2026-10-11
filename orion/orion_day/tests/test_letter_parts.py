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


def test_a_rule_inside_the_carry_forward_takes_no_number():
    items = [p for p in split_carry("- a\n* * *\n- b\n- - -\n- c") if p.kind == "carry"]
    assert [p.text for p in items] == ["- a", "- b", "- c"]


def test_only_the_opening_fence_marker_closes_a_fence():
    paragraphs = [p for p in split_note("```\n~~~\n\nx\n```\n\nafter") if p.kind == "paragraph"]
    assert [p.text for p in paragraphs] == ["```\n~~~\n\nx\n```", "after"]


def test_a_bare_date_is_not_an_integer_claim():
    assert extract_claim_tokens("On 2026-10-09 the 211-node graph") == [("211", "integer")]


def test_quotes_with_json_escaped_characters_are_found():
    material = _material()
    run = material.curiosity_runs[0]
    planted = run.model_copy(update={"journal_body": 'I ran `foo("bar")` and wrote "a path C:\\tmp\\x was here".'})
    material = material.model_copy(update={"curiosity_runs": [planted]})
    results = {t.token: t for t in claim_check('`foo("bar")` and "a path C:\\tmp\\x was here"', material)}
    assert results['foo("bar")'].found
    assert results["a path C:\\tmp\\x was here"].found


def test_a_decimal_inside_a_version_string_is_not_support():
    material = _material()
    planted = material.curiosity_runs[0].model_copy(update={"journal_body": "running v0.27.1"})
    material = material.model_copy(update={"curiosity_runs": [planted]})
    assert not claim_check("a floor of 0.27", material)[0].found


# --- reread helpers (orion-introspect `orion_day`) -------------------------------------------

from orion.orion_day.letter_parts import (  # noqa: E402
    SECTION_PREFIXES,
    parse_ref,
    record_excerpt,
    record_text,
    section_records,
)
from orion.schemas.introspect import OrionDaySection  # noqa: E402


def test_parse_ref_inverts_letter_part_ref():
    for part in split_note(NOTE) + split_carry(CARRY):
        ref = part.ref("2026-09-29")
        if ref is not None:
            assert parse_ref(ref) == ("2026-09-29", part.kind, part.index)
    for bad in ("2026-09-29 ¶", "2026-09-29 note 3", "¶3", "2026-09-29 carry five", ""):
        assert parse_ref(bad) is None


def test_section_names_match_the_tool_contract_and_cover_every_record_but_themes():
    assert set(SECTION_PREFIXES) == set(OrionDaySection.__args__)
    material = _material()
    covered = {ref for name in SECTION_PREFIXES for ref, _ in section_records(material, name)}
    rest = set(material_ref_records(material)) - covered
    assert rest and all(ref.startswith("reverie_theme:") for ref in rest)
    curiosity = [ref for ref, _ in section_records(material, "curiosity")]
    assert curiosity[0] == f"curiosity:{material.curiosity_runs[0].run_id}"
    assert any(ref.startswith("curiosity_failed:") for ref in curiosity)
    assert all(ref.startswith("reverie:") or ref.startswith("visual_reverie:")
               for ref, _ in section_records(material, "reveries"))


def test_record_excerpt_is_title_then_body_and_capped():
    material = _material()
    records = material_ref_records(material)
    run = records[f"curiosity:{material.curiosity_runs[0].run_id}"]
    excerpt = record_excerpt(run, 60)
    assert excerpt.startswith("Curiosity: I tested the hop written_at prior.")
    assert len(excerpt) <= 60 and excerpt.endswith("…")
    dream = next(r for ref, r in records.items() if ref.startswith("dream:"))
    assert record_text(dream) == "A library of recent work: I walked between stacks of PRs."
    news = next(r for ref, r in records.items() if ref.startswith("world_pulse_digest:"))
    assert record_text(news) == "Daily World Pulse: 12 tracked items. Items: Fuel standards rolled back"
    failed = next(r for ref, r in records.items() if ref.startswith("curiosity_failed:"))
    assert record_text(failed) == "HoldLost: x"
    assert record_excerpt(None) == "" and record_excerpt({}) == ""
    assert record_excerpt({"body": "short"}, 300) == "short"
