## Summary

- Orion's Day note prompt now plainly asks Orion to copy the bracket from its closed-day chronology (`[arc:<id>]` for one arc, `[day:<date>]` for a whole-day count) into any sentence about that recorded day, with an example, and repeats the ask once after the record (the record is up to ~70k tokens, so the original instruction sat far from where Orion writes).
- The old line "you do not need to cite the bracketed references" now scopes itself to everything outside the chronology, so the prompt no longer tells Orion two opposite things.
- Closed-day block attention lines no longer put a group total next to one arc's id. Multi-arc subjects read `<total> over N arcs [day:<date>]; longest <arc dwell>, <arc returns> returns [arc:<id>]`; a one-arc subject states only that arc's numbers.
- The day record (`records["day:<date>"]`) gains `attention_groups` (arcs, cumulative dwell per subject shown) so the `[day:]` cite resolves to the number it sits beside.
- No gate change: uncited sentences stay kept and labelled UNVERIFIED, never blocked or retried; no word filters.

## Outcome moved

Live letters for 10-11 and 10-12 were shown 20 and 17 arcs and cited 0 (`sources->'arc_citations'->'note'`: `UNVERIFIED`, cited 0, uncited 1). The prompt now asks for the ids where Orion writes. The block's own truth eval on the 10-09 fixture goes from 2 mismatches / 14 matches to 0 mismatches / 16 matches.

## Current architecture

`orion/orion_day/budget.py:build_llm_view` puts the closed-day block (`orion/temporal_self/day_block.py:render_closed_day_block`) into the digest; `orion/cognition/prompts/orion_day_note_v1.j2` renders it for cortex-exec; `services/orion-durable-runs/app/orion_day_graph.py:gate_arc_citations` checks the reply with `orion/temporal_self/citations.py`. The prompt had a one-line "exception" asking for `[arc:<id>]`, right after a line saying citations were not needed.

## Architecture touched

Prompt text and the closed-day block renderer only. No bus, schema-model, env or service-code change.

## Files changed

- `orion/cognition/prompts/orion_day_note_v1.j2`: citation instruction + trailing reminder (both only when the digest holds `[arc:`).
- `orion/temporal_self/day_block.py`: attention-line wording; `attention_groups` in the day record.
- `orion/temporal_self/tests/test_day_block.py`: exact-string fixture updated; new test that each number matches what it cites.
- `orion/temporal_self/tests/test_citation_truth.py`: the pinned-mismatch test becomes "no mismatches, Harness line matches".
- `services/orion-durable-runs/tests/test_orion_day_arc_citations.py`: prompt contains the instruction before and after the record; no block means no ask; a reply citing ids passes the gate as `verified`; the live uncited shape stays `UNVERIFIED` and is kept verbatim.

## Schema / bus / API changes

- Added: `attention_groups` key inside the `day:<date>` compact record (a free-form `dict[str, Any]`; additive).
- Removed / Renamed: none.
- Behavior changed: attention-line wording in the closed-day block.
- Compatibility notes: no consumer parses the attention-line text; stored letters keep their old block text.

## Env/config changes

- Added / Removed / Renamed keys: none. `.env_example` not touched; no sync needed.

## Tests run

```text
pytest -q orion/temporal_self/tests orion/orion_day/tests \
          services/orion-durable-runs/tests/test_orion_day_arc_citations.py \
          services/orion-durable-runs/tests/test_orion_day_graph.py           -> 272 passed
pytest -q services/orion-hub/tests/test_orion_day_temporal_day.py             -> 6 passed
```

## Evals run

```text
python orion/orion_day/evals/run_orion_day_eval.py              -> all PASS
python orion/temporal_self/evals/run_arc_precision_eval.py      -> passed: true
python orion/temporal_self/evals/run_day_citation_truth_eval.py -> W42: match 3, mismatch 0, undecidable 11
```

Letters already written cannot be re-scored without regenerating them; the next letter after deploy is the live check (below).

## Docker/build/smoke checks

Not run; this PR is not deployed (Juniper's instruction).

## Review findings fixed

Code review ran in a subagent against `origin/main...fix/orion-day-letter-cites-arcs`. No must findings.

- Finding (should): the prompt offered `[day:]` as a valid citation, but the gate counts only `[arc:]`. A sentence naming an attention subject with only `[day:]` stays UNVERIFIED.
  - Fix: the prompt now says `[day:<date>]` alone covers only a whole-day count, and that a sentence naming an attention subject or a time also needs that fact's `[arc:<id>]`. The gate is unchanged.
  - Evidence: `test_a_day_bracket_alone_does_not_verify_an_attention_sentence`, plus updated prompt assertions.
- Finding (should): "each of its lines ends with the record" is false, because some lines carry several brackets in the middle.
  - Fix: the wording is now "next to each fact in it sits the record it came from" and "copy the [arc:<id>] next to that fact".
- Finding (should): `attention_groups` was filtered by label prefix, so a same-label group could leak after the cap.
  - Fix: each group is kept only if its exact rendered line survives in the block text.
  - Evidence: `test_attention_groups_follow_the_lines_that_survive_the_cap`, `test_same_label_groups_keep_only_the_one_whose_line_survives`.
- Finding (nit): "(next point)" rendered even when there is no chronology block.
  - Fix: there are now separate no-block and with-block wordings.
  - Evidence: `test_note_prompt_without_a_block_asks_for_no_ids`.
- Finding (nit): nothing reads `attention_groups` yet, and the truth checker does not check `[day:]` totals.
  - Not fixed. It is stored so the `[day:]` cite has a record behind it, but no check verifies it. Noted under Risks.

## Restart required

Rebuild the services that bake `orion/` into their image: `orion-cortex-exec` (renders the prompt) and `orion-hub` (gathers the block). `orion-durable-runs` code is unchanged.

```bash
scripts/safe_docker_build.sh orion-cortex-exec up -d --build && scripts/safe_docker_build.sh orion-hub up -d --build
```

Post-deploy live check (the next letter written after the rebuild):

```sql
SELECT letter_date,
       sources->'arc_citations'->'note'->>'label'                      AS label,
       jsonb_array_length(sources->'arc_citations'->'note'->'cited_valid') AS cited,
       sources->'arc_citations'->'note'->>'shown'                      AS shown,
       sources->'arc_citations'->'note'->>'uncited_day_sentences'      AS uncited,
       sources->'arc_citations'->'note'->'uncited_examples'            AS examples
FROM orion_day_letter WHERE letter_date >= DATE '2026-10-13' ORDER BY letter_date;
```

Success: `cited > 0`. Baseline (10-11, 10-12): cited 0, shown 20/17, uncited 1, `UNVERIFIED`.

## Risks / concerns

- Severity: low. Concern: `attention_groups` is stored but no check reads it yet, and `citation_truth.py` checks only arc rows. A `[day:]` group total is therefore recorded but not machine-checked. Mitigation: it is additive and cheap to wire into the truth eval later.

- Severity: low. Concern: the prompt asks, it cannot force; Orion may still cite nothing. Mitigation: the gate already labels that UNVERIFIED, and the SQL above measures it.
- Severity: low. Concern: attention lines are ~40 chars longer for multi-arc subjects. The 291-arc 10-09 fixture renders 2,079 of 2,400 chars with nothing dropped; on a busier day the cap drops lowest-priority lines and says so.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2631
