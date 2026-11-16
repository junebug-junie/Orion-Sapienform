## Summary

- Reading choice options are now only links to something else to read (#3443). Links that are the read page itself (its arXiv id or DOI in the URL, or its own URL inside a share/lookup query), licence deeds, vocabulary namespace URIs, and links written only inside real code are left out. The rules use URL structure, not word lists.
- Option text is the link's anchor text, else its own sentence (or the words between it and the next URL on a URL row), with URLs and markdown taken out. When two or more links share one label ("Full URL:"), each takes the title line of its list item.
- One-time relabel of choice-built tendencies (#3444). A pure planner rebuilds each label from the node's own stored choices with the current label maker. A durable-runs admin endpoint writes the result under the reducer's tick lock. A host script does the read-only dry run and the section 14 snapshot, then calls the endpoint. The apply has NOT been run.
- `ChoiceOptionV1.detail` (Juniper approved 2026-11-16): Prime fills it with a curiosity-run promise's own sentence. Only the Hub Personality tab's tendency card shows it (in "About"). It is never put in option text, a label, an embedding, or the Prime `tending:` line (S-1.8).

## Outcome moved

- The live 11-15 read of `arxiv.org/pdf/2611.17316` offered 6 options. All 6 were the paper itself, its licence, or XMP namespaces. With this patch it offers none, and no choice is recorded.
- Replay over the last 7 days (live, read-only, same 286-pass window for main and this branch): decisions go from 16.6 to 15.6 a day (about 133 to 125 over 8 days), and choice rows from 48.2 to 47.0 a day. Passes with an empty list go from 25 to 34. Of the options the old code showed (304 passes of parent text, before the already-read filter), 301 are dropped: 242 same document, 46 licence, 11 namespace, 2 code only.
- Relabel dry run against live (read-only): 155 non-retired choice nodes, 125 labels change, 0 anomalies. Labels with an old category lead or a "(+k more)" tail: 125 before, 0 after.

## Current architecture

- `candidate_links` listed every link in Stage 1's retained text, minus fetched pages, assets and top-level site navigation. The label was the whole line with the URL cut out.
- A tendency label is set only when a choice folds into it (`_Tick.reinforce`). Label-rule fixes reached a node only on its next choice.
- The reducer's tick lock is an in-process `asyncio.Lock` in orion-durable-runs. A tick writes the whole node row back after it folds a choice.

## Architecture touched

- `orion/world_pulse_read/next_read_choice.py` (Hub Stage 2 producer of reading choices).
- `orion/personality/relabel.py` (new, pure), orion-durable-runs personality store/driver/route.
- `orion/schemas/personality.py` (`ChoiceOptionV1.detail`), `orion/prime/choice.py` (producer), Hub Personality tab JS (consumer).

## Files changed

- `orion/world_pulse_read/next_read_choice.py`: `classified_links`, `non_content_reason`, `document_ids`, `_code_free`, sentence/segment labels, shared-label title fallback.
- `orion/world_pulse_read/tests/test_next_read_choice.py`, `services/orion-hub/evals/test_reading_choice_eval.py`: live 11-15 shape, rule tests, label tests.
- `services/orion-hub/evals/run_reading_choice_replay.py`: prints drop reasons with examples, and the options as shown.
- `orion/personality/relabel.py`, `orion/personality/tests/test_relabel.py`: the planner and its tests.
- `services/orion-durable-runs/app/{main,personality_driver,personality_store}.py`: `POST /personality/relabel`, `relabel_choice_nodes`, `_option` passes `detail` through. `option_texts` now uses the shared `choice_sides`.
- `services/orion-durable-runs/tests/test_personality_{tab_postgres,tab_routes,loop}.py`: Postgres relabel and detail tests, route tests, a no-leak test.
- `scripts/personality_relabel_choice_nodes.py`, `tests/test_personality_relabel_choice_nodes.py`: the host script and its tests.
- `orion/schemas/personality.py`, `orion/prime/choice.py`, `orion/prime/tests/test_prime_choice.py`: `detail` and its never-in-text/label test.
- `services/orion-hub/static/js/personality.{js,test.js}`: About block shows "The promise, in its own words".
- `services/orion-durable-runs/README.md`, `.github/workflows/personality-tests.yml`.

## Schema / bus / API changes

- Added: `ChoiceOptionV1.detail: Optional[str]` (max 600 characters). Added `POST /personality/relabel` on durable-runs.
- Removed / Renamed: none.
- Behavior changed: the reading choice options and their texts (#3443).
- Compatibility notes: `ChoiceOptionV1` is `extra="forbid"`. Durable-runs must deploy before orion-mind, or durable-runs refuses Prime's choices that carry `detail` as invalid. Old rows have no `detail` and read as before.

## Env/config changes

- Added keys / Removed keys / Renamed keys: none.
- `.env_example` updated: no. Local `.env` sync: not needed.

## Tests run

```text
orion/personality orion/prime orion/world_pulse_read orion/schemas: 751 passed
services/orion-hub evals/test_reading_choice_eval.py + tests/test_world_pulse_read_stage2.py: 69 passed
services/orion-durable-runs tests (with a throwaway pgvector:pg16 for ORION_ADMISSION_TEST_DSN): all passed
tests/test_personality_relabel_choice_nodes.py tests/test_personality_schemas_registered.py: 5 passed
node --test services/orion-hub/static/js/personality.test.js: 16 passed
```

## Evals run

```text
services/orion-hub/evals/run_reading_choice_replay.py --days 7 (live, read-only)
  window: 286 Stage 2 passes (run 2026-11-16 ~04:20Z, main and branch back to back)
  before (main): decisions=16.6/day rows=48.2/day links_shown mean=7.7 zero=25
  after:         decisions=15.6/day rows=47.0/day links_shown mean=7.2 zero=34
  links left out (parent lists, after already-read): same_document=412 license=57 schema_namespace=3
scripts/personality_relabel_choice_nodes.py (dry run, live, read-only)
  155 nodes, 125 label changes, 0 anomalies, stale leads 125 -> 0
  files: /tmp/personality-relabel/20261116T042500Z-dry/{snapshot.jsonl,progress.log,before_after.csv,report.md}
```

Before/after option text examples (old code -> new code, same link):

- `alphaXiv: \` \` (alphaxiv.org)` -> `alphaXiv (alphaxiv.org)`
- `Full URL (wsoctv.com)` (x4) -> `How to use the WSOC-TV weather app for forecasts in the Carolinas (wsoctv.com)`, ...
- `“ALSO READ: Violence prevention program in west Charlotte creates positive results” (wsoctv.com)` -> `ALSO READ: Violence prevention program in west Charlotte creates positive results (wsoctv.com)`
- `Wireless Emergency Alerts → ; Weather-Ready Nation → ; Brochures → ; ...` -> `Wireless Emergency Alerts (weather.gov)`
- `10 48550 arXiv 2611 17316 (doi.org)` -> dropped (same document). A different paper's bare DOI link now reads `arXiv 2611.07965 (doi.org)`.

Relabel examples (old -> new):

- `A read, "The Definitive Fine-Tuning Guide, 1 of 1` -> `The Definitive Fine-Tuning Guide for 3D, 1 of 1`
- `a thread from my letter over a promise I made and a thread from my letter, 3 of 3` -> `The walkway render break is unresolved over Curiosity run promise and Chat memory has never been promoted, 3 of 3`
- `A daydream: a person driving, 1 of 1` -> `a person driving a car, 1 of 1`
- `induction ledger dangling over ring is doorway projection (+14 more), 6 of 6` -> `induction ledger dangling, 6 of 6`

## Docker/build/smoke checks

```text
No image built. Live smoke UNVERIFIED until deploy (see Risks).
```

## Review findings fixed

A code review subagent ran on the branch. Fixed:

- Finding (blocker): every producer dumped `detail: null` on every option. A durable-runs built before the field would then refuse every reading, outreach and Prime choice, not only Prime's promises.
  - Fix: `ChoiceOptionV1` leaves `detail` out of the dump when it is None.
  - Evidence: `tests/test_personality_schemas_registered.py::test_choice_option_without_detail_dumps_like_the_pre_detail_model` validates the dump against a frozen copy of the old model.
- Finding: a sibling paper whose DOI started with the page's DOI was dropped as the same document.
  - Fix: the DOI now needs an end boundary. A version or format tail (`v1.full.pdf`) still counts as the same work.
  - Evidence: `test_review_fixes_sibling_doi_stray_backtick_and_document_pages_on_mixed_hosts`.
- Finding: a stray backtick used as an apostrophe ("Don`t") swallowed a prose link as code.
  - Fix: inline code is judged per line, only where the line's backticks pair up.
  - Evidence: the same test.
- Finding: `--apply` approved its own fresh plan, not the reviewed dry run.
  - Fix: `--apply` now needs `--from <dry run>` and sends that run's CSV pairs.
  - Evidence: `tests/test_personality_relabel_choice_nodes.py`.
- Finding: no test proved that the replay matches the real reducer.
  - Fix: a differential test folds choices with `graph.fold` over 4 ticks (swaps, a category-only choice, an echo, an opposed choice). It then asserts that the planner gives back every label and bar unchanged.
  - Evidence: `orion/personality/tests/test_relabel.py::test_relabel_of_a_graph_folded_by_the_real_reducer_changes_nothing`.
- Finding: no test proved that the tick lock is held.
  - Fix: a test shows a tick blocks while the relabel holds the lock, and a route test checks that `expected={}` returns 400.
  - Evidence: `test_relabel_holds_the_tick_lock`, `test_relabel_route_dry_run_by_default_and_apply_needs_expected`.
- Nits fixed:
  - `COLLATE "C"` on the evidence id order (it matches Python's sort).
  - The w3.org and purl.org namespace rules now drop only namespace-shaped URIs (`w3.org/2020/.../report.html` and `purl.org/net/x` stay).
  - The licence rule is limited to deed registries, versioned `licenses/<name>/<n.n>` deeds, and `LICENSE` files.
  - After an apply, the report counts only the written ids.
  - Each promise sentence is tagged Chose or Passed over.
  - The single-process limit of the lock is noted in the docstring.

## Restart required

Deploy order: durable-runs first (the consumer of `detail`, and the relabel endpoint), then orion-mind (the Prime producer), then Hub (reading options + tab JS). From the clean primary main checkout after merge:

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-mind up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Then, only with Juniper's approval (production write):

```bash
PYTHONPATH=. .venv/bin/python scripts/personality_relabel_choice_nodes.py --apply --from /tmp/personality-relabel/20261116T042500Z-dry
```

`--apply` writes only the pairs in the reviewed run's `before_after.csv`. If the run is old, do a fresh dry run, review it, and pass that directory instead.

## Risks / concerns

- Severity: medium. Concern: the relabel apply is a production write and has not run. #3444 stays open (Refs, not Fixes) until it runs and the issue's check query shows no stale leads. Mitigation: the dry run is reviewable, the apply writes only the reviewed pairs, and the snapshot is the rollback source.
- Severity: low. Concern: if orion-mind deploys before durable-runs, Prime's choices with `detail` are refused as invalid and lost until durable-runs is rebuilt. Mitigation: the deploy order above.
- Severity: low. Concern: Orion's own dig tools can still read `personality_choice` rows, `detail` included. That is Orion's own data, not turn context. Nothing in the turn path reads the field.
- Severity: low. Concern: site footers and navigation deeper than one path level still show (for example pbs.org author pages). This patch only drops links with a structural reason. A footer rule would need page-region evidence that the digests do not keep.
- Live UNVERIFIED: the first reading choice after deploy (`select alternatives from personality_choice where lane='reading' order by occurred_at desc limit 5`), and the first tendency card that shows a promise sentence.
- Issues: Fixes #3443. Refs #3444 (closes after the apply).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3504
