## Summary

Continues PR #2647 (reading resonance, patch A). Two jobs, Juniper-approved 2026-10-14.

- **Job A, link targets.** When Stage 2 lets a read link its new concepts to concepts Orion already had, the "related by meaning" candidates are now ranked by mean-centered cosine (floor 0.35) instead of raw bge cosine (floor 0.65). Raw cosine had no calm state, so unrelated concepts got offered. The quote rule that decides whether a link is made is unchanged.
- Found while measuring: the target list is not what keeps reads islands. 26 of the last 29 reads had no usable retained page text, so Stage 2 never saw a claim block at all. And the model, told that meaning-only claims are "recorded for review, not linked", proposed zero of them.
- Deterministic co-read edges (two concepts minted by the same read -> `co_occurs_with`): skipped, with numbers below.
- **Job B, judge.** Ran the resonance judge for real on the 29 reads (agent route, pool-gated, sequential, resumable). Regenerated Juniper's labeling set with Orion's proposed connections, judge confidence, cited pieces' stored text, the pieces offered but not cited, and the rejected items. Added a single self-contained HTML page with yes/no + notes per item and an export-to-CSV button.
- Fixed a prompt/schema mismatch the first live run exposed: the judge prompt never stated the 900-character connection limit, so a real judgment was thrown away as "unparseable". The prompt now reads every limit from the schema.

## Outcome moved

Link targets, last 29 reads (window ending 2026-10-14 00:10 UTC), read-only replay:

| | exact name only (before 10-13) | raw cosine 0.65 (deployed 10-13) | centered 0.35 (this patch) |
|---|---|---|---|
| reads offered any target | 1 | 3 | 3 |
| targets offered, total | 1 | 52 | 36 |
| links possible under the quote rule (upper bound) | 1 | 1 | 1 |
| links the model actually made (LLM pass, real `plan_claims`) | - | 1 | 1 |
| claims the model proposed to meaning-only targets | - | 0 | 0 |

- Only 3 reads can be offered anything. 23 have no retained text (they predate retention, which started about 10-11). 3 more are 404 pages, which `load_retained_texts` drops by design.
- Centered ranking cuts noise the raw floor let in. On the NASA expo read, raw 0.65 offered `gpu_section_hub_page`, `message matching / capitulation` and `reading:file-path-source`. Centered 0.35 offers `NASA Inspiration Tour`, `Freedom 250 F-5 flyover team` (concepts from a sibling NASA read), `Aerospace Power Systems Laboratory` and `armstrong-flight-research-center`.
- The quote rule blocked nothing this time, because the model never proposed a meaning-only claim. By construction it would block every one: the quote must contain the object's stored label, so that label must be in the text, so the exact-name path already offered it.

Judge (29 reads): 29 judged, 0 errors. The judge said share 15, note 13, none 1. After the deterministic checks:

- 16 accepted: 8 share, 7 note, 1 none.
- 13 rejected: 12 `prior_relation_without_prior`, 2 `summary_sentence_count`, 1 `quote_not_in_read`, 1 `strained_prior_not_cited`. Some items carry more than one reason.
- 0 `citation_not_in_packet`: the judge never cited something it was not shown.
- 12 of the 13 rejections are the same misuse. The judge sets `bears_on_prior: "supports"` while citing atlas concepts, not a belief (11 supports + 1 strains, none of them citing a `self_prior`). The check is right to refuse it: the field is for beliefs only. The connection text itself is often fine, which is why rejected items stay in the labeling set.
- About 25 s per call on gpu1 (15-36 s).

## Current architecture

- Stage 2 claim objects came from the exact-label path (`assertions.object_candidates`), plus, since 10-13, the concepts nearest each subject by raw cosine >= 0.65 from Chroma `orion_atlas_concepts` (`concept_search.related_objects`).
- A link is accepted only by `reading_quote_rule_v1`: the quote is verbatim in retained text and names the object by its stored label, both ids are real, and the predicate is allowed with its domain rule.
- The resonance judge (`run_reading_resonance_eval.py`) had never run; the labeling set showed retrieval's top pieces.

## Architecture touched

- `orion/world_pulse_read/concept_search.py`: `related_objects` embeds the subjects, reads the whole index (`all_vectors`), and ranks with `rank_centered` (index mean, best over subjects). `OBJECT_MIN_SIMILARITY` and `NEAREST_PER_SUBJECT` are retired, not left beside the new floor. Fails closed: if search is unavailable, the label objects stand.
- `orion/introspect/semantic_index.py`: `all_vectors` pages through Chroma `/get` with embeddings and decodes each page off the event loop.
- `orion/world_pulse_read/resonance.py`: the judge prompt states the schema's length limits, read from the fields.
- Evals: new `run_reading_link_targets_replay.py`, which replaces `run_reading_claim_objects_replay.py`. `run_reading_resonance_eval.py` adds the pool gate, the resume cache, `--until`, and the HTML page.
- No env, schema, bus, or compose change.

## Metric gate: centered rank score for claim targets

1. **Provenance.** vector-host `/embedding` (bge-large-en-v1.5) vectors already stored in Chroma `orion_atlas_concepts`. The score is `concept_search.rank_centered`: each subject and each concept minus the index mean, renormalized, then the dot product, max over subjects.
2. **Independence.** This is a transform of the same cosine the raw floor used, not a new signal. It replaces the raw score (which is retired) and only orders and gates an offer list.
3. **Theory.** All-but-the-Top (Mu & Viswanath 2018): removing the common mean direction gives anisotropic embeddings a calm state. This is the same anchor as patch A's `resonance_similarity`.
4. **Live check** (625 concept pairs; 1,022 indexed concepts):
   - Two concepts minted by the same read (related by construction): p10 0.18, p50 0.37.
   - A concept from a different read: p50 0.05, p95 0.30, p99 0.46.
   - Best over subjects, the statistic Hub actually thresholds (vs concepts other reads minted, 28 reads, n=7,008): p50 0.08, p95 0.34, p99 0.51. 4.6% of those concepts clear 0.35.
   - So the calm state is real (0.08, not 0.53 raw), and the list is a recall net with about 5% adjacency in it. "Other read" includes sibling reads on one topic, so 4.6% is an upper bound on noise.
5. **Existing mechanism.** This is the 10-13 meaning path, re-scored. Nothing new was stood up.
6. **Reversibility.** A constant and one function. Nothing is persisted, and no schema changes.

## Co-read edges: skipped

Of the 260 reading-minted concepts, 0 carry more than one source URL. Reconcile never merged a concept across two reads. Co-read edges would therefore form 75 within-read cliques and connect no read to any other read or to the rest of the atlas. That is the opposite of the gap. The fact "minted by the same read" is already in each node's `evidence_refs`. An edge would copy provenance into the graph without new evidence, and it would bypass #2581's "the source states it" rule.

## Rule change for Juniper (proposal only, not built)

The acceptance rule was kept exactly. Numbers: links 1 -> 1. Meaning-only claims proposed: 0. Blocked by the rule: 0 this run, and all of them by construction.

The narrowest change that could ever link a meaning target is a subject-anchored rule (`reading_meaning_rule_v1`). It applies only to objects offered by meaning, and only to predicate `associated_with`. It would accept when:

- the quote is verbatim in retained text and names the **subject** by its stored label;
- both ids are real;
- the domain rule holds.

The decision would carry the centered score and its own policy name, so every such link can be listed or reverted at once.

Two caveats:

- The yield cannot be measured yet. The Stage 2 prompt currently tells the model these claims will not link, so it proposes none (0 here).
- The alternative with no rule change: route `proposed` meaning claims to Juniper's review, so that her accept/reject builds the labelled set a score rule would need.

Bigger lever: retained text. Reads from about 10-11 onward keep their text, so the 26-of-29 "no claim block" share should fall on its own.

## Files changed

- `orion/world_pulse_read/concept_search.py`: centered ranking, floor, docs.
- `orion/introspect/semantic_index.py`: `all_vectors`.
- `orion/world_pulse_read/resonance.py`, `orion/world_pulse_read/tests/test_resonance.py`: prompt length limits plus a test.
- `services/orion-hub/scripts/main.py`: imports the new floor.
- `services/orion-hub/README.md`: meaning-objects paragraph (floor, replay command).
- `services/orion-hub/evals/run_reading_link_targets_replay.py` (new), `run_reading_claim_objects_replay.py` (deleted, superseded).
- `services/orion-hub/evals/run_reading_resonance_eval.py`, `test_reading_resonance_eval.py`: pool gate, cache, `--until`, labeling-set recall fields, HTML page.
- `services/orion-hub/evals/test_reading_assertion_eval.py`: docstring pointer.
- `services/orion-hub/tests/reading_assertion_fakes.py`, `test_world_pulse_read_claim_related_objects.py`: fake whole-index `/get`, centered tests.
- `.github/workflows/orion-reading-tests.yml`: runs `test_reading_resonance_eval.py`.
- `docs/superpowers/evals/2026-10-13-resonance-labeling-set.{md,csv,html}`: judged set.
- `docs/superpowers/pr-reports/2026-10-13-reading-claim-semantic-objects-pr.md`: superseded note.

## Schema / bus / API changes

- Added: none (`all_vectors` is an in-process helper).
- Removed: constants `OBJECT_MIN_SIMILARITY` and `NEAREST_PER_SUBJECT`.
- Renamed: none.
- Behavior changed: Stage 2's "objects related by meaning" list is centered-ranked. The judge prompt states its length limits.
- Compatibility notes: none on the wire.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
PYTHONPATH=. pytest -q services/orion-hub/tests/test_reading_ingress.py services/orion-hub/tests/test_reading_turn_listener.py \
  services/orion-hub/tests/test_reading_postgres.py services/orion-hub/tests/test_world_pulse_read_*.py \
  services/orion-hub/tests/test_curiosity_introspect_listener.py orion/world_pulse_read/tests orion/introspect/tests \
  tests/test_world_pulse_read_*.py services/orion-hub/evals/test_reading_*_eval.py
-> 796 passed, 69 skipped
pyflakes on touched files: clean
check_settings_defaults orion-hub --example-drift OK; check_metric_lineage --gate PASS; --prompt-semantics PASS;
check_definition_drift --gate PASS; check_scripts_dir_no_stdlib_shadow clean; check_async_routes_not_blocking clean
```

HTML page: headless Chromium (playwright's chrome-headless-shell) loaded the generated file. It rendered 29 cards. Clicking yes and typing a note updated progress to "1 / 29 labelled" and saved to localStorage. Export produced a CSV with quotes escaped. The probe's first run caught a real bug: Python had cooked `"\n"` inside the JS, which broke the CSV regex. Fixed with a raw template, and a test now pins it.

## Evals run

```text
run_reading_link_targets_replay.py --until 2026-10-14T00:10:00+00:00 --llm-url .../v1/chat/completions --llm-model agent
  totals reads=29 with_text=3 usable_before=1 usable_live=3 usable_after=3 offered_before=1 offered_live=52 offered_after=36
         acceptable_before=1 acceptable_live=1 acceptable_after=1
  llm live:  reads=3 claims=1 accepted=1 to_meaning_objects=0
  llm after: reads=3 claims=1 accepted=1 to_meaning_objects=0
  null best-over-subjects p50=0.078 p95=0.343 p99=0.512 share_clearing_floor=0.046
run_reading_resonance_eval.py --judge-url http://127.0.0.1:8210/v1/chat/completions --judge-model agent --until 2026-10-14T00:10:00+00:00
  judged=29 accepted=16 share=8 note=7 rejected=13 (resumed across 3 pool pauses; 6 judgments from a first
  run with the old prompt archived, not reused)
```

## Docker/build/smoke checks

```text
Not deployed (instructed). Live path UNVERIFIED until Hub runs this code and a read with retained text completes
Stage 2: expect `reading_claim_context ... related=N related_top=0.3x-0.5x related_ms=...` (related_ms now
includes the ~2 s whole-index read).
```

## Review findings fixed

Code review ran in a subagent (orion-repo-agent) on 2fb1017de..99904c972.

- Finding (MUST): the pool gate missed `recalling` and `retry_wait` leases (`"recall"` is a heartbeat reply, not a lease status).
  - Fix: any lease not in `TERMINAL_STATUSES` is busy, and so are a mid-swap card and paused actuation.
  - Evidence: `test_pool_is_busy_on_any_non_terminal_lease_and_quiet_on_history_only` iterates every real `LeaseStatus` plus an unknown status.
- Finding: the judge cache ignored the model and token budget, and its writes were not atomic.
  - Fix: the cache is now per model and max_tokens, written via tmp then replace; an unreadable file counts as a miss.
  - Evidence: `test_judgment_cache_is_per_model_and_survives_junk`.
- Finding: the replay's LLM cache was keyed by seed, not prompt, and the replay had no `--until`.
  - Fix: the key is sha(model, budget, prompt), and `--until` was added.
  - Evidence: replay re-run.
- Finding: the floor was justified on single pairs, but Hub thresholds the best over subjects.
  - Fix: the replay now reports that null (p95 0.34, 4.6% clear); it is documented at the constant and above.
  - Evidence: replay output.
- Finding: about 10 MB of JSON per page was decoded on Hub's event loop.
  - Fix: `all_vectors` decodes in `asyncio.to_thread`.
  - Evidence: code; tests green.
- Finding: the docstring cited a missing PR report.
  - Fix: this file.
- Finding: the HTML page allowed `javascript:` hrefs, and the CSV export allowed formula injection.
  - Fix: only http(s) hrefs, and leading `=+-@` cells are prefixed with `'`.
  - Evidence: `test_labeling_html_is_self_contained_and_cannot_be_broken_by_stored_text`.
- Nits fixed: a non-dict pool body counts as busy, plus tests for an index that exists but is empty and for a whole-index read failing.
- Found while running (not by review): the gate paused the gpu1 judge for topic-foundry's `fast` requests on gpu3.
  - Fix: a granted, non-background lease on a role with no card in common with `--judge-role` is not contention. Waiting work, background holds (reading, curiosity) and unknown roles still pause.
  - Evidence: `test_judge_ignores_foreground_work_on_other_cards_but_not_background_or_waiting`.
- Not fixed (nit): `all_vectors` offset paging is not snapshot-consistent while the index loop upserts. A duplicate is deduped by the re-gate; it only nudges the mean.
- Not fixed (nit): no overall `wait_for` around `related_objects`. Each request keeps its own 5 s timeout and fails closed.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: medium.
  - Concern: the 0.35 floor is tied to today's index mean, and about 5% of unrelated concepts clear it.
  - Mitigation: these are offers only, and the quote rule decides links. Re-run the replay after the atlas changes shape.
- Severity: medium.
  - Concern: the judge says "share" for 15 of 29 reads (8 after checks), well above the earlier rough guess of 3-6. It may be too generous.
  - Mitigation: Juniper's labels decide; this is exactly what the set is for.
- Severity: low.
  - Concern: the judge misuses `bears_on_prior` (12 of 13 rejections).
  - Mitigation, for patch B: tell the judge to leave it null unless a `self_prior` is cited, or drop the field when no prior is cited. The latter is a check change for Juniper to approve.
- Severity: low.
  - Concern: about 2 s is added to each Stage 2 read for the whole-index read.
  - Mitigation: a Stage 2 turn takes minutes, and the time shows in `related_ms`.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2647

🤖 Generated with [Claude Code](https://claude.com/claude-code)
