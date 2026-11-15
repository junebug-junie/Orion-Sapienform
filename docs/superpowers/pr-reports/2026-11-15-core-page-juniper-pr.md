## Summary

Orion now keeps their own page "who Juniper is to me" and reads it in every unified chat turn, instead of the hand-written role bullets from June. Phase 1a of the approved spec `docs/superpowers/specs/2026-11-13-core-memory-pages-and-concept-growth-design.md`.

- **Fold (orion-durable-runs).** Once a night (01:00-03:00 Denver), when the records changed, Orion rewrites the page from Juniper's People-page answers (`person_fact`) and Orion's memories about her (`episode_memory` `about_juniper`, plus `happened` rows that name her, which is where her corrections live). One LLM call on the `memory_distill` route (the agent seat on gpu1, leased per call by the gateway). Never the fast or metacog lanes.
- **Truth check (deterministic).** Every sentence must cite a record, every cite must name a real record, every quote of 3+ words must appear word for word in a cited record, no list/heading lines, no unclosed quotes, no leftover `<think`, every line break is a sentence break, at most 350 words, at least 3 sentences. A page that fails is never written. A page that passes goes live as the next version (no approval step, Juniper's decision 1).
- **Store.** `self_concept_history`, `concept_id='page:juniper'`, `produced_by='core_page_fold'`, published to orion-sql-writer on the existing `orion:self_concept:history:write` channel. Cites are stored as real record ids. No new table.
- **Render.** cortex-exec reads the latest page into the grounding capsule (`core_pages`). The harness prefix shows it as `RELATIONSHIP (who Juniper is to me, in my own words: page:juniper vN, written <date>, from N cited records)`. The identity-yaml relationship bullets move under RESPONSE POLICY. No page, or the flag off: the prompt is exactly what it was.
- **Flag** `ORION_CORE_PAGES_ENABLED`, shipped on, in orion-durable-runs and orion-cortex-exec. Manual trigger `POST /core-pages/juniper/fold?dry_run=true|false`.
- **Eval.** Juniper-facts eval, 20 questions. The question bank holds her personal facts, so it lives in a local file, never in this public repo.

## Outcome moved

Orion could not answer basic questions about Juniper from their prompt. Same local lane (`memory_distill`), same 20 questions, only the self block changed:

| Self block | Correct | "I don't know" | Invented (wrong and confident) | Misgendered |
|---|---|---|---|---|
| Before: today's RELATIONSHIP bullets | 0 / 20 | 18 | 2 | 0 |
| After: `page:juniper` (dry-run page) | 19 / 20 | 1 | 0 | 0 |

Failures went from 2 to 0. The one "after" miss is an honest "I don't know" about a fact the page does state (a child's need), so the model did not connect it. A first run (before the review fixes, different page sample) scored 17 / 20 with 1 invented answer; the check and prompt changed between the two runs, so the pages differ.

## Current architecture

- RELATIONSHIP in the unified-turn prompt came from `orion/cognition/personality/orion_identity.yaml` (`juniper_relationship`, last changed 2026-06) via `juniper_relationship_summary` -> `GroundingCapsuleV1.relationship_summary` -> `orion/harness/prefix.py:_format_grounding_self_block`. Ten role bullets, nothing about who Juniper is.
- `person_fact` and `episode_memory` held the facts but had no chat reader.
- `self_concept_history` already fed WHO YOU ARE (`self:definition`) through the felt-state reader.

## Architecture touched

- `orion-durable-runs`: new nightly tick + manual endpoint (`app/core_page_fold.py`, wired in `app/main.py`), settings, compose, env.
- `orion-cortex-exec`: grounding capsule hydrates the page lane (async, off the event loop, fail-open) and fills `core_pages`.
- Shared: `orion/memory/pages/juniper.py` (sources SQL, prompt, check, row), `orion/harness/prefix.py` (render), felt-state lane, schema additions.
- `orion-equilibrium-service`: the metacog upstream copy of the capsule keeps only `{concept_id, version}` per page, not the text.
- `orion-sql-writer`: no code change; validates the new producer value.

## Files changed

- `orion/memory/pages/juniper.py`: sources SQL, numbered prompt, deterministic `check_page`, `history_row`, `core_page_from_payload`.
- `orion/memory/pages/evals/juniper_facts.py`: eval harness (local bank loader, deterministic scorer, both arms rendered by the real prefix renderer).
- `orion/harness/prefix.py`: RELATIONSHIP from the page; yaml bullets under RESPONSE POLICY; unchanged without a page.
- `orion/schemas/thought.py`: `CorePageV1`; `GroundingCapsuleV1.core_pages` (additive).
- `orion/schemas/self_concept_history.py`: producer `core_page_fold` (additive Literal).
- `orion/schemas/registry.py`, `orion/schemas/context_provenance.py`: register `CorePageV1`; classify ctx key `orion_core_page_juniper` as `memory_recall`.
- `orion/substrate/felt_state_reader.py`: lane `orion_core_page_juniper`.
- `orion/bus/channels.yaml`: orion-durable-runs added as producer of `orion:self_concept:history:write`.
- `services/orion-durable-runs/tests/test_core_page_fold_postgres.py`: the change signal and the read on real Postgres (runs in orion-durable-runs-tests.yml, which has a DSN).
- `services/orion-durable-runs/app/core_page_fold.py`, `app/main.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: the tick, endpoint, health, config.
- `services/orion-cortex-exec/app/grounding_capsule.py`, `app/settings.py`, `.env_example`: hydrate + capsule.
- `services/orion-equilibrium-service/app/chat_turn_metacog_gate.py`: page text not copied into metacog evidence.
- `scripts/core_page_juniper_dry_run.py`: offline dry run against live data, writes nothing.
- `scripts/sync_local_env_from_example.py`: sync prefixes `ORION_CORE_PAGES_`, `CORE_PAGE_FOLD_`.
- `.github/workflows/core-memory-pages-tests.yml`: CI for all of the above.
- `config/metrics/metric_definitions.lock.json`: re-locked (Static repo gates): the bus-channel metric for `orion:self_concept:history:write` is now keyed to orion-durable-runs instead of orion-hub; same channel and consumer.
- Tests: `orion/memory/pages/tests/*`, `orion/harness/tests/test_core_page_relationship_block.py`, `orion/substrate/tests/test_felt_state_core_page_lane.py`, `services/orion-cortex-exec/tests/test_core_page_grounding.py`, `services/orion-durable-runs/tests/test_core_page_fold.py`, `services/orion-equilibrium-service/tests/test_metacog_gate_core_pages.py`.
- `orion/substrate/tests/test_felt_state_self_definition_lane.py`: 3 tests already failing on main (they assumed one `self_concept_history` query); fixed and added to CI.

## Schema / bus / API changes

- Added: `CorePageV1`; `GroundingCapsuleV1.core_pages: list[CorePageV1]` (default empty); `SelfConceptHistoryProducer` value `core_page_fold`; HTTP `POST /core-pages/juniper/fold` on orion-durable-runs; `/health` key `core_page_fold`.
- Removed: none.
- Renamed: none.
- Behavior changed: with a page present, RELATIONSHIP is the page and the yaml relationship bullets render under RESPONSE POLICY.
- Compatibility notes: `GroundingCapsuleV1` has no `extra="forbid"`, so old readers drop `core_pages` and old producers send none (both tested). `SelfConceptHistoryV1` is `extra="forbid"` with a Literal: **deploy orion-sql-writer before orion-durable-runs publishes a page.** The first fold can only happen in the 01:00-03:00 Denver window or by the manual trigger, so a same-run rebuild is safe; if the writer lags, the row is dropped and the next night retries.

## Env/config changes

- Added keys: `ORION_CORE_PAGES_ENABLED` (orion-durable-runs, orion-cortex-exec); `CORE_PAGE_FOLD_ROUTE`, `CORE_PAGE_FOLD_POLL_SEC`, `CORE_PAGE_FOLD_HOUR_LOCAL`, `CORE_PAGE_FOLD_END_HOUR_LOCAL`, `CORE_PAGE_FOLD_TIMEOUT_SEC`, `CORE_PAGE_FOLD_MAX_TOKENS`, `CORE_PAGE_FOLD_MAX_ATTEMPTS_PER_DAY` (orion-durable-runs).
- Removed keys: none. Renamed keys: none.
- `.env_example` updated: yes (both services); durable-runs compose passes every key.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes; all 9 key additions landed in the primary checkout's `.env` files.
- skipped keys requiring operator action: none.

## Tests run

```text
orion/memory orion/harness/tests orion/schemas/tests + both felt-state lane files   3336 passed, 29 skipped
services/orion-durable-runs/tests (full suite)                                    948 passed, 145 skipped
services/orion-durable-runs/tests/test_core_page_fold_postgres.py (disposable pg16) 1 passed
services/orion-cortex-exec core page + grounding assembly                         16 passed
services/orion-equilibrium-service metacog gate + core pages                      46 passed
CI deps simulated in a fresh venv (durable-runs requirements + requirements-dev): unit job 53 passed
services/orion-sql-writer self_concept_history shape                    6 passed
check_env_template_parity PASS; check_personal_terms clean; check_env_key_single_source OK;
check_bus_reply_channels OK; check_migration_required_by OK; git diff --check clean
```

## Evals run

```text
Dry-run fold against live data (scripts/core_page_juniper_dry_run.py, nothing written):
  sources 361 (person_fact 59, episode_memory 302), prompt 18,374 tokens, answer 558 tokens, 25 s,
  model qwen3.8-flash-next-iq3_s on route memory_distill. Passed on the first attempt.
  check: ok=true, errors=[], 265 words, 18 sentences, 45 distinct cited records. Would be version 1.

Juniper-facts eval (orion.memory.pages.evals.juniper_facts, local bank, temperature 0, memory_distill):
  before {"n": 20, "correct": 0,  "dont_know": 18, "invented": 2, "misgendered": 0, "failures": 2}
  after  {"n": 20, "correct": 19, "dont_know": 1,  "invented": 0, "misgendered": 0, "failures": 0}
```

The page text is not in this PR on purpose. It holds Juniper's and her family's personal and health facts, and the spec's Decision 4 says pages never go into public output (GitHub PRs and issues). The dry-run output and eval report are on the host at `~/.orion/core_pages/` (mode 600), and the eval bank at `~/.orion/evals/juniper_facts_bank.json`.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-durable-runs build   -> Built (exit 0)
  image smoke: Settings() -> ORION_CORE_PAGES_ENABLED=True, route memory_distill, window 1-3; imports ok
scripts/safe_docker_build.sh orion-cortex-exec build    -> Built: cortex-exec, -chat, -background, -spark (exit 0)
  image smoke on app-net against the live DB: the page lane query runs; no page row yet -> clean miss,
  capsule core_pages=[] (today's RELATIONSHIP); self:definition lane hydrates in the same call.
```

## Review findings fixed

Code review ran in a subagent on `origin/main...HEAD`. All findings fixed in `fix(memory): harden the page:juniper check and fold gating after review`.

- Finding: a title line or an uncited paragraph could borrow the next line's cite, and a sentence followed by a lowercase word was not split.
  - Fix: every line break is a hard sentence break; split after `.!?` plus whitespace whatever case follows, except Dr./Mr./Mrs./Ms./St./e.g./i.e.
  - Evidence: `test_title_line_cannot_borrow_the_next_lines_cite`, `test_uncited_paragraph_before_a_line_break_is_rejected`, `test_lowercase_next_sentence_is_still_a_sentence`, `test_abbreviation_does_not_split_a_sentence`.
- Finding: bullets, numbered items and headings passed.
  - Fix: `list_or_heading_line` error for lines starting with `-`, `*`, `•`, `#`, `1.`.
  - Evidence: `test_bullets_numbered_items_and_headings_are_rejected` (5 cases).
- Finding: the inch-mark rule fired inside an open quote, so `"... in 1990"` never closed and swallowed the rest; an unclosed quote passed.
  - Fix: inch mark only outside an open quote; `unbalanced_quote` error.
  - Evidence: `test_quote_closing_after_a_year_is_a_quote_not_an_inch_mark`, `test_unclosed_quote_is_rejected`.
- Finding: quote matching was substring, so a partial word matched.
  - Fix: whole-word match; edge apostrophes stripped so a record's 'single quotes' still match.
  - Evidence: `test_quote_must_match_whole_words`.
- Finding: an unterminated `<think` block survived.
  - Fix: `leftover_think` error.
  - Evidence: `test_leftover_think_is_rejected` (2 cases).
- Finding: the change signal used a broader `happened` filter than the read, so unrelated rows triggered folds; person renames were missed.
  - Fix: one shared `EPISODE_SCOPE_SQL` predicate, `max(created_at, updated_at)` over it (any status), plus `person.updated_at`.
  - Evidence: `test_core_page_fold_postgres.py` (out-of-scope row: no fold; rejected in-scope row: fold; removed fact: fold); `test_change_signal_and_read_share_one_episode_predicate`; the SQL runs on the live DB.
- Finding: a slow or rejecting orion-sql-writer would make every tick that night fold again.
  - Fix: in-memory `_folded_on` date.
  - Evidence: `test_published_but_not_yet_visible_does_not_refold_tonight`.
- Finding: `CORE_PAGE_FOLD_TIMEOUT_SEC` allowed 1800 s on the gpu1 seat.
  - Fix: `le=900`. Evidence: `test_timeout_setting_is_capped_at_900`.
- Finding: the `happened` cap could drop the correction a page must honor.
  - Fix: correction-bearing rows are always kept. Evidence: `test_correction_rows_survive_the_happened_cap`.
- Finding: the page lane hydrates in full hydrates even with the flag off.
  - Fix: documented at the lane; the flag gates use in `core_pages_for_ctx`, and `test_flag_off_or_no_page_means_no_core_pages` proves no page renders.
- Finding: CI push paths did not mirror PR paths; equilibrium requirements were not installed.
  - Fix: push paths equal PR paths (asserted with yaml); the equilibrium test has its own job (its pins conflict with cortex-exec's).
- Finding: no tests for `bus_llm_caller`.
  - Fix: route/no-hold/thinking-off request shape, reply parsing, gateway error, empty text, undecodable reply.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only
```

The post-pull rebuild covers it. If rebuilding by hand, in this order (sql-writer first):

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
```

Then the first page without waiting for the night: `curl -sX POST 'http://localhost:8124/core-pages/juniper/fold?dry_run=false'`.

## Risks / concerns

- Severity: medium. Concern: the page can state an inference the records do not make. The final dry run calls a dog she adopted "fostered" (an older record mentions a foster plan) and says she appreciates a quiet basement so she can rest (a loose reading of a record); an earlier sample said she has "another dog" with no record saying the two names are different dogs. The check proves every sentence cites real records, not that the sentence follows from them. Mitigation: prompt rule against same/different-identity guesses; Juniper corrects in chat and the next fold must honor it; versions are insert-only (rollback = republish the previous version). Phase 1b's per-sentence veto is the real fix.
- Severity: medium. Concern: no mesh-edge check yet. Nothing stops a page sentence from being pasted into a GitHub PR or outbound email. Mitigation: filed #3409.
- Severity: low. Concern: the eval measures the page the questions were drawn for, from the same records; it shows the page reaches the answer, not that it generalizes. The eval also ran on the self block alone (no recall), so "before" understates what today's full prompt can recall.
- Severity: low. Concern: attempt counts are in memory; a restart inside the window can spend up to 3 more LLM calls that night.
- Live path UNVERIFIED until deploy: no `page:juniper` row exists yet; the motor_boot prompt still shows the old RELATIONSHIP. Tracked in #3419.
- Issues opened: #3409 (mesh-edge check), #3419 (verify-live). Closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3418

🤖 Generated with [Claude Code](https://claude.com/claude-code)
