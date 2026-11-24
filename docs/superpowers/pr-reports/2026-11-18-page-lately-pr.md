Fixes #3463

## Summary

- Orion gets a third page in their own words: "what has mattered lately" (`page:lately`, issue #3463). It is rewritten overnight from cited records and shown in every unified chat turn, at the top of DURABLE MEMORY, above the recall digest (the spec's slot).
- **Built ahead of the `matters` grade (#3590).** That grade does not exist yet. Until it does, when there are too many rows the page keeps follow-ups, memories about Juniper, high-stakes memories and corrections first, then the newest of the rest. The prompt asks the model to weigh the same things.
- **Sources.** The last 14 days of `episode_memory`, plus active `follow_up` rows up to 45 days old, plus Hub chat where Juniper spoke that the episode distiller has not distilled yet.
- **Reader filter.** Episode rows go through the lived-memory serving SQL, unedited, plus `invalid_at IS NULL`. So a row is read only when it is active, and never when it is superseded (status form or pointer form), rejected, expired, or invalidated. A real-Postgres test proves this from day one.
- **One extra check.** When any source is from the newest week, the page must cite at least one of them. A "lately" page that talks only about old things is rejected.
- Same fold machinery as page:juniper and page:origin (`FoldSpec`, the registry, the same nightly tick, the same `ORION_CORE_PAGES_ENABLED` flag). Nothing was forked.

## Outcome moved

- Before: no turn had a short, cited account of the last two weeks. The only recent-memory text was the recall digest, which shows whatever retrieval hits.
- After deploy: every unified turn carries a cited page of what mattered recently. It is refolded nightly, and it changes when recent episodes change.
- **Dry run against live data, final code.** Nothing was written. Model `qwen3.8-flash-next-iq3_s` on `memory_distill`.
  - The check passed on attempt 1 of 3.
  - 270 words, 13 sentences, 13 distinct cited records.
  - Sources: 243 episode rows (about 47k characters of prompt). Undistilled Hub chat: 0 turns today, because Juniper's last chat was already distilled. The chat seam is covered by the pg test, not by this run.
  - An earlier run on the first commit also passed on attempt 1: 311 words, 15 sentences, 15 records. The run before the length rule was tightened failed on length (410 and 419 words).
- **Does it cover last week's real events?** Counts from `orion.memory.pages.evals.lately_coverage`, last 7 Denver days:
  - Cited records from those 7 days: 10 of 13.
  - Days with memories that the page cites: 5 of 6.
  - Episodes cited: 8 of 11.
  - Follow-ups cited: 6 of 10. About-Juniper rows cited: 4 of 19. High-stakes rows cited: 4 of 25.
- **Chat prompt growth:** the page added 1,620 characters to the rendered self block (page text plus a one-line header). The 350-word cap bounds it at about 2,100 characters.

The page text is not reproduced here. It is kept host-side in `~/.orion/core_pages/` (mode 600), because pages stay inside Orion's mesh (spec Decision 4) and the repo is public.

## Current architecture

- Two folded pages: page:juniper (RELATIONSHIP) and page:origin (WHO YOU ARE). Both share `orion/memory/pages/fold.py` and are listed in `orion/memory/pages/registry.py`.
- orion-durable-runs folds every page in `CORE_PAGE_FOLD_PAGES` in the 01:00-03:00 Denver window, one at a time, and skips a page when nothing it reads changed (`no_new_sources`).
- cortex-exec hydrates each page from a felt-state lane into the grounding capsule, and `orion/harness/prefix.py` renders it.

## Architecture touched

- **New page** `orion/memory/pages/lately.py`:
  - `EPISODE_SQL` wraps `lived_memory.EPISODE_SQL` and adds `invalid_at IS NULL`, the 14-day window, and follow-ups up to 45 days old.
  - `CHAT_SQL` reads `chat_history_log` turns with Juniper's words, inside a `memory_episode_shadow` episode that has no `episode_distill_run` row. Command turns are left out.
  - `NEWEST_SOURCE_SQL` is the change signal. It moves on:
    - any change to an in-scope episode or correction;
    - an expiry;
    - a row aging out of the window, or a follow-up passing 45 days;
    - a new chat turn, or a chat turn aging out;
    - a distill run finishing.
  - Sources are dated on Juniper's calendar (Denver), the same as the prompt's "Today is".
- **Shared fold:** a new cite kind `chat_history_log` in `fold.REF_KINDS`.
- **Read path:**
  - felt-state lane `orion_core_page_lately` (a page older than 21 days is not shown);
  - a context-provenance entry;
  - grounding capsule (after page:origin);
  - prefix: the top of DURABLE MEMORY. The header also renders when there is no digest.
- **Mesh edge:** page:lately versions are guarded. The `chat_history_log` cite is stripped before matching.
- **Dry-run script:** `--no-page` keeps the page text off stdout. `--out` is written mode 600.

## Files changed

- `orion/memory/pages/lately.py` (new): sources, selection, prompt, recency check, `FOLD_SPEC`.
- `orion/memory/pages/evals/lately_coverage.py` (new): the coverage eval. It prints counts only.
- `orion/memory/pages/fold.py`, `registry.py`, `__init__.py`: the chat cite kind, and registering the page.
- `orion/harness/prefix.py`: the DURABLE MEMORY slot.
- `orion/substrate/felt_state_reader.py`, `orion/schemas/context_provenance.py`: the lane.
- `orion/memory/mesh_edge.py`: guard the page, and strip chat cites.
- `services/orion-cortex-exec/app/grounding_capsule.py`, `app/settings.py`, `.env_example`: carry the page; comments.
- `services/orion-durable-runs/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`, `app/main.py`, `app/core_page_fold.py`: `CORE_PAGE_FOLD_PAGES` default `juniper,origin,lately`, plus docs.
- `scripts/core_page_dry_run.py`: `--no-page`, and private `--out`.
- `orion/schemas/thought.py`, `orion/schemas/self_concept_history.py`: comments only.
- `docs/superpowers/specs/2026-11-13-core-memory-pages-and-concept-growth-design.md`: status, read-filter table, build order.
- Tests:
  - `orion/memory/pages/tests/test_core_page_lately.py` (new)
  - `orion/memory/pages/tests/test_lately_coverage_eval.py` (new)
  - `services/orion-durable-runs/tests/test_core_page_lately_postgres.py` (new)
  - additions to `test_core_page_relationship_block.py`, `test_felt_state_core_page_lane.py`, `test_mesh_edge.py`, `test_core_page_grounding.py` and `test_core_page_fold.py`
- `.github/workflows/core-memory-pages-tests.yml`: new paths.

## Schema / bus / API changes

- Added:
  - `concept_id='page:lately'` rows in `self_concept_history` (`produced_by='core_page_fold'`, already accepted by orion-sql-writer);
  - felt-state lane `orion_core_page_lately`;
  - `POST /core-pages/lately/fold` (through the existing slug route);
  - the stored cite kind `chat_history_log:<id>`.
- Removed / renamed: none.
- Behavior changed: the nightly tick folds page:lately after page:origin. The mesh edge guards page:lately.
- Compatibility: no new bus channel or schema. `CorePageV1` and the capsule field are unchanged.

## Env/config changes

- Added keys: none.
- Changed default: `CORE_PAGE_FOLD_PAGES=juniper,origin,lately` (orion-durable-runs: `.env_example`, settings, compose).
- `.env_example` updated: yes (orion-durable-runs value; orion-cortex-exec comment only).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: run. It reports `CORE_PAGE_FOLD_PAGES` as **diverged** (`juniper,origin` locally) and leaves it alone, by design.
- **Skipped key requiring operator action:** `CORE_PAGE_FOLD_PAGES` in `services/orion-durable-runs/.env`. I deliberately left it at `juniper,origin`. Code that does not know `lately` (today's main) fails at startup on an unknown slug, by design. So writing the key before this merges would crash-loop orion-durable-runs on any rebuild from main. Set it in the same step as the deploy (see Restart required).

## Tests run

```text
orion/memory orion/harness/tests context_provenance felt-state core-page lane     3508 passed, 41 skipped
orion/memory/pages/tests + test_mesh_edge.py                                     119 passed
services/orion-durable-runs: full tests/ with a throwaway Postgres DSN          1227 passed, 1 skipped
  (one timing flake, test_physiology_market_replay_off_loop, failed once under -x and passed on re-run and alone)
services/orion-durable-runs: core-page fold + all three pg tests                 25 passed
services/orion-cortex-exec: test_core_page_grounding + test_grounding_capsule_assembly   20 passed
services/orion-equilibrium-service: test_metacog_gate_core_pages                 2 passed
check_integration_seam OK; check_env_template_parity PASS; check_metric_lineage --gate PASS;
check_definition_drift --gate PASS (no definition changes); tests/test_no_personal_terms.py 7 passed; git diff --check clean
```

- **pg test** (`test_core_page_lately_postgres.py`, real Postgres). None of these reach the fold prompt:
  - superseded rows (status form);
  - rows an active correction points at (pointer form);
  - rejected, expired and `invalid_at` rows;
  - rows older than 14 days, and follow-ups older than 45 days;
  - outreach turns with no Juniper words, command turns, turns in a distilled episode, turns older than the window, and turns in no episode.

  The change signal is quiet when nothing changed. It moves for a new chat turn, an invalidated row, a row aging out of the window, and a distill run finishing.
- **Mutation checks:** I removed each of these in turn: the `invalid_at` filter, the distilled-episode filter, the aging-out term, the command-turn filter, the distill-run term, and the follow-up cap. Each removal failed the pg test. I restored each one. The mesh-edge test also fails with the old cite regex.

## Evals run

```text
python -m orion.memory.pages.evals.lately_coverage --page-json ~/.orion/core_pages/lately_dry_run.json
  cited 13; from the last 7 days 10; days 5/6; episodes 8/11; follow-ups 6/10; about_juniper 4/19; high stakes 4/25
scripts/core_page_dry_run.py --page lately --attempts 3 --no-page --out ~/.orion/core_pages/lately_dry_run.json
  check ok on attempt 1: 270 words, 13 sentences, 13 cited records. Nothing written.
```

## Docker/build/smoke checks

```text
No image build. Python only; both service Dockerfiles already copy orion/.
The dry run used the live DB (read-only connection) and the live gateway (memory_distill).
No deploy (not requested).
```

## Review findings fixed

The code review ran in a subagent on the first commit.

- Finding: the live `.env` keeps `CORE_PAGE_FOLD_PAGES=juniper,origin`, so the page would never fold after deploy.
  - Fix: not written now, because of the crash-loop risk above. The exact one-line command is under Restart required. The README now states the deploy and rollback order for this key.
- Finding: rolling durable-runs back to an older image while `.env` lists `lately` fails startup.
  - Fix: documented in the README: remove `lately` from the key before rolling back. The strict unknown-slug check stays: origin chose it on purpose, so that a typo cannot turn a page off.
- Finding: sources were dated in UTC, but the prompt's "Today is" was a Denver date. Evening events were misdated by one day.
  - Fix: sources are dated on the Denver calendar.
  - Evidence: `test_sources_are_dated_on_juniper_s_calendar`.
- Finding: every active follow-up was labelled "still open", but nothing can close a follow-up, and the oldest dates from 10-04.
  - Fix: the label and the rules now say "follow-ups I wrote down" and never claim open or done. Old follow-ups are capped at 45 days. Filed #3758.
- Finding: the mesh-edge test passed even without the regex change.
  - Fix: it now uses a chat-only cite.
  - Evidence: the test fails against the old regex.
- Finding: the change signal missed a distill run that keeps no memory, and a chat turn aging out.
  - Fix: both terms added.
  - Evidence: a new pg step, mutation-checked.
- Finding: the chat source counted command turns.
  - Fix: command turns are dropped.
  - Evidence: the `DEAD-COMMAND` row in the pg test.
- Finding: a closed episode the distiller never ran on reads as "not yet distilled". The core-pages workflow lists a pg test it cannot run.
  - Fix: both documented, in the docstring and in a workflow comment.

## Restart required

Nothing is live until the merge and the deploy. Run these from the primary checkout on main, after the merge. Do them in this order, and do the `.env` line in the same sitting as the durable-runs rebuild:

```bash
sed -i 's/^CORE_PAGE_FOLD_PAGES=juniper,origin$/CORE_PAGE_FOLD_PAGES=juniper,origin,lately/' services/orion-durable-runs/.env
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
```

The fold runs 01:00-03:00 Denver. To fold now: `curl -X POST 'http://localhost:8124/core-pages/lately/fold?dry_run=false'`. Check `/health` `core_page_folds.lately`.

## Risks / concerns

- Severity: medium. The cite check proves that a cited record exists and that quotes are word for word. It does not prove that the record says what the sentence claims. This is the same limit as the other pages. The Hub page diff view is not built yet (#3588).
- Severity: medium. **No `matters` grade (#3590).** Selection uses purpose, stakes, corrections and recency. The coverage numbers show the page cites only 4 of 25 recent high-stakes rows. Stakes alone is a coarse signal.
- Severity: low. The page will refold almost every night: rows age out of the 14-day window daily. The cost is one `memory_distill` call with about 47k characters of prompt (latency 3.5 to 5 minutes in the dry runs), plus retries, up to 3 a night.
- Severity: low. The page quotes Juniper's undistilled chat. The mesh edge guards it only after the services that import `orion/memory/mesh_edge.py` (Hub, notify, the FCC sandbox) are rebuilt. The first page:lately row cannot exist before the 01:00 fold, so a same-day deploy is fine.
- **UNVERIFIED live:**
  - no `page:lately` row has been written;
  - no turn has rendered the page;
  - the chat source returned 0 turns live (its behavior is proven by the pg test only).
  - #3463 should get `verify-live` until a fold row exists and a turn shows "What has mattered lately".
- Issues: opened #3758 (follow-ups cannot be closed). Closed: #3463 by this PR, with live proof still owed (see above).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3759

🤖 Generated with [Claude Code](https://claude.com/claude-code)
