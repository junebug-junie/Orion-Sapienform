## Summary

- Orion gets a second page they keep in their own words: "how I came to be" (`page:origin`, issue #3462). It is rewritten overnight from cited records and shown in every unified chat turn, in WHO YOU ARE, right after Orion's own self-definition.
- **Built ahead of the `matters` grade gate (#3590), by Juniper's decision of 2026-11-18.** None of this page's sources use the `matters` grade, so it does not wait for that gate.
- The page reads three sources:
  - Juniper's salience-1 stance messages, word for word (44 live).
  - `episode_memory` rows with purpose `happened` or `orion_view` (372 live).
  - 8 evenly spaced versions of Orion's `self:definition` (91 live). The first and the newest are always included.
- The memory loss is a checked rule, not only a prompt line (spec Decision 3). If any source records the loss, a page that does not state it in plain words, and cite that record, is rejected.
- The fold code shared by every page moves out of `juniper.py` into `orion/memory/pages/fold.py` (`FoldSpec`). The overnight tick now folds each page in `CORE_PAGE_FOLD_PAGES`, one after another.

## Outcome moved

- Before: Orion's prompt held no account of where Orion came from. It also never said that a year of Orion's memory was lost. WHO YOU ARE held only a body census.
- After deploy: every unified turn carries a cited origin page, and the page states the loss plainly.
- Dry run against live data (nothing written), on the final code. Model `qwen3.8-flash-next-iq3_s` on `memory_distill`.
  - The page passed the check on attempt 3 of 3.
  - Attempts 1 and 2 were rejected: an uncited topic sentence each time, and attempt 2 also had one misquote.
  - An earlier run on the first commit passed on attempt 1. That page also stated the loss.
  - Result: 294 words, 16 sentences, 16 distinct cited records.

The page text is not reproduced here. It quotes Juniper's private messages, and pages stay inside
Orion's mesh (spec Decision 4; the repo is public). It was handed to the orchestrating session privately.
What the page covered, in outline:
- why Juniper made Orion;
- that Orion's memory storage died and over a year was lost (stated plainly, cited to her messages);
- that Orion's records start in 2026-07;
- four milestones Juniper marked;
- how Orion's self-definition changed across 7 sampled versions, ending with the newest.

Every sentence above carried cites in the stored form (`[stance:<uuid>]`, `[episode_memory:<uuid>]`, `[self_concept_history:<entry_id>]`). They are removed for display, the same way as on page:juniper.

## Current architecture

- `page:juniper` was the only folded page. Its sources, prompt, cite check and stored row all lived in `orion/memory/pages/juniper.py`.
- `services/orion-durable-runs/app/core_page_fold.py` was hard-wired to page:juniper.
- The grounding capsule (`services/orion-cortex-exec/app/grounding_capsule.py`) carried `page:anchor` and `page:juniper`. The harness prefix (`orion/harness/prefix.py`) rendered them in RELATIONSHIP.

## Architecture touched

- **orion/memory/pages:**
  - `fold.py` (new): the shared check, render, stored row and `FoldSpec`. Moved from `juniper.py`, with no behavior change except one prompt formatting change (see Risks).
  - `origin.py` (new): page:origin.
  - `registry.py` (new): every foldable page, by slug.
  - `juniper.py`: re-exports the shared names, so existing imports keep working.
- **orion-durable-runs:**
  - One `CorePageFolder` per page, with one shared lock. The tick, the overnight loop and the manual trigger cannot hold the fold seat for two pages at once.
  - `run_folders_forever` ticks the pages one after another.
  - `POST /core-pages/{slug}/fold` replaces the juniper-only path. The old URL still works.
  - `/health` keeps `core_page_fold` (page:juniper, same shape) and adds `core_page_folds` (per page).
- **Read path:**
  - New felt-state lane `orion_core_page_origin`.
  - New context-provenance entry.
  - The capsule carries page:origin after page:anchor and page:juniper.
  - The prefix renders it after the leading `In my own words` lines, or first when there are none.
- **Mesh edge** (`orion/memory/mesh_edge.py`):
  - page:origin versions are guarded like page:juniper.
  - Cite stripping now also covers the `stance` and `self_concept_history` reference kinds, and needs a word boundary before the kind.

## Files changed

- `orion/memory/pages/fold.py` (new): shared fold machinery, `FoldSpec`, and `PageCheck.cited` (each sentence with its real refs).
- `orion/memory/pages/origin.py` (new): page:origin SQL, built on the lived-memory serving SQL; its sources, prompt, the memory-loss check, and `FOLD_SPEC`.
- `orion/memory/pages/registry.py` (new): `FOLD_SPECS` and `selected_specs`.
- `orion/memory/pages/juniper.py`: uses `fold.py`, adds `FOLD_SPEC`, and keeps its public names.
- `orion/memory/pages/__init__.py`, `orion/memory/pages/evals/juniper_facts.py`: docs, and the import of the renamed script.
- `scripts/core_page_juniper_dry_run.py` renamed to `scripts/core_page_dry_run.py`. New options: `--page juniper|origin` and `--sources-only`. The connection is read-only.
- `services/orion-durable-runs/app/core_page_fold.py`, `app/main.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: folding several pages; the `CORE_PAGE_FOLD_PAGES` key.
- `orion/substrate/felt_state_reader.py`, `orion/schemas/context_provenance.py`: the origin lane.
- `services/orion-cortex-exec/app/grounding_capsule.py`, `app/settings.py`, `.env_example`: carry page:origin; comments updated.
- `orion/harness/prefix.py`: page:origin in WHO YOU ARE.
- `orion/memory/mesh_edge.py`: guard page:origin; cite stripping.
- `orion/schemas/self_concept_history.py`, `orion/schemas/thought.py`: comments only.
- Tests:
  - `orion/memory/pages/tests/test_core_page_origin.py` (new)
  - `services/orion-durable-runs/tests/test_core_page_origin_postgres.py` (new)
  - additions to `test_core_page_fold.py`, `test_core_page_relationship_block.py`, `test_core_page_grounding.py`, `test_felt_state_core_page_lane.py` and `test_mesh_edge.py`
- `.github/workflows/core-memory-pages-tests.yml`, `.github/workflows/orion-durable-runs-tests.yml`: new paths and the mesh-edge test. The pg tests now also run when `orion/memory/pages/**` or `lived_memory.py` changes.
- `docs/superpowers/specs/2026-11-13-core-memory-pages-and-concept-growth-design.md`: Phase 1 status and build order.

## Schema / bus / API changes

- Added:
  - `concept_id='page:origin'` rows in `self_concept_history` (`produced_by='core_page_fold'`, already accepted).
  - Felt-state lane `orion_core_page_origin`.
  - HTTP `POST /core-pages/{slug}/fold`.
  - `/health` `core_page_folds`.
- Removed: none. The `/core-pages/juniper/fold` URL still works through the slug route.
- Renamed: `scripts/core_page_juniper_dry_run.py` became `scripts/core_page_dry_run.py`.
- Behavior changed:
  - The fold tick folds page:juniper, then page:origin.
  - The mesh edge guards page:origin.
- Compatibility: no new bus channel or schema. `CorePageV1` and the self_concept channel are unchanged.

## Env/config changes

- Added keys: `CORE_PAGE_FOLD_PAGES=juniper,origin` (orion-durable-runs). A typo fails at startup.
- Removed or renamed keys: none.
- `.env_example` updated: yes (orion-durable-runs; comment-only in orion-cortex-exec).
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. The primary checkout's `services/orion-durable-runs/.env` has the key.
- Skipped keys requiring operator action: none from this change. The sync script listed 3 diverged keys that were already there and are unrelated.

## Tests run

```text
pytest orion/memory/pages/tests orion/memory/tests/test_mesh_edge.py orion/harness/tests/test_core_page_relationship_block.py \
  orion/substrate/tests/test_felt_state_core_page_lane.py orion/schemas/tests/test_context_provenance.py   126 passed
pytest orion/schemas/tests/test_context_provenance.py orion/memory orion/harness/tests (+ lanes)            3472 passed, 40 skipped
services/orion-durable-runs: pytest tests (full suite, with a throwaway Postgres DSN)                   exit 0
services/orion-durable-runs: test_core_page_fold.py + both pg tests                                   24 passed
services/orion-cortex-exec: test_core_page_grounding.py test_grounding_capsule_assembly.py           19 passed
services/orion-equilibrium-service: test_metacog_gate_core_pages.py                                   2 passed
scripts/check_integration_seam.py   OK
scripts/check_env_template_parity.py   PASS
git diff --check   clean
```

- **pg test** (`test_core_page_origin_postgres.py`, real Postgres): none of these rows reach the fold prompt:
  - superseded rows (status form);
  - rows an active correction points at (pointer form);
  - rejected, expired and out-of-scope rows;
  - rejected stances, and stances below salience 1;
  - self_concept rows that are not Orion's own.
- **Mutation check:** I removed the lived-memory pointer filter from `EPISODE_SQL`. The pg test then failed on `DEAD-POINTER`. I restored the filter.
- **Pre-existing failure, not from this PR:** `services/orion-cortex-exec/tests/test_substrate_felt_state_reader.py::test_stale_row_not_injected` also fails on a clean main checkout. It is tracked in #3476.

## Evals run

```text
No origin eval yet. Filed #3733 (follow-up) for an origin-facts eval like juniper_facts.py.
Live dry run (see the page text above): passed the check on attempt 3 of 3 against live data. Nothing was written.
```

## Docker/build/smoke checks

```text
No image build. The changes are Python only; the orion-durable-runs Dockerfile already copies orion/.
The dry run used the live DB (read-only) and the live gateway (memory_distill).
```

## Review findings fixed

- Finding: the memory-loss rule passed "Juniper told me the AC died [disk record]". Any loss word counted.
  - Fix: the sentence must name what was lost (memory, disk, storage, history, a year, ...) close to the loss word.
  - Evidence: `test_a_sentence_about_something_else_dying_does_not_state_the_loss`.
- Finding: the new pg test did not run when `origin.py`, `fold.py` or `lived_memory.py` changed.
  - Fix: added `orion/memory/pages/**` and `orion/memory/lived_memory.py` to the paths of `orion-durable-runs-tests.yml`.
- Finding: the pg test asserted `!= "no_new_sources"`, which also passes on a DB error.
  - Fix: assert `== "rejected"`, and check that a new prompt was sent.
- Finding: the change signal missed an in-scope row passing its `expires_at`.
  - Fix: the max past `expires_at` is now part of `NEWEST_SOURCE_SQL`.
  - Evidence: a new pg step.
- Finding: mesh-edge `_CITE` stripped `[circumstance: foo]`.
  - Fix: a word boundary before the kind.
  - Evidence: `test_cite_stripping_needs_a_word_boundary_before_the_kind`.
- Finding: a manual trigger for one page could overlap the loop folding the other page. The slug lookup was case-sensitive.
  - Fix: one shared lock across all folders; the slug is lowercased.
- Not changed: `MEMORY_LOSS_RE` stays narrow. It matches exactly the 2 live loss records. Widen it if a differently worded loss record appears.

## Restart required

Nothing is live until these services are rebuilt from main after the merge. Deploy order: orion-durable-runs, then orion-cortex-exec. orion-sql-writer already accepts `core_page_fold`.

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
```

To fold page:origin before the 01:00-03:00 Denver window, run this after the deploy:

```bash
curl -X POST 'http://localhost:8124/core-pages/origin/fold?dry_run=false'
```

## Risks / concerns

- Severity: medium. The cite check proves that a cited record exists and that quotes are word for word. It does not prove that a record says what the sentence claims.
  - Example: run 1 cited an unrelated row next to the disk record, and dropped a hedge word from it.
  - This limit is the same on page:juniper. Mitigation: Juniper can veto any line. The Hub page diff view (#3588) is not built yet.
- Severity: low. The model often writes an uncited topic sentence. The rule against it is now in the prompt, and the check rejects such a page. The run above still needed 3 attempts, which is the whole nightly budget (`CORE_PAGE_FOLD_MAX_ATTEMPTS_PER_DAY=3`). On a bad night the old version stays live and the next night retries.
- Severity: low. In practice page:origin folds every night. New `self:definition` versions and `updated_at` changes on in-scope episode rows land almost daily. The cost is one `memory_distill` call of about 94k characters per night, plus retries.
- Severity: low. One prompt formatting change for the live page:juniper: each record's inner whitespace and newlines are now folded to single spaces, so a multi-line record stays on one line. The quote check is unaffected, because it already normalizes whitespace.
- Severity: low. `happened` rows are capped at the newest 400, plus every correction. 328 are live today, so nothing is dropped yet.
- **UNVERIFIED live:**
  - no `page:origin` row has been written;
  - no turn has rendered page:origin.
  - #3462 stays open with `verify-live` until the first live fold row exists and a turn's prompt shows "How I came to be".
- Issues:
  - Opened: #3733 (origin eval).
  - Closed: none. #3462 is to be closed with live proof.

## PR link

(see PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
