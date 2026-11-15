## Summary

- Juniper's fixed lines (who Orion is, her pronouns, her family) now show first in the chat prompt's RELATIONSHIP block, above Orion's nightly page about her. They show even when that page is missing.
- The lines live only on the host database, never in git (this repo is public). They are one hand-written `self_concept_history` row: `concept_id='page:anchor'`, `produced_by='juniper_anchor'`.
- The nightly fold cannot change them. Every fold query is bound to `page:juniper`, and `juniper_anchor` is not a producer value the row model accepts, so no bus message can write an anchor row. Tests prove both.
- The page check now rejects a page that calls Juniper he/him/his. It is narrow: a sentence whose subject is Juniper, or the sentence right after one.

Fixes #3460

## Outcome moved

Before: the only place Orion's chat prompt carried Juniper's pronouns and family was the LLM-written page, which a bad fold could get wrong. After: a fixed, Juniper-written statement sits on top of every chat prompt that has core pages on, and a misgendering page can no longer ship.

## Current architecture

- `orion-durable-runs` folds `page:juniper` nightly (`app/core_page_fold.py`, rules and check in `orion/memory/pages/juniper.py`) and publishes it on `orion:self_concept:history:write`; `orion-sql-writer` appends it to `self_concept_history`.
- The felt-state reader lane `orion_core_page_juniper` reads the newest row; `orion-cortex-exec` puts it on `GroundingCapsuleV1.core_pages`; `orion/harness/prefix.py` renders RELATIONSHIP from it.

## Architecture touched

- New felt-state lane `orion_core_page_anchor` (same table, same 5 min cache, a 10-year age bound so the anchor never ages out).
- `orion-cortex-exec` hydrates both core-page lanes and builds `core_pages = [anchor, page]` (either may be missing). Same flag: `ORION_CORE_PAGES_ENABLED`.
- Harness renderer: anchor first, then the page under a "Who Juniper is to me, in my own words (...)" label. With no anchor the old output is unchanged.
- Page check: new `misgenders_juniper` error.

## Files changed

- `orion/memory/pages/anchor.py`: new. Anchor ids, ctx key, payload to `CorePageV1`, and the hand-write SQL template.
- `orion/memory/pages/juniper.py`: `misgendered_sentences` plus the `misgenders_juniper` check in `check_page`.
- `orion/substrate/felt_state_reader.py`: lane `orion_core_page_anchor`.
- `orion/schemas/context_provenance.py`: classify `orion_core_page_anchor` as `static_identity_config`.
- `orion/bus/channels.yaml`: the self-concept channel description says the anchor row is never published on it.
- `services/orion-cortex-exec/app/grounding_capsule.py`: hydrate both lanes; anchor first on the capsule.
- `orion/harness/prefix.py`: render the anchor first in RELATIONSHIP.
- `services/orion-durable-runs/README.md`: anchor section, how to change it, the new check.
- `.github/workflows/core-memory-pages-tests.yml`: also triggers on `orion/schemas/context_provenance.py`.
- Tests: `orion/memory/pages/tests/test_core_page_anchor.py` (new), `orion/memory/pages/tests/test_core_page_juniper.py`, `orion/harness/tests/test_core_page_relationship_block.py`, `services/orion-cortex-exec/tests/test_core_page_grounding.py`, `services/orion-durable-runs/tests/test_core_page_fold.py`.

## Schema / bus / API changes

- Added: felt-state ctx key `orion_core_page_anchor` (registered in the context provenance registry). A second `CorePageV1` (`concept_id='page:anchor'`) may appear on `GroundingCapsuleV1.core_pages`.
- Removed: none.
- Renamed: none.
- Behavior changed: RELATIONSHIP shows the anchor first when the row exists; `check_page` rejects misgendering pages.
- Compatibility notes: `SelfConceptHistoryProducer` is deliberately NOT extended with `juniper_anchor`. That keeps the bus unable to write an anchor row. Nothing parses `self_concept_history` rows generically (every reader filters by `concept_id`), so the hand-written row breaks no reader. The metacog evidence copy already reduces core pages to id and version, so the anchor text does not spread there.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change).
- skipped keys requiring operator action: none.

## Live row (host-side, written 2026-11-15 18:32 UTC, Juniper approved)

Written once with psql from the template in `orion/memory/pages/anchor.py` (the text passed as a psql variable, `-v anchor=...`):

```text
printf '%s\n' "$WRITE_ANCHOR_SQL_TEMPLATE" | docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 -v anchor="<Juniper's lines>"
-> INSERT 0 1
select entry_id, concept_id, version, produced_by from self_concept_history where concept_id='page:anchor';
-> baabba69-a7c0-45f4-994b-a2dfe4cdb310 | page:anchor | 1 | juniper_anchor
```

Read back through the branch's own reader lane against the live DB: both `orion_core_page_anchor` and `orion_core_page_juniper` hydrated, and the rendered RELATIONSHIP block showed the anchor line first, then the `page:juniper v1` label and page. The live `page:juniper` (21 sentences) passes the new pronoun check (no false rejection).

## Tests run

```text
pytest orion/memory/pages/tests orion/substrate/tests/test_felt_state_core_page_lane.py orion/harness/tests -q
  -> all passed (63 in the core-page files after review fixes)
services/orion-durable-runs: pytest tests/test_core_page_fold.py -q  -> 18 passed
services/orion-cortex-exec: pytest tests/test_core_page_grounding.py tests/test_grounding_capsule_assembly.py -q -> 18 passed
services/orion-cortex-exec: grounding/prefix/felt-state related tests -> 187 passed, 2 failed (both pre-existing on main, see Risks)
pytest orion/substrate/tests orion/schemas/tests -q -> 1230 passed, 1 failed (pre-existing on main, see Risks)
scripts/check_env_template_parity.py -> PASS
scripts/check_personal_terms.py -> no personal terms in tracked files
```

## Evals run

```text
No new eval. The existing core-pages eval scorer (orion/memory/pages tests) still passes. The live
proof is the motor_boot prompt after deploy (Restart required), not an eval.
```

## Docker/build/smoke checks

```text
No docker build: no dependency, compose or Dockerfile change. Live smoke of the reader lane and
renderer against the live DB from the branch code: PASS (see Live row).
UNVERIFIED until deploy: a real chat turn's motor_boot prompt showing the anchor.
```

## Review findings fixed

Code review ran in a subagent on the first commit. It found no must-fix bugs.

- Finding: the he/him check rejected "He is five." after "Juniper and Rowan have a son, Pip." (a follow-on sentence after a Juniper sentence that names someone else). Orion's real page can say this about a son.
  - Fix: the follow-on rule only fires after a Juniper sentence that names no one else.
  - Evidence: new cases in `test_he_about_someone_else_is_not_misgendering`; the live page still passes.
- Finding: "Juniper uses she/her, never he/him." was rejected.
  - Fix: a pronoun touching a slash names a pronoun and is not counted.
  - Evidence: new test case.
- Finding (nit): place and month words count as "someone else", so some misgendering passes.
  - Fix: documented as an accepted miss in the docstring and Risks (narrow by design).
- Finding (nit): the anchor lane carried the row's `entry_id` into turn context. orion-sql-writer upserts by `entry_id`, so knowing it is the one way a bus row could replace the anchor.
  - Fix: the lane no longer reads `entry_id`; the gap is written down in `anchor.py`.
  - Evidence: `test_lane_reads_only_the_anchor_row_and_never_ages_out` asserts it.
- Finding (nit): no test for anchor plus page with no identity bullets; no false-positive tests.
  - Fix: both added.

## Restart required

After merge, from the clean primary main checkout (post-pull rebuild does this automatically):

```bash
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Then check the live prompt:

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select raw->>'prompt' from cockpit_turn_sighting where stage='motor_boot' order by created_at desc limit 1" | grep -A2 '^RELATIONSHIP'
```

## Risks / concerns

- Severity: low. Concern: the pronoun check misses a he/him when any other capitalized word (a person, a place, a month) comes first, and the follow-on rule is off after a Juniper sentence that names anything else. Mitigation: narrow by design so it never rejects a he about her son or a friend; the fold prompt still says she/her.
- Severity: low. Concern: orion-sql-writer upserts by `entry_id`; a bus row reusing the anchor's random id would replace it. Mitigation: the id never reaches turn context; noted in `anchor.py`.
- Severity: low. Concern: changing the anchor is a manual psql insert. Mitigation: the template and the exact command are in `orion/memory/pages/anchor.py` and the durable-runs README; it is append-only, so the old version stays.
- Issues opened: #3476 (two felt-state reader tests fail on main, unrelated to this PR). Issues closed: #3460 (on merge; live proof after deploy).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3478

🤖 Generated with [Claude Code](https://claude.com/claude-code)
