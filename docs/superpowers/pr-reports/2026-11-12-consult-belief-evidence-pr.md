## Summary

- Each consult outcome now ends up on one of Orion's beliefs, and Hub checks that it did. An outcome is an answered consult, or Juniper's Yes/No on a brief.
- The next curiosity run is shown the outcome, the beliefs the consult came from (with their current numbers and full ids), and one fixed finding id to write: `consult:<id>:answer` or `consult:<id>:decision`.
- Orion writes that `:Finding` with MERGE and joins it to a `:Prior`. At the end of the turn, Hub reads the graph (read only) and stamps the consult row with the finding and the beliefs.
- If Orion did not record it, the outcome goes to a later run, 3 runs at most. A run that never reached its turn gives its outcomes back without counting the offer. A claim no run settled is taken again after 12 hours.
- Juniper's answer is joined with `ABOUT` and must not move confidence: it says what she wants, not what is true. An answered consult is tested like any evidence (`SUPPORTS`/`CONTRADICTS`, `times_tested + 1`).
- Hub chat shows, for each consult, whether Orion recorded it and on which belief.

## Outcome moved

Before: run `f310d71391e9` was told about consult 9f138a849520 (2026-11-12 07:08Z). No node in `orion_worldview` mentions that consult (`MATCH (n) WHERE ... CONTAINS '9f138a849520'`: none). Nothing checked. The design's acceptance checks 6 and 7 (decision evidence on a belief, answered increases `times_tested`) were not met.

After: every outcome is either recorded (finding + belief ids stamped on `orion_consult`) or visibly not recorded after 3 offers. Consult 9f138a849520 is offered again (its claim predates offer counting).

## Current architecture

`take_outcomes` stamped `orion_told_run_id` once and the kickoff printed a short "record each on a belief" section (`orion/consult/core.py consult_outcomes_section`). No belief was named, no finding id was given, nothing was read back, and outcomes claimed by a run that was cancelled were never given back.

## Architecture touched

- `orion/consult/beliefs.py` (new, pure): finding ids, source beliefs, the two fixed graph reads, read-back matching, the kickoff section.
- Hub: `scripts/consult_loop.py` (store), `scripts/curiosity_investigation.py` (offer, read-back, release), `static/js/consult.js` (status line).
- DB: `orion_consult` belief columns + index.
- No bus, schema-registry or env changes.

## Files changed

- `orion/consult/beliefs.py`: new module (above). `consult_outcomes_section` moved here from `orion/consult/core.py`.
- `orion/curiosity/kickoff_prompt.py`: imports the section from its new home.
- `services/orion-hub/scripts/consult_loop.py`: `take_outcomes` counts offers (cap 3, skips recorded, re-offers pre-patch claims); `outcomes_offered_to`, `stamp_belief`, `release_outcomes`; `belief_state` on every row for Hub chat.
- `services/orion-hub/scripts/curiosity_investigation.py`: `_take_consult_outcomes` attaches the source beliefs; `_stamp_consult_beliefs` at both turn ends; consult release on cancel; `_release_offers_after_failed_run` replaces `_release_corrections_if_unseen` (corrections semantics unchanged) and also handles consult outcomes.
- `services/orion-hub/static/js/consult.js`: `statusText` adds the belief line.
- `services/orion-sql-db/manual_migration_orion_consult_belief_v1.sql`: 5 additive columns + 1 index.
- Docs: Hub README consult section, design spec "Patch 1d".
- Tests: `orion/consult/tests/test_beliefs.py`, `services/orion-hub/tests/test_curiosity_consult_beliefs.py`, `test_consult_belief_store_sql.py`, `test_consult_js.py`, `test_consult_loop.py`, `test_orion_day_correction_listener.py` (renamed method).

## Schema / bus / API changes

- Added: `orion_consult.belief_offers`, `belief_finding_id`, `belief_prior_ids`, `belief_recorded_run_id`, `belief_recorded_at`; index on `orion_told_run_id`. Consult API rows carry these plus a derived `belief_state` (`recorded` | `not_recorded` | `waiting` | null).
- Graph contract (Orion writes, Hub reads): `:Finding {finding_id: "consult:<id>:answer|decision"}` joined to a `:Prior`.
- Removed: `orion.consult.core.consult_outcomes_section` (moved to `orion.consult.beliefs`).
- Renamed: `CuriosityInvestigation._release_corrections_if_unseen` → `_release_offers_after_failed_run`.
- Behavior changed: an outcome can be offered to up to 3 runs instead of exactly 1.
- Compatibility notes: the migration is additive and on the post-pull auto-apply allowlist (dry-run: "would apply"). Hub needs it before boot; the deploy wrapper refuses `up` until it is applied.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no env template changed).
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-hub/tests (from repo root): 4709 passed, 148 skipped, 1 failed
  the failure is test_fcc_model_labels_api.py::test_fcc_model_labels_disabled_by_default:
  PermissionError on /home/athena/.fcc/.env; fails the same on main (host permission, unrelated)
orion/curiosity/tests + orion/consult/tests: 301 passed, 29 skipped (orion/orion_day: passed)
test_consult_belief_store_sql.py: 4 passed against local Postgres (temp table, rolled back;
  live orion_consult checked unchanged afterwards)
scripts/check_migration_required_by.py: OK
apply_post_pull_migrations.py --dry-run: NOT APPLIED -> would apply (single transaction)
```

## Evals run

```text
No consult eval covers the belief step. Live read-only check instead: both new graph reads (the scoped read-back included) run
against orion_worldview (0 consult findings today, 457 prior rows), and each existing consult
names source beliefs that fit its topic (e.g. e8d34fdb8236, the walkway handoff consult, names 5
handoff priors from its source run).
```

## Docker/build/smoke checks

```text
Not run: Juniper deploys (merge + git pull applies the migration, then rebuilds orion-hub).
```

## Review findings fixed

Review ran in a subagent (orion-repo-agent) on the first commit.

- Finding (high): a claimed outcome could stay claimed forever when its run died on a path that neither stamped nor released (unconfirmed durable receipt, cancel during dispatch, a `completed` run whose turn-end stamp failed, an exception between take and turn).
  - Fix: the take also reclaims a claim older than `STALE_CLAIM_SEC` (12 h); the `completed` run-state branch reads back and settles; `belief_state` reads a stale last offer as `not_recorded`.
  - Evidence: `test_a_claim_no_run_settled_is_taken_again_after_the_stale_age` (real SQL), `test_a_last_offer_whose_run_never_settled_reads_not_recorded`.
- Finding (medium): a durable attempt released unrecorded outcomes at once, so a retry that then wrote the finding was never stamped and the outcome could end "not recorded" while recorded.
  - Fix: a durable attempt's turn end stamps only (`release_missing=False`); the run's terminal state gives back the rest.
  - Evidence: `test_a_durable_attempt_stamps_but_gives_back_nothing_until_the_run_ends`.
- Finding (medium): a failed graph read spent an offer and re-offers could re-test a prior for the same evidence.
  - Fix: a failed read keeps every claim; a repeat offer tells Orion to only join the existing finding and not test again.
  - Evidence: `test_an_unreadable_graph_keeps_every_claim`, `test_a_repeat_offer_says_not_to_test_again`.
- Finding (medium): the read-back did not enforce "Juniper's answer is not proof": a decision finding joined with SUPPORTS counted.
  - Fix: `EDGES_FOR`: decision counts only with `ABOUT`; answer with ABOUT/SUPPORTS/CONTRADICTS. A wrong join is not recorded and is logged (`wrong_edge=`).
  - Evidence: `test_only_the_expected_finding_joined_by_an_allowed_edge_counts`, `test_juniper_answer_joined_as_proof_is_not_recorded`.
- Finding (low): the read-back scanned every consult finding under a row cap and credited the reading run, not the writer.
  - Fix: the read lists only the expected ids (validated hex, so nothing untrusted reaches Cypher); `belief_recorded_run_id` is the finding's own `run_id`.
  - Evidence: `test_the_read_back_lists_only_well_formed_consult_finding_ids`; ran live against `orion_worldview`.
- Finding (low): the mid-turn cancel path awaited a graph read during shutdown.
  - Fix: bounded with `asyncio.wait_for(..., 5 s)`; anything left is reclaimed by the stale rule.
- Finding (low, no change): release order in `_release_offers_after_failed_run`. Every step there already swallows its own errors (`release_corrections`, `_release_consult_outcomes`, `_stamp_consult_beliefs` never raise), so one cannot skip another.
- Finding (low, stated): the legacy clause re-offers every outcome told before this patch (today only 9f138a849520), up to 3 times. Intended.

## Restart required

```text
No manual restart. The post-pull rebuild applies the migration and rebuilds orion-hub.
```

## Risks / concerns

- Severity: medium. Concern: Orion may still not record outcomes (the instruction is in the prompt; Orion chooses). Mitigation: Hub now sees it: after 3 offers the consult says "Orion did not record it on a belief" in Hub chat, and the log line `curiosity_consult_beliefs run=... not_recorded=...` names each miss.
- Severity: low. Concern: the source beliefs are a guess (priors the source run formed or last tested); `last_run_id` can move to a later run. Mitigation: Orion may join any prior it holds or make a new one; the list is a menu.
- Severity: low. UNVERIFIED live: the first run after deploy should be offered 9f138a849520 and the 3 decided briefs as Juniper answers them. Check: `SELECT consult_id, belief_offers, belief_finding_id, belief_prior_ids FROM orion_consult`.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
