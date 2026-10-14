## Summary

Stacked on PR #2647 (centered link targets). Two follow-ups Juniper approved on 2026-10-14.

- **Step 4, a narrow new link rule.** Until now, a reading could only link its new concept to an existing one if the page named that existing concept. Concepts offered "by meaning" could never link. New rule `reading_meaning_target_v1`: a claim to a meaning-only concept is accepted as provisional only when the predicate is `associated_with`, the quote is verbatim in the retained page text and names the read's own concept (the subject), both ids are real, and the concept was actually offered to this read. Everything else stays `proposed`.
- The rule has its own policy name on the journal decision and its own receipt reason (`accepted_meaning_target`), so its links can be counted and withdrawn together. One flag turns it off, `HUB_WORLD_PULSE_READ_MEANING_TARGET_LINKS_ENABLED` (default on). The Stage 2 prompt tells the model the rule, and only while the flag is on.
- Withdraw tool: `services/orion-hub/scripts/reading_meaning_links.py list|withdraw`. It appends a `rejected` decision on top of each link the policy made. Hub's projector then closes the link on its next tick.
- **Step 5, backfill prepared, not run.** 26 of the last 29 reads have no usable page text, so Stage 2 never had anything to quote. `scripts/reading_reread_backfill.py` re-reads them through the normal reading path, one at a time, pausing while the GPU pool is busy. The read-only snapshot and URL probe have run. The writing part waits for this code to be deployed.
- Exact-name targets keep the full #2581 rule set, unchanged.

## Outcome moved

- **Rule, offline eval** (`test_reading_assertion_eval.py`, meaning batch, 7 recorded claims to one meaning-only concept): recall 1.00 on the claims that meet the rule, 0 links from the 5 that miss a condition, 0 links citing bytes that are not in the text.
- **Rule, live replay** (`run_reading_link_targets_replay.py --tags meaning_rule`, the same 29 reads #2647 used, ending 2026-10-14 00:10 UTC): only 3 reads have text. The model, now told meaning targets can link, made 1 claim. It is the same exact-name link as before. It proposed 0 meaning-only claims. **Extra links on today's data: 0.** The rule's yield depends on reads having text, which is what Step 5 provides.
- **Backfill dry run** (read-only, `/tmp/reading-reread-backfill/`): 26 targets.
  - 19 answer 200.
  - 4 are inconclusive and will still be tried, because the reader records its own outcome: three `governor.utah.gov` pages return 403, and the Falcon-ASR model page returns 401.
  - 3 are gone: the three coreweave.com pages return 404. They are recorded and never re-read.
  - Snapshot: 872 KB.

## Current architecture

- `plan_claims` accepted a claim only under `reading_quote_rule_v1`: the quote had to name the object by its stored label. A meaning-only object's label is not in the text, so the rule could never accept one. The prompt told the model so, and live it proposed none.
- A URL is read once (`READ_URL_SQL`). Nothing could re-read a finished read. The stale digest sweep skips any pending `digest_item` older than 5 days.

## Architecture touched

- `orion/world_pulse_read/assertions.py`:
  - `ClaimContextV1.meaning_links` is bound once with the prompt, so the prompt and the rule never disagree.
  - `EndpointV1.score` holds the centered score at offer time. It is written into the decision rationale.
  - New rule branch in `plan_claims`. "Meaning-only" means `via == "meaning"`, the object is in this context's list (not re-found by `object_lookup`), and its label is not in the haystack. A meaning-offered concept whose label is in the text gets the full old rule.
  - Prompt wording.
- `orion/world_pulse_read/meaning_links.py` (new): lists decisions by policy, seed and time, joined to each assertion's latest decision. Reports each link as standing, withdrawn or superseded. Builds idempotent withdrawal decisions.
- `orion/world_pulse_read/operator.py`: `reread_read` puts a finished read (both stages terminal, nothing bound, no other row holding the URL) back to Stage 1 pending on the same seed row. The old handoff stays until Stage 1 replaces it.
- `orion/world_pulse_read/queue.py`: the stale digest sweep spares a pending row that already holds a read.
- Hub: settings, `.env_example`, `main.py` and the Stage 2 pipeline carry the flag. `reading_claims_recorded` logs `meaning_rule_decisions`.

## Metric gate

No new metric. The centered score is #2647's ranking score, already gated there. Here it is only copied into the decision rationale and does not gate anything new.

## Files changed

- `orion/world_pulse_read/assertions.py`: rule, prompt, context flag, endpoint score.
- `orion/world_pulse_read/concept_search.py`: `with_related_objects` stamps the score.
- `orion/world_pulse_read/meaning_links.py`: list and withdraw by policy (new).
- `orion/world_pulse_read/operator.py`: `reread_read`.
- `orion/world_pulse_read/queue.py`: stale sweep exclusion.
- `orion/schemas/world_pulse_read.py`: receipt reason comment (no field change).
- `services/orion-hub/app/settings.py`, `.env_example`, `scripts/main.py`, `scripts/world_pulse_read_stage2.py`: flag wiring, log.
- `services/orion-hub/scripts/reading_meaning_links.py`: withdraw CLI (new).
- `services/orion-hub/scripts/reading_reread_backfill.py`: backfill driver (new).
- `services/orion-hub/tests/test_world_pulse_read_meaning_target_links.py`: accept/reject matrix (new).
- `services/orion-hub/tests/test_world_pulse_read_reread_postgres.py`: reread transitions, sweeps, withdraw (new, real Postgres).
- `services/orion-hub/tests/test_world_pulse_read_reread_backfill_postgres.py`: the backfill's whole lifecycle with a stand-in worker (new, real Postgres).
- `services/orion-hub/tests/test_world_pulse_read_claim_related_objects.py`: the old strict-rule test is pinned to flag off.
- `services/orion-hub/evals/test_reading_assertion_eval.py`: meaning-rule batch.
- `services/orion-hub/evals/run_reading_link_targets_replay.py`: `--tags`, `meaning_rule` pass.
- `services/orion-hub/README.md`: rule, revert, backfill.

## Schema / bus / API changes

- Added: policy name `reading_meaning_target_v1` (a value of the existing `SubstrateGraphDecisionV1.policy`). Receipt reasons `accepted_meaning_target`, `meaning_target_predicate`, `quote_not_about_subject` (free-text `reason` field, no schema change). Withdrawal decisions use policy `<policy>_withdrawn` and actor `reading_policy_withdraw`.
- Removed / renamed: none.
- Behavior changed: with the flag on, a meaning-only claim that meets the rule becomes a provisional link. A downstream "link made" finding and a link curiosity seed follow from it, exactly as for any reading link.
- Compatibility notes: no wire field added. `findings.py` and `link_accepted_seeds.py` key on the outcome, the latest decision and the actor, not the policy, so they treat these links like any other.

## Env/config changes

- Added keys: `HUB_WORLD_PULSE_READ_MEANING_TARGET_LINKS_ENABLED=true` (orion-hub).
- Removed / renamed keys: none.
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. `/mnt/scripts/Orion-Sapienform/services/orion-hub/.env:892`.
- Skipped keys requiring operator action: none.

## Tests run

```text
RUN_READING_POSTGRES=1 PYTHONPATH=$PWD pytest -q <the orion-reading-tests.yml list>   -> 1633 passed, 6 skipped
RUN_READING_POSTGRES=1 pytest -q services/orion-hub/tests/test_world_pulse_read_*.py orion/world_pulse_read/tests \
  tests/test_no_private_eval_artifacts.py (after merging #2647's latest)              -> 473 passed
mutation checks: dropping the predicate check, the subject check, the label-not-in-text test, or the
  "offered to this read" check each fails a test; skipping the restore of a failed reread and re-reading
  "gone" pages each fail the backfill lifecycle test
scripts/check_env_template_parity.py PASS; check_settings_defaults.py orion-hub --example-drift OK;
check_metric_lineage.py --gate PASS; check_definition_drift.py --gate PASS
```

## Evals run

```text
pytest services/orion-hub/evals/test_reading_assertion_eval.py -s
  meaning_rule_recall=1.00 meaning_rule_false_links=0 meaning_unsupported_links=0
pytest <the six reading evals in orion-reading-tests.yml>    -> 37 passed
run_reading_link_targets_replay.py --until 2026-10-14T00:10:00+00:00 --tags meaning_rule --llm-url .../v1/chat/completions
  llm totals meaning_rule: reads=3 claims=1 accepted=1 accepted_meaning_rule=0 (pool-gated, one pass)
reading_reread_backfill.py snapshot -> targets=26 fetchable=19 inconclusive=4 gone=3, snapshot 872561 bytes
```

## Docker/build/smoke checks

```text
Not deployed (instructed). The backfill's run step refuses until orion-athena-hub has this code and the flag on
(it checks with docker exec).
```

## Review findings fixed

Code review ran in a subagent on `origin/feat/reading-resonance-crosswalk...HEAD`.

- Finding (blocker): the backfill would re-read nothing. Hub's success path never consumes a durable run. Live, all 26 targets still have unconsumed stage 1 and stage 2 runs, so `reread_read` refused every one with `active_durable_binding`. Dropping the check alone would have been worse: `bind_turn` hands back an unconsumed binding, so Stage 1 would resubmit the OLD run and get its cached result (no fetch, no text). The test had hidden it by consuming every run before the success path.
  - Fix: `reread_read` consumes the seed's finished runs in its own transaction (both stages are terminal by then) and returns their ids, so the next bind numbers a new attempt.
  - Evidence: the test fixtures now leave runs unconsumed, the live shape. The tests assert the new run is attempt 1 and never the old run id. Removing the consume step fails `test_reread_puts_a_finished_read_back...`. Live check: `SELECT stage, consumed_at IS NULL, count(*) FROM reading_durable_turn WHERE seed_id IN (<26>)` -> 26 unconsumed per stage.
- Finding: a read that timed out was skipped on resume and never repaired. Resume could also start a second read while that one still ran.
  - Fix: a timed-out read stays out of `done_ids`. On resume it is waited for first and never re-queued.
  - Evidence: `test_a_timed_out_read_is_waited_for_on_resume_never_requeued` checks for exactly 2 stage 1 attempts.
- Finding: `--restore-seeds` (and the restore after a failed re-read) could overwrite a row still being read.
  - Fix: `restore_seed` locks the row with `FOR UPDATE` and refuses while a stage is pending or claimed. Once both stages are terminal it consumes any finished runs.
  - Evidence: the lifecycle test restores a failed re-read.
- Finding: re-reads would jump ahead of live reads, because a re-queued row keeps its old `created_at`.
  - Fix: before each re-read, the run waits for the reading queue to be otherwise idle (`QUEUE_BUSY_SQL`, which ignores stale digest items the sweep will skip).
  - Evidence: `test_queue_gate_counts_live_work_but_not_stale_digest_items_or_the_target`.
- Finding (nit): a re-read that failed in Stage 2 left the new handoff paired with the old Stage 2 result.
  - Fix: any outcome other than `stage2_done` restores the old row. The new row is kept in `results.jsonl`.
- Finding (nit): withdrawal decisions were labelled `user_asserted`. They are now `local_inferred`.
- Finding (nit, noted rather than changed): a rollback rejection applies to the shared assertion (its id comes from the statement key). If a non-backfill read had hit `already_decided` on the same statement, it loses that link too. `list` shows every affected assertion before `--yes`.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
```

Then, from `services/orion-hub` with `PYTHONPATH=.:../..`, run the backfill:

```bash
python3 scripts/reading_reread_backfill.py snapshot          # refresh the snapshot right before running
python3 scripts/reading_reread_backfill.py run --yes         # one read at a time; resumable
tail -f /tmp/reading-reread-backfill/progress.log
python3 scripts/reading_reread_backfill.py report            # report.md + before_after.csv
python3 scripts/reading_reread_backfill.py rollback --yes --restore-seeds   # only if it must be undone
```

## Risks / concerns

- Severity: medium. Concern: a re-read is a whole new read. Stage 1 re-materializes the read's concepts (merged by identity, but it can mint new ones). It writes new journal entries. It can send Juniper a "link made" message per re-read. Mitigation: concept nodes, rows and journal entries are snapshotted. Rollback withdraws every link decided for these seeds since the run started, and restores the seed rows. It cannot delete appended journal rows or minted concept nodes (append-only stores); the report lists what ran.
- Severity: low. Concern: the replay shows 0 extra links today, and one model pass is a small sample. Mitigation: the yield gets measured on the backfill's re-reads (`links_meaning_rule` per read in `results.jsonl`).
- Severity: low. Concern: re-reads take turns on the agent lane and the reading queue. Mitigation: one at a time, only when the GPU pool is quiet and no live read is queued, and a read still unfinished after 4 h stops the run (resume waits for it).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2652 (stacked on #2647; retarget to main when #2647 merges)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
