## Summary

Orion can now look up their own memories on demand, and is told to do so before saying "I don't remember". Recall now also writes down, for every chat turn, whether memory search ran and why it brought nothing back. (Recall rev 2, step 3.)

- New command `python -P -m orion.dig know <words>`. It asks orion-recall, which runs the same memory search a chat turn gets, on Orion's own words. It prints up to 8 short hits: kind, record id, date, statement.
  - **Strong** hits cleared recall's bar.
  - **Weak** hits are below the bar but share a word with the search. They are shown and labelled, so a partial match is never hidden behind "nothing found".
  - A degraded search (exact words only) says so. A search that could not run says "this is NOT the same as having no memory".
- New orion-recall endpoint `POST /lived/know`. Each lookup writes one `recall_telemetry` row (verb `dig.know`, `corr_id` = the turn id, `selected_ids` = the strong hits), so a memory Orion looks up counts toward read demand. One lookup runs at a time; a second gets `busy`, never an empty answer.
- One prompt rule, added to the dig-kit line that is already in every chat turn's motor prompt: "Before telling Juniper you don't remember or have no record of something, run `orion.dig know <words>`."
- `recall_telemetry` gets two new columns. `lived_status` records why memory search injected what it did: `ok`, `below_bar`, `skipped_short`, `degraded`, `timeout`, `error` or `off` (NULL = lived recall did not run on that call). `lived_trace` records the rewritten searches, every candidate id that reached the reranker, the injected ids and the backend.

Fixes #3535
Fixes #3628

## Outcome moved

- **False "no memory" can now be counted.** Before, a turn where recall never searched (too few words, timeout, error) looked the same in the table as a turn where it searched and found nothing. Now each turn's row says which one happened.
- **Orion has a way to look before denying.** On 2026-11-14, Orion told Juniper it had no memory of something that `episode_memory` held. A by-hand run of `dig know` with the words of that turn returns the target record (proof below).

## Current architecture

- Lived-memory recall (steps 1 and 2: PR #3571, PR #3622) runs once per chat turn inside orion-recall's continuity phase. Messages with fewer than 5 content words are not searched. Results land in the prompt's DURABLE MEMORY block.
- `recall_telemetry` held `backend_counts.lived_memory` (a count) and lived timings. Why nothing was injected lived only in `recall_debug`, which is sent back in the reply but never stored.
- The dig kit had no way to search memories. The motor prompt's dig line pointed only at body, history and camera commands.

## Architecture touched

- orion-recall: new HTTP endpoint; lived reader gets a deliberate-lookup mode; per-call status and trace written to `recall_telemetry`.
- Dig kit (runs in the harness governor and Hub; Orion's turns import it from the FCC sandbox): new `know` command, one more sentence in the always-on dig line.
- Postgres: two additive, nullable columns on `recall_telemetry`.
- No bus channel, schema registry or env change. `RecallDecisionV1` (extra=forbid) is unchanged: the status rides in the existing `recall_debug` and is read out at insert time.

## Files changed

- `services/orion-recall/app/lived_memory.py`: `recall()` takes `min_query_tokens` / `rewrite` overrides; records every candidate that reached the rerank; new `know()`, `status_of()`, `trace_of()`.
- `services/orion-recall/app/worker.py`: marks backstop timeout/error (`outer`), stores `status` on the lived debug, writes `lived_status` / `lived_trace`.
- `services/orion-recall/app/main.py`, `app/http_models.py`: `POST /lived/know`.
- `services/orion-recall/sql/recall_telemetry.sql`, `services/orion-sql-db/manual_migration_recall_telemetry_lived_status_v1.sql`: the two columns.
- `orion/dig/know.py`, `orion/dig/__main__.py`: the `know` command and its help line.
- `orion/dig/brief.py`: the look-before-you-deny sentence (brief cap 520 -> 640 chars).
- `services/orion-recall/README.md`: endpoint, statuses, a 14-day count query.
- `services/orion-hub/static/catalog/orion-abilities.json`: a "Do you remember...?" entry for `know` (CI: every dig command needs one).
- `services/orion-recall/evals/run_dig_know_bank_eval.py`: live eval over the host-local bank.
- Tests: `services/orion-recall/tests/test_lived_know_and_status.py`, `evals/test_dig_know_bank_eval.py`, `orion/dig/tests/test_know_cmd.py`, updates to `tests/test_recall_telemetry_persist.py` and `orion/dig/tests/test_dig.py` (help cap 4000 -> 4250).
- `.github/workflows/lived-memory-recall-tests.yml`: runs the new tests.

## Schema / bus / API changes

- Added: `POST /lived/know` on orion-recall (`{"words", "turn_id"?, "max_hits"?}` -> `{"status", "backend", "degraded", "hits": [{strength, kind, purpose, id, date, text, fit}], "unverified", "corr_id"}`; status adds `busy` and `unverified` to the list below). Columns `recall_telemetry.lived_status text`, `recall_telemetry.lived_trace jsonb`.
- Removed: none.
- Renamed: none.
- Behavior changed: chat-turn lived recall is unchanged, except its debug now carries `candidate_ids`, `status` and (after a backstop timeout/error) `outer`.
- Compatibility notes: new columns are nullable; rows before this deploy read NULL. orion-recall's own boot DDL also adds the columns if missing (information_schema check first, bounded lock), and inserts only columns it confirmed.

## Env/config changes

- Added keys: none. (`ORION_DIG_RECALL_URL` is an optional override read by the dig client; not in any `.env_example`, because the defaults already reach recall from both places the motor runs.)
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change).
- skipped keys requiring operator action: none.

## Tests run

```text
cd services/orion-recall && PYTHONPATH=.:<repo> python -m pytest -q -p no:cacheprovider \
  tests/test_lived_memory.py tests/test_lived_memory_step2.py evals/test_lived_memory_replay_eval.py \
  tests/test_lived_know_and_status.py tests/test_recall_telemetry_persist.py evals/test_dig_know_bank_eval.py
  -> 98 passed (after review fixes)
python -m pytest -q -p no:cacheprovider orion/dig/tests   -> 254 passed (after review fixes)
python -m pytest -q -p no:cacheprovider orion/harness/tests   -> passed (dig brief in the prefix)
services/orion-recall/tests (whole folder): 2 failures, both also fail on clean main
  (test_recall_policy_harness vector gating, test_recall_vector_amputation import); not touched here.
scripts/check_metric_lineage.py --gate / --prompt-semantics: PASS
scripts/check_definition_drift.py --gate: PASS (no definition changes)
scripts/check_integration_seam.py --require-base: OK
scripts/check_migration_required_by.py --base origin/main: OK
scripts/check_env_template_parity.py: PASS
scripts/apply_post_pull_migrations.py --dry-run (new migration): NOT APPLIED, "would apply (single transaction, lock_timeout=10s)" -> fits the auto-apply allowlist
```

## Evals run

`dig know` against live data, before deploy: the branch's endpoint was served from a throwaway container on `app-net` (no bus consumer, telemetry write replaced by a print), reading the live stores read-only. `python services/orion-recall/evals/run_dig_know_bank_eval.py` over the host-local bank (`~/.orion/evals/lived_memory_recall_bank.json`), with plain lookup words per item (host-local, not committed):

```text
found the target for 8 of 9 items (bar: 5) -> PASS
  el Coco (the 2026-11-14 false-denial turn): target strong, rank 3
  birthday: 3 targets strong, ranks 1/2/4
  hecate V100: strong, rank 4
  Zelda (game name): strong, rank 5
  beat the Zelda game: weak, rank 6
  lost memory / storage failure: strong, rank 1
  two dogs (person fact): strong, rank 1
  walkway camera IP: strong rank 1, two more weak (ranks 5, 7)
  cooling baseline: MISS (the memory is worded around the AC and the season, not "cooling baseline")
latency per lookup: 0.3-0.8 s (no rewrite)
```

End-to-end `python -P -m orion.dig know el Coco` from the harness-governor image against the throwaway endpoint: printed 8 strong hits, the 2026-11-11 target third; the would-be telemetry row had `corr_id` = the turn id from `ORION_TURN_CORRELATION_ID`, verb `dig.know`, `lived_status` `ok`. With "Coco stitches" the same target came back as a weak hit (rank 3), showing the weak label doing its job.

**Turn replay: not run.** A safe eval path exists (Hub `POST /api/chat`, `no_write`, eval session, as PR #3603 does), but it would exercise the deployed code, which has neither `dig know` nor the prompt rule. Replay after deploy. UNVERIFIED until then: that Orion actually runs `know` before denying.

Reachability: Hub (host network) reaches recall at `http://127.0.0.1:8260` (`RECALL_SERVICE_URL`, already set); the harness governor (`app-net`) reaches `http://orion-athena-recall:8260` (both checked with `/health`).

### Latency note (report only, budget not changed)

Real Hub turns (`stance_react`, eval sessions excluded):

```text
since step 2 deploy (18:01Z, 8 turns):  recall total p50 4.5 s, p90 5.2 s, max 5.6 s
                                        lived reader p50 4.0 s, p90 4.8 s, max 5.2 s (budget 5.5 s)
                                        0 over budget, 0 degraded, 0 deadline hits
step 1, 06:30-18:00Z (84 turns):        recall total p50 3.5 s, p90 4.5 s; lived wait p90 3.8 s
eval session, same window (33 searched): lived p50 2.1 s, p90 3.4 s
```

The budget is not being hit, but headroom on real turns is thin: about 0.7 s at p90 and 0.3 s at the max, from 8 turns. The rerank is the big piece on real turns (p90 3.4 s, against 1.9 s on the eval session). Not broken, so not changed. Recheck once `lived_status` is live: a `timeout` or `degraded` count above zero is the signal.

## Docker/build/smoke checks

```text
No image build or deploy (instructed). Live read-only smoke as above, from the
running orion-recall image with the worktree mounted read-only.
```

## Review findings fixed

Review by a separate agent (by hand; read-only). All material findings fixed in `80e3698c2`.

- Finding: if the extra serving-gate re-check failed, `know` returned `below_bar`, and the CLI printed "No stored memory", a failed check shown as no memory.
  - Fix: new status `unverified` and an `unverified` count; the CLI says the matches could not be checked and is NOT no memory. The re-check is time-bounded; `know_recheck_error` goes in the trace.
  - Evidence: `test_know_a_failed_recheck_is_unverified_never_below_bar`, `test_render_unverified_matches_are_said`.
- Finding: a degraded search with no hits printed both "a miss here does not mean no memory" and "No stored memory".
  - Fix: no "No stored memory" line when degraded; degraded hits are labelled "no bar was applied".
  - Evidence: `test_render_degraded_empty_never_says_no_stored_memory`, `test_render_degraded_hits_are_not_called_above_the_bar`.
- Finding: `dig.know` rows put weak hits in `selected_ids` and in `injected_ids`.
  - Fix: `selected_ids` and `injected_ids` = strong only; weak ids in `lived_trace.weak_ids`. README says only strong hits count toward read demand.
  - Evidence: `test_know_endpoint_returns_hits_and_logs_the_turn`.
- Finding: the reply waited on the Postgres insert, so a slow insert could turn a finished lookup into "could not run".
  - Fix: the row is written in a background task (reference kept).
  - Evidence: endpoint test still sees the persisted decision.
- Nits fixed: re-check dropped ids persisted (`recheck_dropped`) and asserted in the gate test; ad-hoc calls with no turn id tagged `adhoc:<uuid>` / `lived_memory.know.adhoc`; one lookup at a time (`busy`); reader off returns status `off` (200) instead of a bare 503.
- Nit not changed: dig help is 4038 chars, so the help cap moved 4000 -> 4250. Help loads only on demand, not every turn.

## Restart required

Apply the migration, then rebuild orion-recall. The post-pull rebuild does both on its own from the clean main checkout (the migration fits the auto-apply allowlist). By hand:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_recall_telemetry_lived_status_v1.sql
scripts/safe_docker_build.sh orion-recall up -d --build
```

The dig command and prompt rule reach Orion's turns when the FCC sandbox syncs after the deploy, and the harness governor / Hub pick up `orion/dig/brief.py` on their next rebuild.

## Risks / concerns

- Severity: low. Concern: weak hits are anything below the bar that shares one word with the search; with common words this can list loosely related memories. Mitigation: they are labelled "check before relying on one", capped at 8 total, and strong hits always come first.
- Severity: low. Concern: `dig.know` rows join the read log. Read-demand counts (#3589) must decide whether a lookup counts as "opened". Mitigation: distinct verb `dig.know`, strong hits only in `selected_ids`, ad-hoc and eval runs tagged (`adhoc:` / `eval:dig-know`).
- Severity: medium. Concern: the prompt rule is one sentence; whether Orion follows it is UNVERIFIED until a post-deploy replay. Mitigation: the scoreboard (#3536, PR #3603) counts false "no memory" answers; `lived_status` now lets that count be split by cause.
- Issues opened: none yet. Issues closed: #3535, #3628 (by this PR).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3645

🤖 Generated with [Claude Code](https://claude.com/claude-code)
