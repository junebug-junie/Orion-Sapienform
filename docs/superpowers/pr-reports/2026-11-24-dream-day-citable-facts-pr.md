# Day dream: memories can pass the grounding gate (citable facts, a totals ref, no more cut-short nights)

Fixes #3739

## Summary

- The judge now reads each cited record's real content, not just its counters:
  - a curiosity run's finding;
  - a camera's caption, and which camera took it;
  - a chat turn's words;
  - the time of every experience;
  - how many pairs a sleep cycle declined.
- A day's totals ("75 gpu waits") now have a citable home, `dream_day_read:<day>`. `dig dream-day` saves the roundup's count lines beside its read receipt, stamped with the same read time. The run checkpoints them, the judge reads them, and the night's row keeps them.
- A reminiscence turn's reply is only a save receipt. For the judge, that receipt now reads as the note it saved (#3739). This is a third, narrow, named exception to the self-ledger partition: the note goes to the judge only. The stored evidence quote never carries it, and neither does the row's raw judge text.
- The writer's prompt (v2) puts the answer shape first. It says plainly that the judge reads only the cited records and refuses any fact they do not state. It says to prefer fewer, smaller memories and to cite `dream_day_read:<day>` for totals.
- The real cause of the 11-19 failures was a turn cut short by the clock. The 11-19 turns ran out of time mid-dig before writing any answer. A cut-short reply is now named as its own failed attempt and is never parsed. A retry tells the turn the real reason. The attempt is longer (900 s to 1200 s). A retry or a new run needs only 900 s of night left.

## Outcome moved

- Failure mode: four nights (11-18 and 11-20..22) each wrote 5 memories, and the gate refused all 5.
  - Re-judged read-only with the new resolver, 1 of 5 flips on 11-22 and 1 of 5 on 11-21.
  - In a counterfactual, two 11-22 claims that cite the totals ref pass. Each claim matches a refused candidate's own wording.
  - The rest stay refused, correctly (see Evals run).
- #3739: the 11-17 memory b6879512 used to be refused on every fact. Now one claim is left ("13 commits on main"), because the judge reads the note.
- 11-19: the turns had no answer block because the motor stopped them at its 360 s budget (`fcc_timeout`) after 60 and 30 steps. 7 of the 14 dream_day turns since 11-18 ended the same way.

## Current architecture

- `dream.day`: gather, then hold, then one unified turn, then the receipt check, then the answer parse, then the grounding judge (one call per memory, reading the cited records' text + detail), then persist.
- The detail per kind was thin. An experience gave only its `payload_json`: counters for curiosity_run, entities for vision_percept, the speaker for chat_turn.
- No record could back a day's total.
- A reminiscence turn resolved to its receipt.
- The turn's limit was 900 s. Hub keeps 330 s of a held turn for finalize, and the gate reserve is 180 s, so the motor got 360 s.

## Architecture touched

- `orion/memory/dream_day.py`:
  - `RESOLVE_SQL` with a per-kind detail. Every source table and column is named in the comment.
  - New kind `dream_day_read` for stored nights.
  - `totals_text` and `totals_resolved`.
  - `looks_like_answer` and a `raw_decode` parse.
  - A per-memory page cap for the judge (`GROUNDING_RECORDS_CHARS` 16000; 6500 per record).
- `orion/schemas/dream_day.py`:
  - `DreamDayTotalsV1`, `DREAM_DAY_TOTALS_SUFFIX` and the `dream_day_read` ref kind.
  - `DREAM_DAY_MIN_ATTEMPT_SEC` = 900.
  - Prompt version `dream_day.v2`.
- `orion/dig/__main__.py`: `dig dream-day` writes `<self>/dream_day/<day>.totals.json` before the receipt. If any digs part is UNREAD, the totals say UNREAD instead of a count.
- orion-durable-runs:
  - `dream_day_graph.py`: prompt v2; cut-short check; totals read with a `read_at` match; totals merged into ground and persist; `stored_read`; `retry_attempt_sec`; a retry note that names the real cause; raw judge text blanked for memories judged on a note.
  - `dream_day_store.py`: `resolve_refs(full, self_dir)` swaps a receipt for its note; `read_totals`.
  - `dream_day_receipts.py` (new): the judge-only reader of the reminiscence ledger.
  - `admission_runtime.py`: wiring.
  - `dream_day_poll.py`: the new-run time bound.
  - `settings.py`, `.env_example`, compose.
- `tests/test_organ_names_partition.py`: the third exception, plus a test that pins its one call site to the judge's `full and self_dir` branch.

## Files changed

- `orion/memory/dream_day.py`: resolver detail, totals helpers, tolerant parse, judge page cap.
- `orion/schemas/dream_day.py`: totals contract, ref kind, min attempt, prompt version.
- `orion/schema_skew_discovery.py`: declares the writer of `DreamDayTotalsV1` (the governor's dig).
- `orion/dig/__main__.py`, `orion/dig/README.md`, `orion/dig/tests/test_dream_day_cmd.py`: the totals file.
- `services/orion-durable-runs/app/{dream_day_graph,dream_day_store,dream_day_receipts,dream_day_poll,admission_runtime,settings}.py`
- `services/orion-durable-runs/{.env_example,docker-compose.yml}`: `DREAM_DAY_TURN_TIMEOUT_SEC` 900 to 1200.
- `services/orion-durable-runs/evals/run_dream_day_eval.py`: `--judge-candidates` re-judges a night's stored gate candidates, refused ones included. `--self-dir` sets the ledger directory.
- `services/orion-durable-runs/tests/test_dream_day_citable_facts.py` (new) and `test_dream_day_postgres.py`: real-Postgres checks of every new detail kind, the caps, the receipt swap and the stored totals ref.
- `tests/test_organ_names_partition.py`: the exception.
- `.github/workflows/orion-durable-runs-tests.yml`: path triggers (`orion/harness/cut_short.py`, `orion/dig/__main__.py`) and a dig-side step.

## Schema / bus / API changes

- Added:
  - `DreamDayTotalsV1` (a file contract, forbid).
  - Ref kind `dream_day_read:<day>`.
  - `dream_day.read` jsonb gains the optional keys `totals` and `totals_read_at`. There is no column change.
- Removed / renamed: none. No bus change. No migration.
- Behavior changed:
  - Prompt `dream_day.v2`.
  - A cut-short reply is `dream_day_cut_short`, never parsed.
  - The parse takes the last answer-shaped JSON object, fenced or bare. A printed receipt or the echoed template is not an answer.
- Compatibility:
  - A run checkpointed before deploy has no totals. Its `dream_day_read` items are dropped, which fails closed.
  - An older dig writes no totals file. Same result.

## Env/config changes

- Changed default: `DREAM_DAY_TURN_TIMEOUT_SEC=1200` (was 900): `.env_example`, settings and compose.
- Added / removed keys: none.
- Local `.env` synced with `python3 scripts/sync_local_env_from_example.py --all-keys`. The key existed with 900 (reported as diverged), so I set it to 1200 by hand in `services/orion-durable-runs/.env`. Nothing else was forced.
- Skipped keys: none.

## Tests run

```text
services/orion-durable-runs: pytest tests -q (ORION_ADMISSION_TEST_DSN = throwaway postgres:16)   1249 passed, 1 skipped (final, after review fixes)
tests/test_dream_day_citable_facts.py + test_dream_day.py + test_dream_day_postgres.py (after review fixes)   56 passed
orion/dig/tests                                                   373 passed
orion/dig/tests/test_dream_day_cmd.py + tests/test_organ_names_partition.py (after fixes)   15 passed
tests/scripts/test_schema_skew_discovery.py + partition            30 passed
tests/test_no_personal_terms.py, test_no_private_eval_artifacts.py  passed
scripts/check_integration_seam.py                                OK
scripts/check_env_key_single_source.py                           OK
scripts/check_service_env_compose_parity.py orion-durable-runs   OK
scripts/check_migration_required_by.py                           OK (no new migration)
```

New tests:
- Each detail kind's shape and cap, against real Postgres: curiosity finding (the newest finish row, never another workflow's), the camera caption, chat words, dream_cycle declines, a SQL cap on a 5,000-character finding.
- The reminiscence receipt swap, only with a ledger directory.
- The totals ref:
  - it resolves only for its own night and its own `read_at`;
  - it is stored on the row and resolves again from the row;
  - its short text is unchanged.
- The prompt carries the rule, the ref and the shape first.
- The real 11-19 reply shape gives `dream_day_cut_short`. Fenced, unfenced, trailing-prose and unclosed-brace answers parse.
- Retry time:
  - a retry runs with 1020 s left on a 1200 s brief and is skipped with 600 s left;
  - the tick submits with 950 s left.
- The retry note names the real cause.
- The page cap.
- A partial digs read gives UNREAD in the totals, never a count.
- A note-judged memory keeps no raw judge text.

## Evals run

Read-only, no rows written. Same resolver, prompt and parser as the run, on route `memory_distill` through the gateway's HTTP door.

`python3 services/orion-durable-runs/evals/run_dream_day_eval.py --day 2026-11-22 --judge-candidates`:

```text
#0 UNSUPPORTED -> UNSUPPORTED  "75 gpu waits" (+ "the run closed explored"); cites one gpu_wait: a total needs the totals ref
#1 UNSUPPORTED -> UNSUPPORTED  was 6 claims, now 2: "SelfDefinition v100" (record says "revises v98"), "sent the reach-out"
                                (record says reach_out flag true, not that it was sent). Ogden / 318 parts / GGUF now backed.
#2 UNSUPPORTED -> SUPPORTED    1.257 / 8.489: the curiosity run's finding now reaches the judge
#3 UNSUPPORTED -> UNSUPPORTED  cites two office-camera frames (cam0, 11-23) for a walkway vehicle and three PRs: wrong records
#4 UNSUPPORTED -> UNSUPPORTED  "three hub.promise acts" is a total; "still unkept" / "no promise_kept" stated nowhere
1/5 flipped
```

11-21 (`--day 2026-11-21 --judge-candidates`): 1 of 5 flipped (#4, the substrate.perception run). #3 dropped from 4 refused claims to 1 ("later proved to mint false zeros"). The reminiscence candidate now shows its note to the judge and is still refused: "stood in the 2025-11-17 checkout, compared to deployed main" is not what the note says.

11-17 (`--day 2026-11-17 --judge`, #3739 check): b6879512 used to fail on every claim. Now only "13 commits on main" is refused (the note says 27 commits for that week). bb1b5a42 is still refused for the camera, kids and chat-count claims, as before.

Counterfactual, read-only. The totals text is built from the roundup file the 11-22 run's own read wrote (09:39Z):

```text
"My body queued hard on 2026-11-22: 75 gpu waits, the longest 8,177 seconds ..."  [gpu_wait, dream_day_read]  SUPPORTED
"On the 22nd three hub.promise acts ran that no organ claims."                    [dream_day_read]           SUPPORTED
candidate #0 + dream_day_read                                                     UNSUPPORTED (only "I answered my own question about ... reach a GPU" left)
```

Honest read: the detail fix alone flips about 1 in 5. Most of the remaining refusals are correct: wrong record, a total with no totals ref, or a claim no record makes. The totals ref and the v2 prompt have to change what Orion writes before more pass. That shows only after the next live nights (#3738).

## Docker/build/smoke checks

```text
scripts/check_service_env_compose_parity.py orion-durable-runs  OK (compose default 1200)
```

No image build: no new dependency. No live deploy from this branch. The live claims above are UNVERIFIED until the next night runs.

## Review findings fixed

Review by a subagent (orion-repo-agent) on the first commit. It ran EXPLAIN ANALYZE on the new SQL live: every subquery is indexed, 270 ms cold for 9 ids.

- Finding (material): the totals stated "0 in background runs" when the digs half was UNREAD, so a false count became citable.
  - Fix: when any digs part is UNREAD, the totals carry the UNREAD lines instead of a count.
  - Evidence: `test_dream_day_says_unread_in_page_and_receipt`.
- Finding: the judge's raw reply could quote a reminiscence note into `dream_day.grounding`, which leaks past the partition.
  - Fix: `raw` is None for any memory judged on a swapped-in note.
  - Evidence: `test_a_memory_judged_on_a_reminiscence_note_keeps_no_raw_judge_text`.
- Finding: a non-UTF-8 ledger raised ValueError, so every memory that night went unjudged.
  - Fix: catch `(OSError, ValueError)` and log it. The judge then reads the receipt.
- Finding: the retry note always said "ran out of time".
  - Fix: the note names the cause from `last_error` (cut short, not read, no answer), and otherwise says "saved nothing".
  - Evidence: the prompt test.
- Finding: an echo of the prompt's own template would parse as an answer.
  - Fix: `looks_like_answer` refuses the template's placeholders.
  - Evidence: `test_a_cut_short_reply_is_its_own_failed_attempt_never_parsed`.
- Finding (nit): the fixture's `substrate_durable_run_state` types differed from live (varchar).
  - Fix: the fixture matches live.
- Finding (nit): the CI step duplicated tests the suite already runs.
  - Fix: the step now runs only the dig-side test.

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Run this from the clean primary `main` checkout after merge; the post-pull rebuild does it. Deploy outside the dream window (01:00-03:00 Denver). Order:

1. orion-durable-runs: reads the new file and the new kind.
2. The FCC sandbox sync, which the post-pull run does after the rebuild, so `dig dream-day` writes the totals file.

In either order nothing is written wrongly. Before both have landed, `dream_day_read` items are simply dropped. No migration.

## Risks / concerns

- Severity: medium. Concern: the totals file sits in a directory the turn's own Bash can write. A turn could forge `<day>.totals.json` with its receipt's `read_at` and back its own counts. It is the first grounding evidence that does not come from Postgres. Mitigation: the `read_at` match ties the file to this read. The dig writes it, and the turn has no reason to write it by hand. Follow-up: cross-check the headline experience count against `temporal_self_event` (issue below).
- Severity: low. Concern: a turn that wrote a full answer block and was then cut short during a trailing dig is now refused and retried, not parsed. Mitigation: this fails closed. The prompt now asks for the answer straight after a few digs.
- Severity: low. Concern: the third partition exception widens who reads reminiscence words. It is pinned by a test to one call site: the judge's read. The stored quote and the row's raw judge text never carry them. Claims the judge lists come from the memory's statement.
- Severity: low. Concern: the motor budget rises from 360 s to 660 s, so a turn holds the agent GPU longer. Mitigation: it is a cap, not a cost. Answered turns used 298-346 s.
- Issues closed: #3739. Opened: #3793 (totals cross-check). #3738 (7-night acceptance) still tracks live proof.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3798
