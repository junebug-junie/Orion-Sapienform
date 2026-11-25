## Summary

Reveries that ask, Patch 3 (spec `docs/superpowers/specs/2026-11-18-reverie-questions-design.md`, "Threads (idea 9) and read-back"). Approved by Juniper 2026-11-25.

- A reverie question that Hub turned away only because the curiosity focus slot was full now **waits** and is offered again when the slot is free. Before, it was lost (2 of 4 questions on 2026-11-25). The re-offer makes no model call and does not spend the quiet stretch. The 3-a-day focus cap and the 14-day near-duplicate rule still hold. A question that waits 14 days lapses.
- When the curiosity run that took a reverie question ends, the question's **outcome** is filled in: what the run decided, the finding's register, whether the run said to continue, and the run's own note on what is still open.
- A dig that said "continue" makes its **thread** the next reverie's seed, before the picker rotates kinds. The prompt shows the earlier question, what the dig found, and what is still open.
- Threads close and the status is queryable in a new `reverie_thread` table: `concluded`, `exhausted`, `lapsed`.
- One kill switch: `ORION_REVERIE_THREADS_ENABLED=false` gives back Patch 1 behavior.

## Outcome moved

Failure mode fixed: a good reverie question was dropped whenever another focus was pending (`keep_pending`). Read-back: `reverie_question.outcome` was always NULL. It now holds the dig's result, which the day letter and situation brief already render (`orion/reverie/questions.py outcome_words`).

## Current architecture

orion-thought polls every 60 s. Quiet gate, then seed picker (least recently used kind), then one `reverie_question` model call, then code checks, then one `reverie_question` row. An accepted question goes to Hub's focus slot over the introspect RPC. Hub refuses while any focus is pending. The refusal was stored as `routed_to='none'` with Hub's words in `route_note`, and nothing ever looked at the row again. `outcome` existed as a column but had no writer.

## Architecture touched

- orion-thought only (reverie worker). No new service, scheduler, or bus channel.
- Each poll now starts with a cheap step: (a) read back finished digs, (b) record thread statuses, (c) re-offer the oldest waiting question if the slot is free. Then the normal quiet gate and reverie run, with a continuing thread as the seed when there is one.
- Read-only reads of `curiosity_focus_request`, `curiosity_finding_register`, `substrate_durable_run_state` (finish detail) and `substrate_attention_schema` (`predicted_next`).

Where `continue_note` comes from: the run's `:TurnOutcome.continue_note` lives in the worldview graph, and orion-thought has no graph reader. orion-durable-runs mirrors that same note into `substrate_attention_schema.predicted_next` (`orion/curiosity/attention_schema.py`, entry `curiosity-<run_id>`). Patch 3 reads it there. `continue_line` comes from the run's finish state in `substrate_durable_run_state.detail`.

## Files changed

- `services/orion-thought/app/reverie_threads.py`: new, pure. Slot-full refusal check (Hub's own `PENDING_KEPT`/`JUNIPER_PROTECTED` words), re-offer pick, outcome builder, thread status rules and the continuation pick. Holds the one-line fold-in note for the inquiry ledger (#3442).
- `services/orion-thought/app/reverie.py`: the poll runs read-back, thread statuses and re-offer first. Thread-first seed. The plan request carries the thread context. A slot-full refusal stores `offer_status=waiting`.
- `services/orion-thought/app/store.py`: new reads and writes (adopt old refused rows, waiting rows, slot check, re-offer near-duplicate texts, reconcile SQL, outcome write, thread rows, thread upsert). The cap now counts `COALESCE(routed_at, created_at)`. The seed picker never joins a closed thread.
- `services/orion-thought/app/settings.py`, `.env_example`: `ORION_REVERIE_THREADS_ENABLED=true`.
- `orion/cognition/prompts/reverie_question.j2`: thread block (shown only for a continuing thread).
- `services/orion-sql-db/manual_migration_reverie_question_threads.sql`: new migration.
- `scripts/sql/2026-11-25_grant_orion_readonly_reverie_thread_table.sql`: read grant for Orion's sandbox. NOT APPLIED, needs Juniper.
- `services/orion-thought/tests/test_reverie_threads.py` (new), `tests/test_reverie_questions.py` (fake signatures).
- `services/orion-thought/README.md`, spec "As built" note.

## Schema / bus / API changes

- Added: `reverie_question.offer_status` (`waiting | sent | lapsed | dropped`, NULL = never waited or sent), `reverie_question.routed_at`, `reverie_question.outcome_at`. New table `reverie_thread (thread_id PK, thread_key, seed_kind, status, status_reason, last_question_id, last_activity_at, status_at, opened_at)`.
- Removed / renamed: none.
- Behavior changed: `outcome` jsonb is now written (`dug, focus_status, run_id, disposition, finding (register), register_reason, novelty, continue_line, continue_note, finding_excerpt, finish_found`). The keys `disposition`, `finding` and `continue_note` match what `outcome_words` already renders.
- Compatibility: `ReverieQuestionV1` and `orion/reverie/questions.py` COLUMNS are unchanged, so Hub, cortex-exec and durable-runs read the same columns as before. REQUIRED-BY: `orion-thought` only (the only reader of the new columns and table). Additive and on the post-pull auto-apply allowlist (checked: `auto_apply_refusals = []`). There is no index on `offer_status`, because an index on an existing table would need CONCURRENTLY.
- No bus changes.

## Env/config changes

- Added keys: `ORION_REVERIE_THREADS_ENABLED=true`, `ORION_REVERIE_THREAD_MAX_DIGS=3` (orion-thought).
- Removed / renamed: none.
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (key present in the primary checkout's `services/orion-thought/.env`).
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-thought: pytest tests -> 618 passed, 31 skipped
  (new tests/test_reverie_threads.py: 25 passed. Covers: re-offer when the slot frees; no model call and no
   stretch spent on re-offer; oldest first; cap holds; lapse; near-duplicate at re-offer -> dropped; turned
   away again keeps waiting; cap refusal does not wait; flag off = Patch 1; outcome from fixtures (ran+finish,
   finish grace, pending/taken/expired/missing); reconcile writes finished only; concluded / exhausted /
   lapsed; follow-up already asked; thread book writes only changes and skips closed; thread-first before
   the picker; prompt and plan carry the thread)
  Mutation check: disabling the slot check, the cap check, or any of the three close rules fails 1-2 tests each.
orion/orion_day/tests orion/situational/tests orion/temporal_self/tests -> 827 passed, 1 skipped
services/orion-hub reverie + curiosity focus tests -> 84 passed
tests/test_deploy_migration_gate.py tests/test_no_personal_terms.py tests/test_post_pull_apply_migrations.py -> 197 passed
tests/test_curiosity_urgent_prompt.py orion/substrate/tests/test_signal_history.py (grant files) -> 28 passed
scripts/check_migration_required_by.py --base origin/main -> OK (1 new migration file)
scripts/check_env_template_parity.py -> PASS; check_env_key_single_source.py -> OK
scripts/check_integration_seam.py --require-base -> OK; check_verb_physiology.py -> OK
check_scripts_dir_no_stdlib_shadow.py -> clean; git diff --check -> clean
orion/integration/evals (now-ablation fixtures render reverie_question.j2) -> pass; run_now_ablation_eval --offline -> PASS
  (first CI run failed here: the new thread block added one blank line to the non-thread prompt; fixed with
   whitespace so the prompt is byte-identical when no thread is shown)
```

## Evals run

```text
No new eval. The quality question (are reverie questions worth digging) is Juniper's sample read, ~2026-12-02 (#3984).
Store SQL run end to end against a throwaway postgres:15 holding the live schema (pg_dump -s) and a copy of
the live rows: migration applied twice (idempotent); adopt 2 -> 0 on second run; waiting = the 2 turned-away
rows; slot pending = True -> decision slot_full; slot free -> offers 1d7bb8a5 (15:58); near-duplicate check
False for both; reconcile writes outcomes for both dug questions; thread statuses written 12, second pass 0
changes; continuation = thread ea26513a (the af2666143649 dig); cap count 2 -> 3 after a re-offer.
```

## Docker/build/smoke checks

```text
Live read-only smoke (BEGIN READ ONLY ... ROLLBACK on conjourney, using the PR's own SQL):
- Rows the adopt step turns into waiting, in re-offer order:
    1d7bb8a5 2026-11-25 15:58:27  "Did Juniper return and speak after the brief step-away ..."
    d62ec863 2026-11-25 17:23:50  "Which self-bets in the 0.8-0.9 bin for turn.self.fetches_outside ..."
- Focus slot right now: 6ab4ebeb (self, 14:08) is pending, so neither is offered until a run takes it.
- store.RECONCILE_SQL verbatim:
    73f99e85 ran af2666143649 explored operational novelty 0.155 finish=t continue_line=true
             note "Two open ends, both about what the restored corpus IS: ..."
    8da16a7b ran 6eed301f7743 explored operational novelty 0.231 finish=t continue_line=true
             note "Explored the focus and the arrow reversed. Still unknown: (1) ..."
    e0bdc190 pending -> no outcome yet (correct)
No docker build run: no dependency, Dockerfile, or compose change.
```

## Review findings fixed

#3997 was merged before the review findings were fixed, so the fixes ship in a follow-up PR from the same branch (link below).

- Finding: a thread crowds out the other seeds. Live, 77% of curiosity finishes set `continue_line=true`, so one thread could hold the reverie for days.
  - Fix: a thread closes `exhausted` after `ORION_REVERIE_THREAD_MAX_DIGS` (3) dug questions. A continuation goes first on at most every other quiet stretch: if the newest reverie row was a continuation (`store.last_reverie_was_continuation`), the picker goes first, and the thread is used only when the picker has no material.
  - Evidence: `test_a_thread_is_exhausted_after_three_dug_questions`, `test_a_continuation_goes_first_on_at_most_every_other_stretch`, `test_the_alternate_stretch_falls_back_to_the_thread_when_the_picker_has_nothing`.
- Finding: an outcome written on the 30-min grace miss was permanent.
  - Fix: `RECONCILE_SQL` also returns rows whose outcome has `finish_found=false` once a finish row exists. `write_reverie_outcome` may replace only such an outcome. The first write keeps the 30-min grace. (Live example of the miss: run `a32417ee8a57` is `ran` with no finish row.)
  - Evidence: `test_an_outcome_written_before_the_finish_row_is_rewritten_when_it_lands`, `test_write_outcome_may_replace_only_a_finish_missing_outcome`.
- Finding: a waiting question showed as "kept, not sent".
  - Fix: `orion/reverie/questions.py` COLUMNS gains `offer_status`, and `routed_words` renders "waiting for the focus slot" / lapsed / dropped. The day letter gets the field through `ReverieQuestionDayV1.offer_status`. This is an additive field on a forbid model, so deploy orion-hub and orion-durable-runs together. The Hub tab (`reverie_routes.question_view`, `reverie-tab.js`) shows an amber "waiting for the focus slot" badge. The situation brief selects only question text, so it never showed routing. REQUIRED-BY is now `orion-thought, orion-hub`.
  - Evidence: `test_routed_words_say_waiting`, hub `test_question_view_says_waiting_for_a_waiting_question`.
- Finding: re-offering a question whose focus had expired looped until the question lapsed.
  - Fix: new constant `orion/curiosity/focus.py QUEUED_EARLIER`, used by Hub's refusal text. A re-offer refused with it is marked `lapsed`.
  - Evidence: `test_a_reoffer_refused_as_queued_earlier_lapses_instead_of_looping`, `test_hubs_queued_earlier_refusal_uses_the_constant`.
- Finding: the adopt sweep wrote on every poll.
  - Fix: it now runs once per process start (`_SWEEP_DONE`).
  - Evidence: `test_the_adopt_sweep_runs_once_per_process`.
- Finding: the lapse loop compared naive and aware timestamps.
  - Fix: `_aware(created_at)` in the lapse loop and in `_question_from_row`.
  - Evidence: `test_the_lapse_check_takes_a_naive_timestamp`.

Re-run after the fixes:

```text
services/orion-thought tests -> 628 passed, 31 skipped (test_reverie_threads.py 35)
services/orion-hub reverie/focus/orion_day tests -> 191 passed
orion/orion_day orion/situational orion/temporal_self orion/integration/evals orion/curiosity tests -> 1159 passed, 30 skipped
run_now_ablation_eval --offline -> PASS; migration REQUIRED-BY OK; auto_apply_refusals [] (REQUIRED-BY orion-thought, orion-hub)
deploy-migration-gate / personal-terms / post-pull tests -> 197 passed; env parity PASS; env single source OK; integration seam OK
New env key ORION_REVERIE_THREAD_MAX_DIGS=3 synced to local .env
```

## Restart required

Merge, then the post-pull rebuild applies the migration (additive, allowlisted) and rebuilds orion-thought. By hand, if needed:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_reverie_question_threads.sql
scripts/safe_docker_build.sh orion-thought up -d --build
```

Optional, needs Juniper's yes: `scripts/sql/2026-11-25_grant_orion_readonly_reverie_thread_table.sql`.

## Risks / concerns

- Severity: medium. Concern: both live digs said continue. After deploy, the next two quiet stretches go to those two threads before any new kind. That is the spec, but it shifts the ~12-02 sample (#3984) toward follow-up questions. Mitigation: `ORION_REVERIE_THREADS_ENABLED=false` returns Patch 1 behavior. Thread rows are tagged in `reverie_thread`, so the sample can tell them apart.
- Severity: low. Concern: re-offer runs on every poll, not only in a quiet stretch. Delivering an already-asked question is not a reverie, so it is not gated on quiet. Mitigation: it still needs a free slot and cap room. A Hub refusal for any reason but a full slot is retried at most every 15 min.
- Severity: low. Concern: "an open item" (spec) is not a separate signal. Only `continue_line` continues a thread. `continue_note` comes from `substrate_attention_schema.predicted_next` (a mirror of the graph note, capped by its writer), not from the graph.
- Severity: low. Concern: `exhausted` counts a code-refused question like a `none_reason` (spec says `none_reason` twice). Without this, a thread whose follow-ups keep failing the checks would be retried every stretch for 14 days.
- UNVERIFIED: live re-offer, live outcome write, and a live thread continuation. They need the deploy plus a freed slot and a quiet stretch.
- Issues opened: none. Issues closed: none. Related: #3984 (Juniper's ~12-02 quality read).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3997

🤖 Generated with [Claude Code](https://claude.com/claude-code)
