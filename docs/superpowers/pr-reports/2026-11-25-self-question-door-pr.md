# Self-question door (self-reflection drift, Patch 1)

## Summary

- Every question Orion writes for its own self-inquiry (a sitting's `:SelfQuestionMint`, and the self-monitor's
  "Why?" question) is now stored `door = 'held'`. A held question is never picked for a sitting, never shown as
  an open lived question, and never listed as an open self-question by the curiosity introspect tool.
- A new Hub loop (`self_question_door`) asks the same local judge that labels findings (prompt v2, Flash-Next,
  only a model that passed the eval counts) what each held question is about. Philosophical -> `admitted`
  (pickable as before). Operational -> `routed`: offered to the investigate line as a self focus when the slot
  is free. A bad judge reply stays held and becomes `unjudged` after 3 tries; it is never admitted or routed
  by default.
- After a sitting: when the finding register labels a self-inquiry finding operational, the admitted Orion
  question it answered becomes `demoted` (with the run that exposed it) and is routed. This is the trace for
  dressed questions.
- Juniper's questions never pass the door.
- Ships the one-time backfill for the 88 existing Orion rows (Juniper runs it), a replay eval, a read-only
  cross-organ share eval, and the unlabelled ~100-item anchor set for Juniper to label, with its scorer.

Spec: `docs/superpowers/specs/2026-11-25-orion-self-reflection-drift-design.md` (PR #3908). Juniper's answers
to its Missing questions are recorded on the spec's own branch (commit `d0694698e` on
`docs/self-reflection-drift-design`), not here, so the two PRs do not conflict on the file.

## Outcome moved

The self-reflection loop's entry point. Before: a new Orion question went to the front of the pool (never-asked
first), whatever it was about; an operational Orion question never produced a philosophical answer (0 of 21,
spec section 2). After: only an admitted question can reach a sitting. Acceptance checks 1-6 in the spec are
the live measure (14 days after deploy). UNVERIFIED live until deploy.

## Current architecture

`orion/curiosity/self_question_pool.py` merged the seed with `curiosity_self_questions` and picked any
non-parked row. `_upsert_orion_minted_questions` (Hub, at a self-inquiry finish) and the self-monitor minter
inserted Orion rows straight into the pool. The finding register labelled findings but nothing read the label
back onto the question.

## Architecture touched

- `orion/curiosity/self_question_pool.py`: `door` on `SelfQuestion`; `passes_door`; pick/eligibility skip
  non-admitted Orion rows (kill switch respected); `UPSERT_MINT_SQL` holds new rows and re-holds new text
  (demoted stays demoted); new `INSERT_HELD_SQL` for the self-monitor.
- `orion/curiosity/register.py`: `QUESTION_LINE` (no prompt change, sha unchanged).
- `orion/curiosity/focus.py`: door focus provenance (`DOOR_CORRELATION_PREFIX`, `is_door_focus`) in the prompt
  and the focus story; the door's reason is shown as Hub's words; Prime's `keep_pending` queue may replace a
  pending door focus.
- `orion/curiosity/self_inquiry.py`: `pinned_floor_overdue(door_enabled=...)`.
- Hub: new `scripts/self_question_door.py`; `curiosity_investigation.py` (pick, open lived list, demotion
  after the register row); `self_monitor_questions.py` (held insert); `curiosity_introspect_listener.py`
  (door filter); `main.py` (loop start/stop); settings, `.env_example`, README.
- SQL: `services/orion-sql-db/manual_migration_curiosity_self_question_door_v1.sql` (ADD COLUMN only,
  `REQUIRED-BY: orion-hub`, passes the post-pull auto-apply allowlist).

## Files changed

- `orion/curiosity/self_question_pool.py`: the door in the pool.
- `orion/curiosity/register.py`: `QUESTION_LINE`.
- `orion/curiosity/focus.py`: door focus provenance; Prime may replace a door focus.
- `orion/curiosity/self_inquiry.py`: kill switch through the pinned floor check.
- `orion/curiosity/tests/test_self_question_pool.py`: door pick rules.
- `services/orion-hub/scripts/self_question_door.py`: the loop, demotion, routing.
- `services/orion-hub/scripts/curiosity_investigation.py`: pick, open-lived list, demotion hook.
- `services/orion-hub/scripts/self_monitor_questions.py`: "Why?" questions held.
- `services/orion-hub/scripts/curiosity_introspect_listener.py`: open self-questions skip non-admitted rows.
- `services/orion-hub/scripts/main.py`: start/stop the loop.
- `services/orion-hub/app/settings.py`, `services/orion-hub/.env_example`, `services/orion-hub/README.md`.
- `services/orion-hub/tests/test_self_question_door.py`, `test_self_question_door_pg.py`, and updated
  `test_self_monitor_questions.py`, `test_curiosity_self_question_persistence.py`.
- `services/orion-hub/evals/run_self_question_door_replay.py`: fixture gates (CI) + `--live` replay.
- `services/orion-sql-db/manual_migration_curiosity_self_question_door_v1.sql`.
- `scripts/self_question_door_backfill.py` + `tests/test_self_question_door_backfill.py`.
- `scripts/analysis/eval_reflection_register_share.py`, `scripts/analysis/judge_anchor_set.py`,
  `scripts/analysis/reflection_register_sources.py`.
- `.github/workflows/self-question-door-tests.yml`: the door's tests, the real-SQL test, the replay gates.

## Schema / bus / API changes

- Added: `curiosity_self_questions.door`, `door_reason`, `door_judged_at`, `door_judge_model`,
  `door_prompt_sha256`, `door_attempts` (NOT NULL DEFAULT 0), `door_demoted_by_run`.
- Removed / renamed: none.
- Behavior changed: an Orion-written question is pickable only when admitted (or written before the door).
  A door focus row in `curiosity_focus_request` uses `origin='self'`, correlation
  `self_question_door:<question_id>[#n]`.
- Compatibility: no bus channel, no schema registry entry, no new table. The Hub image needs the migration
  first: `REQUIRED-BY: orion-hub` makes the deploy gate refuse Hub until it is applied, and the post-pull step
  applies it (additive allowlist).

## Env/config changes

- Added keys: `CURIOSITY_SELF_QUESTION_DOOR_ENABLED=true`, `CURIOSITY_SELF_QUESTION_DOOR_TICK_SEC=300`,
  `CURIOSITY_SELF_QUESTION_DOOR_MAX_ATTEMPTS=3`.
- `.env_example` updated: yes (orion-hub).
- local `.env` synced with `python3 scripts/sync_local_env_from_example.py --all-keys orion-hub`: yes, the three
  keys are in the primary checkout's `services/orion-hub/.env`.
- Skipped keys: none.

## Tests run

```text
orion/curiosity/tests                                   320 passed, 29 skipped
services/orion-hub tests -k curiosity|self_question|self_monitor|focus|self_inquiry|introspect|replica
                                                        796 passed, 6 skipped
services/orion-hub/tests/test_self_question_door.py     24 passed
services/orion-hub/tests/test_self_question_door_pg.py  1 passed (throwaway postgres:16 container)
tests/test_self_question_door_backfill.py               5 passed
full services/orion-hub/tests (before review fixes)     5318 passed, 4 failed, 4 errors -- the same 8 fail
                                                        on origin/main (attention loops UI smoke, async-routes
                                                        repo scan, fcc model labels, consult SQL fixture)
mutation: passes_door -> always True                    5 pool tests, 1 hub test and the replay gate fail
static gates: check_settings_defaults orion-hub, check_migration_required_by, check_integration_seam
  --require-base, check_env_key_single_source, check_scripts_dir_no_stdlib_shadow, check_metric_lineage --gate,
  check_definition_drift --gate, check_async_routes_not_blocking, check_env_template_parity: all OK
```

## Evals run

```text
PYTHONPATH=. python services/orion-hub/evals/run_self_question_door_replay.py      PASS (9 gates)
PYTHONPATH=. python services/orion-hub/evals/run_self_question_door_replay.py --live --judge-url <agent seat 8015>
  88 Orion questions judged once (first QUESTION_LINE): admitted 44, routed 13, held 31 (26 no_finding, 5 not one
  JSON object). Self-inquiry findings (live register, 64 labelled): 45% philosophical (29/64, 95% 34-57%) today;
  with the door (Juniper's + admitted questions only) 67% (29/43, 53-80%). Juniper's questions 20/24 (83%);
  admitted Orion 9/19 (47%); routed 0/7; held 0/14. A routed question gave a philosophical finding: 0 times.
  The 26 no_finding replies were the door's own bug: a question states no conclusion, which is prompt v2's
  no_finding test, so a third of Orion's questions would have ended unjudged (acceptance check 5 fails).
  Fixed QUESTION_LINE (tells the judge a question is never no_finding); re-judged the 31 held + 19 labelled:
  31/31 labelled (26 operational, 5 philosophical); 16/19 kept their label, 3 moved philosophical -> operational.
  Files: /tmp/self-question-door-replay/2026-11-25/ (result.json, questions.jsonl, sittings.jsonl).
python3 scripts/analysis/judge_anchor_set.py build      100 unlabelled rows -> ~/.orion/judge-anchor/anchor_set_v1.tsv
python3 scripts/analysis/eval_reflection_register_share.py --per-organ 8 (11-11..11-25; small smoke sample, not the baseline)
  philosophical share: self-inquiry findings 4/8, investigate findings 0/7 (1 judge error), Orion questions 1/8
  (new QUESTION_LINE), outreach 4/8, reverie thoughts 1/8, reverie questions 0 labelled (1 judge error),
  day letter paragraphs 5/8. Wide intervals; acceptance check 6 runs it at 30 per organ before and after deploy.
  Report: /tmp/reflection-register-share/2026-11-25/report.md
```

No outside model was called. Every live read was SELECT-only (the analysis sessions run as read-only
transactions); the judge calls went to Orion's own agent seat.

## Docker/build/smoke checks

```text
No Hub image build here: the change is Python + SQL inside the existing image. Live smoke (spec step 7: one held
row admitted with a judge model stamped, one routed, next self-inquiry pick shows no held row) is UNVERIFIED
until deploy.
```

## Review findings fixed

- Finding: backfill dry run overwrote/removed the apply snapshot; dry-run-then-apply was refused.
  - Fix: dry run writes `snapshot.dryrun.jsonl`; apply refuses only over an applied snapshot.
  - Evidence: `test_a_dry_run_writes_nothing_and_leaves_the_apply_snapshot_alone`.
- Finding: door offers could keep Prime's focus questions out of the one slot.
  - Fix: one door offer per 6 h; Prime's `keep_pending` queue replaces a pending door focus.
  - Evidence: `test_prime_replaces_a_pending_door_focus_but_not_a_chat_or_juniper_focus`, the real-SQL test.
- Finding: the focus prompt showed Hub's text as "Why, in your words then"; demoted wording was wrong.
  - Fix: "Why the door sent it (Hub's words, not yours)"; separate routed/demoted reasons, third person.
  - Evidence: `test_a_routed_focus_says_where_it_came_from_not_a_chat_with_juniper`.
- Finding: a re-mint mid-judgement got the old text's verdict.
  - Fix: `SET_DOOR_SQL` guarded on the judged text; a 0-row write logs `skipped_changed`.
  - Evidence: `test_a_re_mint_with_new_words_while_judging_keeps_the_new_words_held`, real-SQL test.
- Finding: one stuck row could starve the batch.
  - Fix: held rows ordered least recently tried first.
- Finding: a short spell on an unevaluated model would make every held question unjudged for good.
  - Fix: `model_not_evaluated` is not counted (held, retried), like a gateway refusal.
- Finding: a replaced or expired door offer was never offered again.
  - Fix: re-offer (`#2`, `#3`) until a run takes one, at most 3 offers.
- Finding: the curiosity introspect tool still listed held/routed questions as open.
  - Fix: `SELF_QUESTIONS_SQL` skips non-admitted Orion rows.
- Finding: no test ran the SQL.
  - Fix: `test_self_question_door_pg.py` (mint rules from every state, guards, demotion, real focus slot).
- Finding: backfill rollback not atomic.
  - Fix: one `psql -1` transaction over the snapshotted ids.
- Finding: analysis scripts' read-only claim was a convention.
  - Fix: their psql sessions set `default_transaction_read_only=on`.
- Nits fixed: pinned floor honours the kill switch; unused `question_judge_messages` removed; progress never
  counts a vanished row as done.
- Not fixed here: the curiosity replica frame also mislabels a door focus. Fixing it needs a field on a
  cross-service `extra=forbid` schema (consumer-first deploy). Filed #3932.

## Restart required

```bash
# From the clean primary main checkout after merge (post-pull applies the migration first):
scripts/safe_docker_build.sh orion-hub up -d --build
# Then the one-time backfill (production write; snapshot + monitor + report under /tmp/self-question-door-backfill/):
python3 scripts/self_question_door_backfill.py            # dry run: lists the targets, writes nothing
python3 scripts/self_question_door_backfill.py --apply    # holds them; Hub's loop judges them; ~11 ticks
tail -f /tmp/self-question-door-backfill/progress.log
```

## Risks / concerns

- Severity: medium. Concern: until the backfill runs, the 88 pre-door Orion rows stay pickable (by design:
  `door IS NULL` = written before the door). Mitigation: run the backfill right after deploy.
- Severity: medium. Concern: the judge on question texts is UNVERIFIED as an instrument (anchored on findings,
  not questions); 9 of 88 replies were not clean JSON in the spec's measurement. Mitigation: retries,
  `unjudged` never admits, acceptance check 5; Juniper's anchor set will check it.
- Severity: low. Concern: routed questions borrow the one focus slot. Mitigation: 6 h spacing; Prime and
  Juniper replace it.
- Severity: low. Concern: `QUESTION_LINE` wording is new (the spec's measurement used an unrecorded line) and
  was tuned once on live data (see Evals). Mitigation: the anchor set and acceptance check 5 measure it after deploy.
- Issues opened: #3932 (replica frame door label). Closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3933

🤖 Generated with [Claude Code](https://claude.com/claude-code)
