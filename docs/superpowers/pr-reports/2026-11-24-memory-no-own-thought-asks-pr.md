## Summary

Orion no longer asks Juniper to confirm Orion's own thoughts. Only Orion knows what Orion thought.

- A high-stakes memory in Orion's own voice (`orion_thought`) is now stored as `own_thought`. The confirmation asker never picks that state. Before, the validator put every high-stakes memory in `pending_confirmation`, whatever its voice.
- Recall, the Juniper page, the Lately page and the daily report show these rows as Orion's own view: "My own thought (reverie, 11-20), not checked with Juniper: ...". They no longer say "Unconfirmed, check with Juniper".
- Every confirmation tick moves any `orion_thought` row still in an asking state (`pending_confirmation`, `unconfirmed`, `not_asked`) to `own_thought` and withdraws its open card. The tick also writes one `episode_memory_event` (`own_thought`, reason `orion_thought_never_asked`) for each row it moves.
- A one-time audited script, `scripts/memory_own_thought_settle.py`, runs the same SQL. It follows the backfill protocol (AGENTS.md section 14): snapshot, progress log, before/after CSV, report, rollback.
- Supersede: `own_thought` follows the same rules as `not_asked`. A settled correction may retire it. It never retires anything itself.

Fixes #3840

## Outcome moved

Juniper stops getting cards that ask her to confirm Orion's reveries and private thinking. On live data, the replay eval opens 0 cards on `orion_thought` rows, where it opened cards for them before. The 76 live `orion_thought` rows that sit in an asking state now render as Orion's own thought.

## Correction to the evidence in the brief

The brief said `orion_thought` cards were "0 confirmed". That number comes from a join artifact. When Juniper confirms a memory, `confirmation.apply_outcome` rewrites its voice to `worked_out_together`. A join on the current voice therefore hides every confirmed row. The `confirmed` event keeps the original voice (`evidence->>'prior_voice'`). Live counts by the voice at ask time (2026-11-24):

| original voice / channel | confirmed | rejected | revised |
|---|---|---|---|
| orion_thought / chat | 18 | 24 | 2 |
| orion_thought / reverie | 6 | 22 | 0 |
| juniper_said / chat | 41 | 9 | 3 |
| worked_out_together / chat | 1 | 0 | 0 |

So Juniper confirmed 24 of 72 answered `orion_thought` cards (33%). She confirmed 41 of 53 `juniper_said` cards (77%). The brief's `juniper_said` numbers (0 confirmed, 9 rejected, 3 revised) come from the same artifact. No follow-up issue was filed for "why are juniper_said cards mostly rejected": they are mostly confirmed. Issue #3840 carries the corrected table and the `needs-juniper` label, because her approval quoted the wrong numbers.

## Current architecture

- `orion/memory/episode/validate.py` stored every high-stakes memory as `pending_confirmation`.
- `orion/memory/episode/confirmation.py` (run by the orion-memory-consolidation ticker) opened an "Orion is asking" card for every `pending_confirmation` row, whatever its voice.
- `not_asked` (#3537 backfill) already existed for rows nobody will ask. It renders as "Unconfirmed, check with Juniper if natural".

## Architecture touched

- Validator: `initial_confirmation_state(stakes, voice)`.
- Confirmation loop: an opener guard (`voice <> 'orion_thought'`), plus `SETTLE_OWN_THOUGHTS_SQL` / `settle_own_thoughts` as a tick step after answers are applied and before the opener. A row whose answer is still waiting to be applied is left alone, so her answer wins.
- Readers: voice renderer, Juniper page, Lately page, daily report flag.
- Writers: the ungrounded-name repair script now writes `own_thought` for `orion_thought` rows.

### Why a new state instead of reusing `not_asked`

`not_asked` means "a claim we decided not to put to Juniper". For a `juniper_said` row that is still correct, and the "Unconfirmed, check with Juniper" label is right for it. Orion's own thought is a different thing: there is nothing for Juniper to check. If we reused `not_asked` and keyed the label on voice, every reader (recall, three pages, the report, supersede, the backfill counts) would have to repeat the voice rule. A missed reader would show the wrong label. A separate state keeps that rule in one place, at write time. Supersede already treats any state that is neither open nor settled the same way, so `own_thought` needed no new code there, only a comment and a test.

## Files changed

- `orion/memory/episode/validate.py`: `OWN_THOUGHT_*` constants, plus `initial_confirmation_state`.
- `orion/memory/episode/confirmation.py`: opener guard, `SETTLE_OWN_THOUGHTS_SQL`, `settle_own_thoughts`, tick step, docs.
- `orion/memory/voice_render.py`: `own_thought` render.
- `orion/memory/pages/juniper.py`, `orion/memory/pages/lately.py`: own-thought label (Lately now reads `confirmation_state`).
- `orion/memory/episode/report.py`: `own_thought` flag.
- `orion/memory/episode/supersede.py`: the documented rule.
- `scripts/memory_own_thought_settle.py`: the one-time audited pass (dry run / `--apply` / `--rollback`).
- `scripts/repair_episode_memory_ungrounded_names.py`: writes `own_thought` for Orion's voice.
- `services/orion-memory-consolidation/evals/run_memory_confirmation_replay_eval.py`: uses the validator rule, plus a new `cards_on_orion_thought` count.
- `services/orion-memory-consolidation/evals/results/2026-11-24-memory-confirmation-replay-own-thought.json`: eval output (counts only).
- Tests: `test_validate.py`, `test_voice_render.py`, `test_supersede.py`, `test_confirmation.py`, `test_confirmation_pg.py`, `test_own_thought_settle_script.py` (new), `test_backfill_jul_sep_script.py`, `test_core_page_juniper.py`, `test_core_page_lately.py`, `test_memory_confirmation_replay_eval.py`.
- `.github/workflows/orion-memory-episode-tests.yml`: path filter for the new script.

## Schema / bus / API changes

- Added: the `episode_memory.confirmation_state` value `own_thought`. The column has no CHECK constraint, so no migration is needed. Added: the `episode_memory_event` op `own_thought`. Added: the tick summary key `own_thought`.
- Removed / renamed: none.
- Behavior changed: high-stakes `orion_thought` memories are never asked. Their open cards become `withdrawn`.
- Compatibility: readers that filter on `confirmed` or `auto` (orion-thought store, lived memory) treat `own_thought` like `pending_confirmation`. Nothing changes for them.

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced: not needed.
- skipped keys: none.

## Tests run

```text
PYTHONPATH=. pytest -q orion/memory/episode/tests orion/memory/referents/tests services/orion-memory-consolidation/tests services/orion-memory-consolidation/evals orion/memory/tests orion/memory/pages/tests
  (ORION_MEMORY_EPISODE_TEST_DATABASE_URL = disposable pgvector:pg16)  -> 3642 passed, 11 skipped
PYTHONPATH=.:services/orion-durable-runs pytest -q services/orion-durable-runs/tests/test_episode_distill_{reconcile,graph,referent_policy}.py -> 25 passed
PYTHONPATH=. pytest -q services/orion-hub/tests/test_ask_resolve_pg.py -> 7 passed
PYTHONPATH=. pytest -q orion/memory -> 3354 passed, 7 skipped
```

## Evals run

```text
run_memory_confirmation_replay_eval.py --scratch-dsn <disposable> --live-dsn <live, read-only>
  live rows: 25 cards opened, 0 on orion_thought, 76 rows end in own_thought, max 5 open (cap held)
  v3 synthetic: 10 cards (was 15), 0 on orion_thought, 5 own_thought, cap held
```

## Docker/build/smoke checks

```text
No image or compose change. The new code runs in orion-memory-consolidation (confirmation tick) and
orion-durable-runs (validator) after their normal post-pull rebuild.
```

## One-time data pass (backfill protocol)

Dry run against live, 2026-11-24 (no writes). Files are under `/tmp/memory-own-thought/` (dir 700, files 600):

- Targets: 76 `orion_thought` rows in an asking state. 21 are `pending_confirmation` (chat 15, reverie 5, dream 1). 55 are `not_asked` from the backfill (chat 45, reverie 10).
- Rows held back because an answer is waiting: 0. Open cards to withdraw: 1.
- After the pass: `pending_confirmation` by voice goes from juniper_said 24 / orion_thought 21 / worked_out_together 1 to juniper_said 24 / worked_out_together 1.

Apply status: see Risks.

## Review findings fixed

Code review ran in a subagent. It found no blockers.

- Finding: a race could drop Juniper's answer. If her Hub resolve commits while the settle step runs, the settle cannot see her answer and moves the row anyway. `apply_outcome` then ignored her answer.
  - Fix: `apply_outcome` also accepts `own_thought` rows as answerable.
  - Evidence: `test_her_answer_still_applies_when_settle_moved_the_row_mid_resolve`.
- Finding: the next tick undoes a rollback, and that second move logged no event.
  - Fix: the docstring and report now say rollback only sticks with the code reverted or the loop off. The event id is salted with the row's prior `updated_at`, so a second move logs a new event. Rollback writes its own `own_thought_rolled_back` event.
  - Evidence: the end-to-end script test counts the events.
- Finding: rollback could reopen a card whose memory had left `own_thought`.
  - Fix: rollback reopens a card only when its memory was restored.
  - Evidence: script test, "corrected" row case.
- Finding: the READMEs did not describe the new state.
  - Fix: added the settle step and an `own_thought` row to the `services/orion-memory-consolidation/README.md` table, and the render row to `orion/memory/README.md`.
- Finding: the constants were defined twice.
  - Fix: `confirmation.py` now imports them from `validate.py`.

## Restart required

```text
No manual restart. The post-pull rebuild picks up orion-memory-consolidation and orion-durable-runs.
```

## Risks / concerns

- Severity: medium. Concern: Juniper approved this change on numbers that were wrong. She does confirm a third of these cards. Mitigation: #3840 is labelled `needs-juniper` and has the corrected table. The PR is not merged.
- Severity: low. Concern: 55 backfilled `not_asked` `orion_thought` rows also move to `own_thought`. That goes past the 21 in the brief. Their label changes, and nothing else does: no cards, and the same supersede rules. Mitigation: `--rollback` restores them from the snapshot.
- Issues opened: #3840. Issues closed: #3840 (on merge). No `juniper_said` follow-up was filed (see the correction above).

## PR link

(filled after open)
