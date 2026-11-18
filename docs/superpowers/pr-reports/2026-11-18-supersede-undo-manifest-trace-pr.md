## Summary

Orion asked 8 questions on the PR #3734 thread about the memory-correction pass. Three of them found real gaps in `scripts/memory_supersede_pass.py`. This PR fixes them.

- **An undo now updates the file Orion reads.** The pass manifest (`/mnt/orion-fcc/notes/memory_supersede_pass_manifest.json`) still said "2 applied" after both pairs were undone. Now an undo rewrites it: an `undone` list (who, when, the note), `undone_at` on each undone applied entry, and `in_effect`, which is read from the database. The proposal's check version is kept, so the file does not relabel a v3.1 proposal as v3.4.
- **An undo must say who decided and why.** `--actor` and `--note` are now required, and a blank note is refused before any database connection. The note was already stored (`episode_memory_event.evidence->>'note'`), but the command let it be empty, and both live undos have `note = ""` and `actor = operator`.
- **A new read-only `manifest` command** rewrites the manifest from the database. Run it once after deploy to fix today's stale file.
- **The apply run's audit row now carries a per-pair trace.** Before, `memory_supersede_run.decisions` for `pass_apply` was `[]`. Now each pair has its cosine, shared rare words/referents, votes, reason and outcome. There are no statements.
- **Read-only is now real.** The dry run's `SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY` ran inside an already-open transaction, so it covered nothing. Both read-only commands now use `SET TRANSACTION READ ONLY` as the first statement. A Postgres test proves this, and a mutation back to the old form fails the test.
- **The write-time meaning-index failure is logged with its cause** (`supersede_dense_unavailable`). The first live v3.4 distill run refused with `dense_unavailable` and recorded no reason (#3693).

### Merge with main (after #3750)

`orion/memory/episode/supersede.py` conflicted with #3750. Both sides are kept:

- **The meaning-index retry and the failure log are combined.** #3750's bounded retry stays: each attempt is capped at 40 s in total, with one retry per plan. This branch's `supersede_dense_unavailable` warning now logs every failed attempt with its cause.
- **The apply audit row counts what apply did.** `record_run` now adds `counts.outcomes` (for example `{"applied": 2, "undone_never_reapplied": 1}`), taken from apply's per-pair decisions. That sits next to #3750's `counts.abstained` and the `dense_calls` / `dense_ms` / `dense_retries` counts, so nightly, dry-run and apply rows all have the same shape.
  - Test: `test_record_run_counts_apply_outcomes_from_per_pair_decisions`.
- **Tests after the merge:** 411 episode tests passed, Postgres-backed included, and 14 durable distill-graph tests passed.

## Outcome moved

The manifest stops claiming that undone corrections are in effect. Each future undo carries a reason and a decider that Orion can read. Each pass apply leaves a per-pair trace in a table that Orion can already read (`orion_readonly` has SELECT on `memory_supersede_run` and `episode_memory_event`, checked live 2026-11-18).

## Current architecture

`undo` called `undo_link` and printed a line. It did not touch the manifest or the history file. `apply` wrote `decisions: []`. `--actor` defaulted to `operator`, and `--note` defaulted to empty.

## Architecture touched

`scripts/memory_supersede_pass.py` only, plus one log line in `orion/memory/episode/supersede.py` (`index_dense`). No schema, bus, env, or service change.

## Files changed

- `scripts/memory_supersede_pass.py`: undo rewrites the manifest; `--actor`/`--note` are required; there is a new `manifest` command; `in_effect`/`undone` come from the database; apply has a per-pair trace; read-only transactions are fixed; an earlier watch list is kept.
- `orion/memory/episode/supersede.py`: `index_dense` logs why the index could not answer.
- `orion/memory/episode/tests/test_supersede_pass_script.py`: manifest after undo, the history line, a broken earlier file, a required note/actor, a trace with no statements, the log line.
- `orion/memory/episode/tests/test_supersede_pg.py`: CLI undo stores the note and actor and rewrites the manifest; the refresh is read-only and builds `in_effect` from the database; a second undo leaves the file alone.
- `orion/memory/README.md`: lists the new subcommand.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: `memory_supersede_pass.py undo` needs `--actor` and `--note`. The manifest JSON gains `in_effect`, `undone`, and `undone_rule`.
- Compatibility notes: nothing in code reads the manifest. Older manifests (no `undone` key, or not a JSON object) are handled.

## Env/config changes

- None. No `.env_example` changed, so no sync was needed.

## Tests run

```text
PYTHONPATH=. pytest -q orion/memory/episode/tests                      370 passed, 34 skipped
ORION_MEMORY_EPISODE_TEST_DATABASE_URL=<throwaway postgres:16> pytest -q orion/memory/episode/tests/test_supersede_pg.py   12 passed
mutation: refresh back to SET SESSION CHARACTERISTICS -> test_pass_undo_stores_the_note_and_rewrites_the_manifest fails (seen["ro"] != "on")
```

## Evals run

```text
None new: the classifier and check version are unchanged (supersede_classify.v3.4, passing eval 09:31Z).
Read-only re-check for Orion's question 6 (no apply): both undone pairs under v3.4 -
  65590f70 -> 4da33de0: still a candidate (cosine 0.64 + project:circe), votes "compatible", not the same referent (2 of 2 repeats)
  af9d823c -> 06c7cf45: still a candidate (cosine 0.70 + one rare word), stopped by the numbered-item rule before any vote (2 of 2)
```

## Docker/build/smoke checks

```text
Not run: an operator script and one log line; no dependency or compose change.
```

## Review findings fixed

- Finding: the read-only guard did nothing (SET SESSION CHARACTERISTICS inside the open transaction).
  - Fix: `SET TRANSACTION READ ONLY` first, in the dry run and the refresh.
  - Evidence: the pg test asserts `transaction_read_only = on`; the mutation fails it.
- Finding: `in_effect` came from the file's apply history, so it was wrong after a missing file, a second apply, or a later revision.
  - Fix: `IN_EFFECT_SQL` reads it from `episode_memory` + `episode_memory_event`.
  - Evidence: the pg refresh test with a corrupt earlier file.
- Finding: a manifest error after the committed undo exited non-zero, so a re-run said "nothing to undo".
  - Fix: catch the error, print "undo applied; manifest NOT rewritten", and exit 0.
- Finding: `manifest --no-manifest` and `undo --watch` were accepted and ignored.
  - Fix: `manifest` has no `--no-manifest`; `undo` honours `--watch`.
- Nits fixed: the history line keeps the proposal's check version; a label-less watch entry gets no "None" label; the README row is updated.
- Not changed: the trace keeps the model's one-sentence reason. It can paraphrase a statement. The `superseded` event and the distill decisions already store the same reason.

## Restart required

No service restart required: `scripts/memory_supersede_pass.py` is run by hand. The `index_dense` log line takes effect on the next orion-durable-runs rebuild, which the post-pull rebuild does when this merges.

After merge, fix today's stale manifest (read-only on the database; rewrites the file):

```bash
.venv/bin/python scripts/memory_supersede_pass.py manifest
```

## Risks / concerns

- Severity: low.
  - Concern: the two existing undo events still say `actor=operator` and have an empty note. Changing them is a production write and Juniper's call (asked on #3629).
- Issues opened: #3748 (person:juniper referent threshold, needs Juniper), #3749 (Orion cannot challenge a core page sentence, needs Juniper).
- Issues commented: #3629 (undo permanence; needs-juniper added), #3693 (first v3.4 distill run refused, dense_unavailable), #3733 (origin page cite mix).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3751

🤖 Generated with [Claude Code](https://claude.com/claude-code)
