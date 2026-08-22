# A gate for the 78 hand-applied SQL migrations

## Summary

- `services/orion-sql-db/*.sql` is **78 files applied by hand**, with no migration table, no
  version stamp, and no ordering guarantee. A migration that was written, reviewed, merged and
  **never applied** looks identical in git to one that is live.
- Adds `scripts/check_sql_migrations_applied.py` — parses each migration for the objects it
  declares and checks them against the running database.
- **On its first run it found 6 files not fully applied**, including one whose entire schema
  (4 columns + 1 index) had never reached the database while a shipped service used all of it.
- Two in-file escape hatches (`SUPERSEDED-BY`, `NOT-A-MIGRATION`) so the gate can reach a clean
  state honestly rather than being permanently red.
- `make check-sql-migrations-applied` / `-quiet`.

## Outcome moved

"Is the schema we declare the schema we are running" went from unanswerable-without-reading-78-
files to one command. It is currently **green**, and it was not when I started.

## Current architecture

There is no migration framework here. Files are applied with a `docker exec -i ... psql < file`
recorded in a commit message, if at all. CLAUDE.md's own migration guidance is prose in a
README. This is precisely the shape the repo contract says to turn into a script rather than a
reminder.

## Architecture touched

New standalone script + Makefile targets + tests. No service, no runtime path, no contract.
Read-only against the database.

## Files changed

- `scripts/check_sql_migrations_applied.py`: new.
- `tests/test_sql_migration_drift_gate.py`: new, 19 tests, DB-free.
- `Makefile`: two targets.
- `services/orion-sql-db/manual_migration_{proposal,policy_decision,execution_dispatch}_frame_v1.sql`:
  `SUPERSEDED-BY` header added.
- `services/orion-sql-db/collapse_dump{,_pg}.sql`: `NOT-A-MIGRATION` header added.

## What it checks, and what it deliberately does not

Three statement forms are declarative — the object exists in the live database or it does not —
and they cover the overwhelming majority of the corpus:

```text
CREATE [UNIQUE] INDEX [CONCURRENTLY] [IF NOT EXISTS] <name> ON <table>
CREATE TABLE [IF NOT EXISTS] <name>
ALTER TABLE <table> ADD COLUMN [IF NOT EXISTS] <column>
```

Everything else — INSERT/UPDATE/DELETE/DROP backfills, `ALTER TABLE ... SET (...)`, functions,
views — is **not** checked, because "did this backfill run" is not answerable from the presence
of a schema object. Those are **counted and reported as UNCHECKED per file, never silently
skipped**, and a file with nothing declarative reports **UNKNOWN, not APPLIED**. A gate that
returns a false clean bill of health is worse than no gate.

**An INVALID index is a failure, not a pass.** An interrupted `CREATE INDEX CONCURRENTLY`
leaves `indisvalid = false`: invisible to `IF NOT EXISTS` (so re-running the migration is a
silent no-op) and unusable by the planner (so queries quietly seq-scan). CLAUDE.md's migration
instructions warn about exactly this.

Parsing is comment-aware. These files carry very long `--` headers that discuss their own SQL
in English; a parser that reads its own documentation invents objects and the gate goes
permanently red on fiction.

## What it found on the first run

```text
78 migration file(s): 67 applied, 6 missing, 5 unknown
```

**1. `manual_migration_substrate_reverie_thought_expectation.sql` — never applied.** All four
columns (`expectation`, `expectation_checkable_by`, `expectation_verdict`,
`expectation_scored_at`) and the partial index were absent, while
`services/orion-thought` ships `app/store.py`, `app/reverie.py`, `app/settings.py`,
`orion/schemas/reverie.py`, a README section and a test suite that all use them — and the
container runs with `ORION_REVERIE_EXPECTATION_SCORING_ENABLED=true`. The migration's own
header says, in as many words: *"Apply before enabling
ORION_REVERIE_EXPECTATION_SCORING_ENABLED."* It was not.

Applied it (purely additive, `if not exists` throughout, instant — a non-volatile DEFAULT does
not rewrite the table). Gate now reports it applied.

**2. Three `idx_*_source_self_state` indexes — legitimately superseded.** The matching
`*_v2_drop_self_state.sql` migrations explicitly `drop index if exists` them. Correct history;
correctly absent. Annotated.

**3. Two `collapse_dump*.sql` — not migrations at all.** pg_dump output that happens to live in
this directory. Live `collapse_enrichment` has only its primary key and 0 rows. Annotated.

## A correction to my own first reading

I initially reported that the missing columns were why `substrate_reverie_thought` had not
accepted a write in 39 hours (last row 2026-08-19 01:26). **That was wrong, and I checked
before saying it in a PR rather than after.**

The reverie tick is running normally, every ~90s. Every thought is dropped at
`app/reverie.py:660` — `if thought.hollow: return None` — with `reason=zero_grounding` and
`terminal=no_coalition`, which happens **before** the envelope is built and long before any
INSERT. The write path is never reached, so the missing columns were never exercised.

Both problems are real and they are independent:

- the schema drift would have broken persistence the moment a non-hollow thought appeared, and
  is fixed here;
- **reverie has produced zero groundable thoughts for 39+ hours**, which this patch does not
  address and which deserves its own investigation.

## Escape hatches, and why they live in the file

A permanently-red gate is a gate people learn to ignore. But an exception recorded in a commit
message or someone's memory is not an exception — it is drift with an alibi. Both markers must
be written into the migration itself:

```sql
-- ORION-MIGRATION-SUPERSEDED-BY: <file>      -- a later migration removed what this created
-- ORION-MIGRATION-NOT-A-MIGRATION: <reason>  -- never meant to be applied
```

A `SUPERSEDED-BY` naming a file that does not exist is itself reported as drift.

## Schema / bus / API changes

None. Read-only.

## Env/config changes

None. Connection defaults to `localhost:55432/conjourney` and honours `ORION_PG_*` /
`PGPASSWORD`.

## Tests run

```text
$ pytest tests/test_sql_migration_drift_gate.py -q
19 passed in 0.13s
```

DB-free on purpose: the parser decides which objects to look for, and a parser that
under-reports produces a green gate that means nothing. Every test is about the parser being
wrong in the direction that *looks fine*.

Six mutations:

```text
Q1 stop stripping comments (parser reads its own prose)  -> 3 failed
Q2 invalid index treated as applied                      -> SURVIVED, then fixed
Q3 empty file reports APPLIED instead of UNKNOWN         -> 1 failed
Q4 dangling SUPERSEDED-BY accepted                       -> 1 failed
Q5 unchecked statements not counted                      -> 1 failed
Q6 column key drops the table qualifier                  -> SURVIVED, then fixed
```

Q2 and Q6 are the interesting ones, and both were my tests being weak rather than the code:

- **Q2**: my "invalid index" test asserted on `FileReport.status` given a `Finding` someone had
  already labelled `"invalid"` — testing the arithmetic, not the lifecycle. Replacing
  `check()`'s `elif not indexes[name]` with `elif False` left the whole suite green. Rewritten
  to drive the real `check()` across all three live states (valid / invalid / absent). This is
  the single most important thing the gate does, and it was untested.
- **Q6**: the live-state set is keyed `"table.column"`. Dropping the qualifier makes
  `foo.status` satisfy a check for `bar.status` — a missing column reported as applied, the
  exact direction of error the gate exists to prevent. Two tests added.

## Evals run

No eval harness for `scripts/`. The gate's own live run against all 78 real migrations is the
integration evidence, and two tests assert it stays meaningful against the real corpus — one
that every file parses, and one that the corpus still yields >100 declarative objects, which
guards the failure mode where a regex change makes the gate silently blind while still
exiting 0.

## Docker/build/smoke checks

```text
$ make check-sql-migrations-applied-quiet
78 migration file(s): 68 applied, 2 skipped, 3 superseded, 5 unknown
Every declaratively checkable object in every migration is present and valid.
exit=0
```

Migration applied live:

```text
$ docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
    < services/orion-sql-db/manual_migration_substrate_reverie_thought_expectation.sql
ALTER TABLE x4
CREATE INDEX
```

## Review findings fixed

Self-reviewed via mutation testing (above); Q2 and Q6 were found and fixed that way. No
subagent review was run on this branch — the two review agents in this session were spent on
the other two PRs. Flagged under concerns.

## Restart required

```text
No restart required.
```

The reverie schema change is additive and takes effect immediately; `orion-thought` opens new
connections per operation and does not cache the table shape.

## Risks / concerns

- Severity: medium. Concern: **no independent code review on this branch** — mutation testing
  is not the same thing, and it already showed my own tests can be weak in exactly the places I
  am most confident. Mitigation: worth one review pass before merge.
- Severity: medium. Concern: **reverie has produced zero groundable thoughts for 39+ hours**
  (`reason=zero_grounding`, `terminal=no_coalition`, every ~90s tick). Found while
  investigating this gate's first finding; not addressed here. Needs its own look.
- Severity: low. Concern: the parser covers three statement forms. Everything else reports
  UNCHECKED/UNKNOWN, so coverage is honest, but a migration whose only effect is a backfill
  still cannot be verified. Mitigation: stated in the script's docstring and in the output.
- Severity: low. Concern: the gate needs a live database, so it cannot be a pre-commit hook or
  a CI gate as written. Mitigation: exit code 2 for "could not connect" is deliberately
  distinct from exit 1 for "drift found", so a future CI wiring cannot mistake an infra failure
  for a pass.
- Severity: informational. Concern: 5 files report UNKNOWN. That is not a pass and is printed
  as such, but it does mean 5 migrations remain unverifiable by this tool.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1781
