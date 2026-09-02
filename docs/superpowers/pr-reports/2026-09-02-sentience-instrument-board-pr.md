# PR #2026 — Sentience Striving Program: instrument board + drift gate

https://github.com/junebug-junie/Orion-Sapienform/pull/2026

## Summary

- The program's narrative record cannot detect its own decay. A read of `orion/sentience_striving_program/README.md` on 2026-09-02 found **four** claims live data had already moved past — including the finding Objective 7 was *closed* on.
- Adds `instruments.yaml`: each instrument's module, backing table, retention ceiling, outcome (O1–O4) linkage, and the claims resting on it — recorded as **re-runnable queries**, not sentences.
- Adds `scripts/check_sentience_instruments.py` (`make check-sentience-instruments`): re-runs every claim against live repo + database state, exits non-zero on drift.
- Adds Hub `/sentience-program`: the operator view of the same join — what each instrument is doing now, how far its history reaches and what bounds it, what it affects, which outcome it unlocks.
- Reuses the existing metric semantic layer (`orion/metrics/`) for blast radius rather than rescanning, so the board and `check_metric_lineage.py` cannot disagree.
- Read-only throughout. No new consumer, no cognition change, no fused cross-instrument score (Objective 7 closed on exactly that).

## Outcome moved

Four staleness failures found in one sitting, now each either fixed or gated:

| Finding | State |
|---|---|
| §15d: PR #1894 "unmerged, undeployed, zero real runs" | Wrong on all three — merged `975f437e8`, enabled, **26 runs**. §15b's detector **PASSES** (4 refuted priors; real downward revisions `0.85 → 0.6 → 0.35 → 0.15 → 0`, ending refuted). README corrected. |
| Objective 7 closed on "only 5 distinct `reason_narrative` strings" | Now **16**, unnoticed for 13 days. Now a gated claim. |
| "19,417 rows, full history" | Never full history — capped at **7 days** by `SUBSTRATE_ATTENTION_SELF_MODEL_LOG_RETENTION_HOURS`. Now rendered on the board. |
| `pg_stat_user_tables.n_live_tup` = 0 | Real count **19,774**. The reducer uses `COUNT(*)` deliberately; a stats-view read would render a live instrument as dead. |

## Architecture touched

`orion/sentience_striving_program/` (new package), `scripts/`, `services/orion-hub/` (one new route module, one template, one JS panel, two lines in `api_routes.py`), `Makefile`.

Deliberately **not** a second metric registry. Every mechanical fact — who writes a signal, who reads it, whether it is degenerate — resolves at read time from `orion/metrics/`. The manifest declares only what that layer structurally cannot know: that a thing is an instrument *of this program*, which outcome it ladders to and why (irreducibly editorial → freshness-gated, not correctness-gated), and the claims resting on it.

## Files changed

- `orion/sentience_striving_program/instruments.yaml`: the manifest — 7 instruments, 4 outcomes, 7 claims
- `orion/sentience_striving_program/instruments.py`: the reducer; owns no facts of its own
- `scripts/check_sentience_instruments.py`: the gate
- `services/orion-hub/scripts/sentience_program_routes.py`: `/sentience-program` + `/api/sentience-program`
- `services/orion-hub/templates/sentience_program.html`, `static/js/sentience-program.js`: the board
- `orion/sentience_striving_program/README.md`: §15d corrected; new §16 / §16a
- `Makefile`: `check-sentience-instruments`

## Schema / bus / API changes

- Added: `GET /sentience-program` (HTML), `GET /api/sentience-program?consumers=` (JSON). No collision — grep-verified.
- No bus channel, schema-registry, or payload changes. Nothing published or consumed.

## Env/config changes

None. No `.env_example` touched, so no sync required.

**Flagged, not done:** `SUBSTRATE_ATTENTION_SELF_MODEL_LOG_RETENTION_HOURS=168.0 → 8760.0` is a live config change left for Juniper. The 7-day ceiling is **not** a storage decision — the setting's own comment says it was sized to clear a 48h replay window. Measured: the table is **50 MB**/7d against a **41 GB** database and **136 GB** free disk, so a full year costs ~2.6 GB (vs `grammar_events` at 6.5 GB). §16a records why summarising beyond day 7 was argued against rather than built: every bug this program has caught was found by recovering a *pre-aggregation* value, and a rollup forces the choice of which aggregates survive before anyone knows what future-you needs.

## Tests run

```text
orion/sentience_striving_program/tests/   14 passed in 0.27s
services/orion-hub/tests/test_sentience_program_api.py   6 passed in 1.08s
```

## Evals run

```text
No eval harness exists for either touched surface. The gate itself is the
periodic-eval lane for this program: `make check-sentience-instruments` is a
real-data measurement, not a unit test, and is designed to be run on a cadence.
```

## Docker/build/smoke checks

```text
Live API against real Postgres:  HTTP 200, db_error none, 7 instruments
  O3 ast_hot_reducer            rows=19,772  hist=7.0d  cap=7.0d
  O2 prediction_error_domains   rows=123,899 hist=3.0d
  O3 rpt_lamme_recurrence       rows=123,768 hist=3.0d  cap=3.0d
  O2 goal_provenance            rows=1 (singleton, no history by construction)
  O4 emergent_clustering_probe  rows=123,768 hist=3.0d
  O4 curiosity_worldview        MANUAL (FalkorDB, not SQL)
  O1 capability_policy_salience no table

Hub container import check:
  yaml OK / metrics.liveness OK / metrics.consumers OK / psycopg MISSING
  -> route uses SQLAlchemy raw_connection(), not the psycopg helper.

Gate mutation-tested 4 ways, baseline green (exit 0):
  M1 recorded 16 -> 5      DRIFT caught (exit 1)  <- reproduces the real drift
  M2 deleted symbol back   DRIFT caught (exit 1)
  M3 module path moved     MISSING caught (exit 1)
  M4 review 609d stale     STALE caught (exit 1)
```

## Review findings fixed

- Finding: route returned HTTP 500 on any database failure — found by a live run, not a test.
  - Fix: degrades to manifest-only and reports `db_error`; the UI renders it as an outage, never as an empty board.
  - Evidence: `test_board_still_renders_when_the_database_is_unreachable`; live run with an unreachable host now returns 200.
- Finding: retention silently resolved to `None` in any worktree — `.env` is gitignored, and the lookup had no fallback, so the board showed no ceiling at all: exactly the fact it exists to surface.
  - Fix: falls back to `.env_example` and reports which source it read.
  - Evidence: `test_retention_resolves_for_instruments_that_declare_one`; board now renders `capped at 7.0d (.env_example)`.
- Finding: manifest named two paths that did not exist (`novelty_for_target` had moved to `scoring.py`; `capability_policy.py` is under `orion/autonomy/`).
  - Fix: corrected; `test_every_instrument_module_exists` now prevents recurrence.

## Restart required

```bash
sudo docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build orion-athena-hub
```

Hub mounts the repo read-only from `ORION_HOST_REPO_ROOT` (defaults to the primary checkout), so this needs a merge to `main` before the container sees it. The CLI gate needs no restart.

## Risks / concerns

- Severity: low. Concern: `?consumers=true` walks ~4,300 source files and takes tens of seconds. Mitigation: off by default, behind an explicit button, and the UI says what it is doing.
- Severity: low. Concern: the `manual` claim kind (the §15b FalkorDB detector) is reported, never auto-passed — a human must re-run it. Mitigation: deliberate, and `test_manual_claim_is_never_auto_passed` pins it; auto-passing a check nobody ran is the failure mode this program keeps rediscovering.
- Severity: informational. Concern: a DRIFT is not automatically a regression — it means live data moved past what was recorded. The gate's own output says so and names the required response.

