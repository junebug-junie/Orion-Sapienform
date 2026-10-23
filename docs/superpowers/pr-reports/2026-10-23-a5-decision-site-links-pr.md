## Summary

- New read-only rollup in orion-sql-writer: `app/act_downstream_use.py` writes `act_downstream_use`, one row per `grammar_acts` act. Each row answers: did a later decision use this act's output within 72 h?
- The chain it follows is the only output→decision link that works today: act → output row `<table>:<id>` → a Temporal Self arc that holds that ref in `evidence_refs` → a `thought_decision` that cites `arc:<id>` in `day_citations`.
- An act whose family has no join path reads `no_link_path` with `used` NULL. It never reads as 0% used.
- A new additive migration with an `ORION-MIGRATION-REQUIRED-BY: orion-sql-writer` header. It passes `auto_apply_refusals` with no refusals. No bus channel is added.
- A per-family report: `python -m app.act_downstream_use 14`.

## Outcome moved

Body-economy Organ 3b, weight source (a), "downstream use by decision sites" (Patch A results, item 7). Before this, nothing measured whether an act's output was ever used. Organ 3b's `family_weight` and replay R7 now have a per-act input.

### Per family, last 14 days (live, 2026-10-23 ~20:10Z)

Measured read-only: the rollup ran inside a rolled-back transaction, against a temp table.

| family (actor) | acts | linked | linked, window closed | used | used, window closed | use rate (used/linked, closed) | use rate incl. failed acts | citing arcs (closed) / 1-per-N credit | status |
|---|---|---|---|---|---|---|---|---|---|
| hub.chat_reply | 36 | 36 | 20 | 25 | 13 | 0.65 | 0.65 | 7 arcs / 6.07 | linked |
| thought.painting | 33 | 4 | 0 | 2 | 0 | — (4 linked windows still open) | 0.00 (29 failed acts, closed) | — | linked |
| hub.retro_leisure | 1 | 1 | 0 | 0 | 0 | — (window open) | — | — | linked |
| hub.world_pulse_read | 159 | — | — | — | — | — | — | — | **no_link_path** |
| thought.dream_hop | 24 | — | — | — | — | — | — | — | **no_link_path** |
| hub.outreach | 13 | — | — | — | — | — | — | — | **no_link_path** |
| hub.retro_coplay | 0 | | | | | | | | path `direct` ready. Producer is PR #2885, not deployed |
| curiosity, dream, actions, topic-foundry, ask-Claude, self-monitor | 0 | | | | | | | | no acts yet. A `<arc source table>:<id>` ref (e.g. `curiosity_run_outcomes:<run_id>`) links with no change here |

How to read the chat row: 13 used chat replies come from only 7 cited conversation arcs. With 1/N credit per arc, they are worth 6.07 "whole" uses. Chat use is arc-level: it measures that a later decision cited the conversation the reply was part of.

Why each `no_link_path`:
- **world_pulse_read**: no reading table is a Temporal Self arc source.
- **dream_hop**: dream-carry pictures write no `reverie_visual_chain` row, by design (`services/orion-thought/app/store.py`).
- **outreach**: the chat row exists, but Temporal Self binds Orion's own chat rows as context and never as evidence (`orion/temporal_self/sources.py` `chat_turn`, danger mode 9). Live, 0 of 13 outreach rows appear in any arc's `evidence_refs`, while 36 of 36 chat-reply rows do.

Outreach was first wired as `chat_row`. That read 0% used, which was wrong. This live check found the error, and outreach was moved to `no_link_path`.

## Metric gate

1. **Provenance.**
   - `used` comes from `ROLLUP_SQL` in `services/orion-sql-writer/app/act_downstream_use.py`.
   - Its inputs are `grammar_acts` (act ledger, `orion/grammar/act_ledger.py`), `temporal_self_arc.arc_json.evidence_refs` (written by `orion/temporal_self/arcs.py` from `sources.py`), and `thought_decision.day_citations.cited_valid` (stance_react_v1, `orion/temporal_self/citations.py`).
   - The only decision site today is `stance_react_v1`. It made 1,279 decisions in 14 days, and 167 of them cite arcs.
2. **Independence.**
   - `used` is not computed from the act's own outcome (`outcome_value`), cost, or prediction.
   - It shares one upstream with the act's grade: the output row exists only if the act produced something. `no_output` acts therefore read `used=false`, and use rate is computed over linked acts only.
   - For chat replies, the obvious short chain would be "the next turn of the same conversation cites this turn's arc". Live, only 3 of 25 used chat replies were cited by a decision in the same session. The median lag is 4.1 h (p10 0.4 h, p90 5.0 h), so it is not mainly the next turn.
   - All families share one confound: the only decision site is the chat stance. Every use is therefore gated on Juniper chatting. This scales all families together, so it does not favor one family. It does mean use rates fall on quiet days.
3. **Theory anchor.** Value of information. An output has value to the system when a later decision actually conditioned on it. Here, "conditioned on" means the decision cited it through its arc. This measures realized value, not predicted value.
4. **Rest state.**
   - With no decisions, every linked act reads `used=false` once its window closes. Rest is 0, and it is reachable.
   - `no_link_path` stays NULL, so absence of a path cannot pose as rest.
   - `window_closed=false` marks a false that is not final yet, so an open window is not read as "not used".
   - Live values are not degenerate. Chat reply reads 13 of 20 closed. Painting reads used 2 of 4 so far.
   - The arc evidence cap (256) is not hit by any arc today (0 arcs at the cap), so a used output is not silently dropped from `evidence_refs`.
5. **Existing-mechanism check.** Nothing in the repo computes downstream use. `day_citations` is only checked for truth (`orion/temporal_self/citation_truth.py`) and is not joined back to acts.
6. **Reversibility.** One table, one loop, and one env key (`ACT_DOWNSTREAM_USE_INTERVAL_SEC=0` disables it). No producer, schema, or bus contract depends on it. It is read by no one yet.

## Current architecture

`grammar_acts` grades whether an act produced something. Nothing recorded whether the product was ever used. The only output→decision link was arc-level citations in `thought_decision.day_citations`.

## Architecture touched

orion-sql-writer only:
- a new periodic task beside `orion_energy_rollup`;
- a new table, with a model and a migration;
- one env key.

No bus, schema-registry, or API changes.

## Files changed

- `services/orion-sql-writer/app/act_downstream_use.py`: the rollup SQL, the family report, the loop, and a CLI.
- `services/orion-sql-writer/app/models/act_downstream_use.py`, `app/models/__init__.py`: the model (create_all).
- `services/orion-sql-db/manual_migration_act_downstream_use_v1.sql`: the migration (REQUIRED-BY orion-sql-writer, auto-applicable).
- `services/orion-sql-writer/app/main.py`: starts and stops the loop.
- `services/orion-sql-writer/app/settings.py`, `.env_example`: `ACT_DOWNSTREAM_USE_INTERVAL_SEC` (900, ON).
- `services/orion-sql-writer/README.md`: documents the rollup.
- `services/orion-sql-writer/tests/test_act_downstream_use.py`: pure contract tests (arc-source drift, path targets, outreach, model/migration parity, allowlist, flag on).
- `services/orion-sql-writer/tests/test_act_downstream_use_pg.py`: real-Postgres behavior tests.
- `.github/workflows/orion-sql-writer-tests.yml`: the pure tests in the unit step, plus a must-not-skip Postgres step.

## Schema / bus / API changes

- Added: table `act_downstream_use`.
- Bus: none.
- Compatibility: purely additive.

## Env/config changes

- Added keys: `ACT_DOWNSTREAM_USE_INTERVAL_SEC=900` (orion-sql-writer).
- `.env_example` updated: yes.
- Local `.env` synced: yes. The worktree has no `.env`, so the key was appended to the operator `.env` at `/mnt/scripts/Orion-Sapienform/services/orion-sql-writer/.env` (gitignored). The value equals the settings default.
- Skipped keys: none.

## Tests run

```text
PYTHONPATH=<wt>:<wt>/services/orion-sql-writer pytest -q tests/test_act_downstream_use.py tests/test_act_downstream_use_pg.py
  (ORION_TEST_PG_URI = throwaway postgres:16-alpine, session TZ America/Denver)   10 passed
Mutations (tz cast, frozen cutoff, output_at guard) each fail >= 1 pg test
sql-writer full suite: 14 failed / 1080 passed. The same 14 fail on clean main locally (14 failed / 1074 passed): env-sensitive tests, unrelated.
make check-body-economy-contract                                          OK
scripts/check_migration_required_by.py --base origin/main                 OK
scripts/check_definition_drift.py --gate / check_metric_lineage.py --gate PASS
scripts/check_env_key_single_source.py                                    OK
scripts/check_service_env_compose_parity.py orion-sql-writer              N/A (env_file)
tests/test_no_personal_terms.py tests/test_compose_logging_journald.py    9 passed
auto_apply_refusals(migration)                                            []
```

## Evals run

```text
Live dry run (read-only, rolled-back txn, temp table): first pass 266 acts in 0.07 s, incremental pass 126 acts.
The per-family table above is its output. No separate eval harness exists for weight sources yet; replay R7 is the intended consumer.
```

## Docker/build/smoke checks

```text
Not deployed (per brief). After deploy:
docker logs orion-athena-sql-writer 2>&1 | grep ACT_DOWNSTREAM_USE
docker exec orion-athena-sql-writer python -m app.act_downstream_use 14
```

## Review findings fixed

- Finding (material): the incremental cutoff was wall-clock only. A writer stalled for about 38 h would freeze rows at `window_closed=false`, and an act replayed late would never get a row.
  - Fix: each pass recomputes every act with no row and every act whose window is still open, plus the last 5 days.
  - Evidence: `test_a_stalled_writer_is_caught_up_not_frozen`. Mutating the condition back fails 3 tests.
- Finding (material): the link reads the arc's current `evidence_refs`. Arcs are rewritten in place, with no history.
  - Fix: the decision must also come after the output row existed (`output_at`). The snapshot limit is stated in the docstring, the README, and this report.
  - Evidence: the `pre` case in `test_close_before_open_painting_two_outputs_and_chain_time_guard`. Removing the guard fails it.
- Finding (material): use is arc-granular. One cited conversation arc credits all of its chat rows.
  - Fix: `arc_evidence_n` is stored per row. The report gives `citing_arcs_closed` and a 1-per-N `credit_closed`. The chat caveat is above.
- Finding (minor): the arc-source pin missed `substrate_attention_broadcast_log`, which comes from arcs.py.
  - Fix: the table is added. The pin now also scans arcs.py `f"<table>:{` refs.
- Finding (minor): CI did not trigger on `orion/temporal_self/**`.
  - Fix: added to the pull_request and push paths.
- Finding (minor): the time zone test could not fail, because the session was UTC.
  - Fix: the fixture session runs in America/Denver. Removing `AT TIME ZONE 'UTC'` now fails 2 tests.
- Finding (minor): there were no tests for the chain-time guard, for two outputs where the used one is not first, or for `opened_at` NULL.
  - Fix: all three are covered in the new pg test.
- Finding (minor): `use_rate` dropped failed (`no_output`) acts.
  - Fix: added `use_rate_all` = used / (linked + no_output), documented.
- Finding (nit): the column was named `opened_at` but stored a fallback.
  - Fix: renamed to `started_at` and documented.
- Finding (nit): the decision filter could not use an index.
  - Fix: it now compares the raw column.

## Restart required

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

The post-pull rebuild auto-applies the migration first, because it is additive DDL with a REQUIRED-BY header.

## Risks / concerns

- Severity: medium.
  - Concern: one decision site (the chat stance) means use is gated on Juniper chatting. Weights learned from this alone would track chat volume.
  - Mitigation: record it. Add more decision sites (curiosity focus, allocator choice) as they start citing evidence.
- Severity: low.
  - Concern: the chat-reply output row is the whole turn (Juniper's message and Orion's reply). A citation of it is a citation of the turn, not only the reply.
- Severity: low.
  - Concern: `direct` refs are trusted without checking that the row exists. A producer that names an output it never wrote reads `linked`, and `used` stays false unless the ref is cited.

## PR link

(filled on open)
