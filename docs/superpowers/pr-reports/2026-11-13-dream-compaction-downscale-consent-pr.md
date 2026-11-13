# Dream compaction downscales settled reverie salience under Juniper's standing consent

## Summary

- Juniper's standing consent `dream-compaction-downscale` is now in `config/physiology/consent.yaml`. It
  covers downscale only. Prune stays `refused`.
- The applier has a real memory store for the first time: `services/orion-dream/app/compaction_store.py`.
  It lowers `substrate_reverie_thought.salience` and nothing else. It refuses prune and gist-card writes.
- REM now proposes downscale entries. Each reverie thought that a consolidate entry folds in (a thought of
  a settled reverie chain) gets new salience = salience x 0.8, once.
- The sleep cycle asks consent and applies through the store. Apply ships on
  (`ORION_DREAM_COMPACTION_APPLY_ENABLED=true`). Snapshots move to a persistent mount.
- Lifecycle tests on live-shaped data, a prune regression, mutation checks, and a live eval check of
  applied downscales.

## Outcome moved

Before: every dream compaction ask read `needs_consent`, and even with consent the applier had no store
(`no_store`). Memory never changed. Live eval, 3 days: 4 deltas, 5 asks, all `needs_consent`.

After: when Orion sleeps, the thoughts of reverie chains that settled lose 20% of their salience. Orion's
day reading ranks reverie thoughts by salience (`orion/orion_day/budget.py`), so settled thoughts give way
to unsettled ones. On live data (read-only dry run, 50 pending requests): 142 thoughts would be proposed,
all 142 found, none already lowered, 0 prune entries.

## Current architecture

- REM (`app/rem_compaction.py`) built deltas with consolidate entries only. `downscale=[]` was hard-coded.
  121 of 121 live deltas had `edges_downscaled=0`.
- `app/compaction_consent.ask_consent_without_store` asked the market with `store=None`, so the best
  possible result was `no_store`.
- `consent.yaml` had `standing: {}`.
- Snapshots went to `/tmp/dream-compaction-apply` inside the container, lost on restart.

## Architecture touched

- orion-dream: REM producer, applier (gist-less store support), new store, consent ask, sleep-cycle wiring.
- orion-durable-runs: no code change. `consent.yaml` is baked into its image, so it must be rebuilt.
- Postgres: `substrate_reverie_thought.salience` is now written by orion-dream (UPDATE only, no DDL).

## Files changed

- `config/physiology/consent.yaml`: the standing consent; rule comment names the real store.
- `services/orion-dream/app/compaction_store.py` (new): `ReverieSalienceStore` (`downscale_many` in one
  transaction; restore compare-and-set on the planned value), `restore_snapshot_file`, rollback CLI
  `python -m app.compaction_store restore <snapshot_before.json>`.
- `services/orion-dream/app/compaction_applier.py`: honors `supports_gists = False` (skips consolidate,
  counts `cards_skipped_no_gist_store`); a delta with nothing the store runs is `empty`, not `applied`;
  uses the store's batch downscale; new status `rollback_failed`; writes `receipt.json` next to the
  snapshot; the applied log line names the snapshot and rule.
- `services/orion-dream/app/rem_compaction.py`: `downscale_entries`, `consolidate_evidence_ids`, a
  `salience_loader` seam. A salience read failure still stages the gist delta.
- `services/orion-dream/app/rem_store.py`: `load_thought_salience` (read-only).
- `services/orion-dream/app/compaction_consent.py`: `ask_consent_and_apply` replaces
  `ask_consent_without_store` (removed); `preview=True` asks with no store.
- `services/orion-dream/app/main.py`: `_rem` preloads salience in a thread and applies; `/dreams/rem-preview`
  asks with `preview=True` and never writes.
- `services/orion-dream/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: flags and docs.
- `services/orion-dream/tests/test_compaction_store.py` (new), `test_compaction_consent.py`,
  `orion/physiology/tests/test_consent.py`, `services/orion-durable-runs/tests/test_physiology_consent_market.py`:
  tests follow the shipped consent; the needs_consent path is tested with the standing consent removed.
- `services/orion-durable-runs/evals/consent_gate_fixture.py`: shipped allow / prune refuse cases; live
  check of applied downscales.

## Schema / bus / API changes

- Added: `CompactionApplyReceiptV1.cards_skipped_no_gist_store` and status `rollback_failed` (not
  bus-published; returned, logged, and written as `receipt.json`).
- Removed: `app.compaction_consent.ask_consent_without_store` (replaced, no other callers).
- Renamed: none.
- Behavior changed: `MemoryCompactionDeltaV1.downscale` is now populated by REM;
  `metrics.edges_downscaled` is now nonzero. The hub preview reads the same staging table unchanged.
- Compatibility notes: no bus channel or registry change. No migration.

## Env/config changes

- Added keys: `DREAM_COMPACTION_DOWNSCALE_FACTOR=0.8` (validated in (0, 1); a bad value fails at boot).
- Changed defaults: `ORION_DREAM_COMPACTION_APPLY_ENABLED` false -> true;
  `DREAM_COMPACTION_SNAPSHOT_DIR` `/tmp/dream-compaction-apply` -> `/app/logs/compaction-snapshots`.
- Compose now passes `ORION_DREAM_COMPACTION_APPLY_ENABLED`, `ORION_DREAM_COMPACTION_DOWNSCALE_ONLY`,
  `DREAM_COMPACTION_SNAPSHOT_DIR`, `DREAM_COMPACTION_DOWNSCALE_FACTOR` with those defaults.
- `.env_example` updated: yes.
- local `.env` synced: `python3 scripts/sync_local_env_from_example.py orion-dream --all-keys --force`
  (scoped to orion-dream; changed exactly these 3 keys).
- skipped keys requiring operator action: none.

## Proposal mode (CLAUDE.md, memory change)

**What capability changes.** Sleep can now change memory. It lowers the salience of reverie thoughts from
chains that settled, by 20%, once per thought. It cannot raise a value, delete a row, write a card, or
touch any other table or column.

**What data is touched.** One column: `substrate_reverie_thought.salience`, for thought ids named by a
staged, consented delta. The original value stays in `thought_json.salience` (never written). The
before-state is also written to `/app/logs/compaction-snapshots/<delta_id>/snapshot_before.json`
(host: `${ORION_DATA_ROOT:-/mnt/graphdb}/logs/dream/logs/compaction-snapshots/`). Reads: the same table
(salience and `thought_json->'salience'`).

**Privacy boundary.** No new data leaves the database. The snapshot file holds thought ids and numbers
only, no reverie text. No gist card or dream content is written anywhere, so nothing dream-derived can
reach fact-grade recall. The test fixture copies ids and salience values from live Postgres, never text.

**The trace that proves it worked.**
- Log line in orion-dream: `compaction applied delta=<id> downscaled=N ... snapshot=<path>
  rule=dream.compaction.downscale_only`, then `rem compaction consent asked id=<id> status=applied ...`.
- `want_ledger` row: `want_id = dream.compaction:compact:<delta_id>`, consent verdict `allow`,
  consent_ref `standing:dream-compaction-downscale`.
- The snapshot file and `receipt.json` (status `applied`) on the host, under
  `/mnt/graphdb/logs/dream/logs/compaction-snapshots/<delta_id>/`.
- SQL: `SELECT count(*) FROM substrate_reverie_thought WHERE salience < (thought_json->>'salience')::float`
  moves above 0.
- Eval: `PYTHONPATH=. python services/orion-durable-runs/evals/consent_gate_fixture.py --live --days 3`
  checks that every lowered thought is a target of an allowed delta and that none was raised. Before the
  rebuild it says UNVERIFIED.

**The dangerous failure mode.**
1. Prune slips through. Guarded three times: the rule is `refused` (the parser refuses any other consent
   for an irreversible rule), the consent ask for a pruning plan is verb `prune`, and the store raises on
   `prune`. Mutation-checked.
2. Downscale compounds every sleep until salience collapses. Guarded: REM proposes a thought only while
   its column equals the original in `thought_json`; the store's UPDATE is compare-and-set on the old value;
   a value too small to lower (or that would round to 0) is skipped.
3. A partial apply leaves memory half-changed. Guarded: snapshot first; all downscales of a delta run in
   one transaction (a stale or missing row writes nothing); restore is compare-and-set, so it never undoes
   another writer; a failed restore reads `rollback_failed`, never `rolled_back`.
4. Salience is lowered on thoughts Orion still needs. Bounded: only thoughts of chains that reported
   `reverie_chain_settled`; lowered by 20%, not zeroed; restorable.

**How to disable or roll back.**
- Stop all applies: set `ORION_DREAM_COMPACTION_APPLY_ENABLED=false` in `services/orion-dream/.env` and
  rebuild dream. Or delete the standing consent from `consent.yaml` (a PR) and rebuild durable-runs: every
  ask returns to `needs_consent`.
- Roll back one apply: `docker exec orion-athena-dream python -m app.compaction_store restore
  /app/logs/compaction-snapshots/<delta_id>/snapshot_before.json`.
- Roll back every apply at once (read the original from `thought_json`):
  `UPDATE substrate_reverie_thought SET salience = (thought_json->>'salience')::float WHERE
  jsonb_typeof(thought_json->'salience') = 'number' AND salience < (thought_json->>'salience')::float;`
  (a production write: Juniper runs it, not an agent).

## Metric gate (salience is an existing signal, not a new one)

No new metric is wired in. This changes an existing signal's values. Recorded anyway: the producer is
`services/orion-thought/app/store.py` (INSERT); the consumer that changes behavior is
`orion/orion_day/budget.py` (sort by salience). Live, 3 days: 695 thoughts, salience 0.25 to 1.0,
median 0.73, 37 distinct values, column equals `thought_json` on 695 of 695, so "already lowered" is
detectable. Theory anchor: the synaptic homeostasis hypothesis (Tononi and Cirelli): sleep downscales
proportionally, so relative order among the downscaled is kept (tested). Reversible: yes, the original is
in every row.

## Tests run

```text
pytest services/orion-dream/tests -q                                   382 passed, 9 skipped
pytest orion/physiology/tests -q                                       311 passed
(cd services/orion-durable-runs) pytest tests/test_physiology_consent_market.py -q   8 passed
python scripts/check_env_template_parity.py                            PASS
python scripts/check_env_key_single_source.py                          OK
python scripts/check_compose_no_relative_mounts.py                     PASS
python scripts/check_metric_lineage.py --gate                          PASS
python scripts/check_definition_drift.py --gate                        PASS
python scripts/check_service_hostname_refs.py                          OK
```

Read-only live check of the new Postgres reader (`load_thought_salience`, Postgres-only SQL), in a
`default_transaction_read_only=on` session: 50 requests, 142 ids, 142 rows, 0 already lowered; delta 50
cards, 142 downscale, 0 prune.

Mutation checks (each guard broken, tests run, file restored):

```text
M1 gate prune refusal removed (consent.py)           dream 1 failed | physiology 2 failed
M2 applier consent-required removed                  dream 12 failed
M3 store prune refusal removed                       dream 1 failed
M4 prune plan disguised as compact                   dream 4 failed
M5 store upscale guard removed                       dream 2 failed
M6 compare-and-set removed                           dream 1 failed
M7 missing standing consent -> allow (consent.py)    dream 2 failed | physiology 4 failed | eval FAIL (6)
After review fixes:
M8 restore compare-and-set removed                   dream 2 failed
M9 batch downscale not atomic                        dream 1 failed
M10 rollback_failed reported as rolled_back          dream 2 failed
M2 (rerun) applier consent-required removed          dream 12 failed
```

## Evals run

```text
PYTHONPATH=. python services/orion-durable-runs/evals/consent_gate_fixture.py           PASS (13 checks)
PYTHONPATH=. python services/orion-durable-runs/evals/consent_gate_fixture.py --live --days 3
  live 3 d: staged deltas=4 consent asks=5 (all needs_consent: pre-deploy)
  live downscale: lowered thoughts=0 raised=0 lowered without an allowed delta=0
  UNVERIFIED: no applied downscale yet
  apply receipts in /mnt/graphdb/logs/dream/logs/compaction-snapshots: none               PASS
```

## Docker/build/smoke checks

```text
docker build -f services/orion-dream/Dockerfile -t orion-dream-wt-downscale-check .     ok (scratch tag, removed)
docker run ... python -c "import app.main; ..."  -> apply True dir /app/logs/compaction-snapshots factor 0.8 ReverieSalienceStore
docker run ... python -m app.compaction_store --help  -> restore CLI present
docker exec orion-athena-dream: /app/logs writable as root (snapshot dir parent)
```

`scripts/safe_docker_build.sh orion-dream build` needs a root `.env`, which a worktree does not have; the
plain build above was used instead.

## Review findings fixed

A review subagent reviewed PR #3254 and found no blockers. It ran the new SQL (salience loader, snapshot)
against live Postgres in a read-only session and confirmed it works. All 9 findings are fixed:

- Finding: restore overwrote every snapshotted row, so an overlapping apply's rollback could undo another
  writer's committed value.
  - Fix: the snapshot records the planned value; restore is compare-and-set on it.
  - Evidence: `test_a_failed_apply_never_undoes_another_writer`; mutation M8 caught.
- Finding: each downscale committed on its own, so a killed container could leave half a delta.
  - Fix: `downscale_many` runs the whole delta in one transaction; the applier prefers it.
  - Evidence: `test_a_stale_target_writes_nothing_even_when_restore_cannot_run`; mutation M9 caught.
- Finding: a failed restore still read `rolled_back`.
  - Fix: new status `rollback_failed`, logged at error.
  - Evidence: `test_never_raises_even_if_rollback_also_fails`; mutation M10 caught.
- Finding: the outcome lived only in logs, and want_ledger cannot tell a preview's allow from an applied one.
  - Fix: `receipt.json` next to each snapshot; the live eval reads receipts and fails on `rollback_failed`.
    The want_ledger ambiguity remains (see Risks); the receipts are the durable record of what applied.
  - Evidence: lifecycle test reads `receipt.json`; eval prints receipt counts.
- Finding: the preview endpoint read salience synchronously on the event loop.
  - Fix: it loads requests and salience in threads, like the sleep cycle.
  - Evidence: `test_the_preview_endpoint_asks_with_no_store`.
- Finding: tiny values could make no-op or zero proposals.
  - Fix: REM skips unless 0 < new < old; the store requires new < old.
  - Evidence: `test_downscale_entries_skip_unknown_lowered_and_bad_factor`, new `0.742 -> 0.742` refusal case.
- Finding: the factor had no validation.
  - Fix: `Field(gt=0, lt=1)`.
  - Evidence: `test_apply_ships_on_with_a_persistent_snapshot_dir` checks 0, 1 and 8 fail.
- Finding: source-string tests and an env-dependent settings test.
  - Fix: behavioral tests drive the real `_rem` and `rem_preview_endpoint`; settings test reads code
    defaults.
  - Evidence: `test_the_sleep_cycle_loads_salience_stages_persists_then_applies`.
- Finding: docs (docstring count, line length, two salience numbers, snapshot retention).
  - Fix: README and docstrings updated.

## Restart required

From the primary checkout on `main`, after merge and `git pull`. durable-runs first (it reads the
consent), then dream:

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-dream up -d --build
```

Verify after the next sleep:

```bash
docker logs orion-athena-dream 2>&1 | grep -E "compaction applied|rem compaction consent asked" | tail -5
```

## Risks / concerns

- Severity: medium. Concern: the first sleep after deploy lowers up to 200 thoughts at once (the cap),
  if many settled requests are in its window. Mitigation: each is -20% once, snapshotted, and restorable.
- Severity: low. Concern: gist cards are still not written (no gist table). The receipt counts them as
  skipped, so the gap is visible. Mitigation: a gist-card store is a separate design (dream content into
  memory needs its own privacy review).
- Severity: low. Concern: live apply is UNVERIFIED until both services rebuild and Orion sleeps.
- Severity: low. Concern: `/dreams/rem-preview` records an `allow` in want_ledger that is never applied,
  so "allowed" in want_ledger is a superset of "applied". Mitigation: `receipt.json` per real apply; the
  live eval reports receipts.
- Severity: low. Concern: snapshot directories accumulate (one small directory per applied sleep).
  Mitigation: none yet; a retention rule can follow once real volume is known.
- Severity: low. Concern: the Postgres SQL of `load_thought_salience` and the store has no CI test (CI
  uses SQLite). Mitigation: both were run by hand against live Postgres, read-only, by the author and the
  reviewer.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3254
