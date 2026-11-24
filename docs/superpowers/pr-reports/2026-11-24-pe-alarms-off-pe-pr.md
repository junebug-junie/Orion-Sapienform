## Summary

Orion's three alarms (cabinet warming, strain news, street event) stop living in the prediction-error slot. Before this patch Orion's own dig tools and metacog still met them as "prediction errors". Refs #3808 (design rev 2), alarms steps 2 and 3.

- **Writers.** The alarms write their value to node metadata `signal_value` (with `signal_kind: alarm`) and to `substrate_node_signal_history` only. No `prediction_error` key, no PE-table row. The first write after deploy clears the old key on the graph node.
- **Readers.** Every history reader reads one shared source with a per-node read rule. Normal prediction-error nodes come from the PE table. The alarms are read from both tables (deduped on `(node_id, observed_at)`) until their old PE rows age out by the normal 7-day prune, around 2026-12-01. Nothing is deleted.
- **Meaning.** The glossary files the three under their own channel, `alarm`. World-first attention and self-readings look each node up under its own channel. All three are still contest candidates and are still judged as triggers. The walkway's p99 floor (`ATTENTION_MIN_BAND_BY_NODE_ID`) is unchanged.
- **Words.** Self-readings, the node-history dig tool, metacog episode refs, labels and glossary text call them alarms and cite `substrate_node_signal_history`. They never say "prediction error".
- **Fences.** CI fails any producer that writes `prediction_error` for an alarm id. At runtime, the PE writer refuses an alarm id and the store drops an alarm row sent to the PE table. A stale `prediction_error` key on an alarm node is never read.

## Outcome moved

- A pixel-free alarm firing can no longer reach Orion as "my prediction error". Orion's dig tool now says, for example: "node:substrate.cabinet (cabinet warming alarm: an alarm, above 0 = it fired; stored in substrate_node_signal_history)".
- The design's dangerous failure 1 is prevented. Strain news no longer goes silent around 2026-12-01, when the old PE rows age out. The restart seed and warm-up count now read both tables.

### Before numbers (live, read-only, 2026-11-24 ~08:50Z)

Alarm contest win share. Relative rates, per Juniper's rule. `substrate_attention_frames` keeps only 3 days (122,770 frames from 11-21 08:40Z, about one frame every 2.1 s), so this is a 3-day window, not 7.

| node | own events in window | eligible frames | expected eligible (events x 300 s / 2.11 s) | eligibility | wins (rank 1) | win when eligible |
|---|---|---|---|---|---|---|
| strain | 6 | 830 | 853 | 0.97 | 641 | **0.772** |
| cabinet | 0 (0 in 7 d) | 0 | 0 | n/a | 0 | n/a |
| street | 0 (1 ring in 7 d, 11-20 23:02Z, outside the frame window) | 0 | 0 | n/a | 0 | n/a |

- Strain news events in 7 days: **14** (value > 0 rows). The old PE-only count and the read-both count are equal.
- Restart seed warm-up count for strain, 7 days: old reader **3,007**, read-both **3,007**. Equal today because Patch 1 dual-writes and the backfill was run. After cutover the old reader would fall below `DEFAULT_MIN_READINGS` (200) about 6.5 days later and strain news would stop. The read-both reader stays near 3,000.
- History source equivalence: over 7 days, the new shared source returns exactly the old PE query's rows (23,964 vs 23,964; 0 in either `EXCEPT ALL` direction). Today's contest inputs are therefore identical, so the expected "after" win share equals "before". The real after-number is **UNVERIFIED** until it is deployed and re-queried (see the verify-live issue).

## Current architecture

- `services/orion-substrate-runtime/app/worker.py` wrote all five signal nodes through `_write_prediction_error_node` (`metadata.prediction_error`). The history sampler wrote PE-table rows and dual-wrote the five to `substrate_node_signal_history` (Patch 1, #3800). Strain also wrote its event row to the PE table directly.
- Every history reader read `substrate_node_prediction_error_history` only: attention-runtime store, equilibrium `pe_history_reader`, thought `self_readings_cue`, the substrate-runtime seed, `orion/introspect/node_history.py`, and the strain restart seed.
- The glossary had the three alarms under `channel: prediction_error` with `value_kind: trigger`. `world_first` looked up only `prediction_error`.

## Architecture touched

- Contract: the node metadata keys `signal_value` and `signal_kind`, persisted by `falkor_codec` and `falkor_store` (hydration, externally-owned skip set).
- New stdlib-only module `orion/prediction/signal_history_sql.py`. It holds the moved set, the per-node read rule (`HISTORY_READ_MODE_BY_NODE_ID`: `read_both` / `signal_only`), and the shared derived-table SQL (no bind parameters, so it works under sqlalchemy, psycopg2 and asyncpg). Per design rev 3 (#3827), the cameras are not in any read-both union. Patch 3 flips them to `signal_only`.
- Readers: substrate-runtime, attention-runtime, equilibrium, thought, Hub node-history dig tool (through `orion/introspect`).
- Glossary and metric lock: the three URNs move from `.prediction_error` to `.alarm`.
- No bus channel, no schema registry, no migration change.

## Files changed

- `orion/prediction/signal_history_sql.py` (new): moved set, per-node read rule, shared history SQL, glossary channel per node.
- `orion/substrate/signal_history.py`: re-exports; `node_signal_value`, `split_history_samples`, metadata key names.
- `orion/substrate/falkor_codec.py`, `orion/substrate/falkor_store.py`: persist `signal_value` / `signal_kind`.
- `services/orion-substrate-runtime/app/worker.py`: `_write_signal_node`; the PE writer refuses alarms; cabinet/strain/street use the signal writer; the sampler routes alarm rows to the signal table only; the strain seed reads both tables; signal prune always runs.
- `services/orion-substrate-runtime/app/store.py`: PE-table write drops alarm rows; seed fetch uses the shared source; `fetch_signal_history_read_both` defaults to the read-both ids only.
- `services/orion-substrate-runtime/app/brain_frame_producer.py`, `orion/substrate/pressure.py`, `orion/substrate/attention_broadcast.py`: read an alarm's `signal_value`, never its stale `prediction_error`.
- `services/orion-attention-runtime/app/store.py`, `services/orion-equilibrium-service/app/pe_history_reader.py`, `services/orion-thought/app/self_readings_cue.py`, `orion/introspect/node_history.py`: shared history source.
- `orion/attention/world_first.py`, `orion/attention/self_readings.py`: glossary lookup under the node's own channel; self-reading refs cite the signal table.
- `orion/metacog/self_signal_episode.py`: episode refs cite the signal table for alarms.
- `config/field/field_channel_glossary.v1.yaml`: three entries move to `channel: alarm`, with a "re-test" line for Orion.
- `orion/metrics/semantics.py`, `config/metrics/metric_definitions.lock.json`: URN rename.
- `scripts/check_pe_alarm_readers.py`, `config/substrate/prediction_error_readers.yaml`: moved-node writer fence; registry updated.
- `orion/regulation/strain_news.py`, `orion/substrate/prediction_error_freshness.py`: docstrings.
- `services/orion-substrate-runtime/{settings.py,.env_example,README.md}`: the flag's meaning narrowed (comments only); the Patch 2 section.
- `.github/workflows/retired-pe-guards.yml`: new tests and paths.
- Tests: `tests/test_pe_alarms_off_pe.py` (new), `services/orion-substrate-runtime/tests/test_alarm_signal_writer.py` (new), and updates to existing alarm, glossary, metric, codec, strain, cabinet, street, dual-write and node-history tests.

## Schema / bus / API changes

- Added: node metadata `signal_value` (float) and `signal_kind` (`alarm`) on the three alarm nodes. Glossary channel `alarm`. Metric URNs `node:substrate.{cabinet,strain,street}.alarm`. Introspect item kind `node_alarm_history` for alarm nodes.
- Removed: `prediction_error` key and PE-table rows for the three alarms (the old rows age out). URNs `...{cabinet,strain,street}.prediction_error`.
- Renamed: none.
- Behavior changed: history readers read the alarms from both tables; `SUBSTRATE_SIGNAL_HISTORY_ENABLED` now gates only the camera dual-write.
- Compatibility notes: readers are switched in the same patch as the writers, and they read both tables. Either deploy order keeps the alarms' full 7-day window. Old readers that are not redeployed would see the alarms' history thin out over 7 days, so restart all consumers listed below.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: the comment for `SUBSTRATE_SIGNAL_HISTORY_ENABLED` only (its meaning narrowed). The value is unchanged (`true`).
- local `.env` synced: the key already exists with `true`; no key change, so there was nothing to sync.
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider, compared with main:
  tests/test_pe_alarms_off_pe.py tests/test_pe_alarm_refile.py tests/test_check_pe_alarm_readers.py  all pass
  services/orion-substrate-runtime/tests: 17 failed / 1 collection error, the SAME set as main (pre-existing); all new and updated tests pass
  services/orion-attention-runtime tests+evals: 0 failed (main has 2 extra tests from a later merge)
  services/orion-equilibrium-service tests+evals: 289 passed (main 289)
  services/orion-thought test_stance_self_readings: 26 passed; Hub node-history + attention organ: 52 passed
  orion/substrate|attention|introspect|metacog|field|regulation tests: see the CI section; local failures match main
  root alarm/glossary/metric/world-first suite: see the CI section; failures match main
```

## Evals run

```text
services/orion-attention-runtime/evals and services/orion-equilibrium-service/evals: pass (unchanged).
Live read-only equivalence check (history source vs old PE query, 7 d): 23,964 = 23,964 rows, 0 diff.
No new eval harness: this patch changes where values are stored and how they are named, not what is judged.
The contest-win-share comparison after deploy is the live check (verify-live issue).
```

## Docker/build/smoke checks

```text
Not run: no Dockerfile, compose or dependency change. orion/prediction/ is already copied into every image that copies orion/.
Live SQL smoke (read-only, sqlalchemy text() against conjourney): the shared source compiles with one bind
parameter (:since); node ids such as 'node:substrate.cabinet' are not taken as bind parameters; it returns
all 12 live nodes. EXPLAIN: the WHERE predicates are pushed into both branches; 7-day read takes 87 ms.
```

Static gates (derived from `.github/workflows/orion-static-gates.yml`): `check_definition_drift --gate`, `check_env_key_single_source`, `check_inner_state_registry`, `check_integration_seam --require-base`, `check_metric_lineage --gate` / `--prompt-semantics`, `check_migration_required_by`, `check_pe_alarm_readers`, `check_scripts_dir_no_stdlib_shadow`, `check_sentience_instruments --static-only`, `check_system_health_producers`, `check_async_routes_not_blocking`, `check_env_template_parity`. All pass. The fence was mutation-tested: switching the cabinet and strain writers back to `_write_prediction_error_node` fails the gate at `worker.py:1520` and `:2157`.

## Review findings fixed

See the review section appended below.

## Restart required

Order: readers first, then the writer. Readers read both tables, so either order is safe. Readers-first means no reader is ever on the old PE-only query while the writer has stopped writing PE rows.

```bash
scripts/safe_docker_build.sh orion-attention-runtime up -d --build
scripts/safe_docker_build.sh orion-equilibrium-service up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

(Run from the clean primary `main` checkout after merge. The post-pull rebuild picks these up from the diff.)

## Risks / concerns

- Severity: medium. Concern: a reader that was not rebuilt keeps the old PE-only query and sees the alarms' history thin out over 7 days. Mitigation: the restart list above. The verify query checks zero new PE rows for the alarms.
- Severity: low. Concern: rollback by revert brings back PE-only readers, which would miss the signal-only rows written during cutover. Mitigation: rows are not deleted, and a revert writes PE rows again from the next tick.
- Severity: low. Concern: the evidence list key on alarm nodes keeps its old name (`prediction_error_evidence_event_ids`). It is attribution only, not read for alarms, and not shown to Orion.
- Issues opened: see the PR link section. Issues closed: none.

## PR link

(filled in after creation)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
