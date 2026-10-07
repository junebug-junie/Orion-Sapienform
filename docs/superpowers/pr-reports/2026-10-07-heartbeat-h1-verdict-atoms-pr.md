# Heartbeat H1 verdict atoms + per-producer unrouted counts

## Summary

- orion-heartbeat now writes a small, bounded record of its H1 verdict onto
  `orion:grammar:event`. It publishes a transition atom when the verdict
  class changes and then holds for 3 ticks (~90 s), and one summary atom per
  hour. Both are marked with the trace prefix `heartbeat.h1:`.
- Nothing reaches the field. Live history shows the verdict has no rest
  state and does not follow organ activity (numbers below), so no honest
  consumer exists yet. The atom is a trace in the ledger only.
- The organ→site map is unchanged. Grammar producers heartbeat cannot route
  are now counted per producer. They show on `/health` and in the hourly
  atom, and are named against the channels.yaml catalog. The previous
  single anonymous `events_skipped_organ` total is still there alongside.
- Heartbeat skips its own atoms when they come back on the shared channel,
  so it never absorbs its own output.
- `orion-heartbeat` is added to the `orion:grammar:event` producers in
  channels.yaml (the catalog gate checks this). The metric lock is
  regenerated for that real producer change.
- Adds a replay eval, `services/orion-heartbeat/evals/replay_organ_map_options.py`.

## Outcome moved

Before this, the heartbeat's verdict only existed in a 30 s HTTP poll and in
one AST/HOT field. Now it leaves a durable, debounced record in
`grammar_events` at about 1 transition per hour (on the same history, a naive
"emit on every change" would be ~32 per hour). It also leaves an hourly
summary that says when H1 failed instead of going quiet. Grammar dropped by
heartbeat now names its producers: about 40k atoms a day from llm-gateway,
sql-writer, vision-frame-router, harness-governor, substrate-runtime and
gpu-pool. Before, they were one unlabeled counter.

## Current architecture

- orion-heartbeat consumed `orion:grammar:event` and routed 5 hardcoded
  organs onto boundary sites 0-4 of a 10-site MPS (`routing.ORGAN_SITE_MAP`).
  Every other producer was dropped as `UnroutableOrganError` into one
  anonymous counter.
- H1 ran every 30 s. Its result was published nowhere. orion-substrate-runtime's
  AST/HOT tick already polls `/h1` (`SUBSTRATE_HEARTBEAT_H1_URL`) and stores
  it in `substrate_attention_self_model.heartbeat_*`. That was the source
  for the 7-day replay below.
- Readers of `grammar_events` in substrate-runtime select by
  `source_service = ANY(...)` and `trace_id LIKE prefix`, one reducer per
  producer. No reducer exists for `orion-heartbeat`, so these atoms cannot
  reach the field.

## Architecture touched

- orion-heartbeat: a new publisher path inside the H1 loop, a self-skip on
  intake, per-producer unrouted counters, and catalog naming.
- Contract: `orion/bus/channels.yaml` adds `orion:grammar:event`
  producer `orion-heartbeat`. Schema: existing `GrammarEventV1`, unchanged.
- No field topology, reducer, or semantic-layer YAML changes.

## Files changed

- `services/orion-heartbeat/app/substrate/verdict_atoms.py`: new, pure.
  Contains the debounced transition tracker, the hourly window, and the
  event builders.
- `services/orion-heartbeat/app/service.py`: publishes from the H1 loop with
  a cap and failure accounting; self-skip; per-producer unrouted counts; new
  `/health` fields.
- `services/orion-heartbeat/app/substrate/routing.py`: adds
  `catalog_grammar_producers()` and `SELF_SOURCE_SERVICE`.
- `services/orion-heartbeat/app/settings.py`, `.env_example`,
  `docker-compose.yml`: 4 new `HEARTBEAT_VERDICT_*` keys, all ON.
- `services/orion-heartbeat/requirements.txt`: explicit PyYAML (the catalog
  loader).
- `services/orion-heartbeat/tests/test_verdict_atoms.py`: new tests.
- `services/orion-heartbeat/evals/replay_organ_map_options.py`: the
  organ-map replay.
- `services/orion-heartbeat/README.md`: what it emits, why nothing reaches
  the field, and the organ-map decision.
- `orion/bus/channels.yaml`: the producer entry, and a consumer note.
- `config/metrics/metric_definitions.lock.json`: re-locked (`routing_changed`
  for the grammar channel producers).

## Schema / bus / API changes

- Added: `orion-heartbeat` as a producer on `orion:grammar:event`. There are
  two atom roles, `h1_verdict_transition` and `h1_hourly_summary`, with
  `atom_type=observation` and `layer=substrate`.
- Removed: none.
- Renamed: none.
- Behavior changed: heartbeat publishes, where before it published nothing.
  `/health` gains `events_skipped_self`, `events_skipped_organ_by_source`,
  `catalog_loaded`, `catalog_producers_unrouted`, `uncatalogued_sources_seen`
  and `verdict_atoms`.
- Compatibility notes: these are additive. `GrammarEventV1` is unchanged.
  sql-writer persists the atoms like any other producer's.

## Env/config changes

- Added keys: `HEARTBEAT_VERDICT_ATOMS_ENABLED=true`,
  `HEARTBEAT_VERDICT_SETTLE_TICKS=3`,
  `HEARTBEAT_VERDICT_SUMMARY_INTERVAL_SEC=3600.0`,
  `HEARTBEAT_VERDICT_MAX_TRANSITIONS_PER_HOUR=12`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-heartbeat`:
  yes. The primary checkout's `services/orion-heartbeat/.env` now has all 4
  keys, with the flag ON.
- Skipped keys requiring operator action: none.

## Why nothing reaches the field (metric gate, §0A)

Evidence: `substrate_attention_self_model`, 16,642 samples from 2026-09-30 to
10-07. This is the verdict as AST/HOT recorded it.

1. **Provenance.** `compute_h1_ensemble` → `classify_ensemble_verdict`
   (`reconstruction.py`) over the MPS ensemble.
2. **Independence.** The reheat comes from `bus_synaptic` (FalkorDB), which
   already drives a field signal. The absorbed atoms are the same grammar
   that substrate-runtime reduces. So it is not independent at the input.
3. **Theory anchor.** Holographic boundary/bulk redundancy (the design doc)
   is a hypothesis. It has no validated mapping to any of Orion's states.
4. **Live sanity.** This fails.
   - The verdict never reaches a rest state. `mean_ratio` minimum was
     0.668. The silence branch (`<= 0.2`) fired 0 times.
   - Every `concentrated` tick (2,792 of 2,792) came from the bulk band.
     That band's edges are percentiles of heartbeat's own past output, so it
     reads about 17% concentrated by construction.
   - The concentrated share is 14-20% in every UTC hour; it does not follow
     the day.
   - `std_ratio` against total organ fires: r = -0.008. Bulk against fires:
     r = 0.022.
   - Raw class changes run at 31.8 per hour, and `redundant` runs never
     last more than 3 samples.
   - Replaying the same 3 h of atoms with only the RNG seed changed gives a
     different verdict on 31-35% of ticks.

**Conclusion:** I shipped the atom only. I invented no field channel.
AST/HOT already reads `/h1`.

## Organ-map decision (replay)

**Decision:** keep the hand-checked 5-organ `ORGAN_SITE_MAP` and count the
producers it drops, per producer, named against the channels.yaml catalog.
Do not derive site assignment from the catalog.

Replay: `services/orion-heartbeat/evals/replay_organ_map_options.py` ran the
service's own ensemble code over real `grammar_events`. Two 3 h windows,
about 300 H1 ticks each. Reheat was held at 0.0054, and the first 30 min was
dropped as warm-up.

| Window | Option | concentrated | mixed | redundant | ticks agreeing with current |
| --- | --- | --- | --- | --- | --- |
| 10-06 21:14 - 10-07 00:15 | current (seed 42) | 25.4% | 74.3% | 0.3% | -- |
| | fold 6 extras onto boundary 0-4 | 20.5% | 71.6% | 7.9% | 62.7% |
| | extras onto bulk 5-8 | 19.5% | 77.6% | 3.0% | 68.6% |
| | control: current, seed 142 | 24.8% | 73.9% | 1.3% | 69.3% |
| | control: current, seed 242 | 25.7% | 71.3% | 3.0% | 65.3% |
| 10-05 21:14 - 10-06 00:13 | current | 28.0% | 70.7% | 1.3% | -- |
| | fold onto boundary | 28.0% | 69.3% | 2.7% | 67.3% |
| | onto bulk 5-8 | 24.0% | 74.0% | 2.0% | 69.3% |

What the table shows:

- Seed noise alone flips about a third of ticks, so per-tick agreement
  cannot separate a remap from noise.
- At distribution level, folding onto the boundary moved `redundant` from
  0.3% to 7.9% and `concentrated` from 25% to 20% in window 1. That is
  outside the seed spread. In window 2 it barely moved.
- Bulk placement changes what "bulk" means. Site 9 cannot take an organ at
  all.

A remap would silently change H1, so the drops are counted instead. The
unrouted share is about 40k atoms a day (about 21% of the atoms heartbeat
sees).

## Tests run

```text
pytest services/orion-heartbeat/tests -q                  -> 147 passed
pytest tests/test_grammar_event_producer_catalog.py -q     -> 3 passed
python scripts/check_definition_drift.py --gate            -> PASS (re-locked once for the producer change)
python scripts/check_metric_lineage.py --gate              -> PASS
python scripts/check_env_template_parity.py                -> PASS
python scripts/check_service_env_compose_parity.py orion-heartbeat -> OK (27 keys)
other orion-static-gates (inner_state, hostname refs, relative mounts, system_health producers, async routes, sentience instruments) -> all pass
```

## Evals run

```text
Manual research eval (needs a live CSV export, not CI):
services/orion-heartbeat/evals/replay_organ_map_options.py
  2 windows x {current, fold_boundary, extend_bulk} + 2 seed controls -> table above
Debounce replay over substrate_attention_self_model (16,642 samples, 7 days):
  settle 1 -> 31.8 atoms/h, 2 -> 5.0/h, 3 -> 1.1/h, 4 -> 0.24/h
```
No automated eval harness exists for orion-heartbeat. Follow-up: turn the
replay into a fixture-backed eval on a small committed atom sample.

## Docker/build/smoke checks

```text
docker build -f services/orion-heartbeat/Dockerfile -t orion-heartbeat-verify:h1atoms .   -> OK
In-image: catalog_loaded=True, catalog_producers_unrouted = 6 producers, flag ON, settle 3
Isolated smoke (throwaway redis on a private docker network, NOT the Orion bus;
H1 every 2 s, settle 1, summary every 5 s):
  /health verdict_atoms.published=2 (1 transition from=none to=mixed, 1 summary),
  events_seen=2, events_skipped_self=2 -> own atoms echoed back and skipped.
Smoke containers, network and image removed afterwards. Production not touched.
```

## Review findings fixed

The review subagent ran on `git diff origin/main...HEAD`. It found no blockers.

- Finding: the README still said "publishes nothing", and the referenced
  README section was missing.
  - Fix: updated the intro, the scope bullet and the thresholds paragraph.
    Added "What heartbeat emits", "Why nothing reaches the field" and the
    organ-map section.
  - Evidence: `services/orion-heartbeat/README.md`.
- Finding: a transition whose publish failed was lost without trace.
  - Fix: `transitions_publish_failed` in the hourly summary. The next
    published atom carries `suppressed_since_last`.
  - Evidence: `test_publish_failure_is_counted_not_raised`.
- Finding: a capped transition made the next atom's `from=` look like a gap.
  - Fix: added `suppressed_since_last=N` on the next published transition.
  - Evidence: `test_cap_frees_up_after_a_rolling_hour_and_reports_suppressed`.
    This test also covers the rolling-hour expiry.
- Finding: the tracker's streak list grew without bound during a long hold.
  - Fix: constant-size state (count, since, std sum).
  - Evidence: `test_long_hold_keeps_constant_state_and_true_held_since`.
- Finding: the "dead loop reads as h1_ticks=0" claim was overstated.
  - Fix: reworded. A failed H1 computation reads as `h1_ticks=0`; a dead
    loop reads as missing summary rows.
  - Evidence: `test_h1_loop_failure_still_flushes_an_absence_summary`.
- Finding: an unreadable catalog showed as `catalog_producers_unrouted: []`.
  - Fix: added `catalog_loaded`. The list is `null` when the catalog is
    unknown.
  - Evidence: `test_unreadable_catalog_reads_unknown_not_empty`.
- Finding: the self-skip test used a hand-built dict.
  - Fix: it now feeds back the exact published wire payload. Also added
    loop-glue tests (failure path, flag off).
  - Evidence: `test_own_published_atom_echo_is_skipped_...` and
    `test_h1_loop_publishes_nothing_when_disabled`.
- Finding: the per-source key was not sanitized, and settings had no bounds.
  - Fix: keys go through `safe_token`; `ge=1`/`gt=0` on the new settings.
  - Evidence: `test_settings_reject_degenerate_values`.
- Finding: was the lock's `_last_change` replacement legitimate?
  - Fix: none needed. It is the tool's normal behavior: the block describes
    this branch against its merge base.
  - Evidence: `check_definition_drift.py --gate` passes.

## Restart required

Not deployed. To deploy, from the primary checkout on main after merge:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && docker compose --env-file .env --env-file services/orion-heartbeat/.env -f services/orion-heartbeat/docker-compose.yml up -d --build heartbeat
```

Verify: `curl -s localhost:7251/health | jq .verdict_atoms`. Within ~2 min
there should be one `from=none` transition row:
`docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select created_at, event_json->'atom'->>'summary' from grammar_events where source_service='orion-heartbeat' order by created_at desc limit 5"`

## Risks / concerns

- Severity: low
  - Concern: about 30 grammar rows a day are added to `grammar_events`. That
    ledger has 3-day retention and ingests about 400k rows a day.
  - Mitigation: the cap and the debounce. The flag is
    `HEARTBEAT_VERDICT_ATOMS_ENABLED`.
- Severity: low
  - Concern: a future generic reader of `grammar_events` (one that does not
    filter by source) would see these atoms.
  - Mitigation: they are `observation` atoms with a distinct prefix and
    source, and their summaries say they are ledger-only.
- Severity: info
  - Concern: the replay is a manual research eval that needs a live CSV
    export. It is not a CI eval.
  - Mitigation: the procedure is documented in the script docstring.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
