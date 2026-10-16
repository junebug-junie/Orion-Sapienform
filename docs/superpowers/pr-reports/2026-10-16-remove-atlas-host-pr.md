# Remove atlas as a live host

## Summary

- atlas (a machine taken apart on 2026-08-21) moves out of the live node list into a new
  `retired_nodes:` section of `config/biometrics/node_catalog.yaml`.
- The biometrics loop now drops retired hosts from its two live tables on the next tick
  (`prune_retired_nodes` in `orion/substrate/biometrics_loop/pipeline.py`). The node reducer,
  the pressure organ (both sweeps and the per-event path) and the pressure reducer refuse them,
  so nothing puts them back.
- atlas's 8 frozen readings stop counting in the biometrics surprise score
  (`biometrics_prediction_error`), and the ~58,000 junk atlas pressure emissions per day stop.
- The metacog trend cue no longer asks for atlas (`METACOG_TREND_CUE_NODES`); verb activation,
  smoke scripts and one persona string drop atlas.
- Tests that used atlas as a live machine now use circe (live GPU node) or hecate (expected
  offline). A new CI workflow runs the biometrics loop tests, which had no CI; four of them were
  red on main.

## Outcome moved

Live, read-only, 2026-10-16 ~03:05Z (`docker exec orion-athena-sql-db psql`):

- `substrate_node_biometrics_projection` held atlas with `last_seen_at`
  2026-08-20T22:43:26Z, cached `expected_online: true`, and 8 `pressure_hints` (gpu, strain,
  fan, disk, power, memory, thermal, disk_capacity). The catalog said `false` since 08-21; the
  cache never updated because atlas never reported again.
- `substrate_active_node_pressure_projection` held atlas with `active_pressures:
  ["availability"]`, status `suppressed`, and `strain` re-accepted at 03:02:16Z today.
- `substrate_organ_emissions`, last 24 h: 58,082 of 341,374 emissions (17%) are atlas pressure
  candidates, about 115 MB. 181,207 since 2026-10-13. The "self-terminating" suppression sweep
  never terminated for atlas (see Risks).
- Biometrics surprise score, last 24 h of stored `node:substrate.biometrics` receipts
  (n=118, retention-sampled). **Approximate**: the stored value is `min(1, mean|delta|/0.30)`
  over every key of every node; atlas adds 8 always-zero deltas to 44 keys (athena 17, circe 10,
  hecate 9, atlas 8 in today's projection), so without atlas the value is the stored value
  times 44/36 = 1.222 (assumes the key count held all day):

  | | mean | median | p95 | max | share >= 0.03 (top of rest range) |
  |---|---|---|---|---|---|
  | with atlas (stored) | 0.0336 | 0.0272 | 0.0815 | 0.109 | 42.4% |
  | without atlas (approx) | 0.0411 | 0.0332 | 0.0996 | 0.133 | 59.3% |

  So the score reads about 22% louder without atlas, and more readings leave the rest band.
  That interacts with the attention-quota work (spec
  `docs/superpowers/specs/2026-10-16-attention-continuous-signal-quota-design.md`, branch
  `docs/attention-continuous-signal-quota`), which measured biometrics with atlas in.

## Current architecture

`orion-substrate-runtime` reduces biometrics grammar events into two projections (one row
each in Postgres). A node entry is only rewritten when that node sends an event. The catalog
was consulted for `expected_online`, but a node that stops existing stays in both
projections forever, and every generic reader (`biometrics_prediction_error`, the absence and
suppression sweeps, Hub routes) keeps reading its last values.

## Architecture touched

- Contract: `config/biometrics/node_catalog.yaml` gains `retired_nodes:`.
  `NodeCatalog.retired`, `NodeCatalog.is_retired()`, `NodeProfile.retired`. Retired names
  still resolve (historical trace ids keep a name), always `expected_online=False`, never in
  `NodeCatalog.profiles`. A name in both sections fails at load.
- Runtime seam: `process_biometrics_grammar_events` prunes first and writes one receipt
  (`reducer_id=biometrics_retired_node_prune`, one `suppress` delta per removed entry with the
  full `before` state, `after=None`).
- No bus, channel or schema-registry change.

## Classification of `atlas` references

Inventory: `rg -n -i '\batlas\b'` excluding site-packages/orion_dev/graphify-out: 565 files.

**(a) atlas the host -- changed in this PR**

- `config/biometrics/node_catalog.yaml`: node entry -> `retired_nodes:`.
- `orion/cognition/verbs/active.yaml`: `nodes.atlas` removed (same as default rule).
- `services/orion-cortex-exec/app/metacog_trend_reader.py`, `.env_example`: trend cue nodes
  `athena,atlas,circe` -> `athena,circe`.
- `scripts/smoke_biometrics_closed_loop.sh`, `scripts/smoke_field_digester_biometrics.sh`:
  query circe instead of atlas.
- `services/orion-cortex-orch/app/conversation_front.py`: persona node list (also named a
  nonexistent "Apollo"). Not imported by any live entry point; fixed for truth.
- `orion/core/bus/bus_schemas.py`: field description example.
- `mesh-utilities/common/exclude_services.txt`: ai-town comment now says it moved on to circe.
- Tests that treated atlas as live: `tests/test_biometrics_grammar_extract.py`,
  `test_node_biometrics_reducer.py`, `test_biometrics_pressure_organ.py`,
  `test_node_pressure_reducer.py`, `test_biometrics_pipeline.py`,
  `test_capability_absence_signal.py`, `services/orion-biometrics/tests/test_node_catalog.py`,
  `services/orion-cortex-exec/tests/test_metacog_trend_reader.py`,
  `orion/substrate/tests/test_metacog_trend_signals.py`. Coverage kept: atlas -> circe for
  live-GPU cases, -> hecate for expected-offline cases.

**(a) atlas the host -- already retired earlier, left as the retirement mechanism**

- `services/orion-field-digester/app/tensor/channels.py` `RETIRED_PSEUDO_NODES` (`node:atlas`),
  `config/field/orion_field_topology.v1.yaml` (removed 08-21), `services/orion-hub/scripts/
  biometrics_node_client.py` (`LIVE_NODES` excludes atlas), field-digester `GPU_NODES`.

**(a, not changed) live names that still say atlas but run on circe -- flagged, separate patch**

- `services/orion-llamacpp-host/docker-compose.atlas-workers.yml`, `ATLAS_*` env keys
  (`ATLAS_CHAT_PROFILE_NAME`, `ATLAS_AGENT_BURST_*`, `ATLAS_METACOG_PROFILE_NAME`, ...),
  container names `orion-atlas-llamacpp-*`, `config/gpu_pool.yaml`, `config/llm_profiles.yaml`
  profile keys, gpu-lane-controller tests. These are circe's running LLM workers. Renaming
  them is a cross-host container and env rename with restarts on circe.
- `services/orion-bus/docker-compose.atlas-standby.yml` and its runbook: cold-standby bus whose
  only target was atlas; already marked BROKEN/UNSET in its header. Delete or re-target is a
  Juniper call.

**(b) the word means something else -- left**

Concept Atlas, Substrate/Grammar Atlas, Self Atlas, Curiosity Atlas (Hub tabs, routes, JS,
templates), `atlas_graph` / `HUB_CURIOSITY_GRAPH_ATLAS` (the `orion_substrate` FalkorDB graph),
`orion/curiosity/atlas.py`, world-pulse concept atlas, `orion/grammar/constants.py`
("Substrate Atlas"), topic-foundry Self Atlas.

**(c) historical -- left**

`docs/**`, PR reports, plans, specs, and code comments that narrate atlas incidents
(`pressure_organ.py`, `pressure_reducer.py`, `inner_state_registry.py`, biometrics
`pdu.py`/`settings.py`/`.env_example`, field-digester comments, test docstrings).

## Files changed

- `config/biometrics/node_catalog.yaml`: atlas -> `retired_nodes:`.
- `orion/biometrics/node_catalog.py`: retired profiles, `is_retired`, both-sections guard.
- `orion/substrate/biometrics_loop/pipeline.py`: `prune_retired_nodes` + call + receipt.
- `orion/substrate/biometrics_loop/node_reducer.py`: no-op for retired hosts.
- `orion/substrate/biometrics_loop/pressure_organ.py`: no candidates, not swept.
- `orion/substrate/biometrics_loop/pressure_reducer.py`: rejects retired candidates.
- `services/orion-cortex-exec/app/metacog_trend_reader.py`, `.env_example`: default nodes.
- `orion/cognition/verbs/active.yaml`, `orion/core/bus/bus_schemas.py`,
  `services/orion-cortex-orch/app/conversation_front.py`,
  `mesh-utilities/common/exclude_services.txt`, two smoke scripts: atlas references.
- `orion/substrate/cabinet_ambient_spike_consumer.py`: no-op for retired hosts.
- `services/orion-field-digester/app/ingest/state_deltas.py`: never perturb a retired lattice
  node (was pseudo-nodes only).
- `tests/test_retired_biometrics_node.py` (new), 12 test files moved off atlas.
- `.github/workflows/biometrics-loop-tests.yml` (new).

## Schema / bus / API changes

- Added: `retired_nodes:` catalog section; `NodeProfile.retired`; `NodeCatalog.is_retired`.
- Removed: none.
- Renamed: none.
- Behavior changed: retired hosts are removed from `substrate_node_biometrics_projection` and
  `substrate_active_node_pressure_projection`; Hub `/api/substrate/biometrics-node/atlas/latest`
  will have no atlas node after the first tick.
- Compatibility notes: `biometrics_prediction_error` formula unchanged; its node population
  shrinks by one, so its level rises ~22% (see Outcome). `check_definition_drift --gate` passes
  (no definition change); this is a population change Juniper asked for.

## Env/config changes

- Changed value: `METACOG_TREND_CUE_NODES` `athena,atlas,circe` -> `athena,circe`
  (`services/orion-cortex-exec/.env_example`).
- `python3 scripts/sync_local_env_from_example.py` run from the primary root: it compares the
  primary checkout's (still unmerged) `.env_example`, so it reported nothing for this key. The
  one line in `services/orion-cortex-exec/.env` was edited by hand to `athena,circe`.
- Skipped keys requiring operator action: none for this change.

## Tests run

```text
pytest tests/test_retired_biometrics_node.py ... (biometrics loop set, 8 files)   84 passed (minimal CI venv)
pytest biometrics loop set + orion/substrate/tests + verb activation              1233 passed, 3 failed
  (3 failures: orion/substrate/tests/test_felt_state_self_definition_lane.py, also fail on clean main)
services/orion-biometrics: pytest tests/test_node_catalog.py                       5 passed
services/orion-cortex-exec: pytest tests/test_metacog_trend_reader.py              8 passed
services/orion-field-digester: pytest tests (minus main-import test)               282 passed, 6 skipped
CI set in a clean venv (pytest pydantic PyYAML requests)                           86 passed
Mutation: disabling the prune call, or the pressure-reducer retired guard, each turns a test red.
Before this PR, on main: 4 tests in this set were red (circe asserted expected-offline).
scripts/check_metric_lineage.py --gate PASS; check_definition_drift.py --gate PASS;
check_inner_state_registry.py OK; check_env_template_parity.py PASS.
```

## Evals run

```text
No eval harness for the biometrics loop. The live-data replay above (stored PE receipts,
rescaled) stands in for it and is labelled approximate.
```

## Docker/build/smoke checks

```text
None run. No deploy (task rule). Live effect UNVERIFIED until orion-substrate-runtime restarts.
```

## Review findings fixed

Review ran in a subagent on `git diff origin/main...HEAD`. No must-fix findings.

- Finding: the field digester turns deltas that name `node:atlas` into perturbations, so atlas
  pressure deltas (and the new prune receipt's `suppress` delta, whose `before` lists
  `availability`) re-created `node:atlas` for one tick, because reconcile runs before
  perturbations and the filter only covered pseudo-nodes.
  - Fix: `delta_to_perturbations` now drops every id in `PRUNED_NODE_IDS` (retired lattice
    nodes + pseudo-nodes). Three digester test files moved off atlas.
  - Evidence: `test_retired_lattice_node_deltas_produce_no_perturbations`; reverting the
    filter to pseudo-nodes only turns it red. Digester suite 282 passed.
- Finding: the absence-sweep call (no-op node-bio saver) wrote a prune receipt claiming a
  node-bio removal it never saved, every tick during a total outage.
  - Fix: `prune_retired_nodes(prune_node_bio=...)`; the pipeline prunes node-bio only when it
    runs the node reducer (the call that saves it).
  - Evidence: `test_absence_sweep_path_does_not_claim_a_node_bio_prune_it_never_saves`.
- Finding: the cabinet ambient-spike consumer could re-create a retired host.
  - Fix: returns a no-op receipt for a retired host.
  - Evidence: `test_cabinet_spike_cannot_recreate_a_retired_host`.
- Finding (nit): docstring named a test that does not exist. Fixed.
- Finding (nit): CI push filter narrower than PR filter, missing the runtime worker and cabinet
  consumer. Both filters now match and include them.
- Finding (nit, not changed): offline-node tests read hecate's real catalog entry. Kept on
  purpose: with this CI workflow, flipping hecate in the catalog now turns these tests red on
  that PR, which is the gate the circe tests lacked.
- Finding (note): Hub outreach vocabulary (`outreach_vocabulary.py`) builds names from
  `NodeCatalog.profiles`, so `atlas` is no longer a known name there. Intended.

## Restart required

```bash
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-field-digester up -d --build
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-cortex-exec up -d --build
```

Restart orion-field-digester together with orion-substrate-runtime: the first runtime tick
writes the prune receipt, and the digester must already refuse `node:atlas` deltas.

orion-biometrics and orion-hub read the catalog too but only to resolve names; no restart needed
for correctness. orion-cortex-orch's change is in an unused code path.

## Risks / concerns

- Severity: medium. Concern: the suppression sweep is not self-terminating for a stale
  expected-offline node that still carries `availability`. `node_pressure_suppressed` maps to
  pressure kind `strain` (`pressure_reducer.py` `ROLE_TO_PRESSURE_KIND`), so `availability` is
  never cleared, and Rule C re-adds `strain` from frozen hints. That is why atlas produced 58k
  emissions/day. Retiring atlas removes the only live case; hecate (expected offline, empty
  `active_pressures`) does not hit it today. Mitigation: follow-up fix in the organ/reducer.
- Severity: low. Concern: hecate's 9 keys are frozen since 2026-10-15T06:31Z and dilute the
  biometrics mean the same way while hecate is off. Not retired (on-demand box). Mitigation:
  the attention-quota work should decide whether the mean should skip stale nodes generally.
- Severity: low. Concern: the biometrics score rises ~22% after deploy. Mitigation: expected and
  asked for; downstream attention thresholds were measured with atlas in.

## Data cleanup proposal (not done)

Historical atlas rows, read-only counts 2026-10-16:

- `orion_biometrics`, `orion_biometrics_induction`, `orion_biometrics_summary`: 48,669 rows
  each, 2026-07-23 -> 2026-08-20. Real history of a real machine; recommend keep.
- `substrate_organ_emissions`: 181,207 atlas pressure emissions since 2026-10-13 (~115 MB/day).
  Junk. They age out under existing retention; a purge would only speed that up. Recommend no
  manual delete.
- Projection rows: the prune removes atlas on the first tick after restart; no manual write.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2714

🤖 Generated with [Claude Code](https://claude.com/claude-code)
