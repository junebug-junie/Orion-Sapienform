# fix(field-digester): an unmeasured capability no longer reads as perfect

## Summary

- When every input to a capability channel had expired, the field digester wrote pressure 0.0, confidence 1.0 and spare capacity 1.0, and recorded no source. A capability nobody was measuring looked perfectly healthy. Now those keys are left out of the vector instead.
- This uses the existing convention: a dropped key means "unmeasured" for node channels (`decay.py` `expire_unrefreshed_channels`), and an absent provenance entry already meant "nobody measured it" (`measured_zero_source`). This patch applies it one layer up, in `apply_diffusion`, which is the last writer before the tick is saved.
- The replay eval (`scripts/eval_capability_unmeasured_replay.py`) runs on 2,051 real ticks. This option changed **no** downstream reading in any simulated outage. The "confidence = 0" alternative would have made a dark camera the attention selector's most urgent capability on every outage tick.
- The metric definitions changed, so the glossary meaning text for `pressure`, `confidence`, `available_capacity` and `reliability_pressure` was updated and the metric lock re-locked.
- The hub's field debug route returns `null` for a missing capability pressure instead of inventing 0.0. The hub's transport lattice pressure gate reads a missing channel as "unknown", not "quiet".
- Also fixed, found during the replay: a derived confidence/capacity kept a stale direct-edge source label. Live, `capability:transport` confidence named `node:athena` on 123,095 of 123,095 ticks, left over from the direct edge retired on 09-25.
- Five follow-ups that would change cognition-loop decisions are **not** implemented. They are listed below as decisions for Juniper.

## Outcome moved

Failure mode closed: "outage reads as recovery" in the stored field. Example: the eye reports "no camera" (`capability:vision` pressure 0.85), then the frame router dies. Before, the next tick 300 s later read pressure 0.0, confidence 1.0, capacity 1.0. Now that tick has no pressure, confidence, capacity or reliability keys and no provenance. The test pinning this is `test_vision_outage_after_alarm_does_not_read_as_recovery`, and it fails on main.

## Current architecture

- The digester tick runs in this order: `reconcile_field_state_with_lattice` (re-seeds every capability channel to 0.0, with confidence and capacity at 1.0), then `apply_decay`, `expire_unrefreshed_channels` (drops node channels in `EXPIRING_NODE_CHANNELS`), `apply_diffusion`, and finally save.
- In `apply_diffusion`, every diffusion-target channel that nobody contributed to was set to `0.0`. If pressure was a target, confidence was derived as `1 - 0.5*pressure` and capacity as `1 - pressure`. That gives 1.0 and 1.0 for an unmeasured pressure.
- Capability channels affected through the expiring inputs:
  - `capability:vision` pressure and reliability (`vision_frame_staleness`, `vision_processing_failure_pressure`, 300 s)
  - `capability:storage` reliability (`write_failure_pressure`, 180 s)
  - `capability:transport` reliability (`rpc_timeout_pressure`, 120 s)

## Semantic layer (mandatory pre-work)

`make check-metric-lineage METRIC=…` was run for pressure, confidence, available_capacity, reliability_pressure and all four expiring channels, plus `inference_failure_pressure`. `make check-metric-generic-consumers` was also run: 15 confirmed and 4 likely whole-vector readers. None of these metrics has a liveness verdict registered.

### Consumer table

Each entry lists what the consumer does with: 0.0, 1.0, a missing key, confidence 0.0, and None/NaN.

- **`selectors._current_pressure_proxy` and novelty** (attention winner) — DECISION
  - 0.0: adds nothing.
  - 1.0 confidence: inverted to 0, reads calm.
  - Missing key: not counted.
  - Confidence 0.0: **becomes proxy 1.0, the top target, and causes a novelty spike.**
  - None/NaN: skipped.
- **`pressure.collect_field_channel_pressures`** (proposal dimensions, feedback, corpus, anomaly scorer) — DECISION
  - 0.0: max-merged, loses unless it ties.
  - 1.0 confidence: min-merged, loses.
  - Missing key: not counted.
  - Confidence 0.0: **becomes the merged confidence and capacity (0.0).**
  - None/NaN: None raises TypeError. **NaN becomes 1.0.**
- **`pressure.map_channels_to_dimensions` and `field_pressures`** — DECISION
  - 0.0: max into the dimension.
  - Missing key: the dimension disappears only if no capability has that channel. Never happens live.
  - Confidence: not routed.
  - NaN: becomes 1.0.
- **`credit_integrity.channel_write_backed`** (feedback credit gate)
  - Works on provenance only. A winner with no provenance resolves to the capability id, which reads as "not backed".
  - Missing key: a measured sibling wins instead, which is more honest.
- **`feedback.extractors.pressure_delta`** — DECISION
  - Missing dimension: filled with 0.0. Only matters if a whole dimension vanishes, which never happens live.
- **`feedback.builder.reliability_delta`**
  - Missing: treated as stale, so credit is withheld.
- **`proposals.scoring`**
  - Missing dimension: urgency 0 and confidence 0. Not reachable here, because the dimensions always have measured siblings.
- **`outcome_resolution`**
  - Missing dimension: no observation.
- **Corpus and anomaly scorer** (`worker.py`, which feeds an equilibrium metacog trigger)
  - Missing key: filled with 0.0.
  - Confidence 0.0: **the merged confidence pins at 0, a distribution shift.**
- **`fit_encoder`** (offline)
  - Missing key: filled with 0.0.
- **Hub glossary routes** (display)
  - Missing everywhere in the window: classed "never_produced". Not the case here.
- **Hub `substrate_field_routes:72`** (display)
  - Before: a missing pressure became 0.0. **Now null.**
- **Hub `substrate_lattice_routes` M4** (display)
  - Missing becomes 0.0, shown as "quiet". Unchanged, because transport pressure and reliability never go unmeasured live.
- **`substrate-lattice.js`** (display)
  - Missing key: not shown.
- **Not consumers** (node vectors only, or schema only):
  - `orion-mind`, curiosity, tension, the goal producer, equilibrium (except through the anomaly scorer)
  - `contrast.py`, `self_study.py`

Full audit with file:line: see the consumer audit in this PR's conversation, summarised above.

### Options evaluated

Replay used every 60th tick of `substrate_field_state` from 2026-10-04 01:10 to 10-07 01:10 UTC: 2,051 of 123,099 ticks. Both implementations run on the same pre-diffusion state. "legacy" is a frozen copy of main's `apply_diffusion`, not a reconstruction. Legacy reproduces the stored production vectors on 1,949 of 2,051 ticks. The other 102 are all `capability:orchestration`, whose capability-to-capability edge reads the previous tick's llm_inference value (ratio about 1.000). Every channel that both implementations wrote agrees on 2,051 of 2,051 ticks in every scenario (`a_measured_channel_mismatch` = 0).

**Natural data (no simulated outage):** 0 ticks had an unmeasured capability channel. Every capability channel that has an edge was attributed on 123,112 of 123,112 ticks. At the node level, `rpc_timeout_pressure` was absent on 34 ticks; its largest age was 118.97 s against a 120 s TTL. During those ticks transport reliability stayed "measured" through athena's `observer_failure_pressure`. So **in the last 72 h no option would have changed anything**. The numbers below come from simulated outages on those real ticks.

| Option | What a dark capability reads as | vision outage | storage outage | all outages |
|---|---|---|---|---|
| legacy / (c) marker | 0.0 pressure, 1.0 confidence and capacity (under-index) | baseline | baseline | baseline |
| **(a) drop key — chosen** | absent | **0 diffs** on dimensions, dimension winners, credit, merged values, attention top. 1 tick: merged-confidence source label (the stale-label fix) | same | same |
| (b0) confidence = capacity = 0 | alarm via inverted channels | vision is attention top on **2051/2051** ticks; merged confidence and capacity are 0.0 on 2051/2051 | 0 diffs (storage pressure still measured) | attention top changes 2051/2051; merged confidence 0.0 on 2051/2051 |
| (b1) b0 plus pressure 1.0 | alarm | dimensions, winners and credit change on 2051/2051 | storage is attention top on 2051/2051; reliability credit flips on 2051/2051 | `resource_pressure` = **1.0 on 2051/2051** |

- **(c), an explicit `measured:false` marker:** numerically identical to legacy for every consumer. Only `credit_integrity` reads provenance, and it already treats an absent provenance as "not backed". Values stay 0.0 and 1.0, so every other reader still sees "healthy". This fails the goal.
- **(d), an existing mechanism:** yes, and (a) *is* that mechanism, moved one layer up from node channels.
- **Over-index risk:** b0 and b1. A dark sensor becomes an alarm, which feeds attention winners, proposal targets, anomaly and metacog triggers, and spurious curiosity.
- **Under-index risk:** legacy and (c). A dark sensor reads as perfect.
- **(a)** removes the false "perfect" claim from the stored vector without creating an alarm. For max()- and min()-style consumers it matches legacy wherever another capability is measured, which is the case on every live tick.

### Prior incidents checked against this change

- **bus_synaptic ~0.27 floor; saturated constant read as signal:** no new value is introduced.
- **prediction_error decaying to 0 under the 0.92 loop:** this is the dual case. This patch stops writing a fabricated 0. Separately, decay.py's "dead weight" comment is wrong: the capability decay loop *does* reach orchestration through the capability-to-capability edge, as the replay showed. The comment is corrected but the behaviour is unchanged.
- **Held calm values through outages:** see decision 3 below.
- **Raw counts:** not applicable.

### Metric gate

1. **Provenance:** `diffusion.py` `apply_diffusion`, in the `possible_targets` loop.
2. **Independence:** no new metric. This redefines when the existing ones are present.
3. **Theory anchor:** the absent-key-means-unmeasured contract that `decay.py` and `credit_integrity` already use.
4. **Live data:** 72 h replay above.
5. **Existing mechanism:** reused.
6. **Reversibility:** a single-function revert. No schema, manifest or stored training default changes. Old rows keep their 0.0 and 1.0 values.

## Architecture touched

orion-field-digester diffusion (cognition substrate, write side only), the field glossary, the metric lock, and the hub debug route. No bus, schema or env changes.

## Files changed

- `services/orion-field-digester/app/digestion/diffusion.py`: drop unmeasured target channels and derived channels.
- `services/orion-field-digester/app/digestion/decay.py`: comment noting the capability layer, plus a correction to the "dead weight" note.
- `services/orion-field-digester/tests/test_capability_unmeasured_is_absent.py`: new. Covers the real outage shapes (vision after alarm, storage writer, RPC partial coverage, generic consumers) through reconcile, tick and a JSON round trip.
- `services/orion-field-digester/tests/test_eval_capability_unmeasured_replay.py`: new. Gates the eval's headline claim: (a) changes nothing downstream, and b0 hands attention to the dark eye.
- `services/orion-field-digester/tests/test_diffusion_measured_zero_provenance.py`, `test_diffusion_provenance.py`, `test_field_rpc_delivery_perturbations.py`: these pinned "unmeasured = 0.0" and now assert absence.
- `scripts/eval_capability_unmeasured_replay.py`: new read-only replay eval.
- `config/field/field_channel_glossary.v1.yaml`: meaning text for the 4 capability channels.
- `config/metrics/metric_definitions.lock.json`: re-locked (4 high-severity `semantics_changed`).
- `config/field/orion_field_topology.v1.yaml`: the capability:vision comment.
- `services/orion-field-digester/README.md`: new section.
- `services/orion-hub/scripts/substrate_field_routes.py` and its test: null, not 0.0.
- `services/orion-hub/scripts/substrate_lattice_routes.py` and its test: a missing transport pressure or reliability reads as "unknown".
- `services/orion-field-digester/app/projections/node_field_projection.py`: same null change (module has no live caller).

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed:
  - `FieldStateV1.capability_vectors[cap]` may now lack `pressure`, `reliability_pressure`, `confidence` or `available_capacity` on a tick when nothing measured them. The schema already allowed this (`dict[str, float]`).
  - Hub `/api/substrate/field/node/{id}` `connected_capabilities[].pressure` can be `null`.
  - Hub transport gates: the `pressure` gate can be `unknown` when a channel is unmeasured.
  - A capability-to-capability edge treats its source as measured only when that source channel has provenance.
  - A derived confidence/capacity no longer carries a stale direct-edge label.
- Compatibility notes: every consumer was audited, as above. No consumer indexes these keys with `[]`.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: no.
- Local `.env` synced: not needed.
- Skipped keys: none.
- No feature flag. This is a correctness fix with zero measured downstream change. Roll back by reverting.

## Tests run

```text
services/orion-field-digester: pytest tests -q --ignore=tests/test_heartbeat_chassis.py -> 281 passed, 6 skipped
  (test_heartbeat_chassis.py: collection FileNotFoundError, identical on clean main)
test_capability_unmeasured_is_absent.py vs main's diffusion.py: 6 of 7 fail on main
  (the 7th pins unchanged partial-coverage behaviour on purpose)
services/orion-hub: test_substrate_lattice_routes.py + test_substrate_field_debug_api.py -> 70 passed
  (test_hub_ui_polish::test_hub_main_layout_is_fifty_fifty_and_scrollable_chat fails identically on main)
root: 14 field/capability test files -> 162 passed, 6 failed; the same 6 fail on clean main
  (test_field_digestion_rules::test_reasoning_load_diffuses_to_orchestration,
   test_field_execution_perturbations x2, test_field_transport_perturbations x1,
   test_field_deterministic_replay x2)
tests/test_field_topology_edges.py tests/test_field_channel_glossary.py -> 24 passed
scripts/check_metric_lineage.py --gate -> PASS
scripts/check_definition_drift.py --gate -> PASS (after --update)
scripts/check_inner_state_registry.py, check_scripts_dir_no_stdlib_shadow.py -> OK
```

## Evals run

```text
python scripts/eval_capability_unmeasured_replay.py --dump <2,051-tick 72h sample>
  legacy (frozen main apply_diffusion) vs recorded: 1949/2051 match (residual = orchestration cap->cap prior-tick read)
  a vs legacy, every channel both wrote: 0 mismatching ticks in all 5 scenarios
  natural: 0 ticks unmeasured
  outage:vision / storage / rpc / all: option a -> 0 differences in dimension values/winners, credit,
    merged values, attention top; 1 tick merged-confidence label (stale-provenance fix)
  outage:rpc: 0 ticks unmeasured (athena's observer channel still covers transport reliability)
  b0: vision attention-top 2051/2051, merged confidence 0.0 2051/2051
  b1: resource_pressure 1.0 on 2051/2051 (all-outage)
```

## Docker/build/smoke checks

```text
Not run: the deploy belongs to Juniper from the primary checkout after merge. UNVERIFIED live until the proof queries below are run.
```

## Review findings fixed

The /code-review subagent reviewed `git diff origin/main...HEAD` and returned 12 findings. All 12 are fixed:

- **Finding (major):** the eval's "legacy" was rebuilt from the branch's own output, so "zero downstream change" held by construction.
  - Fix: legacy is now a frozen copy of main's `apply_diffusion`, run on the same pre-diffusion state. Added the `a_measured_channel_mismatch` and merged-winner counters.
  - Evidence: 0 mismatches. Only one difference was surfaced: the stale label.
- **Finding (major):** the generic-consumer test passed on main.
  - Fix: it now uses an all-calm tick where the fabricated 1.0 ties, and asserts the confidence key is absent.
  - Evidence: fails on main.
- **Finding:** a capability-to-capability source that reconcile had re-seeded counted as a measured zero.
  - Fix: require provenance when the source channel is diffused itself.
  - Evidence: `test_unmeasured_upstream_capability_is_not_a_measured_zero_downstream`.
- **Finding:** a direct measured-zero confidence was dropped, and the derived value kept a stale direct label.
  - Fix: keep the measured zero; clear the label on overwrite.
  - Evidence: `test_direct_measured_zero_confidence_survives_unmeasured_pressure`, plus the 123,095-tick live finding.
- **Finding:** the docstring/comment still said "zeroed", and the glossary/README over-claimed.
  - Fix: reworded, and scoped to channels an edge feeds. Non-target seeded zeros are named as a follow-up.
- **Finding:** the hub lattice gate read a missing key as "quiet".
  - Fix: it now reads "unknown". Evidence: new hub test.
- **Findings (minor/nit):** dead b1 code removed; node projection mirrored; significance store stubbed; eval test asserts measured-channel equality; README says "simulated outages only".

## Decisions for Juniper (cognition-loop changes, not implemented)

1. **Feedback can still credit an outage as a recovery.** If a capability was winning `resource_pressure` or `reliability_pressure` and then goes dark, the dimension falls to the next measured capability. `credit_integrity` only checks the *after* winner, so a "pressure decreased" outcome can be credited. This is true of legacy and of (a) alike. A possible guard: withhold credit when the *before* winner's source is unmeasured after. This changes feedback decisions.
2. **Attention novelty treats going dark as a change.** For example, the vision proxy goes from 0.85 to 0 when the router dies. Should novelty skip transitions into "unmeasured"? This changes attention decisions.
3. **`inference_failure_pressure` is held, not expired.** Over 72 h, circe's value was held for up to 16,745 s (4.6 h). It was older than 120 s on 43,620 of 123,099 ticks, older than 600 s on 1,956, and older than 1 h on 67. Putting it in `EXPIRING_NODE_CHANNELS` would make llm_inference reliability unmeasured whenever the gateway is quiet. The TTL needs her call.
4. **`observer_failure_pressure` is a degenerate constant.** It read 0.0 on 123,099 of 123,099 ticks, with 1 distinct value. It is the reason transport reliability stays "measured" when the RPC bridge expires. Retire it under the metric gate?
5. **Channels that no edge ever targets stay at 0.0.** Examples are vision's contract, execution and reasoning pressure, and graph and memory reliability. Reconcile keeps re-seeding them at 0.0 forever, so they are structurally unmeasured but read as calm. Dropping them changes the reconcile contract, so it is left as a follow-up.

## Restart required

No SQL migration.

```bash
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-field-digester up -d --build
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
```

## Proof queries (after deploy)

```sql
-- invariant: a target channel is present iff it has provenance (expect 0 violations)
select count(*) from substrate_field_state s,
  jsonb_each(s.field_json->'capability_vectors') cv(cap, vec),
  (values ('pressure'),('reliability_pressure')) ch(c)
where s.generated_at > '<deploy_ts>'
  and (vec ? ch.c) and not ((s.field_json->'capability_provenance'->cap) ? ch.c)
  and cap in ('capability:vision','capability:storage','capability:transport','capability:llm_inference');
-- vision unmeasured ticks now carry no confidence (expect 0)
select count(*) from substrate_field_state where generated_at > '<deploy_ts>'
  and not (field_json->'capability_provenance'->'capability:vision' ? 'pressure')
  and (field_json->'capability_vectors'->'capability:vision' ? 'confidence');
```

## Risks / concerns

- **Severity: low.** A future consumer could index a capability channel with `[]` and raise KeyError. Mitigation: today's consumers were audited, and the README and glossary state the contract.
- **Severity: low.** Pre-deploy rows keep the old fabricated 0.0 and 1.0 values. A window that straddles the deploy mixes the two encodings.
- **Severity: info.** In the last 72 h this would have changed nothing. It is protection for the next outage, not a fix for an observed one.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2534

🤖 Generated with [Claude Code](https://claude.com/claude-code)
