# fix(field): D3 rename, outage-proof credit, outage-proof novelty, retire observer_failure_pressure

## Summary

This PR implements four of the follow-ups Juniper approved on 2026-10-07 ("yeah sure on those decision"). Three are #2534 decisions; the fourth is decision D3 from the transport-lattice spec. #2534 decisions 3 and 5 were deferred by her and are not touched.

- **D3, honest name.** The transport capability's "contract pressure" was always the bus catalog-drift reading (0.85 x node:athena's catalog drift), just under the wrong name. It is renamed `catalog_drift_pressure`, with the same values. Stored rows that still carry the old name have it pruned on the next tick, so the same drift is never stored under two names.
- **Decision 1: an outage no longer counts as a recovery.** Sometimes the capability that set a pressure dimension goes dark between the "before" and "after" ticks, and the dimension drops to the next source. That drop is no longer credited to the action. It is now withheld, both in the feedback frame and in the outcome scoring (both arms).
- **Decision 2: a channel going dark is no longer "news".** Attention novelty used to count a channel disappearing, or coming back, as a change. Now, when the set of measured channels changes, novelty compares only the channels measured on both ticks. The frame records when this happened.
- **Decision 4: `observer_failure_pressure` is gone, end to end.** It read 0.0 on 123,099 of 123,099 ticks. Its only effect was to keep transport reliability looking "measured, calm" whenever the RPC bridge went quiet. Now an RPC-bridge outage reads as **unmeasured**, which is the outcome Juniper accepted.
- Added a replay eval that runs main's real code and this branch on the same 72 h of real ticks and diffs every downstream reader.

## Outcome moved

- **Natural data (2,037 consecutive tick pairs, 72 h):** no change in anything a cognition consumer reads. That covers:
  - dimension values and winners (channel and source)
  - the merged channel values
  - attention novelty, pressure, winners and dominant targets
  - feedback write-evidence
- **Only difference on natural data:** on 1 tick in 2,037, transport reliability is unmeasured. That is decision 4 working, on a tick where the RPC bridge's reading had expired.
- **Simulated outages:** the dark eye or writer or bridge stops creating novelty, and credit is withheld instead of scored.
- **Real feedback frames (2,995 sampled):** the new credit guard withheld nothing. That is expected, because nothing went fully dark in this window.

## Current architecture

- **Transport capability, before this PR:** fed by three edges.
  - bus_synaptic -> `pressure`
  - node:athena -> `contract_pressure` (actually catalog drift) and `reliability_pressure` (from `observer_failure_pressure`)
  - rpc_delivery -> `reliability_pressure`
- **Feedback credit:** the write-evidence gate (`channel_write_backed`) checked only the AFTER tick's winner.
- **Attention novelty:** the current max-channel proxy minus the previous frame's proxy. It could not tell "went dark" apart from "calmed down".
- **The #2534 consumer table** (in its PR report) lists every reader of these channels.

## Semantic layer / metric gate

- **Lineage:** checked with `make check-metric-lineage` for `contract_pressure`, `catalog_drift_pressure` and `observer_failure_pressure`.
  - `observer_failure_pressure` had 9 named readers. All are removed: bus observer, extract, reducer, decay, state_deltas, channels, the hub gate, and the substrate-runtime incident set.
  - `contract_pressure` at capability level had 7 named readers; all are renamed or commented.
  - There are 23 whole-vector generic consumers. These are covered by the replay rather than by grep.
- **Double counting:** none. Node and capability now share the name `catalog_drift_pressure`.
  - `collect_field_channel_pressures` takes the max of the two, and 0.85 x the node value is always below the node value.
  - The replay shows the merged value and its winner unchanged on every tick.
  - The channel is in neither `PRESSURE_CHANNELS` nor `CHANNEL_DIMENSION_MAP`, so no dimension changes.
  - Mood-arc encoder v4 uses `catalog_drift_pressure`, which is unchanged. It never used `contract_pressure` or `observer_failure_pressure`.
- **One under-index risk found in review, and fixed:** `commensurability.observe_source_to_channel` let the capability's 0.85x value overwrite node:athena's own value under the same key. It now keeps the value the merge keeps.
- **Theory anchor:** absent key means unmeasured, the convention already used by `decay.py` and #2534's `apply_diffusion`. Decision 1 is the R5b "do not learn from an outage" rule applied to the BEFORE winner. Decision 2 makes novelty a change in measured state, not a change in what is being measured.
- **Reversibility:** no schema migration and no SQL.
  - Decisions 1 and 2 are code-only and revert with a single commit.
  - Decision 1 rides the existing `write_evidence_guard_enabled` policy flag, which is ON. No new flags.
  - D3 and decision 4 change the metric lock; it was re-locked.

## Files changed

- **Config:**
  - `config/field/orion_field_topology.v1.yaml`: rename, the observer edge mapping removed, comments.
  - `config/field/field_channel_glossary.v1.yaml`: `catalog_drift_pressure` is now level [node, capability] with a truthful meaning. Entries for `contract_pressure` and `observer_failure_pressure` removed. `reliability_pressure` meaning updated.
  - `config/substrate-lattice/transport_lattice_policy.v1.yaml`: the `observer_failure_pressure` (m3) row is replaced by `transport_reliability_pressure` (m4 `capability:transport.reliability_pressure`, same thresholds). The hub gate already used that threshold for this reading, under the wrong name.
  - `config/metrics/metric_definitions.lock.json`: re-locked. 2 metrics removed, 2 had their meaning text changed, 1 annotation changed.
  - `config/proposals/proposal_policy.v1.yaml`: comment only.
- **Field digester** (`services/orion-field-digester/app/`):
  - `tensor/channels.py`: renamed in `CAPABILITY_CHANNELS`. `RETIRED_CAPABILITY_CHANNELS` gains `contract_pressure`. `RETIRED_NODE_CHANNELS` gains `observer_failure_pressure`.
  - `digestion/decay.py` and `ingest/state_deltas.py`: updated to match. The README has a new section.
- **Transport reducer:**
  - `orion/substrate/transport_loop/extract.py` and `reducer.py`: observer failure no longer counted. A pre-deploy `tick_failed` trace becomes a no-op.
  - `orion/schemas/transport_projection.py`: the retired fields are dropped on read.
- **orion-bus:** `app/bus_observer.py`, `app/grammar_emit.py`, `README.md` and `SUBSTRATE_TRACE_MAP.md`. A failed tick now publishes nothing; the log line and heartbeat cover it.
- **Credit guard (decision 1):**
  - `orion/field/pressure.py`: `collect_field_channel_pressures_with_holders` (same merge, also records which vector won).
  - `orion/field/credit_integrity.py`: `before_winner_went_unmeasured`, `dimension_value_holders` (tie-aware) and `_vector_measures` (stale node stamps count as unmeasured).
  - `orion/feedback/builder.py` and `orion/feedback/outcome_resolution.py`: the guard.
- **Novelty (decision 2):**
  - `orion/attention/field_attention/selectors.py`: compares only the channels measured on both ticks, and records a reason.
  - `orion/attention/field_attention/builder.py`: `previous_field`, ignored if its tick does not match the previous frame.
  - `services/orion-attention-runtime/app/worker.py` and `store.py`: an in-memory previous field, plus `load_field_for_tick` (a primary-key read, only after a restart).
- **Hub:**
  - `services/orion-hub/scripts/substrate_lattice_routes.py`: the pressure gate judges the two halves independently, so an RPC lull no longer hides an active bus reading.
  - `services/orion-hub/static/js/substrate-lattice.js`: the observer line on the M3 card removed.
- **Other code:**
  - `services/orion-substrate-runtime/app/worker.py`: incident fields.
  - `orion/substrate/relational/adapters/transport_ctx.py`: comment.
  - `orion/field/commensurability.py`: the collision fix.
  - `scripts/analysis/*`: notes on historical names.
- **Eval:** `scripts/eval_field_decisions_replay.py`, with modes `run`, `compare` (a gate) and `feedback`.
- **Tests:**
  - New: `tests/test_credit_before_winner_unmeasured.py`, `tests/test_attention_novelty_ignores_measurement_changes.py`, `tests/test_eval_field_decisions_replay.py`, and `services/orion-attention-runtime/tests/test_worker_previous_field.py`.
  - Updated: reducer, schema, glossary, topology, digester, hub, bus and commensurability tests.

## Schema / bus / API changes

- **Added:**
  - Field channel `capability:*.catalog_drift_pressure`.
  - Lattice row `transport_reliability_pressure`.
  - `AttentionRuntimeStore.load_field_for_tick`.
  - Optional `previous_field` argument on `build_attention_frame` and the selectors.
  - Feedback withheld reason `before_winner_unmeasured`, and outcome skip reason `before_winner_unmeasured:<signal>`.
- **Removed:**
  - Capability channel `contract_pressure`.
  - Node channel `observer_failure_pressure`.
  - `TransportBusStateV1.observer_failure_count` and `observer_failure_pressure`, dropped on read via `RETIRED_TRANSPORT_BUS_STATE_FIELDS`.
  - Grammar role `bus_observer_tick_failed` is no longer emitted. It remains a terminal role and is ignored by extract.
  - Lattice row `observer_failure_pressure`.
- **Renamed:** capability `contract_pressure` -> `catalog_drift_pressure`.
- **Behavior changed:**
  - Transport reliability is absent during an RPC-bridge outage.
  - The feedback and outcome guard (decision 1).
  - Novelty (decision 2).
  - The hub pressure gate: "watch" if either measured half is at or above its threshold; "unknown" if nothing is active and a half is unmeasured; "quiet" only when both halves are measured and calm.
- **Compatibility:**
  - Persisted rows are cleaned by reconcile and the projection validator.
  - `FieldAttentionTargetV1` is unchanged; only a new reasons string is added.
  - No `channels.yaml` or registry changes: no new bus channel or schema.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: no. Local `.env` sync: not needed. Skipped keys: none.
- New flags: none. Decision 1 rides `write_evidence_guard_enabled: true`.

## Tests run

```text
root (focused, 25 files incl. credit/feedback/outcome/attention/transport/glossary/topology/proposal/commensurability/eval): 533 passed
tests/test_field_transport_perturbations.py (PYTHONPATH=services/orion-field-digester:.): 3 passed (was failing on main: missing run_digestion_tick args; fixed so its assertion runs)
services/orion-field-digester: 281 passed, 6 skipped (test_heartbeat_chassis.py excluded: same relative-path FileNotFoundError on main)
services/orion-hub lattice + glossary + field debug: 90 passed (glossary endpoint count test was failing on main at 52 vs 39; now pinned to 50)
services/orion-bus (PYTHONPATH=.:../..): 38 passed
services/orion-attention-runtime: 43 passed (heartbeat chassis excluded, same as main)
services/orion-feedback-runtime: 58 passed (heartbeat chassis excluded, same as main)
services/orion-substrate-runtime transport incident logging: 5 passed
scripts/analysis capability health + transport history: 31 passed
scripts/check_definition_drift.py --gate: PASS (after merge commit, then --update)
scripts/check_metric_lineage.py --gate: PASS
scripts/check_inner_state_registry.py, check_scripts_dir_no_stdlib_shadow.py, check_env_template_parity.py: PASS
```

New tests that fail on main:
- The before-winner guard in the gate, the feedback frame, outcome scoring and the control arm. These need the new code; main credits the outage.
- Novelty "went dark" and "came back". Main's novelty reads 0.75 for the dark eye.
- Hub "active bus half watches while reliability is unmeasured". Main reads "unknown".
- The pre-deploy `tick_failed` trace becomes a no-op. Main emits reliability 1.0 and drift 0.0.
- A failed observer tick publishes nothing. Main publishes a failed trace.
- RPC outage leaves transport reliability unmeasured. Main reads 0.0 from node:athena.
- The D3 transition tick and both-level prune.
- The commensurability collision.

## Evals run

```text
Dump: substrate_field_state 2026-10-04 03:00 -> 10-07 03:00 UTC, every 60th tick + its predecessor (2,037 consecutive pairs, 4,074 rows)
before = primary checkout at origin/main 088cf9f3e (real main code); after = this branch
python scripts/eval_field_decisions_replay.py run --tree <main> ... ; run --tree . ... ; compare main.jsonl branch3.jsonl
rows: both=10185 only_before=0 only_after=0
natural:        0 diffs in dims / dim winner channel+source / merged values / attention top+dominant / novelty / proxy / credit backed; guard fired 0
                1 tick: capability:transport reliability unmeasured (RPC reading expired; decision 4 as intended)
outage:vision:  0 consumer diffs (the eye read staleness 0.0 when it went dark, so the proxy did not move)
outage:storage: novelty changed on 102 ticks (node:substrate.storage_write) / 34 (capability:storage); attention top capability changed 33, dominant 38;
                credit guard fired on reliability_pressure 100 / 2037 (storage reliability was the before winner)
outage:rpc:     transport reliability unmeasured 2037/2037 (was 0.0 via observer on main); novelty changed 48 (rpc_delivery) / 43 (capability:transport);
                attention top capability changed 41; guard fired 7
outage:all:     union of the above; dimension values and winners unchanged in every scenario
GATE PASS: natural data -- no consumer-visible change beyond the rename/retirement

Feedback replay (decision 1): 120,863 real feedback frames 2026-10-04..07, every 40th = 3,022; 2,995 with a 30 s after-tick
  (before = dispatch source tick, after = first tick within 30 s, the runtime's own rule)
  stored ticks and re-digested-with-branch ticks: guard withheld 0 / 8,985 dimension decisions; R5b withheld 1 (pre-existing)
```

The guard firing rate on real frames is 0, so it is rare, as asked. It only fires when a capability actually goes dark, and nothing went dark in this window.

## Over- and under-indexing, by consumer

- **Proposal arena** (`field_pressures` dims): unchanged on natural data. Dimension values and winners are identical in every scenario, including outages.
- **Feedback credit** (builder plus outcome resolution):
  - Natural data: unchanged.
  - Outages: credit for the dark winner's dimension is withheld, in both directions, instead of being credited. This is deliberately under-crediting the outage window.
  - Ties are handled. If any vector that held the winning value still measures it, credit is kept. Before the review fix, a 0 -> 0 tie could be withheld spuriously.
- **Attention novelty:** unchanged on natural data. In outages, a channel going dark no longer becomes the top novelty. There is no over-index from "dark = alarm", and no under-index for a real move: channels measured on both ticks still count.
- **Corpus, anomaly scorer and fit_encoder:**
  - The merged dict loses the `contract_pressure` key, which was a redundant 0.85x copy of drift, and the always-zero `observer_failure_pressure`.
  - The v4 encoder's channel list contains neither, so the live anomaly score is unchanged.
  - Future encoder refits lose two dead or duplicate columns.
- **Hub:**
  - The lattice values panel gets a `transport_reliability_pressure` row. It would have read "watch" on 1,837 of 123,086 ticks (1.5%) over 72 h, which is the same threshold the hub pressure gate already applied to this same reading.
  - `_compute_salience` is display-only: the simulate and latest endpoints.
- **Relational adapter:** confidence = 1 - reliability. It is unchanged, because the observer input never fired.

## Review findings fixed

The `/code-review` subagent reviewed `git diff origin/main...HEAD`. It found no major issues, 6 minor and 4 nits. All are fixed.

- **Finding:** the guard could fire on a tie at zero when only the merge's last-picked holder expired.
  - **Fix:** `dimension_value_holders`. The guard fires only if every holder of the winning value went dark.
  - **Evidence:** `test_tie_at_the_winning_value_is_not_an_outage_when_another_holder_still_measures`.
- **Finding:** decaying node channels never tripped the guard.
  - **Fix:** with the feedback policy's `stale_after_sec`, a stale node stamp now reads as unmeasured.
  - **Evidence:** `test_decaying_node_winner_with_a_stale_stamp_counts_as_unmeasured`.
- **Finding:** an RPC lull would blank the whole hub pressure gate.
  - **Fix:** the two halves are judged independently.
  - **Evidence:** `test_gates_pressure_active_bus_half_still_watches_when_reliability_unmeasured`.
- **Finding:** the commensurability checker reported athena's drift as the 0.85x value.
  - **Fix:** it keeps the value the merge keeps.
  - **Evidence:** a new test in `test_merge_commensurability.py`.
- **Finding:** several new paths had no tests: worker previous-field, the store read, the reliability reason in both branches, and the control arm.
  - **Fix:** `test_worker_previous_field.py` plus 3 builder and outcome tests.
- **Finding:** nothing recorded when novelty was adjusted.
  - **Fix:** a reason `novelty_common_channels_only went_dark=[...] came_back=[...]` on the target.
- **Nits:**
  - Eval compare: now reports unpaired rows, ignores the intended key changes, and exits non-zero on any natural-data move. Gated by `tests/test_eval_field_decisions_replay.py`.
  - Overstated reducer test docstring: scoped down.
  - Stale topology comments: fixed.
  - Analysis scripts: retired-name notes added.

## Restart required

**Manual SQL migration:** none.

Deploy from the primary checkout on main after merge. Order:

```bash
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-field-digester up -d --build
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-bus up -d --build --no-deps bus-observer
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-attention-runtime up -d --build
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-feedback-runtime up -d --build
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
```

- The digester goes first, so stored rows get pruned and renamed before readers see them.
- `--no-deps bus-observer` stops Redis (bus-core) from being restarted.
- Proposal, execution-dispatch, consolidation and any other `orion/` importers pick up the code on their next normal rebuild. Nothing in them changes behaviour.

## Proof queries (after deploy)

```sql
-- D3: no capability vector carries the old name; transport carries the new one (expect 0, then ~all)
select count(*) filter (where cv.vec ? 'contract_pressure'), count(*) filter (where s.field_json->'capability_vectors'->'capability:transport' ? 'catalog_drift_pressure')
from substrate_field_state s, jsonb_each(s.field_json->'capability_vectors') cv(cap, vec) where s.generated_at > '<deploy_ts>';
-- decision 4: observer channel gone from every node (expect 0); transport reliability unmeasured only when the RPC reading is absent
select count(*) from substrate_field_state s, jsonb_each(s.field_json->'node_vectors') nv(n, vec) where s.generated_at > '<deploy_ts>' and nv.vec ? 'observer_failure_pressure';
select count(*) from substrate_field_state where generated_at > '<deploy_ts>'
  and not (field_json->'capability_vectors'->'capability:transport' ? 'reliability_pressure')
  and (field_json->'node_vectors'->'node:substrate.rpc_delivery' ? 'rpc_timeout_pressure');  -- expect 0
-- decision 1: guard firings (expect rare)
select count(*) from substrate_feedback_frames where generated_at > '<deploy_ts>' and feedback_frame_json::text like '%before_winner_unmeasured%';
-- decision 2: adjusted novelty is visible on frames (non-zero only when a channel went dark/came back)
select count(*) from substrate_attention_frames where generated_at > '<deploy_ts>' and frame_json::text like '%novelty_common_channels_only%';
```

## Risks / concerns

- **Severity: low.**
  - **Concern:** transport reliability is now absent whenever the RPC bridge has no counted calls for 120 s: 53 of 123,086 ticks over 72 h. Readers already treat absence as unmeasured (#2534). The hub gate now reads "unknown" only when nothing else is active.
  - **Mitigation:** this is the outcome Juniper accepted.
- **Severity: low.**
  - **Concern:** decision 1 in `outcome_resolution` checks key and provenance only, with no stamp-age check, because the scoring window is settle-length and has no staleness bar of its own.
  - **Mitigation:** documented in code. Expiring channels are still caught.
- **Severity: low.**
  - **Concern:** live evidence is UNVERIFIED until deploy. The proof queries above check it.

## PR link

(filled on open)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
