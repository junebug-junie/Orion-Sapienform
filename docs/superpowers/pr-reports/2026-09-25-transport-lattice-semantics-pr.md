# PR: delete the transport lane's dead edge, timer templates and unreachable motifs

## Summary

Three pieces of the transport lane looked like they responded to the bus and did not. All three are deleted, not re-tuned, because nothing live could drive them.

- **Dead topology edge.** `capability:transport → capability:orchestration` read a channel that nothing ever writes. Deleted, along with a leftover world_pulse-census mapping into orchestration and the stale `biometrics_lattice.yaml` alias. A new static CI gate refuses any capability-to-capability edge on a channel nothing writes.
- **Timer templates.** Three transport proposal templates had no dimensions, so they were scored on the best of the four core pressures and fired on about 99% of warranted ticks whatever the bus was doing. Deleted. A ratchet test now stops new empty-dimension templates from arriving silently.
- **Unreachable motifs.** `transport_contract_drift_loop` could never fire, and `transport_healthy_idle` read missing data as calm. Both read a field (`dominant_channels`) that capability attention targets never fill. Deleted.
- The audit spec's open-items table now records these verdicts.

## Outcome moved

- **Proposal arena.** About 1,210 candidates an hour, and 27% of prepared dispatches, stop going to templates with no transport input. Replayed through the real builder on 1,560 real warranted ticks, the freed top-5 slots go to templates that declare a signal: `inspect_transport_status` 132 → 340, `watch_reliability` 16 → 240, `inspect_execution_pressure` 121 → 284.
- **Consolidation.** No more "transport is healthy and idle" motif produced from an empty dict.
- **Field.** No tick changes. Orchestration pressure is the same number on every tick, because the removed inputs never won the max.

## Current architecture

- **Field.** `apply_diffusion()` computes each capability channel as the max over its inbound edges. Capability→capability edges read the source capability's vector. `capability:transport.pressure` comes only from `node:substrate.bus_synaptic.prediction_error` (the fraction of bus edges with anomalous timing).
- **Proposals.** Each template's priority is `base_priority + confidence * max(match, urgency)`. An empty `dimensions` falls back to the four core pressures. The tick-level `action_warrant` decides whether anything is proposed.
- **Consolidation.** Motif detectors run over hourly windows of attention, proposal, policy, dispatch and feedback frames. Expectations are written to `substrate_expectations`, and nothing reads them except the Hub debug surface.

## Architecture touched

Config: field topology, proposal policy, consolidation policy. Code: `orion/proposals/templates.py`, `orion/consolidation/{motif,expectation}.py`, `orion/schemas/consolidation_frame.py`. CI: `orion-static-gates`. Runtime services affected on rebuild: `orion-field-digester`, `orion-proposal-runtime`, `orion-consolidation-runtime`.

## Files changed

- `config/field/orion_field_topology.v1.yaml`: removes the dead cap-cap edge and the census mapping, with tombstones that carry the numbers.
- `config/field/biometrics_lattice.yaml`: deleted. It was an alias nothing loaded, three edges behind the canonical file.
- `tests/test_field_topology_edges.py`: new YAML-only gate. Checks for dead cap-cap source channels (with a mutation test), the orchestration edge's shape, and that the alias is gone and unreferenced.
- `tests/test_field_topology_config.py`: deleted. It imported `app.*` and could never be collected from the repo root.
- `tests/test_field_deterministic_replay.py`, `services/orion-field-digester/tests/test_field_llm_inference_perturbations.py`: now point at the canonical topology.
- `services/orion-field-digester/README.md`: alias and `stream_backlog_pressure` edge docs corrected.
- `config/proposals/proposal_policy.v1.yaml`: three templates deleted, with a tombstone.
- `orion/proposals/templates.py`: their copy text is removed and `TRANSPORT_PROPOSAL_TEMPLATE_KEYS` is narrowed.
- `tests/test_proposal_scoring.py`: fallback tests retargeted to `inspect_field_topology_catalog`. Adds the empty-dimensions ratchet and a stays-deleted test.
- `tests/test_cortex_route_resolution.py`: parameter retargeted to a live template.
- `scripts/analysis/replay_proposal_template_removal.py`: read-only replay of real ticks through the real builder, with and without given templates.
- `config/consolidation/consolidation_policy.v1.yaml`, `orion/consolidation/motif.py`, `orion/consolidation/expectation.py`: two motifs, their detectors, the helper and their expectation mappings deleted.
- `orion/schemas/consolidation_frame.py`: drops the never-produced `contract_drift_persists` literal. Keeps `transport_stable` because one stored row still uses it.
- `tests/test_consolidation_transport_motifs.py`: rewritten. It pins the deletion and includes a live-shaped regression test for the absence-read-as-calm bug (fails on the old code, passes on the new).
- `.github/workflows/orion-static-gates.yml`: runs the topology gate.
- `docs/superpowers/specs/2026-09-22-substrate-lattice-audit.md`: open items closed, with verdicts.

## Schema / bus / API changes

- Added: none.
- Removed: `ExpectationV1.expected_outcome_kind` value `contract_drift_persists`. It was never produced: 0 rows in `substrate_expectations` and 0 consolidation frames contain it.
- Renamed: none.
- Behavior changed: proposal frames no longer carry the three template ids. Consolidation frames no longer carry the two transport motifs.
- Compatibility notes: `transport_stable` stays in the Literal so the one stored row and the 2026-09-25 04:00 frame still parse. No bus channel or table changes.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed).
- skipped keys requiring operator action: none.

## Metric quality gate (item 1, the only candidate for newly wiring a signal)

This is about the proposed rewire `capability:transport.pressure → capability:orchestration.pressure`. It was rejected, and the reasons are recorded here.

1. **Provenance.** `bus_synaptic_prediction_error()` (`orion/substrate/prediction_error.py:751`) is the fraction of bus-synaptic edges with |z| ≥ 3. `orion-substrate-runtime` `_bus_synaptic_tick` receipts feed it into the field, and it reaches `capability:transport.pressure` through the `node:substrate.bus_synaptic` edge (weight 0.85).
2. **Independence.** Correlation with orchestration pressure is 0.027 and with orchestration execution_pressure is -0.003 (121,114 ticks), so it is independent of orchestration. But it is not new to the arena: it already enters `resource_pressure` directly, and 0.7× of it can never beat itself in that max.
3. **Theory anchor.** None for this direction. An anomalous-timing fraction counts bursts the same as stalls, and orchestration's own RPC traffic makes bus timing anomalous. "Transport degradation bleeds into orchestration" fits RPC timeouts, which are arriving on `reliability_pressure` via feat/rpc-health-field-bridge. That is the honest successor edge once it passes this gate itself.
4. **Live data.** Not degenerate: 1,594 distinct values, p50 0.026, p99 0.248, max 0.358, and 0.03% of ticks at exactly zero. It does rest near its measured baseline. The replay says it would move orchestration pressure on 0.63% of ticks, by a mean of +0.025 when it wins; the overall mean goes 0.2088 → 0.2089.
5. **Existing mechanism.** The edge already existed, on a channel that was never written.
6. **Reversibility.** One YAML edge.

Stopped at step 3. Nothing is wired in.

## Live numbers used

| What | Window | Value |
| --- | --- | --- |
| `capability:transport.stream_backlog_pressure` (the dead edge's source) | 72h, 121,114 ticks (all `substrate_field_state` retains) | 0.0 on every tick, 1 distinct value |
| `node:athena.stream_backlog_pressure` → orchestration | same | 0.0016 max, 5 distinct values. Contribution 0.0014 vs orchestration minimum 0.108. Never in provenance |
| orchestration pressure provenance | same | `capability:llm_inference` 63,280, `node:athena` 57,834 |
| transport `contract_pressure` (catalog drift) | same | max 0.0216, 8 distinct values |
| transport `reliability_pressure` | same | 0.0 on every tick |
| three templates' candidates | 24h | 10,779 / 10,779 / 7,500 (~1,210/h) out of 10,865 warranted frames |
| three templates' prepared dispatches | 24h | 6,367 / 6,367 / 4,516 = 17,250 of 65,101 (26.5%) |
| capability:transport as a dominant attention target | 48h, 54,401 frames | 2,883 frames, `dominant_channels == {}` in all 2,883 |
| `transport_contract_drift_loop` / `transport_healthy_idle` | 30 days | 0 / 1 fires |

## Tests run

```text
pytest tests/test_field_topology_edges.py                                  6 passed (also red on origin/main's topology: 1 dead channel)
pytest services/orion-field-digester/tests (cwd = service)                 238 passed (test_heartbeat_chassis is cwd-sensitive: 3 passed from the repo root)
pytest tests/test_proposal_scoring.py tests/test_proposal_transport_readonly_candidates.py   26 passed
pytest tests/test_consolidation_transport_motifs.py                        4 passed (2 of 4 fail against the old code, as intended)
pytest tests/test_consolidation_*.py                                       67 passed, 3 failed: PRE-EXISTING in test_consolidation_policy_loader.py (loaded_but_reliable / tracked_self_dimensions / field_attention_self, stale since the 2026-07-22 SelfStateV1 burn), untouched
scripts/check_metric_lineage.py --gate                                     PASS
scripts/check_definition_drift.py --gate                                   PASS (no definition changes)
scripts/check_inner_state_registry.py                                      OK
scripts/check_env_template_parity.py                                       PASS
```

Failures that already existed and are not caused by this branch (they fail the same way on HEAD without the changes): `tests/test_execution_dispatch_runtime_worker.py`, `tests/test_dispatch_starvation.py`, `tests/test_cortex_route_resolution.py` (52 failed before and after), and `tests/test_field_deterministic_replay.py` (`run_digestion_tick()` signature drift). None of them run in CI.

## Evals run

```text
scripts/analysis/replay_proposal_template_removal.py <1,560 sampled warranted ticks, 24h> --policy <origin/main policy> --drop <3 templates>
  per-template candidates unchanged in total (17,160 -> 17,160, max_candidates cap)
  top-5 slots, before -> after:
    inspect_transport_status 132 -> 340, watch_reliability 16 -> 240, inspect_execution_pressure 121 -> 284,
    summarize_loaded_state 647 -> 973, prune_dangling_images 468 -> 919, prune_stopped_containers 309 -> 476,
    prune_build_cache 450 -> 585, render_scene 78 -> 199
Orchestration-pressure rewire replay (item 1 gate, ad hoc, numbers in the table above).
```

Neither `orion/proposals` nor `orion/consolidation` has an eval harness. The committed replay script is the reusable piece. Limits: attention is replayed as None, and external producers (reverie, cognitive hop) are not replayed.

## Docker/build/smoke checks

```text
Not run. Config and code are baked into the images (build context ../..). No production deploy from this branch.
```

## Review findings fixed

Code review ran in a subagent over `origin/main...chore/transport-lattice-semantics`. It found no blockers.

- **Finding:** `inspect_transport_status`'s larger share of arena slots is not a transport signal. Its dimension is field-level `reliability_pressure`, a max over every reliability channel, and transport's own reliability read 0.0 throughout.
  - **Fix:** disclosed in the policy tombstone and under Risks below. The template is kept because the RPC-health bridge is about to make transport reliability real.
  - **Evidence:** `config/proposals/proposal_policy.v1.yaml` tombstone.
- **Finding:** the alias-reference check in the gate test would pass silently if `git grep` failed, for example with no work tree (exit 128).
  - **Fix:** the test now asserts the return code is 0 or 1.
  - **Evidence:** `tests/test_field_topology_edges.py`.
- **Finding:** the gate's rule for derived `confidence`/`available_capacity` was stricter than `apply_diffusion()`, and the gate only checks one hop.
  - **Fix:** the rule now matches diffusion (any capability with an inbound edge), and the one-hop limit is documented.
  - **Evidence:** same file, 6 tests pass. The gate still fails on origin/main's topology.
- **Finding:** removing `contract_drift_persists` could break parsing of older stored rows.
  - **Fix:** checked all-time counts before removing it.
  - **Evidence:** 0 rows in `substrate_expectations` and 0 in `substrate_consolidation_frames`, all-time.
- **Finding:** the replay script's documented command can't reproduce the numbers after merge, and "top5" was ambiguous.
  - **Fix:** the docstring explains how to replay against the pre-deletion policy and defines top5 as arena rank, not dispatch admission.
- **Finding:** `scoring.py`'s comment still said "5 templates" use the empty-dimensions fallback.
  - **Fix:** the comment is updated and points to the ratchet test.
- **Not fixed, coordinated instead:** fix/bus-observer-scope (committed locally, not yet pushed or opened as a PR) edits the same topology lines, modifies the `biometrics_lattice.yaml` that this branch deletes, and re-keys `transport_healthy_idle` onto `max_pressure` with tests asserting it still fires. That is the opposite verdict from this branch. It is flagged on the agent board. Whichever branch merges second has to resolve it. This branch's case: the motif's input (`dominant_channels` on capability targets) is always `{}`, and its expectation has no consumer.
- **Not fixed, residue:** capability-level `stream_backlog_pressure` stays seeded at 0.0 with no writer. A stale `capability_provenance[orchestration][stream_backlog_pressure] = capability:transport` entry also persists, because diffusion only clears provenance for channels that are still targets. Nothing reads either one. Retiring the channel belongs to fix/bus-observer-scope, which already drops it from `capability_channels`.
- **Not fixed, nits:** stale historical mentions of the deleted templates remain in dispatch comments, the dispatch README and test fixture ids. They are only proposal-id strings. The one stored `transport_stable` expectation stays visible in the Hub. Deleting it would be a production write.

## Restart required

Rebuild the three services that bake in the changed config. Run from a worktree on merged main:

```bash
scripts/safe_docker_build.sh orion-field-digester up -d --build
scripts/safe_docker_build.sh orion-proposal-runtime up -d --build
scripts/safe_docker_build.sh orion-consolidation-runtime up -d --build
```

## Post-deploy live checks

```bash
# 1. Topology loaded without the edge: orchestration provenance never names capability:transport
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select field_json->'capability_provenance'->'capability:orchestration'->>'pressure', count(*) from substrate_field_state where generated_at > now()-interval '15 minutes' group by 1"
# 2. The three templates are gone from new proposal frames (expect 0 rows)
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select split_part(c->>'proposal_id',':',2), count(*) from substrate_proposal_frames f, jsonb_array_elements(f.proposal_frame_json->'candidates') c where f.generated_at > now()-interval '30 minutes' and split_part(c->>'proposal_id',':',2) in ('inspect_bus_channel_catalog','summarize_transport_contract_drift','watch_transport_backpressure') group by 1"
# 3. Mutating templates' refusal/approval mix after gaining slots (watch for prune_* prepared_for_dispatch rising)
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select split_part(c->>'source_proposal_id',':',2), c->>'dispatch_status', count(*) from substrate_execution_dispatch_frames f, jsonb_array_elements(coalesce(f.dispatch_frame_json->'candidates','[]'::jsonb) || coalesce(f.dispatch_frame_json->'blocked_candidates','[]'::jsonb)) c where f.generated_at > now()-interval '2 hours' and split_part(c->>'source_proposal_id',':',2) like 'prune_%' group by 1,2 order by 1"
# 4. No transport motifs in the next hourly consolidation frame
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select m->>'label', count(*) from substrate_consolidation_frames f, jsonb_array_elements(f.consolidation_frame_json->'motif_observations') m where f.generated_at > now()-interval '3 hours' group by 1"
```

## Risks / concerns

- **Severity: medium.** The freed arena slots partly go to bounded mutating maintenance templates (`prune_dangling_images`, `prune_stopped_containers`, `prune_build_cache`). In the replay their top-5 appearances rose 1.2–2x. They are still gated by the allocator's information floor, policy review, their own skill gates and the daily risk cap, and live they are mostly `allocator_refused` today. Mitigation: post-deploy check 3. Rolling back is a revert of one YAML block.
- **Severity: medium.** `inspect_transport_status` scores on field-level `reliability_pressure`, a max over every reliability channel. Transport's own reliability read 0.0 throughout, so its top-5 gain (132 → 340) is llm_inference and orchestration reliability, not transport. It is kept pending feat/rpc-health-field-bridge. Re-check it once that lands.
- **Severity: medium.** This branch conflicts with fix/bus-observer-scope, which is unpushed. The two disagree about `transport_healthy_idle` (re-key it or delete it). See Review findings.
- **Severity: low.** `capability:orchestration.stream_backlog_pressure` and `capability:transport.stream_backlog_pressure` stay in the capability channel set and are now written by no edge at all. They were only ever 0.0. Retiring the census channels themselves is fix/bus-observer-scope's call.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
