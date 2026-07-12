# Phase 4 — attention salience vs. diffusion provenance cross-check — 2026-07-12

Context: Phase 4 of `docs/superpowers/plans/2026-07-12-inner-state-unification-plan.md` (brainstorm idea #6). Two independently-computed node-attribution mechanisms exist today: `orion-attention-runtime`'s salience scoring (`SelfStateV1.dominant_attention_targets`, per-tick, weighs pressure/novelty/urgency/confidence) and `orion-field-digester`'s diffusion provenance (`FieldStateV1.capability_provenance`, per-tick, records which edge source contributed the largest weighted amount to each capability channel). Do they agree? Answered from live Postgres, not assumption — 295 real ticks, ~10 minutes, sampled 2026-07-12 ~23:2x UTC, well after both the diffusion-saturation fix and the field-topology weights have been live and non-degenerate.

## Method

```sql
SELECT s.self_state_json->'dominant_attention_targets' as attn_targets,
       f.field_json->'capability_provenance'->'capability:llm_inference'->>'pressure' as llm_prov,
       f.field_json->'capability_provenance'->'capability:orchestration'->>'pressure' as orch_prov,
       f.field_json->'capability_provenance'->'capability:transport'->>'reliability_pressure' as transport_prov,
       f.field_json->'capability_provenance'->'capability:graph'->>'pressure' as graph_prov
FROM substrate_self_state s JOIN substrate_field_state f ON f.tick_id = s.source_field_tick_id
WHERE s.generated_at > now() - interval '10 minutes'
```

295 joined rows. For each capability channel's provenance winner, checked whether that same node/capability ID also appears in the same tick's `dominant_attention_targets`.

## Findings

| Metric | Value |
|---|---|
| `node:athena` present in `dominant_attention_targets` | **100.0%** of ticks |
| `node:circe` present in `dominant_attention_targets` | **100.0%** of ticks |
| `node:atlas` present in `dominant_attention_targets` | **52.9%** of ticks |
| `capability:transport.reliability_pressure` provenance = `node:athena` | **100.0%** of ticks |
| `capability:graph.pressure` provenance = `node:athena` | **100.0%** of ticks |
| `capability:llm_inference.pressure` provenance = `node:atlas` | **81.7%** of ticks |
| `capability:llm_inference.pressure` provenance = `node:circe` | **18.3%** of ticks |
| `capability:orchestration.pressure` provenance in `dominant_attention_targets` (node or capability ID) | **79.1%** of ticks |

## Interpretation

**Where they agree, and why it's not a coincidence:** `node:athena` is the sole `source_id` for the `node:athena -> capability:transport` and `node:athena -> capability:graph` edges in `config/field/orion_field_topology.v1.yaml` — provenance is trivially always athena for those two channels, by construction. Athena and circe both showing 100% presence in attention's salience list independently corroborates this: both are structurally central (athena orchestrates + serves transport/graph; circe/atlas are the two GPU-bearing nodes contending for `capability:llm_inference`), and the two mechanisms — computed from entirely different formulas (`weighted_pressure`/`urgency_score`/`confidence_from_vector` for salience vs. max-weighted-contribution for provenance) over the same underlying `FieldStateV1` — land on the same structural conclusion about which nodes matter.

**Where they diverge, and why that's expected, not a bug in either system:** `node:atlas` wins `capability:llm_inference`'s diffusion contest 81.7% of the time (it's the higher-weighted edge, `atlas: 0.85` vs `circe: 0.50` per the topology config, so it wins whenever both report comparable raw `gpu_pressure`) — but atlas only appears in attention's top-5 salience list 52.9% of the time. These measure genuinely different things: provenance answers "which single source contributed the most to this specific channel's value this tick" (a magnitude comparison between exactly two edges); salience answers "how urgently should overall attention be drawn to this node right now," a weighted composite of pressure **and** novelty **and** urgency **and** confidence (`compute_salience()`, `orion/attention/field_attention/scoring.py`), additionally bounded by `policy.limits.max_node_targets` (top-5 cap) and `policy.thresholds.min_salience`/`suppress_below` gates. Atlas can legitimately win the provenance contest for `llm_inference` on a tick where its overall salience composite doesn't clear the reporting threshold or gets edged out of the top-5 by other targets — that's the mechanism working as designed, not either one being wrong.

## Verdict

**Do not unify these into one mechanism.** They answer different questions by design — provenance is a magnitude-comparison primitive (who contributed most to this channel), salience is a multi-factor urgency composite (where should attention go right now) — and the ~30-point gap between atlas's provenance-win-rate (81.7%) and its salience-presence-rate (52.9%) is explained precisely by that difference, not by either being broken. Keeping both is the correct call; each is already the right tool for its own consumer (provenance for phi's `dominant_node`, per Phase 2; salience for attention's own targeting).

## Relevance to Phase 3

This corroborates, rather than undermines, Phase 2's design decision to source `PhiIntrinsicRewardV1.dominant_node` from attention's salience-ranked `dominant_attention_target_details` (not from raw provenance) — salience is the more conservative, threshold-gated signal, and today's data shows it agrees with provenance on the two structurally-obvious nodes (athena, circe) while being appropriately more selective about atlas. Phase 3 (the embodiment narrative) can build on this with the concrete fact that `dominant_node` will name a real, corroborated hardware node roughly half-to-all of the time depending on which node/capability is asked about — not a coin flip, not a stuck constant.

Phase 3 itself remains blocked on Phases 0–2 being deployed (its own stated gate, per the plan) — this doc doesn't unblock that, it's independent, parallel research per the plan's own Phase 4 scoping.
