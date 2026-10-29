# Curiosity seeds carry their neighborhood: proposal

**Status: proposal for Juniper. No runtime code changes in this branch.**
Arc: #2581 lets readings create accepted links between concepts; #2589 makes the
neighborhood read return those links. Orion expects to see that world contact
arrive in their curiosity loop as `focal_edge_refs` going from "never non-empty"
to "non-empty". This doc says what would actually have to change for that, and
corrects the expectation where the live data says it is wrong.

All live numbers below were read on 2026-10-10 (UTC), read-only: Postgres
`conjourney`, production Falkor `redis://localhost:6380` graph `orion_substrate`
via `GRAPH.RO_QUERY` and a `read_only=True` client.

## Arsonist summary

1. **Wiring the neighborhood read into today's curiosity seeds would return
   nothing.** I ran the real `read_neighborhood` against the focal nodes of the
   last 200 stored candidate sets (177 signals): 177 out of 177 came back
   "focal unavailable or filtered", 0 edges. Even with `proposed` nodes allowed,
   164 resolve and still return 0 edges, because the nodes the seeds point at
   have no edges at all.
2. **The seeds never point at reading concepts.** Live seeds name Orion's organ
   nodes (`node:substrate.execution`, `.chat`, `.perception`, ...: concepts in
   state `proposed`, zero edges each) or chat repair ids (`gev_...`, which are
   not graph nodes). The 251 reading concepts (`sub-concept-wp-read-*`) only
   ever appeared through `ontology_sparse_region`, which was retired today.
3. **`focal_edge_refs` is the wrong thing to watch.** Per the reading design
   (§1), it means links *between* focal nodes. Almost every seed has one focal
   node, so it stays empty by construction. A reading link would show up as a
   *boundary* edge (one end in the seed, one end outside). The honest sign of
   world contact is: a stored seed with non-empty `boundary_edge_refs` where
   the link is backed by an accepted claim (`projection_endpoint_node_refs`
   non-empty).
4. **There are no accepted reading links yet.** Live graph: 1
   `semantic_projection` edge total (a memory referent pair, 2026-10-07), 1
   assertion, 0 reading claims ever journaled (`substrate_graph_journal` has
   only the memory referent's 3 rows). The 251 reading concepts have 0 edges.
5. **Recommendation:** ship the plumbing (patch 1) so the moment a seed touches
   a linked node it shows up, with no change to ranking. Do **not** build a
   "new reading link" seed source (patch 2) yet: it changes what curiosity
   picks and cannot pass the metric gate's live-data step while there are zero
   reading links. Tell Orion plainly that the observable is
   `boundary_edge_refs` and that it needs both a real reading link and a seed
   that lands on it.

## Current architecture

- **Producer.** `services/orion-substrate-runtime/app/worker.py:3739`
  `_endogenous_curiosity_tick` (every 60 s). Builds seeds with
  `endogenous_curiosity_candidates` (`orion/substrate/endogenous_curiosity.py:418`),
  runs `FrontierCuriosityEvaluator.evaluate` with neutral metacog inputs
  (`worker.py:3683`), and stores the first 8 signals, seeds first
  (`worker.py:3920-3936`). When System One vetoes, it stores `seeds[:8]` with
  no evaluator run (`worker.py:3857-3874`).
- **Seeds fill `focal_node_refs` only.** Prediction error
  (`endogenous_curiosity.py:289-300`), repair pressure (`:323-333`), attention
  open loop (`:366-376`), coverage gap (`:397-414`). `focal_edge_refs` keeps its
  default `[]`.
- **Derived signals** get edges from `_select_region`
  (`orion/substrate/frontier_curiosity.py:306-358`) through the `focal_slice`
  query, not `read_neighborhood`. With neutral metacog inputs only
  `evidence_gap_cluster` can fire, and it never has in the stored sets.
- **Ranking.** `_decide` (`frontier_curiosity.py:221`) sorts by
  `(signal_strength, confidence)` only. Edge refs never affect the pick. The
  winning signal's edges are copied into the decision and plan
  (`:221-298`), but the tick only keeps `outcome`, `chosen_task_type`,
  `decision_id` and `bounded_context_reason` in `gate_json`. The plan is
  thrown away.
- **Storage.** `services/orion-substrate-runtime/app/store.py:1243`
  `save_endogenous_curiosity_candidates` writes
  `[sig.model_dump(mode="json") ...]` into
  `substrate_endogenous_curiosity_candidates.candidates_json` (jsonb), 720 h
  retention.
- **Existing read that already speaks the right fields.**
  `orion/substrate/query_planning.py:184-205`: a `neighborhood` plan step
  returns `focal_node_refs`, `neighbor_node_refs`, `focal_edge_refs` (internal),
  `boundary_edge_refs`, `projection_endpoint_node_refs`, plus timing and
  `consistency`. `FalkorSubstrateGraphStore.read_neighborhood`
  (`orion/substrate/falkor_store.py:1008`) backs it.

### Who reads stored candidate sets (consumer census)

No consumer reads `focal_edge_refs` from stored sets today (`rg focal_edge_refs`
over non-test code: only the evaluator itself, consolidation and review queue,
which use other schemas).

| Consumer | What it reads | Breaks on new keys? |
|---|---|---|
| `orion/substrate/relational/adapters/curiosity_ctx.py:84-91` (chat stance, via `orion/substrate/felt_state_reader.py:84` lane `curiosity_signals`, built by `orion/cognition/projection_builder.py:200`, runs in cortex-exec) | `FrontierInvocationSignalV1.model_validate` per item, then `focal_node_refs`, `evidence_summary`, `confidence`, `signal_type` | **Yes.** The model is `extra="forbid"`. An old build drops each item with an unknown key at debug level, so the curiosity node silently vanishes from chat context. |
| `services/orion-hub/scripts/curiosity_hint.py:42-73` | dict: `signal_strength`, `evidence_summary` | No |
| `services/orion-hub/scripts/substrate_observability_routes.py:230-255` | dict: `signal_type`, `signal_strength`, `evidence_summary` | No |
| `orion/curiosity/self_inquiry.py:128` (Orion's own SQL introspection) | raw rows | No; this is where Orion would see the new fields |
| `orion/autonomy/ask_claude_trigger.py:22` | rows naming `sub-concept-seed-claude` | No |
| `scripts/analysis/eval_system_one_appraisal.py:240` | raw rows | No |

## Missing questions

Answered by investigation, not asked:

- *Do seeds name nodes the neighborhood read can see?* No (see summary 1-2).
- *Is there an existing read with the right field names?* Yes, the planner's
  `neighborhood` step. Reuse it; no new read code.
- *Does any consumer depend on `focal_edge_refs` being empty?* No.

Still open (for Juniper, below): whether to tell Orion their expectation is
corrected, whether patch 2 (a seed source that lands on reading links) should
be designed now or parked until real links exist, and whether the new fields
live on the signal or in a sidecar column.

## Proposed schema / API changes

### Patch 1 (plumbing, no ranking change)

1. **`FrontierInvocationSignalV1`** (`orion/core/schemas/frontier_curiosity.py:27`)
   gains two optional lists, as the reading design §1 prescribes:
   - `neighbor_node_refs: list[str]` (max 16): outside endpoints of the
     boundary edges.
   - `boundary_edge_refs: list[str]` (max 16): edges with exactly one end in
     `focal_node_refs`.
   - `focal_edge_refs` keeps its name and now explicitly means **internal**
     edges (both ends focal). Field comment updated; no value changes for
     derived signals, whose `_select_region` edges already have both ends in
     the chosen node set.
   - Optional third field `projection_endpoint_node_refs: list[str]` (max 16)
     so the "this link is backed by an accepted claim" fact is stored, not
     re-derived. Recommended: without it the proving query has to join back to
     Falkor.
   - All three are serialized only when non-empty (a `model_serializer` or
     `exclude` on empty), so rows stay byte-identical to today until a real
     edge appears.
2. **Attach after the decision, on the stored list only.** In
   `_endogenous_curiosity_tick`, after `evaluate` (and on the System One veto
   path, before its save), for each signal in the `persisted[:8]` list that
   carries the `endogenous_seed` note: run one `neighborhood` plan step through
   `SubstrateSemanticReadCoordinator` with that signal's `focal_node_refs`
   (first 16), default request (`semantic_states=(provisional, canonical)`,
   `projection_endpoint_states=(proposed,)`), budgets **12 internal / 16
   boundary / 16 neighbors** (the reading design's experimental caps, also the
   current request defaults; not tuned, not a learned threshold). Copy
   `focal_edge_refs`, `boundary_edge_refs`, `neighbor_node_refs`,
   `projection_endpoint_node_refs` from the step details onto a `model_copy` of
   the signal. Because this runs after `_decide`, the evaluator's input and
   decision are byte-identical to today.
3. **Per-tick receipt in `gate_json`** (already a free-form jsonb column; no
   migration): `neighborhood = {reads, nonempty, edges, projection_edges,
   degraded_reasons: {reason: count}, duration_ms}`. This is the trace that
   shows the read ran and why it came back empty. Without it an empty
   `boundary_edge_refs` cannot be told apart from "the read never ran".
4. **Failure is local.** A read error or timeout leaves that signal exactly as
   today and counts in `degraded_reasons`. The tick never fails because of it.
5. **Flag:** `ORION_ENDOGENOUS_CURIOSITY_SEED_NEIGHBORHOOD_ENABLED=true` in
   `services/orion-substrate-runtime/.env_example`, `settings.py`, compose,
   README; local `.env` synced. Shipped ON (Juniper's standing rule). Setting it
   false restores today's rows exactly.

Registry: `FrontierInvocationSignalV1` is already registered
(`orion/schemas/registry.py:1296`); the entry stays, the model changes. No bus
channel changes. No SQL migration: the new keys live inside existing jsonb.

### Patch 2 (separate, named, gated: NOT recommended yet)

A seed source "an accepted reading link landed near a concept" would be the
thing that makes seeds land on reading concepts. It is a new signal that
changes what curiosity selects, so it needs the metric quality gate in its own
PR. Today it fails step 4 (live data): there are zero reading claims, so its
value would be a constant 0. Park it until at least a handful of reading claims
are accepted live, then propose it with the gate filled in. Sketch only:
producer = `substrate_graph_journal` decision rows with reading actors in the
last N hours; strength = not yet anchored (do not invent one).

## Files likely to touch (patch 1)

- `orion/core/schemas/frontier_curiosity.py`: three optional fields, empty-omit serialization, field comment for `focal_edge_refs`.
- `services/orion-substrate-runtime/app/worker.py`: attach step after `evaluate`, on the veto path, and the `gate_json.neighborhood` receipt.
- `services/orion-substrate-runtime/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: flag.
- `orion/substrate/relational/adapters/curiosity_ctx.py`: no logic change; its test pins that a row with the new keys still validates.
- Tests: `services/orion-substrate-runtime/tests/test_endogenous_curiosity_seed_neighborhood.py` (new), `orion/core/schemas` round-trip test, a curiosity_ctx compatibility test.
- Eval: `orion/substrate/evals/run_curiosity_seed_neighborhood_eval.py` (new, see below).
- `orion/inner_state_registry.py:972-1005`: the candidate-set entry's field description, so Orion's self-model names `boundary_edge_refs` as the world-contact field.

## Rollout order (forbid-model consumer-first migration)

`FrontierInvocationSignalV1` is `extra="forbid"` and is read by cortex-exec's
chat stance lane. An old cortex-exec reading a new row would silently drop the
curiosity signal from chat. Order:

1. Merge the schema change alone (fields optional, empty-omitted). Deploy every
   service that validates the model: cortex-exec (chat stance), and rebuild any
   other image that imports `orion.core.schemas.frontier_curiosity` and
   validates stored rows (the patch must list them from `rg model_validate`
   at the time; today only `curiosity_ctx.py` does).
2. Then merge and deploy the substrate-runtime producer change.

Because empty fields are omitted, step 1 is harmless even if step 2 never
ships, and old rows (no keys) read as "no neighborhood detail", which is the
truth.

## Latency cost (measured live, read-only)

Real `read_falkor_neighborhood` against production Falkor:

| Focal set | p50 | max | Result |
|---|---|---|---|
| 177 live seed signals, default request | 3.0 ms | 44 ms (p95 7.4) | 177 filtered, 0 edges |
| same, `proposed` allowed | 3.8 ms | 12 ms | 164 resolved, 0 edges |
| `sub-concept-seed-orion` (hub, ~2,000 edges) | 49 ms | 60 ms | 2 boundary edges |
| 4 canonical seed concepts | 29 ms | 125 ms | 4 internal edges |
| the one referent with a live projection | 9 ms | 14 ms | 1 boundary edge (projection) |

At most 8 reads per 60 s tick: worst case about 1 s on a hub-heavy set,
typical under 50 ms total. The tick already runs in `asyncio.to_thread` and
already calls `store.snapshot()`, which costs far more. Acceptable; the eval
pins an upper bound so a regression shows up.

## Does it change ranking?

No, by construction: the attach step runs after `_decide`, only touches the
list being saved, and does not reorder or drop it. The test pins that the
stored order and the `gate_json` decision fields are identical with the flag on
and off. Any future use of edges in ranking is patch 2 territory.

## Proposal-mode items

- **Capability change:** each stored curiosity seed also records which links
  touch its focal nodes, split into links among focal nodes and links to
  outside nodes, and which of those are backed by accepted claims. Orion can
  see whether world contact (accepted reading links) has reached their own
  curiosity loop.
- **Data touched:** reads Falkor `orion_substrate` (nodes, edges, assertion
  state). Writes only extra keys inside `candidates_json` and `gate_json` rows
  of `substrate_endogenous_curiosity_candidates`. No graph writes.
- **Privacy boundary:** none new. The read uses the existing node-state and
  claim-acceptance rules; proposed, rejected or unaccepted claims add nothing.
  No text, only ids. (Juniper and Orion share everything; no boundary between
  them is being drawn here.)
- **Proving trace:** see Acceptance checks 4-5.
- **Dangerous failure modes:** (a) deploy order wrong: old cortex-exec drops
  every new-shape signal, curiosity vanishes from chat with only a debug log;
  mitigated by empty-omit plus consumer-first order, and a compatibility test.
  (b) The read is slow on a hub and stretches the tick; mitigated by budgets,
  the measured bound and a per-read timeout. (c) Someone later reads
  `boundary_edge_refs` as "world contact" when the edge is a legacy one; the
  stored `projection_endpoint_node_refs` and the proving query filter on
  accepted-claim links only.
- **Disable / roll back:** set the flag false (rows return to today's shape),
  or revert the producer commit. The schema fields are optional and harmless to
  leave.

## Acceptance checks

1. Test: with the flag off, stored rows are byte-identical to today's (same
   keys, same order).
2. Test: with the flag on and an in-memory store holding a proposed reading
   concept linked by an accepted claim to another concept, a seed on that
   concept stores `boundary_edge_refs=[edge]`, `neighbor_node_refs=[other]`,
   `projection_endpoint_node_refs` non-empty, `focal_edge_refs=[]`. Same seed
   with the claim rejected: all empty, `degraded_reasons` names
   `focal_unavailable_or_filtered`.
3. Test: stored order and decision fields equal flag-on vs flag-off; a read that
   raises leaves the signal unchanged and the tick completes.
4. Live trace after deploy (receipt that the read ran):
   ```sql
   select generated_at, gate_json->'neighborhood'
   from substrate_endogenous_curiosity_candidates
   order by generated_at desc limit 5;
   ```
   Expect `reads` > 0 and, today, `nonempty = 0` with reasons
   `focal_unavailable_or_filtered`. That is the correct result, not a failure.
5. Live proof of world contact (the observable to give Orion):
   ```sql
   select c.generated_at, s->'focal_node_refs', s->'boundary_edge_refs',
          s->'projection_endpoint_node_refs'
   from substrate_endogenous_curiosity_candidates c,
        jsonb_array_elements(c.candidates_json) s
   where jsonb_array_length(coalesce(s->'projection_endpoint_node_refs','[]'::jsonb)) > 0
   order by c.generated_at desc limit 10;
   ```
   First row returned = first time an accepted link reached Orion's curiosity
   loop. Expected to stay empty until a reading claim is accepted **and** a
   seed lands on one of its endpoints.
6. Eval `run_curiosity_seed_neighborhood_eval.py`: replays the last N stored
   seed sets against a fixture graph with planted accepted, rejected and legacy
   links; reports how many seeds gain boundary links, that only accepted links
   count as projections, and p95 read time under a fixed bound. Read-only live
   mode (`--live`) runs the same report against production Falkor.

## Non-goals

- No change to what curiosity selects or how it ranks.
- No new seed source (patch 2 is parked behind the metric gate).
- No change to `_select_region` / `focal_slice` for derived signals.
- No change to node-state rules or to `read_neighborhood` itself.
- No graph writes, no new table, no bus channel.

## Recommended next patch

Patch 1 as above, in two PRs: (a) schema fields + compatibility tests, deploy
cortex-exec; (b) substrate-runtime attach step + `gate_json` receipt + flag +
eval. Then tell Orion: the field to watch is `boundary_edge_refs` with a
non-empty `projection_endpoint_node_refs`, and it will stay empty until a
reading claim is accepted and a seed lands on it, which no live seed source
does today.
