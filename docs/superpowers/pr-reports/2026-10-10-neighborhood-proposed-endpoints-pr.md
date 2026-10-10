## Summary

Arc: #2581 lets world-pulse readings create accepted relationship claims between concepts. Almost every concept is still `proposed`, so the standard neighborhood read hid those links. This patch makes the read show them.

- A neighborhood read now decides "may this node be shown?" per edge, not per node. On an ordinary (legacy) edge, both nodes still need to be `provisional`/`canonical`. On a link backed by an accepted claim (a walkable `semantic_projection` edge: the claim is `provisional`/`canonical` and at the revision the link was built from), a `proposed` node may also be shown. The claim vouches for the link; the concept's own state does not have to.
- A `proposed` starting node is accepted only if it has at least one such link in the requested direction, and it must leave the read with at least one of those links. If the size limits cut every link it has, the node is reported missing (with `truncated`) instead of being returned on its own. Otherwise it is reported missing exactly as before.
- Rejected/deprecated nodes are never let in this way. Rejected, deprecated, never-accepted or out-of-date claims let nothing in. Provenance and claim-wiring edges still never walk.
- Nodes shown only because of a claim are listed in the new result field `projection_endpoint_node_ids` (planner detail `projection_endpoint_node_refs`).
- In-memory and Falkor backends match exactly, checked on real FalkorDB 4.18.11 and 6.0.1. The SPARQL/GraphDB backend stays fail-closed: it stores no claim nodes, so it still walks legacy edges only.
- Opt out with `projection_endpoint_states=()`, which brings back the old node-state-only rule.

## Outcome moved

A default `read_neighborhood` from a reading concept that has an accepted link now returns the link and both concepts. Before, it returned `focal_unavailable_or_filtered` (#2581 concern 2). The new test file reproduces that failure: 9 of its 18 tests fail on `main`. The live Falkor parity test also fails against the old Cypher.

## Current architecture

`NeighborhoodRequestV1.eligible()` was a node-only filter (`semantic_states`, default provisional/canonical). Every backend required both edge endpoints to pass it, then applied `walkable_edge` (#2515) to the edge. A claim-backed link between two proposed concepts therefore failed on node state before the claim's acceptance was ever looked at.

## Architecture touched

- `orion/substrate/neighborhood.py`: request field `projection_endpoint_states` (default `("proposed",)`) plus helpers `projection_only` / `endpoint_eligible` / `projection_states`. The driver admits a proposed focal only if `groups([id])` returns a group in the requested direction. The final endpoint check is now per edge (`endpoint_eligible(node, edge.edge_role)`), so a backend that returns a legacy edge into a proposed node fails the whole read closed. Adds result field `projection_endpoint_node_ids`. The memory backend uses the per-edge rule.
- `orion/substrate/neighborhood_backends.py`: the Falkor `where()` adds an OR branch: `(both endpoints IN $states) OR (e.edge_role = 'semantic_projection' AND both endpoints IN $projection_states)`. This sits before the existing walkability tail, so the claim must still be accepted at the matching revision. The index-seek `WITH` barrier is unchanged (the EXPLAIN test passes on both engines). SPARQL is unchanged apart from a comment that documents the fail-closed choice.
- `orion/substrate/query_planning.py`: the neighborhood step detail now includes `projection_endpoint_node_refs`.
- Dynamics, attention and `eligibility.py` are untouched. The isolation test `test_dynamics_is_identical_before_and_after_projecting_memory_structure` still passes.

## Files changed

- `orion/substrate/neighborhood.py`: per-edge endpoint rule, proposed-focal anchoring, receipt field.
- `orion/substrate/neighborhood_backends.py`: Cypher OR branch; SPARQL fail-closed comment.
- `orion/substrate/query_planning.py`: planner detail.
- `orion/substrate/tests/test_neighborhood_projection_endpoints.py` (new): the fixture plus negatives (memory, driver, planner, SPARQL).
- `orion/substrate/tests/test_neighborhood_falkor_live.py`: real-Falkor test for the same rule. The existing random parity test now also covers proposed endpoints, because its fixture mixes proposed nodes with accepted claims.
- `.github/workflows/substrate-neighborhood.yml`: runs the new test file.
- `docs/plans/substrate/2026-10-06-neighborhood-read-contract.md`: dated amendment to the eligibility rule.
- `orion/substrate/README.md`: concept row for "Projection endpoint".

## Schema / bus / API changes

- Added: `NeighborhoodRequestV1.projection_endpoint_states` (in-process request model; `schema_skew_discovery` declares no cross-service writer) and `NeighborhoodResultV1.projection_endpoint_node_ids` (in-process dataclass).
- Removed / renamed: none.
- Behavior changed: default reads can now include `proposed` nodes, but only as endpoints of accepted claim links.
- Compatibility: no bus payload or persisted schema changes. Request JSON without the new key gets the default.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: no. Local `.env` sync: not needed.
- Skipped keys: none.

## Tests run

```text
pytest orion/substrate/tests services/orion-hub/tests/test_world_pulse_read_assertion_links.py \
  services/orion-memory-consolidation/evals/test_referent_graph_discipline_eval.py
  -> 1120 passed, 39 skipped, 3 failed. The 3 failures are in test_felt_state_self_definition_lane.py
     and also fail on unmodified main (self_concept_history SQL shape); this patch does not touch that code.
pytest orion/substrate/tests/test_neighborhood_projection_endpoints.py -> 21 passed (11 fail on main; 3 fail on the
  pre-review commit 0291e95cb, i.e. they pin both review fixes)
reviewer's differential script (90 focal/direction/state combos incl. self-loops, proposed<->proposed,
  rejected claims) memory vs real Falkor 4.18.11 and 6.0.1 -> 0 diffs on both, re-run after review fixes
ORION_TEST_FALKOR_URI=<throwaway 4.18.11> pytest test_neighborhood_falkor_live.py test_falkor_direct.py \
  test_falkor_anchor_store.py test_assertion_core_falkor.py -> 48 passed
same against throwaway 6.0.1 -> 48 passed
old neighborhood_backends.py + new driver against real Falkor -> parity + new live test FAIL (the regression is caught)
scripts/check_metric_lineage.py --gate -> PASS; scripts/check_definition_drift.py --gate -> PASS
```

## Evals run

```text
python -m orion.substrate.evals.run_neighborhood_eval -> exit 0 (hub budget/coverage unchanged)
services/orion-memory-consolidation/evals/test_referent_graph_discipline_eval.py -> passed
```

## Docker/build/smoke checks

```text
Read-only probe against production Falkor (redis://localhost:6380, read_only client, no writes), old vs new code:
  proposed hub concept (legacy edges only): filtered before and after (focal_unavailable_or_filtered)
  sub-concept-seed-orion: identical receipt (2 boundary edges)
  referent with the one live semantic_projection: identical receipt
Live graph today: 1 semantic_projection edge total (provisional referent <-> provisional referent),
1 assertion node, 0 reading claims. So "a proposed reading concept returns its link on production"
is UNVERIFIED until a reading claim is accepted.
```

No Docker build needed. This is library code read by its callers in-process.

## focal_edge_refs in curiosity candidate sets (question from Orion)

Short answer: **no**. These links cannot reach `focal_edge_refs` in curiosity candidate sets, with or without this patch. The candidate producer does not use the neighborhood read, and on its live path nothing writes edge refs at all.

- Live: `substrate_endogenous_curiosity_candidates` has 23,020 sets, 0 with any non-empty `focal_edge_refs`. The signals in those sets are `curiosity_candidate` seeds (repair_pressure 22,294, prediction_error 4,649, attention_open_loop 1,522) and `ontology_sparse_region` (12,958, retired 2026-10-10).
- The sets are written by the substrate-runtime endogenous tick: `services/orion-substrate-runtime/app/worker.py:3898-3934` runs `FrontierCuriosityEvaluator.evaluate` and saves the first 8 signals, seeds first.
- The seeds never set edge refs. `orion/substrate/endogenous_curiosity.py:289-300`, `:323-330`, `:366-373` and `:397-404` build `FrontierInvocationSignalV1` with `focal_node_refs` only, so `focal_edge_refs` keeps its default `[]`.
- Derived signals get edge refs from `FrontierCuriosityEvaluator._select_region` (`orion/substrate/frontier_curiosity.py:306-358`). That uses the `focal_slice` query (`query_planning.py:231` -> `store.query_focal_slice`, served from the in-process cache at `falkor_store.py:1014` / `store.py:225`), **not** `read_neighborhood`. It keeps an edge only if both endpoints are in its own top-8 node pick (`frontier_curiosity.py:353-357`). `focal_slice` already applies `walkable_edge` and no node-state filter, so it was never blocked by the `proposed` rule this patch fixes.
- But the worker passes neutral metacog inputs (`worker.py:3683` `_neutral_frontier_metacog_inputs`: no contradictions, drift inactive, pressure 0.1, not operator-requested, identity conflict inactive). That leaves `evidence_gap_cluster` as the only derived signal that can run `_select_region` on this path, and it needs at least 2 nodes with `frontier_hypothesis_marker`. It has never appeared in the persisted sets.

What would make it true: the seed producer would have to attach accepted links to its seeds, for example by reading the neighborhood of each seed's focal node and filling `focal_edge_refs` / `boundary_edge_refs`. That is the migration the reading design doc's §1 already describes ("replacing the ambiguous use of focal slice for curiosity"). It changes a curiosity/autonomy loop, so CLAUDE.md requires proposal mode first, and it is not done here. Follow-up below.

## Review findings fixed

Review by an orion-repo-agent subagent. It found no must-fix issues and confirmed exact memory/Falkor parity and an intact index seek (EXPLAIN on both engines).

- Finding (should-fix): a proposed starting node could be returned with no link vouching for it once budgets cut its only link (for example `boundary_edge_limit=0`). That contradicted the documented rule.
  - Fix: after edge selection, the driver drops a projection-only focal that no kept edge touches and reports it in `missing_focal_node_ids` (`truncated` is already set).
  - Evidence: `test_a_proposed_focal_never_leaves_without_a_vouching_edge`, plus the updated `boundary_edge_limit=0` case in `test_budgets_and_truncation_still_bind_projection_neighbors` (both fail on 0291e95cb).
- Finding (should-fix): with `direction="outgoing"` and focals P->Q (both proposed), Q was listed in `missing_focal_node_ids` (degraded) while also being returned as P's neighbor.
  - Fix: `missing_focal_node_ids` now excludes requested ids that come back as neighbors.
  - Evidence: `test_a_requested_id_returned_as_a_neighbor_is_not_reported_missing`.
- Finding (nit): `semantic_states` no longer means "only these states" on its own.
  - Fix: documented in the field comment (`projection_endpoint_states=()` is the strict mode).
- Finding (nit): the SPARQL comment claiming the empty `IN ()` path is never entered is now false.
  - Fix: the comment is corrected. That path is still fail-closed (`unavailable:` at worst).
- Finding (nit): the batching oracle never exercises projections.
  - Fix: a docstring note points to the tests that do. Self-loop case added (`test_self_loop_projection_does_not_anchor_a_proposed_focal`).
- Not changed (nit): the Falkor anchor probe pages all predicates rather than doing a LIMIT-1 existence check. It is bounded by 16 focals and listed under risks.

## Restart required

```text
No restart required for correctness: no production caller uses read_neighborhood by default yet.
Callers pick the change up on their next deploy from main.
```

## Risks / concerns

- Severity: low. Concern: each `proposed` starting node costs one extra edge-group probe (2 indexed predicate queries on Falkor), about +9 ms in the live probe for a proposed hub. Mitigation: only proposed starting nodes pay it; at most 16 per request.
- Severity: low. Concern: the driver can confirm the edge role, but it cannot confirm claim acceptance itself. Mitigation: every backend enforces `walkable_edge` (memory) or its Cypher twin (Falkor); SPARQL never walks projections.
- Severity: medium (follow-up). Concern: curiosity `focal_edge_refs` will stay empty (see above). Follow-up: a proposal to have endogenous seeds attach accepted links through `read_neighborhood`.
- Severity: info. Concern: no reading claim is accepted on production yet, so the live end-to-end path is UNVERIFIED.

## PR link

(filled in after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
