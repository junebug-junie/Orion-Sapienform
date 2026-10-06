# Bounded bidirectional neighborhood reads

This implements the **recommended first patch** of
[the reading property graph design](2026-10-06-reading-property-graph-design.md).
It adds a direct read contract and diagnostic replay. The assertion pipeline,
complete hydration repair, persisted curiosity-candidate migration, and migration
of existing runtime consumers remain subsequent patches. No existing caller is
silently given a different graph or eligibility predicate.

## API

```python
from orion.substrate.neighborhood import NeighborhoodRequestV1

result = store.read_neighborhood(NeighborhoodRequestV1(
    focal_node_ids=("node-a", "node-b"),
    internal_edge_limit=12,
    boundary_edge_limit=16,
    neighbor_node_limit=16,
))
```

The query coordinator also accepts a `neighborhood` step with the same request
fields. Its `details` distinguish `focal_node_refs`, `neighbor_node_refs`,
`focal_edge_refs` (internal), and `boundary_edge_refs`. These are query receipts,
not additions to the persisted candidate or bus schemas.

Only Concept/Entity endpoints in the requested promotion states and anchor scopes
are eligible. Default states are `provisional` and `canonical`; `proposed` is an
explicit diagnostic opt-in. The existing edge schema has no acceptance state;
node eligibility does **not** assert that a relationship was reviewed. Evidence
and other nonsemantic endpoint kinds do not compete for budgets. This contract
must be extended with assertion acceptance/edge-role filtering when that schema
lands. No new cognitive metric is introduced.

Anchor scope is a subject filter, **not an ACL**. This is an internal store API
under the caller's existing graph access boundary. It is not a public endpoint or
an implementation of the future reading-source visibility contract.

Internal edges have both endpoints in the resolved focal set. Boundary edges
have exactly one focal endpoint. Direction selects incoming, outgoing, or both
boundary directions; internal edges preserve their actual direction regardless.
Separate budgets prevent a large boundary from starving internal edges. Boundary
allocation rotates through focal nodes, then directions and predicates, then
stable edge IDs. Every returned edge has both endpoint nodes in the result.

Caps: at most 16 requested focal IDs and 256 each for internal edges, boundary
edges, and neighbors; zero is supported. Durable reads filter topology before
limits, enumerate existing groups, and fetch at most budget+1 edges per group.
Keyset pages continue across short pages. A neighbor cap can underfill the edge
budget: this is deliberately reported as truncation, not exhaustive coverage.
This is bounded response/candidate work, not a bound on database index traversal.

## Receipts and consistency

- `complete_for_request`: all eligible edges in the requested directions were
  returned, no budget omitted data, and every requested focal node resolved.
- `truncated`: at least one candidate edge was omitted by an edge/neighbor budget.
- `degraded`: missing/filtered focal nodes, query/decoding failure, invalid
  endpoint linkage, or an unsupported continuation. A query failure returns no
  partial successful graph and never substitutes a stale cache.
- `reason`, missing focal IDs, read start/end, and source backend distinguish an
  empty graph from an unavailable one. Backend exception messages are redacted.
- `consistency=best_effort_non_atomic`: multiple read statements do not establish
  snapshot isolation. Detected endpoint changes fail closed; concurrent changes
  that do not violate an invariant may still yield a mixed-time result.
- Continuations are **not issued in this patch**. Supplying one returns
  `restart_required:continuation_unsupported`. Restart with a new request. A
  stable, scope-bound paging token needs a separate consistency contract.

Falkor reads native durable properties directly with `hydrate=False` supported.
They do not migrate legacy blobs, warm the full cache, or mutate any data. SPARQL
uses the existing RDF payload contract with equivalent filtering. Routed reads
use only the primary backend. Existing `snapshot` and focal-slice APIs retain
their old behavior, including their known completeness limitations.

## Verification and replay

```bash
python -m pytest orion/substrate/tests/test_neighborhood.py orion/substrate/tests/test_neighborhood_backends.py -q
python -m orion.substrate.evals.run_neighborhood_eval
python -m scripts.replay_substrate_neighborhood --request /tmp/request.json --uri redis://localhost:6380
```

The replay requires an explicit request JSON and endpoint. It constructs a
read-only client, does not hydrate, and outputs graph IDs and a read receipt,
without source text. Run it only under the operator's existing graph permissions.
The URI names FalkorDB, not the Orion bus. This patch does not access the bus.

Live read-only replay on 2026-10-06, candidate
`curiosity-e914a662ce620e47d36e1932`: eight focal nodes, four internal edges,
3,355 incoming boundary edges and no outgoing boundary edges. Including proposed
nodes explicitly, budgets 12/16/16 returned 4/16/16 in 1,186.61 ms; all endpoints
were present; `truncated=true`, `degraded=false`. The independently queried
eligible census matched the all-edge census for this focal set. This latency is
an experimental diagnostic result, not a per-tick performance claim.

Local evidence: `/tmp/reading-neighborhood/candidate.json`, `request.json`,
`live-replay.json`, and `live-census.json`. No service restart or production write
was performed. The proposed reading assertion → review → projection → UI path
remains **UNVERIFIED** and outside this patch.

## Round trips and index seeks (memory Stage 2, PR E)

Same contract, same receipts; fewer and cheaper queries.

- **Neighbors are fetched in one call.** The algorithm first decides which
  boundary edges fit the budgets (that needs only ids), then calls `nodes()`
  once with every admitted neighbor. `nodes()` now runs at most twice per read
  (focal, then neighbors). A missing, duplicate or ineligible neighbor still
  fails the whole read closed. Falkor reads them with one indexed
  `n.node_id IN $node_ids` query (`LIMIT 2n`, so duplicates stay visible).
  SPARQL keeps per-id reads, because an endpoint may cap result rows.
- **Group reads seek the `node_id` index.** Both FalkorDB 4.18 and 6.0 planned
  every *incoming* group read as a label scan over all source nodes (50–120 ms
  each on the live graph). Group reads now bind the focal endpoint first
  (`MATCH (target:SubstrateNode) WHERE target.node_id = $focal WITH target …`);
  the WHERE clause is unchanged. `test_neighborhood_falkor_live.py` EXPLAINs
  every query a real read issues and fails on any label scan.

Measured on throwaway FalkorDB copies of production (read-only DUMP, 5,051
nodes, 38,393 edges, `node_id` index present), 20 runs each:

| Request | 4.18.11 p95 before → after | 6.0.1 p95 before → after |
|---|---|---|
| circe entity, 4/8/8 | 249.6 → 26.5 ms | 391.6 → 25.2 ms |
| GPU concept, 12/16/16 | 430.8 → 53.7 ms | 330.7 → 71.9 ms |
| #2497's 8 hub nodes, 12/16/16 | 1,077.4 → 104.5 ms | 1,326.0 → 127.8 ms |

Equivalence: 305 requests (5 named + 300 random focal sets, budgets, states,
directions) gave byte-identical receipts before and after on both engines.

## Evidence handles: not in this patch

A bounded `read_evidence_handles` read (which source items back a node, as of
a time) was built and measured for this patch, then cut under the rule that no
concept lands without a runtime consumer. It lands in memory Stage 2 PR F
together with recall-by-referent, its first consumer. Concept table:
`orion/substrate/README.md`.
