# Shared FalkorDB analytics seam + Concept Atlas structure read

## Summary

- FalkorDB ships graph algorithms as stored procedures. `rg "algo\."` across the repo returned **zero hits** — pageRank, WCC, betweenness, label propagation and friends have been sitting unused in the engine while `concept_atlas_routes.py` hand-rolled union-find and a degree loop in Python, per request, inside a 1732-line module no other service can import.
- Adds `orion/graph/analytics.py`: `components()`, `rank()`, `communities()`, `neighborhood()`, `path()`, `summary()`. Graph-agnostic — it takes a client plus configurable id/label properties, so `orion_substrate`, `orion_worldview`, `orion_bus_synapse` and `orion_recall` are one constructor argument apart.
- Adds `read_only=True` to `RedisGraphQueryClient`, routing to `GRAPH.RO_QUERY`.
- Adds Hub `GET /api/substrate/concepts/structure` — the whole-graph read, unfiltered, including **bridges**: nodes that top betweenness but not pageRank.
- Atlas UI: folds evidence nodes into a count on the concept they support (**136 nodes → 56** on the live graph) and adds a Graph structure card.

## Outcome moved

The Atlas could already say "136 nodes, 461 edges, 12 components" and none of it told Juniper what was in there. Now measurable and visible:

| | before | after |
|---|---|---|
| default node count on the canvas | 136 | 56 |
| algorithms callable by any service | 0 | 5 |
| graphs the analytics work on | — | 4, verified |
| whole-graph structural read | none | one route, one card |

## Current architecture

`concept_atlas_routes.py::concept_atlas_network` fetched a 300-node/600-edge slice, then computed activation-weighted degree and connected components in Python over the **already-filtered** node/edge lists. `_compute_connected_components` is a private union-find in a module nothing else imports. No other graph in the deployment (`orion_worldview`, `orion_bus_synapse`, `orion_recall`) had any structural read at all.

## Architecture touched

- New shared module under `orion/graph/`, the home the FalkorDB property-graph doctrine already names for adapters. No substrate vocabulary in it.
- Hub gains one read-only route. No writer, no new env key, no schema or bus contract change.
- `RedisGraphQueryClient` gains an opt-in read-only mode; default `False`, so every existing writer is unchanged.

**Deliberately NOT changed:** `_compute_connected_components` stays. It answers "what is connected in the view you are looking at" over the post-filter lists; `algo.WCC` answers "what shape is the whole graph in". Swapping one for the other would have reported components whose members the caller cannot see. Both are real; they are not duplicates.

## Files changed

- `orion/graph/analytics.py`: new. The seam.
- `orion/graph/falkor_client.py`: `read_only` mode via redis-py's own `read_only=True` keyword.
- `orion/graph/tests/test_analytics.py`: new, 45 tests.
- `services/orion-hub/scripts/concept_atlas_routes.py`: `/structure` route + `_build_graph_analytics`.
- `services/orion-hub/tests/test_concept_atlas_structure_route.py`: new, 11 tests.
- `services/orion-hub/static/js/concept-atlas.js`: evidence collapse, structure card, diagnosis helpers.
- `services/orion-hub/static/js/concept-atlas.test.js`: +21 tests (61 → 82).
- `services/orion-hub/templates/concept_atlas.html`: structure card + "Fold in evidence" toggle.

## Runtime truth: what the engine actually does

`MODULE LIST` on the live instance reports `graph` v4.18.11. Being listed by `dbms.procedures()` is **not** evidence a procedure runs:

| procedure | result | wrapped? |
|---|---|---|
| `algo.WCC` | 9.4ms | yes |
| `algo.pageRank` | 4.6ms | yes |
| `algo.betweenness` | 4.5ms | yes |
| `algo.HarmonicCentrality` | 5.1ms | yes |
| `algo.labelPropagation` | 18.6ms | yes |
| `algo.BFS` | **zero rows**, every argument form tried | no |
| `algo.SPpaths` | **zero rows** | no |
| `algo.SSpaths` | **zero rows** | no |
| `shortestPath()` | "does not currently support undirected shortestPath traversals" | no |

So `neighborhood()` and `path()` are plain variable-length Cypher, not procedure wrappers — a deliberate downgrade to the primitive that demonstrably returns rows.

**The arities are inconsistent and getting one wrong is not always loud.** `algo.pageRank()` raises `requires 2 arguments, got 0`; `algo.betweenness(null, null)` returns a header and **zero rows, no error**. The same mistake raises on one procedure and silently returns an empty ranking on the next. Each call form is pinned as a whole string in `MEASURES` and asserted verbatim in the tests.

## Extensibility, verified rather than asserted

Same object, one constructor argument apart:

```
orion_substrate      nodes= 136 edges= 461 comps=  12 largest= 116 singles= 10
orion_worldview      nodes=  48 edges=   0 comps=  48 largest=   1 singles= 48
orion_bus_synapse    nodes= 313 edges= 645 comps=   1 largest= 313 singles=  0
orion_recall         nodes=3864 edges=5036 comps= 131 largest=2455 singles=  0
```

Timing on the largest (3864 nodes): summary 36ms, pagerank 4.3ms, betweenness 4.3ms, communities 23ms.

## Findings this surfaced

**1. The Atlas is a hairball because its dominant edge carries almost no information.** 307 of 461 edges (67%) are `co_occurs_with`, a same-day co-occurrence proxy. Over 56 concepts that links **19.9% of every possible pair**. `algo.labelPropagation` returns exactly one community spanning the whole connected graph — and still one community when restricted to `associated_with`. No centrality measure or layout change fixes this; it is an upstream producer problem. The structure card states it with the numbers inline.

**2. Orion and Juniper are ubiquitous, not load-bearing.** pageRank #1 and #2; **absent from the betweenness top 8**. The graph's actual bridges are `Light folding concept`, `messy middle authenticity`, `Home lab infrastructure` — all rendered as ordinary dots before this. The existing comment at `concept_atlas_routes.py:1205` suspected degree was ranking vocabulary ubiquity; betweenness measures it. *Honest caveat: betweenness partially correlates with degree here — `Light folding concept` is #1 in both. It is not clean independent signal on a 136-node single-blob graph; it disagrees precisely where it matters.*

**3. `orion_worldview` has 48 nodes and zero relationships.** `db.relationshipTypes()` is empty. `worldview.py:459`'s `run_edges_cypher` counts `SUPPORTS`/`CONTRADICTS` by type, and its own docstring says it exists *because* "the footprint could not see an edge at all" — Orion has been instructed to write edges and has written none, ever. **Not fixed here** (producer-side, touches Orion's live turn instructions). Filed as the follow-up below.

**4. 59% of the Atlas canvas was scaffolding.** 80 of 136 nodes are Evidence nodes with no `label` field; the route names each one "Evidence for \<concept\>". Folding them into a count is the single biggest legibility win and needs no algorithms.

**5. The 10 singleton components are all retired telemetry** — eight `*_prediction_error` nodes plus `node:substrate.transport` and `node:substrate.harness_closure`. `substrate.transport` is a metric CLAUDE.md records as **retired outright** yet still holding a node.

## Metric quality gate

Applied to the three new numbers, per CLAUDE.md §0A.

**pageRank / betweenness / harmonic**
1. *Provenance:* FalkorDB stored procedures, `orion/graph/analytics.py::rank`. Not derived by us.
2. *Independence:* **partially fails.** Betweenness correlates with degree on this graph (`Light folding concept` tops both). Recorded rather than hidden; the value is that it disagrees on the god nodes, not that it is orthogonal.
3. *Theory anchor:* standard named centralities, not invented.
4. *Live sanity:* non-degenerate, non-zero, distinct orderings on all four graphs. Not merely "it varies" — the three measures produce *different* top-N sets, which is the check that they are not one metric wearing three names.
5. *Existing mechanism:* none — zero `algo.` calls repo-wide before this.
6. *Reversibility:* cheap. Read-only, no schema, no manifest, no stored value.

**saturation** (edges ÷ possible pairs) is arithmetic over two counts, hand-computed in the tests (307/1540 = 0.1994). Denominator is the **concept** count, not the node count: a concept-concept edge cannot land on an Evidence node, and using 136 would report 0.05 — a sparse graph where the real one is 20% saturated.

## Tests run

```text
orion/graph/tests                                    89 passed
services/orion-hub/tests/test_concept_atlas_routes.py            33 passed
services/orion-hub/tests/test_concept_atlas_structure_route.py   11 passed
services/orion-hub/static/js  (node --test)          82 passed, 0 failed
```

Mutation-tested, every mutation asserted present in the file before running so a no-op replacement cannot read as a pass:

```text
python  6/6 killed   wrong betweenness arity · a write in a query ·
                     dropped id/label validation · hardcoded node.label ·
                     no depth clamp · no rel-type validation
route   5/5 killed   the original path collision · duplicate handler name ·
                     bridges stop excluding pagerank · flattering denominator ·
                     route 500s instead of degrading
js      12/12 killed evidence counting · in-place mutation · dropped count ·
                     label replacement · kept edges · diagnosis cutoffs ·
                     component split · plural · NaN guard
```

One JS mutation initially **survived** (`edges > 0` guard). Investigated rather than patched over: the branch is unreachable from the real route, since `edge_count` is the sum of `edge_type_counts`. Kept as defence against a stale cached bundle and covered by a test feeding exactly that inconsistent payload — without the guard the card renders "NaN%".

## Evals run

```text
services/orion-hub has no evals/ directory; none added.
orion/graph has no eval harness.
```

Not claimed as covered. These are deterministic structural reads over a live graph; the live-verification block below is what stands in for an eval here, and a real eval harness for `orion/graph/` is a reasonable follow-up if this seam grows consumers.

## Docker/build/smoke checks

```text
No Docker build run. The patch adds no dependency, port, healthcheck or
compose wiring; it is one new pure module, one read-only route, and static
assets. It DOES require a Hub restart to take effect (route + static bundle).
```

Live verification against the real FalkorDB, via the route function itself:

```text
available=True graph=substrate
nodes=136 concepts=56 edges=461
components=12 largest=116 singletons=10
dominant=co_occurs_with saturation=0.1994
edge types: {'co_occurs_with': 307, 'supports': 80, 'associated_with': 74}

BRIDGES (high betweenness, not top-pagerank):
    63.70  Home lab infrastructure
    58.78  Light folding concept
    56.07  messy middle authenticity

pagerank top3: Orion 0.0993 · Juniper 0.0964 · Code Review Process 0.0339

singletons: Chat / Execution / Bus synaptic / Codebase / Perception / Route /
            Biometrics / Vision prediction error,
            substrate:node:substrate.transport,
            substrate:node:substrate.harness_closure
```

Read-only enforcement, live: `CREATE (:Tmp)` through the read-only client →
`graph.RO_QUERY is to be executed only on read-only queries`.

## Bugs caught while building, in this patch

- **Route path collision.** `/api/substrate/concepts/summary` already existed, and my handler shared the name `concept_atlas_summary`. FastAPI does **not** raise on a duplicate path — it silently serves whichever registered first, so the new route was dead on arrival. Renamed to `/structure`; added two gates asserting no duplicate path+method and no duplicate handler name in the router.
- **Silent list corruption.** Hand-issuing `GRAPH.RO_QUERY` without `--compact` returns a `collect()` column as its *string repr*. It does not fail — it yields single characters when iterated, so `path()` returned `['O','r','i','o','n',...]`. Fixed by using redis-py's own `read_only=True` keyword so both modes share one parser.
- **`ro_query` does not exist** in redis-py 5.0.1. Caught by running it, not by reading docs.
- **Wrong aitown default.** I wrote `orion_aitown_substrate`; the real default is `orion_substrate_aitown`. A test now pins it against the store builder's own source.
- **`os.getenv` vs pydantic-settings.** The route returned `available=false` under any in-process caller. Switched to `settings.FALKORDB_URI`.
- **Substrate-shaped assumptions.** `node.label`/`node.node_id` hardcoded would have returned `label=None` for all 3864 `orion_recall` nodes while looking like it worked.

## Review findings fixed

Review ran in a subagent (`/code-review high`). See the follow-up commit on this branch.

## Restart required

```bash
sudo docker compose \
  --env-file .env \
  --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml \
  up -d --build orion-hub
```

Static assets are served from the image, so a plain restart is not enough — the bundle must be rebuilt.

## Risks / concerns

- **Severity: low.** `path()` is undirected variable-length enumeration — 143ms at depth 4 on 136 nodes, and it grows badly. Capped at `MAX_PATH_DEPTH` and not called from any render path. It is an operator query.
- **Severity: low.** `communities()` returning one community is a real answer, not a failure, and is documented as such. A reader who takes it as a bug will chase nothing.
- **Severity: informational.** Betweenness is partially correlated with degree here. Do not build a ranking product on it without re-checking independence on a graph with real semantic edges.

## Follow-ups (not in this patch)

1. **`orion_worldview` has never had an edge written.** Producer-side; touches Orion's live turn instructions and so wants proposal mode per CLAUDE.md §0A.
2. **No semantic-relatedness edge exists.** Every current edge type is a co-occurrence mesh, a 2-node star, or pendant evidence. topic-foundry already computes embeddings; a centroid-similarity edge is the missing ingredient that would give the graph structure worth running community detection on.
3. **Retire `node:substrate.transport`'s node.** CLAUDE.md records the metric as retired outright; the node is still in the graph.
4. `orion/curiosity/worldview.py` has a private `rows_from_reply` equivalent to `falkor_client._rows_from_query_result`. Consolidating is right but touches Orion's live loop, so it was left alone.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/graph-analytics
