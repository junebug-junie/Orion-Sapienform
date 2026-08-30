# Engine-side traversal, communities, and an honest canvas

## Summary

- **Finishes the graph-analytics workstream.** Three of `orion/graph/analytics.py`'s capabilities had **zero callers** — `communities()`, `neighborhood()`, `path()`. By this repo's own rule that is a keyword cathedral: a producer with no consumer.
- Adds `GET /api/substrate/concepts/neighborhood` and `/path`, which query the **engine over the whole graph** rather than filtering a fetched slice.
- Both resolve a label to a node and **refuse to guess** when it is ambiguous.
- `communities()` joins `/structure` — now that there is something for it to find.
- The canvas finally says **how much of the graph it is showing**.

## Outcome moved

| | before | after |
|---|---|---|
| analytics capabilities reachable from Hub | 3 of 6 | **6 of 6** |
| neighbour query scope | the fetched 600-edge window | the whole graph |
| ambiguous name | silently resolved to something | candidates returned, nothing guessed |
| canvas coverage | unstated | `showing 102 of 671 nodes (15%) and 600 of 1464 edges` |

## Current architecture

`/network` fetches `query_concept_region(limit_nodes=300, limit_edges=600)`, then hydrates edge-reachable non-concept nodes up to `_NETWORK_HYDRATION_MAX_EXTRA_NODES = 100`. Its `focus` parameter filters **that already-truncated slice**, so it answers *"neighbours among the ones we happened to fetch"*.

That distinction was academic when I built the analytics: the graph was 136 nodes and fit inside every cap. It is not academic now.

## The graph grew 5× under us

Measured live 2026-08-30, after the entity fix (#1968) landed:

```
Entity    464      <- was impossible to persist
Evidence  135
Concept    72
            671 nodes / 1464 edges
```

The binding cap is the **edge** one, confirmed against the deployed Hub: `/network` returns exactly 600 edges and `truncated: true`. All 72 concepts fit; `limit_nodes=300` never binds. The 464 entities hang off `associated_with` edges outside that 600-edge window, so **the canvas renders 102 of 671 nodes and zero entities** — and the payload has carried `truncated: true` the entire time with nothing rendering it.

## Communities: my earlier conclusion is overtaken by the data

I previously reported the graph had *no community structure to find*, and left `communities()` unwired for exactly that reason — a reader for a thing with nothing to read. That was true at 136 nodes / 307 edges, where label propagation returned exactly **one** community for every edge-type restriction tried.

At 671 nodes / 1464 edges it finds real structure, and the most useful part is not thematic grouping but **duplicate detection**:

```
communities(rel=associated_with)
     2  ['Rest and support', 'Rest and recovery']
     2  ['AI Model Transparency', 'AI Model Configuration']
     2  ['Hospital and family chaos', 'Hospital and medical concerns']
     3  ['Conceptual Pressure and Ontology', "Athena's unresolved signal", ...]
```

Those are near-duplicate concepts the induction pipeline minted separately. The UI highlights small communities for that reason rather than the 645-member blob.

## Ambiguity is returned, never guessed

`resolve_node()` matches exact id, then exact label, then prefix — and returns **candidates**, because labels are not unique here:

```
resolve_node("Hospital") -> Hospital and family chaos
                            Hospital and fear experience
                            Hospital and medical concerns
```

A silent first-match would answer a traversal question about a node the caller did not mean. The routes return `node_ambiguous` / `endpoint_not_resolved` with the candidate list, and never traverse before the node is pinned down (asserted by test).

## Cost, measured on the real graph

```
neighborhood  depth 1   6ms    depth 2  15ms    depth 3  48ms
path          depth 2   2ms    depth 3  25ms    depth 4 384ms
```

`path` at depth 4 was **143ms when the graph was 136 nodes** and is 384ms at 671 — the same query, 5× the graph. The default drops to depth 3; `MAX_PATH_DEPTH` is the ceiling, not the recommendation, and the module docstring now says to re-measure before raising it.

## Files changed

- `orion/graph/analytics.py`: `resolve_node()`; measured-cost docstring.
- `services/orion-hub/scripts/concept_atlas_routes.py`: two traversal routes, communities in `/structure`, hydration-truncation fields.
- `services/orion-hub/templates/concept_atlas.html`: drill-down card, communities column, coverage line.
- `services/orion-hub/static/js/concept-atlas.js`: `coverageLine`/`candidateHint`/`formatPath`, drill-down wiring, click-to-drill.
- 3 test files touched/added.

## Schema / bus / API changes

- **Added:** two GET routes; `communities`, `hydration_truncated`, `hydrated_count`, `hydration_limit` on existing payloads. All additive.
- **Behavior changed:** none for existing fields.

## Env/config changes

None — no keys added, removed or renamed; `.env_example` untouched, so no sync required.

## Tests run

```text
orion/graph/tests + substrate falkor suites              191 passed
services/orion-hub concept-atlas suites (5 files)        129 passed
services/orion-hub/static/js (node --test)                98 passed
                                                   418 passed, 0 failed
```

Mutation-tested, every mutation asserted present in the file before running:

```text
16/16 killed
  routes  ambiguity silently resolved · traverses before resolving ·
          truncation never reported · path `found` always true ·
          communities dropped · dead procedure blanks the card ·
          rel_types no longer parsed
  analytics  exact-id priority collapsed · prefix query always runs ·
             blank needle hits the graph · rows with no id kept
  js      coverage silent when truncated · coverage warns on every render ·
          empty path claims disconnection · candidates hidden
```

One survived the first sweep — `resolve_node` had no analytics tests at all, only route-level coverage through a stub. Eight added; all four of its mutations then died.

## Evals run

```text
No eval harness covers these routes; none added.
```

Not claimed as covered. The live measurements above are what stands in for one.

## Docker/build/smoke checks

Live against the deployed FalkorDB via a TestClient over the real router:

```text
neighborhood(Orion,d=1): available=True  56 nodes  truncated=False
neighborhood(Hospital):  node_ambiguous  ['Hospital and family chaos',
                          'Hospital and fear experience',
                          'Hospital and medical concerns']
neighborhood(nope):      node_not_found
path Orion->Juniper:     found  ['Orion','Sync Issue Resolution','Juniper']
path Orion->Hospital:    endpoint_not_resolved
rel_types="x) RETURN 1 //":  invalid_rel_types  (refused)
structure communities:   [(645,[Orion,Juniper]), (10,[Burst Test...]),
                          (3,[AI model updates...]),
                          (2,[Hospital and family chaos, Hospital and medical concerns])]
degraded_measures: []
```

## Review findings fixed

`/code-review high` ran in a subagent; see the follow-up commit on this branch.

## Restart required

```bash
sudo docker compose \
  --env-file .env \
  --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml \
  up -d --build orion-hub
```

Static assets are baked into the image, so a plain restart is not enough.

## Risks / concerns

- **Severity: medium, pre-existing, NOT fixed here.** The canvas still renders 15% of the graph. This patch makes that *visible* and gives an unbounded way to explore around it; it does not raise the caps. Raising `limit_edges` is a rendering decision (cytoscape at 1464 edges) that deserves its own measurement, not a number bumped in passing.
- **Severity: low.** `path` is the one expensive call and it degrades with graph growth. Bounded, defaulted to 3, and never called from a render path — but re-measure before raising the cap.
- **Severity: low.** `communities()` on the whole graph is dominated by a 645-member blob. The useful signal is in the small communities, which is why the UI highlights those; a reader who only looks at the top row learns nothing.

## Follow-ups

1. **The duplicate concepts are a real finding, not just a demo.** `Rest and support` / `Rest and recovery` and friends are the induction pipeline minting near-identical concepts. Worth a dedup pass keyed on community membership.
2. Raise or window the 600-edge cap with a rendering measurement behind it.
3. `orion_worldview` still has 48 nodes and zero edges (from the earlier PR) — the traversal routes work against it, they just have nothing to traverse.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/atlas-graph-traversal
