## Summary

- The substrate graph reload now survives a bad FalkorDB cached query plan. When every row of a page fails to decode, the store re-reads that page once with a fixed variant of the query text (one trailing space), which misses the plan cache. It logs `falkor_hydrate_plan_cache_suspect` with the shifted field and a sample id. If the variant fails too, the reload fails exactly as before.
- After every reload (good or failed) the store counts the durable graph's nodes and compares with its in-memory node count. The gap rides on `CompleteScanReceipt` (`durable_node_count`, `node_count_gap`, `plan_cache_retries`) and the 5-minute `falkor_substrate_hydrate_stats` line. A gap above max(50, 5%) logs `falkor_substrate_hydrate_node_gap` on every reload until it closes.
- Refs #3455 (stays open until a live recovery or a clean live week is seen).

## Outcome moved

Failure mode from #3455: a stale plan made 15,568 of 15,568 node rows undecodable for ~28 h, every reload failed, and readers (attention broadcast, curiosity tick) ran on a 40-node graph while the durable graph held 15,355. Nothing complained. Now the reload self-heals from that exact shape, and a partial in-memory graph is loud.

## Current architecture

`FalkorSubstrateStore._hydrate_from_durable` pages four scans through `_scan_pages` (native nodes, legacy nodes, native edges, legacy edges) and decodes each row in the caller. Any decode error failed the whole reload; the last good cache stayed. No surface compared in-memory coverage with the durable graph.

## Architecture touched

- `orion/substrate/falkor_store.py`: `_scan_pages` takes a per-row `decode` and yields `(row, decoded)`; whole-page-failure retry with `_PLAN_CACHE_BYPASS_SUFFIX`; `_record_node_count_gap`; new counters; stats line fields.
- `orion/substrate/store.py`: three additive defaulted fields on `CompleteScanReceipt` (in-process dataclass, not a bus schema).
- `orion/graph/falkor_client.py`: `RecordingFalkorClient` answers the node-count query.

## Files changed

- `orion/substrate/falkor_store.py`: retry guard and gap signal.
- `orion/substrate/store.py`: receipt fields.
- `orion/graph/falkor_client.py`: test double answers `count(n)`.
- `orion/substrate/tests/test_falkor_hydrate_plan_cache_guard.py`: new tests.
- `orion/substrate/tests/test_falkor_store.py`, `test_complete_hydration.py`: account for the one extra count query per reload.
- `docs/superpowers/pr-reports/2026-11-17-falkor-hydrate-plan-cache-guard-pr.md`: this report.

## Schema / bus / API changes

- Added: `CompleteScanReceipt.durable_node_count`, `.node_count_gap`, `.plan_cache_retries` (defaults keep every constructor working).
- Removed / Renamed: none.
- Behavior changed: one extra read query (`MATCH (n:SubstrateNode) RETURN count(n)`) per reload; at most one extra page read per scan when a whole page is undecodable.
- Compatibility notes: no bus, channel, or registry change.

## Env/config changes

- None. Thresholds are class constants. `.env_example` untouched; no sync needed.

## Metric quality gate (node_count_gap)

1. Provenance: `durable` = `MATCH (n:SubstrateNode) RETURN count(n)` read in `_record_node_count_gap`; `cache` = `len(self._cache._nodes)` under `_cache_lock`, right after the reload swap (or the kept old cache on failure). Gap = durable - cache.
2. Independence: `last_hydrate_ok` / `failed_total` say a reload failed, not how much of the graph readers lack. The gap is the only reading of coverage. It is correlated with failures by design (a failed reload is the main cause), but it also catches the never-hydrated process and silent drops (skipped unknown kinds), which the failure counter cannot.
3. Theory anchor: coverage by definition. Readers read the in-memory cache; the durable graph is the truth it mirrors. durable - cache is the count of nodes readers cannot see.
4. Live data (read-only replay against `orion-athena-falkordb`, 2026-11-17, this branch's code, `read_only=True` client): three full reloads read `cache_nodes=15783 durable=15783 gap=0 retries=0` (35-46 s each). An unhydrated store read `gap=15783` and logged the alarm. So the rest state is a true 0, not a floor, and the starved state (the #3455 shape) reads the full graph size. Not a decayed-to-zero artifact: 0 comes from two independent counts agreeing, recomputed every reload. In the live runtime a write landing between swap and count can make the gap a few nodes; the 50-node floor absorbs that.
5. Existing mechanism: `CompleteScanReceipt.node_count` is the scanned count, not a comparison; `last_hydrate_node_count` same. No health endpoint exposes hydrate stats (grep of `services/*/app`). Extended the receipt and the stats line rather than adding a new surface.
6. Reversibility: cheap. Defaulted dataclass fields, one log line, one query. No schema registry, manifest, or training default.

## Tests run

```text
pytest orion/substrate/tests/test_falkor_hydrate_plan_cache_guard.py  -> 7 passed
pytest orion/substrate/tests/test_complete_hydration.py test_falkor_store.py test_falkor_store_hydrate_signal.py test_falkor_identity_dedupe.py -> 109 passed
```

New tests: shifted cached plan + good variant -> reload succeeds, one suspect log naming `decay_floor`, exactly two distinct query texts; both bad -> reload fails, old cache kept; partly-bad page -> no retry; clean scan never sends the variant; gap 0 on a full reload; gap 200 + alarm when a failed reload leaves a partial cache; count error -> gap None, reload still ok.

## Evals run

```text
Existing orion/substrate/evals/run_complete_hydration_eval.py covers the scan contract; not re-run separately (its fixture is exercised by test_complete_hydration.py).
```

## Docker/build/smoke checks

```text
Read-only live replay (see gate item 4). No image build; runtime picks this up on the normal post-pull rebuild.
```

## Review findings fixed

REVIEW_PLACEHOLDER

## Restart required

```text
No manual restart. The post-pull rebuild of services importing orion/substrate (orion-substrate-runtime, hub, recall) picks it up.
```

## Risks / concerns

- Severity: low. Concern: the variant's own plan could go bad too. Mitigation: then the reload fails as today and the gap alarm fires; a FalkorDB restart is still the operator fix.
- Severity: low. Concern: each reload during a bad-plan episode reads one wasted page and logs one suspect line. Mitigation: deliberate, keeps the fault visible; plan cache gets at most four extra entries (one per scan).
- Issues: refs #3455 (kept open; close with a live `ok_total` rising / gap 0 comment, or a live suspect-then-recovered line). No new issues opened.

## Upstream FalkorDB bug report (draft, not filed)

> **Cached execution plan returns a stale/shifted projection column.** A read query `MATCH (n:SubstrateNode) WHERE n.payload_json IS NULL AND id(n) > $after_id RETURN n.node_id AS node_id, ... (49 projected properties) ..., id(n) AS object_id ORDER BY object_id LIMIT $page_size`, issued repeatedly with different parameters, began returning the value of `n.node_id` in the `decay_floor` column for every row (`Cached execution: 1`). The stored property was correct (`RETURN n.decay_floor` alone -> `0`; `WHERE n.decay_floor = n.node_id` -> 0 rows). The identical query with one extra space (`Cached execution: 0`) returned correct values. A server restart cleared it. Server uptime was 9.2 days; trigger unknown. Graph ~15.5k nodes, ~64k edges; concurrent writers active.

## PR link

PR_LINK_PLACEHOLDER
