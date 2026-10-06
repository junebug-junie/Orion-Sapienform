## Summary

Orion's memory step that writes claims and referents into the shared graph waits until every service that reads that graph has said "I can read the new shapes". After the 2026-10-06 deploy only 3 of 9 services had said so, so nothing was ever written. This patch makes every graph-reading service say so the moment it starts, and keep saying so.

- New `advertise_at_startup()` in `orion/substrate/reader_capability.py`: a background thread that writes the "I can read it" key right away and every 10 minutes. Never blocks boot, never raises.
- The key now expires after 30 minutes. A reader that dies, or is rolled back to older code without the refresher, drops out of the gate by itself instead of holding a stale "ready" forever.
- Wired into the startup of orion-substrate-runtime, hub, recall, cortex-exec (all four containers: main, chat, spark, background run the same entrypoint), cortex-orch and spark-concept-induction.
- Removed orion-field-digester, orion-world-pulse and meta-tags from the required list: none of them ever opens the `orion_substrate` graph, so new node kinds there cannot hurt them, and they could never have advertised.
- Fixed the code default `DEFAULT_REQUIRED_READERS`, which still had the wrong `orion-hub`-style names #2522 fixed only in env/compose.

## Outcome moved

The referent/assertion projector readiness gate (in orion-memory-consolidation) can open after a rebuild of the six readers, instead of waiting forever for services that build their graph store lazily or never.

## Current architecture

A reader advertised only when it first constructed a Falkor substrate store (`falkor_store.bootstrap_substrate_reader`) or in recall's background bootstrap. Many readers build that store on first request, and cortex-exec's main store path (`falkor_anchor_store.build_unification_store_from_env`) passes its own client, so it skipped the advertisement entirely. Keys had no expiry.

## Architecture touched

- `orion/substrate/reader_capability.py`: `advertise(..., ttl_s=1800)` (SET with EX), new `advertise_at_startup()`, `ADVERTISE_INTERVAL_S=600`, `CAPABILITY_TTL_S=1800`, corrected `DEFAULT_REQUIRED_READERS`.
- Startup of six services (FastAPI lifespan / `on_event("startup")` / async `main()`).
- `SUBSTRATE_ASSERTION_REQUIRED_READERS` list in memory-consolidation `.env_example` + compose default.
- Gate logic (`readiness`) unchanged: it reads with GET, and an expired key reads as missing, which is the intended behavior.

## Files changed

- `orion/substrate/reader_capability.py`: TTL + startup refresher + default list fix.
- `services/orion-substrate-runtime/app/main.py`, `services/orion-hub/scripts/main.py`, `services/orion-recall/app/main.py`, `services/orion-cortex-exec/app/main.py`, `services/orion-cortex-orch/app/main.py`, `services/orion-spark-concept-induction/app/main.py`: call `advertise_at_startup()` at boot.
- `services/orion-memory-consolidation/.env_example`, `docker-compose.yml`, `README.md`: shorter required list, explanation.
- `orion/substrate/README.md`: how and when readers advertise; rollback now self-heals.
- `orion/substrate/tests/test_reader_capability_startup.py` (new), `test_reader_capability.py`, `test_required_reader_names_match_services.py`: tests below.

## Why three services were removed (evidence)

- **orion-field-digester**: no `falkor`, `FALKORDB_*`, `SUBSTRATE_STORE_BACKEND` or substrate store construction anywhere in `services/orion-field-digester` (code, `.env_example`, compose). Its `orion.substrate` imports are pure helpers (`causal_geometry_producer`, `field_topology_*`). Live container env has no `FALKORDB_URI`.
- **orion-world-pulse**: only substrate import is `bus_synaptic_surprise.latest_bus_synaptic_prediction_error`, which takes a SQL `Engine`. No Falkor config in code, env or live container.
- **meta-tags**: talks to Falkor, but only to `FALKORDB_RECALL_GRAPH=orion_recall` (its own tag graph, `app/main.py` `_get_falkor_client`). Never touches `orion_substrate`, where assertion/referent nodes live.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: the reader capability key now has a 30 min TTL and is refreshed every 10 min.
- Compatibility notes: the three existing live keys (no TTL) get a TTL on the next refresh after the rebuild.

## Env/config changes

- Added/removed/renamed keys: none.
- Changed value: `SUBSTRATE_ASSERTION_REQUIRED_READERS` now `orion-substrate-runtime,hub,recall,cortex-exec,cortex-orch,spark-concept-induction`.
- `.env_example` updated: yes. Compose default updated: yes.
- local `.env`: the key already existed, so the sync script does not overwrite it; edited by hand in `/mnt/scripts/Orion-Sapienform/services/orion-memory-consolidation/.env`. `sync_local_env_from_example.py --all-keys orion-memory-consolidation` run afterwards (no change; reports one unrelated pre-existing divergence, `CONCEPT_RELATION_RESOLUTION_ENABLED`).

## Tests run

```text
pytest orion/substrate/tests/test_reader_capability.py test_reader_capability_startup.py test_required_reader_names_match_services.py -> 16 passed
  - every required reader has an advertise_at_startup() call in its service (mutation-checked: removing cortex-orch's call fails it)
  - code default list == .env_example list == compose default
  - TTL: key readable until expiry, gate closes after
  - refresher renews the key; after stop, key expires and the gate closes
  - startup returns in <0.1s with a hung/down Redis; no URI = no-op; idempotent; internal errors swallowed
py_compile of all six edited entry modules -> ok
```

## Evals run

```text
None: this is a liveness/plumbing change, covered by the gate tests above. No eval harness change.
```

## Docker/build/smoke checks

```text
Static gates from .github/workflows/orion-static-gates.yml: all pytest gate files 366 passed / 2 skipped;
check_definition_drift --gate, check_metric_lineage --gate and every other check_* script exit 0.
Not deployed (per instruction). Live behavior UNVERIFIED until rebuild: expect 6 keys under
orion:substrate:reader_capability:* with TTL ~1800, and memory-consolidation /health referent_projector.state=running.
```

## Review findings fixed

- Review subagent not run, per instruction for this task.

## Restart required

```bash
for s in orion-substrate-runtime orion-hub orion-recall orion-cortex-exec orion-cortex-orch orion-spark-concept-induction orion-memory-consolidation; do scripts/safe_docker_build.sh $s up -d --build; done
```

Then check: `docker exec orion-athena-falkordb redis-cli --scan --pattern 'orion:substrate:reader_capability:*'`

## Risks / concerns

- Severity: low. Concern: the four cortex-exec containers share one name (`cortex-exec`), so one key covers all four; if only one of them were rolled back the others would keep the key alive. Mitigation: all four are built from the same image and deployed together.
- Severity: low. Concern: once the gate opens, assertion/referent writes start for the first time. Mitigation: that is the intended rollout; readers built from this code skip unknown shapes.
- Severity: low. Concern: if FalkorDB is unreachable for >30 min the gate closes and writes pause. Mitigation: that is the safe direction; it reopens on the next refresh.

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
