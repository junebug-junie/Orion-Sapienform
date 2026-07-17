# fix: gate substrate dynamics tick writes on real change, not every tick

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1098
Branch: `fix/substrate-dynamics-unconditional-upsert`

## Summary

- `SubstrateDynamicsEngine.tick()` was calling `store.upsert_node()` unconditionally for every node on every 30s tick, regardless of whether anything changed.
- On the SPARQL/Fuseki backend that's a full DELETE+INSERT transaction per node per tick. Traced live in production on 2026-07-16 to ~2.5 SPARQL updates/sec sustained (78 substrate nodes / 30s tick), a material contributor to `orion-rdf-store` running out of TDB2 compaction headroom (real incident: disk filled to the point `make compact` could no longer run).
- Now only writes a node when its activation crosses the existing `1e-6` change threshold, its dormancy state flips, or its pressure changed.
- Code review (8-angle) surfaced a real bug the guard exposed: the dormancy check read the stale, top-of-tick stored `recency_score` instead of the freshly computed value. The unconditional write had been masking this — once persistence stops for a quiescent node, the stored value never refreshes and the node could never transition to dormant via pure recency decay. Fixed in the same patch by having the dormancy check read fresh recency directly.
- Enables the retention scheduler in `orion-rdf-writer` (already built, previously `RDF_RETENTION_ENABLED=false` with no policies) as an ongoing safety net against the unbounded `autonomy/{identity,goals}` accumulation found during the same incident.

## Outcome moved

Eliminates the dominant *active* source of TDB2 write-amplification found during the 2026-07-16 Fuseki disk-exhaustion incident. (The other 98% of that incident's disk usage was historical `autonomy/{drives,identity,goals}` accumulation via now-disabled/removed write paths — a one-off SPARQL cleanup, not a code change, tracked separately in the incident's own operational log.)

## Current architecture

`orion-substrate-runtime`'s `_dynamics_tick_loop` calls `SubstrateDynamicsEngine.tick()` every `SUBSTRATE_DYNAMICS_TICK_INTERVAL_SEC` (default 30s) when `SUBSTRATE_DYNAMICS_TICK_ENABLED=true`. `tick()` recomputes pressure and activation for every node in the store snapshot and previously persisted all of them back unconditionally.

## Architecture touched

`orion/substrate/dynamics.py` (the tick loop's write gate), `orion/substrate/tests/test_dynamics.py` (new), `services/orion-substrate-runtime/README.md` (docs), `services/orion-rdf-writer/.env_example` (retention config).

## Files changed

- `orion/substrate/dynamics.py`: gate `upsert_node()` on real change; fix dormancy check to use fresh recency instead of stale stored value; move `model_copy` construction inside the write-guard branch (avoids building objects that end up discarded).
- `orion/substrate/tests/test_dynamics.py`: new — 4 tests covering the no-op case, activation-changed, pressure-changed, and the recency/dormancy regression.
- `services/orion-substrate-runtime/README.md`: documents the write guard and the recency/dormancy gotcha for future maintainers.
- `services/orion-rdf-writer/.env_example`: `RDF_RETENTION_ENABLED=true`, `RDF_RETENTION_INTERVAL_HOURS` 168→24, adds `RDF_RETENTION_POLICIES` for `autonomy/{identity,goals,drives}` with context comment.

## Schema / bus / API changes

None. `SubstrateDynamicsResultV1`'s shape is unchanged; this only changes when a node gets persisted, not what gets returned or logged.

## Env/config changes

- Changed: `RDF_RETENTION_ENABLED` (false→true), `RDF_RETENTION_INTERVAL_HOURS` (168→24) in `services/orion-rdf-writer/.env_example`.
- Added: `RDF_RETENTION_POLICIES` in the same file.
- Local `.env`: no `.env` exists for `orion-rdf-writer` in this worktree (expected — gitignored, worktrees don't inherit one). The **live** `.env` on the host running the actual `orion-rdf-writer` container was updated directly with matching values during incident response, outside this worktree/PR.
- No skipped keys.

## Tests run

```
cd orion/substrate && pytest tests/ -q
265 passed, 17 warnings (pre-existing deprecation warnings, unrelated)
```

## Evals run

No eval harness exists for `orion/substrate/dynamics.py` specifically; this service's `evals/` directory covers other reducers (brain-frame, biometrics). Not adding one here — out of scope for a targeted bug fix, flagged as a gap rather than silently skipped.

## Review findings fixed

- Finding: dormancy check read stale stored `recency_score` instead of the freshly computed value, a regression this PR's own write-guard would introduce (once a node's activation ratchets to a peak — the common case, since `decay_half_life_seconds` defaults to `None` — it can go dormant-check-blind indefinitely).
  - Fix: dormancy check now reads `activations[f"{node_id}:recency"]` directly instead of `node.signals.activation.recency_score`.
  - Evidence: new test `test_dormancy_uses_fresh_recency_not_stale_stored_value` reproduces the exact scenario (activation and pressure held constant across two ticks, only recency moves) and fails without the fix, passes with it.
- Finding: `graph_cognition/features.py`'s `dormant_count` reads the same persisted `dormant` metadata flag exposed to the staleness bug above.
  - Fix: resolved as a side effect of the dormancy-check fix above (same root cause).
  - Evidence: same regression test.
- Finding: `activation_bundle`/`updated_signal`/`updated_node` (three Pydantic `model_copy` calls) were built unconditionally every loop iteration even when the write guard ends up skipping that node.
  - Fix: moved the construction inside the write-guard `if` branch.
  - Evidence: existing tests still pass (265/265); no behavior change, pure efficiency cleanup.
- Finding (not fixed, follow-up flagged): the actual unsafe primitive — `SparqlSubstrateStore.upsert_node()` doing an unconditional DELETE+INSERT — is untouched, and at least 4 other call sites (`orion/substrate/materializer.py:64`, `services/orion-substrate-runtime/app/worker.py:742,1057,1135`) still call it unconditionally and remain exposed to the same write-amplification bug class via a different trigger (perception messages, drive-state ticks, materializer re-runs). Scoped out of this patch deliberately — fixing the shared primitive is a bigger, riskier change than this incident's timeline allowed for; noting here so it isn't lost.

## Docker/build/smoke checks

Not run from this worktree — no live Fuseki/Docker access configured for a full smoke test here. The write-guard behavior itself was effectively live-verified during the incident: production `dynamics.py` was manually patched with an equivalent gate during firefighting, and Fuseki's `/update` request rate visibly dropped from ~2.5/sec sustained to near-zero for unchanged nodes.

## Restart required

```bash
docker compose restart orion-substrate-runtime
docker compose restart orion-rdf-writer
```

## Risks / concerns

- Severity: low
- Concern: the follow-up (other unconditional `upsert_node()` callers) is real and unaddressed in this PR.
- Mitigation: documented above and in review findings; the dynamics-tick path was the dominant, currently-active contributor per live traffic analysis, so this patch addresses the incident's actual cause even though the shared primitive remains generically unsafe.

## Related: 2026-07-16 Fuseki disk-exhaustion incident (context, not part of this PR)

This PR's root-cause investigation was part of a larger same-day incident response:

- `autonomy/drives` graph (37.9M triples, orphaned since its writer was removed 2026-07-15) — cleared via Graph Store Protocol DELETE.
- `autonomy/identity` graph (34.7M triples, `RDF_SKIP_KINDS` already gating new writes off) — pruned to latest-per-subject via a batched SPARQL DELETE script.
- `autonomy/goals` graph (10.6M triples) — pruned via 30-day age-based batched SPARQL DELETE.
- Two Fuseki OOM/unresponsive incidents occurred during cleanup attempts (an unbounded `+` transitive property-path DELETE evaluated from 400k+ free-variable bindings, then a batch size still too large); both self-corrected or were recovered via `make recover` / a direct `docker compose restart`. No data loss — TDB2 transactions are all-or-nothing, and a full server-side backup (`orion_2026-07-16_16-23-37.nq.gz`) was taken before the historical-data cleanup began (later deleted per operator instruction once the cleanup's correctness was established).
- A hardened, bounded-memory batched-delete script (SELECT a small ground-term batch of stale artifact URIs, then DELETE only from that bounded VALUES set, retry-with-backoff on transient failures) was used for the final successful cleanup pass.
- `make compact` still needs to run afterward to actually reclaim disk space — SPARQL DELETE alone doesn't shrink TDB2's on-disk footprint, only `tdbcompact` does.
