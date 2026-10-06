## Summary

Two processes were writing over each other in the substrate graph, and one of them kept throwing away its own refreshes.

- **One decay owner.** The Hub and orion-substrate-runtime both faded concept activation on the same Falkor nodes, each from its own cached copy, so their writes landed out of order. The Hub's decay scheduler is removed outright (code, settings, env keys, tests). The runtime's 30 s dynamics tick is now the only decay writer and defaults ON.
- **Refreshes no longer abort on local writes.** The runtime's Falkor cache refresh gave up whenever the same process wrote during the scan ("local mutation during scan; retry required"), which with several tick loops writing at once meant about one refresh in two failed. Writes made during a scan are now recorded and replayed onto the fresh cache at swap time, so the refresh completes and still contains every write.
- **Refresh success rate is visible.** New counters (`hydrate_ok_total`, `hydrate_failed_total`, `hydrate_replayed_writes_total`) and a `falkor_substrate_hydrate_stats` log line at most every 5 minutes.
- Guard test: fails if any Hub module decays activation or writes the decay stamp again.

## Outcome moved

- Bug A: the two writers that made `activation_decayed_at` step backwards (live: 05:28:22 then 05:28:18; 05:33:02 then 05:32:36) and activation tick up are now one writer.
- Bug B: `orion-athena-substrate-runtime` logged 73 `local mutation during scan` aborts out of 80 hydrate failures in 2 h (read-only `docker logs --since 2h`, 2026-10-06 ~05:40 UTC). That failure class is removed. In tests, a process writing during every scan now completes 5 of 5 refreshes; before, it completed none.

## Current architecture

- `services/orion-hub/scripts/main.py` started `substrate_decay_task`, which called `api_routes.decay_concept_activations()` every 120 s. That function re-upserted all 868 concept nodes, stamping them with a `now` taken before a multi-second loop.
- `orion/substrate/dynamics.py::SubstrateDynamicsEngine.tick()` ran every 30 s in orion-substrate-runtime (`worker.py::_dynamics_tick`). It decayed the same nodes and wrote the same stamp.
- `FalkorSubstrateStore._hydrate_from_durable()` (#2500, with the swap lock from #2508) captured `_write_generation` at scan start and raised if it had changed by swap time.

## Architecture touched

- `orion/substrate/falkor_store.py`: `_scan_journal`, `_apply_node_to_cache`, replay-at-swap, counters and stats log.
- orion-hub: decay scheduler removed.
- orion-substrate-runtime: `SUBSTRATE_DYNAMICS_TICK_ENABLED` default flipped to true in `settings.py` and compose. `.env_example` was already true.
- Comments in `activation.py`, `dynamics.py`, `falkor_codec.py` and `cognitive_substrate.py` now name a single decay writer.

## Files changed

- `orion/substrate/falkor_store.py`: journal local writes during a scan and replay them at swap. Replayed edges whose endpoint the scan lacks are skipped. Adds counters and the stats log.
- `orion/substrate/tests/test_complete_hydration.py`: the abort-asserting test is replaced. Adds tests for convergence, replayed edges and skip-key merging, the dangling-edge skip, and journaling stopping after a failed refresh.
- `services/orion-hub/scripts/{api_routes,main}.py`: `decay_concept_activations` and its scheduler task removed.
- `services/orion-hub/app/settings.py`, `.env_example`, `README.md`: `SUBSTRATE_DECAY_SCHEDULER_ENABLED`, `SUBSTRATE_DECAY_SCHEDULER_INTERVAL_SEC` and the Hub's `SUBSTRATE_DYNAMICS_DECAY_MODE` removed; docs updated.
- `services/orion-hub/tests/test_substrate_concept_decay_scheduler.py`: deleted (it tested the removed function).
- `services/orion-hub/tests/test_hub_does_not_decay_substrate_activation.py`: new kill guard.
- `services/orion-hub/tests/test_heartbeat_chassis.py`: drops the monkeypatch of the removed setting.
- `services/orion-hub/scripts/concept_atlas_routes.py`: docstring now names the real decay writer.
- `services/orion-substrate-runtime/{app/settings.py,docker-compose.yml,.env_example,README.md,app/worker.py,tests/test_worker_falkor_routed_store.py}`: tick default ON; docs.
- `orion/substrate/{activation,dynamics,falkor_codec}.py`, `orion/core/schemas/cognitive_substrate.py`, `orion/substrate/tests/test_falkor_store.py`: comments only.

## Schema / bus / API changes

- Added: none.
- Removed: none on the bus or in schemas. One Hub Python function was removed; it had no HTTP route.
- Renamed: none.
- Behavior changed: the Hub no longer writes activation. A Falkor refresh that overlaps a local write now succeeds instead of failing.
- Compatibility notes: none.

## Env/config changes

- Added keys: none.
- Removed keys (orion-hub): `SUBSTRATE_DECAY_SCHEDULER_ENABLED`, `SUBSTRATE_DECAY_SCHEDULER_INTERVAL_SEC`, `SUBSTRATE_DYNAMICS_DECAY_MODE`.
- Default changed (orion-substrate-runtime): `SUBSTRATE_DYNAMICS_TICK_ENABLED` is now true in code and compose. `.env_example` was already true, so the live value does not change.
- `.env_example` updated: yes, in both services.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: run. The sync only adds keys; no keys were added. The three removed Hub keys stay in the local Hub `.env` as harmless leftovers, since Settings uses `extra="ignore"`. The operator can delete them.
- Skipped keys requiring operator action: none.

## Tests run

```text
orion/substrate/tests/test_complete_hydration.py + test_falkor_store.py + test_dynamics_decay_since_last.py: 112 passed (after merging origin/main)
New Bug B tests against the pre-fix falkor_store.py: 4 of 4 fail (abort loop reproduced)
orion/substrate/tests (full): 974 passed, 3 failed. The 3 failures are in test_felt_state_self_definition_lane.py and also fail on the base branch.
services/orion-substrate-runtime/tests: 14 failures plus 2 collection errors, the identical set on the base branch.
services/orion-hub/tests (full): 43 failed, 2 errors, 3257 passed, the identical set on the base branch.
Hub kill guard + heartbeat chassis: 5 passed. Runtime test_worker_dynamics_tick: 5 passed.
scripts/check_env_template_parity.py: PASS. git diff --check: clean.
```

## Evals run

```text
python -m orion.substrate.evals.run_complete_hydration_eval: passed=true
python -m orion.substrate.evals.run_decay_since_last_eval: passed=true (legacy cliff 0.9781/0.9567/0.9357 reproduced)
```

## Docker/build/smoke checks

```text
Not run (no deploy or restart per task scope). The live effect is UNVERIFIED until restart.
```

## Review findings fixed

- Finding: a replayed edge could be cached with an endpoint node that was deleted elsewhere mid-scan.
  - Fix: the replay skips an edge whose endpoint is missing from the fresh scan.
  - Evidence: `test_replayed_edge_with_missing_endpoint_is_not_cached`.
- Finding: an older replayed local write can mask a newer write from another process that the scan read, until the next refresh.
  - Fix: documented as a trade-off on `_scan_journal`. This matches steady-state cache behavior.
  - Evidence: comment in `falkor_store.py`.
- Finding: a stale docstring in `test_worker_falkor_routed_store.py` said the dynamics tick stays off.
  - Fix: updated.
  - Evidence: diff.
- Finding: the counters are updated outside the lock.
  - Fix: a comment explains why this is safe (hydrate callers are serialized).
  - Evidence: diff.

## Restart required

Restart orion-substrate-runtime first, so decay is never unowned (worst case, briefly double-owned), then the Hub.

```bash
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build && scripts/safe_docker_build.sh orion-hub up -d --build
```

Deploy from the primary checkout on main after merge, not from this worktree.

## Risks / concerns

- Severity: low.
  - Concern: if orion-substrate-runtime is down or its tick is disabled, nothing decays activation.
  - Mitigation: this is intended (one owner, no fallback). The tick now defaults on.
- Severity: low.
  - Concern: about 8 per 2 h hydrate failures of a different kind remain: `missing or mismatched endpoint: sub-entity-topicfoundry-...`. These come from another process writing mid-scan, a mixed-time view that this patch does not address.
  - Mitigation: follow-up. `falkor_substrate_hydrate_stats` will now show the rate.
- Severity: UNVERIFIED.
  - Concern: no live evidence yet that the stamp is monotonic and that hydrate failures dropped.
  - Mitigation: after restart, check `docker logs orion-athena-substrate-runtime | grep falkor_substrate_hydrate_stats` and re-read `sub-concept-seed-juniper`'s `activation_decayed_at` over a few ticks.

## PR link

See the PR page.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
