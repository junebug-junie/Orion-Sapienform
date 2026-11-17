## Summary

- Orion's substrate graph no longer pretends to look for contradictions. Nothing ever produced a contradiction node, and the durable store refuses the kind, so every contradiction query came back empty. Juniper chose option (B) on 2026-11-17: contradictions are out of the graph for now. The query (`query_contradiction_region` / `read_contradiction_region`) is gone from the store protocol, all four stores, the curiosity seed plan, review bootstrap and frontier curiosity.
- The Hub graph panels now say "not available" (`available: false`, reason names #3544) for contradictions on every backend, and `contradiction_count` is `None` instead of a false `0`.
- The durable whitelist (`orion/substrate/falkor_codec.py`, `DURABLE_NODE_KINDS`) carries a comment: adding `contradiction` needs a producer and a consumer built together.
- `scripts/analysis/curiosity_target_return_days.py` counts days in America/Denver, not UTC, and `--repeats` prints each repeat target with its dates.
- `orion/world_pulse_read/resonance_records.py`: comments now say the previous-answer neighbour is the answer before the current one, even a draft, on purpose.
- #3542 (retire `metacog_perception`, `coherence_state`, `goal_pressure` and the stub `GraphCognitionAdapter`) was already done on main by PR #3522; the issue is closed. `make check-definition-drift UPDATE=1` left the lock unchanged.

## Outcome moved

The curiosity seed and review bootstrap stop spending a query on a region that cannot hold anything. The Hub stops showing "0 contradictions" when the true answer is "not represented".

## Current architecture

Contradiction nodes had no live producer (`emit_contradiction` test-only), and Falkor refused them at write. Three readers still queried the region: curiosity seed plan, review bootstrap, Hub graphdb payloads.

## Architecture touched

`orion/substrate` store protocol and implementations, query planning, review bootstrap, frontier curiosity; Hub `api_routes.py` graphdb payloads. No bus, schema or env change.

## Files changed

- `orion/substrate/store.py`, `graphdb_store.py`, `falkor_store.py`, `routed_store.py`: remove contradiction-region methods.
- `orion/substrate/query_planning.py`, `frontier_curiosity.py`, `review_bootstrap.py`: drop the step and its consumers.
- `orion/substrate/falkor_codec.py`: whitelist comment for #3544.
- `services/orion-hub/scripts/api_routes.py`: honest `available: false` payload.
- `scripts/analysis/curiosity_target_return_days.py`: Denver days, `--repeats`.
- `orion/world_pulse_read/resonance_records.py`: comment fix.
- Tests in `orion/substrate/tests/` and `services/orion-hub/tests/test_substrate_review_runtime_hub_debug.py`.

## Schema / bus / API changes

- Removed: `SubstrateGraphStore.query_contradiction_region` / `read_contradiction_region`; `contradiction_region` step from the `curiosity_seed` plan.
- Behavior changed: Hub `/graphdb` overview `contradiction_count` is `None`; `regions.contradiction` and `contradiction_hotspots` are `{available: false, reason}` on every backend.
- Compatibility notes: no Hub JS/template reads these fields (grep).

## Env/config changes

- None. `.env` sync not needed.

## Tests run

```text
pytest orion/substrate orion/signals orion/world_pulse_read --ignore=orion/substrate/experiments
  1600 passed, 1 failed (test_felt_state_lived_answers: fails on main too, tracked in #3476)
pytest services/orion-hub/tests -k "substrate or graphdb"   228 passed
make check-definition-drift GATE=1                          PASS (lock unchanged)
scripts/check_metric_lineage.py --gate                      PASS
scripts/check_integration_seam.py                           OK
```

## Evals run

```text
No eval harness covers the removed query; this is a retirement. Live script run below stands in.
```

## Docker/build/smoke checks

```text
python3 scripts/analysis/curiosity_target_return_days.py --days 14 --repeats   (live DB, read-only)
  targets 2011, targets_2plus_days 63, world_2plus_days 41; repeat lines print with Denver dates.
  created_at is timestamptz, so `at time zone 'America/Denver'` converts correctly.
```

## Review findings fixed

- Code-review subagent: no material findings. It confirmed no remaining callers (py/js/html/ts), the 2-tuple unpack in frontier_curiosity, and the script SQL (live run `--days 7 --repeats`). Optional note on the consolidation.py label is listed under Risks.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

Substrate consumers in other services pick up the change on their next normal rebuild; the post-pull rebuild covers it.

## Risks / concerns

- Severity: low. Concern: `orion/substrate/consolidation.py` still labels unresolved regions with the string `contradiction_region` and counts contradiction nodes in a region; that is a label, not the retired query. Mitigation: left as is to keep this patch narrow.
- Issues: Fixes #3544. #3542 already closed. #3476 (pre-existing felt-state test failure) referenced, not changed.

## PR link

(filled on open)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
