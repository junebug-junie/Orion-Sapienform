## Summary

- A small background image job no longer kills a long reading or curiosity turn on gpu2.
- New scheduler rule H5 (`orion/gpu_pool/scheduler.py`): when background owner work (every diffusion job today) is the only reason a swap seat drains, its holds are recalled with reason `draining_background` and a 3600 s grace instead of 600 s. The running step finishes. An idle hold still gives the seat back at its next step start.
- If the seat later drains for anything else (a non-background owner, max_hold, an operator), the hold gets the normal 600 s, counted from its recall.
- New setting `Defaults.background_drain_grace_sec` (code default 3600). It is deliberately not in `config/gpu_pool.yaml` yet: `Defaults` is `extra="forbid"`, and circe's lane controller parses that file, so a new key would break any reader still on older code.

## Outcome moved

Self-monitor handoff B, finding 1: 13% of Orion's turns fail (`turn.run`). The 7-day breakdown (10-28..11-04, 262 failed of 2,114) traced about 87 failures to the GPU pool taking the card back mid-turn. 81 of the 106 expired turn leases were `agent-gpu2` drains for diffusion: 319 background image jobs, each about 1 minute (reverie-visual 190, orion-thought 129). Each drain killed the running turn after 600 s and restarted it from scratch, even when the turn had run for 10-60 minutes. This patch keeps those turns running. Images now wait up to 1 h for the running step, which is inside the 2.5 h wait Juniper allowed on 2026-09-29.

## Current architecture

`agent-gpu2` is a swap seat that borrows gpu2 from diffusion (`swap.evicts: [diffusion]`, warm standby). Any queued diffusion lease that fits drains the seat. Every granted lease there is then recalled with `hold_clawback_grace_sec` (600 s) for holds, then aborted (`recall_grace_exceeded`) and re-queued. Durable-runs gives an idle hold back at its next node start, but a node running a 10-60 min harness turn was killed.

## Architecture touched

- `orion/gpu_pool` scheduler and config only. No bus, schema, or API change. The new recall reason string goes into existing `reason` fields. No code branches on `"draining"` (searched).

## Files changed

- `orion/gpu_pool/scheduler.py`: H5 rule; `BACKGROUND_DRAIN` / `BACKGROUND_DRAIN_CUT` reasons; `_Ctx.background_drain`; early abort once a later non-background drain arrives.
- `orion/gpu_pool/config.py`: `Defaults.background_drain_grace_sec` (3600).
- `config/gpu_pool.yaml`: a comment only, explaining why the key is not there yet.
- `orion/gpu_pool/tests/test_background_drain_grace.py`: new tests.
- `orion/gpu_pool/tests/test_warm_standby.py`, `test_stage7_3_max_holds.py`, `test_lease_graph.py`: pinned expectations updated to the new reason and grace.

## Schema / bus / API changes

- Added: recall reason `draining_background` and abort reason `background_drain_cut` (values, not fields).
- Removed / Renamed: none.
- Behavior changed: the recall grace for holds on a seat drained only by background demand.
- Compatibility notes: the YAML is unchanged in shape, so older readers are unaffected.

## Env/config changes

- None. No `.env_example` changed, so no `.env` sync was needed.

## Tests run

```text
pytest orion/gpu_pool/tests services/orion-gpu-pool/tests -q  -> 736 passed, 15 skipped
```

## Evals run

```text
No eval harness exists for the scheduler. The live check after deploy is below.
```

## Docker/build/smoke checks

```text
Not run. Pure scheduler logic. Live check after deploy:
  durable_resource_events resource.lease_expired joined to gpu_pool_events recalled reason=draining
  on lane agent-gpu2 should drop to ~0; gpu_pool_events should show reason=draining_background.
```

## Review findings fixed

Code review ran in a subagent. It found no material bug: no double abort with the expiry path, the drain sets are computed before use on every path, nothing branches on the `"draining"` reason, and the lease graph accepts the new reason and an early abort.

- Finding (low): the same lease could get Expire (heartbeat lost) and the new Abort in one tick.
  - Fix: the cutback skips a lease whose heartbeat already expired.
  - Evidence: `test_a_hold_whose_heartbeat_was_lost_gets_one_decision_not_two`.
- Finding (low/medium, observability): the cutback abort fired before the published `recall_by` with the generic `recall_grace_exceeded` reason.
  - Fix: it now uses its own reason, `background_drain_cut`, so the lease history says why.
  - Evidence: `test_a_later_interactive_owner_cuts_the_long_grace_back_to_the_normal_one`.
- Finding (low): the YAML rollback note contradicted the "not in this file yet" note.
  - Fix: the comment now says that rollback today is a code change.
- Policy note (no change): for up to 1 h, a background image now waits behind a running text turn. That inverts "diffusion trumps text" in `visual_chain.py` for background work, on purpose. One-off orion-thought image requests (90 s deadline) already deferred under the 600 s grace, so that is not a regression.

## Restart required

The pool reads the scheduler code, so redeploy the GPU pool service from clean main after merge:

```bash
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
```

## Risks / concerns

- Severity: low. Concern: an image waits up to 1 h when it lands during a long turn. Mitigation: all diffusion today is background; a non-background image still gets the old 10 min cut-off.
- Severity: low. Concern: a turn longer than 1 h is still killed. Mitigation: that is the old behavior with a longer limit.

## PR link

(filled on open)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
