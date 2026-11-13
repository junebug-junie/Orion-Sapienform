## Summary

DESIGN ONLY. No runtime code. Juniper approves before any build. Do not merge until she has
answered the open questions.

- Replay evals call the model servers directly, so the GPU pool never sees them. On 2026-11-12
  03:20-04:30Z a replay ran on chat's card and the agent card. The pool recorded one chat grant in
  that window. One live judge call timed out at 04:10Z.
- The existing fences miss evals. The CI address check skips `evals/` folders. The circe firewall
  lets athena through, and the evals run on athena.
- The design has three phases. Phase 1: evals take a `background` pool lease and call the gateway.
  Phase 2: a "door" in front of each model server reports calls that skipped the gateway. Phase 3:
  the door refuses them.
- Placement recommendation: a new `operator` work class, not a body-economy tier. It never bids, and
  it sorts last in the queue. Its energy is reported as `sponsored` (Juniper's work), not as Orion's
  activity.

## Outcome moved

None yet (design). When built, the failure this targets is: an eval takes a GPU seat the pool
believes is free, and Orion's live work (a judge call, Juniper's chat) times out behind it.

## Current architecture

See the spec's "Current architecture" and "Direct-caller inventory". In short:

- 13 callers reach a model server without the gateway: 10 unleased, 2 leased but direct, and
  1 runtime service (substrate -> kev).
- Operator holds exist, but at `interactive` priority, and the heat reflex never sheds them.
- No server log can attribute a call to its caller today.

## Architecture touched

Docs only: `docs/superpowers/specs/2026-11-12-leased-evals-and-server-refusal-design.md` and this
report.

## Files changed

- `docs/superpowers/specs/2026-11-12-leased-evals-and-server-refusal-design.md`: the design (design
  mode plus the proposal-mode checklist).
- `docs/superpowers/pr-reports/2026-11-12-leased-evals-and-server-refusal-design-pr.md`: this report.

## Schema / bus / API changes

- Added: none in this PR. Proposed: operator classes and routes in `config/gpu_pool.yaml`, gateway
  dispatch headers `X-Orion-Lane-Key` / `X-Orion-Lease-Id`, and a lane-door service.
- Removed: none
- Renamed: none
- Behavior changed: none
- Compatibility notes: the design avoids a new `Priority` value on purpose (`extra=forbid` readers).

## Env/config changes

- Added keys: none in this PR. Proposed: `GPU_POOL_OPERATOR_LAST`,
  `PHYSIOLOGY_OPERATOR_LOAD_SPONSORED`, `LANE_DOOR_MODE`.
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template
  changed)
- skipped keys requiring operator action: none

## Tests run

```text
None. The PR changes docs only. CI static gates run on push.
```

## Evals run

```text
None (design). The evidence queries are in the spec: gpu_pool_events (30 days), circe docker ps,
the Strata engine log, llama.cpp journald, check_circe_worker_refs.py --live-env.
```

## Docker/build/smoke checks

```text
Not applicable (docs only).
```

## Review findings fixed

- Design PR. No code review was run, because there is no code. Juniper's review of the open
  questions is the gate.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: high (when built)
  - Concern: the Phase 2/3 door sits on Juniper's chat path. A dead door or a key mismatch mutes
    Orion.
  - Mitigation: chat flips last, only after 7 clean shadow days. The door reports a key hash, and the
    gateway compares it at boot. Rollback is one env value plus a restart.
- Severity: medium
  - Concern: how often evals hit the servers today is UNVERIFIED. Neither the pool nor the server
    logs record it. Phase 2 exists to measure it.
- Severity: medium
  - Concern: conflicts with earlier decisions: operator holds at `interactive`, and the 10-30
    energy ruling. The spec names each conflict and proposes a ruling. Juniper decides.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3237

🤖 Generated with [Claude Code](https://claude.com/claude-code)
