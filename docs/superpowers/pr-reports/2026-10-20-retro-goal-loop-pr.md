## Summary

Build the next P3/P4 implementation from the Retro design (#2751/#2796):
verified Nova observations → bounded attempt → optional Orion background goal
selection → native acknowledgment → attributed input → checked/unknown outcome.

## Outcome moved

Fixed scripts can now be handed off to a goal motor with five-second goals,
epoch/attempt validation and immediate human takeover. The browser exposes the
opt-in experiment and its evidence. Unsupported observations stay unknown.

Actual Orion cognition proof is UNVERIFIED: the live gateway refused the eval
with `gpu_pool_unavailable`; logs report thermal shedding (`cabinet_hot`). No
fallback model, scripted reply or thermal override was used. The independent
native mechanic and synthetic-planner browser evals pass and are labeled as such.

## Architecture touched

Retro-local contracts, worker, observation adapter and mailbox; Hub's existing
relay/router, LLM gateway client, template and linked JavaScript. No new service,
global bus schema, migration, memory write or autonomy scheduler.

## Schema / bus / API changes

- Add strict service-local Goal and optional ScriptControl.cognition.
- Add native `/sessions/{id}/goal` and same-origin Hub `/sessions/{id}/plan`.
- Existing registered `llm.chat.request` is used on `metacog_background`, with
  one in-flight request and five-second minimum request spacing per process.
- Attempts, accepted goals and action attribution are bounded local projections
  and evidence. Goal retries cannot extend deadlines. Late replies cannot
  override newer epochs or a human takeover.

## Env/config changes

- `RETRO_COGNITION_ENABLED=false` and `HUB_RETRO_COGNITION_ENABLED=false`.
- Templates, settings and affected compose contract updated.
- `sync_local_env_from_example.py --all-keys orion-retro orion-hub` ran;
  new keys also added default-off to primary operator env files. All env files
  remain ignored. Existing deployment-specific overrides were preserved.
- No dependency changes; local test venv installed Hub's already-pinned Redis
  dependency. npm-generated artifacts were moved outside the worktree.

## Tests and evals

- Retro/shared tests: 95 passed; Hub Retro tests: 5 passed.
- JavaScript syntax checks: PASS.
- Docker Retro build: PASS; isolated service fault smoke: PASS.
- Native Nova mechanic: PASS; blocked walk followed by fixture jump-right changes
  applied actions and actual state, with observed health decrease (not success).
- Normal and 350ms delayed-ack native browser evals: PASS, including goal
  handoff/ack/applied wait/takeover with explicitly synthetic planner.
- Real Orion planner eval: FAIL/UNVERIFIED, GPU-pool thermal admission refusal.
  Evidence: `/tmp/retro-cognition/native-loop.json`.
- Metric lineage, prompt semantics, definition drift, env parity and Hub settings
  defaults: PASS. Observation provenance/dependence/theory/rest/reversibility
  documented in the implementation record.

## Review findings fixed

- Outcome baseline included planner waiting time: capture it at the first actual
  applied frame; deterministic delayed-planner regression added.
- Offer expiry could truncate an accepted goal: accepted execution uses its own
  deadline; late-acceptance regression added.
- Native eval still used pre-wait baseline: changed it to the applied baseline
  and reran the real-core mechanic eval successfully.
- Independent reviewer ran 73 Retro/Hub tests and found no further material
  implementation defects after the two runtime fixes.

## Restart required

No live deployment performed. After merge, enable both gates only when the
experiment is wanted and the current game can be ended. From clean primary
main synchronized with origin/main:

```bash
scripts/safe_docker_build.sh orion-retro up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Retro restart loses its current paused game. Defaults keep cognition off.

## Risks / concerns

- Real planner acceptance is blocked by thermal admission; it must pass before
  claiming the first cognitive loop demonstrated.
- This implements the next goal-loop seam, not all six research phases. WebRTC,
  save/load, SNES qualification, 30-minute client/policy gates, independent leisure,
  learned policy, world-model/transfer and longitudinal memory remain open.
- Two exact core fingerprints and one Nova ROM are qualified; other builds/ROMs
  intentionally remain unknown. No general game understanding is claimed.

## References

[Implementation and observation quality gate](../specs/2026-10-20-retro-goal-loop-implementation.md)
