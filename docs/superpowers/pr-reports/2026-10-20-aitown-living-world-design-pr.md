## Summary

- Finish the living-world design in #2666; no runtime changes.
- Separate LLM actions from state reads and validated commits.
- Define weather snapshot provenance and a public town-region source.
- Extend mirror acceptance checks and record remaining implementation choices.

## Outcome moved

The design now specifies implementable LLM orchestration and source references for both digest and weather events. The next patch remains the dialogue echo fix.

## Current architecture

The town uses a patched Convex engine with line-by-line NPC dialogue. The existing self-repeat guard checks a speaker's own lines. The proposed living-world simulation is not implemented.

## Architecture touched

Design documentation only. Proposed simulation stays in Convex; the bridge stays in embodiment.

## Files changed

- `docs/superpowers/specs/2026-10-14-aitown-living-world-design.md`: simulation proposal and reviewed contract corrections.
- `docs/superpowers/pr-reports/2026-10-20-aitown-living-world-design-pr.md`: review and validation record.

## Schema / bus / API changes

- Added: none at runtime. Proposed internal actions, typed mirror source references and retained weather snapshots.
- Removed: none.
- Renamed: none.
- Behavior changed: none at runtime.
- Compatibility notes: future mutations revalidate state and commit each scene at most once.

## Env/config changes

- Added keys: none at runtime; two optional public-weather settings proposed.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed; no template changes.
- skipped keys requiring operator action: none. Selecting a weather provider and town region belongs to implementation PR 5.

## Tests run

```text
python3 scripts/check_personal_terms.py -> passed
 git diff --check -> passed
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-ai-town/tests/test_npc_no_self_repeat_patch.py services/orion-ai-town/tests/test_mechanical_leave_patch.py -q -> 8 passed
```

The eight checks cover existing patch text, not live dialogue. System Python lacked pytest; the repository virtual environment ran them successfully.

## Evals run

None: design-only. A1–A9 specify future implementation checks, including scene quality and forecast scoring. No runtime capability is claimed.

## Docker/build/smoke checks

Not applicable: documentation only. Live simulation behavior remains UNVERIFIED.

## Review findings fixed

- Finding: queries and mutations cannot orchestrate external LLM calls.
  - Fix: specify internal actions, with queries for snapshots and mutations for validated commits.
  - Evidence: spec §7 and schema/API section; independent review returned no findings after revision.
- Finding: weather models lack the digest item ID required by the original mirror contract.
  - Fix: typed source references, immutable weather snapshots, public-provider acquisition and explicit town-region configuration.
  - Evidence: spec §5, schema/API section and A9; independent review returned no findings after revision.

## Implementation handoff (2026-10-21)

Another agent started building this design, then stopped before it committed anything. The work is now saved on branch `feat/aitown-living-world-implementation` (commit `27edb72e7`, pushed, no PR). It is a draft. Nobody has reviewed it, and it is not deployed.

What the draft contains (44 files, about 1,560 lines):

- The simulation core as Convex overlay files (`services/orion-ai-town/overlay/convex/sim/`), hook patch `orion-sim-hooks.patch`, and `scripts/test_sim.sh`.
- The bus contract: five `orion:town:sim:*` channels and `orion/schemas/town_sim.py` (registered). This adds `TownSimStoredV1`, which this spec does not name.
- The embodiment bridge and mirror (`orion/embodiment/town_sim_bridge.py`, `town_mirror.py`), the cortex verbs `town_forecast` and `town_add_event`, the hub form, the sql-writer consumer, and `manual_migration_town_sim_v1.sql`.

Evidence on 2026-10-21: `scripts/test_sim.sh` passed 14/14. The pytest run on `services/orion-ai-town/tests` and `services/orion-embodiment/tests/test_town_sim_bridge.py` passed 148. Live behavior is UNVERIFIED.

The draft does not follow the build order above:

1. PR 1 is missing. There is no `orion-live-turn-roles.patch`, no partner-echo check, and no `c:578946` fixture. The draft also rewrites `orion-npc-no-self-repeat.patch`, and its retry prompt is shorter than the reviewed version. Check that rewrite before you keep it.
2. All five PRs are in one branch. Split it in the order PR 1 → PR 5. Use the draft as source material, not as one merge.
3. There is no A5 scene-writer eval.
4. `services/orion-embodiment/.env_example` and `services/orion-hub/.env_example` changed. Run `python scripts/sync_local_env_from_example.py` when each slice lands.
5. Excluded on purpose: the `services/orion-hub/node_modules/` changes (a local jsdom install, not part of this work). They stay uncommitted in that worktree.

Next action: start PR 1 from `main` in a fresh worktree. Take only the echo/role pieces from the draft branch.

## Restart required

No restart required.

## Risks / concerns

- Severity: implementation risk.
- Concern: scene quality and forecast usefulness have not been demonstrated.
- Mitigation: retain the proposed quality evals and metric gate before using scores in cognition. Weather polling remains disabled until its public source and town region are configured.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2666
