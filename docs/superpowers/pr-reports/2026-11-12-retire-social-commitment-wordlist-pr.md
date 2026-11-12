## Summary

- Social memory used to scan Orion's room replies for promise phrases ("I'll come back to", "I'll summarize", ...) and keep them as open "commitments" on each room. This PR removes that detector completely, as Juniper approved on 2026-11-12.
- Removed the producer, all five promise word lists, the two schemas (`SocialCommitmentV1`, `SocialCommitmentResolutionV1`), both bus channels, four env keys, and the `active_commitments` field on room continuity.
- Removed every consumer: bridge routing and repair, the Hub prompt metadata, the social-room prompt block, the inspection section and its Hub badge, context-window candidates, and freshness records.
- Stored data is left in place. The DB column still holds the old rows. Nothing reads or writes it now.
- A judge-based promise finder replaces this on the unified turn (intention-ledger design PR #3188, step D, built separately).

## Outcome moved

A word list that found 1 promise in 18,702 room turns no longer runs, no longer shapes routing, and can no longer grow. Promise tracking gets one owner (step D) instead of two.

Live evidence (Postgres, 2026-11-12):

```text
select count(*), count(*) filter (where active_commitments::text not in ('[]','null'))
  from social_room_continuity;                         -> 67 | 1
select count(*) from social_room_turns;                -> 18702
the one commitment: aitown c:272389, 2026-07-30 05:51Z, type yield_then_reenter,
  "Yield for now, then re-enter ...", state open, expired 07:21Z (room never had another turn)
rows in any social table naming a commitment artifact/candidate/trigger kind -> 0
(2 room turns mention "social commitment channel" as conversation text only)
```

So the "1 commitment in 67 rooms" figure from the design doc is confirmed.

## Current architecture

`services/orion-social-memory/app/synthesizer.py` ran `update_commitments` on every room turn: `extract_commitments` matched `_COMMITMENT_*_HINTS` phrases in Orion's reply, then TTL/fulfil/supersede logic kept up to 3 open items on `SocialRoomContinuityV1.active_commitments`. Social memory published `orion:social:commitment` and `orion:social:commitment:resolution`. The bridge read `active_commitments` to steer routing (summarize / wait), to raise repair confidence, and to pass `social_open_commitments` to Hub, which rendered an `OPEN COMMITMENTS:` prompt block. Inspection built a "commitments" section; the Hub debug panel showed a Commitments badge. Context-window selection and floor/ritual decisions also changed when commitments existed.

## Architecture touched

- orion-social-memory: producer, settings, compose, ORM model, README.
- orion-social-room-bridge: routing and repair no longer read commitments.
- orion-hub: cortex request builder metadata, social-inspection JS.
- Shared contracts: schemas, registry, channels.yaml, metric definition lock.
- Prompts: `chat_social_room.j2`, one example line in `chat_general.j2`.

## Files changed

- `orion/schemas/social_commitment.py`: deleted.
- `orion/schemas/registry.py`, `orion/bus/channels.yaml`: schemas and both channels removed.
- `orion/schemas/social_memory.py`: `active_commitments` removed; a before-validator drops a legacy `active_commitments` key for registry/bus consumers during rollout (see Risks).
- `orion/schemas/social_{context,epistemic,freshness,inspection,repair}.py`: removed Literal values only the detector produced (`commitment`, `open_commitment`, `commitments`, `active_commitment`).
- `services/orion-social-memory/app/synthesizer.py`: producer, word lists, TTL/fulfil/supersede logic, freshness record, context candidate and every commitment condition removed.
- `services/orion-social-memory/app/service.py`: no call, no publish, no log lines, no row read.
- `services/orion-social-memory/app/models.py`: ORM column removed (DB column kept).
- `services/orion-social-memory/app/settings.py`, `.env_example`, `docker-compose.yml`: four keys removed.
- `services/orion-social-room-bridge/app/policy.py`, `app/service.py`: commitment-steered routing, `_relevant_commitments`, and `social_open_commitments` removed.
- `services/orion-hub/scripts/cortex_request_builder.py`: `social_open_commitments` metadata removed.
- `services/orion-hub/static/js/social-inspection.js` (+ test): Commitments badge and label removed.
- `orion/inspection/social.py`: commitments section removed.
- `orion/cognition/prompts/chat_social_room.j2`, `chat_general.j2`: block and stale wording removed.
- `orion/social/scenario_replay.py`: fixture key removed.
- `config/metrics/metric_definitions.lock.json`: re-locked (2 high "removed" deltas, the two channels).
- `README.md`, `services/orion-social-memory/README.md`, `docs/architecture/social-conversational-commitments.md`: marked retired.
- Tests: see below.

## Schema / bus / API changes

- Added: none.
- Removed: `SocialCommitmentV1`, `SocialCommitmentResolutionV1`; channels `orion:social:commitment`, `orion:social:commitment:resolution`; field `SocialRoomContinuityV1.active_commitments`; Hub request metadata `social_open_commitments`; Literal values listed above.
- Renamed: none.
- Behavior changed: bridge `commitment_contradiction` repair (a peer-message contradiction cue, not the detector) now always has confidence 0.58, trigger `peer_message`, and decision `clarify`. That is what it already did in 66 of 67 rooms, because commitments were almost always empty.
- Compatibility notes: no code subscribed to the two channels. A legacy `active_commitments` key is dropped unread by the schema, so old producers do not fail validation during a rolling rebuild.

## Env/config changes

- Added keys: none.
- Removed keys: `SOCIAL_MEMORY_COMMITMENT_CHANNEL`, `SOCIAL_MEMORY_COMMITMENT_RESOLUTION_CHANNEL`, `SOCIAL_MEMORY_COMMITMENT_TTL_MINUTES`, `SOCIAL_MEMORY_COMMITMENT_MAX_OPEN`.
- Renamed keys: none.
- `.env_example` updated: yes (`services/orion-social-memory/.env_example`).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: ran, "No changes needed". The sync script never deletes keys, so the four dead keys were removed from the primary checkout's `services/orion-social-memory/.env` by hand (settings use `extra="ignore"`, so they were harmless either way).
- skipped keys requiring operator action: none.

## Tests run

```text
services/orion-social-memory: pytest tests -q            -> 71 passed, 2 failed (same 2 fail on main: test_bridge_summary_yields_back_to_relevant_peer, test_stale_consensus_candidate_is_excluded_in_favor_of_fresher_divergence)
services/orion-social-room-bridge: pytest tests -q       -> 72 passed
tests/test_social_inspection.py tests/test_social_room_prompt.py -> 24 passed
hub social tests (test_social_room, test_cortex_request_builder, test_social_room_turn_publish, test_hub_direct_social_room_isolation, test_hub_direct_inspection_cache) + tests/test_social_room_scenario_replay.py -> 83 passed, 2 failed (same 2 scenario failures on main)
bus catalog / registry tests (12 files)                   -> 56 passed
orion/body_atlas/tests                                    -> 275 passed
node --test social-inspection.test.js                     -> 2 passed
every run: step in .github/workflows/orion-static-gates.yml -> no failures
check_definition_drift.py --gate                          -> PASS
check_env_template_parity.py                              -> PASS
git diff --check                                          -> clean
```

New regression tests: `test_retired_promise_detector_leaves_no_trace` (social memory: old trigger phrases create no field, no commitment channel publish, no candidate, no freshness record); `test_legacy_active_commitments_no_longer_steer_routing` and `test_contradiction_cue_asks_for_clarification` (bridge); prompt and inspection tests now assert a legacy key is ignored; JS test asserts no Commitments badge.

## Evals run

```text
No eval harness covers social-memory promise detection. Nothing to run; the detector is gone.
The social-room scenario replay (tests/test_social_room_scenario_replay.py) ran: same 2 pre-existing failures as main.
```

## Docker/build/smoke checks

```text
Not run. Deploy happens from main after merge via the post-pull rebuild.
```

## Review findings fixed

- Finding (low): the rollout note claimed the schema shim protects Hub. Hub never validates `SocialRoomContinuityV1`. It does validate context-window candidates and inspection snapshots, whose `commitment`/`commitments` kinds are removed. If Hub is rebuilt before social-memory and room `c:272389` (the one stored open commitment) gets a turn in that gap, that one turn fails validation.
  - Fix: shim comment corrected (it protects registry/bus consumers of `orion:social:room:continuity`); restart order below is now social-memory first, then bridge, then Hub.
  - Evidence: `orion/schemas/social_memory.py` comment; "Restart required" section.
- Finding (low): no test pinned the shim.
  - Fix: `test_room_continuity_schema_drops_retired_active_commitments_key` added; removal condition added to the shim comment.
  - Evidence: `tests/test_social_inspection.py` -> 24 passed.
- Finding (nit): bridge `commitment_contradiction` decision kept an unreachable `>= 0.75 -> repair` branch.
  - Fix: decision is `clarify` directly; comment says the cue does not read commitments.
  - Evidence: bridge tests 72 passed.
- Finding (nit): unused `fuchsia` badge tone. Checked: the tone is still used elsewhere in Hub (`app.js`, `thought-process.js`). No change.
- Reviewer mutation check: all 7 new or changed tests fail against `origin/main` code, so they catch the detector coming back.

## Restart required

```bash
scripts/safe_docker_build.sh orion-social-memory up -d --build && scripts/safe_docker_build.sh orion-social-room-bridge up -d --build && scripts/safe_docker_build.sh orion-hub up -d --build
```

(Run from the clean primary `main` checkout after merge, in this order: social-memory first. The post-pull rebuild normally does this by itself.)

## Risks / concerns

- Severity: low. Concern: rollout order. Registry consumers of the room-continuity channel validate with a forbid model; the schema drops a legacy `active_commitments` key, so an old social-memory does not break them. Hub validates context candidates and inspection snapshots, so if Hub is rebuilt first and room `c:272389` gets a turn in the gap, that one turn fails. Mitigation: rebuild social-memory first.
- Severity: low. Concern: stored data. `social_room_continuity.active_commitments` (JSON) still exists: 67 rows, 66 hold `[]`, 1 holds the single July commitment. Mitigation: unread and unwritten. Drop later with a backup first; that needs Juniper.
- Severity: low. Concern: the bridge still has a `commitment_contradiction` repair type. It is a separate peer-message cue, not this detector, and its name is now a little misleading. Left alone to keep this PR to the approved scope.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
