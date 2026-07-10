# feat(orion-substrate-runtime): wire goal-context listener for voluntary attention

**Status:** IMPLEMENTED, tested, reviewed. Closes the last-mile gap in the
already-merged voluntary-attention-override feature (PR #891).

## Summary

`orion/substrate/attention/top_down.py` (biased competition / voluntary
attention) has been merged, fully wired into the live attention-frame
pipeline, and gated off by `ORION_ATTENTION_TOPDOWN_ENABLED=false` since
PR #891. Traced the entire codebase and found `set_active_goal()` — the only
function that populates the goal-context store the feature reads from — was
never called anywhere except its own definition and a unit test. Even with
the flag enabled, voluntary attention would have been a permanent no-op.

This adds the missing bus consumer: `orion-substrate-runtime` now subscribes
to `orion:memory:goals:proposed` (`GoalProposalV1`, already producer-wired by
`orion-spark-concept-induction`, already permitted for any consumer via the
channel's wildcard) and calls `set_active_goal()` on every valid proposal.

**`ORION_ATTENTION_TOPDOWN_ENABLED` stays `false`** — this patch only makes
the store have real data ready; enabling voluntary attention is a separate,
later decision.

## Architecture touched

- New live bus-pubsub consumer in `orion-substrate-runtime`, mirroring the
  existing `post_turn_closure_listener.py` pattern almost 1:1 (same
  decode→kind-check→validate→dispatch shape, same never-raise discipline,
  same lifespan start/stop wiring).
- No changes to `orion/substrate/attention/goal_context.py`,
  `top_down.py`, or `attention_broadcast.py` — those were already correct.

## Files changed

- `services/orion-substrate-runtime/app/goal_context_listener.py` (new) —
  the consumer. No separate enable flag (always runs when the bus is
  enabled) — harmless when idle, since nothing reads the store unless
  `ORION_ATTENTION_TOPDOWN_ENABLED` is also true.
- `services/orion-substrate-runtime/app/settings.py` — `channel_goal_proposal`
  field.
- `services/orion-substrate-runtime/.env_example` — `CHANNEL_GOAL_PROPOSAL`
  key.
- `services/orion-substrate-runtime/app/main.py` — third listener wired into
  `lifespan()` alongside the two existing ones (finalize-appraisal,
  post-turn-closure), same start/stop pattern.
- `orion/bus/channels.yaml` — added `orion-substrate-runtime` to the
  `orion:memory:goals:proposed` entry's `consumer_services` (documentation
  clarity; the wildcard already permitted this, no contract change).
- `scripts/sync_local_env_from_example.py` — `CHANNEL_GOAL_PROPOSAL` added
  to `SYNC_EXACT` (this exact allowlist gap has bitten this repo's
  config-sync workflow repeatedly this week — closed proactively this time).
- `services/orion-substrate-runtime/tests/test_goal_context_listener.py`
  (new) — 5 tests: valid envelope → `set_active_goal` called with decoded
  goal; decode failure → no-op; wrong `kind` → no-op + warning logged;
  malformed payload → no-op; bus disabled → returns before ever calling
  `subscribe()`.

## Schema / bus / API changes

- Added: none (reuses the existing `GoalProposalV1` schema and
  `orion:memory:goals:proposed` channel, both already registered).
- Compatibility: purely additive — `orion-substrate-runtime` added as an
  explicit consumer of an already-wildcard-permitted channel.

## Env/config changes

- Added key: `CHANNEL_GOAL_PROPOSAL=orion:memory:goals:proposed`
  (`orion-substrate-runtime`).
- `.env_example` updated; local `.env` synced via
  `python scripts/sync_local_env_from_example.py` (not committed,
  gitignored, confirmed via `git status --short`).

## Tests run

```text
pytest services/orion-substrate-runtime/tests/test_goal_context_listener.py \
       services/orion-substrate-runtime/tests/test_post_turn_closure_listener.py \
       services/orion-substrate-runtime/tests/test_execution_trajectory_endpoint.py -q
  → 13 passed

pytest services/orion-substrate-runtime/tests --ignore=.../test_grammar_consumer_integration.py -q
  → 93 passed, 14 failed — same 14 pre-existing unrelated failures confirmed
    identical (via git-stash comparison) multiple times this session; this
    is a known cross-test module-global-state artifact in this suite, not
    caused by this patch.

pytest tests/test_autonomy_goals_bus_catalog.py -q
  → 3 passed (confirms the channels.yaml edit doesn't break the existing
    catalog contract test, which only checks producer_services/schema_id,
    not an exact consumer_services list).
```

## Review findings fixed

- One stylistic note from the build agent's own review pass: the
  function-local `from orion.substrate.attention.goal_context import
  set_active_goal` import (inside `_handle_bus_message`, not at module
  top-level) lacked a comment explaining why — it's there specifically so
  `monkeypatch.setattr("orion.substrate.attention.goal_context.set_active_goal", ...)`
  in tests actually takes effect (a module-level import would bind the
  original function object at import time, making the monkeypatch a no-op
  for this module's already-bound reference). Comment added.
- Orchestrator re-review (code-review skill, medium effort): no further
  findings. Verified no other test snapshot-asserts the exact
  `consumer_services` list for this channel (only checks membership), so
  the `channels.yaml` edit is safe.

## Docker/build/smoke checks

Not run against live containers in this environment. No restart is required
for this branch by itself to be inert-safe (the listener is always-on but
the feature it feeds stays gated off); restart `orion-substrate-runtime` to
pick up the new consumer once merged:
```bash
docker compose --env-file .env --env-file services/orion-substrate-runtime/.env \
  -f services/orion-substrate-runtime/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low. New consumer only writes to an in-memory store nothing
  reads unless a separate, still-off flag is also flipped. Never raises
  (decode failure, wrong kind, malformed payload all log-and-return).
- Note for whoever eventually flips `ORION_ATTENTION_TOPDOWN_ENABLED`: the
  actual producer (`GoalEngine.propose()`) only ever emits `proposal_status`
  `"proposed"` or `"active"` on this channel — never terminal states like
  `"completed"`/`"failed"`. The store's terminal-clear branch
  (`goal_context.py`'s `_ACTIVE_STATES` check) will essentially never fire
  in practice; behavior will be "latest-goal-wins" rather than clean
  clearing on completion. This matches `goal_context.py`'s own documented
  "MVP proxy" design intent — not a bug introduced by this patch, just worth
  knowing before enabling.

## PR link

Branch pushed: `feat/goal-context-listener-production`.
Compare: https://github.com/junebug-junie/Orion-Sapienform/compare/main...feat/goal-context-listener-production
