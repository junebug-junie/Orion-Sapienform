## Summary

- Retro co-play now shows up in the body economy with its real GPU cost. Before this change, the economy saw co-play sessions but charged them nothing.
- Each co-play handoff gets one stable id (`coplay_turn_id`). Three things carry it: the play act, its shadow bid, and every planner call made during the handoff.
- The act ledger charges an act for the GPU leases that carry the act's id. So each planner call is now charged to the act.
- `hub.retro_coplay` now records the lane it uses: `lanes: [metacog]`, `placement: one_of`. The tier stays at 2.

## Outcome moved

- **Before (live, 2026-10-29):** 32 co-play acts in `grammar_acts`, and all 32 have `gpu_slot_s_by_lane = null`. In the same period, `orion-hub-retro` held 83 metacog leases in 2 hours.
- **After:** a co-play act closes with `gpu_slot_s_by_lane.metacog` set. Once Patch D makes bids real, the market can price long "Let Orion play" sessions.

## Current architecture

- The gateway stamps the request envelope's `correlation_id` on each GPU lease (`turn_correlation_id`).
- `fill_close_cost` in `orion/grammar/act_ledger.py` sums the leases whose `turn_correlation_id` equals the act's `correlation_id`.
- Co-play planner calls used a fresh uuid each time.
- The act's correlation id was the row key `<sid>:<epoch>`. That key is not a UUID, so no envelope could carry it.
- The join therefore never matched.

## Architecture touched

Hub only, plus `config/physiology/families.yaml`. No bus, schema or gateway change.

## Files changed

- `services/orion-hub/scripts/act_emit.py`: `coplay_turn_id(key)` replaces `_coplay_envelope_id`. It returns the same uuid5 as before, now as a string. The act's `correlation_id` and the envelope id both use it.
- `services/orion-hub/scripts/retro_memory_export.py`: the shadow bid's `want_id` is `coplay_turn_id(key)`, so it still equals the act's correlation id.
- `services/orion-hub/scripts/retro_routes.py`: the watcher records each watched handoff's id in `coplay_turns`. `plan_v2` passes that id to the planner.
- `services/orion-hub/scripts/retro_planner.py`: `plan_zelda` and `_gateway_chat` now take `turn_id`, which becomes the envelope correlation. The reply channel and `planner_id` keep a fresh id for each call.
- `config/physiology/families.yaml`: `hub.retro_coplay` now has `lanes: [metacog]` and `placement: one_of`.
- `services/orion-hub/README.md`: the shadow-bid table row now describes the new `want_id`.
- Tests: `tests/retro/test_planner_v2.py`, `tests/retro/test_experience.py`, `tests/test_physiology_shadow_bid.py`, `tests/test_retro_memory_export.py`.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: the `correlation_id` of new `hub.retro_coplay` acts and the `want_id` of new co-play bids are now a UUID instead of the row key. The `act_id` (built from the key) and `test_before` (`retro_play_sessions:<key>`) do not change.
- Compatibility notes: acts opened before the deploy keep the row key. If a handoff is open during the deploy, its close carries the new id while its open carried the old one. `act_id` still pairs the two.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-hub: tests/retro tests/test_physiology_shadow_bid.py tests/test_retro_memory_export.py -> 247 passed
orion/physiology/tests -> 160 passed
services/orion-durable-runs: test_run_bids.py test_physiology_bid.py -> 16 passed, 3 skipped
```

Mutation checks:
- When `plan_v2` stops passing the id, `test_coplay_planner_calls_carry_the_play_act_id` fails.
- When `_gateway_chat` ignores `turn_id`, `test_gateway_chat_sends_the_turn_id_on_a_fresh_reply_channel` fails.

## Evals run

No new eval. `evals/retro_coplay.mjs` covers the handoff path and is unchanged. The live check below is the proof.

## Docker/build/smoke checks

Live check after the deploy, once one co-play handoff has ended:

```sql
select correlation_id, cost_json->'gpu_slot_s_by_lane' from grammar_acts
 where actor='hub.retro_coplay' order by opened_at desc limit 3;
```

Expected result: the newest row has a UUID correlation id and a non-null `metacog` value. Status: UNVERIFIED until a session runs.

## Review findings fixed

- Finding (low): `planner_id` carried only the per-call id. That id appears in no gateway log line and no GPU lease, so a goal could not be traced.
  - Fix: under a turn, `planner_id` is `llm-gateway:<route>:<turn id>:<call id>` (98 characters, Retro's limit is 128).
  - Evidence: `test_gateway_chat_sends_the_turn_id_on_a_fresh_reply_channel` asserts both ids and the length.
- Noted, not changed (low): the gateway's reply-embedding doc id is the envelope correlation, so the planner replies of one handoff now share one vector doc (last write wins). The replies are JSON goals, not prose.
- Noted, not changed (low): planner calls made before the act opens (the policy ends before the first poll, or there is no bus) tag leases that no act claims. This is harmless.
- The reviewer confirmed: the turn id matches end to end; nothing else joins on the old row key (`act_downstream_use` uses `test_before`, and the shadow-vs-real report matches both sides); RPC replies stay unique; `coplay_turns` is cleaned up in the watcher's `finally`. 707 + 247 + 16 tests pass.

## Restart required

Hub rebuild on `git pull` (the post-pull rebuild covers it).

## Risks / concerns

- **Severity: low.** Solo leisure (`hub.retro_leisure`) has the same gap. Its act correlation is the dispatch id, which is not a UUID, so it is still uncharged and still has `lanes: []`. Fixing it means changing the dispatch-linked correlation id. That is left as a follow-up.
- **Severity: low.** Many planner calls now share one envelope correlation id. Chat turns already do this. RPC replies are keyed by reply channel plus id, and the reply channel stays unique for each call.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
