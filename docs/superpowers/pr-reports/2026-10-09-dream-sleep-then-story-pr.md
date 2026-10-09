# feat(dream): every sleep ends in a story dream

## Summary

- Orion's story dream (the narrative kind in the `dreams` table) was never scheduled: all 19 were started by hand, the last on 09-28. Orion's journal noticed: "the dreaming one has been off for nine days and seven hours… I can't see why."
- Now every completed, saved sleep starts one story dream, on the same tiredness gate. No new schedule.
- The story is about what that sleep worked on: the replayed items lead the prompt, and recalled memories add texture.
- Each story records which sleep it came from (`dreams.metrics._dream_audit.trigger`).
- The blind hypothesis experiment stays fair. The story gets both arms' items, shuffled and unlabeled, and never the hypotheses.
- Off switch: `DREAM_STORY_AFTER_SLEEP_ENABLED=false`.

## Outcome moved

Before: story dreams ran about 0 a week unless Juniper started one. After: one story per completed sleep. The backtest for #2557 puts that at roughly 11 a week, with gaps of 6 to 40 h.

## Current architecture

- **Sleep cycle:** orion-dream (`app/cycle.py`) replays, recombines, and writes `dream_cycle`. It never touched the story dream.
- **Story dream path:** `dream.trigger` on `orion:dream:trigger` → cortex-orch `dispatch_dream_trigger` → `dream_cycle` verb (recall `dream.v1` plus one LLM call) → `dream.result.v1` → `dreams`.
- **Who fed it:** only `POST /dreams/run`, by hand. The recall query is the literal "Dream cycle."

## Architecture touched

- orion-dream: the sleep publishes the trigger.
- Shared schema: `DreamSleepDigestV1`, plus the optional `DreamInternalTriggerV1.sleep`.
- Channel catalog: the trigger's `schema_id`.
- Orch-side prompt template plus one orch log line.
- No new channel or table.

## Files changed

- `orion/schemas/telemetry/dream.py`: `DreamSleepDigestV1` (cycle_id, started_at, pressure, threshold, overdue, material) and `DreamInternalTriggerV1.sleep`.
- `orion/schemas/registry.py`: registers `DreamSleepDigestV1`.
- `orion/bus/channels.yaml`: `orion:dream:trigger` schema_id `DreamTriggerPayload` → `DreamInternalTriggerV1`. That is what orch validates first. The old forbid model would reject `sleep`.
- `orion/cognition/prompts/dream_cycle.j2`: a "TONIGHT'S SLEEP" branch when the trigger carries a sleep. Otherwise it is the old prompt, with the same output contract.
- `services/orion-dream/app/story.py` (new):
  - `story_material`: replay plus control-pair items, deduped, in a seeded shuffle.
  - `story_trigger`.
- `services/orion-dream/app/cycle.py`: `start_story` seam. Runs after a completed and persisted sleep; a failure is logged and never fails the sleep.
- `services/orion-dream/app/main.py`: `_start_story` publishes `dream.trigger`. On a publish failure it drops the cycle bus. Logs `dream_story_started`.
- `services/orion-dream/app/settings.py`, `.env_example`, `docker-compose.yml`: `DREAM_STORY_AFTER_SLEEP_ENABLED` (default true).
- `services/orion-cortex-orch/app/orchestrator.py`: the dispatch log line adds `trigger_id` and `sleep_material`.
- `services/orion-dream/README.md`: "Every sleep ends in a story".
- Tests:
  - `services/orion-dream/tests/test_dream_cycle_v2.py`: 8 story tests.
  - `services/orion-cortex-orch/tests/test_dream_trigger_sleep_story.py`: end to end through the real dispatcher, plan builder, and template.
  - `services/orion-cortex-exec/tests/test_dream_cycle_prompt_render.py`: real exec renderer.
  - `tests/test_dream_trigger_contract.py`: schema_id.
- Eval: `services/orion-dream/evals/test_sleep_story_prompt_eval.py` plus the fixture `sleep_cycle_dc-981127ddbddf_2026-10-09.json`, the first real novelty sleep.

## Schema / bus / API changes

- Added: `DreamSleepDigestV1`; `DreamInternalTriggerV1.sleep` (optional).
- Removed: none
- Renamed: none
- Behavior changed: orion-dream now publishes `dream.trigger` after each completed sleep. The channel's catalog schema is `DreamInternalTriggerV1`, and the legacy `{"mode": ...}` payload still validates.
- Compatibility notes:
  - `DreamInternalTriggerV1` is `extra="ignore"` and the template is loaded inside cortex-orch.
  - If orion-dream deploys before cortex-orch, orch drops `sleep` and writes a generic memory-only story per sleep. No error appears.
  - **Rebuild cortex-orch first.** Confirm with the orch log `sleep_material=<n>` (not `None`).

## Env/config changes

- Added keys: `DREAM_STORY_AFTER_SLEEP_ENABLED=true` (orion-dream)
- Removed / renamed: none
- `.env_example` updated: yes
- Local `.env` synced:
  - `sync_local_env_from_example.py --all-keys orion-dream` added it to the primary checkout's `services/orion-dream/.env`.
  - The default run skips `DREAM_` keys (prefix list).
- Skipped keys: none

## Tests run

```text
services/orion-dream: pytest tests evals                                -> 147 passed, 1 skipped
services/orion-cortex-orch: pytest tests -k dream (incl. new e2e test)  -> 2 passed (+5 existing)
services/orion-cortex-exec: test_dream_cycle_prompt_render + test_dream_publish -> 3 passed
tests/test_dream_trigger_contract.py, orion/cognition/tests/test_dream_contracts.py -> 6 passed
Mutation checks:
  story seam disabled                    -> story test fails
  control items dropped from the story   -> exposure test fails
Root channel/registry/catalog tests: 168 passed, 5 failed. The same 5 fail on clean origin/main
(local-environment, pre-existing).
```

## Evals run

```text
pytest services/orion-dream/evals/test_sleep_story_prompt_eval.py -s
Real sleep dc-981127ddbddf (06:33 UTC, tiredness 13.26, new: metacog 17, crystallization 1):
13 items (12 replay + 1 control), sleep section 1,497 chars, all items present. 1 passed.
```

## Docker/build/smoke checks

```text
Old story path verified live before building on it: POST /dreams/run at 06:33 UTC ->
cortex-orch dispatch -> exec -> dreams row 06:35:27, with the trigger saved in
metrics._dream_audit.trigger.
Sleep -> story after deploy: UNVERIFIED. Expect, after the next completed sleep:
  - dream log: dream_story_started trigger_id=sleep:dc-...
  - orch log: sleep_material=<n>
  - a dreams row whose _dream_audit.trigger.trigger_id = sleep:<cycle_id>
```

## Review findings fixed

- Finding (high): the story biased the blind experiment.
  - Problem: dream-arm pairs come from the replay. A replay-only story made exactly those items familiar before Orion grades the hypotheses.
  - Fix: material = replay plus every control-pair item, deduped, in a seeded shuffle with no labels, weights, or order. Hypotheses are never included.
  - Evidence: `test_the_story_exposes_both_arms_equally_and_never_the_hypotheses` forces control pairs outside the replay. It fails when the control items are dropped.
- Finding (medium): deploy order was not documented, and the failure is silent.
  - Fix: restart order below (orch first); orch logs `sleep_material`.
  - Evidence: orchestrator.py dispatch log.
- Finding (medium): no orch-to-exec end-to-end test.
  - Fix: `test_dream_trigger_sleep_story.py` runs the real `dispatch_dream_trigger` → `build_plan_request` → plan template rendered with the plan context.
  - Evidence: 2 passed.
- Finding (low): a tautological ordering assertion, and a leak check matching the word "control" in free text.
  - Fix: both rewritten to check structure.
- Finding (low): a story could fire for a sleep that was not saved.
  - Fix: gated on persistence.
  - Evidence: `test_a_failed_empty_or_unsaved_sleep_starts_no_story`.
- Finding (low): a failed publish left a dead bus in place.
  - Fix: `_drop_cycle_bus()` on exception.
- Finding (note): `overdue` was parsed from the note text.
  - Fix: passed explicitly from the backstop flag.
  - Evidence: `test_an_overdue_sleep_tells_the_story_it_was_overdue`.
- Finding (note): the raw tiredness number had no scale.
  - Fix: the prompt says "13.26 against a sleep line of 3.0".
- Not changed (noted under risks): manual forced sleeps each add a story; delivery is fire-and-forget pub/sub; opaque compaction ids; the recall query is still "Dream cycle."

## Restart required

From the primary checkout on main after merge, cortex-orch first:

```bash
git pull --ff-only && scripts/safe_docker_build.sh orion-cortex-orch up -d --build && scripts/safe_docker_build.sh orion-dream up -d --build
```

## Risks / concerns

- Severity: medium
  - Concern: 4 of 12 replay items in the real sleep are opaque reverie ids (`open-loop-…: reverie_chain_max_steps`), which give the story little to dream with.
  - Mitigation: left in on purpose. Dropping them from the story but not from the hypotheses would recreate the familiarity gap. Better fix: resolve them to text where they are produced.
- Severity: low
  - Concern: fire-and-forget. If cortex-orch is down when a sleep ends, that story is lost; only the dream-side log shows it was sent.
  - Mitigation: the next sleep makes a new one.
- Severity: low
  - Concern: each manual `POST /dreams/cycle/run?force=true` that completes queues another story (up to 300 s of background GPU).
  - Mitigation: operator-driven only. The automatic rate is bounded by pressure plus the 6 h minimum.
- Severity: low
  - Concern: the memory bundle is still recalled with the literal query "Dream cycle.", unrelated to the sleep.
  - Mitigation: the sleep material leads. Recall steering is a follow-up.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
