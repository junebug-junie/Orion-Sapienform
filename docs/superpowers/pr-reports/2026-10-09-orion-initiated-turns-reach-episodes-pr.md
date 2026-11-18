## Summary

- **Orion's own messages now reach episode memory.** sql-writer only announced turns that had both a prompt and a response, so every message Orion sent on their own was dropped before memory consolidation. That was 53 of 101 chat turns over 14 days.
- **They never closed a stale conversation.** The 10-07 goodnight waited 23 h for Juniper's next message, though Orion wrote 4 times in between.
- **They were never remembered,** and Juniper's replies were remembered without the message they answered. The episode spec assumed these turns "persist and are classified like any other turn".
- **New contract field** `MemoryTurnPersistedV1.initiated_by`. It is set to `orion` only for real outreach (`client_meta.unsolicited`). Other prompt-less rows, such as Claude speaking in the room, stay out of memory.
- **Episode rules for Orion's turns:**
  - an Orion message closes an open conversation only after 3 h of silence;
  - Juniper's reply **joins** an episode of Orion-only messages when it comes within 24 h, even though her own phase stamp says `long_gap`;
  - an unanswered stretch is skipped (`no_juniper_turn`).
- **Distiller prompt v6** renders these turns as `ORION FIRST`. The coverage metric counts Juniper's turns only.

## Outcome moved

Replay of the last 14 days of live chat through the old and new boundary code (read-only):

```text
before: episodes=16 orion_turns=0  in an episode with Juniper=0
after:  episodes=20 orion_turns=53 in an episode with Juniper=36  orion-only (skipped)=4
```

So 36 of Orion's 53 outreach messages are now distilled together with the reply they got. The other 17 went unanswered for more than 24 h. Stale conversations also close when Orion next writes after 3 h of silence, instead of waiting for Juniper.

## Current architecture

1. sql-writer writes `chat_history_log`, then publishes `orion:memory:turn:persisted`.
2. memory-consolidation classifies each turn, appends it to the legacy windows, and runs the shadow episode tracker (Rule 3).
3. A closed episode is distilled by durable-runs.

Prompt-less turns were filtered out at step 1.

## Architecture touched

- **Contract:** `MemoryTurnPersistedV1` (additive field, `extra="forbid"`).
- **orion-sql-writer:** both emit paths, the envelope path and the row readback.
- **orion-memory-consolidation:** turn routing (Orion turns go to the episode tracker only, skipping the classifier and the legacy windows) and episode boundaries and counting.
- **Shared episode code:** the distill prompt and the coverage metric.

## Files changed

- `orion/schemas/memory_consolidation.py`: the `initiated_by` field and its deploy-order note.
- `services/orion-sql-writer/app/worker.py`: `_memory_turn_initiator()`, used by both emit paths.
- `services/orion-memory-consolidation/app/worker.py`: routes Orion turns to the episode tracker only.
- `services/orion-memory-consolidation/app/episode_shadow.py`:
  - `episode_boundary()` (the Orion close gap, the reply-joins rule, Rule 3 otherwise);
  - Orion turns are not counted as Juniper turns;
  - `no_juniper_turn` skip reason;
  - dedup of Orion turns against recently closed episodes.
- `services/orion-memory-consolidation/app/settings.py`, `.env_example`, `docker-compose.yml`: `MEMORY_EPISODE_ORION_CLOSE_GAP_SEC=10800`, `MEMORY_EPISODE_ORION_REPLY_WINDOW_SEC=86400`.
- `orion/cognition/prompts/memory_episode_distill.j2`, `orion/schemas/memory_episode.py`: prompt v6.
- `orion/memory/episode/validate.py`: coverage counts Juniper turns only.
- Tests:
  - `services/orion-memory-consolidation/tests/test_orion_initiated_turns.py` (new, 10);
  - sql-writer: `test_fetch_chat_turn_for_memory_emit.py` (+3), `test_memory_turn_persisted_once.py` (+2, the envelope path);
  - episode: `test_distill_prompt.py`, `test_validate.py`, plus the v6 bumps.

## Schema / bus / API changes

- Added `MemoryTurnPersistedV1.initiated_by: "juniper" | "orion"`, default `"juniper"`.
- Behavior changed:
  - `orion:memory:turn:persisted` now carries Orion's outreach turns (empty `prompt`);
  - new episode reasons `v2:orion_after_silence`, `v2:orion_joins` and `v2:reply_to_orion`;
  - new skip reason `no_juniper_turn`.
- Compatibility: the model is `extra="forbid"`, so **orion-memory-consolidation must deploy before orion-sql-writer**. The only consumer is memory-consolidation.

## Env/config changes

- Added: `MEMORY_EPISODE_ORION_CLOSE_GAP_SEC=10800`, `MEMORY_EPISODE_ORION_REPLY_WINDOW_SEC=86400` (memory-consolidation).
- Local `.env` synced with `sync_local_env_from_example.py --all-keys orion-memory-consolidation`: both keys added.
- `check_service_env_compose_parity`: both keys are exposed; the pre-existing gaps are unchanged from main.

## Tests run

```text
orion/memory/episode/tests                      294 passed, 22 skipped
orion-memory-consolidation tests                270 passed, 11 skipped
orion-sql-writer tests                          12 failed (all pre-existing; main has 13), rest pass
orion-durable-runs test_episode_distill_graph   9 passed
tests/test_metric_definition_drift.py           56 passed (no lock change)
```

## Evals run

```text
Live 14-day replay (above). Distill eval on v6 not yet run against the model (follow-up, with the v5 one).
```

## Docker/build/smoke checks

```text
Not built from this branch. Deploy from the primary checkout on main after merge, consolidation first.
```

## Review findings fixed

- Finding (HIGH): outreach never joined Juniper's reply. Her reply is phase-stamped from HER last turn (`long_gap`), which closed every Orion-only episode. That was 25 of 25 live outreach messages.
  - Fix: a Juniper turn joins an episode of Orion-only messages within 24 h (`v2:reply_to_orion`).
  - Evidence: `test_juniper_reply_joins_the_outreach_it_answers_despite_her_long_gap_phase`; the replay shows 36 of 53.
- Finding (MEDIUM): every prompt-less row was tagged Orion, including 17 rows of Claude speaking in the room (`client_meta.room_claude`).
  - Fix: `orion` only when `client_meta.unsolicited` is true; other prompt-less rows are still dropped.
  - Evidence: `test_promptless_row_that_is_not_outreach_stays_out`, `test_other_promptless_turn_envelope_is_not_memory`.
- Finding (LOW): outreach at a 105-minute gap would split a conversation Juniper then resumed.
  - Fix: an Orion turn closes only after 3 h (`MEMORY_EPISODE_ORION_CLOSE_GAP_SEC`).
  - Evidence: `test_orion_message_soon_after_juniper_joins_her_conversation`.
- Finding (LOW): a late duplicate Orion turn could land in a second episode.
  - Fix: Orion turns are deduped against episodes closed in the last 3 days.
- Finding (LOW): the envelope emit path was untested.
  - Fix: 2 envelope-path tests.

## Restart required

After merge, from the primary checkout on main, consolidation first:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && for s in orion-memory-consolidation orion-sql-writer orion-durable-runs; do docker compose --env-file .env --env-file services/$s/.env -f services/$s/docker-compose.yml up -d --build; done
```

durable-runs picks up the v6 distill prompt.

## Risks / concerns

- **Medium: more episodes, and more distill runs.**
  - Concern: 20 episodes instead of 16 over 14 days, so about 2 more distill runs a week.
  - Mitigation: the 4 Orion-only episodes are skipped and never distilled.
- **Low: an unrelated reply joins an outreach episode.**
  - Concern: if Juniper writes about something else within 24 h of an unanswered outreach, it joins that episode.
  - Mitigation: the distiller still splits memories by claim, and the episode is merely wider.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
