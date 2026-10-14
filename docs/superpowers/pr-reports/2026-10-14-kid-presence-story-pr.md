# Orion tells a whole story to a child who is listening

## Summary

- Relational turns may now narrate the people a story is about. The old single-reply rule banned "narration attributed to the other person", which also banned telling a story about a present child.
- Presence (who is in the room) can no longer crash and wipe the situation brief. One normalizer is shared by the Hub save path, the brief builder and the cache key.
- The prompt now names who is present. When a child is present, a protected caution asks for child-level language and a story with a real ending.
- Memory recall for the chat answer now pulls older turns about people named in the message. Each turn is cut to the part that mentions them.
- The Hub presence form uses labeled dropdowns. It used to be five unlabeled text boxes.

## Outcome moved

Incident corr `21c8a610-f852-4af3-a855-c337a30486da` (2026-10-14). Juniper asked for a space story for herself and her 7-year-old son, with presence set to show him in the room. Orion returned a ~2,000-character adult vignette that never gave the child a line and stopped at lift-off.

The raw FCC transcript shows `stop_reason=end_turn` after 450 output tokens, so no length limit was hit. The prompt caused it, in four ways:

1. **The reply rule.** "Never write dialogue, narration ... attributed to the other person" applied to the child the story was about. The text visibly dodges it: *"they would have said, if spacewalks came with subtitles."* The turn was also labeled "Relational/minimal ... Acknowledge and stay present".
2. **Presence crashed.** The child's name was saved into `relationship` and `age_band`. `PresenceContextV1` raised, and the hub logged `unified_turn_situation_context_failed` and dropped the whole situation brief. Even on a working turn, the prompt only said `audience_mode=kid_present` and never named anyone.
3. **Recall missed who he is.** Phase-3 purposeful recall (intent `open_loop`) never ran entity injection, so it returned three unrelated `active_packet` lines. The later reflect recall for the same turn did find the "my son ... 7th birthday" turn.

## Current architecture

- `orion/harness/operator_brief.py` adds `HARNESS_RELATIONAL_TOOL_DISCIPLINE` to every relational FCC motor turn.
- Hub stores presence unvalidated (`POST /api/presence`). `orion/situational/context.py::_presence_from_ctx` builds strict `PresenceContextV1`. Any failure drops the situation brief (`turn_orchestrator._build_situation_prompt_fragment`).
- `services/orion-recall/app/worker.py::process_recall` ran `_compute_entity_relatedness_boost_map` only in its default branch, not in the PCR continuity/purposeful branches.

## Architecture touched

Harness motor instructions, situation brief (shared `orion/situational`), Hub presence API and UI, and orion-recall PCR belief fusion. No bus, schema or env changes.

## Files changed

- `orion/harness/operator_brief.py`: the relational rule keeps "exactly one reply" and "never invent the other person's next message". It now allows story characters, dialogue and endings, and drops the "minimal / acknowledge" framing.
- `orion/situational/presence.py` (new): `normalize_presence_context` (bad values fall back to defaults; `audience_mode` is derived from companions) and a bounded `companions_phrase`.
- `orion/situational/context.py`: `_presence_from_ctx` and `_presence_cache_fingerprint` use the normalizer (the fingerprint now includes `context_note`). The fragment gains a "With Juniper right now: ..." line and a child caution, kept whole in priority order.
- `services/orion-hub/scripts/api_routes.py`: `POST /api/presence` stores normalized presence.
- `services/orion-hub/static/js/presence-form.js` (new) and `app.js`, `templates/index.html`: labeled Name and Note fields; dropdowns for who they are, age and doing; the audience-mode select is removed (now derived).
- `services/orion-recall/app/worker.py`: entity injection runs for purposeful intents whose plan has `rdf_chat` (relational, semantic, open_loop), under the existing deadline. Injected turns carry `meta.entity_terms`.
- `services/orion-recall/app/fusion.py`: up to 4 reserved belief slots for injected turns ahead of `active_packet`. Each is cut to a window around the matched name, and together they take half the budget.
- Tests: `orion/harness/tests/test_harness_prefix.py`, `orion/situational/tests/test_presence_normalize.py`, `services/orion-hub/tests/test_presence_api.py`, `services/orion-hub/static/js/presence-form.test.js`, `services/orion-recall/tests/test_pcr_purposeful_entity_injection.py`.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed:
  - `POST /api/presence` returns and stores normalized presence. `audience_mode` is derived when companions are given.
  - The recall PCR purposeful digest can now contain `falkor_chat` entity turns.
- Compatibility notes: the presence dict shape is unchanged. Invalid values that used to crash a turn now fall back to defaults.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change).
- skipped keys requiring operator action: none.

## Tests run

```text
orion/situational/tests + orion/harness/tests: 656 passed (after merging origin/main)
services/orion-recall/tests: 374 passed, 2 failed; the same 2 fail on main
  (test_recall_policy_harness::...gating_suppression_and_selection,
   test_recall_vector_amputation::...without_vector_adapter)
services/orion-hub/tests: 36 failed / 3573 passed on the branch vs 43 failed on main;
  the branch's failures are a subset of main's except
  test_memory_graph_routes::test_memory_graph_validate_fixture_roundtrip,
  which also fails on pristine origin/main (d2ce12b85), so not this branch
services/orion-cortex-exec presence/stance/situation tests: pass
node --test presence-form.test.js: 6 passed
New regression tests fail on the old code: incident presence payload
(raised ValidationError live), relational instruction wording, recall
injection count (0 before).
```

## Evals run

```text
No eval harness covers story completeness or presence. Gap: add an eval that
replays a child-present story request and checks for a resolved ending.
Follow-up.
```

## Docker/build/smoke checks

```text
Live read-only recall smoke: a throwaway orion-recall-recall container on
app-net with the live recall env, running the exact incident message
(open_loop, purposeful).
  main:   falkor_chat_entity_injected absent; digest = header only
  branch: falkor_chat_entity_injected=4; digest leads with
          "... We celebrated my son, <child>'s, 7th birthday ..." and
          "... My son <child> was losing his mind ..."
End-to-end turn after deploy: UNVERIFIED (not deployed).
```

## Review findings fixed

- Finding: new presence body lines could push the safety cautions out of the 1200-char fragment.
  - Fix: the child guidance is now a caution (kept whole, in priority order); `companions_phrase` is capped at 4 people, 40-char notes and 240 chars.
  - Evidence: `test_cautions_survive_the_cap_with_a_child_and_a_crowded_body`.
- Finding: `context_note` is rendered but was not in the cache fingerprint.
  - Fix: added to the fingerprint.
  - Evidence: `test_note_change_busts_the_situation_cache`.
- Finding: the new form had no note field, so Save erased existing notes.
  - Fix: labeled, escaped Note field.
  - Evidence: `a saved note is shown and survives the read-back` (node).

## Restart required

Deploy from the clean primary `main` checkout after merge:

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-recall up -d --build
```

## Risks / concerns

- Severity: medium.
  - Concern: recall latency. Purposeful recall for relational, semantic and open_loop now makes up to 4 Falkor round trips plus one Postgres lookup.
  - Mitigation: bounded by the existing recall deadline; on a timeout, injection is skipped.
- Severity: low.
  - Concern: a frequent first name can pull up to 4 loosely related turns into the belief digest.
  - Mitigation: the existing degree discount; the turns are capped to half the belief budget.
- Severity: low.
  - Concern: the relational rule now permits narrating present people. The original fabricated-dialogue incident (610c51a60) was about inventing the partner's replies, which is still banned.
  - Mitigation: the regression test keeps both halves of the rule.

## PR link

(filled in on the PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
