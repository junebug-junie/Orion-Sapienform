# feat(memory): voice renderer + intent fix

Memory Stage 2, PR D (design: `docs/superpowers/specs/2026-10-06-memory-stage2-referent-graph-design.md`, PR #2496, section 4.2 and row D of 7.5; voice contract: rev 3 section 7). This PR does two things. It makes Orion's remembered items say whose thought they are. And it stops chat recall from picking the same "open loop" mode on every single turn.

## Summary

- **One voice renderer, `orion/memory/voice_render.py`.** It turns a remembered item into the line Orion reads: "Juniper told me (10-04): …", "I told Juniper …", or "Something I was turning over on my own (reverie, 10-04), not something Juniper and I discussed: …". It is a pure function with two live callers in this PR.
- **A reverie can never be rendered as something Juniper said.** Anything on an internal channel (reverie, curiosity, dream, journal, topic_model) renders as Orion's own private thought, whatever voice it carries. "Juniper told me" needs a `juniper_said` chat memory *and* a verified quote from one of her prompts. Every other mismatch falls back to "My own note…, not Juniper's words". It never falls toward Juniper.
- **Live caller 1, the chat stance's reverie glimpse.** Today the latest reverie reaches the stance prompt as a bare `reverie_glimpse:` line, with nothing saying it was Orion's own idea. It now arrives labelled. Checked on the 20 newest real reveries: 20 of 20 render as Orion's own thought.
- **Live caller 2, the daily episode report.** Each memory now shows exactly the line Orion would read. The old label map (`_VOICE_LABEL`) is deleted, not kept alongside.
- **The intent fix.** Purposeful recall used `open_loop` for 630 of 630 chat turns in the last 7 days. Any loop in the attention frame forced it, and live chat frames always carry several substrate "prediction error" loops. Now the intent comes from model judgments (stance, turn-change appraisal, and the same-turn LLM's typed reading of the message) and explicit ids. There are no word lists. The old capitalized-word regex and the "plan/step/…" substring list are removed.

## Outcome moved

- **Source monitoring on the live path.** Before: the stance prompt received reverie text unlabelled. After: every reverie is labelled as Orion's own thought, "not something Juniper and I discussed" (20/20 real reveries). Run through the renderer, the 31 live episode memories give 27 "Juniper told me" (all 27 have a verified prompt quote) and 4 "I told Juniper".
- **Intent before (live, read-only `recall_telemetry`, 7 days):** `chat.belief.open_loop.v1` 630; every other `chat.belief.*` profile 0. The cortex log confirms `rule_id=open_loops_present` on the turns still in the log buffer.
- **Intent after: UNVERIFIED live** (not deployed, per instruction). Offline evidence:
  - The per-turn inputs are mostly not persisted (the stance brief and attention frame are never stored), so a full replay is impossible.
  - The one stored model signal, the turn-change appraisal in `chat_history_log.spark_meta`, covers 44 of the phase-3 turns over 30 days. It says TOPIC shift ≥ 0.35 on 15 and STANCE shift on 3. Under the old rule all 44 were `open_loop`; under the new rules those 18 alone become `semantic` or `relational`.
  - Caveat: nothing in cortex-exec puts that appraisal into the turn context before phase 3 runs (it is computed after the turn). The shift rules may therefore rarely fire live. The live variation is expected to come from the stance fields and the same-turn LLM signals, which are present on human turns (cortex log: `current_turn_llm_read … items=1`).
- **Post-deploy check (the spec's acceptance, "≥ 3 intents over 7 days"):**
  ```sql
  SELECT profile, count(*) FROM recall_telemetry
  WHERE created_at > now() - interval '7 days' AND profile LIKE 'chat.%' GROUP BY 1 ORDER BY 2 DESC;
  ```
  Turns whose intent is `continuity` skip phase 3, so they show only `chat.continuity.v1`. Per-turn `rule_id` is in the cortex log line `pcr_phase3_* … rule_id=`.

## Current architecture

- `derive_retrieval_intent` checked `attention_frame.open_loops` first, and returned `open_loop` if the list was non-empty.
- After that it used a capitalized-word regex (`entity_query`, which matches the first word of almost any sentence) and a substring list over `response_priorities`.
- The procedural rule needed `task_mode == "instrumental"`, which is not one of the stance template's `task_mode` literals.
- `voice_render.py` did not exist. The reverie glimpse was passed through raw. The episode report used its own label dict.

## Architecture touched

- `orion/memory/voice_render.py` (new): `VoicedMemory`, `render_memory`, `speaker`. It reuses the Stage 1 validator's `INTERNAL_CHANNELS`, so there is one definition of "internal".
- `orion/memory/retrieval_intent.py`, rule order:
  - skip;
  - relational stance;
  - STANCE shift;
  - TOPIC shift;
  - REPAIR shift (now the only `open_loop` trigger);
  - contradiction seed;
  - procedural stance (`conversation_frame=planning` or `task_mode=technical_collaboration`);
  - the turn-signal referent rule (person → relational, plan → procedural, anything else named → semantic);
  - brain-lane default;
  - continuity.
- `services/orion-cortex-exec/app/pcr_chat_memory.py`: passes `ctx["current_turn_llm_signals"]` instead of the raw user message.
- `services/orion-cortex-exec/app/chat_stance.py`: the reverie glimpse goes through the renderer.
- `orion/memory/episode/report.py`: renders each memory through the renderer. Its query adds `remembered_at` and a `has_verified_juniper_quote` EXISTS check.

## Concepts (producer → consumer → test)

| Concept | Producer | Consumer | Test |
|---|---|---|---|
| `render_memory` / `VoicedMemory` / `speaker` | reverie glimpse; episode report rows | stance prompt (`chat_stance_brief.j2` `reverie_glimpse`); daily report file | `test_voice_render.py` (672-case matrix + contract rows), glimpse tests, PG report test |
| "Juniper told me" (`juniper`) | live distiller `juniper_said`/chat with verified prompt quote (27 rows) | report | matrix, `test_juniper_said_needs_a_verified_prompt_quote` |
| "Juniper and I worked out" (`together`) | validator keeps `worked_out_together` when prompt and response quotes both verify | report | matrix |
| "I told Juniper" (`orion_to_juniper`) | live `orion_thought`/chat (4 rows) | report | matrix |
| "…on my own…, not something Juniper and I discussed" (`orion_private`) | every reverie glimpse; any internal-channel memory | stance prompt; report | matrix, `test_the_180_hecate_reveries_case`, `test_glimpse_is_never_presented_as_juniper_or_as_shared` |
| "My own note…, not Juniper's words" (`orion_note`) | any other mismatch (e.g. `juniper_said` without a verified quote) | report | PG report test (unverified row) |
| "I read" / "From my own code and docs" | validator keeps `orion_read` / `orion_self_knowledge` with response evidence | report | `test_contract_table_rows` |
| "Unconfirmed, check with Juniper if natural:" | validator sets `pending_confirmation` for high stakes | report | `test_pending_marker` |
| Intent rule ids (`relational_mode`, `stance_shift`, `topic_shift`, `repair_shift`, `contradiction_seed`, `procedural_mode`, `brain_lane_belief_default`, `continuity_only`) | `derive_retrieval_intent` | `run_pcr_phase3` → `chat.belief.<intent>.v1` or skip; `rule_id` in recall `task_hints`, log line, `ctx.debug.pcr` | `tests/test_retrieval_intent.py` (22), `test_phase3_profile_follows_the_turn_not_the_frame` |
| `turn_names_person` / `turn_names_plan` / `turn_names_topic` (new) | same-turn LLM signals (`current_turn_llm_signals.py`) | same as above | `test_turn_signal_referent_rule`, caller test |

The same table is in `orion/memory/README.md` (new) with a plain-English column, and `services/orion-cortex-exec/README.md` points to it.

**Cut on purpose (no producer or no consumer in this PR):**
- **The intent → memory-kind table.** Live PCR does not retrieve episode memories at all, so wiring the table in means building recall-by-referent. That is PR F.
- **`MemoryItemV1` voice/`recall_reason` fields.** Their consumer is recall's render, which is PR F.
- **The renderer's reading title and claim status, graphify build date, and "faded" suffix.** Nothing produces them today; `status=faded` has no writer.
- **The open-loop rules for "a loop that persists across turns" and "an open follow-up whose referents appear in the turn".** No in-turn input exists for either until PR F.

**How intent avoids a word list:** each input is a model's own closed-vocabulary output or an id:
- stance `task_mode`/`conversation_frame` literals from `chat_stance_brief.j2`;
- turn-change `shift_kind` from the LLM classifier in `classify.py`;
- the same-turn LLM's `type` field (`person|place|plan|belief|concept|activity|other`, `_ALLOWED_TYPES` in `current_turn_llm_signals.py`);
- seed and contradiction ids.

No rule reads the message text.

## Files changed

- `orion/memory/voice_render.py`: new renderer.
- `orion/memory/retrieval_intent.py`: intent rules from model signals; `open_loops_present`, `entity_query` and the priority word list removed.
- `orion/memory/episode/report.py`: renders through `voice_render`; `_VOICE_LABEL` deleted; query adds `remembered_at` and `has_verified_juniper_quote`.
- `services/orion-cortex-exec/app/chat_stance.py`: reverie glimpse rendered as Orion's own thought.
- `services/orion-cortex-exec/app/pcr_chat_memory.py`: passes `current_turn_llm_signals`.
- `orion/memory/tests/test_voice_render.py` (new), `tests/test_retrieval_intent.py`, `services/orion-cortex-exec/tests/test_pcr_chat_memory.py`, `services/orion-cortex-exec/tests/test_chat_stance_reverie_glimpse_projection.py`, `services/orion-memory-consolidation/tests/test_episode_report_pg.py`: tests.
- `.github/workflows/memory-voice-intent-tests.yml` (new): runs the renderer, intent and cortex caller tests. Nothing in CI ran `tests/test_retrieval_intent.py` or the cortex PCR tests before.
- `orion/memory/README.md` (new), `services/orion-cortex-exec/README.md`: concepts.

## Schema / bus / API changes

- Added: none on the bus or in the registry. `derive_retrieval_intent` takes `turn_signals` instead of `user_message` (one in-repo caller, updated).
- Removed: rule ids `open_loops_present` and `entity_query`.
- Behavior changed:
  - PCR phase 3 profile selection, as above.
  - The stance prompt's `reverie_glimpse` text now carries a voice label.
  - The episode report's memory lines changed format.
- Compatibility: `RetrievalIntentV1` values and `PROFILE_FOR_INTENT` are unchanged.

## Env/config changes

- None. No `.env_example` touched, so no `.env` sync was needed. No new flags. Rollback is a revert; the old rule was the bug.

## Tests run

```text
renderer + intent (PYTHONPATH=.):                       699 passed
orion/memory/episode/tests + orion/memory/tests + orion-memory-consolidation tests + evals,
  against a throwaway postgres:16 (ORION_MEMORY_EPISODE_TEST_DATABASE_URL set):  1145 passed, 0 skipped
orion-cortex-exec PCR + glimpse + grounding + retrieval-query callers:          49 passed
new CI workflow's dependency set, verified in a clean venv:                      699 + 21 passed
orion-cortex-exec full suite: 1084 passed; failure set equals origin/main 029322db2's except
  2 situation-freshness tests (test_fresh_capture_is_available, test_fresh_percept_is_available) that
  fail only in full-suite order and pass alone; they touch nothing changed here
mutation check: restoring "any frame loop -> open_loop" fails 4 caller tests and 5 intent tests
static gates (all 25 run steps of orion-static-gates.yml, incl. check_definition_drift --gate): 25/25 PASS
pyflakes on touched files: clean; git diff --check: clean
```

## Evals run

```text
No eval harness exists for PCR intent selection. Live/offline checks instead:
- 31 live episode memories through the renderer: 27 "Juniper told me" (27/27 with a verified prompt quote), 4 "I told Juniper"
- 20 newest live reverie thoughts through the real glimpse projection: 20/20 "on my own ... not something Juniper and I discussed"
- recall_telemetry 7 d before: chat.belief.open_loop.v1 = 630 of 630 purposeful recalls
Follow-up: the spec's eval 4 (source monitoring over 7 days of shadow recall) belongs to PR F's evals/referent/.
```

## Docker/build/smoke checks

```text
Not deployed (instruction). No dependency changes. Throwaway postgres:16 container for the PG-backed tests.
Production was only read (read-only transactions on recall_telemetry, episode_memory*, substrate_reverie_thought,
chat_history_log, attention_salience_trace; docker logs).
```

## Review findings fixed

Review subagent not run (instruction). Self-found during the work:
- Finding: the procedural rule required `task_mode == "instrumental"`, which the stance template never emits.
  - Fix: use the template's own literals (`conversation_frame=planning`, `task_mode=technical_collaboration`).
  - Evidence: `test_derive_retrieval_intent_rules` procedural rows.
- Finding: a `juniper_said` chat memory without a verified quote would have read "on my own… not something we discussed", which is false for a chat item.
  - Fix: a separate `orion_note` fallback ("My own note…, not Juniper's words").
  - Evidence: PG report test.
- Finding: the new caller test passed alone but failed in the full suite, because other tests re-import `app.*`.
  - Fix: patch `run_pcr_phase3.__globals__` directly.
  - Evidence: full-suite run.

## Restart required

After merge, from the primary checkout on main (one line each):

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-cortex-exec up -d --build
cd /mnt/scripts/Orion-Sapienform && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-memory-consolidation up -d --build
```

## Risks / concerns

- Severity: medium. Concern: fewer purposeful recalls. A turn with no relational/procedural stance, no shift, no seed and no typed signal now gets `continuity` and skips phase 3. Before, every such turn ran an `open_loop` belief recall (p50 ~1.2 s). Orion's own turns never have same-turn signals. Mitigation: this is the designed behavior that the open-loop flood was masking. Watch `chat.belief.*` volume in `recall_telemetry` after deploy. Restoring the old rule is a revert.
- Severity: medium. Concern: the intent mix after deploy is unknown. The stance brief is not persisted, so `relational_mode` could dominate as `open_loop` did. Mitigation: the SQL above. If one intent dominates, the next step is persisting the stance fields per turn, not a word list.
- Severity: low. Concern: the reverie glimpse wording changes the stance prompt. Mitigation: the statement text is unchanged and only a label is added.
- UNVERIFIED: the live intent distribution; whether `turn_change_appraisal` is ever present in the context at phase-3 time (no producer found in cortex-exec).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2521

🤖 Generated with [Claude Code](https://claude.com/claude-code)
