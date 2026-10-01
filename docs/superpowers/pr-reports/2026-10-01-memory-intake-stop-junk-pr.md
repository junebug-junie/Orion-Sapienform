# fix(memory): stop saving greetings and commands as memories (redesign Stage 0A)

Part A of Stage 0 of the memory redesign (`docs/superpowers/specs/2026-09-30-memory-episode-redesign-design.md`, PR #2440, approved by Juniper 2026-10-01). It stops junk entering memory. It does not change how real memories are written; that is Stage 1. The `daily_metacog_v1` fix is Stage 0 part B and is not in this PR.

## Summary

- **"Repair" now means real repair.** The Hub marked almost every chat turn as a repair, because the check was "did a repair appraisal run?" instead of "was there repair pressure?". It now only marks a repair when the pressure is high enough to change how Orion answers. That threshold is 0.45, the level the repair contract already uses to switch its reply mode.
- **Greetings and commands are judged on Juniper's words alone.** Before, the gate judged her prompt together with Orion's reply. Orion's reply is never small talk, so "sup" and "Run github compactor." passed as real content. Those checks now run first, before the repair shortcut. The command list comes from the Hub's real workflow registry, not a hand-written list.
- **A kept window is summarised by its real content.** A window like "Headed to Austin…" followed by "sup yo" was saved under the summary "sup yo". The summary is now the last prompt that isn't a greeting or a command.
- **Auto-saves leave an audit trail.** Every memory the policy saves on its own now gets an `auto_activate` history row (actor `system:formation_policy`). The reverie image seed still ignores these rows, and a test pins that.
- **No more "Juniper approved" for auto-saved rows.** The self-study source text, the curiosity menu cards and the curiosity journal footer now say "auto-saved by policy, not reviewed by Juniper" They say "approved by Juniper" only when a real `op='approve'` history row exists (37 rows live).
- **Graphiti writes can work, and a failure is visible.** The Hub uses host networking, so it cannot resolve the adapter's container name. It now uses `http://127.0.0.1:8640`. A failed write shows up in the approve response, the sync response and a red Hub status line, not only in a warning log.

## Outcome moved

Replay of 30 days of real chat intake windows (`services/orion-memory-consolidation/evals/run_intake_gate_replay_eval.py`):

| | Before | After |
|---|---|---|
| Windows admitted | 107 of 107 | 101 of 107 |
| Rows summarised by a greeting or command | 23 | 0 |
| Turns flagged as repair | 75.7% | 4.1% |
| Austin and offsite (judged on real text) | kept | kept |
| Labs and family (redacted in the fixture) | kept | kept, but see below |
| 13 synthetic must-keep messages | — | 13 of 13 kept |

- **The 6 dropped windows** are all commands and greetings: "Run github compactor." ×6 across windows, "Compact the last 24 hours of chat into a memory digest." ×3, "Do a journal pass." ×2, "ty!", "sup". The same result comes from the real, unredacted text of the same 107 windows.
- **What the named keepers prove.** Only Austin and offsite are checked against their real text. Labs and family are stored as redacted placeholders, which the rule can never call junk. In the fixture they only prove the window was not dropped for some other reason.

  The real protection against dropping messages like theirs is a synthetic, non-private set the eval judges on real text every run:
  - non-Latin scripts: Cyrillic, Japanese, Hebrew, accented Spanish;
  - short negations: "I'm not ok", "not good", "rough day";
  - a command followed by real content: "Do a journal pass about my labs", "run a self review on my divorce";
  - short questions about real things: "when is the surgery?", "where is mom?", "who is Sarah?";
  - a greeting in front of real news: "hi, my son was diagnosed today".

  All 13 are kept.
- **The spec's live check of "0 new active rows under 40 characters"** would still not pass: 13 kept rows have a short summary, for example "sleepy", "meow", "another test", "I've got the blues.", "hey, which queue?". The rule keeps short statements and any question with a real topic in it, following Juniper's "over-index on remembering". If Juniper wants these gone, Stage 1's writer is the place to do it.
- **The novelty and significance floors no longer decide anything.** After the junk check, every window with at least one non-junk prompt is proposed. The floors in `consolidation_gate.py` only choose which reason is recorded on the row. Whether a real window deserves a memory is now the Stage 1 writer's job; a note to that effect is in the gate code and in the spec's Stage 1 section.
- The spec's ~96% repair figure counted grammar atoms. The replay's 75.7% counts turns that have an appraisal row. Both measure the same defect.

## Current architecture

- **The repair atom.** `orion/hub/turn_orchestrator.py` (unified turn) and `services/orion-hub/scripts/websocket_handler.py` (classic path) emitted the `repair_signal` grammar atom whenever an appraisal bundle existed. Memory consolidation (`orion/memory/consolidation_grammar.py`) reads that atom's presence as "a repair happened". `consolidation_gate.py` then proposed the window before any text check.
- **The text check.** It required BOTH the prompt and Orion's reply to be low-info, and that never happens.
- **The auto-activation history.** `intake_pipeline.py` threw away the history dict that `formation_executor.auto_activate` returns.
- **The Graphiti URL.** The Hub `.env_example` pointed at `http://orion-athena-graphiti-adapter:8000`. Failures were logged and turned into an empty result.

## Architecture touched

- **Hub grammar trace:** the `repair_signal` atom is emitted only at level ≥ 0.45. A new `repair_pressure_reading` atom carries sub-floor readings, so the chat projection's `repair_pressure_level` stays the same number it was before.
- **Consolidation gate:** a prompt-only junk check (`orion/memory/intake_junk.py`) runs first.
- **Crystallization intake:** writes the history row, and the summary skips junk.
- **Projection:** the Graphiti error is carried through to the responses and the UI.
- **Labels:** curiosity and self-study wording.

## Metric quality gate (repair-signal floor)

1. **Provenance.** The level is `reduce_repair_level` in `orion/substrate/appraisal/paradigms/repair_pressure_v2.py`, a weighted sum of 7 kind scores. It reaches the Hub as `bundle.grammar_scalars.repair_pressure.level`.
2. **Independence.** This is not a new metric. It is a threshold on an existing one. The threshold is the contract's own `_LEVEL_MID`, so the reply mode and the repair signal now agree.
3. **Theory anchor.** 0.45 is where `assemble_repair_contract_delta` switches the reply contract to `concrete_bias`. Below it, the appraisal changes nothing about the reply.
4. **Live data.** `repair_pressure_appraisal_log`, 2026-09-01 to 10-01, n=1,823:
   - 418 rows at 0.0 (no evidence);
   - 1,037 at 0.087, the classifier's confident all-NO floor;
   - 289 between 0.19 and 0.24;
   - 56 between 0.29 and 0.34;
   - 13 between 0.38 and 0.44;
   - 10 at 0.45 or above (0.55%).

   The grammar trace (n=498, 09-28 to 10-01) has 2 at 0.45 or above. The signal can return to a calm state: 98% of turns read below the floor.
5. **Existing mechanism.** The floor reuses the contract constant (`REPAIR_SIGNAL_LEVEL_FLOOR = _LEVEL_MID`).
6. **Reversibility.** It is one constant and two call sites.

**Consumer audit of the `repair_signal` atom:**
- `consolidation_grammar.py`: the intended change.
- `orion/substrate/chat_loop/grammar_extract.py`: reads the level, has_repair_signal and confidence. The level now also comes from the reading atom, so `chat_prediction_error`'s input is unchanged; a test pins this. `ChatTurnStateV1.has_repair_signal` has no consumer.
- `services/orion-cortex-exec/app/pcr_chat_memory.py`: reads the atom only from `ctx["grammar_events"]`, which nothing populates. It also returns True on `metadata.repair_pressure_contract`, so it is unaffected. That fallback has the same "an appraisal ran" problem, and it is out of scope here.

## Files changed

- `orion/substrate/appraisal/contract.py`: `REPAIR_SIGNAL_LEVEL_FLOOR`, `is_repair_signal()`
- `orion/hub/turn_orchestrator.py`, `services/orion-hub/scripts/websocket_handler.py`: an honest `has_repair_signal`, plus the sub-floor reading flag
- `services/orion-hub/scripts/grammar_emit.py`: the `repair_pressure_reading` atom
- `orion/substrate/chat_loop/grammar_extract.py`: reads the level from the reading atom
- `orion/memory/intake_junk.py` (new): greeting/filler and workflow-registry command checks
- `orion/memory/consolidation_gate.py`: the junk check runs first, on the prompt alone
- `orion/memory/crystallization/intake_consolidation_window.py`: the summary skips junk prompts
- `orion/memory/crystallization/intake_pipeline.py`: persists the `auto_activate` history row
- `orion/memory/crystallization/projection_graphiti.py`, `projector.py`: carry the Graphiti error
- `services/orion-hub/scripts/crystallization_routes.py`: the sync route returns `errors`
- `services/orion-hub/static/js/memory-crystallization-ui.js`: red status line on a Graphiti failure
- `services/orion-hub/.env_example`: `GRAPHITI_ADAPTER_URL=http://127.0.0.1:8640`
- `services/orion-cortex-exec/app/self_study_analysis.py`, `orion/curiosity/study_material.py`, `orion/curiosity/journal.py`: honest labels
- `services/orion-memory-consolidation/requirements.txt`: declares `requests` (the eval imports `orion.substrate`; required by the static gate)
- `services/orion-memory-consolidation/README.md`: documents the gate change and the eval
- Tests and evals: listed below

## Schema / bus / API changes

- **Added:** the `repair_pressure_reading` grammar atom role (hub.chat traces, `layer=organ_signal`). It is consumed by `grammar_extract`.
- **Behavior changed:**
  - the `repair_signal` atom is emitted only at real repair pressure;
  - the approve response's `projection.errors` and the sync response's new `errors` include `graphiti_sync_failed:…`;
  - `memory_crystallization_history` gains `op='auto_activate'` rows.
- **Compatibility notes:** no schema or table changes. `grammar_events` and `memory_crystallization_history` already accept these values.

## Env/config changes

- **Changed:** `services/orion-hub/.env_example` `GRAPHITI_ADAPTER_URL` (was `http://orion-athena-graphiti-adapter:8000`).
- **Local `.env` synced:** ran `python scripts/sync_local_env_from_example.py --all-keys orion-hub`. It reported `GRAPHITI_ADAPTER_URL` as diverged and did not overwrite it. I set that one line by hand in the local Hub `.env` instead of using `--force`, which overwrites the whole file and flattens secrets.
- **Skipped keys needing operator action:** none.

## Tests run

All runs use `/mnt/scripts/Orion-Sapienform/.venv/bin/python` from the worktree.

| Suite | Before | After |
|---|---|---|
| orion-memory-consolidation `tests` (+ `evals` after) | 119 passed | 229 passed |
| orion-hub targeted (grammar, turn orchestrator, crystallization UI, graphiti, curiosity) | 311 passed | 321 passed |
| repo-root targeted (curiosity, crystallization, self-study, chat reducer) | 421 passed, 1 failed | 429 passed, 1 failed |
| orion-thought `test_store.py` (+ reverie pin after) | 57 passed | 59 passed |
| orion-cortex-exec self-study | 18 passed | 18 passed |

The one root failure (`test_memory_card_v1_unchanged_in_registry_gap`) also fails on main.

These failures also fail on a clean main worktree at `f44dc1127`, unchanged by this PR:
- orion-durable-runs collection error (`test_admission_review_regressions.py`);
- `orion-thought test_settings_mind_enrichment`;
- 3 in `orion/substrate/tests/test_felt_state_self_definition_lane.py`;
- 3 in `orion-hub test_substrate_effect_endpoint.py`;
- the orion-cortex-exec duplicate-verb collection errors.

**Static gates** (`.github/workflows/orion-static-gates.yml`), all passing locally:
- `check_definition_drift.py --gate`: PASS, no re-lock needed;
- `check_metric_lineage.py --gate`;
- the grammar producer catalog;
- `test_substrate_services_declare_requests.py`, after the requirements fix;
- the Hub node tests;
- all the remaining checks in the workflow.

## Evals run

```text
python services/orion-memory-consolidation/evals/run_intake_gate_replay_eval.py
  107 windows; old gate admitted 107 (23 rows summarised by junk); new: kept 101, dropped 6
  repair-signal share 75.7% -> 4.1%; named keepers austin/offsite/labs/family: kept
  (only austin/offsite judged on real text); synthetic keepers kept: 13/13
same 107 windows replayed on their live, unredacted text: kept 101, dropped 6 (identical)
pytest services/orion-memory-consolidation/evals -q   -> 7 passed
```

**Fixture privacy.** A prompt is stored word for word only if the new rule calls it junk, or if the spec already quotes it (the Austin and offsite lines). Every other prompt is stored as `[redacted prompt: N chars]`. The labs and family keepers are checked by correlation id only.

**Known limit:** this fixture cannot catch a future rule that over-drops private content. `--refresh` re-judges the real text and warns when the real-text and redacted replays disagree.

## Docker/build/smoke checks

```text
Not run (instructed: no deploy/restart).
Live DNS check from inside the running Hub container (read-only):
  orion-athena-graphiti-adapter -> FAIL [Errno -3] Temporary failure in name resolution
  127.0.0.1:8640 /health -> {"service":"orion-graphiti-adapter","postgres":true,...}
```

## Review findings fixed

The code review found that the junk filter dropped real messages. That contradicts Juniper's "over-index on remembering". Fixed in commit `977b03d61`.

- **Finding:** non-Latin text was always dropped. The word pattern only knew `a-z`, so 'мама умерла сегодня', '母が亡くなった' and 'אמא שלי חולה' had "no words" and read as small talk.
  - **Fix:** a Unicode-aware word pattern. Any letter the English word lists can't judge (any non-ASCII letter, including accented Latin) means the text is never junk.
  - **Evidence:** `test_review_overdrop_cases_are_kept` and `test_non_latin_text_survives_…` (Cyrillic, Japanese, Hebrew, "Mamá está enferma", "café?").
- **Finding:** short negative feelings were filler. "not" was a stopword, so "I'm not ok" and "not good" were dropped.
  - **Fix:** negations (not, no, never, any "n't") count as content, and a prompt containing one is never junk.
  - **Evidence:** the same test covers "I'm not ok", "not good", "not great", "I'm sad", "rough day", "I can't sleep" and "no".
- **Finding:** a command with real content after it was dropped. "Do a journal pass about my labs" read as a command.
  - **Fix:** the extra words beyond the command may only be politeness ("please", "now", "hey orion"), never content.
  - **Evidence:** `test_command_with_content_is_not_a_command`. `test_command_with_only_politeness_is_still_a_command` shows "please run github compactor now" still drops.
- **Finding:** short questions about real things were dropped ("when is the surgery?", "where is mom?").
  - **Fix:** a question is junk only if every content word in it is social small talk (back, new, mind, happening, …).
  - **Evidence:** `test_review_overdrop_cases_are_kept` covers the real questions, and `test_pure_social_question_is_still_junk` shows "you back?", "what's up?", "how are you?", "what's new?" and "what else is on your mind?" still drop.
  - **Behavior change:** "hey, which queue?" is now kept. That moves the window "hi | hey, which queue?" from dropped to kept (7 drops become 6), and adds "hi Orion, how was your weekend?" to the short kept summaries. Both are questions about a real topic, which the review asked to keep. `test_review_change_is_the_only_new_keep` pins that this is the only window that changed.
- **Finding:** the eval's labs and family keepers proved nothing, because redacted placeholders can never be judged junk.
  - **Fix:** the report now says only 2 of the 4 named keepers are real checks. The eval adds 13 synthetic, non-private must-keep messages covering every finding above, plus "hi, my son was diagnosed today". The eval exits non-zero if any of them drops.
  - **Evidence:** `test_synthetic_keepers_are_all_kept`; the replay prints "Synthetic keepers kept: 13/13".
- **Finding (nit):** the novelty and significance floors can no longer fire.
  - **Fix:** a note in `consolidation_gate.py`, in this report, and in the spec's Stage 1 section.
  - **Evidence:** the diff to `docs/superpowers/specs/2026-09-30-memory-episode-redesign-design.md`.
- **Finding (nit):** "approved by Juniper" was inferred from `approval_mode != auto_policy`.
  - **Fix:** the menu cards and the "N approved by hand" count both read an `op='approve'` row from `memory_crystallization_history`. The self-study text now says the same. A `manual_required` row with no approval reads "no recorded approval from Juniper".
  - **Evidence:**
    - `test_manual_required_without_an_approve_row_is_not_called_approved`;
    - `test_approval_is_read_from_history_in_both_queries`;
    - a read-only live run of both queries: stance 36 approved, semantic and open_loop 0, and sample rows "Run github compactor." / "meow" with `juniper_approved=f`.

## Restart required

```bash
./scripts/safe_docker_build.sh orion-hub up -d --build
./scripts/safe_docker_build.sh orion-memory-consolidation up -d --build
./scripts/safe_docker_build.sh orion-cortex-exec up -d --build      # self-study label
./scripts/safe_docker_build.sh orion-durable-runs up -d --build     # curiosity journal footer
```

The Hub must be restarted for the Graphiti URL to change. Run these from a worktree at the merged main, not from this branch.

## Risks / concerns

- **Severity: medium.** The shift classifier still admits most windows: 92 of the 100 kept ones were admitted by `substantive_shift`. That is the 8B shift classifier, and this patch does not touch it. Real junk only falls out when every prompt in a window is junk. Stage 1's writer is the real fix.
- **Severity: low.** Short real-looking statements are kept on purpose ("sleepy", "meow", "another test"). So the spec's live check of "0 rows under 40 chars" will not reach 0. The command-pattern half of that check should.
- **Severity: low.** The PCR recall gate (`pcr_chat_memory._has_repair_grammar_signal`) still treats "a repair contract exists" as a repair signal. That is the same disease in another consumer, out of scope here.
- **Severity: low.** A failed `auto_activate` history insert is logged at error level (`crystallization_auto_activate_history_failed`) but does not fail the window. The row is already written by then, so failing the window would not undo it.
- **UNVERIFIED (live):**
  - the 48 h live checks: repair share under 20%, 0 command rows, every auto row has a history row;
  - an approval producing `graphiti_episode_ids ≠ []` after the Hub restart.

  None of these can be measured until the code is deployed.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2457

🤖 Generated with [Claude Code](https://claude.com/claude-code)
