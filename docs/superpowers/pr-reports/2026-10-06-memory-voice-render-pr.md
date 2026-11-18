# feat(memory): voice renderer

Memory Stage 2, PR D (design: `docs/superpowers/specs/2026-10-06-memory-stage2-referent-graph-design.md`, PR #2496, row D of 7.5; voice contract: rev 3 section 7). This PR makes Orion's remembered items say whose thought they are. The intent fix that was first in this PR has been removed after review: its signals are dead on the live path. It moves to PR F (see "Input for PR F" below).

## Summary

- **One voice renderer, `orion/memory/voice_render.py`.** It turns a remembered item into the line Orion reads:
  - "Juniper told me (10-04): …"
  - "Juniper and I worked out …"
  - "I told Juniper …"
  - "Something I was turning over on my own (reverie, 10-04), not something Juniper and I discussed: …"
  - "My own note …, not Juniper's words: …"
  - It is a pure function.
- **It uses the Stage 1 validator's own rules, so it can never be looser than the writer.** It calls `validate.py`'s `_supported_voice` and its channel rule:
  - "Juniper and I worked out" needs a chat memory with verified quotes from both her prompt and Orion's reply.
  - "Juniper told me" needs a verified prompt quote.
  - Juniper's voices exist only on `chat` (not `confirmation`).
  - Missing evidence moves the voice only away from Juniper.
- **A reverie is never rendered as something Juniper said.** Anything on an internal channel renders as Orion's own private thought, whatever its voice claims.
- **Rejected and corrected memories never render as truth.** They read "Something I had remembered (…) that Juniper rejected, not true: …" or "…that Juniper corrected, superseded: …".
- **The live consumer is the daily episode report.** Each memory now shows exactly the line Orion would read; the report's old label map is deleted.
- **The reverie glimpse is also wired, but only on the legacy path.** `chat_stance._project_reverie_glimpse` renders through the renderer. Only `chat_stance_brief.j2` (the legacy `chat_general` stance) shows `chat_reverie_glimpse`, and it had no live turns on 2026-10-06. The live `stance_react.j2` never renders it. Adding it there is pending Juniper's decision and is not done here.

## Outcome moved

- **The daily report shows source monitoring as rendered**, not just the stored voice label. The 31 live episode memories render as 27 "Juniper told me" (each with a verified prompt quote) and 4 "I told Juniper" (each with a verified quote). Checked read-only.
- **No live chat behavior changes in this PR.** The stance prompt that live chat uses is untouched, and so is PCR intent selection (reverted to main).

## Current architecture

- `voice_render.py` did not exist.
- The episode report printed its own label map (`_VOICE_LABEL`) next to the raw statement.
- The reverie glimpse passed the raw interpretation into `chat_stance_brief.j2` (legacy path).

## Architecture touched

- `orion/memory/voice_render.py` (new): `VoicedMemory`, `render_memory`, `speaker`. It imports `INTERNAL_CHANNELS` and `_supported_voice` from `orion/memory/episode/validate.py`.
- `orion/memory/episode/report.py`: renders through the renderer. The query adds `remembered_at` and two EXISTS checks: a verified prompt quote and a verified reply quote.
- `services/orion-cortex-exec/app/chat_stance.py`: the legacy reverie glimpse goes through the renderer.

## Concepts (producer → consumer → test)

| Concept | Producer | Consumer | Test |
|---|---|---|---|
| `render_memory` / `VoicedMemory` / `speaker` | episode report rows (live); legacy reverie glimpse | daily report file (live); `chat_stance_brief.j2` (legacy, 0 live turns) | `test_voice_render.py`, `test_episode_report_pg.py`, glimpse tests |
| "Juniper told me" | live `juniper_said`/chat rows with a verified prompt quote (27) | report | matrix, `test_juniper_said_needs_a_verified_prompt_quote`, PG test |
| "Juniper and I worked out" | validator keeps `worked_out_together` only with prompt + reply quotes on chat | report | `test_worked_out_together_needs_both_quotes_and_chat`, PG test |
| "I told Juniper" | live `orion_thought`/chat rows with a verified quote (4) | report | matrix |
| "…on my own…, not something Juniper and I discussed" | internal-channel memories; legacy glimpse | report; legacy stance prompt | matrix, `test_the_180_hecate_reveries_case`, PG test |
| "My own note…, not Juniper's words" | any row failing the checks above | report | matrix, PG test |
| "Unconfirmed, check with Juniper if natural:" | validator sets `pending_confirmation` for high stakes | report | `test_pending_marker_and_whitespace` |
| "…that Juniper rejected, not true" / "…corrected, superseded" | `episode_memory.confirmation_state` | report | `test_rejected_and_corrected_are_never_plain_truth`, matrix, PG test |

The same table, with a plain-English column, is in `orion/memory/README.md` (new). `services/orion-cortex-exec/README.md` notes that the glimpse is legacy-only.

**Cut on purpose (no producer or no consumer):**
- **The intent change** (below).
- **"I read" / "From my own code and docs".** The validator turns `orion_read`/`orion_self_knowledge` into `orion_thought` for chat episodes, so nothing produces them. This PR's first report wrongly said the validator kept them.
- **Reading titles, claim status, graphify build dates and the "faded" marker.** Nothing produces them yet.
- **The intent → memory-kind table and the `MemoryItemV1` voice fields.** Both go to PR F.

## Input for PR F: why the intent change was removed

This PR first changed `derive_retrieval_intent` so that `open_loop` stopped winning on every turn (630 of 630 purposeful recalls in 7 days used `chat.belief.open_loop.v1`). Review of #2521 (orchestrator, 2026-10-06) replayed 590 live turns and found the replacement does not work on the live path:

- **The turn-change appraisal is post-turn.** `chat_stance_belief_log.shift_kind` is null on 56,520 of 56,520 rows, so the TOPIC/STANCE/REPAIR shift rules never fire live.
- **Repair pressure is absent too.** `thought_decision.repair_pressure_level` is null on 659 of 659 rows.
- **The net effect would have been a shift, not a fix.** On the 590-turn replay, the change cut phase 3 by about 50%, and the remaining intents came out about 85% `relational`. One dominant intent replaced another.

This PR's own work had flagged the post-turn appraisal as UNVERIFIED; the review confirmed it dead.

For PR F: intent and memory selection should key on **referents known before the turn**, i.e. recall by referent from the alias map, not on post-turn appraisals. These two observations also hold and stay useful there:
- the same-turn LLM signal (`current_turn_llm_signals`) is present on human turns;
- the old `entity_query` regex matches the first capitalized word of nearly any sentence.

`orion/memory/retrieval_intent.py`, its tests and its cortex-exec caller are byte-identical to origin/main in this PR.

## Files changed

- `orion/memory/voice_render.py`: new renderer.
- `orion/memory/episode/report.py`: renders through `voice_render`; `_VOICE_LABEL` deleted; query adds `remembered_at`, `has_verified_juniper_quote`, `has_verified_orion_quote`.
- `services/orion-cortex-exec/app/chat_stance.py`: legacy reverie glimpse labelled.
- `orion/memory/tests/test_voice_render.py` (new): 2,016-case matrix over voice × channel × evidence × the 6 confirmation states, plus contract tests.
- `services/orion-memory-consolidation/tests/test_episode_report_pg.py`: five memories through the real SQL (verified, unverified, reverie, worked-out-together with both quotes, rejected).
- `services/orion-cortex-exec/tests/test_chat_stance_reverie_glimpse_projection.py`: the glimpse expects the labelled line, and never presents a reverie as Juniper's or shared.
- `.github/workflows/memory-voice-render-tests.yml` (new): the renderer matrix and glimpse tests.
- `orion/memory/README.md` (new), `services/orion-cortex-exec/README.md`: concepts.

## Schema / bus / API changes

- None on the bus or in the registry. New internal function `orion.memory.voice_render.render_memory`.
- Behavior changed:
  - The daily episode report's memory lines.
  - The legacy `chat_general` stance prompt's `reverie_glimpse` text, which gains a label.

## Env/config changes

- None. No `.env_example` touched, so no `.env` sync was needed. No flags.

## Tests run

```text
After merging origin/main (incl. #2517's confirmation loop):
renderer matrix + contract (orion/memory/tests):                           2021 passed
orion/memory/episode/tests + orion/memory/tests + orion-memory-consolidation tests + evals,
  against a throwaway postgres:16 (ORION_MEMORY_EPISODE_TEST_DATABASE_URL set): 2601 passed, 0 skipped
orion-cortex-exec glimpse + PCR + grounding:                                  27 passed
tests/test_retrieval_intent.py (main's behavior, unchanged):                 8 passed
static gates (all 25 run steps of orion-static-gates.yml, incl. check_definition_drift --gate): 25/25 PASS
pyflakes on touched files: clean; git diff --check: clean
```

## Evals run

```text
No eval harness for rendering. Live read-only check: 31 live episode memories through the renderer with
their real evidence flags -> 27 "Juniper told me", 4 "I told Juniper", 0 other.
The spec's eval 4 (source monitoring over 7 days of shadow recall) belongs to PR F's evals/referent/.
```

## Docker/build/smoke checks

```text
Not deployed (instruction). No dependency changes. Throwaway postgres:16 container for the PG-backed tests.
Production was only read (read-only transactions; docker logs).
```

## Merge with main (2026-10-06, after #2517)

`origin/main` was merged into this branch with a merge commit (no rebase, no force push).
- `orion/memory/episode/report.py`: #2517 added `CONFIRMATION_FLAG`, which records what the confirmation loop did to each memory. Both are kept. Each line is `[purpose, voice/channel, stakes]{CONFIRMATION_FLAG} {rendered line}`: the bracket and flag are the operator's audit, and the rest is what Orion would read.
- `orion/memory/voice_render.py`: #2517 introduced the state `unconfirmed` (asked, no answer in 7 days). It now keeps the "Unconfirmed, check with Juniper if natural:" prefix, matching the spec ("an expiry is not a resolution"). It is in the test matrix.
- `services/orion-cortex-exec/app/chat_stance.py`: an import-only conflict. Main removed the now-unused `build_substrate_store_from_env` import (a2992e217); this branch's `voice_render` import is kept.

## Review findings fixed

Review of #2521 (orchestrator, 2026-10-06):
- Finding: the intent change's signals are dead live (post-turn appraisal; null repair pressure). The 590-turn replay showed a ~50% phase-3 cut and ~85% relational.
  - Fix: removed; the intent files are identical to main. Recorded above as input for PR F.
  - Evidence: `git diff origin/main` on those 4 files is empty.
- Finding: the claim that the reverie label changes live chat is false. Only the legacy `chat_stance_brief.j2` renders the glimpse.
  - Fix: claims corrected here and in both READMEs. The live consumer is the daily report. `stance_react.j2` is untouched pending Juniper.
- Finding: `worked_out_together` rendered without evidence and on `confirmation`.
  - Fix: the renderer runs the validator's `_supported_voice` (prompt and reply quotes) and its chat-only rule.
  - Evidence: `test_worked_out_together_needs_both_quotes_and_chat`, matrix.
- Finding: `rejected` was missing from the test states, and a rejected memory could render as "Juniper told me".
  - Fix: rejected and corrected render explicitly marked as not true or superseded.
  - Evidence: matrix now covers 5 states; PG test includes a rejected row.
- Finding: tests injected an appraisal no live turn has.
  - Fix: removed with the intent change.

Self-found earlier: a `juniper_said` chat memory without a verified quote would have read "on my own…", which is false for chat; it now reads "My own note…, not Juniper's words".

## Restart required

After merge, from the primary checkout on main (one line each). The report change is picked up by orion-memory-consolidation. orion-cortex-exec only affects the legacy path.

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-memory-consolidation up -d --build
cd /mnt/scripts/Orion-Sapienform && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-cortex-exec up -d --build
```

## Risks / concerns

- Severity: low. Concern: live chat does not use the renderer yet. Mitigation: that is PR F's recall render, plus Juniper's pending decision on `stance_react.j2`; stated here and in the READMEs.
- Severity: low. Concern: the renderer imports a private validator helper (`_supported_voice`). Mitigation: it is deliberate, so writer and renderer cannot drift apart; any change to it is covered by both test sets.
- UNVERIFIED: the daily report file after deploy (not deployed).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2521

🤖 Generated with [Claude Code](https://claude.com/claude-code)
