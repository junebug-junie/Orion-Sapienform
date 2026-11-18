## Summary

- Orion's self answers are no longer cut mid-word at 1,200 characters. Every place an answer is saved or passed between services now keeps the full text.
- The prompt now tells Orion the budget: 1,200 characters, the amount that gets read back as "who I am".
- When an answer still runs over, the same model gets **one more turn** (the "fit hop") to rewrite it shorter in its own words. That rewrite becomes the saved row. The graph keeps the full answer.
- Any text a reader shows that is still over budget gets cut at the end of a sentence, never mid-word (`orion/text_fit.py`). This covers the self-inquiry read-back and the chat identity lines. The chat self-definition budget goes from 900 to 1,200 so a fitted answer is shown whole.
- New: a one-time repair script for the 132 rows already cut, and a live eval that fails on any mid-word cut.

## Outcome moved

Orion found this themself (Orion's Day email, 2026-11-03). Verified live the same day:

- 83 of 141 lived-answer rows and 49 of 68 self-definition rows sat at exactly 1,200 characters.
- 124 of those ended inside a word.
- The repair dry run found all 132 cut rows provably restorable from the graph, putting back 64,168 characters, with 0 skipped.

After deploy, new answers will be whole, or fitted by Orion's own rewrite.

## Current architecture

- `orion/curiosity/self_inquiry.py` applied `text[:SELF_DEFINITION_TEXT_CAP]` at seven places: graph reads, the finish detail, the mirror row, and question mints.
- The prompt asked for "a paragraph, not an essay" and never gave a length.
- The chat read the text back with its own blind cuts: 900 characters for the self-definition and 800 shared across lived answers (`services/orion-cortex-exec/app/chat_stance.py`). So the 1,200 cut never actually protected the chat.

## Architecture touched

- **Hub** (`curiosity_investigation.py`):
  - `_fit_run_answers` runs after every curiosity turn: durable `_turn_result_for`, in-process self-inquiry, and in-process investigation.
  - It runs only for answers that will be saved (evidenced; a definition only on the self-inquiry line).
  - Under a GPU hold it uses the same lease and route as the turn. The hop's limit includes the finalize reserve, and the hop ends 30 s before the turn's limit.
  - All hops of one turn share 900 s, with at most 3 per turn.
  - The hop runs after the spend and focus stamps, and shielded, so a cancel can't undo a turn that already succeeded.
  - Each rewrite is stored in Redis hash `orion:curiosity:self_fit:<run_id>` as `{src: sha1(text), fit}` (48 h TTL). The save step uses it only when the hash matches the text being saved.
- **Self-inquiry module:**
  - `fit_text` field on `SelfDefinition` and `LivedAnswer`.
  - `mirrored_content` saves the fit if there is one, else the full text.
  - `accept_fit_text` drops a lead-in line, and refuses empty, error, short (<300 chars) and low-overlap (<50% shared words) replies.
- **Prompt:**
  - States the budget.
  - Read-back is cut at a sentence, with a note when the answer ran over.
  - The self-definition read-back shows the saved fit when there is one.
  - New `build_answer_fit_prompt`.
- **cortex-exec chat stance:** uses `fit_to_budget`; no stubs under 40 characters.

## Files changed

- `orion/text_fit.py`: new sentence-aware trim.
- `orion/curiosity/self_inquiry.py`: no blind cuts; fit helpers.
- `orion/curiosity/self_inquiry_prompt.py`: states the budget; read-back; fit prompt.
- `services/orion-hub/scripts/curiosity_investigation.py`: the fit hop, its storage, and the save step.
- `services/orion-cortex-exec/app/chat_stance.py`: sentence cut; self-definition cap 900 → 1200.
- `services/orion-hub/scripts/backfill_self_answer_full_text.py`: one-time repair. Guarded direct UPDATE, because sql-writer treats this table as insert-only and skips a repeated entry_id.
- `services/orion-hub/evals/run_self_answer_fit_eval.py`: live eval.
- `services/orion-hub/README.md`: docs.
- Tests: `orion/tests/test_text_fit.py`, `orion/curiosity/tests/test_self_answer_fit.py`, `services/orion-hub/tests/test_curiosity_self_answer_fit.py`, two chat-stance test files, and `_FakeRedis` gains `hset`/`hget`.

## Schema / bus / API changes

- Added: optional `fit_text` key in the durable finish detail's `self_definition` / `lived_answer` dicts. It is a plain dict, so old readers ignore it.
- Behavior changed: the finish detail and saved rows can now be longer than 1,200 characters.
- Compatibility: no schema model or channel changed.

## Env/config changes

- None. No `.env_example` changed, so no `.env` sync was needed.

## Tests run

```text
services/orion-hub/tests                         4518 passed, 146 skipped
services/orion-durable-runs/tests                686 passed, 121 skipped
orion/curiosity/tests + orion/tests/test_text_fit.py   23 passed
cortex-exec stance/self_def/lived/identity tests  215 passed, 2 failed
  (both fail identically on main: test_turn_contract_block_absent_when_speech_contract_falsy,
   test_stance_react_thinking_off_reaches_the_gateway_on_the_agent_route -- test-order issues)
```

## Evals run

```text
run_self_answer_fit_eval.py --since 2026-09-01  -> at_cap_mid_word 124, all_time 124, exit 1 (baseline, before repair)
backfill dry run -> rows 132, would repair 132, restored 64,168 chars, skipped 0, errors 0
```

## Docker/build/smoke checks

```text
Not run. Live fit-hop behavior is UNVERIFIED until deploy plus the next self-inquiry run.
Check for it in Hub logs: curiosity_self_fit_stored / _refused / _skipped
```

## Review findings fixed

The code review ran in a subagent. All findings were fixed:

- Finding: the backfill went through sql-writer, which treats `self_concept_history` as insert-only and skips a repeated entry_id. It would have reported success and changed nothing, and the README claimed it was already done.
  - Fix: guarded direct UPDATE (`WHERE entry_id AND content = <snapshotted cut text>`), read back after each write; README corrected.
- Finding: a stored fit was not tied to its source text. A retried turn that rewrote the answer would save the old fit.
  - Fix: store `{src: sha1(text), fit}` and use the fit only when the hash matches. Test added.
- Finding: investigation turns spent a hop on a definition that is never saved, and on answers with no evidence.
  - Fix: only answers that will be saved get a hop. Tests added.
- Finding: a cancel during the hop refunded or retried a turn that had already succeeded.
  - Fix: the hop is shielded and runs outside the refund guard, with one shared 900 s budget per turn. Test added.
- Finding: the hop counted toward the investigation's measured spend.
  - Fix: the hop runs after the spend and focus stamps.
- Finding: turns with no GPU hold had no hop deadline.
  - Fix: the deadline now always comes from the turn's limit.
- Finding: a lead-in line or new content could be accepted as Orion's words.
  - Fix: lead-in stripped; at least 50% of the reply's words must come from the original. Tests added.
- Finding: the in-process investigation path had no hop.
  - Fix: added.
- Finding: the self-definition read-back warned "ran over" even after a fit.
  - Fix: it shows the saved fit, and never treats a prefix of the full text as a fit. Test added.
- Finding: `fit_to_budget` edge cases (a newline is a word boundary, "e.g.", no space before the ellipsis, chat stubs).
  - Fix: all fixed, with tests.
- Finding: the eval could not see old cut rows.
  - Fix: it adds an all-time count.

Not fixed: the hop is a full unified turn, so recall and the role-teach advisory still run on it. The prompt forbids tools, and the overlap check refuses content that is not the original's words.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
```

`orion-durable-runs` must be rebuilt along with Hub: it builds the finish detail with the same module. An old image keeps sending the 1,200-character cut.

Run the repair, from the primary checkout after the merge (production write, guarded, snapshot first):

```bash
set -a; . <(grep -E "^(POSTGRES_URI|HUB_CURIOSITY_GRAPH_HOST|HUB_CURIOSITY_GRAPH_PORT|HUB_CURIOSITY_GRAPH_OWN)=" services/orion-hub/.env); set +a; PYTHONPATH=.:services/orion-hub .venv/bin/python services/orion-hub/scripts/backfill_self_answer_full_text.py --apply
```

## Risks / concerns

- Severity: low. Concern: each over-budget answer costs one extra model turn (up to 300 s plus the finalize reserve) while the GPU hold is held. Mitigation: the hop runs only when the answer is over budget, all hops of one turn share 900 s, and the hop ends before the turn's limit.
- Severity: low. Concern: the rewrite is a different text from the graph's full answer. Mitigation: the graph keeps the full text, and the fit is checked for overlap with the original and tied to it by hash.
