## Summary

- Orion is now taught to write each stop (`:Hop`) and each turn decision (`:TurnOutcome`) with MERGE on its identity, so sending the same command twice no longer makes a second node (#3543).
- `:Hop` is keyed on `{run_id, n}`; the note and `written_at` go in `ON CREATE SET`. `:TurnOutcome` is keyed on `{run_id}`; `written_at` is on create, and the decisions are a plain `SET`, so they still land when the focus anchor (`orion/harness/task_anchor.py`) made the node first.
- The "do not resend a :Hop or :TurnOutcome" sentence is replaced by one that is now true.
- New rule in the prompt: match both ends of an edge by id first. Never write `-[:ABOUT]->(:Prior)` inline (that minted the blank Prior node 6510).
- Hub's question scrape and the resonance reader now read a `:SelfQuestionMint`'s words from `text`, else from `question`. A mint that still has no words is dropped and logged by question id, never silently (#3592).

## Outcome moved

A resent Hop/TurnOutcome command leaves one node (checked on a scratch FalkorDB). Two live question mints keyed under `question` will now reach the question pool and resonance on the next read.

## Current architecture

`kickoff_prompt.py` taught `CREATE (:Hop {...written_at: timestamp()})` and `CREATE (:TurnOutcome {...})`. Live 2026-11-17: 82 duplicated Hop pairs (95 extra rows), 22 duplicated TurnOutcome runs (23 extra). `build_self_question_mint` returned None for any mint without `text`, with no log.

## Architecture touched

Prompt text only on the write side. Read side: `orion/curiosity/self_inquiry.py` (`mint_text`, `self_question_mint_drop_reason`, warning log), `orion/world_pulse_read/resonance_records.py` (same fallback, drop log).

## Files changed

- `orion/curiosity/kickoff_prompt.py`: MERGE templates, resend sentence, match-first edge rule.
- `orion/curiosity/self_inquiry.py`: read `question` beside `text`; log dropped mints.
- `orion/world_pulse_read/resonance_records.py`: same fallback; log blank mints.
- `tests/test_curiosity_taught_writes_resend.py`: new; checks the rendered commands, and replays them twice on a scratch FalkorDB (after the real focus-anchor statement).
- `tests/test_self_question_mint_read.py`: new; `question`-keyed mint read, blank mint dropped and logged.
- `tests/test_curiosity_hop_identity.py`, `orion/curiosity/tests/test_focus_lifecycle.py`, `orion/curiosity/tests/test_current_lived_answer.py`: updated/added assertions.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: taught Cypher for `:Hop`/`:TurnOutcome` is MERGE. Readers accept `question` as a fallback key for mint text.
- Compatibility notes: existing duplicate nodes are not touched.

## Env/config changes

None.

## Tests run

```text
pytest orion/world_pulse_read/tests orion/curiosity/tests orion/harness/tests tests/test_curiosity*.py tests/test_*worldview*.py tests/test_self_question_mint_read.py services/orion-curiosity-peer/evals
2160 passed, 31 skipped
ORION_TEST_FALKORDB_PORT=<scratch falkordb/falkordb:latest> pytest tests/test_curiosity_taught_writes_resend.py tests/test_curiosity_write_stamp.py
2162 passed, 29 skipped with the scratch FalkorDB set, including both real-FalkorDB tests (resend leaves one Hop, one TurnOutcome, one Prior, one ABOUT edge; write stamp unaffected)
```

## Evals run

```text
services/orion-curiosity-peer/evals (agency episode eval) passed in the run above. No eval measures resend behaviour of the live model; the scratch-graph replay is the check.
```

## Docker/build/smoke checks

```text
Scratch container falkordb/falkordb:latest (same image as live): worldview_pool() over mints {text:''}, {question:'q words?'}, {text:'t words?'} returned the two worded ones and logged
resonance_self_question_mint_dropped count=1 qids=['a'] reason=blank_text
No live graph was written.
```

## Review findings fixed

- Finding: a different stop that reuses an existing n was silently dropped (ON CREATE SET keeps the old note, no error). Resumed sittings have restarted at n=1 before.
  - Fix: the taught Hop command ends `RETURN h.note`; the prompt says a mismatched note means that n is taken, so resend with the next n.
  - Evidence: real-FalkorDB test sends a reused n and checks the old note comes back and the count stays 1.
- Finding: the new Hop->Prior ABOUT example taught an edge no code reads (keyword-cathedral rule).
  - Fix: kept the match-both-ends rule, dropped the Hop example; it points at the existing Finding/Concept examples.
  - Evidence: scratch test now checks Finding-SUPPORTS->Prior sent twice leaves one Prior and one edge.
- Finding: `build_lived_answer` was accidentally given the mint fallback (a replace hit two identical blocks).
  - Fix: restored the plain `text` check for answers.
  - Evidence: focused suites green.
- Finding: one leftover `continue_line: false` in map syntax; scratch test stripped `//` comments instead of sending the command as taught.
  - Fix: now `t.continue_line = false`; the test sends the multi-line block with its comments.
  - Evidence: real-FalkorDB test passes on the raw block.
- Not changed: TurnOutcome `written_at` now records the first write only; reviewer confirmed no reader needs the latest-decision time.

## Restart required

Hub (mint scrape) and world-pulse-read (resonance) pick this up on their normal post-merge rebuild. The prompt text reaches Orion when the FCC sandbox syncs after deploy.

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only
```

## Risks / concerns

- Severity: low. Concern: the existing 95 Hop + 23 TurnOutcome duplicates, blank Prior 6510, and the blank mint `lived.self_monitor.act_grade_is_not_a_reliability_rate` stay in the live graph. Removing or repairing them is a production write and needs Juniper. Mitigation: both issues are Refs, not Fixes. #3543 asks for the standing counts re-run after the fix (no new duplicate pairs after deploy), which is live proof this PR cannot give. #3592 stays open until the blank node gets its words or is retired.
- Severity: low. Concern: a model can still leave the template. Mitigation: readers now fall back and log instead of dropping silently.
- Issues: refs #3543, refs #3592. None opened or closed.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3643
