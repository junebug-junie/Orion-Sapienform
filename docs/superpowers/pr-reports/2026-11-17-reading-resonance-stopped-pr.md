## Summary

- Reading resonance (comparing world reading with Orion's own answers) stopped on 2026-11-14 22:31Z. This patch restarts it.
- Cause: one lived question in the worldview graph has empty text. Its answer became Orion's current answer at 22:17Z. The resonance pool then built a "this answers question X" neighbour with empty text. The schema requires at least one character, so building the pool raised `ValidationError`.
- Both resonance paths build that pool. Submit (`reading_resonance_prepare_failed`) and findings (`reading_resonance_findings_unavailable`) failed on every tick.
- Fix: `worldview_pool` skips questions with blank text. The answer stays in the pool without its question neighbour. Findings with an empty id are also skipped (review hardening).

## Outcome moved

The resonance submit path can build its candidate pool again. Live read-only run of the fixed function against the real graph built 519 priors, 68 current answers, and 54 questions with no error.

## Current architecture

`services/orion-hub/scripts/world_pulse_read_stage2.py` `_reconcile_resonance` -> `resonance_submit.reconcile` -> `resonance_retrieval.prepare_brief` -> `worldview_index` -> `resonance_records.worldview_pool`. The findings path reaches the same function: `findings.findings_for_reads` -> `resonance_findings.resolve_citations` -> `worldview_pool`.

## Architecture touched

Hub reading resonance only. No contract change.

## Files changed

- `orion/world_pulse_read/resonance_records.py`: skip blank question text and empty finding ids.
- `orion/curiosity/tests/test_current_lived_answer.py`: regression test using the live shape (a blank question whose answer is current).

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: questions with blank text are no longer resonance candidates.
- Compatibility notes: none

## Env/config changes

None. No `.env_example` changed.

## Tests run

```text
pytest orion/curiosity/tests/test_current_lived_answer.py   # new test FAILS before the fix (1 failed, 8 passed)
pytest orion/curiosity/tests/test_current_lived_answer.py tests/test_curiosity_worldview.py   # 167 passed after the fix
```

## Evals run

None. The resonance eval (`services/orion-hub/evals/run_reading_resonance_eval.py`) needs live graph and LLM access. A read-only live pool build stood in for it (see Outcome moved).

## Docker/build/smoke checks

```text
Live diagnosis in orion-athena-hub, read-only:
- All 98 stored reading_resonance result_json rows validate.
- resonance_findings traceback: resonance_records.py:92 ResonanceNeighborV1 text='' string_too_short
- Falkor: SelfQuestionMint lived.self_monitor.act_grade_is_not_a_reliability_rate text='' (1 of 58 nodes)
- Its LivedAnswer d3f6ab8e4143 has written_at 2026-11-14 22:17:39Z. The last resonance row is 22:31:02Z.
```

## Review findings fixed

- Finding: no material findings. Optional hardening: a Finding whose `finding_id` is exactly '' would raise the same way.
  - Fix: the finding filter now also requires a non-empty id.
  - Evidence: 167 tests passed.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build   # from the clean primary main checkout after merge
```

## Risks / concerns

- Severity: low
- Concern: the writer that minted the blank question is not fixed here. Issue opened: #3592. Refs #3585; it stays open until new rows appear live.
- Mitigation: the reader now tolerates blank questions.

## PR link

See the PR.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
