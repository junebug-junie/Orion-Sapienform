## Summary

- Orion's graph writes now get a `written_at` timestamp guaranteed by code, not by Orion copying the prompt's example.
- Before each Orion turn starts, the harness motor records which run nodes already lack `written_at` (the legacy ones). After each graph-writing shell command finishes, it stamps only nodes created during this turn.
- A last stamp at turn end catches writes the command check can't see (e.g. a script); kills and timeouts get it too.
- The 186 legacy Hops (and 126 Findings, 1 InvestigationRole) stay unstamped, so their unknown real time is never faked. Stamped nodes carry `written_at_source='harness_stamp'` so readers can tell the harness's clock from Orion's.
- Covers every run-node label the atlas reads `written_at` from (`RUN_NODE_FIELDS`): Hop, Finding, TurnOutcome, InvestigationRole, PriorRevision, SelfDefinition, LivedAnswer.

## Outcome moved

A run where Orion forgets `written_at` no longer drops out of the date-windowed atlas strip (`run_ids_since_cypher`), and its hops no longer sort as "unknown time".

## Current architecture

Orion writes `orion_worldview` from inside the `claude -p` subprocess with `redis-cli GRAPH.QUERY`. No code ran those writes, so `written_at` relied on the kickoff prompt example (fixed in e88766059, not enforced). Live 2026-10-01: 186/594 Hops had no `written_at`.

## Architecture touched

- `orion/harness/fcc_motor.py` stream loop: this is the only code that sees each in-turn graph command finish. It already holds Orion's curiosity graph credentials (`sandbox_env.py`).
- The model's Cypher is not rewritten. The stamp is a separate, deterministic query run as Orion's own ACL user (`orion_curiosity`, read/write on `orion_worldview` only).

## Files changed

- `orion/curiosity/write_stamp.py`: baseline + stamp queries, `WriteStamper`, and stream-event helpers.
- `orion/harness/fcc_motor.py`: arms the stamper before spawning; stamps after each GRAPH.QUERY tool result and at turn end.
- `tests/test_curiosity_write_stamp.py`: stamp semantics (new node stamped, legacy spared, model timestamps untouched, fails closed with no baseline), plus a real-FalkorDB test gated on `ORION_TEST_FALKORDB_PORT`.
- `orion/curiosity/tests/fake_worldview_graph.py`: shared in-memory fake of the two stamp queries.
- `docs/superpowers/pr-reports/2026-10-01-hop-written-at-stamp-pr.md`: this report.
- `orion/harness/tests/test_fcc_motor_write_stamp.py`: drives the motor loop. A model-style Hop created without `written_at` is stamped before the next model step; a legacy Hop is not.

## Schema / bus / API changes

- Added: graph property `written_at_source = 'harness_stamp'` on harness-stamped nodes. Readers ignore unknown properties.
- Removed / Renamed: none
- Behavior changed: run nodes created without `written_at` get it within seconds of the write.
- Compatibility notes: no bus or schema registry changes.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none
- Kill switch: the existing curiosity graph keys in `~/.fcc/.env`. If any is absent, there is no stamper.

## Tests run

```text
pytest tests/test_curiosity_write_stamp.py orion/harness/tests/test_fcc_motor_write_stamp.py  -> 14 passed, 1 skipped
ORION_TEST_FALKORDB_PORT=<scratch falkordb/falkordb container> pytest tests/test_curiosity_write_stamp.py -> 9 passed
pytest orion/harness/tests orion/curiosity/tests tests/test_curiosity_write_stamp.py tests/test_curiosity_worldview.py -> 658 passed, 1 skipped
services/orion-harness-governor: pytest tests -> 58 passed
Mutation: disabling the per-tool-result stamp fails test_hop_written_without_written_at_is_stamped_after_its_tool_result
```

## Evals run

```text
None. There is no eval harness for the motor. The real-FalkorDB test is the behavioral check.
```

## Docker/build/smoke checks

```text
Cypher checked against live orion_worldview with read-only queries and GRAPH.EXPLAIN only (313 legacy unstamped run nodes found).
End-to-end stamp checked on a throwaway local FalkorDB container. No production writes.
Live path: UNVERIFIED until a deployed curiosity turn logs `write_stamp_applied` and new Hops show `written_at_source='harness_stamp'` (or a model-written `written_at`).
```

## Review findings fixed

- Finding (must): the turn-end stamp swallowed `asyncio.CancelledError`.
  - Fix: catch `Exception` only, so cancellation propagates.
  - Evidence: `fcc_motor.py` finally block.
- Finding: added latency on every chat turn (a 3s timeout, plus a turn-end write even on turns that ran no tools).
  - Fix: timeout cut to 1s, and the turn-end stamp is skipped when the turn ran no tool.
  - Evidence: `test_toolless_turn_skips_the_turn_end_write`; flipping the guard to `True` makes it fail.
- Finding: the `GRAPH.QUERY` match was case-sensitive, so a lowercase `graph.query` write only got the later turn-end stamp.
  - Fix: case-insensitive match. `graph.ro_query` still does not match.
  - Evidence: added a case to `test_graph_write_detection_from_stream_events`.
- Finding: the per-turn redis client was never closed.
  - Fix: added `WriteStamper.close()`, called in the motor's finally block.
  - Evidence: the main motor test asserts `graph.closed`.
- Finding: the fail-closed test didn't prove fail-closed, and the kill/timeout path was untested.
  - Fix: the graph now recovers mid-turn in the test, which asserts no write query ran and the new Hop stayed unstamped. Added a timed-out-turn test.
  - Evidence: `test_graph_unreachable_at_turn_start_fails_closed`, `test_timed_out_turn_still_stamps_at_turn_end`.
- Finding: known gaps were not stated (a write still in flight when a turn is killed, and FalkorDB id reuse). All of them fail in the safe direction.
  - Fix: documented in the `write_stamp.py` module docstring.
- Finding: one test module imported a fake from another test module, and one test only checked that an import existed.
  - Fix: moved the fake to `orion/curiosity/tests/fake_worldview_graph.py` and dropped the trivial test.
- Not fixed (nit): the full node scan instead of per-label scans. Cheap at the current graph size (about 3k nodes).
- Finding: no PR report was committed.
  - Fix: this file.

## Restart required

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

## Risks / concerns

- Severity: low
  - Concern: adds one read-only FalkorDB query (1s timeout) at the start of every non-reading FCC turn, including chat turns.
  - Mitigation: if it fails, the stamper is disarmed and the turn runs normally.
- Severity: low
  - Concern: the turn-end stamp records turn-end time, not the real write time, for writes the GRAPH.QUERY check missed.
  - Mitigation: these nodes are marked `written_at_source='harness_stamp'`.

## PR link

(this PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
