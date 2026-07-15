# PR report: post-hoc tool-provenance mismatch audit for the FCC motor

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1060
Branch: `feat/fcc-tool-provenance-audit`

## Summary

- `orion/harness/tool_provenance_audit.py`: pure `detect_tool_provenance_mismatch(draft_text, receipts)` — flags when a turn's draft uses live-immediacy language ("this turn", "right now", "happening now", "in the background") while that same turn's tool trace shows a fetch-shaped tool call (`get_file_contents`, `read_file`, web fetch, etc.).
- Wired into `HarnessRunner.run()`: computed once after the FCC subprocess stream completes, before either lifecycle-publish path, so it reaches both the empty-draft and success return paths.
- `HarnessMotorResult`/`HarnessDraftMoleculeV1` both get a new `tool_provenance_audit: str | None` field, kept deliberately separate from the existing `grounding_status` field.
- `HarnessGrammarCollector.record_tool_provenance_mismatch()` publishes a `GrammarAtomV1(atom_type="uncertainty_marker")` on the same grammar bus channel every other harness event already uses, plus a correlation-ID'd warning log.

## Outcome moved

Closes the audit half of a real incident: Orion narrated a GitHub-MCP file fetch as live substrate computation "happening this turn." A prior PR (#1056) fixed the prevention side (the FCC motor's own system prompt now carries a `CONTEXT PROVENANCE` block). This PR closes the detection side: since the FCC motor is a single-shot `claude -p` subprocess with no mid-run injection point (confirmed by reading `fcc_motor.py` — nothing can reach back into a live run to stop a confabulation before it's generated), the only remaining lever is a post-hoc check comparing what the draft claims against what the turn's own tool trace shows. That's what this PR builds.

## Current architecture

`orion/harness/fcc_motor.py::run_fcc_turn` spawns the FCC motor as a single-shot `claude` subprocess and streams back JSON events; `_summarize_content_blocks`/`_tool_result_body_text` already parse every `tool_use`/`tool_result` block. `HarnessRunner.run()` (`orion/harness/runner.py`) consumes that stream, publishing each step as a `GrammarReceiptV1` via the existing grammar bus/schema contract (`orion/grammar/publish.py`), and captures `draft_text` from the `final` event — both available together, in the same function, before the turn's result is returned. `HarnessMotorResult.grounding_status` already exists but is an overloaded error/overflow code with real downstream consumers (`orion/hub/turn_orchestrator.py`, `services/orion-harness-governor/app/bus_listener.py`) that surface it as `base["error"] = ...` — confirmed via grep that repurposing it for this signal would misreport a soft audit as a hard motor failure, so this uses a new, separate field instead.

## Architecture touched

- `orion/harness/tool_provenance_audit.py` (new)
- `orion/harness/runner.py`
- `orion/harness/grammar_emit.py`
- `orion/schemas/harness_finalize.py`

## Files changed

- `orion/harness/tool_provenance_audit.py`: new, pure detection function + two regexes (fetch-shaped tool names, immediacy language).
- `orion/harness/runner.py`: computes the audit once per turn in `HarnessRunner.run()`, threads it into both `HarnessMotorResult` return sites and `build_draft_molecule()`.
- `orion/harness/grammar_emit.py`: new `HarnessGrammarCollector.record_tool_provenance_mismatch()` method.
- `orion/schemas/harness_finalize.py`: `HarnessDraftMoleculeV1.tool_provenance_audit: str | None = None`.
- Tests: `orion/harness/tests/test_tool_provenance_audit.py` (9 unit tests), extensions to `test_harness_runner.py` (2 integration tests) and `test_harness_grammar_emit.py` (2 tests, including an edge-topology regression test).

## Schema / bus / API changes

- Added: `HarnessMotorResult.tool_provenance_audit: str | None = None`, `HarnessDraftMoleculeV1.tool_provenance_audit: str | None = None` — both additive with safe defaults.
- Added: a new grammar atom `semantic_role="exec_tool_provenance_mismatch"` (`atom_type="uncertainty_marker"`, an existing enum value — no new AtomType needed), published on the existing `orion:grammar:event` channel.
- Removed: none. Renamed: none.
- Behavior changed: none for any turn without a tool-provenance mismatch (field stays `None`, no new atom emitted). `grounding_status`/`compliance_verdict` are explicitly untouched by design.
- Compatibility notes: fully additive; `HarnessDraftMoleculeV1` has no `extra="forbid"`, `HarnessMotorResult` is a plain dataclass with the field appended last with a default — verified no downstream construction site breaks.

## Env/config changes

None.

## Tests run

```text
PYTHONPATH=<worktree> venv/bin/python -m pytest \
  orion/harness/tests/test_tool_provenance_audit.py \
  orion/harness/tests/test_harness_runner.py \
  orion/harness/tests/test_harness_grammar_emit.py \
  -q
20 passed, 1 failed (test_harness_runner_surfaces_fcc_error_code -- confirmed
pre-existing on unmodified main, unrelated to this diff: an error-code string
mismatch, "fcc turn timed out after 120.0s" != "fcc_timeout")

Full orion/harness/tests suite: 123 passed, 3 failed -- same pre-existing
failure above plus two unrelated Jinja UndefinedError failures in
test_grounding_capsule_consumers.py, both independently confirmed present
and identical on unmodified main. Zero regressions from this patch.
```

## Evals run

No dedicated eval harness exists for this seam. The regression tests above (particularly the edge-topology test for the grammar-graph fix below) function as the deterministic gate.

## Docker/build/smoke checks

Not run — this changes harness/schema Python only, no config read at boot, no new dependencies, no compose/port/worker changes.

## Review findings fixed

- Finding: `_IMMEDIACY_PATTERN`'s bare `\bcomputing\b` alternative (in the phase-1 commit) had no temporal/proximity qualifier, so it false-positived on any generic use of the word ("cloud computing costs are rising").
  - Fix: dropped that alternative. The incident's actual phrase ("computing in the background this turn") is still caught by "in the background"/"this turn" alone.
  - Evidence: `test_no_flag_for_generic_use_of_the_word_computing` regression test.
- Finding: `HarnessGrammarCollector.record_tool_provenance_mismatch` didn't update `self._last_completed_atom_id`, unlike its sibling terminal methods (`record_step_completed`, `record_step_failed`) — so the subsequent `record_result_assembled`'s own `derived_from` edge still pointed at the last *step* atom instead of the mismatch atom, leaving it a graph leaf with an incoming edge but no outgoing edge into the result-assembled/emitted chain, despite being semantically about the result.
  - Fix: one-line addition setting `self._last_completed_atom_id` after emitting the atom.
  - Evidence: `test_record_tool_provenance_mismatch_becomes_last_completed_atom`, which asserts the edge from the mismatch atom to the `exec_result_assembled` atom actually exists in the built event graph.

## Restart required

```text
No restart required to review/merge. Once merged, takes effect on the next
deploy/restart of whatever service runs HarnessRunner (orion-harness-governor).
```

## Risks / concerns

- Severity: Low
- Concern: This is a post-hoc audit only — by design, it cannot prevent a confabulation in the same generation (the FCC motor subprocess has already run to completion by the time this check runs). It also doesn't yet feed back into a future turn's context (no cross-turn continuity) or reinstate `external_tool_fetch` in `orion/schemas/context_provenance.py` (from PR #1056), since that touches a different subsystem (`services/orion-cortex-exec`'s chat `ctx` pipeline) and was deliberately scoped out of this patch.
- Mitigation: tracked as an explicit next step, not silently dropped — see `project_orion_substrate_bridge_confabulation` memory and the design-mode spec that preceded this patch.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1060
