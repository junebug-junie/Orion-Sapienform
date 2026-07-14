# PR report: context-provenance registry for chat ctx keys

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1056
Branch: `feat/context-provenance-registry`

## Summary

- New `orion/schemas/context_provenance.py`: a registry classifying every ctx key that flows into a chat turn as one of `live_runtime_projection` / `derived_summary` / `memory_recall` / `static_identity_config` / `user_input` / `external_tool_fetch`, or `PLUMBING_KEYS` for routing bookkeeping with no cognitive content.
- `GroundingCapsuleV1.context_provenance` carries the classification through to `orion/harness/prefix.py`'s `compile_harness_prefix`, which renders it as a `CONTEXT PROVENANCE` block in the literal FCC motor prompt — the actual prompt for the agentic session with MCP tool access.
- `build_metacog_substrate_cue` tags its cue with the registry's source kind.
- `chat_stance.py` folds a standing "only these keys are live" hazard into the existing hazards mechanism.

## Outcome moved

Closes the mechanism behind a real incident: Orion narrated a GitHub-MCP file fetch (`mcp__github__get_file_contents` against `pressure.py`/`dynamics.py`/`attention_broadcast.py`) as live substrate computation "happening this turn." Investigation traced two failure modes: (1) the actual live substrate cue is a terse `self_state`/`eventfulness` string with no mechanism-level detail, and (2) nothing anywhere distinguished "live runtime signal" from "retrieved memory," "static config," or "content fetched via a tool call this turn" at the point Orion talks about itself. This PR closes (2).

## Current architecture

`executor.py` assembles ~80 keys into a single `ctx` dict across a 4000-line turn pipeline with no shared provenance metadata. `GroundingCapsuleV1` already carried a `provenance` dict, but only for `identity_source`/`pcr_ran`. `orion/schemas/cognition/answer_contract.py` already modeled evidence-typed claims (`repo_file`/`runtime_log`/etc.) but only for the investigation/technical verb lane (`supervisor.py`), never wired into casual chat (`executor.py`/`chat_stance.py`).

## Architecture touched

- `orion/schemas/context_provenance.py` (new): the registry + `PLUMBING_KEYS` exemption set + `classify()`.
- `orion/schemas/thought.py`: `GroundingCapsuleV1.context_provenance: dict[str, str]`.
- `services/orion-cortex-exec/app/grounding_capsule.py`: `context_provenance_for_ctx()` populates it.
- `orion/harness/prefix.py`: `_format_context_provenance_block()` renders it into `compile_harness_prefix`'s literal motor prompt.
- `orion/substrate/metacog_trigger_signals.py`: `build_metacog_substrate_cue()` tags its output.
- `services/orion-cortex-exec/app/chat_stance.py`: `_project_context_provenance_hazard()`, folded into the existing `_project_self_state_from_beliefs` hazards mechanism.

## Files changed

- `orion/schemas/context_provenance.py`: new registry (61 classified keys + 27 plumbing-exempt keys, exact coverage of the live `ctx` key snapshot).
- `orion/schemas/thought.py`: added `context_provenance` field to `GroundingCapsuleV1`.
- `services/orion-cortex-exec/app/grounding_capsule.py`: added `context_provenance_for_ctx()`, wired into `build_grounding_capsule()`.
- `orion/harness/prefix.py`: added `_format_context_provenance_block()`, wired into `_format_grounding_self_block()`.
- `orion/substrate/metacog_trigger_signals.py`: cue now tagged with `(source=live_runtime_projection)`.
- `services/orion-cortex-exec/app/chat_stance.py`: added `_project_context_provenance_hazard()`, folded into `build_chat_stance_inputs()`'s hazards list (prepended, not appended — see review findings).
- Tests: `orion/schemas/tests/test_context_provenance.py`, `services/orion-cortex-exec/tests/test_chat_stance_context_provenance_hazard.py`, plus extensions to `test_grounding_capsule_assembly.py`, `test_metacog_trigger_signals.py`, `test_harness_prefix.py`.

## Schema / bus / API changes

- Added: `GroundingCapsuleV1.context_provenance: dict[str, str]` (additive, default `{}`).
- Removed: none.
- Renamed: none.
- Behavior changed: `compile_harness_prefix`'s rendered motor prompt gains a new `CONTEXT PROVENANCE` block when the capsule carries any classified keys; `build_metacog_substrate_cue`'s output string gains a `(source=...)` suffix; `build_chat_stance_inputs`'s `social["hazards"]` list may gain one more entry.
- Compatibility notes: fully additive; existing consumers of `GroundingCapsuleV1` unaffected by the new field (default `{}`).

## Env/config changes

None. No new env keys added, removed, or renamed.

## Tests run

```text
PYTHONPATH=<worktree>:<worktree>/services/orion-cortex-exec venv/bin/python -m pytest \
  orion/schemas/tests/test_context_provenance.py \
  orion/substrate/tests/test_metacog_trigger_signals.py \
  orion/harness/tests/test_harness_prefix.py \
  services/orion-cortex-exec/tests/test_grounding_capsule_assembly.py \
  services/orion-cortex-exec/tests/test_chat_stance_context_provenance_hazard.py \
  services/orion-cortex-exec/tests/test_chat_stance_self_state_projection.py \
  -q
49 passed

Broader regression sweep (services/orion-cortex-exec/tests/test_chat_stance_*.py,
test_grounding_capsule*.py, test_router_grounding_capsule_metadata.py, orion/harness/tests):
223 passed, 7 failed -- all 7 confirmed present and identical on unmodified main
(verified by running the same batch against the unmodified main checkout):
  - 2 test-order-dependent failures (test_chat_stance_brief.py x1 additionally
    order-independent-broken, test_grounding_capsule_assembly.py x1,
    test_router_grounding_capsule_metadata.py x1 -- these 3 only fail when run
    as part of the larger batch, pass in isolation; identical on main)
  - 1 test-order-independent pre-existing failure
    (test_chat_stance_shadow_mode_uses_local_result_and_passes_observer,
    "assert 'continuity' in ['tick']")
  - 1 Jinja UndefinedError ('mind_coloring' is undefined) in
    test_grounding_capsule_consumers.py, x2 tests
  - 1 error-code string mismatch in test_harness_runner.py
    ("fcc turn timed out after 120.0s" != "fcc_timeout")
Zero regressions introduced by this patch.
```

## Evals run

No dedicated eval harness exists for this seam. The registry-coverage tests (`test_live_snapshot_fully_covered`, `test_static_ctx_assignments_covered`) serve as the deterministic gate in place of an eval — the latter is a live source-grounded check, not a static snapshot, and immediately caught 17 real unclassified keys during development (see Review findings below).

## Docker/build/smoke checks

Not run. This patch changes prompt-assembly Python only (schema, registry, string rendering) — no config read at boot, no new dependencies, no compose/port/worker changes.

## Review findings fixed

- Finding: `_project_context_provenance_hazard`'s hazard was appended after two prior `_unique(..., limit=8)` folds in `build_chat_stance_inputs`, so it silently dropped whenever the hazards list was already full at 8 unique entries — exactly on the eventful turns most likely to need it.
  - Fix: prepend instead of append, so it survives the cap regardless of how many other hazards already exist.
  - Evidence: `test_build_chat_stance_inputs_folds_provenance_hazard_when_live_key_present` passes; fix documented inline.
- Finding: `GroundingCapsuleV1.context_provenance` was populated by `context_provenance_for_ctx()` but had zero consumers anywhere in the codebase — write-only data, a schema+producer with no consumer (this repo's CLAUDE.md explicitly names that shape as the thing to avoid: "no keyword cathedral").
  - Fix: wired into `orion/harness/prefix.py`'s `_format_grounding_self_block`, the function that renders the capsule into the literal FCC-motor prompt (`compile_harness_prefix`, which spawns the `claude -p` agentic session with MCP tool access — the exact pathway implicated in the original incident, confirmed by the existing `test_compile_harness_prefix_includes_github_repo_when_mcp_enabled` test in the same file).
  - Evidence: `test_compile_harness_prefix_includes_context_provenance_when_capsule_has_it` / `..._omits_context_provenance_when_capsule_empty`.
- Finding: only the `self_state` clause of `build_metacog_substrate_cue`'s output got the `(source=...)` tag; the `execution_trajectory_projection`/`eventfulness` clauses in the same cue string were left untagged, despite the docstring claiming the whole cue "tags itself."
  - Fix: moved the tag to apply once, after all clauses are assembled, covering the full cue.
  - Evidence: `test_build_metacog_substrate_cue_tags_its_own_provenance`.
- Finding: `_project_context_provenance_hazard` duplicated the registry-scan-and-filter loop already implemented in `grounding_capsule.py`'s `context_provenance_for_ctx`, risking silent divergence between the capsule's provenance map and the hazard's live-key list if the filtering logic changes in one place and not the other.
  - Fix: `chat_stance.py` now imports and reuses `context_provenance_for_ctx` instead of re-scanning the registry independently.
  - Evidence: existing and new hazard tests pass after the refactor.
- Finding: `test_live_snapshot_fully_covered`'s hand-maintained key snapshot (`LIVE_CTX_KEY_SNAPSHOT`) has no automated link to the real ctx-assembly code in `executor.py`, so a future new ctx key only gets caught if an author remembers to update the frozenset by hand — the exact "deterministic gates over repeated yelling" failure mode this repo's CLAUDE.md warns against.
  - Fix: added `test_static_ctx_assignments_covered`, which statically parses `executor.py` for literal `ctx["key"] = ...` / `ctx.setdefault(...)` assignments and asserts registry+plumbing coverage against that parse — an automated safety net for the common case of a new direct assignment (doesn't replace the hand snapshot, still needed for keys set via dict merges in other modules, but removes the hand-maintenance burden for the common case).
  - Evidence: this test immediately surfaced 17 previously-unclassified keys during development (`chat_stance_brief`, `chat_stance_debug`, `collapse_entry`, `collapse_json`, `final_entry`, `journal_pageindex_context`, `metacog_ctx_trim_applied`, `metacog_draft_prompt_chars`, `metacog_draft_section_sizes`, `metacog_enrich_prompt_chars`, `metacog_enrich_section_sizes`, `metacog_entry_id`, `prior_chat_stance_brief`, `prior_stance`, `prior_step_results_by_service`, `speech_contract`, `world_context_capsule`), all now classified; `test_static_ctx_assignments_covered` passes.
- Finding: `"external_tool_fetch"` is defined in `ContextSourceKind` but never assigned to any registry entry or ctx key — the literal incident category (a GitHub MCP fetch) has no producer key anywhere, because harness-level MCP tool calls are invisible to this repo's `ctx` dict entirely.
  - Not fixed in this patch — genuine out-of-repo scope gap, disclosed under Risks/Concerns below rather than silently left uncovered. This was flagged as an open Missing Question during the design-mode spec that preceded this patch, before implementation started.

## Restart required

```text
No restart required for this patch to be reviewable/mergeable. Once merged,
takes effect on the next deploy/restart of cortex-exec and cortex-exec-chat
(prompt-assembly logic only, no config read at boot).
```

## Risks / concerns

- Severity: Medium
- Concern: `"external_tool_fetch"` has no producer anywhere in this repo. The original incident's actual mechanism — an agentic session calling `mcp__github__get_file_contents` mid-turn — happens at the harness/session level, outside this repo's own `ctx`-assembly pipeline entirely, so nothing here can tag that tool call's result with provenance at the source. This PR's `CONTEXT PROVENANCE` block in the FCC motor prompt (`orion/harness/prefix.py`) gives the motor a standing instruction about what *is* live before it reaches for a tool — a preventive nudge, not a detector — so there is no guarantee a future tool-fetched claim gets caught if the model doesn't attend to that instruction.
- Mitigation: flagged during design as requiring a harness-level fix outside this repo's addressable scope (see the `project_orion_substrate_bridge_confabulation` memory and the preceding design-mode spec). Follow-up: instrument the harness/session layer's MCP tool-result formatting to tag fetched content with provenance at its actual source, if and when that layer is addressable from this project.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1056
