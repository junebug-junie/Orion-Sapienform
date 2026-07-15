# PR report: cross-turn tool-fetch continuity (same-service redesign)

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1065
Branch: `feat/cross-turn-tool-fetch-continuity`

## Summary

- `orion/harness/last_tool_fetch_cache.py`: `publish_last_tool_fetch()` (write, end of a turn) and `read_last_tool_fetch()` (read, start of the next turn), both keyed by `thought.session_id`, both fail-open.
- `HarnessRunner.run()` reads at the start (before prompt compilation) and writes at the end (success path only).
- `compile_harness_prefix()` renders "Last turn you fetched content via tool: X, Y" directly into the FCC motor's own prompt when present.

## Outcome moved

Gives a harness-mode conversation memory of its own recent tool use across turns — closing the disclosed gap from PR #1060 (the tool-provenance audit), which could detect a same-turn mismatch but had no way to carry that fact forward.

## This is a rebuild — the design changed mid-flight, twice

**First attempt** wired the cache cross-service: write in `orion-harness-governor`, read in `orion-cortex-exec`, on the theory that a cortex-exec chat turn is "what comes next" after a harness turn in the same conversation.

- **Review round 1** caught real bugs in that version (hazard-string truncation past its own warning clause, a hazard-eviction regression, a metacog-cue truncation bug, a false-liveness classification on empty-dict fallbacks, a confirmed per-turn double registry scan) — all fixed at the time.
- **Review round 2** traced the actual dispatch path end-to-end and found the whole premise was wrong: `orion/hub/turn_orchestrator.py::run_unified_turn` (harness path) and `cortex-exec`'s `chat()` RPC are mutually exclusive per-message branches in `services/orion-hub/scripts/websocket_handler.py` — confirmed directly by Juniper and then verified in source (`client_mode == "orion"` hits `continue` right after the harness call, never reaching the cortex-exec branch). The two services are never sequential turns of one conversation, so the write-key (`session_id`, stable per-conversation) and read-key (a fresh per-turn UUID) could never have matched regardless of any bug fix.

**This version** is rebuilt entirely within `orion/harness/` — same service for both sides, same `thought.session_id` key on both. The session_id-stability assumption was verified end-to-end via source trace this time, not just asserted: backend (`services/orion-hub/scripts/websocket_handler.py:718`, read fresh from client-sent data on every message, no server-side regeneration) *and* frontend (`services/orion-hub/static/js/app.js`'s `orionSessionId`, loaded once from `localStorage` at page load and resent unchanged on every WS message).

**Review round 3** on this corrected design confirmed the session_id theory holds end-to-end, and caught two more real bugs, both fixed (see Review findings below).

## Current architecture

`orion/harness/fcc_motor.py::run_fcc_turn` spawns the FCC motor as a single-shot `claude` subprocess. `HarnessRunner.run()` (`orion/harness/runner.py`) drives one turn: compiles the prompt via `compile_harness_prefix`, streams the subprocess's events into `GrammarReceiptV1`s, captures `draft_text`. Prior work (PR #1060) added `detect_tool_provenance_mismatch()`, a same-turn post-hoc audit with no cross-turn memory.

## Architecture touched

- `orion/harness/last_tool_fetch_cache.py` (new)
- `orion/harness/runner.py`
- `orion/harness/prefix.py`
- `orion/harness/tool_provenance_audit.py` (one function made public: `fetch_shaped_tool_names`)

## Files changed

- `orion/harness/last_tool_fetch_cache.py`: new, `publish_last_tool_fetch()` + `read_last_tool_fetch()`.
- `orion/harness/runner.py`: reads at the start of `run()`, writes at the end (success path, explicitly excluding the error/partial-draft path via a new `error_path_taken` flag).
- `orion/harness/prefix.py`: `compile_harness_prefix()` gains `prior_tool_fetch_names: list[str] | None = None`, rendered as one line when present.
- `orion/harness/tool_provenance_audit.py`: `_fetch_shaped_tool_names` → public `fetch_shaped_tool_names` (no behavior change).
- Tests: `orion/harness/tests/test_last_tool_fetch_cache.py` (new, 14 tests covering write/read/round-trip/malformed-payload/element-validation/never-raises), extensions to `test_harness_runner.py` (6 new tests including the read-wiring, the partial-draft-error exclusion, and the session_id-keying assertion) and `test_harness_prefix.py` (2 new rendering tests).

## Schema / bus / API changes

- None. No new schema, no new bus channel — plain Redis key via the existing `OrionBusAsync.redis` property both services already hold.
- `build_harness_prompt()`/`compile_harness_prefix()` gain one new optional kwarg each (`prior_tool_fetch_names`), additive, defaults to `None`.

## Env/config changes

None.

## Tests run

```text
PYTHONPATH=<worktree> venv/bin/python -m pytest orion/harness/tests -q
145 passed, 3 failed -- all 3 confirmed pre-existing and identical on
unmodified main (2 unrelated Jinja UndefinedError failures in
test_grounding_capsule_consumers.py, 1 unrelated error-code string
mismatch in test_harness_runner_surfaces_fcc_error_code). Zero
regressions from this patch.

services/orion-cortex-exec side verified back to clean baseline after
reverting the first (wrong) design: 15 passed, no lingering changes
(`git status --short services/orion-cortex-exec/` empty before commit).
```

## Evals run

No dedicated eval harness exists for `orion/harness/` covering prompt-content behavior (i.e. whether the FCC model actually uses the injected continuity line correctly without over-claiming immediacy from it). Flagged as a gap, not silently skipped — only unit tests cover this patch.

## Docker/build/smoke checks

Not run — Python-only change to prompt assembly and a Redis read/write already using the existing shared bus connection; no config read at boot, no new dependencies, no compose/port changes.

## Review findings fixed

**Round 1 (first, cross-service design — later reverted entirely; findings still worth recording since the same bug class could recur elsewhere):**
- Hazard string truncated to 140 chars by an existing `_unique()` helper, cutting off its own warning clause.
- Hazard-eviction regression from a naive fix to the above.
- Metacog-cue provenance tag truncated on eventful turns.
- `key in ctx` presence-check misclassified empty-dict fallbacks as live.
- Confirmed per-turn double scan of the provenance registry.
(All fixed in the reverted design; superseded by this patch's same-service rebuild, which doesn't touch that registry at all.)

**Round 2 (architecture correction):**
- Finding: the write-key (`session_id`) and read-key (`chat_correlation_id`, unpopulated, falling back to a fresh per-turn UUID) never match — the two services aren't sequential turns of one conversation at all.
  - Fix: full redesign, this PR.

**Round 3 (this design):**
- Finding: the write guard (`if not draft_text: return`) only excludes the fully-empty-draft case; a turn that errors/times out with a salvaged partial draft falls through non-empty and still records a tool-fetch for a turn the user may not have fully seen — contradicting the code's own inline comment.
  - Fix: explicit `error_path_taken` flag, set in the `error` event branch, gates the write regardless of whether a partial draft exists.
  - Evidence: `test_harness_runner_does_not_publish_last_tool_fetch_on_partial_draft_error`.
- Finding: `read_last_tool_fetch` validated `tool_names` was a list but not that its elements were strings — a malformed cache row would crash `prefix.py`'s `", ".join(...)` uncaught, breaking the whole next turn over one bad Redis row.
  - Fix: element-level `isinstance(name, str)` validation.
  - Evidence: `test_read_returns_none_when_tool_names_has_non_string_element`.
- Finding (disclosed, not fixed): `session_id` is loaded once from browser `localStorage` and shared across every tab on the same origin — the same value `websocket_handler.py` already flags as cross-tab-shared for a different reason (its own turn-cancel registry deliberately avoids keying on it). Two tabs on the same session will cross-contaminate this cache.
  - Not fixed: low severity (continuity confusion, not a content/security leak), and multi-tab-same-session may not be a supported usage pattern at all. Documented prominently in `last_tool_fetch_cache.py`'s module docstring for whoever picks up a proper per-tab/per-chain key if that assumption turns out wrong.

## Restart required

```text
No restart required to review/merge. Once merged, takes effect on the next
deploy/restart of orion-harness-governor.
```

## Risks / concerns

- Severity: Low
- Concern: the cross-tab session_id sharing described above.
- Mitigation: disclosed in code and here; not a supported-usage-pattern question this patch can resolve alone.
- Severity: Low
- Concern: no eval harness exercises whether the FCC model actually uses the injected continuity line correctly (vs. ignoring it or over-interpreting it).
- Mitigation: flagged as a gap; unit tests cover the mechanism, not model behavior.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1065
