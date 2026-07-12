# PR: Clamp overlong stance_react imperative/tone instead of fail-closing the turn

## Summary

- `ThoughtEventV1.imperative` enforced a hard `Field(max_length=300)`. When the stance_react LLM overshot it, `ThoughtEventV1.model_validate` raised inside `parse_stance_react_payload`, and `services/orion-thought/app/bus_listener.py`'s generic `except Exception` converted the whole turn into a fail-closed defer via `build_stance_react_failure_thought` — discarding an entire already-paid brain-tier LLM turn over a few stray characters.
- `imperative` is not the user's message — it's an LLM-generated "efference copy" felt-layer directive (`orion/cognition/prompts/stance_react.j2:69`), concatenated directly into the FCC motor prompt alongside the full, uncapped `user_message` (`orion/harness/prefix.py:106,118`). The cap itself is a deliberate design constraint to keep it terse, not a content-loss bug — confirmed during review that raw user input was never at risk.
- `normalize_stance_react_raw` now clamps `imperative`/`tone` (logging a warning when it fires) before Pydantic validation runs, so overflow degrades gracefully instead of dropping the turn.
- Schema cap bumped 300→400 as headroom (the prompt still instructs the LLM to target 300 chars — 400 is enforcement slack, not a new target).
- Added a regression test pinning the clamp constants (`IMPERATIVE_MAX_LEN`, `TONE_MAX_LEN`) to the schema's actual `max_length`, so the two can't silently drift apart and reintroduce the bug.

## Outcome moved

An oversized `imperative`/`tone` field from the stance_react LLM call no longer discards the whole turn's stance output (fail-closed defer, empty imperative, dropped stance slice); it truncates and logs instead, preserving the rest of the already-computed turn.

## Current architecture

`orion/thought/stance_react.py:normalize_stance_react_raw` coerces LLM JSON into typed strings before `ThoughtEventV1.model_validate` runs — handling nulls, bools, wrong types — but previously did no length clamping, relying entirely on Pydantic's hard `max_length` to reject rather than truncate. `stance_react` runs on the full brain-tier "chat" LLM lane (`services/orion-cortex-exec/app/llm_lane.py:34-92` resolves it to `llm_lane="chat"`, high priority), never the deterministic quick lane (`docs/superpowers/specs/2026-07-05-orion-unified-turn-design.md:51,765` — quick lane is explicitly disallowed for stance_react), so a rejected turn here always throws away a real paid LLM call.

## Files changed

- `orion/schemas/thought.py`: `ThoughtEventV1.imperative` max_length 300→400.
- `orion/thought/stance_react.py`: new `_coerce_and_clamp_str` helper (wraps `_coerce_required_str`, truncates and logs on overflow); wired into `normalize_stance_react_raw` for `imperative` (400) and `tone` (200); `correlation_id` extracted up front for log correlation.
- `orion/thought/tests/test_stance_react_pipeline.py`: 3 new tests — clamp behavior for overlong `imperative`, clamp behavior for overlong `tone`, and a drift guard asserting the clamp constants match `ThoughtEventV1.model_fields[...]`'s live schema constraint.

## Schema / bus / API changes

- Behavior changed: `ThoughtEventV1.imperative` max_length 300 → 400 (schema-level cap; not a bus channel or event-shape change).
- Compatibility notes: checked `orion/schemas/harness_finalize.py:111`'s `stance_imperative` field (also capped at 300) — its only writer, `orion/harness/finalize.py:1036-1039`, independently truncates via its own `_excerpt(text, max_len=300)` before assignment, so it can't overflow regardless of this change. No other consumer in the codebase assumes `imperative` stays ≤300 chars.

## Env/config changes

None.

## Tests run

```text
.venv/bin/python -m pytest orion/thought/tests orion/schemas/tests/test_thought.py -q
→ 39 passed
```

Confirmed via `git stash` that these pre-existing failures on `main` are unrelated to this diff:
```text
services/orion-thought/tests/test_reverie_spontaneous_thought.py (3 tests)
services/orion-thought/tests/test_settings_salience_flags.py::test_salience_flags_default_off
orion/harness/tests/test_grounding_capsule_consumers.py (2 tests)
orion/harness/tests/test_harness_runner.py::test_harness_runner_surfaces_fcc_error_code
```

`git diff --check`: clean.

## Evals run

None applicable — no eval harness exists for this seam; this is a schema-validation clamp fix with deterministic unit coverage, not a quality/behavior judgment call.

## Docker/build/smoke checks

Not applicable — pure Python logic change in `orion/thought` and `orion/schemas`, no config, dependency, or runtime-boot surface touched.

## Review findings fixed

Two-angle subagent review (correctness: line-scan/removed-behavior/cross-file-tracing; cleanup/altitude/conventions) plus self-verification.

- Finding: `IMPERATIVE_MAX_LEN`/`TONE_MAX_LEN` in `stance_react.py` duplicated the schema's `Field(max_length=...)` in `thought.py` as two independent hardcoded numbers.
  - Fix: added `test_stance_react_clamp_lens_match_schema_max_length`, which introspects `ThoughtEventV1.model_fields[...]` at test time and asserts the clamp constants match the live schema constraint — a future edit to one without the other now fails CI instead of silently reintroducing the fail-closed bug.
  - Evidence: `.venv/bin/python -m pytest orion/thought/tests/test_stance_react_pipeline.py -q` → 16 passed, including the new drift-guard test.
- Finding: `_coerce_and_clamp_str`'s truncation duplicates `orion/harness/finalize.py`'s `_excerpt` helper (same `text[:max_len]` idea, different field).
  - Fix: not changed — the two live in different layers (`orion.harness` vs `orion.thought`) with different signatures (logging, field name, correlation id vs a bare default), and sharing a one-line slice across modules would add cross-layer coupling for no real benefit. Judged not material.
  - Evidence: confirmed both call sites and signatures directly; no existing shared string-clamp utility module in the repo to route through instead.
- Finding: verified `correlation_id` is actually populated at the point `normalize_stance_react_raw` reads it (not `None` in the real path).
  - Fix: not needed — confirmed `parse_stance_react_payload` calls `raw.setdefault("correlation_id", correlation_id)` before `normalize_stance_react_raw(raw)` runs, so the log line sees the real correlation id on the only production call site (`services/orion-thought/app/bus_listener.py:305`).

## Restart required

No restart required. This changes pure Python validation/coercion logic in a library module consumed at request time by `services/orion-thought`; the running service picks it up on its next normal deploy/restart cycle, no forced restart needed for correctness.

## Risks / concerns

- Severity: low
- Concern: `_coerce_and_clamp_str` truncation is a plain character-count slice; if `imperative`/`tone` ever carry meaningful trailing content (e.g. a closing instruction), it could be cut mid-thought.
- Mitigation: this only fires when the LLM already exceeded the prompt's instructed 300-char target — already a prompt-discipline miss — and truncation is strictly better than the prior behavior of discarding the entire turn. The new `logger.warning` gives visibility if this starts firing often, which would be a signal to revisit the prompt rather than the clamp.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/stance-react-imperative-clamp
