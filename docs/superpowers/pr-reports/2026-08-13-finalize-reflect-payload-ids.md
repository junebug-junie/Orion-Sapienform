# fix(harness): inject finalize-reflection identity fields on the path that actually runs

## Summary

- A live FCC turn died with 4 pydantic errors on `FinalizeReflectionV1`: `correlation_id`, `thought_event_id`, `substrate_appraisal_id`, `draft_hash`, all `Field required`.
- Root cause: `run_finalize_reflection()` injected those four server-owned fields guarded by `if isinstance(raw_payload, dict)`, but `extract_finalize_reflection_payload()` returns model **text** (a `str`) on the normal path. The guard was never true on a real turn — the injection has been dead code since `40cf980bf` (2026-07-05).
- Fix: normalize `str -> dict` with `parse_json_object()` **before** injecting, and move extract+inject+parse inside the existing `try` so genuinely malformed payloads fail closed to the degraded reflection instead of raising out of the turn.
- Changed `setdefault` to plain assignment. `setdefault` let a model-supplied value win — harmless while the block was dead, a live silent-corruption path the moment it starts working.
- Removed the prompt block that told the model to emit those four fields, since the harness owns them.

## Outcome moved

The finalize reflection pass completes instead of killing the turn whenever the quick lane is blocked. Previously every such turn raised out of `run_finalize_reflection()`.

## Current architecture

`orion/harness/finalize.py::run_finalize_reflection()` runs the substrate-informed finalize reflection. Most turns short-circuit at the quick lane (`maybe_quick_lane_verdict`, line ~344) and never reach the LLM path — which is why a 5-week-old defect only surfaced now.

On the LLM path, cortex-exec returns a payload, `extract_finalize_reflection_payload()` pulls the model text out of it, and the result is validated into `FinalizeReflectionV1`. The model is asked for the judgement fields only; the four identity fields are the harness's to supply.

## Architecture touched

One function's payload-normalization order, one prompt template, one test module. No schema, bus channel, or env key changed.

## Files changed

- `orion/harness/finalize.py`: normalize `str -> dict` before injection; move inject+parse inside the guarded `try`; `setdefault` -> assignment; distinct log token for unparseable payloads vs gateway failures; import `ValidationError`.
- `orion/cognition/prompts/harness_finalize_reflect.j2`: dropped `METADATA (include from inputs)` naming the four identity fields; replaced with an explicit instruction not to emit them.
- `orion/harness/tests/test_finalize_reflect_llm_fallback.py`: three new regression tests.

## Schema / bus / API changes

- Added / Removed / Renamed: none
- Behavior changed: a payload that fails to parse or validate now degrades to `degraded_llm_failure_fallback` / `misaligned` instead of raising out of the turn. Model-supplied values for the four identity fields are now overwritten rather than honored.
- Compatibility notes: `FinalizeReflectionV1` itself is unchanged. Prompt change is additive-safe — a model that still emits the four fields is simply overridden.

## Env/config changes

None. No `.env_example` touched, so no sync required.

## Tests run

```text
cd /mnt/scripts/Orion-Sapienform-finalize-reflect-payload-ids
PYTHONPATH=. pytest orion/harness/tests -q
-> 3 failed, 174 passed in 4.71s
```

The 3 failures (`test_grounding_capsule_consumers` x2, `test_harness_runner_surfaces_fcc_error_code`) fail identically on unmodified `main` — pre-existing, unrelated, confirmed by running them there.

All three new tests verified failable, not tautological:

```text
# against unpatched finalize.py
FAILED ::test_reflect_text_payload_gets_server_owned_ids_injected
FAILED ::test_reflect_malformed_payload_uses_degraded_reflection

# with setdefault restored in place of assignment
FAILED ::test_reflect_server_owned_ids_overwrite_model_supplied_values
```

Standalone reproduction of the original crash, before the fix:

```text
returned type: str
isinstance(raw, dict) -> False          <-- the dead guard
VALIDATION FAILED, missing: ['correlation_id', 'thought_event_id',
                             'substrate_appraisal_id', 'draft_hash']
```

## Evals run

No eval harness covers `orion/harness/finalize.py`. The load-bearing evidence here is the reproduction above plus the three failable regression gates.

## Docker/build/smoke checks

**UNVERIFIED at runtime.** `/app` is baked into the `orion-harness-governor` image, so this does not take effect until a rebuild. Not yet observed on a live turn.

## Review findings fixed

- Finding (SHOULD FIX): `setdefault` let a model-supplied value win for the four identity fields — dormant while the block was dead, live silent corruption once it works. `_reflection_id()` and `_verdict_molecule_id()` hash the reflection's copies, while sibling fields on the same outcome molecule come from the real appraisal object, so one transcribed-wrong `draft_hash` silently decorrelates them with nothing visible at runtime. Worse, `harness_finalize_reflect.j2:39-40` actively instructed the model to emit all four.
  - Fix: plain assignment; deleted the prompt block; added `test_reflect_server_owned_ids_overwrite_model_supplied_values`.
  - Evidence: new test fails with `setdefault` restored, passes with assignment.

- Finding (SHOULD): moving the parse inside a bare `except Exception` means a schema regression of our own (e.g. a `FinalizeReflectionV1` field rename) would degrade *every* turn to `misaligned` under an "llm_failed" label instead of failing loudly — and `misaligned` drives `finalize_changed=True` and `surprise_resolved=False` into the post-turn closure, so a code bug would quietly perturb the substrate loop.
  - Fix: `ValidationError`/`ValueError` now log `harness_finalize_reflect_payload_unparseable_using_degraded_reflection`, distinct from the gateway-failure token, so a schema regression is greppable on its own.
  - Evidence: `unparseable` branch at `finalize.py:405`.

- Finding (NIT): the new test's docstring claimed the model is never asked for the four fields, which the live prompt contradicted.
  - Fix: corrected the docstring; the prompt change makes the claim true.

- Reviewer verification worth recording: `parse_json_object()` is byte-identical to the call the old code already made on the same string (it was reached via `parse_finalize_reflection_payload`), just relocated — 10 realistic LLM output shapes (bare, ```json fences, prose preamble, trailing comma, Python `False`, double-encoded) produced **0 divergences** between old and new paths. So no previously-working reflection can be converted to degraded by this change.

## Known issues NOT fixed here

- `finalize.py:387-399` — the retry `maybe_quick_lane_verdict()` inside the `except` is dead: `quick_lane_block_reason()` is a pure function of the same three objects and already returned `None` earlier in the function, so `degraded is not None` can never be true. Pre-existing, not introduced here. Flagged because the new malformed-payload test's assertion silently depends on it staying dead.
- `test_harness_finalize_chain.py:59` and `test_finalize_failure_closure.py:132` both mock `extract_finalize_reflection_payload` with `reflection.model_dump(mode="json")` — a dict already containing all four ids. That is precisely why this bug survived 5 weeks: every integration-level test fed the one shape the real path never returns. They still pass and still encode the wrong contract. Worth switching one to return JSON text.

## Restart required

`/app` is baked into the image, and `scripts/safe_docker_build.sh` refuses to run from the shared/primary checkout, so rebuild from a worktree:

```bash
# after merge, refresh the primary checkout (containers bind-mount it)
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only

# rebuild FROM A WORKTREE
cd /mnt/scripts/Orion-Sapienform-finalize-reflect-payload-ids
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

## Risks / concerns

- Severity: low
  - Concern: malformed payloads now degrade silently where they previously raised. A loud failure is easier to notice than a quiet `misaligned`.
  - Mitigation: the distinct `payload_unparseable` log token makes the quiet path greppable, and `misaligned` is the fail-closed verdict by design (`dbbcf6131`).

- Severity: low
  - Concern: `UNVERIFIED` until the governor image is rebuilt and a real quick-lane-blocked turn is observed.
  - Mitigation: reproduction + three failable gates cover the logic; only the deployment step is outstanding.

## Status

DONE_WITH_CONCERNS — code fix and gates complete and green; live path unverified until rebuild.
