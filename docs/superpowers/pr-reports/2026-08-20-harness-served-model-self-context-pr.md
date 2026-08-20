# PR Report: served-model self-context in the harness system prompt

## Summary

- Follow-up to PR #1755 (harness `response_identity`). That PR made the real
  served backend model persist to `chat_history_log.response_identity`, but
  tracing every consumer showed it's write-only: no code reads it back into
  a prompt, the PCR/recall continuity digest, or the 5a substrate finalize
  appraisal (`HarnessDraftMoleculeV1` has no identity/model field at all).
  Orion has zero self-knowledge of which real backend answered even one
  turn ago, despite the fact now existing in the DB.
- Juniper's proposal: inject it into the harness system prompt instead.
  Confirmed live that the exact seam already exists — `compile_harness_prefix`
  builds a "grounding self block" (Orion's existing "who am I" context)
  fresh every turn.
- The one real wrinkle: the prompt is built and sent *before* the CLI call
  happens, so PR #1755's discovery mechanism (reading the model off the
  completed response) is the wrong tool — it only knows what just happened.
  Reused a different, already-existing mechanism instead:
  `orion-llm-gateway`'s `GET /routes` already live-probes and caches
  (15s TTL) the real model per route for the Hub route picker
  (`route_catalog.py`'s `_probe_model`). Resolves `fcc_model_label` through
  `~/.fcc/.env` to a route key and reads that route's cached model —
  current, not one-turn-stale, and reuses an existing probe rather than
  adding a new one.

## Outcome moved

Every harness/unified-turn system prompt now carries a line like
`Backend model currently serving this turn: Qwen3.6-35B-A3B-UD-Q5_K_M12`
(fails open to no line, never a placeholder, when discovery can't resolve
a real model) — Orion goes from zero self-knowledge of its own serving
backend to having it fresh on every turn, in the same place its other
self-model context ("who am I", stance, autonomy state) already lives.

## Current architecture

`orion/harness/prefix.py::compile_harness_prefix` assembles the FCC motor's
system prompt: unified operator brief, grounding self block (gated on
`thought.grounding_capsule.identity_summary`), Thought imperative/stance
slice, autonomy slice, repair overlay, user message, MCP tool briefs. Called
from `orion/harness/runner.py::build_harness_prompt`, itself called from
`HarnessRunner.run()` — the same async context that already does one other
pre-prompt bus read (`read_last_tool_fetch`).

`orion-llm-gateway`'s `route_catalog.py` already exposes `GET /routes`,
live-probing each route's `/v1/models` (llama.cpp echoes the real served
weights file regardless of requested alias) and caching results for 15s —
built for the Hub's route picker, unused by anything harness-side before
this PR.

## Architecture touched

- `orion/harness/fcc_motor.py` — new probe function
- `orion/harness/prefix.py` — prompt assembly
- `orion/harness/runner.py` — `HarnessRunner`
- `services/orion-harness-governor/` — new env surface, requirements, README

## Files changed

- `orion/harness/fcc_motor.py`:
  - `_weights_file_basename()` (new): shared basename+extension-stripping
    helper, factored out of `_served_model_from_assistant` so both served-
    model paths (post-hoc discovery, pre-turn probe) use identical logic.
  - `_route_key_from_fcc_env_value()` (new): splits a `~/.fcc/.env` value
    like `"llamacpp/chat"` into `("llamacpp", "chat")`; `None` on anything
    without exactly that shape.
  - `probe_current_served_model()` (new, async): resolves `fcc_model_label`
    → route key → live-probed model via `GET {gateway_url}/routes`. Fails
    open to `None` on: no label, missing/malformed env entry, a
    non-llamacpp backend (`MODEL_HAIKU`'s `nvidia_nim` route isn't in this
    route table), unreachable gateway, non-2xx response, or an uncached
    route (worker down).
  - Added `import httpx`.
- `orion/harness/prefix.py`: `compile_harness_prefix` gains
  `current_served_model: str | None = None`; appends
  `"Backend model currently serving this turn: {model}"` to the grounding
  self block area when present, omits entirely when absent. Stays a
  pure/deterministic formatter per its own docstring — the live `/routes`
  call happens in the async caller, not here.
- `orion/harness/runner.py`:
  - `build_harness_prompt` threads `current_served_model` to
    `compile_harness_prefix`.
  - `HarnessRunner.__init__` gains an injectable `served_model_probe`
    (defaults to `probe_current_served_model`), same pattern as the
    existing `fcc_runner` injection point.
  - `HarnessRunner.run()`: the probe and the existing
    `read_last_tool_fetch` bus read now run concurrently via
    `asyncio.gather` (review fix — see below) instead of serially, wrapped
    in belt-and-suspenders `try/except` on top of the probe's own internal
    fail-open.
- `services/orion-harness-governor/.env_example`,
  `services/orion-harness-governor/app/settings.py`,
  `services/orion-harness-governor/docker-compose.yml`: new
  `HARNESS_LLM_GATEWAY_URL` (default `http://llm-gateway:8210`, same
  `app-net` hostname `orion-cortex-exec`/`orion-context-exec` already use).
  Read directly from the environment by `fcc_motor.py` (matching this
  service's existing convention for harness-motor-owned config); mirrored
  in `settings.py` for operator visibility only.
- `services/orion-harness-governor/requirements.txt`: added `httpx==0.27.2`
  (not previously a dependency of this service; matches the version pinned
  in `orion-llm-gateway`/`orion-cortex-exec`).
- `services/orion-harness-governor/README.md`: new "Served-model
  self-context" section (review fix — see below).
- Tests: `orion/harness/tests/test_fcc_motor_served_model.py` (extended:
  `_route_key_from_fcc_env_value` cases, `probe_current_served_model`'s
  success path and every fail-open path, mocked `httpx.AsyncClient`),
  `orion/harness/tests/test_harness_prefix.py` (2 new: line present/absent),
  `orion/harness/tests/test_harness_runner.py` (2 new: probe threaded into
  the built prompt; probe failure omits the line without breaking the turn).

## Schema / bus / API changes

None. No schema, channel, or API contract changed — pure prompt-construction
and config plumbing.

## Env/config changes

- Added keys: `HARNESS_LLM_GATEWAY_URL` (orion-harness-governor)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes
- local `.env` synced: **manually** — `scripts/sync_local_env_from_example.py`
  didn't pick up the new key because it reads `.env_example` from the
  primary checkout, not this worktree's uncommitted copy (known gap, see
  `feedback_env_sync_reads_example_from_primary_checkout` memory). Added
  the key by hand to both the worktree's and the primary checkout's live
  `services/orion-harness-governor/.env`, verified via `grep`.
- skipped keys requiring operator action: none

## Tests run

```text
ENABLE_PRE_TURN_APPRAISAL=false python3 -m pytest \
  orion/harness/tests/test_fcc_motor_served_model.py \
  orion/harness/tests/test_harness_prefix.py \
  orion/harness/tests/test_harness_runner.py \
  services/orion-harness-governor/tests \
  -q
# 83 passed, 1 pre-existing unrelated failure

ENABLE_PRE_TURN_APPRAISAL=false python3 -m pytest \
  orion/harness/tests services/orion-harness-governor/tests -q
# 237 passed, 5 failed
```

All 5 failures confirmed pre-existing and unrelated via `git stash`:
identical failures reproduce on unmodified main with this diff removed
(order-dependent flakiness across the larger suite + one grounding_status
formatting bug, both already documented from PR #1755's review — see
"Review findings fixed" below for why the grounding_status one specifically
was investigated again and still not touched).

## Evals run

No eval harness exists for `orion-harness-governor` or the `orion/harness`
package (same gap noted in PR #1755). Not added here — plumbing-adjacent
feature, covered by the unit/integration tests above.

## Docker/build/smoke checks

Not run. Pure Python + config changes to `orion-harness-governor` (already
running); no Dockerfile changes. `httpx` added to requirements.txt would
need a rebuild before deploy (new dependency, not just an env change) —
see Restart section.

## Review findings fixed

- Finding: the pre-turn served-model probe was awaited strictly after the
  `read_last_tool_fetch` bus read instead of running concurrently, adding
  up to the probe's full timeout (2s default) in serial latency to every
  harness turn.
  - Fix: both now run via `asyncio.gather` in `HarnessRunner.run()`.
  - Evidence: `orion/harness/runner.py` diff;
    `test_harness_runner_threads_served_model_probe_into_prompt` still
    passes with the gather in place.
- Finding: the new `HARNESS_LLM_GATEWAY_URL` env var wasn't documented in
  the service README, unlike other env vars documented there in prose.
  - Fix: added a "Served-model self-context" section.
  - Evidence: `services/orion-harness-governor/README.md` diff.
- Finding: `probe_current_served_model` re-reads and re-parses
  `~/.fcc/.env` from disk, on top of two pre-existing reads elsewhere in
  the same turn (`default_fcc_runner` and `run_fcc_turn` each already
  loaded it independently before this PR).
  - Not fixed. Threading the already-loaded env dict into the injectable
    `served_model_probe` seam would either complicate its single-argument
    test contract, or make a strict assertion on the passed dict
    non-deterministic across environments (a real `~/.fcc/.env` exists
    locally but not in CI). The file is small and this isn't a hot path —
    judged not worth the trade-off. Flagging as a legitimate, low-priority
    follow-up if it ever shows up in real latency data.
- Finding (re-surfaced from PR #1755's own review, in the exact function
  this PR adds new logic to): `grounding_status = apply_context_overflow_hint(error_msg)
  or error_code or error_msg or "failed"` discards `error_code` for any
  non-context-overflow error, because `apply_context_overflow_hint` returns
  its input unchanged (truthy) when the text isn't an overflow message —
  short-circuiting the `or error_code` fallback. Confirmed this makes the
  pre-existing `test_harness_runner_surfaces_fcc_error_code` fail on both
  the pre- and post-diff tree.
  - Investigated further before declining, rather than declining reflexively:
    checked every consumer of `grounding_status` and found
    `orion/hub/turn_orchestrator.py` reads it as **user-facing error text**
    for the Hub UI (`base["error"] = _with_overflow_hint(run.grounding_status)
    or run.grounding_status`), not just a machine-readable code. Reordering
    to prefer `error_code` first (which would flip the failing test to
    passing) would make the Hub UI show `"fcc_timeout"` instead of a
    readable sentence like `"fcc turn timed out after 120.0s"` — a real
    user-facing UX regression traded for a unit-test pass. Left alone,
    same call PR #1755's review made on the same finding; worth a
    dedicated, focused PR that resolves what `grounding_status` is
    actually supposed to be (short code vs. display text) rather than a
    drive-by reorder bundled into unrelated feature work.

## Restart required

```bash
# Rebuild required (new httpx dependency), not just restart:
docker compose \
  --env-file .env \
  --env-file services/orion-harness-governor/.env \
  -f services/orion-harness-governor/docker-compose.yml \
  up -d --build
```

## Risks / concerns

- Severity: low
- Concern: `probe_current_served_model`'s live `/routes` behavior against
  the real deployed gateway was not exercised end-to-end in this session
  (verified via unit tests with mocked `httpx`, plus a live `curl` to
  `/routes` in the prior investigation confirming the response shape this
  code parses). Recommend a real turn + a look at the captured harness
  prompt (or `HARNESS_FCC_LOG_FILE`/prefix logging, if enabled) to confirm
  the self-context line actually appears once deployed.
- Severity: low
- Concern: the deferred `grounding_status` bug above remains broken (not a
  regression from this PR, but flagged twice now). Worth a dedicated
  follow-up PR to decide its intended contract before fixing.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1763
