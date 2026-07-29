# orion-heartbeat's H1 verdict into AST/HOT — design spec

Status: **implemented, this patch.** Touches AST/HOT self-modeling instrumentation, gated by
CLAUDE.md §0A's proposal-mode requirement — Juniper directly asked to give orion-heartbeat a real
consumer, then explicitly chose this path (see conversation: AST/HOT was picked over CollapseMirror's
still-unbuilt `insight`/`flow` gates and over `orion-equilibrium-service`, once
`docs/superpowers/specs/2026-07-29-ast-hot-reducer-live-ticking-design.md`/PR #1459 confirmed AST/HOT
now ticks live and persists durably).

## Arsonist summary

`orion-heartbeat`'s tensor-network ensemble (docs/superpowers/specs/2026-07-24-spark-field-
holographic-lattice-design.md, PR #1439/#1441, live-verified 2026-07-29 in `5d195f822`) computes a
real, independently-derived boundary/bulk entanglement verdict, but had zero consumers — read-only,
publishing nowhere, by design (CLAUDE.md's empty-shell-cognition gate: don't wire a non-discriminating
signal into anything real). Separately, AST/HOT (`reduce_attention_self_model()`) just gained a real
live tick and durable table (PR #1459) but is itself not consumed by anything yet either.

This patch connects the two: `orion-substrate-runtime`'s `_attention_self_model_tick()` now fetches
`orion-heartbeat`'s `/h1` and threads it into the reducer as an additive input, populating three new
`AttentionSelfModelV1` fields. Neither signal is being asked to explain the other — heartbeat's
tensor-network ratio and AST/HOT's prediction-error domains are genuinely independent measurements,
not aliased versions of one thing (same independence-check discipline CLAUDE.md's metric quality gate
requires for any new signal joining a model).

## Current architecture (before this patch)

- `orion-heartbeat`: `/h1` returns `{"ok": bool, "h1": {"mean_ratio", "verdict", "tick_count", ...}}`
  once its first ensemble tick completes. No consumer anywhere in the repo.
- AST/HOT: `reduce_attention_self_model()` (pure, no I/O) took `prediction_error_by_domain`/
  `prediction_error_trend_by_domain` as its only real-signal inputs. `_attention_self_model_tick()`
  (`services/orion-substrate-runtime/app/worker.py`) supplies those from FalkorDB node metadata
  already in-hand from the same tick's broadcast computation — zero extra store round-trip for that
  part.

## What this patch does

- `orion/schemas/attention_self_model.py`: three new additive fields — `heartbeat_mean_ratio`
  (`float | None`), `heartbeat_verdict` (`Literal["redundant","concentrated","mixed"] | None`),
  `heartbeat_basis` (`str`).
- `orion/substrate/attention_self_model.py`: new optional `heartbeat_h1: dict | None` param on
  `reduce_attention_self_model()`, plus `_heartbeat_h1_fields()` — fails open to
  `(None, None, "")` on anything malformed (missing keys, wrong types, an unrecognized verdict
  string), same discipline as every other optional input in this reducer. Still no I/O in this
  module — caller-supplied, matching its own design.
- `services/orion-substrate-runtime/app/worker.py`: new `_fetch_heartbeat_h1()` — one synchronous
  `requests.get` (same blocking-call-inside-an-async-tick pattern this file's other synchronous SQL
  reads already use), short timeout, fails open to `None` on any exception, disabled entirely
  (returns `None` without a request) when `SUBSTRATE_HEARTBEAT_H1_URL` is unset. Wired into
  `_attention_self_model_tick()` right alongside the existing field-lane fetch.
- New settings: `SUBSTRATE_HEARTBEAT_H1_URL` (empty disables the fetch entirely; defaults to the
  real `http://orion-athena-heartbeat:7251/h1` for this deployment, promoted to the active
  `.env_example` default after live verification — see "2026-07-29 env-default incident" below),
  `SUBSTRATE_HEARTBEAT_H1_FETCH_TIMEOUT_SEC` (default `2.0`).

## 2026-07-29 env-default incident (same day, post-merge-prep)

While live-verifying this patch, a concurrent, unrelated session (`chore/gpu-pressure-verify`, a
different worktree) ran its own `docker compose up` for `orion-substrate-runtime`, recreating the
shared container from its own worktree's `.env` — which predated this patch and had neither
`SUBSTRATE_ATTENTION_SELF_MODEL_TICK_ENABLED=true` (PR #1459's own flag, which had only ever been
manually flipped on in the live container, never promoted to `.env_example`'s active default) nor a
real `SUBSTRATE_HEARTBEAT_H1_URL`. The redeploy reused the already-built image (this patch's code was
intact and unaffected — confirmed via `docker exec ... grep` against the running container's files),
but silently reverted both flags to their `.env_example`/`docker-compose.yml` fallback defaults,
killing the self-model tick entirely for over an hour with no error — only a missing
`substrate_attention_self_model_tick_completed` log line, the same "ticks alive elsewhere, this one
silently frozen" failure shape CLAUDE.md's metric quality gate warns about.

Juniper's call: promote both flags to their active default in `.env_example` *and*
`docker-compose.yml`'s own inline `${VAR:-default}` fallback (not just this one container's `.env`) —
the fallback matters because a future worktree with no `.env` for this service at all (this repo's
own `new_worktree.sh` does not provision one) falls through to the compose file's inline default, not
`.env_example`. Both are now `true`/the real heartbeat URL. This does reopen PR #1459's own original
"default-off, flip only after a live-data sanity check" framing for
`SUBSTRATE_ATTENTION_SELF_MODEL_TICK_ENABLED` specifically — that sanity check happened (this doc's
own live-verification section, plus PR #1459's own), and Juniper explicitly signed off on promoting it
here, so this is the intentional next step of that same precedent, not a bypass of it.

## Non-goals

- **Not building the CollapseMirror `insight`/`flow` gates** in `orion-equilibrium-service`. Those
  remain a separately-scoped follow-up with their own unresolved Missing Questions (flow-state signal
  choice, cooldown cadence, draft-mapping) — bundling them here would repeat the "don't build two
  speculative things at once" mistake this repo's MQ3 thread already called out.
  `AttentionSelfModelV1` (including these new fields) still has no bus channel and no downstream
  consumer after this patch — additive-only, same status its own prediction-error fields had before
  PR #1459 gave the whole model a live tick.
- Not reconciling or unifying heartbeat's verdict with AST/HOT's own `confidence`/
  `prediction_error_confidence` — the two stay independent, side-by-side fields. No new combined
  score.
- Not changing `_brain_frame_tick()`'s narrower, separate self-model computation (still
  `field_frame=None`, no heartbeat, no trend) — intentionally left un-unified, per PR #1459's own
  documented reasoning.

## Acceptance checks

1. Reducer unit tests (`orion/substrate/tests/test_attention_self_model.py::TestHeartbeatH1`) cover:
   real payload populates fields; `None` input, `{"ok": false, ...}`, missing `verdict`, an
   unrecognized verdict string, and a non-dict input all fail open to honest defaults; the new fields
   don't perturb `prediction_error_confidence` or vice versa.
2. Worker tests (`services/orion-substrate-runtime/tests/test_worker_attention_self_model_tick.py`)
   cover: disabled-by-default (no request attempted), a successful fetch reaching the persisted model,
   a `{"ok": false}` response, and — critically — an unreachable heartbeat service still lets the tick
   persist normally with honestly-absent heartbeat fields.
3. Live: with `SUBSTRATE_HEARTBEAT_H1_URL` pointed at the real running `orion-heartbeat` container,
   `substrate_attention_self_model` rows show real, non-null `heartbeat_mean_ratio`/`heartbeat_verdict`
   values matching what `/h1` is independently reporting at the same time (verified via direct
   Postgres query, not just container health).

## Files changed

- `orion/schemas/attention_self_model.py`
- `orion/substrate/attention_self_model.py`
- `orion/substrate/tests/test_attention_self_model.py`
- `services/orion-substrate-runtime/app/worker.py`
- `services/orion-substrate-runtime/app/settings.py`
- `services/orion-substrate-runtime/tests/test_worker_attention_self_model_tick.py`
- `services/orion-substrate-runtime/.env_example` + synced local `.env`
- `services/orion-substrate-runtime/docker-compose.yml`
- `services/orion-substrate-runtime/README.md`
