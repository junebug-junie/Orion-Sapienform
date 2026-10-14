# Visual reverie episode-memory seed

## Status

DONE_WITH_CONCERNS pending merge and deployment. The code path, focused tests,
eval, live read query, env sync, and container build are verified. A real
post-deploy visual-chain row remains UNVERIFIED because this patch does not
deploy or trigger a GPU image without an allocator decision.

## Problem

Visual reverie still read the retired `memory_crystallizations` source. The
new governed `episode_memory` type therefore never entered image prompts or
their typed source receipts.

Pre-change live evidence on 2026-10-14:

- 68 `episode_memory` rows existed; 25 passed the initial Stage 2 policy audit.
- 144 visual runs existed since 2026-10-02.
- 0 had `context_selection.source_kind='episode_memory'`.
- 86 used the legacy `memory` kind, representing only 3 distinct legacy ids.

## Changes

- Added `load_episode_memory_seed()` with the approved Stage 2 eligibility
  boundary: recent active `happened`/`about_juniper` memories, unexpired, all
  evidence verified, no disqualifying event, and low/auto or confirmed stakes.
- Rotated over eligible memories by least recent visual use, with never-used
  memories first.
- Made the reader default-on via
  `REVERIE_VISUAL_MEMORY_SEED_SOURCE=episode_memory`; retained
  `crystallization` as a bounded rollback.
- Recorded the selected memory as
  `context_selection.source_kind='episode_memory'` plus its exact `memory_id`.
  `memory_source_kind` also identifies a loaded memory slot when another slot
  wins round-robin selection.
- Preserved historical decoding for old `source_kind='memory'` rows and frozen
  plans written before `memory_source_kind` existed.
- Updated the Hub API/cockpit to label current episode memories separately from
  legacy crystallizations.
- Added the new env family to the deterministic local-env sync contract and
  synced the live ignored `services/orion-thought/.env`.

## Privacy and failure boundaries

- Pending high-stakes memories cannot seed a visual; they require confirmation.
- A memory with any unverified evidence is excluded.
- Expired memories are excluded even if a lifecycle worker has not yet changed
  their `status`.
- The episode reader hard-clamps age to seven days. The legacy rollback env may
  narrow that window but cannot widen it to re-admit pre-policy memories.
- The execution receipt exposes source kind and id, not private memory text.
- Database failure degrades to no memory seed rather than blocking the visual
  chain.
- Rollback: set `REVERIE_VISUAL_MEMORY_SEED_SOURCE=crystallization` and restart
  `orion-thought`. No schema migration or destructive rollback is required.

## Runtime evidence

The exact new eligibility/rotation SQL was executed read-only against the live
database after implementation. At that moment it returned 24 eligible memories
and next selected memory id `686e6e3c-6b5c-58eb-b568-990f83358e8d`. The count
fell from the earlier 25-row audit because the query also enforces live expiry.
The current seven-day candidates are policy-era v5/v6 rows with recorded stakes
reasons. Thirty older v2 low/auto rows still lack stakes reasons, but are now
structurally outside the non-widenable seven-day window.

`scripts/safe_docker_build.sh orion-thought build` completed and produced image
`sha256:2c0224830a14bbcd7e37dbb99313f2fef553cc6d6d6d7614042c9857fb24ef3d`.

## Checks

- Final focused thought/store/visual/durable/eval/Hub/schema/env-sync suite:
  229 passed, 29 skipped.
- Broad `orion-thought` tests and evals plus touched Hub/schema/script tests:
  653 passed, 28 skipped, 1 unrelated pre-existing failure.
- `python3 scripts/check_env_template_parity.py orion-thought`: PASS.
- `python -m compileall` over touched Python packages: PASS.
- `git diff --check`: PASS.
- Isolated PostgreSQL eligibility/rotation integration test: 1 passed.

The unrelated broad-suite failure is
`test_settings_mind_enrichment.py::test_mind_enrichment_defaults_off`: main
currently defines `mind_base_url='http://mind:6611'`, while the unchanged test
expects `http://orion-mind:6611`. This patch changes neither side.

## Eval

`test_episode_memory_seed_keeps_typed_provenance_without_text_egress` proves a
real typed episode source reaches the diffusion prompt, retains its kind/id in
the chain, enters the execution receipt as an id, and does not leak source text
through that receipt. The existing honest-degradation matrix remains green.

## Review findings fixed

- **High:** the reused legacy age env could widen episode eligibility beyond
  seven days and expose pre-policy rows. Fixed with a hard 604800-second clamp
  in the reader plus a test that a 14-day caller still binds seven days.
- **Medium:** SQL substring tests did not execute governance or rotation.
  Added an isolated-Postgres fixture covering low/auto, confirmed high,
  pending high, mixed verified/unverified evidence, disqualifying events,
  expiry, the hard age ceiling, never-used-first, and least-recently-used.
- Independent re-review confirmed both findings resolved and found no new
  material issue.

## Deployment verification

After merge/restart, wait for an allocator-authorized visual run and verify:

1. At least one visual per day uses `source_kind='episode_memory'`.
2. No pending-confirmation memory is selected.
3. At least five distinct episode memory ids appear over seven days.
4. The Hub labels those rows `Memory (validated episode)`.

Until those checks run on the deployed rail, runtime cutover is **UNVERIFIED**.
