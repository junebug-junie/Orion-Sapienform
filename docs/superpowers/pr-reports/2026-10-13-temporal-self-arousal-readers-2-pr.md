# Temporal Self rev 4, order 6: the remaining arousal readers (R3 readers 4, 5, 6)

Three stacked PRs, one per service, merge in order: #2635 -> #2636 -> #2637.

## Summary

- Orion's arousal reading now slows three more kinds of optional work while the body or GPU is under load (`strained`). Before this, only the dream, curiosity and background outreach read it (#2624).
- **Reader 4, visual-reverie baseline image (orion-proposal-runtime, #2635):** when a baseline image is about to be issued and arousal reads `strained`, it is deferred by the policy's `retry_sec` (10 min), exactly like a retry. Engaged / idle: no change.
- **Reader 5, reverie chain pace (orion-thought, #2636):** the pause between text-reverie chains doubles (90 s -> 180 s) while `strained`. Every chain step is a background-lane LLM call. Engaged / idle: no change.
- **Reader 6, self-modification proposals (orion-hub, #2637):** the scheduled mutation cycle makes no new proposal while `strained`. The pressure is kept (not cooled) and is proposed on the first cycle after strain clears. Engaged / idle: no change.
- Every reader reads arousal at its own decision point (not at loop start), and treats missing / stale (> 360 s) / malformed / producer-unknown (including `gpu_fault:*`) / Redis error / its own flag off as "arousal does not exist". Each has its own flag, shipped ON.

## Outcome moved

Behaviour: during a GPU-queue backlog, Orion adds less optional GPU and self-modification work to it. Replay on saved history (below): about 2% of reverie chains and 2% of mutation proposals fall inside a strained span; the baseline image coincided once in 4.5 days.

## Current architecture

- `orion-durable-runs` writes `RegulationStateV1` to Redis `orion:regulation:latest` (TTL 360 s) every 120 s and on every Juniper turn; level changes are `arousal_transition` rows in `temporal_self_event`. Since #2628 a GPU controller fault reads `unknown`, not `strained`.
- Readers 1-3 and 7 live in orion-dream and orion-hub (#2624). Readers 4-6 had no arousal input.

## Architecture touched

- `orion/regulation/arousal_read.py` (new): one Redis GET judged by `arousal_view`, sync and async forms, 2 s timeouts, never raises. Shared by the three readers.
- Reader 4: `orion/reverie/baseline.py::schedule` takes a lazy `arousal_level` callable, invoked only after every other baseline check passed. `ProposalRuntimeWorker._baseline_arousal_level` does the read (the tick runs in `asyncio.to_thread`; about 16 reads a day plus retries).
- Reader 5: `services/orion-thought/app/chain.py` `_next_chain_sleep_sec` reads arousal after each chain, right before the pause; `reverie_chain_sleep_sec` is the pure rule (never shortens).
- Reader 6: `PressureAccumulator.readiness` / `ready_for_proposal` take a lazy arousal callable (after threshold and cooldown). `SubstrateAdaptationWorker.arousal_level` + `_CycleArousalProbe` read at most once per cycle, at the first otherwise-ready pressure; held pressures trace `mutation_proposal_held` and the cycle summary logs `proposals_held_arousal`. Hub wires `_mutation_arousal_level` into the scheduled cycle only; the operator's manual cycle never reads arousal.

## Files changed

- `orion/regulation/arousal_read.py`, `tests/test_regulation_arousal_read.py`: shared fetch.
- `orion/reverie/baseline.py`, `services/orion-proposal-runtime/app/{worker,store,settings}.py`, `.env_example`, `README.md`, `tests/test_visual_baseline_arousal_reader.py`, `services/orion-proposal-runtime/tests/test_visual_baseline_arousal_worker.py`: reader 4.
- `services/orion-thought/app/{chain,settings}.py`, `.env_example`, `README.md`, `tests/test_reverie_chain_arousal_reader.py`: reader 5.
- `orion/substrate/{mutation_pressure,mutation_worker}.py`, `orion/substrate/tests/test_mutation_pressure_arousal_reader.py`, `services/orion-hub/{app/settings.py,scripts/api_routes.py,.env_example,README.md}`, `services/orion-hub/tests/test_regulation_mutation_arousal_reader.py`, `services/orion-hub/tests/test_substrate_mutation_scheduler_runtime.py` (fixture stubs the reader so tests never read the live bus): reader 6.
- `orion/inner_state_registry.py`, `config/metrics/metric_definitions.lock.json`: three declared consumers (routing change, re-locked per branch).
- `scripts/sync_local_env_from_example.py`: exact keys `ORION_VISUAL_BASELINE_AROUSAL_ENABLED`, `SUBSTRATE_AUTONOMY_AROUSAL_ENABLED`.
- `scripts/analysis/measure_arousal_reader_replay.py` + test, `docs/superpowers/evidence/2026-10-13-temporal-self-arousal-readers-2/`: eval.

## Schema / bus / API changes

- Added: none on the bus. Mutation cycle result gains `proposals_held_arousal`; trace event `mutation_proposal_held` (not persisted today). Proposal frames may carry warning `arousal_strained`.
- Removed / Renamed: none.
- Behavior changed: the three dials above, conservative direction only.
- Compatibility notes: `RegulationStateV1` is read, not changed. No migration.

## Env/config changes

- Added keys: orion-proposal-runtime `ORION_VISUAL_BASELINE_AROUSAL_ENABLED=true`, `ORION_REGULATION_READER_MAX_AGE_SEC=360`; orion-thought `ORION_REVERIE_CHAIN_AROUSAL_ENABLED=true`, `ORION_REVERIE_CHAIN_AROUSAL_STRAINED_MULTIPLIER=2.0`, `ORION_REGULATION_READER_MAX_AGE_SEC=360`; orion-hub `SUBSTRATE_AUTONOMY_AROUSAL_ENABLED=true`.
- Removed / renamed: none.
- `.env_example` updated: yes (all three), plus settings defaults and README.
- local `.env` synced with `python scripts/sync_local_env_from_example.py orion-proposal-runtime orion-thought orion-hub`: yes, all keys present in the primary checkout's `.env` files.
- skipped keys requiring operator action: none (the two keys outside synced prefixes were added as exact keys and then synced).
- `check_settings_defaults.py --example-drift orion-hub`: OK. proposal-runtime and thought are not on that checker's allowlist (pre-existing). `check_env_template_parity`: PASS. `check_env_key_single_source`: OK. Compose parity: N/A (all three use `env_file`).

## Tests run

```text
root + proposal-runtime + replay + substrate focused set             160 passed
services/orion-thought/tests (whole suite)                           1 failed, 592 passed, 28 skipped
  -- the 1 failure (test_settings_mind_enrichment default URL) fails identically on origin/main
services/orion-hub/tests (whole suite, branch snapshot)              36 failed, 3515 passed, 86 skipped
services/orion-hub/tests (whole suite, origin/main)                  36 failed, 3510 passed, 86 skipped
  -- identical failure sets (diff empty)
orion/substrate/tests                                                3 failed (felt_state_self_definition_lane, same 3 on main), 1 flaky metacog trend test passes alone
check_definition_drift --gate PASS (re-locked); check_metric_lineage --gate PASS;
check_inner_state_registry OK; tests/scripts/test_sync_local_env_from_example.py 19 passed
```

Per reader: strained holds; engaged / idle / unknown / flag off / Redis error / Redis hang = today. Must-not: every non-issue baseline reason (disabled, stale, history unavailable, not due, active, cooldown) is unchanged under all four levels and arousal is not even read; below-threshold and cooling pressures never read arousal; the mutation kill switch wins over any level; the manual mutation cycle never gets a reader; the reverie pause is never shorter than base for any level or multiplier; the thermal gate, reverie refractory and chain step cap have no arousal input.

## Evals run

`scripts/analysis/measure_arousal_reader_replay.py`, window 2026-10-09 07:00 to 2026-10-13 19:00 UTC (`gpu_pool_state_history` starts 10-09 06:46). Evidence: `docs/superpowers/evidence/2026-10-13-temporal-self-arousal-readers-2/reader_replay.json`, `export.sql`.

Held / total per day, replay lane (shipped reducer over saved inputs, GPU-fault logic on):

| Day (UTC) | Visual baseline (4) | Reverie chain (5) | Mutation proposals (6) | Strained minutes |
|---|---|---|---|---|
| 10-09 (from 07:00) | 0 / 11 | 0 / 178 | 0 / 100 | 0 (whole day reads unknown: GPU controller fault) |
| 10-10 | 1 / 14 | 21 / 306 | 8 / 140 | 88 |
| 10-11 | 0 / 16 | 5 / 321 | 2 / 141 | 20 |
| 10-12 | 0 / 15 | 7 / 334 | 3 / 140 | 26 |
| 10-13 (to 19:00) | 0 / 12 | 0 / 276 | 0 / 111 | 0 |
| **Total** | **1 / 68** | **33 / 1415** | **13 / 632** | |

Live lane (real `arousal_transition` rows, from 10-10 23:00): baseline 0 / 43, reverie 11 / 945, mutation 4 / 398.

What the numbers say:

- All five strained spans were GPU-queue backlogs (`gpu_queue_sustained`), none cabinet heat, so the existing visual thermal gate would not have caught any of them.
- **Reader 4 is rare but not dead.** One baseline image (10-10 09:57) started inside a GPU backlog; deferred, it would have run about 20 min later. Expected rate is about one every 4-5 days at today's strain levels. Kept because it defers the single heaviest GPU job into a backed-up queue and costs one GET per due image.
- **Reader 5:** "held" means that chain's following pause was doubled (+90 s), not that the chain was skipped. About 2% of chains, all inside strained spans.
- **Reader 6:** each held proposal waits until strain clears (14-60 min here). Upper bound: the row does not say whether a proposal came from the operator's manual cycle, which is never held.
- First order only: a held decision would have moved later ones; not simulated.

## Docker/build/smoke checks

```text
UNVERIFIED -- not deployed from these branches (production deploys from the primary checkout on main).
Live read-only check during review: GET orion:regulation:latest parsed as level=idle, age_sec=105 through read_arousal_sync.
```

## Review findings fixed

Review ran in a subagent against `origin/main...feat/arousal-reader-mutation-pressure`. No must-fix findings; all ten hard requirements confirmed (arousal only adds holds, unknown = today, manual cycle untouched, decision-point reads, bounded timeouts, env parity, no live Redis reads from tests).

- Finding (should-fix): `ORION_REVERIE_CHAIN_AROUSAL_STRAINED_MULTIPLIER` had `ge=1.0`, so a typo like 0.5 would stop orion-thought from booting instead of changing nothing.
  - Fix: bound removed; the code guard already treats <= 1 as "no change".
  - Evidence: `test_a_sub_one_multiplier_boots_and_changes_nothing`.
- Finding (nit): baseline strained branch duplicated the `next_attempt_not_before` stamp.
  - Fix: hoisted above the arousal check.
- Finding (nit): two source-text tests used cwd-relative paths.
  - Fix: repo-relative via `__file__`.
- Finding (nit): Hub settings field for `SUBSTRATE_AUTONOMY_AROUSAL_ENABLED` is declared for parity but the cycle reads `_env_flag`.
  - Fix: comment says so (same pattern as its `SUBSTRATE_AUTONOMY_*` neighbours).
- Not changed: `mutation_proposal_held` traces fire every cycle during strain (not persisted today, so harmless); replay caveats are stated above.

## Restart required

From the primary checkout on main after all three merge:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-proposal-runtime up -d --build && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-thought up -d --build && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
```

Live checks after deploy:

- `docker logs <proposal-runtime container> 2>&1 | grep "arousal_view reader=visual_baseline"` appears when a baseline image comes due.
- `docker logs orion-athena-thought 2>&1 | grep "arousal_view reader=reverie_chain"` once per chain.
- `docker logs orion-athena-hub 2>&1 | grep "arousal_view reader=mutation_pressure"` when a pressure is ready.
- During the next strained span (`curl -s localhost:8124/regulation/state`): `grep visual_baseline_deferred`, `grep reverie_chain_paced`, and a non-zero `proposals_held_arousal` in Hub's `mutation_scheduler_cycle_finished` log line.
- Kill-switch smoke: deleting `orion:regulation:latest` leaves all three on today's behaviour.

## Risks / concerns

- Severity: low. Concern: reader 4 fires about once every 4-5 days at current strain levels. Mitigation: numbers stated; off switch `ORION_VISUAL_BASELINE_AROUSAL_ENABLED`.
- Severity: low. Concern: a long GPU-queue backlog that is not a controller fault would hold mutation proposals for its whole length. Mitigation: pressure is kept, not lost; off switch `SUBSTRATE_AUTONOMY_AROUSAL_ENABLED`.
- Severity: low. Concern: stacked PRs each re-lock `metric_definitions.lock.json`; merge in order. If another PR relocks first, rebase and re-run `check_definition_drift.py --update` rather than hand-merging.

## PR link

- https://github.com/junebug-junie/Orion-Sapienform/pull/2635 (reader 4, base)
- https://github.com/junebug-junie/Orion-Sapienform/pull/2636 (reader 5)
- https://github.com/junebug-junie/Orion-Sapienform/pull/2637 (reader 6)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
