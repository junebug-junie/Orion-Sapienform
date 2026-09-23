# PR: Promote System One `curiosity_pull` to endogenous curiosity admission

## Summary

- Promote **only** `curiosity_pull` from shadow into a categorical admission gate over already-built endogenous curiosity candidates.
- Operational contract: `argmax` level **0 → skip** `FrontierCuriosityEvaluator`; levels **1 and 2 → admit** with identical effect (no level-2 boost).
- Fail-open when System One is missing/stale/malformed; explicit kill switch defaults **live** (`false`).
- Leave `reverie_fit`, `attention_interrupt`, and `deliberation_need` observational.
- Publish typed frames on `orion:system_one:appraisal`; persist gate lineage on candidate rows; extend the eval script for post-live calibration.

## Outcome moved

System One can now causally influence whether Orion spends cognition on the full curiosity evaluator, without inventing topics, replacing the evaluator, or bypassing governance.

## Current architecture

Kev / System One compiled `SystemOneAppraisalFrameV1` on the attention-broadcast tick into Postgres + a grammar shadow. Endogenous curiosity always ran `FrontierCuriosityEvaluator` when seeds existed. No behavioral consumer read System One.

## Architecture touched

- `orion/substrate/system_one_access.py` — reusable frame validation + curiosity admission
- `services/orion-substrate-runtime` — gate in `_endogenous_curiosity_tick`; bus publish; settings; `gate_json` writer
- Bus: `orion:system_one:appraisal`
- Migration: `manual_migration_endogenous_curiosity_gate_json_v1.sql`
- Eval: post-live gate metrics

## Live distributions used to justify promotion

Window **2026-09-23 17:36:26Z → ~22:04Z** (~4.46h), **n≈469**, `kev`/`kev-latest`/`orion.system_one.shadow.v1`, 0 malformed.

`curiosity_pull` argmax ≈ **276 / 178 / 15** (0/1/2); score mean ~0.80, stdev ~0.17 — not flat.

`reverie_fit` argmax **468/0/0** — not promoted (degenerate).

Chosen contract is 0-vs-{1,2} so System One admits evaluation rather than claiming “investigate,” and so the ~3% level-2 rate does not collapse live curiosity by ~97%.

## Fallback / hard gates / rollback

- Missing/stale/invalid → `system_one_unavailable_fallback` → legacy evaluator-on-seeds
- `ORION_ENDOGENOUS_CURIOSITY_KILL_SWITCH` remains superior
- `SUBSTRATE_SYSTEM_ONE_CURIOSITY_GATE_KILL_SWITCH=true` restores legacy (default **false**)
- System One cannot mint candidates; evaluator still owns invoke/defer/noop

## Telemetry / lineage

Candidate-set `gate_json` + logs: `frame_id`, probabilities, argmax, gate result, evaluator outcome/task/decision_id when admitted.

## Tests run

```text
57 passed (system-one appraisal + access + bus catalog + observational guard +
endogenous curiosity tick/store + attention broadcast tick) in python:3.12 CI shape
```

## Review findings fixed

- Finding: writer hard-required `gate_json` before migration
  - Fix: fall back to legacy INSERT if column missing; PR restart notes still say apply migration first
  - Evidence: `save_endogenous_curiosity_candidates` dual-path + store tests
- Finding: `candidate_set_id` stamped after persist so it never landed in `gate_json`
  - Fix: store stamps `candidate_set_id` into `gate` before Json serialize
  - Evidence: `test_save_curiosity_candidates_persists_gate_json`

## Restart required (after merge, for Juniper)

```bash
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_endogenous_curiosity_gate_json_v1.sql
# from worktree after merge:
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

## Do not merge

This PR is left for Juniper to review and merge.
