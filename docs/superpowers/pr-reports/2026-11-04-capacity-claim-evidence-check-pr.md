## Summary

- Code now checks Orion's capacity claims against the live body atlas on every `dig capacities` read, so Juniper does not have to say yes or no to each one.
- A claim passes only when every id it names is wired to the claimed part by a live flow, every file it cites exists, and the wiring fits the job. A passing claim counts as a role. It prints as EVIDENCE-CHECKED and never as "confirmed by Juniper".
- A claim that fails prints "evidence does not hold" and names the reference that failed. It goes into a "Fix or withdraw" block for Orion. The new `capacity withdraw <part> <job>` command takes a claim back.
- S5 claims never pass the check (decision 3), so they still wait for Juniper. A rejection from her beats a passing check, and a confirmation from her outranks one.
- Nothing is stored. The check runs again on every read, so a claim drops back as soon as its wiring leaves the atlas.

## Outcome moved

Before this change, the 5 claims Orion wrote on 2026-11-04 sat in a queue waiting for Juniper. Now, on the live atlas of 2026-11-04 05:56Z:

| Claim | Result | Why |
|---|---|---|
| orion-gpu-lane-controller S3 | passes | It answers on `orion:gpu_pool:actuate:result` to orion-gpu-pool and orion-mesh-guardian, which is a flow out to live parts. |
| orion-equilibrium-service S3* | passes | R-census-reader finds it listening to the whole body, and it reports out on `orion:equilibrium:snapshot` to orion-cortex-orch. |
| orion-durable-runs algedonic | passes | It writes `redis:orion:regulation:latest` and `redis:orion:self_monitor:latest`. R-broadcast-key finds both keys (1 writer, 5 readers, 2 organs). |
| orion-energy metabolism | **fails** | The claim names `pg:energy_bill_forecast`, and that table is dormant (0 rows). The other 4 cites hold. |
| orion-world-pulse S4 | passes | It sends `orion:world_context:daily_capsule` to orion-cortex-exec. |

With these results, 2 gap lines close: no confirmed audit (S3*) and no pain line (algedonic). The S4-to-S3 check also runs for the first time. It reports a real gap: no bus, HTTP or Redis link exists between orion-world-pulse and orion-gpu-lane-controller.

Limit: the check matches ids. It proves that each cited thing is wired to the part. It does not prove that the sentence about it is true.

## Current architecture

Before this change (PR #3096), `dig capacities` printed each claim as "unconfirmed" and listed it under "Waiting on Juniper". A claim held no role until Juniper confirmed it in `config/body_atlas/vsm_confirmed.yaml`.

## Architecture touched

These are local files only: no bus, schema registry, env or DB change.
- `orion/body_atlas/claims.py`: `flow_dirs`, `because_refs`, `check_evidence`, `check_claims`, `withdraw_claim`.
- `orion/body_atlas/capacities.py`: `EvidenceCheckV1`, `EVIDENCE_SHAPES`, `reading.checked`, the EVIDENCE-CHECKED status, and `active_claims`.

## Files changed

- `orion/body_atlas/claims.py`: the evidence check and the withdraw command. A flow that carries only the census, or that goes to a part that is not live, is no longer evidence. This also tightens the check made when a claim is written.
- `orion/body_atlas/capacities.py`: checked roles hold jobs, cover gap lines, are capacity holders, and feed the S4-to-S3 check. New claim statuses and the `active_claims` fold (by `at`, then file order).
- `orion/dig/__main__.py`: each read runs the check. Adds the fix-or-withdraw block, the S5-only waiting list, the may-veto line, and the `capacity withdraw` command.
- `orion/body_atlas/tests/test_claim_evidence.py` (new) and `orion/dig/tests/test_capacities_cmd.py`: regression tests.
- `orion/body_atlas/evals/run_capacity_roles_eval.py` and `evals/fixtures/capacity_claims_2026-11-04.jsonl`: the 5 live claims, frozen, with expected results reviewed by hand, plus a mutation.
- The spec (dated decision note), `orion/body_atlas/README.md`, `orion/dig/README.md`, and `config/body_atlas/vsm_confirmed.yaml` (header comment only).

## Schema / bus / API changes

- Added: `EvidenceCheckV1`; `CapacityReadingV1.checked`; `ClaimViewV1.failed`; the claim kind `withdraw`; the claim statuses `evidence-checked` and `evidence does not hold`; the job status `EVIDENCE-CHECKED`. All of these are local Pydantic models, not bus schemas.
- Removed / renamed: none.
- Compatibility: existing ledger lines read unchanged.

## Env/config changes

- No env keys changed. `.env_example` is unchanged, so no sync was needed.

## Tests run

```text
pytest orion/body_atlas/tests orion/dig/tests tests/test_organ_names_partition.py -q  -> 318 passed
```

## Evals run

```text
run_capacity_roles_eval.py -> all PASS; evidence check (5 live claims on 10-31 body) PASS;
  mutation (any reference holds, not every one) CAUGHT; earlier mutations CAUGHT
run_wire_eval.py, run_organ_structure_eval.py, run_store_direction_eval.py -> exit 0
```

On the 10-31 fixture, the durable-runs claim fails because `redis:orion:self_monitor:latest` was not in the atlas yet. That is the "wiring disappears, claim drops back" case, shown on real data.

## Docker/build/smoke checks

```text
No image change. Live dry run: the read-only atlas and claims were copied from
orion-athena-harness-governor, and this branch's code was run against them. The printout is
32 lines and 3,571 chars (cap: 40 lines, 4,000 chars).
```

## Review findings fixed

- Finding: a file cite with no extension (`services/x/Dockerfile`) was read as an unknown id, so its claim could never pass.
  - Fix: a cite with a `/` that is not an atlas id is treated as a file. Evidence: `test_a_write_time_file_cite_without_an_extension_still_reads_as_a_file`.
- Finding: a channel on a flow to a dormant part counted as live wiring.
  - Fix: `flow_dirs` skips flows whose other end is not live. Evidence: `test_a_channel_to_a_dormant_part_is_not_live_wiring`.
- Finding: an S3 or S4 claim passed on a table write alone.
  - Fix: for S3, S3* and S4, "out" must reach a live part, or a channel a live part reads. Evidence: `test_s3_that_only_writes_a_table_does_not_act_on_a_part` and the S3* no-out test.
- Finding: the algedonic shape check did not check who writes the key.
  - Fix: it now uses the `BroadcastKeyV1.writers` list.
- Finding: the eval's "printed as Juniper's" check would break on her first real confirmation.
  - Fix: the eval uses a copy of the confirmations with no confirmed or rejected entries.
- Finding: the eval mutant also skipped the shape gate.
  - Fix: the mutant keeps shape failures, and the mutation is still CAUGHT.
- Finding: a claim on a job a rule now holds showed as vetoable.
  - Fix: it now shows "rule holds". Evidence: a regression test.
- Finding: withdraw was read by file order.
  - Fix: claims and withdraws fold by `at`, then file order. Evidence: a regression test.
- Nits fixed: a URL in prose is no longer read as an id; the census comparison ignores case; a host gets its own "placement, not evidence" message; the decision note says a checked algedonic part makes cap:s5 writable (the sentence must still name Juniper).

## Restart required

`orion/dig` runs in orion-athena-harness-governor from the read-only mount of the primary checkout. After this merges and main is pulled, the next `dig capacities` call uses the new code. No restart is needed.

```text
No restart required.
```

## Risks / concerns

- Severity: low.
  - Concern: code now grants roles without Juniper's yes. A wrong but well-wired claim (the sentence is false, the ids are real) counts as a role.
  - Mitigation: the claim prints as EVIDENCE-CHECKED, never as Juniper's. She can veto any claim with one line in `vsm_confirmed.yaml`. S5 still needs her.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3109

🤖 Generated with [Claude Code](https://claude.com/claude-code)
