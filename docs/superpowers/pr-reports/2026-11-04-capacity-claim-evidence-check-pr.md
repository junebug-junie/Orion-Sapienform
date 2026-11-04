## Summary

- This PR builds on PR #3107 (merged 06:29Z), which let a claim hold its job while its evidence still checked. It does not compete with it: there is still one mechanism, #3107's hold seam (`CLAIMS_HOLD`, `claims.recheck`, `_claim_status`). This PR makes the check behind that seam stricter.
- The new check (`claims.check_evidence`) passes a claim only when every id the claim names is wired to the claimed part by a live flow, every repo file it cites exists, and the wiring fits the job. #3107's check passed a claim when any one id still checked.
- A claim that passes prints as EVIDENCE-CHECKED ("code, not Juniper"), never as "confirmed by Juniper". It closes the job's "no part is shown" gap lines, the same way a confirmation does. That reverses #3107's rule that a claim never clears a gap line; the spec records the change in a dated note.
- An S5 claim never holds through the code check, so it still waits for Juniper. If Juniper rejects a claim, her rejection wins over the check. Her confirmation outranks a passing check.
- A claim that fails prints "evidence does not hold" and names the reference that failed. It goes into a "Fix or withdraw" block. The new `capacity withdraw <part> <job>` command takes a claim back. Juniper's part is one "may veto" line, not a queue she must answer.

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

## What #3107 did: kept vs replaced

Kept from #3107:
- the one hold mechanism: `CLAIMS_HOLD` (the rollback switch), `claims.recheck` (the seam), and the order of the checks in `_claim_status`;
- `reading.claimed` and strength `claimed`;
- the "rule says no" check (a near miss means "not S2") and the "disputed" check (a later dispute of Orion's);
- "rule gives it";
- a confirmation outranking a claim;
- a held claim counting as a capacity holder;
- `body <part>`.

Replaced:
- the recheck itself: it was the floor used when the claim is written (any one id still checks); it is now `check_evidence` (every id, plus the shape of the job);
- the labels: CLAIMED became EVIDENCE-CHECKED ("code, not Juniper"); `claimed` became `evidence-checked`; `evidence gone` became `evidence does not hold`, and the claim now names the reference that failed;
- "a claim never clears a gap line": a passing claim now covers "no part is shown" gaps, and `gap:s4-to-s3` is checked against claimed parts too;
- "Evidence gone" in the dig tail became "Fix or withdraw", which also lists failing claims on jobs another part holds;
- S5 no longer holds through the code check;
- new: `capacity withdraw`, and the one-line "Juniper may veto".

## Current architecture

#3107 is on main. A claim holds its job (CLAIMED) while, on each read, all of these are true:
- its part is live;
- its because still passes the floor used when the claim is written (at least one cited id is on a flow of the part);
- Juniper has not rejected it;
- no near-miss rule says "not S2";
- no later dispute of Orion's stands against it.

A claim never clears a gap line.

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
pytest orion/body_atlas/tests orion/dig/tests tests/test_organ_names_partition.py -q  -> 342 passed
```

## Evals run

```text
run_capacity_roles_eval.py -> all PASS. Claims floor PASS. 7 claim mutants CAUGHT (stored cites
  trusted, rejected, dead part, rule says no, disputed, S5 holds, a held claim clears an atlas-blind
  gap). Evidence check on the 5 live claims (10-31 body) PASS. Any-reference mutant CAUGHT.
run_wire_eval.py, run_organ_structure_eval.py, run_store_direction_eval.py -> exit 0
```

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

Review of the reconciled diff, after the merge with #3107:
- Finding: the README still had #3107's eval and tail text.
  - Fix: rewritten.
- Finding: a failing claim on a job that another part holds showed in no list.
  - Fix: `failed_claims` lists every failing claim. Evidence: test updated.
- Finding: disputes were read in file order, while claims were read by time.
  - Fix: both are now read by `at`, then file order.
- Finding: no test covered the S4-to-S3 recheck with claimed parts.
  - Fix: `test_s4_s3_gap_is_rechecked_with_parts_held_by_passing_claims`.
- Nits fixed:
  - the `cites_in` docstring;
  - a test name;
  - the `evidence` field comment;
  - the "S5 is hers" wording, now shown only when every waiting claim is S5;
  - the #3107 spec section, now marked as superseded;
  - the eval's gap comment.

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
