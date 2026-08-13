# Arena degeneracy instrument, and the gate that discarded every skill verb's result

Branch: `feat/metric-commensurability` (3 commits)
Status: **DONE_WITH_CONCERNS** — see Risks. Both mechanisms deployed and live-verified.

Implements step 1 of `docs/superpowers/specs/2026-08-12-metric-commensurability-handover-spec.md`
§7, and fixes the gate that measurement exposed.

## Summary

- Built the read-only instrument §7 named as the next patch, shipped alone per that spec's own
  throttle rule §6.3.
- It answered §7's question, and in doing so found a strictly larger defect the spec diagnosed
  only in the abstract: **the proposal arena's scalar has almost no resolution across its own
  competitors**. 7.03 of 10 candidates are tied at the frame maximum; 10 of 28 template pairs
  carry an urgency correlation of exactly 1.000000.
- It also found that `prune_build_cache` **had** been dispatching since PR #1599, and every
  result was being discarded at the parse boundary — Orion's only mutating action was
  structurally incapable of reporting an outcome.
- Fixed that. Orion now records real, inspectable judgments about the host.
- Fixed a live false-success this patch introduced, caught by review before it could
  accumulate.

## Outcome moved

| gate | before | after |
| --- | --- | --- |
| skill verb result | `status=empty raw_len=0`, payload discarded | `status=success kind=structured`, payload preserved |
| `bytes_reclaimed` reachable from SQL | no — discarded at parse | yes — `result_json.structured_result` |
| failed verb | (after commit 2) recorded as **success** | `status=failed` with the real reason |
| theater tripwire vs. skill failures | could never fire | fires on any non-success majority |
| §7's question | unmeasured | answered, with a re-runnable instrument |
| section D "starved" | 3 (all false) | 0 (true) |

First real autonomous judgment Orion has recorded about its own host, minutes after deploy:

```json
{"acted": false,
 "decision": "declined_no_pressure",
 "reason": "used_pct 71.426 < 75.0",
 "disk_before":  {"used_pct": 71.426, "used_bytes": 359216766976, "total_bytes": 502922461184},
 "cache_before": {"entries": 12667, "reclaimable_bytes": 111900000000},
 "thresholds":   {"min_disk_pct": 75.0, "min_reclaimable_bytes": 42949672960}}
```

It measured the host, measured the cache, compared both against its own thresholds, declined to
act, and said why. Not a biometric.

## The measurement

`scripts/analysis/measure_arena_degeneracy.py`, 24h live window. Every number pasted, none derived.

### A. The arena was never holding a contest

```
frames measured                 17,496 x 10 candidates
distinct urgency values per frame:
    2 distinct  ->   5,840 frames (33.4%)
    3 distinct  ->  11,363 frames (64.9%)
    4 distinct  ->     293 frames ( 1.7%)
avg candidates TIED at frame max  7.03 of 10.00
```

### B. Ten pairs of "competitors" carry one number

```
inspect_attended_target      inspect_field_topology_catalog    17496   1.000000
inspect_bus_channel_catalog  summarize_transport_contract_drift 17496  1.000000
...
pairs with corr(urgency) >= 0.999999: 10 of 28
```

Root cause, traced to source: `orion/proposals/scoring.py:268` `proposal_urgency()` calls
`_pressure_dimension_ids()`, which falls back to **all four** `PRESSURE_DIMENSIONS` for any
template declaring `dimensions: {}`. Five of thirteen live templates do. So for all five,
`urgency = max(execution, resource, reasoning, reliability)` — literally the same scalar.
`match_score` is provably ≤ `urgency` (it is that same value times a weight ≤ 1 times a policy
weight ≤ 1), so `proposal_priority = base_priority + confidence · urgency` reduces to

```
priority = base_priority + C          # C identical across the tied candidates
```

Rank order is decided entirely by hand-authored `base_priority` constants, and the observed win
rates follow that ladder exactly. Worse, it is **inverted**: a template that declares the
dimension it actually cares about gets that one dimension's value
(`prune_build_cache: resource_pressure ≈ 0.08`), while a template declaring nothing gets the max
of four (≈ 0.9). Describing what you care about is a handicap.

This is the spec's failure mode C (silent defaults) producing failure mode D (argmax starvation),
and it means PR #1599's aging and reserved lane were treating starvation in a ranking that cannot
rank. Not wrong — downstream of the real defect.

### C. Spec §7, answered

```
policy frames                   33,888
frames with zero decisions      15,808 (46.6%)
distinct decision sets              66
frames adding no new decision   27,375 (80.8%)

EMPTY runs        n= 1254  frames= 15,808  mean/median/p95/max = 12.61 / 9 / 39 / 111
identical runs    n= 5259  frames= 18,080  mean/median/p95/max =  3.44 / 2 /  9 /  99
```

**66 distinct decision sets in a day, re-emitted 33,888 times.** §2.1's direction 1 (reduce
production) is correct, and the split says it is two problems: long unbroken runs of *empty*
frames (mean 12.6), and shorter runs of genuinely-identical non-empty ones (mean 3.4). Different
remedies; the combined 80.8% headline hides which dominates.

### D. Reachability

```
template                          proposed  dispatch  blk:policy  blk:lost  win rate
inspect_bus_channel_catalog          16895      2497           8         0    14.78%
...
prune_build_cache                      411        23           3         4     5.60%

STARVED (competed, never won): 0
blocked by policy, NOT starved: 3
```

## The defect the instrument found

Section D showed `prune_build_cache` had dispatched. Every one returned `status=empty raw_len=0`.
The cortex-exec log for the same correlation ids says otherwise:

```
final_text_assembly verb=skills.runtime.builder_prune.v1
  raw_len=739 clean_len=739 final_len=739
```

739 characters returned, 0 stored. `parse_structured_observation` only understood **one** producer
family. Observation verbs emit `{"observation", "salient_facts", "confidence"}`; skill verbs emit
their own result dict via `_skill_result_output`, which sets `final_text = json.dumps(result)` and
has no `observation` key. Missing key → `""` → `len("") == 0` → `status="empty"`.

The verb's four carefully separated verdicts (`declined_no_pressure`,
`declined_nothing_to_reclaim`, `would_act`, `pruned` with a real `bytes_reclaimed`) all collapsed
into the single state a dead executor produces.

## Current architecture (before)

```
proposal-runtime  --(ProposalFrameV1)-->  policy-runtime  --(PolicyDecisionFrameV1)-->
    execution-dispatch-runtime  --RPC-->  cortex-exec  --verb-->  SkillVerbOutput
                                              |
                                   parse_structured_observation  <- understood 1 of 2 shapes
                                              |
                                   substrate_dispatch_results
```

## Architecture touched

- `orion/execution_dispatch/result_extraction.py` — the parse boundary between cortex-exec's
  reply and the stored dispatch result.
- `services/orion-execution-dispatch-runtime/app/worker.py` — the status decision, the emit, the
  replay branch, and the theater tripwire predicate.
- No new service, no new schema version, no bus channel change.

## Files changed

- `scripts/analysis/measure_arena_degeneracy.py`: new read-only instrument (sections A–D).
- `tests/test_measure_arena_degeneracy.py`: 47 tests, all fixtures hand-computed.
- `orion/execution_dispatch/result_extraction.py`: recognise skill-verb results; read the plan's
  own status.
- `services/orion-execution-dispatch-runtime/app/worker.py`: status by `result_kind` not by
  observation length; plan-failure check before content classification; tripwire predicate.
- `services/orion-execution-dispatch-runtime/README.md`: tripwire contract was made false by
  commit 2; status-value table; `raw_len` caveat.
- `tests/test_execution_dispatch_result_extraction.py`: 18 tests (2 replaced — they asserted the
  buggy behaviour).
- `tests/test_execution_dispatch_runtime_worker.py`: +13 lifecycle tests.
- `services/orion-execution-dispatch-runtime/tests/test_theater_tripwire.py`: +3, 1 reversed.

## Schema / bus / API changes

- Added, both additive, no version bump: `result_json.result_kind`
  (`observation` / `structured` / `empty`) and `result_json.structured_result`.
- Added on the failure path: `result_json.plan_status`.
- Behaviour changed: `substrate_dispatch_results.status` is now decided by `result_kind` and the
  plan's own verdict, not by `len(observation)`. A structured skill result is `success`; a plan
  reporting `fail`/`partial`/`blocked` is `failed`.
- Compatibility: pre-existing rows have no `result_kind`, so the replay branch reads `None` and
  they replay with their original semantics. The three readers of this table
  (dispatch-runtime replay, orion-feedback-runtime evidence, the in-process tripwire deque) were
  swept; no `CHECK` constraint, no pydantic model, no dashboard aggregates it.
- **`status='success' ⟺ raw_len>0` no longer holds.** No code depended on it; ad-hoc SQL might.
  Documented in the README.

## Env/config changes

None. No env key added, removed, renamed, or changed in meaning. No `.env_example` touched, so no
`sync_local_env_from_example.py` run was required.

## Tests run

```text
tests/test_execution_dispatch_runtime_worker.py
tests/test_execution_dispatch_result_extraction.py
tests/test_execution_dispatch_runtime_store.py
tests/test_dispatch_starvation.py
tests/test_maintenance_dispatch_gating.py
tests/test_measure_arena_degeneracy.py
tests/test_feedback_builder.py                            211 passed

services/orion-execution-dispatch-runtime/tests/          9 passed
  (--ignore=tests/test_heartbeat_chassis.py: pre-existing FileNotFoundError
   on collection, present on main, unrelated)
```

Red-check verified for every load-bearing test, in a scratch copy, never via `git checkout` on a
file with uncommitted work (spec §6.6):

| break injected | tests that failed |
| --- | --- |
| constant-series correlation returns 0.0 not None | 2 |
| id parser uses first `proposal` marker not last | 1 |
| decision-insensitive redundancy fingerprint | 2 |
| family-2 payloads degrade to empty (pre-patch behaviour) | 6 parse + 5 worker |
| plan-status check disabled | 4 |
| old `s == "empty"` tripwire predicate | (included above) |

The empty-stays-empty guard passes under every break, as it must.

## Evals run

```text
none -- no service touched here has an evals/ directory.
```

The behaviour changed is result classification, whose quality signal is the `result_kind`
distribution this patch begins persisting. Recorded as a follow-up rather than claimed.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d --build   (x2)
  Container orion-athena-execution-dispatch-runtime Started

docker logs --since 8m | grep -icE "error|traceback|exception"   ->  0
docker logs --since 8m | grep -o "status=[a-z]* kind=[a-z]*" | sort | uniq -c
       4 status=success kind=observation
       1 status=success kind=structured
theater_tripwire_active occurrences                              ->  0
```

Live path moved, post-deploy, from `substrate_dispatch_results`:

```
status   kind        observation                                      created_at
success  structured  declined_no_pressure: used_pct 71.426 < 75.0     23:56:16Z
empty                                                                 23:55:30Z   <- pre-deploy
```

## Review findings fixed

Code review run in a subagent at `high` effort. Every finding independently reproduced against
live code before being fixed.

- **Finding (BLOCKING): a failed skill verb was stored as `success` and emitted as `success=True`.**
  - Cause: `main.py:696` hardcodes `CortexExecResultPayload(ok=True)`; `cortex_client` checks only
    the codec's `ok`; `router.py:1396` sets `overall_status="fail"` but still populates
    `final_text`. My `result_kind` branch then graded that content as success — persisting a
    summary reading `"failed: Command 'docker builder prune' timed out after 600 seconds"` as a
    win, and feeding a positive score into `orion/feedback/builder.py`.
  - Fix: `plan_execution_status()` reads the plan's own verdict from the same payload, checked
    before any content classification. `None` is not graded — an unreadable shape falls through
    rather than getting a fabricated verdict.
  - Evidence: reviewer's repro through the real `_send_prepared_candidates`; I independently
    confirmed all three upstream links in source. 7 new failure-path tests, red-checked.

- **Finding: the theater tripwire could never fire for a skill verb.** It counted only
  `status == "empty"`; once skill failures were classified honestly as `"failed"`, a maintenance
  verb failing all ten trailing dispatches would not trip the one mechanism built to catch a dead
  motor nerve. My own patch narrowed it.
  - Fix: counts every non-success status. Widening to RPC failures is intended and disclosed.
  - Evidence: live trip risk measured, not assumed — 30 `failed` / 12,782 `success` over 24h
    (0.23%) against a >50% threshold. The test asserting the old narrower behaviour is reversed
    with its reasoning recorded, not quietly deleted.

- **Finding: section D's "starved templates" headline was 100% false positives.** All three
  reported templates are policy-blocked by design.
  - Fix: separates `blk:policy` from `blk:lost`, prints the block-reason breakdown the report was
    already computing and discarding, and reports `STARVED: 0`.
  - Evidence: 100% of their blocked rows carry `policy_decision:*`; `proposal_kind_to_cortex` has
    no route for `defer` or `request_policy_review`.

- **Finding: dispatch frames windowed on `created_at`, proposals on `generated_at`.** Median live
  lag 287s, max 918s; at `--window-hours 0.1` this produced five false starved templates.
  `created_at` is also the unindexed column.
  - Fix: `generated_at` for both.

- **Finding: section C conflated empty frames with identical non-empty ones** — two findings with
  different remedies behind one 80.8% headline.
  - Fix: split, with separate run-length stats for each population.

- **Finding: section A mixed denominators** (all candidates vs candidates with readable urgency).
  Currently equal only because `urgency_score` is a required schema field — correct by accident.
  - Fix: one denominator; label corrected to describe what it actually measures.

- **Finding: README's tripwire contract line was made false by commit 2, and no PR report was
  committed.**
  - Fix: this document, plus the README correction, status-value table, and `raw_len` caveat.

Cleared by the review, recorded so they are not re-litigated: `template_key_from_proposal_id`
validated against 182,328 live decision ids and 170,428 proposal ids with **0 unparseable**; the
family-1/family-2 discriminator cannot misroute (no skill payload in `verb_adapters.py` contains
an `observation` key); no reachable division by zero; read-only enforcement is real and verified
from inside a transaction.

## Restart required

Already applied on this host during verification:

```bash
scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d --build
```

## Risks / concerns

- **Severity: medium.** The arena degeneracy in section A/B is **diagnosed, not fixed**. Rank
  order is still decided by hand-authored `base_priority` constants. This patch deliberately does
  not touch scoring (§6.1, one mechanism per deploy); that is §5.1's job and it needs the
  calibration layer first.
- **Severity: medium.** 46.6% of policy frames still carry zero decisions and are published
  anyway, in runs averaging 12.6. That is an empty-shell-cognition violation at the top of the
  pipeline and is unaddressed here.
- **Severity: low.** The tripwire now counts RPC send failures, which it did not before. Measured
  as far below the threshold on real data, but it is a real widening of a mechanism that pauses
  dispatch.
- **Severity: low.** A verb returning non-JSON plain text still degrades to `empty`. Does not
  occur in live data (12,782 success results in 24h, all valid JSON), left alone deliberately
  rather than fixed as an unmeasured second mechanism.
- **Pre-existing, not caused here:** `check_service_env_compose_parity orion-execution-dispatch-runtime`
  reports 3 of 24 `.env_example` keys missing from `docker-compose.yml`
  (`EXECUTION_DISPATCH_STALENESS_MIN_SEC`, `_MAX_SEC`, `EXECUTION_DISPATCH_RUNTIME_PORT`).
  Confirmed identical on `main`. Not fixed here.
- **Pre-existing, not caused here:** `services/orion-execution-dispatch-runtime/tests/test_heartbeat_chassis.py`
  fails collection with `FileNotFoundError` on `main` too.

## Follow-ups this creates

1. §5.2 metric contract registry + scheduled gate.
2. §5.1 quantile calibration at the boundary — the actual fix for section A/B.
3. Change-detection on policy-frame production, targeting the 46.6% empty and the 80.8%
   redundancy separately.
4. An eval harness for dispatch result classification; none of the touched services has one.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/metric-commensurability
