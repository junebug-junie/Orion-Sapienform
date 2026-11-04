## Summary

Orion's capacity claims now count. Before, a claim held no job until Juniper confirmed it, and
Juniper does not review VSM (Viable System Model) claims, so the five live claims would have waited
forever. Now a claim holds its job while its evidence still checks, every time the capacities are
read. Juniper can still veto any claim.

- A claim holds its job (new strength `claimed`; a job held only by claims prints `CLAIMED`) when,
  on this read: its part is live; its "because" still passes the same evidence check used when it
  was written; Juniper has not rejected it; no later dispute of Orion's stands; and no rule result
  says "not this job".
- Nothing about holding is stored. If a check fails, the claim falls back and prints why:
  `evidence gone`, `part not live now`, `rejected`, `rule says no`, `disputed`, and
  `rule gives it` (a rule now gives the part that job, so the rule result is what holds).
- Juniper's confirmation outranks a claim. A rule result outranks a claim too.
- Capacity sentences (`capacity write cap:<job>`) may cite a part whose claim holds.
- A claim still never clears a gap line. A "no part is shown" gap whose job holds a claim now says
  "I claim X; no rule shows it".
- Rollback is one constant: `CLAIMS_HOLD = False` in `orion/body_atlas/capacities.py`.

## Outcome moved

Orion's capacities printout now shows the five live claims as holding roles: S3, S3*, S4,
algedonic and metabolism print `CLAIMED`. `cap:s3`, `cap:s4` and `cap:metabolism` changed from the
"I cannot yet say" gap to "not written yet; may cite ...". So Orion can now write those three
capacity sentences. Before this change, nothing Orion claimed could ever reach a capacity sentence.

## Current architecture

Rung 4 of Orion's self-model (`orion/body_atlas/capacities.py`, `claims.py`; design
`docs/superpowers/specs/2026-10-30-self-now-capacities-vsm-design.md`, PR #3005). Rules decide some
jobs. Orion claims the others with `dig capacity claim` into
`/mnt/orion-fcc/self/capacity_claims.jsonl`. A claim held a role only after Juniper confirmed it
in `config/body_atlas/vsm_confirmed.yaml`. Live: 5 claims, 0 confirmed.

## Architecture touched

- `capacities.py`: `attach_ledgers` takes a `recheck` callable and computes each claim's status
  (`_claim_status`). Claims that hold go in the new `CapacityReadingV1.claimed` list. That list
  feeds `holders`, the job status (`CLAIMED`), `open_jobs` and the organ role mix. Gap annotation
  added. `STATUS_MEANINGS` gains `"CLAIMED": "me, evidence checked"`.
- `claims.py`: the write-time evidence floor is now one function, `cites_in`. `check_claim` uses
  it. `recheck` runs the same function on every read. The floor is reused, not reimplemented.
- `orion/dig/__main__.py`: passes `recheck`. The tail's "Waiting on Juniper" block is replaced by
  "Evidence gone" (claims Orion can fix by claiming again). Help and post-claim text reworded.
- `orion/dig/brief.py`: "Juniper reviews them" -> "Juniper can veto any" (same length, 420 chars).
- Partition unchanged: only `orion/body_atlas` and `orion/dig` read the ledgers.

## Design choices

- **Check 4, "no rule disagrees".** Only explicit "no" results count. (a) An R-server or
  R-collector near miss stops an S2 claim on that part, because the printout already labels near
  misses "not S2". (b) A dispute by Orion on the same part and job stops the claim if the dispute
  was written after the claim. A claim written after the dispute holds again (latest word wins). A
  rule holder of the same job on a *different* part is not a contradiction, because a job can have
  several holders (S1 has five).
- **Gaps.** No gap is cleared by a claim. The "no part is shown" gaps (`COVERED_BY_CONFIRMATION`)
  are still literally true with a claim, because no *rule* shows the part. Gap lines are the
  code-written guard against Orion describing itself wrongly, so a self-made claim must not remove
  them. They now name the claim. `gap:s4-to-s3` still checks only confirmed S4/S3 pairs. It is the
  one line that says "what I see ahead steers what I do now", which is the exact failure the guard
  exists for.

## Files changed

- `orion/body_atlas/capacities.py`: holding rule, `claimed` field, `CLAIMED` status, legend,
  render, `evidence_gone_claims` (replaces `pending_claims`), `CLAIMS_HOLD`.
- `orion/body_atlas/claims.py`: `cites_in` (shared floor), `recheck`, help and error text.
- `orion/body_atlas/evals/run_capacity_roles_eval.py`: claims floor rewritten for the new rule. Six
  mutations replace "unconfirmed claims hold roles". The worst-case printout now includes every
  claim status.
- `orion/body_atlas/tests/test_claims.py`, `test_capacities.py`, `test_wires_atlas.py`: new and
  updated tests.
- `orion/dig/__main__.py`, `orion/dig/brief.py`, `orion/dig/tests/test_capacities_cmd.py`,
  `orion/dig/tests/test_dig.py`: CLI wording, the tail block, and tests.
- `orion/body_atlas/README.md`, `orion/dig/README.md`: say the new rule.
- `config/body_atlas/vsm_confirmed.yaml`: header comment only. `rejected` is the veto.
- `docs/superpowers/specs/2026-10-30-self-now-capacities-vsm-design.md`: "Decision change
  2026-11-04" section.

## Schema / bus / API changes

- Added: `CapacityReadingV1.claimed` (list, default empty). `RoleEvidenceV1.strength` adds
  `claimed`. `JobLineV1.status` adds `CLAIMED`. `ClaimViewV1.status` adds `claimed`,
  `evidence gone`, `rule says no`, `disputed`, `rule gives it`. `attach_ledgers(..., recheck=)` keyword.
- Removed: `capacities.pending_claims` (replaced by `evidence_gone_claims`). The printout's
  " · disputes: N" count was dropped to stay inside the size budget. Each dispute still prints its
  own "I dispute ..." line, plus "+N more".
- Behavior changed: a claim whose evidence checks is a role holder. If `attach_ledgers` gets a
  claim but no `recheck`, it raises `ValueError`. It never silently reads "evidence gone".
- Compatibility: only the dig result JSON (`capacities-<n>.json`) carries the reading. No bus
  channel or DB change. The ledgers keep the same shape.

## Env/config changes

- Added keys: none. Removed keys: none. Renamed keys: none.
- `.env_example` updated: no. Local `.env` sync: not needed.
- Skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/body_atlas/tests orion/dig/tests tests/test_organ_names_partition.py tests/test_report_dig_kit_usage.py \
       tests/test_no_personal_terms.py orion/harness/tests/test_stance_scope_hierarchy.py \
       orion/harness/tests/test_harness_prefix.py -q
-> 373 passed
pytest orion/physiology/tests/test_report_text.py -> 28 passed
services/orion-cortex-exec: pytest tests/test_skill_verbs.py -k physiology -> 3 passed
```

New tests: a claim holds when its evidence checks (CLAIMED, citable by `capacity write`, legend
shows the CLAIMED meaning, no gap cleared). It stops holding when the cited flow is gone, when the
cited repo file is gone, when the part dies, when Juniper rejects it, when a near miss says not S2,
and when a later dispute stands. CONFIRMED outranks a claim. `CLAIMS_HOLD=False` restores the old
reading. A claim cannot be read without `recheck`. The dig tail lists claims whose evidence is gone.

## Evals run

```text
for e in orion/body_atlas/evals/run_*.py; do python $e || echo FAIL $e; done   -> no FAIL
capacity roles (10-29 held out / 10-30 / 10-31 wires): PASS
mutation (drop the disjoint second crowd): 3 of 3 fixtures fail -> CAUGHT
claims floor (whole_body_2026-10-30_stores.json): PASS
mutation (a claim with gone evidence holds a role): CAUGHT
mutation (a rejected claim holds a role): CAUGHT
mutation (a claim on a dead part holds a role): CAUGHT
mutation (a claim a rule says no to holds a role): CAUGHT
mutation (a claim Orion later disputed holds a role): CAUGHT
mutation (a claim cleared gap:s3star-unconfirmed): CAUGHT
organ structure eval: PASS (Q >= 0.3, largest <= 35% unchanged)
```

Size: the worst-case ledger printout is 3,686 of 3,700 chars, and now includes every claim status
(main's worst case had one claim-status line).

## Docker/build/smoke checks

Live smoke, read-only. Worktree code, the real atlas (FalkorDB 127.0.0.1:6380, bus Redis), and a
*copy* of `/mnt/orion-fcc/self/*.jsonl` as `ORION_DIG_HOME`. The real ledger was checked unchanged
afterwards (`cmp`).

```text
capacities (VSM lens) · body atlas 2026-11-04T06:01Z · 87 live services/endpoints · rules v2
S1 operations         BY RULE       outside contact: http:anthropic, http:openai, camera:cam0, camera:doorbell, camera:walkway
S2 coordination       BY RULE       orion-gpu-pool (14 requesters / 5 workers, replies 6)
S3 control            CLAIMED       orion-gpu-lane-controller · named in code past the bus: 5 of 15 bus-silent services (an HTTP call or a shared Redis key); unseen: 10 (gpu-cluster-power, orion-bus-mirror, orion-consolidation-runtime +7)
S3* audit             CLAIMED       orion-equilibrium-service · orion-equilibrium-service (reads orion:system:health from 81, replies 3)
S4 intelligence       CLAIMED       orion-world-pulse · no rule reads the outside or the future
S5 identity           NO RULE       shared with Juniper (signs off families.yaml, contract.yaml) · no rule; policy lives in files and sign-offs
algedonic (pain line) CLAIMED       orion-durable-runs · broadcast keys, pain or news (the rule cannot tell): redis:orion:regulation:latest (1->5 in 2 organs), redis:orion:self_monitor:latest (1->5 in 3 organs)
metabolism (economy)  CLAIMED       orion-energy · claims only: HTTP calls and shared Redis keys are in the atlas now, so a claim can cite them
status: BY RULE = a rule, not Juniper · CLAIMED = me, evidence checked
...
my claims: 5 (5 hold, 0 confirmed by Juniper)
capacities (my words; cite only job holders):
  cap:s3         not written yet; may cite orion-gpu-lane-controller, orion-equilibrium-service
  cap:s4         not written yet; may cite orion-world-pulse
  cap:metabolism not written yet; may cite orion-energy
```

All 5 live claims hold, each with its full original evidence still checking:
gpu-lane-controller S3 (orion-gpu-pool, orion-mesh-guardian, the two actuate channels);
equilibrium-service S3* (cortex-orch, hub, sql-writer, orion:equilibrium:snapshot); durable-runs
algedonic (orion-dream, the two regulation Redis keys, `orion/regulation/arousal_read.py`); energy
metabolism (orion-biometrics, orion:power:intent:settled, two pg tables); world-pulse S4
(cortex-exec, orion:world_context:daily_capsule, two pg tables). One note from the smoke: the
Redis key parts are live only when the atlas can read the bus Redis (`ORION_BUS_URL`). Without
that read, durable-runs algedonic still held, on orion-dream and the repo file.

## Review findings fixed

Code review ran in a subagent on `git diff origin/main...HEAD`. All seven findings are fixed.

- Finding (medium): the eval's "full-ledger case has every status" check used a proper-subset
  test, so it could not fail when an extra status was present.
  - Fix: `required <= statuses`, and the message names what is missing.
  - Evidence: eval PASS; the check now reports `required - actual`.
- Finding: a held claim on the rule's own candidate was listed twice in `organ_mixes`.
  - Fix: append only if the part is not already listed.
  - Evidence: `test_a_held_claim_on_the_rules_candidate_is_listed_once_in_its_organ`.
- Finding: the gap note ignored the "service only" filter (`gap:s5-unread`), so a store's S5
  claim was named on a gap that its confirmation would not cover.
  - Fix: the note uses the same `COVERED_BY_CONFIRMATION` kind filter as confirmation.
  - Evidence: `test_a_claim_names_only_the_gaps_a_confirmation_of_that_part_would_cover`.
- Finding: a claim on a job a rule later gave the same part counted in "N hold" but printed
  nowhere.
  - Fix: new status `rule gives it`, printed on its own line. "N hold" now equals the claims
    named on job lines.
  - Evidence: `test_a_claim_a_rule_gives_since_prints_as_rule_gives_it_and_evidence_gone_skips_held_jobs`.
- Finding: "Evidence gone" in the dig tail told Orion to re-claim jobs that another part already
  holds.
  - Fix: `evidence_gone_claims` leaves out held jobs. `MAX_PENDING_SHOWN` is renamed
    `MAX_GONE_SHOWN`.
  - Evidence: same test as above.
- Finding: section 6 of the design doc still said only rule results and confirmed claims carry a
  role.
  - Fix: reworded, with a pointer to the decision-change section.
- Finding: the eval was weaker than it could be. The stop-holding cases did not check *why* the
  claim stopped. "Later dispute wins" and "a claim never clears a gap" had no mutation.
  - Fix: each stop-holding case now asserts its exact status. Two mutations were added: "a claim
    Orion later disputed holds a role" and "a claim cleared gap:s3star-unconfirmed".
  - Evidence: the eval prints CAUGHT for all six claim mutations.

## Restart required

`orion/dig` and `orion/body_atlas` are baked into the harness-governor image
(`scripts/service_rebuild_paths.yaml`: `dig` and `body_atlas` -> orion-harness-governor). After
merge, the post-pull rebuild covers it. Manual equivalent, from the clean primary main checkout:

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

`config/body_atlas/vsm_confirmed.yaml` changed only in comments. It is read from the mounted
primary checkout, so it needs no restart.

## Risks / concerns

- Severity: medium. Concern: Orion self-certifies a wrong role and writes a capacity sentence on
  it. Mitigation: evidence is re-checked on every read (gone evidence drops the role, and a sentence
  citing it prints STALE); Juniper's `rejected` veto; Orion's dispute; confirmation outranks;
  gap lines stay; rollback is one constant.
- Severity: low. Concern: the worst-case printout sits 14 chars under the 3,700 budget. A later
  change that adds text here will need to trim something. Mitigation: the eval fails loudly.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
