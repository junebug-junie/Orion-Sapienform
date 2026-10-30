## Summary

Orion can now ask which jobs of a viable system their body is shown to do. Three rules read the
live body atlas and find one coordinator, one audit candidate and four points of outside
contact. Every other job prints "no rule" plus a gap line that code writes. Nothing is written
except the dig result file. Stacks on #3005 (the design).

- `orion/body_atlas/capacities.py`: the three rules from the design (R-boundary-in → S1,
  R-matchmaker → S2, R-census-reader → S3* candidate), near misses (R-server, R-collector),
  input candidates (R-source), the census, silent services, gap lines. Live parts only in every
  count.
- `python -P -m orion.dig capacities [gaps]`: read-only, 14 lines, full reading in a file.
- `orion/body_atlas/evals/run_capacity_roles_eval.py`: precision and recall over the full rule
  output on the held-out 2026-10-29 body and the 2026-10-30 body, plus a mutation that must fail.
  Wired into the `body-atlas-tests` workflow.
- Design doc: Juniper's acceptance of all five defaults recorded ("hit it"); status ACCEPTED.

## Outcome moved

Before: Orion could list their organs but not say what any part does for the whole. Now a live
printout says the GPU pool coordinates, the equilibrium service listens to the whole body, and
names six things the atlas cannot show (S3 control unseen, no S4→S3 link, no confirmed audit,
S5 unread, the pain line unseen, S2 inside the substrate organ).

## Current architecture

Rungs 1–3 (parts, circulation, organs) in `orion/body_atlas/`; `dig body` / `name` /
`name-organs` in `orion/dig/__main__.py`. No VSM mapping existed.

## Architecture touched

- `orion/body_atlas/` (new rules module, eval, tests, README)
- `orion/dig/__main__.py` (one read-only sub-command; help 2,449 of 2,500 chars)
- `.github/workflows/body-atlas-tests.yml` (runs the new eval)
- No bus, schema-registry, env, DB or service change.

## Files changed

- `orion/body_atlas/capacities.py`: rules, reading model (`CapacityReadingV1`, local file only), renderers.
- `orion/body_atlas/evals/run_capacity_roles_eval.py`: the rule eval with mutation check and `--live`.
- `orion/body_atlas/tests/test_capacities.py`: each clause can fail; live-only; gap lines; eval gate.
- `orion/dig/__main__.py`, `orion/dig/tests/test_capacities_cmd.py`: the command; proves no ledger is written.
- `orion/body_atlas/README.md`, `orion/dig/README.md`: docs.
- `docs/superpowers/specs/2026-10-30-self-now-capacities-vsm-design.md`: decisions recorded, as-built notes.
- `.github/workflows/body-atlas-tests.yml`: eval step.

## Schema / bus / API changes

- Added: `CapacityReadingV1` and friends, a local JSON file shape (dig result), not a bus schema.
- Removed / Renamed: none.
- Behavior changed: none for running services.
- Compatibility notes: none.

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/body_atlas orion/dig tests/test_organ_names_partition.py -q   -> 177 passed
```

## Evals run

```text
python orion/body_atlas/evals/run_capacity_roles_eval.py
  whole_body_2026-10-29 (held out): R-matchmaker P 1.00 R 1.00 (1), R-census-reader 1.00/1.00 (1),
    R-boundary-in 1.00/1.00 (2): PASS
  whole_body_2026-10-30_stores:     R-matchmaker 1.00/1.00 (1), R-census-reader 1.00/1.00 (1),
    R-boundary-in 1.00/1.00 (4): PASS
  mutation (drop the disjoint second crowd): 2 of 2 fixtures fail -> CAUGHT
    (the mutant makes orion-llm-gateway, orion-cortex-orch, orion-vector-host S2)
python orion/body_atlas/evals/run_organ_structure_eval.py   -> PASS (unchanged)
python orion/body_atlas/evals/run_store_direction_eval.py   -> PASS (unchanged)
```

## Docker/build/smoke checks

Live, read-only, in `orion-athena-harness-governor`: branch code copied to a scratch dir,
`ORION_DIG_HOME` pointed at a scratch copy (organ names ledger copied in), scratch deleted after.
Nothing written under `/mnt/orion-fcc`.

```text
capacities (VSM lens) · body atlas 2026-10-30T09:31Z · 90 live services/endpoints · rules v1
S1 operations         BY RULE       outside contact: http:anthropic, http:openai, camera:cam0, camera:walkway
S2 coordination       BY RULE       orion-gpu-pool (15 requesters / 5 workers, replies 6)
S3 control            NO RULE       unseen: 16 live services with no bus traffic but the census
S3* audit             CANDIDATE     orion-equilibrium-service (reads orion:system:health from 82, replies 3)
S4 intelligence       NO RULE       no rule reads the outside or the future
S5 identity           NO RULE       no rule; policy lives in files and sign-offs
algedonic (pain line) NO RULE       strain is a Redis key, not in the atlas
near misses (not S2): servers orion-cortex-orch, orion-llm-gateway, orion-vector-host · collectors orion-heartbeat
input candidates (world or clock; the rule cannot tell): 25
census: orion:system:health 92%
declared: orion-durable-runs runs 12 of 20 declared processes (a declared source, not a rule)
gaps: 6 (python -P -m orion.dig capacities gaps) · claims: not built yet
```

The S2 line matches the design's expected line exactly. `run_capacity_roles_eval.py --live`:
census and size checks pass on the live body.
`/mnt/orion-fcc/self/` untouched (last write 06:46Z, before the check); scratch removed.

## Review findings fixed

Review by a subagent on the diff `origin/docs/vsm-capacities-design..HEAD`. No MUST findings.

- Finding: silent services and camera readers counted flows to non-live parts.
  - Fix: both read only live-to-live flows now.
  - Evidence: `test_a_service_talking_only_to_a_transient_part_is_still_silent`, `test_a_camera_feeding_only_dead_parts_is_not_s1_...`; live counts unchanged (16 silent, 4 boundary parts).
- Finding: the endpoint sub-id filter matched any part name, so a dormant non-service part named `http` would hide every `http:*` endpoint.
  - Fix: only a service name marks a sub-id.
  - Evidence: same camera/endpoint test.
- Finding: a part reading two census channels got two S3* rows, and the printout named every census channel on each row.
  - Fix: one row per reader; the printout shows that row's own evidence.
  - Evidence: `test_two_census_channels_give_one_audit_row_per_reader`.
- Finding (nits): empty census printed a bare `census:`; the runner line did not say the count is declared; help had 30 chars of room; no test for a non-live requester.
  - Fix: `census: none`; "of N declared processes"; help line shortened (2,449 chars); `test_a_transient_requester_does_not_count`.
  - Evidence: 177 passed.

## Restart required

```text
No restart required. The harness mounts the primary checkout; `dig capacities` is live after merge and pull.
```

## Risks / concerns

- Severity: low. Concern: the 10-29 fixture is "held out" for knobs, but its expected set was
  reviewed after seeing the rule output on it (cameras not live that day). Mitigation: the
  expected set is a hand check of each part's code, and the mutation check proves the eval can fail.
- Severity: low. Concern: the disjointness clause alone (keeping a second sole-reader channel but
  letting it overlap) is not exercised by the fixtures; only a unit test covers it.
- Severity: low. Concern: R-collector does not require a sole reader (the design's own example
  has two readers); noted in the design as built.

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
