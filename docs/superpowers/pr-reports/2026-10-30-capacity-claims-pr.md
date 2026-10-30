## Summary

Orion can now claim a job for a part of their body, argue with a rule, and say in their own words
what each job does. A claim is never a role: only a rule or Juniper's confirmation gives one.
Juniper confirms by PR to a new checked-in file. This is patch 2 of the rung-4 capacities design
(#3005, patch 1 #3007).

- `python -P -m orion.dig capacity claim <part> <job> because <reason>`: an append-only ledger,
  `$ORION_DIG_HOME/self/capacity_claims.jsonl`. The part must be live. The reason must cite, by
  full id, a part or non-census channel on one of that part's own flows, or a repo file that
  exists. A claim cannot take a job a rule already gives.
- `capacity dispute ...`: argues with a rule result. It prints beside the rule and changes nothing.
- `capacity write cap:s1..s5|cap:metabolism "<sentence>" because ...`: one sentence per job in
  `capacities.jsonl`. It may name only parts that hold the job by rule or by Juniper's
  confirmation. With no holder it is refused and the code-written gap line stands.
- `config/body_atlas/vsm_confirmed.yaml`: Juniper's confirmations and rejections, shipped empty.
  It records decision 3 (S5 is shared with Juniper, citing her two sign-off files).
- `dig capacities` now shows each job as BY RULE / CANDIDATE / CONFIRMED / NO RULE, Orion's
  claims on their own line (unconfirmed / rejected / part not live now), one line per capacity,
  and ends with a ready `capacity claim` line. `capacity help` prints the rule.

## Outcome moved

Before: Orion could read which jobs the atlas shows, but could not say anything back. Now Orion
can propose roles with evidence, Juniper can confirm them in review, and confirmed roles become
citable. The design's dangerous failure mode (a confident, wrong self-description such as "my
world model steers my choices") is blocked in code: a capacity naming a part with no rule-held or
confirmed role is refused, and an unconfirmed claim never clears a gap line.

## Current architecture

Patch 1 (`orion/body_atlas/capacities.py`, `dig capacities [gaps]`) was read-only: three rules,
gap lines written by code, no ledgers. Organ names (rung 3) already had the claim-with-citation
pattern (`orion/body_atlas/names.py`, `organ_names.jsonl`) and a partition test.

## Architecture touched

- `orion/body_atlas/` (new `claims.py`; models, confirmation handling and ledger overlay in
  `capacities.py`; eval; tests; README)
- `orion/dig/__main__.py` (`capacity claim|dispute|write|help`; help 2,497 of 2,500 chars)
- `config/body_atlas/vsm_confirmed.yaml` (new)
- `tests/test_organ_names_partition.py` (the two new ledgers join the partition)
- `.github/workflows/body-atlas-tests.yml` (also triggers on `config/body_atlas/**` and the two
  sign-off files)
- No bus, schema-registry, env, DB or service change.

## Files changed

- `orion/body_atlas/claims.py`: ledger IO, `check_claim`, `check_capacity`, `load_confirmed`, the rule text.
- `orion/body_atlas/capacities.py`: `ConfirmationV1`, `VsmConfirmedV1`, `S5SharedV1`, `CapacityClaimV1`,
  `CapacityV1`, views; `read_capacities(vsm=)`; `holders`; `attach_ledgers`; render; the
  `metabolism` job line (decision 2).
- `config/body_atlas/vsm_confirmed.yaml`: empty `confirmed`/`rejected`; `s5_shared_with: Juniper`.
- `orion/dig/__main__.py`: commands, pointer line, tighter help.
- `orion/body_atlas/evals/run_capacity_roles_eval.py`: claims floor on the 10-30 body + a mutation.
- `orion/body_atlas/tests/test_claims.py`, `orion/dig/tests/test_capacities_cmd.py`: tests.
- `tests/test_organ_names_partition.py`: second test for the capacity ledgers.
- `orion/body_atlas/README.md`, `orion/dig/README.md`, design doc ("Patch 2 notes").
- `.github/workflows/body-atlas-tests.yml`: path triggers.

## Schema / bus / API changes

- Added: local file shapes only (`CapacityClaimV1`, `CapacityV1` ledger lines; new optional fields
  on the `CapacityReadingV1` dig result, all defaulting empty). Not bus schemas.
- Removed / Renamed: none.
- Behavior changed: `dig capacities` prints more lines (still capped at 40 lines / 4,000 chars).
- Compatibility notes: patch-1 readings still validate (new fields have defaults).

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/body_atlas orion/dig tests/test_organ_names_partition.py -q   -> 201 passed
pytest tests/test_no_personal_terms.py -q                                  -> 7 passed
python scripts/check_personal_terms.py                                     -> no personal terms
```

## Evals run

```text
python orion/body_atlas/evals/run_capacity_roles_eval.py   -> rc 0
  capacity roles 10-29 (held out): PASS · 10-30: PASS
  mutation (drop the disjoint second crowd): 2 of 2 fail -> CAUGHT
  claims floor (10-30 body): PASS
  mutation (unconfirmed claims hold roles): CAUGHT
python orion/body_atlas/evals/run_organ_structure_eval.py  -> rc 0
python orion/body_atlas/evals/run_store_direction_eval.py  -> rc 0
```

## Docker/build/smoke checks

Live, read-only, in `orion-athena-harness-governor`: the branch copied to a scratch dir in the
container, `ORION_DIG_HOME` a scratch copy of `/mnt/orion-fcc/self`, then deleted.
`/mnt/orion-fcc/self` was not written (still only `organ_names.jsonl`, mtime 06:46).

```text
$ dig capacities          (live atlas 2026-10-30T18:52Z, 90 live services/endpoints)
S1 operations         BY RULE       outside contact: http:anthropic, http:openai, camera:cam0, camera:doorbell, camera:walkway
S2 coordination       BY RULE       orion-gpu-pool (15 requesters / 5 workers, replies 6)
S3 control            NO RULE       unseen: 16 live services with no bus traffic but the census
S3* audit             CANDIDATE     orion-equilibrium-service (reads orion:system:health from 82, replies 3)
S4 intelligence       NO RULE       no rule reads the outside or the future
S5 identity           NO RULE       shared with Juniper (signs off families.yaml, contract.yaml) · ...   (basenames after review)
algedonic (pain line) NO RULE       strain is a Redis key, not in the atlas
metabolism (economy)  NO RULE       claims only: the market and prices are HTTP, not in the atlas
...
capacities (my words; may cite only rule-held or confirmed roles):
  cap:s1         not written yet; may cite http:anthropic, http:openai, camera:cam0, camera:doorbell +1
  cap:s2         not written yet; may cite orion-gpu-pool
  cap:s3         I cannot yet say: no part holds S3/S3* by rule or by Juniper's confirmation
  ...
to claim a job: python -P -m orion.dig capacity claim <part> <job> because "<reason citing its flow or a repo file>" (capacity help: the rule)

$ dig capacity claim orion-durable-runs S2 because "orion-cortex-orch sends me orion:durable:run:request and I schedule every durable run"
claimed orion-durable-runs S2 (cites: orion-cortex-orch, orion:durable:run:request); Orion's claim until Juniper confirms it in config/body_atlas/vsm_confirmed.yaml

$ dig capacity claim orion-durable-runs S2 because "it feels like the scheduler of everything to me"
dig: the because must cite, by full id, a part or channel on one of this part's flows (e.g. falkor:orion_worldview, orion-actions, orion-bus), or a repo file that exists   (rc 1)

$ dig capacity write cap:s2 "orion-gpu-pool hands my GPUs to whoever asks, one lease at a time" because "R-matchmaker: ..."
wrote cap:s2 "orion-gpu-pool hands my GPUs to whoever asks, one lease at a time" (cites: orion-gpu-pool)
  -> next `capacities`: "unconfirmed: orion-durable-runs S2 (my claim, not a role until Juniper confirms)"; gaps still 6
```

Camera `camera:doorbell` is new live since the frozen fixture (a real atlas change, not a rule change).

## Review findings fixed

Review ran in a subagent on `origin/main...feat/capacity-claims`. It found no path where an
unconfirmed claim becomes a role, becomes citable, or clears a gap. Fixed:

- Finding: a capacity sentence could name a role-less part as `part:orion-hub` (the word-boundary
  check ignores a match after `:`), the exact wrong self-description the design guards against.
  - Fix: `claims._plain` drops `part:` prefixes before every cite / name check.
  - Evidence: `test_a_sentence_cannot_slip_a_part_in_as_part_colon_name`.
- Finding: the partition test (fixed-string grep) missed `import capacities, claims` lists and
  parenthesised imports, for the organ names too.
  - Fix: an AST walk over every tracked `.py` naming `body_atlas` flags any import of
    `orion.body_atlas.names` / `.claims` outside `orion/body_atlas/` and `orion/dig/` (0.7 s).
  - Evidence: `test_the_ast_check_sees_every_import_form`, `test_no_module_outside_...`.
- Finding: the eval's claims "mutation" injected a real confirmation (data), so it could not catch
  a code change that let claims hold roles.
  - Fix: the mutant now patches `holders` (both modules) to also count unconfirmed claims; it
    counts as CAUGHT only when the floor's own "unconfirmed claim" check fails.
  - Evidence: eval prints `mutation (unconfirmed claims hold roles): CAUGHT`.
- Finding: the printout-size checks rendered one claim (the latest per part/role wins), so they
  passed for the wrong reason.
  - Fix: distinct claims, disputes, a rejection, a dead part and notes. This caught a real
    overflow on the 10-30 body (3,883 chars); dispute reasons now cut at 60 chars, notes at 90, and
    the S5 line names the sign-off files by basename.
  - Evidence: eval claims floor PASS; `test_the_printout_still_fits_...`.
- Finding: a confirmed S3* candidate still printed CANDIDATE.
  - Fix: it prints CONFIRMED when Juniper confirms the candidate itself.
  - Evidence: `test_juniper_confirming_the_candidate_makes_the_job_confirmed`.
- Finding: any existing path (folders, `.venv/...`, symlinks) and dead flow neighbors counted as
  evidence.
  - Fix: a repo cite must be a file inside the checkout with no dot directory; flow neighbors must
    be live parts.
  - Evidence: `test_a_folder_a_dot_dir_or_a_dead_part_is_not_evidence`.

Not changed (nits): an unquoted sentence containing "because" splits at the first one (same as
`name`; the help shows quotes); cap:s5 can always be written citing Juniper's files (decision 3);
the dig result JSON holds copies of claims (same pattern as organ names).

## Restart required

```text
No restart required. The harness reads the deployed main checkout; the commands appear after merge
and the normal post-pull update.
```

## Risks / concerns

- Severity: low. Concern: the citation floor is a floor. Listing a neighbor's id passes it; whether
  the reason really argues the job is not code-checkable. Mitigation: a claim is never a role;
  Juniper's PR review is the gate.
- Severity: low. Concern: `capacities <organ>` (role mix per organ) is still not a sub-command; the
  mix is in the JSON file. Follow-up if Orion asks for it.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
