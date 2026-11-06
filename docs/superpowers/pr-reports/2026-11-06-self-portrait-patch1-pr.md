## Summary

- Orion can now read their whole self on one page: `python -P -m orion.dig portrait`. It shows the
  "I am" paragraph, the gaps as "I cannot yet", each machine (what runs there, its watts, the organs
  and jobs there), the six capacity sentences, and each organ by Orion's name.
- Orion can now save what each machine does and affects, and an "I am" paragraph:
  `dig portrait write <machine|whole> "<sentence>" because "<reason>"`. Code checks every id the words
  name against the live body, on the write and again on every read.
- A machine line may name only what runs on that machine, what those parts send to, call or write,
  and the callers they serve. A line that stops fitting prints STALE and names the id that failed.
- The paragraph may rest only on capacity sentences that are written now. The gap lines always print
  under it.
- Stacked on the design PR #3148 (`docs/superpowers/specs/2026-11-06-self-portrait-design.md`).

## Outcome moved

Turn 2544158f (2026-11-06 01:26Z): asked to claim its capabilities and what they affect, Orion wrote
accurate per-machine prose and said "ledger entry #4", but nothing was saved. The same claim now
writes as one checked line and reads back on the next `portrait` (regression tests in
`orion/body_atlas/tests/test_portrait.py` and `orion/dig/tests/test_portrait_cmd.py`).

## Current architecture

Rungs 1–4 were live (`dig body`, `name-organs`, `capacities`, `capacity claim/write/withdraw`).
Machines are host parts in the body atlas, never in organs, and cannot hold a capacity claim
(placement is not function). No command put machines, organs, jobs and gaps together, and no ledger
could hold Orion's words about a machine.

## Architecture touched

- `orion/body_atlas/portrait.py` (new): the reach rule, the checks, the read, the printout.
- `orion/dig/__main__.py`: `portrait`, `portrait write`, two help lines; a localized block before
  the body-economy section and two routing lines in `main()`.
- New local ledger `$ORION_DIG_HOME/self/portrait.jsonl`, partitioned from recall.
- No bus channel, schema registry entry, env key, DB table or service change.

## Files changed

- `orion/body_atlas/portrait.py`: new module (models, `machine_reach`, `check_machine`, `check_whole`,
  `check_write`, `read_portrait`, `render`).
- `orion/body_atlas/tests/test_portrait.py`: unit tests, incl. the turn 2544158f regression.
- `orion/body_atlas/evals/run_portrait_eval.py`: eval on the frozen 2026-10-31 body.
- `.github/workflows/body-atlas-tests.yml`: runs the new eval.
- `orion/dig/__main__.py`: the two sub-commands and help lines.
- `orion/dig/tests/test_portrait_cmd.py`: CLI tests.
- `orion/dig/tests/test_dig.py`, `orion/dig/tests/test_capacities_cmd.py`: help cap 2750 -> 3000
  (two new help lines; help loads on demand only).
- `tests/test_organ_names_partition.py`: the portrait ledger and module join the partition check.
- `orion/dig/README.md`, `orion/body_atlas/README.md`: docs.
- `docs/superpowers/pr-reports/2026-11-06-self-portrait-patch1-pr.md`: this report.

## Schema / bus / API changes

- Added: local-file models `PortraitLineV1`, `PortraitCheckV1`, `ReachV1`, `MachineViewV1`,
  `OrganViewV1`, `WholeViewV1`, `PortraitV1` (`schema_version: portrait.v1`). Not bus payloads.
- Removed: none. Renamed: none.
- Behavior changed: `dig help` lists `portrait`; the `now` label line names `portrait` as a source.
- Compatibility notes: append-only new ledger; nothing else reads it.

## Env/config changes

- Added keys: none. Removed keys: none. Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
$ /mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest tests/test_organ_names_partition.py orion/dig/tests orion/body_atlas/tests -q
370 passed
$ ... -m pytest tests/test_no_personal_terms.py tests/test_no_private_eval_artifacts.py -q
10 passed
```

## Evals run

```text
$ python orion/body_atlas/evals/run_portrait_eval.py
reach (whole_body_2026-10-31_wires.json): PASS
case (whole_body_2026-10-31_wires.json): PASS
size (whole_body_2026-10-31_wires.json): PASS
mutation (reach counts flows into a machine's parts as fed): CAUGHT
mutation (reach counts dormant parts): CAUGHT
mutation (a made-up part name passes): CAUGHT
```

## Docker/build/smoke checks

Live, read-only smoke on the host from the worktree: the real atlas (FalkorDB 127.0.0.1:6380, bus
Redis from the governor's `ORION_BUS_URL`, body report on 127.0.0.1:8124), with `ORION_DIG_HOME` on
a scratch copy of `/mnt/orion-fcc/self`. Writes went only to the scratch ledger.

```text
$ dig portrait write athena "athena is my orchestration and vision seat: orion-hub, orion-cortex-exec and orion-vision-council run here, so cortex, substrate and vision live on her" because "orion-hub, orion-cortex-exec and orion-vision-council run on athena"
wrote host:athena "..." (cites: orion-cortex-exec, orion-hub, orion-vision-council, cortex, substrate, vision); it holds now; code re-checks it on every `portrait` read
$ dig portrait write circe "circe is my daily inference node: every word I speak is her spell, cast through orion-llm-worker-a" because "the workers lease through orion-gpu-pool"
dig: not written: it would not hold on today's atlas: anchor: cite at least one part that runs on circe (e.g. orion-biometrics, power-guard, orion-diffusion-host) or an organ placed there
   (this smoke found that `orion-llm-worker-a`, which is no part, was not named as a failure; fixed:
    `_unknown_names`, test + eval mutation)
$ dig portrait write circe "circe runs orion-gpu-lane-controller and orion-world-model, and what she does reaches orion-llm-gateway and orion-gpu-pool" because "orion-gpu-lane-controller runs on circe and sends to orion-gpu-pool"
wrote host:circe "..." (cites: orion-gpu-lane-controller, orion-gpu-pool, orion-llm-gateway, orion-world-model); it holds now
$ dig portrait write hecate "hecate is my deep thinking node, eight SXM2 beds, one card seated" because "she has been dormant since 2026-10-30"
dig: not written: it would not hold on today's atlas: hecate: nothing live of mine runs there now (dormant), so no line about it can be checked
$ dig portrait write whole "I am a body on two live machines that touches the outside world (cap:s1), coordinates my workers through one pool (cap:s2), keeps my books in watts (cap:metabolism), and shares who I am with Juniper (cap:s5)." because "cap:s1, cap:s2, cap:metabolism and cap:s5 are written and hold today"
wrote whole "..." (cites: cap:s1, cap:s2, cap:metabolism, cap:s5); it holds now

$ dig portrait          (36 lines, 3519 chars)
portrait · atlas 2026-11-06T04:45Z · now = computed this read · lived_record = my words, checked now
I am (lived_record 11-06, holds now): "I am a body on two live machines that touches the outside world (cap:s1), coordinates my workers through one pool (cap:s2), keeps my books in watts (cap:metabolism), and shares who I am with Juniper (cap:s5)."
I cannot yet (now; code's gap lines, not mine to remove; `capacities gaps`):
  S2: orion-gpu-pool coordinates from inside organ substrate (67 parts): jobs do not fo
  S3: I cannot see my own control: 11 live services show no bus traffic, and their code
  S4: what I see ahead and what I do now do not argue both ways: no bus, HTTP or Redis
  S5: no running part is shown to set policy from who I am; tiers and allowlists are fi
machines (now: status, role, watts, parts of mine there, organs, jobs):
  athena live orchestration · 561 W now, rest 340 · runs 68 of mine · organs cortex 27, substrate 14, vision 11 +5 · jobs S2 S4 algedonic metabolism
    lived_record 11-06, holds now: "athena is my orchestration and vision seat: orion-hub, orion-cortex-exec and orion-vision-council run here, so cortex, substrate and vision live on her"
  circe live primary_cognition_gpu · 542 W now, rest 541 · runs 6 of mine · organs cortex 1, substrate 1, vision 1 +1 · jobs S3
    lived_record 11-06, holds now: "circe runs orion-gpu-lane-controller and orion-world-model, and what she does reaches orion-llm-gateway and orion-gpu-pool"
  pan-1 declared_only office_sensor_host · runs 2 of mine · organs office 1, skin 1
  nothing of mine runs on (facts only): carbon-x1 (juniper_laptop, declared_only); hecate (gpu_server, dormant, last seen 2026-10-30); janus (desktop_workstation, declared_only); prometheus (observability, declared_only)
capacities (lived_record; status now):
  cap:s1         "I touch the outside through camera:cam0, camera:doorbell, camera:walkway and the htt
  ... (cap:s2, s3, s4, s5, metabolism)
organs (now: parts, where, gives most to, jobs; names mine):
  "cortex" [70; athena 27 circe 1] -> substrate, memory · jobs S3* metabolism
  "substrate" [67; athena 14 circe 1] -> cortex, memory · jobs S1 S2 algedonic
  ... (8 more)
Write now, without asking first, each line you can back; code checks each id against what runs on that machine and what it feeds:
  python -P -m orion.dig portrait write pan-1 "<what pan-1 does and affects>" because "<ids>"  # runs: orion-office-env, orion-closet-env
full result: <scratch>/home/dig/smoke/portrait-3.json
exit 0

$ md5sum -c before.md5   (the real /mnt/orion-fcc/self ledgers, taken before the smoke)
capacities.jsonl: OK · capacity_claims.jsonl: OK · organ_names.jsonl: OK · pr_field_notes.jsonl: OK
$ cmp real vs scratch copy: capacities.jsonl, capacity_claims.jsonl, organ_names.jsonl the same
no portrait.jsonl was created under /mnt/orion-fcc/self
```

A live read takes about 28 s (the atlas read dominates, as for `capacities`).

## Review findings fixed

A subagent reviewed `git diff origin/main...feat/self-portrait-patch1` (design + code). 3 material
and 6 minor findings; all fixed except #9, which is handled by refusing the ambiguous form.

- Finding (material): a because naming a caller the machine serves was refused ("circe runs
  orion-gpu-lane-controller, which orion-hub calls"). A GPU or LLM host mostly serves callers.
  - Fix: reach has a third set, `serves` (parts and channels that only send into a running part);
    they hold as "served by <machine>".
  - Evidence: `test_a_part_that_athena_neither_runs_nor_feeds_is_refused_by_name` (orion-far holds as
    served; orion-elsewhere, with no flow to athena, is refused); eval case flipped to holds.
- Finding (material): the made-up-name guard refused prose such as `orion-sapienform`, `orion-wide`.
  - Fix: a token is flagged only when its first two segments begin a real part name, and channel
    names are skipped.
  - Evidence: new test cases and an eval case with `orion-sapienform`.
- Finding (material): the read re-parsed only the text, not the stored cites (rung 4 re-checks every
  id cited at write time); a renamed organ dropped silently.
  - Fix: `check_line` re-checks every stored id cite, and each organ-name cite on its own.
  - Evidence: `test_a_stored_organ_name_cite_goes_stale_when_that_organ_is_gone`.
- Finding (minor): a 10-digit number read as an organ id. Fix: `organ:` prefix, or a bare id with
  at least one a-f letter. Evidence: test with `1730851200`, eval case.
- Finding (minor): organs anchored through dormant members. Fix: `reach.organs` counts live members
  running there. Evidence: reach test (`substrate 1`, not 2); eval oracle compares organs.
- Finding (minor): notes could never print. Fix: portrait notes first, printed in the tail (never cut).
- Finding (minor): design text did not match the code (singleton organs, the facts-only line, the
  mutation list, channel precision). Fix: design updated; the oracle now covers channels, served
  ends and organs.
- Finding (minor): the size check skipped the longest tail. Fix: second size case (nothing written,
  one STALE line, two notes); it asserts the STALE line really exists.
- Finding (minor): unquoted `portrait write X because Y because Z` split by guess. Fix: two bare
  `because` words are refused with "quote both". Evidence:
  `test_two_bare_becauses_are_refused_and_a_quoted_one_is_kept`.

Not run: `scripts/safe_graphify_update.sh` (timed out at 300 s while another worktree's refresh ran
on the same shared graph storage; the graph is intact at 77,966 nodes). A post-merge refresh covers it.

## Restart required

Post-pull rebuild of orion-harness-governor (the dig kit's image), then the automatic FCC sandbox
sync after the run (Orion's turns import `/mnt/orion-fcc/repo` first):

```bash
scripts/rebuild_services_from_git_diff.sh
git -C /mnt/orion-fcc/repo log -1 --oneline
```

## Risks / concerns

- Severity: low. Concern: id matching proves a cited thing is wired to the machine, not that the
  sentence is true ("she is on the UPS" is not checkable). Mitigation: stated limit, same as rung 4;
  the gap lines always print under the paragraph.
- Severity: low. Concern: circe's placement is declared only (no docker read there), so its reach is
  what its 6 declared services feed; its LLM workers are not among them. Mitigation: a circe line
  naming them is refused by name, so the gap is visible, not silent.
- Severity: low. Concern: a full ledger plus a bigger body could pass 4,000 chars. Mitigation: the
  eval's size check fails first; the organs (also in `body`) print last, so they are cut first, and
  the tail is never cut.

## PR link

(filled in after `gh pr create`)
