# PR: dig `body <part>` -- one part's flows, and the exact ids a claim on it may cite

## Summary

- Orion can now look at one part of their body: `python -P -m orion.dig body orion-energy` (or `part:orion-energy`). Before this, the command failed with "no multi-part organ 'organ:orion-energy'".
- The printout shows what the part is (kind, live or not, host, its organ and Orion's name for the organ), then its flows in and out, strongest first. Each flow shows the other part, the relation, the channels, and the weight. Census channels (the health heartbeat every part sends) are marked.
- It shows the part's jobs (by rule, candidate, or Juniper's confirmation) and Orion's own claims on it.
- It ends with `cite in a claim:` and the exact ids a capacity claim on this part may cite, plus a ready `capacity claim` line that already cites the first one. The ids come from `claims.evidence_ids`, the same function `capacity claim` checks against. Nothing re-implements the rule.
- Organ ids keep their old meaning: `organ:` plus 10 hex characters, or the 10 hex characters alone, still list the organ's members.
- Pointers to the new view: the dig `help` line, `capacity help`, the `Unclaimed jobs:` header in `capacities`, the live `body` summary footer, and the README table.

## Outcome moved

On turn 00655f13 (2026-11-04 05:04-05:37Z), Orion took 161 steps and 33 minutes to name 2 organs and write 5 capacity claims. Of 52 tool calls, 12 were hand-written Python scripts that parsed the body file to list one part's flows. About 8 more read `claims.py`, `names.py` and the dig source to learn what a claim may cite. With this patch, one command answers both questions for one part. Whether Orion uses it on the next claims turn is UNVERIFIED until such a turn runs on the rebuilt harness.

## Current architecture

- The dig kit (`orion/dig/__main__.py`) runs as a command-line tool through Bash inside `orion-harness-governor`. The tool is baked into that image.
- `body` with no argument printed the live body. A date printed a past export. Any other word was treated as an organ id (`find_organ`), so a part name failed.
- `capacity claim` requires the because to cite, by full id, a part or a non-census channel on one of the part's flows (`claims.evidence_ids`). No command printed those ids.

## Architecture touched

- Dig kit only: a new `cmd_body_part` view and `_find_part` resolver in `orion/dig/__main__.py`. `_capacity_reading` now accepts an atlas that has already been read, so `body <part>` reads the live body once.
- One footer line in `orion/body_atlas/__main__.py` (`summarize`).
- No service, bus channel, schema, env key, or database change. Read-only: the only write is the dig result file under `$ORION_DIG_HOME/dig/<dig>/`.

## Files changed

- `orion/dig/__main__.py`: part resolution, the part view, help text, `capacity help` line, pointer in the `Unclaimed jobs:` header. An unknown name now fails before any result file is written. Result file names are sanitized.
- `orion/dig/tests/test_body_part.py`: new tests (see below).
- `orion/body_atlas/__main__.py`: the live `body` footer also names `body <part>`.
- `orion/dig/README.md`: a table row for `body <part>`.
- `docs/superpowers/pr-reports/2026-11-04-dig-body-part-pr.md`: this report.

## Schema / bus / API changes

- Added: none (a new form of an existing CLI command, `body <part>`).
- Removed: none.
- Renamed: none.
- Behavior changed: `body <word>` shows a part when the word names a part and is not an organ id. The error for a word that is neither now also says "no part is named '<word>' either".
- Compatibility notes: organ ids behave as before. Existing organ tests pass unchanged.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed)
- skipped keys requiring operator action: none

## Tests run

```text
PYTHONPATH=. .venv/bin/python -m pytest orion/dig/tests orion/body_atlas/tests tests/test_organ_names_partition.py \
  tests/test_report_dig_kit_usage.py orion/harness/tests/test_stance_scope_hierarchy.py \
  orion/harness/tests/test_harness_prefix.py orion/physiology/tests/test_report_text.py \
  orion/curiosity/tests/test_self_inquiry_prompt_dig_body.py tests/test_no_personal_terms.py -q -p no:cacheprovider
404 passed
```

The new tests in `orion/dig/tests/test_body_part.py` (11) cover these cases:
- The flows in and out are listed, capped at 10 per direction with "+N more in the file", and every flow is in the file.
- The citable ids equal `claims.evidence_ids(...)` exactly. The ready line, with the placeholder filled in, writes a claim to a temporary ledger, and the claim cites the first printed id. Run as printed, the line is refused for its placeholder.
- With uneven weights, the heaviest flow across both directions gives the first cite id. A self loop gets its own line.
- A part that sits alone in a singleton organ names no organ id, because `body <id>` cannot open a singleton.
- An unknown part gives a clear error and writes no file, even with `/` in the name.
- Organ refs work as before, with or without `organ:`. Organ refs are matched in any letter case.
- `help` and `capacity help` point at `body <part>`. Help is 2748 chars, under the 2750 cap.

## Evals run

```text
for e in orion/body_atlas/evals/run_*.py; do python $e; done
run_capacity_roles_eval.py exit 0   (claims floor PASS; both mutations CAUGHT; claim-block tail size budget holds)
run_organ_structure_eval.py exit 0
run_store_direction_eval.py exit 0
run_wire_eval.py exit 0
python orion/physiology/evals/physiology_state_skill_eval.py   exit 0   (dig-kit CI job runs it)
```

## Docker/build/smoke checks

The smoke ran on the host from the worktree. It used the governor's FalkorDB graph, reached through `127.0.0.1:6380`. It read a copy of Orion's `self/` ledgers, and the result went to scratch space, not to Orion's dig folder.

```text
$ PYTHONPATH=$PWD ORION_DIG_HOME=<scratch> FALKORDB_URI=redis://127.0.0.1:6380 python -P -m orion.dig body orion-energy
part orion-energy (part:orion-energy): service, live, on athena · organ "cortex" organ:e7da34f2df
flows in (1), strongest first:
  <- orion-biometrics signals [orion:power:intent:settled] w=4.0931
flows out (11), strongest first:
  -> pg:energy_bill_actual writes [orion:energy:bill:actual] w=1
  -> pg:energy_bill_forecast writes [orion:energy:bill:forecast] w=1
  -> pg:energy_cost_accrued writes [orion:energy:cost:accrued] w=1
  -> pg:energy_importer_status writes [orion:energy:importer:status] w=1
  -> pg:energy_reconcile writes [orion:energy:reconcile] w=1
  -> pg:energy_run_cost writes [orion:energy:run_cost:estimated] w=1
  -> pg:energy_stakes_snapshot writes [orion:energy:stakes:snapshot] w=1
  -> pg:energy_usage_interval writes [orion:energy:usage:observed] w=1
  -> orion-cortex-orch signals [orion:system:health (census)] w=0.0037
  -> orion-equilibrium-service signals [orion:system:health (census)] w=0.0037
  +1 more in the file
my claims: metabolism claim unconfirmed
cite in a claim (by full id): orion-biometrics, orion:power:intent:settled, pg:energy_bill_actual, orion:energy:bill:actual, orion:energy:bill:forecast, pg:energy_cost_accrued, orion:energy:cost:accrued, pg:energy_importer_status +12 more in the file
  python -P -m orion.dig capacity claim orion-energy '<job>' because '<what it does>, seen on orion-biometrics'
full result: <scratch>/dig/2026-11-03/body-part-orion-energy-2.json
exit 0

$ ... python -P -m orion.dig body e7da34f2df | head -3     (organ id: unchanged)
organ organ:e7da34f2df "cortex": 70 members (38 svc, 28 store, 4 process) · on athena 27, circe 1
hub orion-cortex-exec · namespaces system/cortex/grammar · internal 629.1043 / external 569.9962
service (38), strongest first:
```

One live read takes about 27 s (the whole body is read, as with `body`). No Docker build was run. `scripts/service_rebuild_paths.yaml` maps `dig` and `body_atlas` to `orion-harness-governor`, which covers every code file this PR touches.

## Review findings fixed

The code review ran in a subagent on `git diff origin/main...HEAD`. It found no material issues and 9 minor ones or nits. 8 are fixed:

- Finding: an unknown bare name wrote a `body-<name>-N.json` file before failing, and a name with `/` crashed with a traceback.
  - Fix: the organ check runs before any write. File names are sanitized.
  - Evidence: `test_an_unknown_name_writes_no_file_even_with_a_slash`.
- Finding: the header printed a singleton organ id, which `body <id>` cannot open.
  - Fix: only a multi-part organ is named. Otherwise the header says "none (a part alone)".
  - Evidence: `test_a_part_alone_names_no_organ_id_it_cannot_open`.
- Finding: self loops showed in the out list as `-> <itself>`.
  - Fix: self loops are left out of both lists and get their own `to itself:` line.
  - Evidence: `test_strongest_flow_first_across_both_directions_and_self_loops_apart`.
- Finding: "strongest first" for cite order was per direction, so a weak in-flow could beat a strong out-flow.
  - Fix: cite order sorts in, out and self flows together by weight.
  - Evidence: same test (`cite[0] == "r2"`, the w=9 out-flow over the w=5 in-flow).
- Finding: every weight in the fixture was 1, so the ordering tests passed whatever the sort did.
  - Fix: a new `_uneven()` fixture. Under the old ordering the test fails.
  - Evidence: same test.
- Finding: the ready line placed an id inside hand-written double quotes, which is unsafe for `$` or a backtick.
  - Fix: `shlex.quote` builds the because.
  - Evidence: live output above, and the end-to-end claim test.
- Finding: the cite line sits in the uncut tail, so very long ids could push the printout past 4000 chars.
  - Fix: the cite line is bounded at 1000 chars, with "+N more in the file".
- Finding: an uppercase organ id gave a misleading "no part is named" suffix.
  - Fix: organ refs are matched in any letter case.
  - Evidence: `ORGAN:0123456789` case in the unknown-name test.
- Not fixed, accepted: `body <part>` reads Juniper's `vsm_confirmed.yaml`, the same way `capacities` does. If that file is malformed, the part view fails with that file's error. The jobs and claims lines need that file, and a loud failure matches how `capacities` behaves.

## Restart required

The dig kit is baked into the harness image. After merge, the post-pull rebuild of `orion-harness-governor` picks this up (`scripts/service_rebuild_paths.yaml`: `dig` and `body_atlas` map to `orion-harness-governor`). To run it by hand from the clean primary main checkout:

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

## Risks / concerns

- Severity: low
  - Concern: a part whose name is exactly 10 hex characters would be read as an organ id.
  - Mitigation: `part:<name>` always selects the part. No such part name exists today.
- Severity: low
  - Concern: `claims.evidence_ids` counts the other end of a flow as citable even when the flow carries only census channels (in the fixture, `watcher` may cite `gateway` because of a health heartbeat). This view prints that rule as it is and does not change it.
  - Mitigation: none in this PR. If the rule is wrong, fix it in `claims.py`, and this view will follow.
- Severity: low
  - Concern: it is UNVERIFIED whether Orion reaches for `body <part>` on the next claims turn.
  - Mitigation: `capacity help`, the `Unclaimed jobs:` header, and the `body` footer all name it.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3106

🤖 Generated with [Claude Code](https://claude.com/claude-code)
