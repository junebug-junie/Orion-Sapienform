## Summary

- After a `git pull`, a change to shared code under `orion/` now rebuilds the services that actually use it. Before, only `orion/curiosity/` did this; every other shared package, plus all of `orion/schemas/` and `orion/bus/`, rebuilt nothing.
- "Uses it" is read from import statements at module level, not package level: a service rebuilds when its runtime code imports the changed module, or imports another module of the same `orion/<pkg>/` that reaches it, or imports a module of a package one hop beyond that which reaches it.
- The schema registry is the one hub we do not walk through. It imports every registered schema and is imported by the bus core, so walking through it would rebuild the whole fleet for any schema edit. Services that import the registry directly still rebuild.
- Non-Python files (verb YAML, prompt templates) map to the package that reads them. `orion/recall/profiles/` and the three `orion/core/storage/sql/*.sql` files get explicit reader entries (modules or services). Any other data file under `orion/core/` is skipped with a reason, because the whole-package fallback there includes the bus core and would rebuild 67 services. Only a non-Python file under `orion/bus/` or `orion/schemas/` with no known reader falls back to "every image that copies orion", and says so on stderr. `orion/bus/channels.yaml` stays skipped (see Risks).
- Service `tests/` and `evals/` folders no longer count as importers. Under `orion/`, `tests/`, `test_*.py`, `conftest.py`, `*.md`, `Makefile`, and non-Python files under `tests/` or `evals/` never trigger a rebuild.

## Outcome moved

PR #2684 changed only `orion/situational/perception_reader.py` and `orion/schemas/situation.py`, and the post-pull deploy rebuilt nothing. Now:

```text
$ ORION_MESH_HOST=athena bash scripts/rebuild_services_from_git_diff.sh --dry-run --base 46e092b93
[dry-run] would rebuild: orion-cortex-exec
[dry-run] would rebuild: orion-hub
[dry-run] would rebuild: orion-cortex-orch      # added by the existing exec->orch coupling
```

Blast radius on the last 10 merges to `origin/main` (host `athena`; "Before" = origin/main's classifier, "After" = this branch; the wrapper adds `orion-cortex-orch` whenever `orion-cortex-exec` is listed):

| Merge | PR | Files | Before | After | Services after |
|---|---|---|---|---|---|
| 1b2cea7e6 | #2684 | 6 | 0 | 2 | cortex-exec hub |
| 46e092b93 | #2683 | 42 | 6 | 11 | hub sql-db sql-writer thought vision-council vision-edge vision-frame-router vision-host vision-retina vision-scribe vision-window |
| acb3c7803 | (main sync) | 12 | 1 | 2 | harness-governor hub |
| 8a3b3385e | #2682 | 42 | 27 | 18 | actions attention-runtime cortex-exec cortex-orch dream durable-runs embodiment equilibrium-service execution-dispatch-runtime field-digester harness-governor hub llm-gateway memory-consolidation proposal-runtime self-experiments substrate-runtime thought |
| d0b975f70 | #2681 | 7 | 1 | 2 | harness-governor hub |
| 8c54869e2 | #2680 | 5 | 0 | 0 |  |
| b3a5a6555 | #2679 | 1 | 0 | 0 |  |
| 44f333c40 | #2678 | 5 | 2 | 2 | durable-runs hub |
| bfb966b46 | #2677 | 35 | 13 | 14 | attention-runtime durable-runs equilibrium-service field-digester hub proposal-runtime sql-writer substrate-runtime thought vision-council vision-edge vision-frame-router vision-host vision-retina |
| 597aa6dca | (main sync) | 14 | 2 | 23 | actions attention-runtime consolidation-runtime cortex-exec cortex-orch dream durable-runs equilibrium-service execution-dispatch-runtime feedback-runtime field-digester graph-compression harness-governor hub memory-consolidation policy-runtime proposal-runtime recall spark-concept-induction sql-writer substrate-runtime thought world-pulse |

How the walk depth was chosen (same 10 merges, counting only `orion/**.py` changes, before the host exclude list):

| Policy | Services per merge |
|---|---|
| Unbounded import walk | 84, 84, 2, 84, 2, 0, 0, 2, 84, 84 |
| Own package + 1 package hop, no registry barrier | 59, 59, 2, 61, 2, 0, 0, 2, 61, 63 |
| Plain 1 module hop | 2, 11, 2, 9, 2, 0, 0, 2, 5, 7 |
| Own package + 1 hop (hop not closed inside its package) + registry barrier (first review round) | 2, 11, 2, 9, 2, 0, 0, 2, 5, 15 |
| **Own package + 1 package hop, closed inside the package it lands in + registry barrier (shipped)** | 2, 11, 2, 9, 2, 0, 0, 2, 5, 24 |

Plain one module hop was the first cut, but it missed a real consumer: `orion-harness-governor` runs `orion.curiosity.worldview` through `orion.harness.fcc_motor -> orion.curiosity.write_stamp -> orion.curiosity.atlas -> worldview` (three hops, two of them inside `curiosity`). The old curiosity-only rule caught that case; the shipped policy still does (`test_orion_curiosity_import_closure`).

The #2682 row went down from 27 to 18. The old rule rebuilt every importer of anything in `orion/curiosity/` (plus one hop at package level) for any curiosity edit; the new one only follows the modules that changed.

## Current architecture

`scripts/git_hooks/post-merge` (opt-in) runs `scripts/rebuild_services_from_git_diff.sh`, which asks `scripts/rebuild_affected_services.py --plan` for affected services. For `orion/` paths the classifier used an index keyed by top-level package (`orion.<pkg>` -> importers), gated by `orion_import_packages: [curiosity]`. `orion/bus/` and `orion/schemas/` were in `skip_prefixes`, so even the "contract path -> all COPY-orion services" rule underneath them never ran.

Why it was narrowed (commit `cbfd353a9`, 2026-08-30): `orion/bus/channels.yaml` was broadcasting a rebuild to all 60+ COPY-orion services, and package-level fan-out restarted the mesh on unrelated shared-lib edits. Both reasons are kept: `channels.yaml` is still never auto-rebuilt, and the module-level graph plus the registry barrier is what keeps schema edits from reaching the fleet.

## Architecture touched

- Classifier only (`scripts/rebuild_affected_services.py` + its YAML). The wrapper script, post-merge hook, and every PR #2680 feature (baked/mounted dependency sources, `--force-recreate`, running profile services, analytics profile) are unchanged; their tests pass untouched.
- New module-level import graph (`build_import_graph`, `module_service_consumers`). Regex import extraction handles `import a, b as c`, `from x import (a,\n b)`, and relative imports. The `ast` version gave identical results on all 10 merges but took ~7s per run; regex takes ~1.3s.
- Path normalization: `raw_path.lstrip("./")` stripped characters, not a prefix, so `.github/...` became `github/...` and `.env` became `env`. Now it strips a literal `./` prefix.
- If `service_rebuild_paths.yaml` exists but PyYAML is missing, the classifier now exits 2 instead of silently dropping every skip rule. Without the mapping, `channels.yaml` would fall back to a fleet rebuild.

## Files changed

- `scripts/rebuild_affected_services.py`: module-level import graph, own-package closure + package hops + barriers, data-file readers, contract fallback with stderr reason, `fleet_wide_paths`, test/evals/markdown skips, path normalization fix, PyYAML guard.
- `scripts/service_rebuild_paths.yaml`: removed `orion_import_packages`; removed `orion/bus/` and `orion/schemas/` from `skip_prefixes`; `contract_paths` made explicit; added `fleet_wide_paths`, `orion_import_hops`, `orion_import_barriers`, `orion_data_readers`.
- `tests/scripts/test_rebuild_affected_services.py`: #2684 regression; tests-only and markdown changes select nothing; schema module selects only its importers (not two hops, not a sibling schema's importers, not evals/tests-only importers); relative imports; package `__init__`; own-package closure + registry barrier; data-file mapping; contract fallback; recall profiles; dot-prefixed paths. Replaced the test that asserted the old allowlist.
- `docs/superpowers/pr-reports/2026-10-15-rebuild-on-shared-orion-code-pr.md`: this report.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: post-pull auto-deploy now rebuilds the importers of changed `orion/` code (table above).
- Compatibility notes: `--list-only`, `--plan` and `--json` keep their output shape. YAML key `orion_import_packages` is gone and no longer read. To stop a package from ever auto-rebuilding, add `orion/<pkg>/` to `skip_prefixes`.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed)
- skipped keys requiring operator action: none

## Tests run

```text
pytest tests/scripts/test_rebuild_affected_services.py tests/scripts/test_rebuild_services_from_git_diff.py -q
  -> all passed
pytest tests/scripts -q
  -> 516 passed, 2 failed: test_eval_vision_organ_replay.py (known order-dependent; passes alone: 2 passed)
All 35 run steps of .github/workflows/orion-static-gates.yml, run locally -> 35 PASS
```

## Evals run

```text
No eval harness for the deploy scripts. The blast-radius replay over the last 10 merges (table above) is the behavioral check.
```

## Docker/build/smoke checks

```text
ORION_MESH_HOST=athena bash scripts/rebuild_services_from_git_diff.sh --dry-run --base 46e092b93
  -> would rebuild orion-cortex-exec, orion-hub, orion-cortex-orch
No docker build or deploy run (by instruction).
```

## Review findings fixed

Fresh-context review subagent (read-only) on 2f7b6cd00:

- Finding (must-fix): the hop into another package only took direct importers, so a sibling module in that package was missed. Example: `orion/core/activation_decay.py` selected only hub, but cortex-exec runs it through `substrate.falkor_anchor_store -> falkor_store -> activation`. About 19% of orion modules were missing consumers this way.
  - Fix: each hop is now closed inside the package it lands in, the same way the changed module's own package is. The reason string, docstring and yaml comment describe this.
  - Evidence: `test_hop_is_closed_inside_the_package_it_lands_in` and `test_real_repo_hop_closure_reaches_cortex_exec`. The blast table was re-run; the max is 23, against a fleet storm of ~60.
- Finding (should-fix): a `.sql` file under `orion/core/storage/sql/` fell back to all of `orion.core` and rebuilt 67 services.
  - Fix: explicit `orion_data_readers` entries for the three files. Entries can now name services directly. A new `orion_data_no_fallback: [orion/core/]` skips (with a reason) any other data file there.
  - Evidence: `test_data_reader_entries_can_name_services` and `test_core_data_file_without_reader_entry_does_not_storm`.
- Finding (should-fix): eval fixtures under `orion/**/evals/` and `orion/cognition/Makefile` triggered whole-package rebuilds of 12-17 services.
  - Fix: non-Python files under `tests/` or `evals/`, and Makefiles, are skipped.
  - Evidence: `test_orion_eval_fixtures_and_makefiles_select_nothing`.
- Finding (should-fix): the PyYAML guard had no test.
  - Fix: added `test_missing_pyyaml_exits_2_instead_of_dropping_skip_rules`.
- Finding (should-fix): the mini-repo tests silently depend on the real yaml.
  - Fix: this is now documented in the fixture docstring (the mapping path is fixed by design).
- Finding (nit): `orion_import_hops:` with an empty value raised `TypeError`.
  - Fix: an explicit None check. Not `or 1`, which would eat a configured 0.
- Finding (nit): `orion/__init__.py` runs in every service, so editing it rebuilds the fleet. This was undocumented.
  - Fix: one line added to the yaml header.
- Not fixed (nit): regex blind spots (`import(b)` with no space, `;`-joined imports, backslash continuations, `)` inside a comment). The reviewer found 0 occurrences outside tests.

## Restart required

```text
No restart required. This takes effect on the next `git pull` on the primary checkout with post-merge rebuild enabled.
```

## Risks / concerns

- Severity: medium
  - Concern: bigger rebuilds after pulls that touch shared code (up to 23 services on the last 10 merges, vs 0-27 before; the 23 is a `orion/schemas/harness_finalize.py` edit). That is the requested behavior, but on a busy day it means more container restarts.
  - Mitigation: module-level resolution, the registry barrier, and the host allow/exclude lists all still apply. To quiet a package, add it to `skip_prefixes`.
- Severity: medium
  - Concern: `orion/bus/channels.yaml` is still never auto-rebuilt, even though every bus service reads it at runtime.
  - Mitigation: deliberate (commit `cbfd353a9`). Channel edits almost always ship with the producer or consumer service change, and that service rebuilds through its own `services/` path.
- Severity: low
  - Concern: the import graph cannot see dynamic imports, or data files read from another package without an `orion_data_readers` entry. It also does not follow two package hops away, or through the registry. A consumer reached only that way will not rebuild.
  - Mitigation: the reason string names the rule that fired. Add an `orion_data_readers` or `orion_package_extra_services` entry when one is found.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
