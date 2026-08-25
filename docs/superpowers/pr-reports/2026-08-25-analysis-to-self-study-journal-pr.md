# Analysis → self-study journal entry, wired to autonomous dispatch

Branch: `feat/analysis-journal-action`
Status: **DONE_WITH_CONCERNS** (see Concerns — the action is not yet deployed; three services need a rebuild, listed below)

## Summary

- New autonomous action `skills.self_study.analyze.v1`: contrast one window of an already-stored telemetry source against the window immediately before it, and write **one** self-study journal entry — only when a disclosed notability rule fires.
- **One action shape, four inputs**, not four actions. The notability rules are shared and source-agnostic; the four sources differ only in which rows to read. Adding a fifth is a config entry, not a fifth copy of the analysis.
- Wired into Orion's own dispatch loop: one proposal template + one dispatch route. This is the part that was missing — the code to reflect has existed for months and was never reachable.
- Anti-spam is the design, not a guard bolted on: `skipped_not_notable` is the common, correct outcome, and a per-source cooldown caps the whole action at 16 journal entries/day (measured 3-5/day over two live replays).
- New `CortexRouteTemplateV1.skill_args` seam: static per-route verb arguments, generalising what `maintain` routes had hardcoded in `envelopes.py`. Refuses `mode`/`run_mode`/`dry_run` outright.

## Outcome moved

**Before:** `journal_entries` held **zero** rows with `source_kind` in (`self_study`, `self_reflection`). Ever. Despite `self_review` being `autonomous_invocable=True` for months (0 invocations in 72h) and `run_self_concept_reflect` being fully implemented, because **nothing** in `config/proposals/proposal_policy.v1.yaml`, `config/execution_dispatch/execution_dispatch_policy.v1.yaml`, or `orion/proposals/templates.py` referenced either one.

**After:** the first `source_kind='self_study'` entry in this database's history, written end-to-end through the real bus:

```
entry_id     d539a4c5-ddfa-4824-88c9-5d06d8e4c31c
created_at   2026-08-25 20:14:29+00
title        Self-study analysis: vision events
source_ref   vision_events:5c146b1b326796cc
```

Its body found a real 44-hour vision outage and a 5x volume collapse:

> `observation_gap`: Longest stretch with no observation was 2648 min, against a bar of 1548 min (max of the 120 min floor and 2x this producer's own 774 min baseline gap).
> `volume_shift`: 860 rows against 4119 the window before (0.21x; bar: outside 0.5x-2x).
> Checked and did not fire: producer_stalled, new_category, lost_category, mean_shift.

## Current architecture (before this patch)

| piece | state |
|---|---|
| `run_self_concept_reflect` (`services/orion-cortex-exec/app/self_study.py:2469`) | implemented, reachable only by manual verb call |
| `self_review` workflow (`orion/cognition/workflows/registry.py:141`) | `autonomous_invocable=True`, **0 invocations / 72h** |
| `publish_self_reflection_artifacts` → `journal.entry.write.v1` → `orion-sql-writer` → `journal_entries` | fully wired and live |
| proposal template / dispatch route pointing at any of it | **absent — this was the whole gap** |

Also inspected and deliberately **not** used: `reflect_self_concepts` (`self_study.py:1077`) is hardcoded prose — its "findings" are prewritten paragraphs selected by which concept kinds exist. Routing an autonomous action at it would have shipped exactly the empty-shell cognition CLAUDE.md section 0A bans.

## Architecture touched

- **New**: `orion/schemas/self_study_analysis.py`, `services/orion-cortex-exec/app/self_study_analysis.py`, `orion/cognition/verbs/skills.self_study.analyze.v1.yaml`.
- **Contract seam**: `CortexRouteTemplateV1.skill_args` + its merge in `build_cortex_request_envelope`.
- **Autonomy config**: one template in `proposal_policy.v1.yaml`, one route in `execution_dispatch_policy.v1.yaml`.
- **Reused unchanged**: the `orion:journal:write` channel, `journal.entry.write.v1`, `orion-sql-writer`'s existing mapping, and the already-valid `self_study` `JournalSourceKind`. No new channel, no new table, no new env key.

### The four inputs

All live and current, verified against Postgres on 2026-08-25:

| source | table | rows | most recent |
|---|---|---|---|
| `concept_induction` | `memory_crystallizations` | 1,282 | 2026-08-25 05:23 |
| `vision_events` | `vision_events` | 730 / 3d | 2026-08-25 19:xx |
| `affective_state` | `juniper_affective_state_log` | 1,033 | 2026-08-25 19:23 |
| `cocreation_signals` | `substrate_codebase_delta_log` | 1,536 | 2026-08-25 19:25 |

`affective_state` is named honestly in the journal body: it is message volume + swear frequency over rolling windows, a coarse agitation proxy, **not** an emotion read. The facial/vocal affect lane (`orion:affectgpt:assessment`) keeps no history to analyse — only a 1h Redis key — so it cannot be a source here.

### The rules (all shared, all source-agnostic)

`evaluate_rules()` takes two `SourceWindow`s and knows nothing about vision or affect or crystallizations.

| rule | bar |
|---|---|
| `producer_stalled` | recent window empty, baseline ≥ 5 rows |
| `observation_gap` | longest silent stretch ≥ max(120 min, **2x this producer's own baseline gap**) |
| `volume_shift` | row ratio outside 0.5x–2.0x, baseline ≥ 5 rows |
| `new_category` / `lost_category` | a label ≥ 3 rows appears/disappears, baseline ≥ 5 rows |
| `mean_shift` | \|Δmean\| ≥ 1 **baseline** sigma, both sides ≥ 5 rows |

Every bar is reported in the journal body next to the number that did or did not cross it, and the rules that did **not** fire are named — the negative space is what makes this an analysis rather than a highlight reel.

## Metric quality gate (CLAUDE.md §0A)

Recorded here rather than passed verbally. **No new metric is wired into any model.** Nothing here feeds field pressure, proposal scoring, action-value, or any cognition loop; the only output is an append-only journal entry.

1. **Provenance** — every number is a `count`, `mean`, or `max-gap` over rows a named producer already wrote. Producing functions traced to their tables; no derived quantity is invented.
2. **Independence** — n/a in the usual sense (nothing joins an existing model). The four sources are causally independent producers on independent hardware paths.
3. **Theory anchor** — the bars are not a theory of anything and are not presented as one. `GAP_MINUTES` is anchored on a real incident class (the 2026-08-21 vision outage: 21h blind, healthy container, green logs). The rest are disclosed uncalibrated starting values, stated as such in code and in the journal body.
4. **Live-data sanity** — the eval below. 3 cells fire, 13 rest, 0 unreadable. It can both speak and stay quiet. Two rules were **fixed by this check, not by review**: `observation_gap` fired on every window of a 2-rows/day producer, and `new_category` fired on everything against an empty baseline.
5. **Existing mechanism** — searched. `self_review` / `self_concept_reflect` exist; their analysis is hardcoded prose, so this does not duplicate a working mechanism, it supplies one.
6. **Reversibility** — delete the template from `proposal_policy.v1.yaml` and rebuild `orion-proposal-runtime`. No schema migration, no data loss, and the journal entries are append-only rows in an existing table.

## Files changed

- `orion/schemas/self_study_analysis.py`: result contract (`SelfStudyAnalysisResultV1`, `AnalysisMetricV1`, `AnalysisFindingV1`).
- `orion/schemas/registry.py`: registers all three; verified through `resolve()`, not by reading the dict.
- `services/orion-cortex-exec/app/self_study_analysis.py`: the analysis, the four `SourceSpec`s, the shared rules, source rotation, cooldown, journal body.
- `services/orion-cortex-exec/app/verb_adapters.py`: `SelfStudyAnalyzeVerb`.
- `orion/cognition/verbs/skills.self_study.analyze.v1.yaml`: verb contract. `services: []` is load-bearing — it is what routes through `executor.py`'s local-verb branch, the only branch that injects a live bus into `VerbContext.meta`.
- `orion/execution_dispatch/policy.py`: `CortexRouteTemplateV1.skill_args` + the validator refusing run-mode keys.
- `orion/execution_dispatch/envelopes.py`: merges route `skill_args`, derived maintenance `mode` last.
- `orion/proposals/templates.py`, `config/proposals/proposal_policy.v1.yaml`: the template; `max_candidates` 10 → 11.
- `config/execution_dispatch/execution_dispatch_policy.v1.yaml`: the route.
- `services/orion-sql-db/manual_migration_self_study_analysis_time_indexes.sql`: three additive indexes. **Not applied** — see Restart required.
- `tests/test_self_study_analysis_{rules,runner,wiring}.py`, `services/orion-cortex-exec/evals/run_self_study_analysis_eval.py`.

## Why ONE template, not four

Four near-identical templates would have competed for the same five dispatch slots to say the same thing four ways, and would have pushed four existing templates out of a 10-candidate arena. Instead the verb picks the most overdue lens itself, and `max_candidates` goes up by exactly one — which matters on a host whose Postgres I/O ceiling is a known constraint.

Checked rather than assumed: **no reserved dispatch slot was added.** The starvation fix already in `execution_dispatch_policy.v1.yaml` (`starvation_aging_bonus_per_tick: 0.002`, cap `0.25`) is explicitly general, and 0.34 + a full aging bonus clears every fresh competitor.

## Schema / bus / API changes

- **Added**: `SelfStudyAnalysisResultV1`, `AnalysisMetricV1`, `AnalysisFindingV1` (registered). `CortexRouteTemplateV1.skill_args` (defaults `{}`; every existing route unchanged).
- **Removed / renamed**: none.
- **Behavior changed**: an envelope for a route declaring `skill_args` now carries `context.skill_args`. Verified byte-identical for every existing route: a `summarize_only` route with no `skill_args` emits no key at all, and a maintenance route with none emits exactly `{"mode": ...}` as before.
- **Compatibility**: `journal.entry.write.v1` unchanged; `self_study` was already a valid `JournalSourceKind`; `orion:journal:write` already lists `orion-cortex-exec` in `producer_services`. No contract patch needed.

## Env/config changes

- Added / removed / renamed keys: **none**.
- `.env_example` updated: **not applicable** — no key changed.
- Local `.env` synced: **not applicable** — no key changed.
- Skipped keys requiring operator action: none.

Deliberate: an earlier draft added `SELF_STUDY_ANALYSIS_DATABASE_URL`. Dropped, because all three fallbacks (`SUBSTRATE_FELT_STATE_DATABASE_URL` → `ENDOGENOUS_RUNTIME_SQL_DATABASE_URL` → `POSTGRES_URI`) are already set in all four live `orion-athena-cortex-exec*` containers, confirmed with `docker exec … env`. A fourth key would be one more surface to drift. `POSTGRES_URI` is last in the chain because it points at `orion-sql-db`, a hostname this service's own `.env_example` records as historically unresolvable here.

## Tests run

```text
$ pytest tests/test_self_study_analysis_rules.py \
         tests/test_self_study_analysis_runner.py \
         tests/test_self_study_analysis_wiring.py -q
73 passed

$ pytest tests/test_execution_dispatch_{builder,envelopes,policy_loader}.py \
         tests/test_proposal_{frame_builder,policy_loader,scoring,transport_readonly_candidates}.py -q
159 passed   (with the three new files)

$ pytest services/orion-cortex-exec/tests/test_self_study_pass1.py -q         → 45 passed
$ pytest services/orion-cortex-exec/tests/test_self_study_consumer_wiring.py  → 3 passed
$ pytest services/orion-cortex-exec/tests/test_self_study_graphdb.py -q       → 7 passed

$ python scripts/check_journal_dispatch_registry.py
check_journal_dispatch_registry: OK -- all 8 trigger_kind(s) ... have a JOURNAL_DISPATCH_REGISTRY row.
```

Pre-existing and **not** caused by this branch (identical on `main`, verified): `pytest services/orion-cortex-exec/tests` as a whole directory has 14 collection errors, and `scripts/check_service_env_compose_parity.py orion-cortex-exec` fails on the compose file's `!override` YAML tag.

### Mutation testing, against the real file

Review found **13 of 30** semantic mutations surviving. All 13 now die, plus 3 written for this patch's own fixes. Verified by mutating `services/orion-cortex-exec/app/self_study_analysis.py` itself and restoring it (baseline 73 passed, restored 73 passed):

```text
M23 drop abs() on mean shift            1 failed   M02 volume ratio <= -> <     1 failed
M11 gap bar max->min                    1 failed   M05 mean sigma >= -> >       1 failed
M08 GAP_RELATIVE_MULTIPLE 2.0->1.0      1 failed   M18 until <= since -> <      1 failed
M22 drop recent-side mean guard         1 failed   M06 MIN_CATEGORY_ROWS 3->1   1 failed
M16 cooldown width x2                   2 failed   producer_stalled floor gone  1 failed
M21 read ceiling 20000->3               1 failed   selector min->max            7 failed
M15 truncated >= -> >                   1 failed   run mark not written         3 failed
```

M06 survived even the **first** closure round, because that fixture expressed the stray count as `MIN_CATEGORY_ROWS - 1` and moved with the constant. Now pinned to the literal.

## Evals run

```text
$ python services/orion-cortex-exec/evals/run_self_study_analysis_eval.py
source                  win status                      recent    base  fired
concept_induction       1.0 skipped_not_notable              0       0  -
concept_induction       6.0 skipped_not_notable              0       0  -
concept_induction      24.0 skipped_not_notable              2       0  -
concept_induction      72.0 journal_failed                   9      21  observation_gap,volume_shift,lost_category
vision_events           1.0 skipped_not_notable             19      13  -
vision_events           6.0 skipped_not_notable             88      75  -
vision_events          24.0 skipped_not_notable            190       0  -
vision_events          72.0 journal_failed                 860    4136  observation_gap,volume_shift
affective_state         1.0 skipped_not_notable              4       4  -
affective_state         6.0 skipped_not_notable             24      24  -
affective_state        24.0 skipped_not_notable             96      95  -
affective_state        72.0 skipped_not_notable            289     204  -
cocreation_signals      1.0 skipped_not_notable              5       4  -
cocreation_signals      6.0 skipped_not_notable             25      24  -
cocreation_signals     24.0 journal_failed                 102      96  new_category
cocreation_signals     72.0 skipped_not_notable            305     243  -
3 fired / 13 quiet / 0 unreadable
PASS
```

(`journal_failed` here is a **fire**, not a failure: the eval passes `bus=None` on purpose so nothing reaches the journal.) Without a DSN the eval exits `2` with `CANNOT RUN: no Postgres DSN reachable. This is not a pass.` — "cannot measure" is never reported as "measured and fine".

## Live smoke checks

```text
# 1. End-to-end through the real bus (redis://100.92.216.81:6379/0) and real Postgres
status        : journaled
fired         : ['observation_gap', 'volume_shift']
journal_write : written / append_only_by_design
entry_id      : d539a4c5-ddfa-4824-88c9-5d06d8e4c31c

# 2. Persisted by orion-sql-writer, read back from Postgres
d539a4c5-... | 2026-08-25 20:14:29+00 | orion | manual | Self-study analysis: vision events
             | self_study | vision_events:5c146b1b326796cc

# 3. Source rotation against live Redis — perfect round-robin, TTL 7d
run 0: concept_induction   run 3: cocreation_signals
run 1: vision_events       run 4: concept_induction
run 2: affective_state     run 5: vision_events

# 4. Deps already present in the running container
$ docker exec orion-athena-cortex-exec-background python -c "import sqlalchemy, psycopg2"
sqlalchemy 2.0.43 / psycopg2 ok
```

## Review findings fixed

### BLOCKER — the rotation measured journal writes, not analysis runs

- **Finding**: `select_least_recently_analysed` queried `journal_entries`, but a `skipped_not_notable` run writes nothing. A source quiet enough that no rule could fire was permanently un-journalable, permanently sorted first, and permanently won. Replayed at a 5-min cadence across the real 2026-08-23 vision outage, `concept_induction` took **560 of 576 runs (97%)** and held the selector for **474 consecutive runs (~39.5h)** — during which the vision pipeline was dead and its lens was never re-examined. "Four inputs" was decorative. It also made a false `producer_stalled` alarm load-bearing for releasing the selector.
- **Fix**: a per-source run mark in Redis (`SETEX`, 7d TTL, same shape as `orion/situational/juniper_affect_state.py`), stamped at **selection** time so a source that errors or returns nothing still rotates away. An operator-pinned `skill_args.source` does not perturb it.
- **Evidence**: same replay after the fix — `{concept_induction: 144, vision_events: 144, affective_state: 144, cocreation_signals: 144}`, longest consecutive run on one source = **1**. Confirmed live against the real Redis (smoke #3 above).

### BLOCKER — `producer_stalled` published a false claim

- **Finding**: the one rule without the `MIN_BASELINE_ROWS` floor, contradicting the module's own docstring. Against live `memory_crystallizations` (~4.6 rows/day) it fired on **72 of 288** sampled 6h windows with baselines of 1–2 rows, writing *"The producer stopped."* about a producer behaving exactly as it always does. The same disease the live smoke had already fixed for `observation_gap`, on the same source, in the same run.
- **Fix**: the same `>= MIN_BASELINE_ROWS` floor every other rule carries.
- **Evidence**: `test_producer_stalled_refuses_on_a_baseline_too_thin_to_call_it_a_stall` walks baselines 1..4 and asserts refusal, then asserts it fires at exactly 5. Mutation removing the floor now fails the suite.

### BLOCKER (found by fixing B1) — the cooldown was not a cap

- **Finding**: not in the review — it was **hidden by** the review's first blocker. The reviewer measured 4 entries/24h and correctly called the spam gate sound; that measurement was of a system where the broken selector was acting as an accidental rate limiter. With rotation fixed, the same replay over the same 24h of real rows produced **130 entries**, because the digest genuinely churns as the window slides (`cocreation_signals` alone alternated `new_category|domain:git` and `lost_category|domain:git` on a bursty producer, each variant carrying its own independent cooldown).
- **Fix**: the cooldown is now per **source**, not per finding-set, with a 6h floor independent of `window_hours` (without which `window_hours=0.5` would allow 48 entries/source/day). Hard ceiling 16/day.
- **Evidence**: both replays re-run with the written entries fed back into the lookup — **6 entries over 48h (3.0/day)** and **5 over 24h (5.0/day)**.

### SHOULD — config could smuggle a run mode past the dry-run gate

- **Finding**: `envelopes.py` forces the derived `mode` only for `MAINTENANCE_SCOPE`, while `allowed_scope` and `cortex_verb` are independent unvalidated config fields. A route declaring a mutating verb under `summarize_only` (which `builder.py::scope_allowed` admits **unconditionally**, without `allow_mutating_dispatch`) plus `skill_args: {mode: execute}` would have reached that verb with `mode=execute` regardless of `dry_run`. Unreachable before this patch, because route `skill_args` did not exist.
- **Fix**: `CortexRouteTemplateV1` refuses `mode` / `run_mode` / `dry_run` at the schema, case-insensitively. Refused, not overridden.
- **Evidence**: `test_config_cannot_declare_a_run_mode_at_all`.

### SHOULD — the index migration was written but not committed

- **Finding**: the file existed in the worktree, three minutes after the commit, and was not in it.
- **Fix**: committed. Its scope also shrank: the selector's `journal_entries` scan (4,306 buffers, 21.9 ms, **zero rows returned**, no index on `source_kind`) is gone entirely rather than indexed around, since rotation state belongs in a short-lived Redis mark and not in a 35k-row append-only table.
- **Evidence**: `git show --stat` includes it; the remaining `recently_journaled` lookup plans as `Index Scan using idx_journal_entries_created_at`, 529 buffers, 1.9 ms.

### SHOULD — 13 of 30 mutations survived

- **Fix + evidence**: see Tests run above. All 13 die; 3 more added for this patch's own fixes.

### NICE (all four fixed)

- Two docstring references to a `NOTABILITY_RULES` symbol that does not exist, and to a route-pinned `skill_args.source` the live route deliberately leaves unset.
- A truncated **baseline** window is now disclosed in the journal body, naming which numbers it skews.
- The `base_priority` comment no longer implies that sharing a base predicts sharing a dispatch rate (0.34 gets 65 and 2,316 dispatches/24h on two existing templates). It now records the fact that actually transfers: this template's whole gate-relevant tuple is identical to three templates dispatching 2,316–2,864 times a day.

## Restart required

`config/proposals/*.yaml` and `config/execution_dispatch/*.yaml` are **baked into the images** (`/app/config/...`), not volume-mounted — confirmed: neither compose file has a `volumes:` entry for them. So all three services need a **rebuild**, not a restart.

**Not deployed by this branch.** The image tag and container name key off the service directory basename only, so a build from a worktree replaces the live image tag and the next restart of the stack would silently pick up unmerged code (`docs/superpowers/pr-reports/2026-07-14-agent-git-safety-hooks-pr.md`). Deploy after merge, from a worktree of merged `main`:

```bash
# 1. cortex-exec — the verb itself (all four lanes share one image)
scripts/safe_docker_build.sh orion-cortex-exec up -d --build

# 2. proposal-runtime — the new template + max_candidates 10 -> 11
scripts/safe_docker_build.sh orion-proposal-runtime up -d --build

# 3. execution-dispatch-runtime — the new route
scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d --build
```

Optional, and a **production write** so it is deliberately not run here — three additive `CREATE INDEX CONCURRENTLY` statements, ~40k buffer reads/hour of sequential scanning avoided. Must run outside a transaction block:

```bash
psql -h localhost -p 55432 -U postgres -d conjourney \
  -f services/orion-sql-db/manual_migration_self_study_analysis_time_indexes.sql
```

Post-deploy check — the action should appear within minutes, and quiet runs are the expected majority:

```bash
docker logs orion-athena-cortex-exec-background --tail 200 | grep -i self_study
psql ... -c "SELECT created_at, title, source_ref FROM journal_entries
             WHERE source_kind='self_study' ORDER BY created_at DESC LIMIT 10;"
redis-cli -h 100.92.216.81 keys 'orion:self_study:last_run:*'
```

## Risks / concerns

- **Severity: medium — not deployed, so the dispatch path is UNVERIFIED end-to-end.** The verb, the journal write, the SQL persistence and the rotation are all live-verified in-process against the real bus, real Redis and real Postgres. What is proven only by traced code and tests is the last hop: that the dispatch loop actually selects this template and calls the verb. Mitigation: `test_self_study_analysis_wiring.py` asserts every joint (template → candidate → route → verb yaml → registered verb), and review independently traced `build_proposal_frame` → `build_policy_decision_frame` → `build_execution_dispatch_frame` against the real config files. Confirm with the post-deploy check above.
- **Severity: low — `concept_induction` is the thinnest source.** ~4.6 rows/day means most rules cannot fire on it at a 6h window, and it will usually be a quiet run. That is correct behaviour, not a defect, but it does mean one of the four lenses contributes little until crystallization volume rises.
- **Severity: low — bursty producers can produce alternating, mutually contradictory readings.** `cocreation_signals` flips between `new_category|domain:git` and `lost_category|domain:git` as a bursty domain enters and leaves the window. Both readings are literally true and the counts are stated honestly, but neither is informative. The per-source cooldown now caps the damage at one entry per 6h; a burstiness-aware category rule is the real fix and is deliberately deferred rather than guessed at.
- **Severity: low — `observation_gap` is inert for producers whose baseline gap exceeds half the window.** A consequence of the relative bar, and the intended direction (do not cry wolf on slow producers), but worth knowing rather than discovering.

## PR link

<to be filled after push>
