# PR Report: Full revamp of root README.md

**PR:** [#1437](https://github.com/junebug-junie/Orion-Sapienform/pull/1437)
**Branch:** `docs/readme-full-revamp`
**Date:** 2026-07-28

## Summary

- Root `README.md` had accreted into ~3 overlapping drafts stapled together across a long
  history of incremental "Update README.md" commits: Spark was explained in full 3 times,
  dreams/collapse-mirrors/social-memory 2-3 times each, and a "References & Conceptual
  Anchors" + "Get Involved" pair was literally duplicated verbatim at the very end.
- It also described architecture that no longer exists: RDF/Fuseki referenced ~30 times,
  even though `orion-rdf-writer`/`orion-rdf-store` were deleted and `orion:rdf:enqueue`
  retired the same day this work started.
- It never mentioned the four biggest architectural shifts of the last ~4 months: the
  `orion-unified` turn pipeline, the FCC motor, the Sentience Striving Program (which
  formally superseded the old six-drive taxonomy), and the `orion-substrate-runtime`
  layered active-inference pipeline.
- Merged every duplicated section into one telling, replaced RDF with FalkorDB throughout,
  added real sections for the four new primitives grounded in actual code paths, brought
  the service inventory and hardware GPU specs current, removed the duplicate blocks.
- Net: 2595 → 1071 lines (59% cut), zero duplicate headers remaining.

## Outcome moved

The README now reflects the mesh that's actually running instead of a pre-cutover snapshot
with three copies of the same architecture narrative pasted on top of each other.

## Current architecture

Root `README.md` only — no code touched. The prior version was effectively 3 unmerged
drafts of the same narrative, never consolidated across years of incremental edits.

## Architecture touched

Documentation only. No services, schemas, bus channels, or config surfaces changed.

## Files changed

- `README.md`: full rewrite/merge (see summary above).

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: none
- Compatibility notes: n/a — documentation only

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: n/a
- local `.env` synced: n/a
- skipped keys requiring operator action: none

## Tests run

```text
N/A — documentation-only change, no code paths touched.
```

## Evals run

```text
N/A
```

## Docker/build/smoke checks

```text
N/A
```

## Review findings fixed

- Finding: `transport` listed as a live active-inference domain alongside its own successor
  `bus-synaptic` (`transport_prediction_error` was retired 2026-07-26 per
  `ACTIVE_INFERENCE_DOMAINS`).
  - Fix: corrected to the real 5-domain list and named the retirement explicitly rather than
    silently dropping the mention.
  - Evidence: `orion/substrate/attention_self_model.py:90` (`ACTIVE_INFERENCE_DOMAINS`),
    `orion/substrate/prediction_error.py:175-195` (retirement docstring),
    `services/orion-substrate-runtime/app/worker.py:2240-2296` (`_transport_tick()` now
    calls `bus_synaptic_prediction_error`, not the retired function).
- Finding: bus channel count stated as "~265" vs. the real 253 in `channels.yaml`.
  - Fix: corrected to "~250".
  - Evidence: reviewer parsed `orion/bus/channels.yaml` directly with PyYAML.
- Finding: dangling reference to `HARDWARE.md`, which doesn't exist anywhere in the repo
  (inherited from the old README, not introduced by this revamp).
  - Fix: removed the reference rather than carrying stale debt forward.
  - Evidence: `git ls-files | grep -i hardware.md` empty in both the main checkout and this
    worktree.

Everything else the review checked passed live verification against current main: all 77
services in the new service-inventory section diffed 1:1 against
`git ls-tree -d --name-only HEAD services/`; `ORION_UNIFIED_TURN_ENABLED=true` confirmed as
the real default in `.env_example`/`.env`; the unified-turn sunset checklist confirmed to
have zero of 8 boxes checked; FCC port/proxy claims confirmed against
`.env_example`/`docker-compose.yml`; `turn_orchestrator.py`'s `orion.fcc.context_budget`
import confirmed; the Sentience Striving Program's PR #1156 and 84,511-tick finding
confirmed via `git log`; substrate-runtime layer numbering (5/7/8/9/10/11, Layer 6
deliberately skipped) matched the canonical layer doc; the FCC/cortex GWT-dispatch design
correctly described as design-stage rather than overclaimed as live; all retired-service
mentions (landing-pad, spark-introspector, rdf-writer, rdf-store) confirmed as real
retirements with their commits present as ancestors of current HEAD.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
- Concern: `graphify-out/` graph update hit the known, previously-documented destructive
  node-loss bug during `scripts/safe_graphify_update.sh` (node count would have dropped
  ~92%). The wrapper caught it and auto-restored `graph.json`/`manifest.json` as designed;
  `graphify-out/` stays stale for this patch. Per existing project guidance
  (`docs/superpowers/pr-reports/` — see memory of the 2026-07-14 incident and its many
  documented recurrences), this is expected wrapper behavior and not something this PR's
  scope should attempt to root-cause.
- Mitigation: none needed for this PR; nothing from the failed update run was committed.
- Severity: low
- Concern: a concurrent session merged an unrelated PR (#1436, the shared-checkout edit
  guard hook) into `origin/main` while this work was in progress. This branch was rebased
  onto the updated `origin/main` cleanly (no conflicts) before opening the PR.
- Mitigation: none needed; confirmed via `git merge-base --is-ancestor` and a clean rebase.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1437
