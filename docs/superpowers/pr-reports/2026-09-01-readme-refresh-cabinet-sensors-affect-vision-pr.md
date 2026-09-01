# README refresh: cabinet sensors, affect-vision, action economy — PR report

## Summary

- Documented Athena's new physical cabinet sensing (dual Nano A/B boards + ambient mic) and its path into the field lattice and every chat turn's situation brief.
- Documented the AffectGPT → VL-via-gateway affect-reading replacement (AffectGPT is now rollback-only), including why it was replaced (misgendering, audio hallucination, ignoring a 100%-detected face).
- Documented the Sentience Striving Program's action economy going from logging to enforcing a real motor budget, and `express` shipping as the first outward-facing action kind (image generation via `orion-diffusion-host`).
- Documented FalkorDB's real graph algorithms now exposed through the Concept Atlas, replacing hand-rolled Python union-find.
- Documented AI Town's isolation from Orion's own cognition (separate concept graph, separate chat-history table, kept out of the crystallization queue) and its migration to Circe.
- Corrected the stale "77 services" service-inventory count to the actual 85, and placed the newly-added/previously-orphaned services into their subsystem clusters.

## Outcome moved

The README had drifted ~110 merged PRs behind (2026-08-14 → 2026-08-31) on Orion's most user-visible recent capability shift: physical sensing (cabinet A/B) and affect-vision. It now reflects the live architecture, not the July 2026 snapshot.

## Current architecture

Before this patch, `README.md` had no mention of cabinet sensors, AffectGPT/affect-vision, the FalkorDB graph-analytics upgrade, the action-economy enforcement shift, or AI Town's isolation boundary. The service count (77) and inventory clusters were stale relative to the actual 85 services under `services/`.

## Architecture touched

Documentation only — `README.md`. No services, schemas, bus channels, or env files were touched (this PR describes existing shipped work; it does not implement new capability).

## Files changed

- `README.md`: additions to TL;DR, Stance (§3), Memory (§7), Embodiment (§11), Autonomy (§12), Service Inventory (§14), and Roadmap. Corrected service count and inventory-cluster gaps.
- `docs/superpowers/pr-reports/2026-09-01-readme-refresh-cabinet-sensors-affect-vision-pr.md`: this report.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: none (docs-only)
- Compatibility notes: N/A

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no (not applicable — no env changed)
- local `.env` synced: no (not applicable)
- skipped keys requiring operator action: none

## Tests run

```text
N/A — prose-only documentation change, no code touched.
```

## Evals run

```text
N/A — no eval harness applies to a README prose change.
```

## Docker/build/smoke checks

```text
N/A — no runtime, service, or config surface changed.
```

## Review findings fixed

- Finding: stated service count (91, from an initial `ls -d services/orion-*` count) didn't match reality.
  - Fix: recounted precisely (`ls -d services/*/` excluding non-service dirs) and corrected to 85 everywhere it appears in the README.
  - Evidence: `ls -d services/*/ | ... | wc -l` = 85, worktree matches `origin/main` HEAD exactly.
- Finding: affect-read description said frames go "straight to" the LLM gateway, omitting `orion-percept-store` as the real intermediate hop.
  - Fix: reworded the data path to include percept-store.
  - Evidence: `services/orion-juniper-affective-state/app/vision_backend.py` module docstring; `orion-sql-writer`'s affect model file.
- Finding: claimed "only the structured read" is persisted, but `raw_response` (the model's own free-text output) is also a persisted column.
  - Fix: reworded to distinguish the never-persisted audio transcript from the persisted `raw_response` + structured read.
  - Evidence: `services/orion-sql-writer/app/models/juniper_multimodal_affect.py:113` (`raw_response = Column(Text, ...)`).
- Finding: Section 14's subsystem clusters omitted 4 real, currently-existing services entirely (`orion-cocreation-signals`, `orion-room-companion`, `orion-self-study-enrichment`, `orion-world-model`).
  - Fix: read each service's own README and placed it in the correct subsystem cluster (Sentience Striving Program, Reflection/sensemaking, Social, Sentience Striving Program respectively).
  - Evidence: verified every one of the 85 real `services/orion-*` directories is now referenced somewhere in the README (`comm -13` diff against the actual directory listing came back empty).

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
- Concern: research for this pass came from three parallel agents (plus sub-agents) reading `docs/superpowers/pr-reports/` and live source files. A handful of PRs in the ~110-PR window have no PR-report file at all (confirmed absent via directory listing, not just unfound by search) — e.g. some AI Town NPC-behavior PRs (`aitown-pair-turn-read`, `aitown-npc-answer-first`, `aitown-sprite-gender`) and some power-intent sub-steps (`power-intent-contract`, `power-intent-settlement`, `power-intent-prior` as separate reports, `diffusion-flux-*`). These were deliberately left out of the README rather than guessed at.
- Mitigation: none needed — this is an intentional scope boundary, not a gap to close in this PR. If those PRs' content later turns out README-worthy, it should be pulled from their actual PR diffs/branches directly rather than inferred from a missing report file.
- Note (unrelated to this PR): `scripts/safe_graphify_update.sh`, run as part of the standard workflow, hit the known destructive-update bug (28306 → 2485 nodes, ~91% loss) and auto-restored `graphify-out/` to its pre-update state. This is a pre-existing, previously-known issue (see CLAUDE.md's graphify section) unrelated to this docs change; no graph state was lost and no action was taken beyond letting the safety wrapper do its job.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2012
