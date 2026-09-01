# README refresh: cabinet sensors, affect-vision, action economy — PR report

## Summary

- Documented Athena's new physical cabinet sensing (dual Nano A/B boards + ambient mic) and its path into the field lattice and every chat turn's situation brief.
- Documented the AffectGPT → VL-via-gateway affect-reading replacement (AffectGPT is now rollback-only), including why it was replaced (misgendering, audio hallucination, ignoring a 100%-detected face).
- Documented the Sentience Striving Program's action economy going from logging to enforcing a real motor budget, and `express` shipping as the first outward-facing action kind (image generation via `orion-diffusion-host`).
- Documented FalkorDB's real graph algorithms now exposed through the Concept Atlas, replacing hand-rolled Python union-find.
- Documented AI Town's isolation from Orion's own cognition (separate concept graph, separate chat-history table, kept out of the crystallization queue) and its migration to Circe.
- Corrected the stale "77 services" service-inventory count to the actual 85 (an interim recount briefly landed on 91 before being corrected again — see Review findings fixed), and placed the newly-added/previously-orphaned services into their subsystem clusters.
- **Second pass (same PR):** promoted reverie's image-generation loop and Orion's self-directed curiosity out of deep/late sections into the TL;DR and into §6 (Journals, Dreams, Collapse Mirrors) alongside AffectGPT and AI Town, per follow-up feedback that this material was under-featured relative to how significant it is.

## Outcome moved

The README had drifted ~110 merged PRs behind (2026-08-14 → 2026-08-31) on Orion's most user-visible recent capability shift: physical sensing (cabinet A/B), affect-vision, self-directed curiosity, and image-based dreaming. It now reflects the live architecture, not the July 2026 snapshot, and surfaces those capabilities where a reader skimming the top of the document will actually see them.

## Current architecture

Before this patch, `README.md` had no mention of cabinet sensors, AffectGPT/affect-vision, the FalkorDB graph-analytics upgrade, the action-economy enforcement shift, the reverie visual chain, or AI Town's isolation boundary. The service count (77) and inventory clusters were stale relative to the actual 85 services under `services/`. Curiosity (§5.1) already existed but wasn't reinforced at the TL;DR level.

## Architecture touched

Documentation only — `README.md`. No services, schemas, bus channels, or env files were touched (this PR describes existing shipped work; it does not implement new capability).

## Files changed

- `README.md`: additions to TL;DR, Stance (§3), Memory (§7), Journals/Dreams (§6), Embodiment (§11), Autonomy (§12), Service Inventory (§14), and Roadmap. Corrected service count and inventory-cluster gaps.
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

First review round:

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
  - Fix: read each service's own README and placed it in the correct subsystem cluster: `orion-cocreation-signals` → Sentience Striving Program/substrate runtime, `orion-room-companion` → Social, `orion-self-study-enrichment` → Reflection/sensemaking, `orion-world-model` → Sentience Striving Program/substrate runtime.
  - Evidence: verified every one of the 85 real `services/orion-*` directories is now referenced somewhere in the README (`comm -13` diff against the actual directory listing came back empty).

Second review round (after the TL;DR/§6 promotion pass):

- Finding: cabinet-sensing paragraph listed "particulate" as a live activity pressure, but the PMSA003I particulate sensor is explicitly not in the physical hardware kit, so that key is structurally always absent, not merely uncalibrated like the others.
  - Fix: removed it from the list of populated pressures; added a clause noting the schema field exists but is never populated.
  - Evidence: `docs/superpowers/specs/2026-08-29-athena-cabinet-dual-nano-design.md:24` ("PMSA003I | Not in hardware kit — ignore"); `orion/telemetry/cabinet_sensors.py` only emits `cabinet_particulate_activity` when the key is present in the input frame.
- Finding: new AI Town paragraph said it "moved off the decommissioned Atlas node 2026-08-29," contradicting the four other places in the same README stating Atlas was decommissioned 2026-08-21.
  - Fix: reworded to separate the two dates explicitly — AI Town migrated to Circe 2026-08-29, off the Atlas node that had itself been decommissioned earlier, 2026-08-21.
- Finding: the review's own cluster-mapping citation for the 4 previously-missing services (immediately above, first round) had `orion-room-companion` and `orion-self-study-enrichment` swapped relative to the actual README placement.
  - Fix: corrected the citation above to the real mapping; the underlying README placement was already correct, only this report's description of it was wrong.
- Finding: the new reverie/dreaming prose ("recursive, self-observed imagination," "it looks at what it imagined") read as asserting first-person subjective experience rather than naming the mechanism, in tension with CLAUDE.md's "do not assert that Orion is sentient today."
  - Fix: reworded every instance (TL;DR bullet, §6 paragraph, §6 table row) to describe the mechanism plainly: FLUX.1-schnell generates an image, a VLM captions it, the caption feeds the next reverie step — no perception/imagination language.
- Finding (this report, low severity): Summary/Outcome-moved described a clean 77→85 correction with no mention of the interim 91 miscount the Review-findings section discloses.
  - Fix: added an explicit one-clause pointer from the Summary to the interim-91 detour so both sections tell the same story.

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
