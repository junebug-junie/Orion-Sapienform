# Hire grounding — Mind / progress disclosure into role teach

**Date:** 2026-09-19
**Status:** shipped (PR #2252); motor splice **VERIFIED** live 2026-09-20 via Soft HUD `motor_boot`
**Parent:** `docs/superpowers/specs/2026-09-15-orion-hire-determination-grounding-design.md`
**Impl parent:** PR #2238 (`feat/hire-determination-impl`)
**Live forensics:** 2026-09-19 (post #2244 / #2246 / #2247); splice soak 2026-09-20
**Follow-up:** `docs/superpowers/specs/2026-09-20-hire-handoff-and-queue-pressure-design.md` — deep strong nudge, denial handoff, budget resume, queue weather

## Arsonist summary

Orion writes `:InvestigationRole` every time and always picks `local_crawl`. The hire wire is fine; the decision is uninformed. Patch 4 shipped a soft-disclosure hook (`extra_lines` on `_role_and_help_section`) and left it empty (“Mind labels later”). Mind already produces Orion-origin work-shape fields on curiosity turns; they color stance and never reach the role teach. This patch closes that seam — still advisory, still Orion-authored — and optionally discloses hop-clock progress on resume. It does **not** auto-hire, does **not** key on attention winners, and does **not** replace self-study / self-model freshness (#2247).

## Current architecture

1. Kickoff / self-inquiry teach role + HelpRequest (`orion/curiosity/kickoff_prompt.py:_role_and_help_section`).
2. World curiosity calls that section with **no** `extra_lines`. Self-inquiry only adds “peer must not draft your answer.”
3. Unified turn runs Mind with `utterance_origin=orion` and subject-sized appraisal (`execute_unified_turn` → thought → `build_light_mind_request` → Mind stance handoff).
4. Soft labels (`expected_depth`, `cross_cutting`, `foresight_note`) land in Mind stance payload and may enter stance_react coloring; they are **not** copied into the kickoff role section.
5. After a short look, Orion may MERGE `:HelpRequest`; Hub enqueues; peer runs Cursor.

**Live (2026-09-19):**

| Check | Result |
| --- | --- |
| `:InvestigationRole` | Always `local_crawl` when present |
| `:HelpRequest` | Zero |
| Peer | Listening; zero jobs |
| Mind on curiosity | Many `session_id=orion_curiosity` runs |
| Soft labels (live Orion-origin probe) | Present; often `unknown` + a useful `foresight_note` |
| Role teach disclosure | Empty |

Adjacent merges that change the ground under this patch:

- **#2244 hop identity** — hops have `written_at`; resume preamble tells a retried sitting what it already wrote. Use for progress disclosure, do not fight.
- **#2246 stance capacity** — curiosity/reading less often fake-empty on stance; Mind/stance more likely to complete.
- **#2247 self-model freshness** — daily self-fact refresh + self-sense eval line in Hub. Parallel pipe (lived self food), **not** hire grounding. Shares `curiosity_investigation.py` — land carefully.

## Missing questions (locked for this patch)

| Question | Answer |
| --- | --- |
| Who authors role / HelpRequest? | Orion only (unchanged) |
| Auto-hire on depth=deep? | **No** |
| Keyword / feeling lists on user text? | **No** |
| Where do labels appear? | Soft lines in role teach (`extra_lines`), advisory |
| When are labels known? | At **turn** time (Mind), not at kickoff-scaffold build. Disclosure must ride the **turn prompt path** (or a mid-run revise teach), not only the frozen first kickoff bytes if those never see Mind. |
| Self-study chewing = hire ground? | **No.** Self-study answers LivedAnswer / SelfDefinition. Hire ground is Mind work-shape + progress on *this sitting*. |

## Capability change

Orion’s role/hire teach for a curiosity or self-inquiry sitting gains an advisory “what Mind just read about this subject / what you already wrote” block when available. Orion still chooses `local_crawl` or `hire_cursor` and still alone writes HelpRequest.

## Proposed data flow

```text
kickoff inventory (unchanged)
    → durable turn
    → execute_unified_turn(origin=orion, mind_appraisal_text=subject)
    → Thought: Mind enrichment → soft labels in coloring
    → NEW: format soft labels (+ optional hop progress) → extra_lines
    → role teach section includes those lines for this turn / revise
    → Orion MERGEs InvestigationRole (and HelpRequest only if hiring)
```

**Timing constraint (load-bearing):** The first durable kickoff is often frozen before Mind runs. Disclosure therefore attaches where the **motor/harness prompt for this attempt** is assembled (turn handler / resume preamble / mid-run revise), not by hoping the pre-Mind kickoff string already contained labels. Prefer:

1. Primary: inject Mind-derived `extra_lines` into the role section on the **first stance-informed turn prompt** (same place resume preamble is prepended today for `attempt > 1`).
2. Secondary: on resume / mid-run, append hop-progress lines from `#2244` clocks (`next_hop_n`, prior hop notes summary already begun in Hub).

If Mind fails open (no coloring), teach stays as today — empty disclosure is valid.

## Proposed schema / API changes

- **No new bus channel.** No new graph label.
- Optional thin helper, e.g. `format_role_teach_disclosure(mind_coloring | None, progress | None) -> list[str]`, pure, unit-tested.
- Optional: persist last Mind coloring summary on the in-memory run map next to `_mind_appraisal_by_run_id` so a later revise/resume can re-disclose without re-running Mind. **Do not** invent a new keyword taxonomy; store only allow-listed soft fields already on `ChatStanceBrief`.
- Follow-up (may ship same PR if tiny): persist `mind_appraisal_text` on `CuriosityTurnRequestV1` (known gap from parent status table) so Hub restart does not fall back to full kickoff for Mind.

## Files likely to touch

| Path | Why |
| --- | --- |
| `orion/curiosity/kickoff_prompt.py` | `extra_lines` consumers; maybe shared formatter |
| `orion/curiosity/self_inquiry_prompt.py` | Same disclosure; keep peer-must-not-draft line |
| `services/orion-hub/scripts/curiosity_investigation.py` | Wire Mind coloring → disclosure at turn/resume (collision surface with #2247 self-sense-eval line) |
| `orion/hub/turn_orchestrator.py` and/or thought reply path | Only if disclosure must be threaded through stance result back to Hub |
| `services/orion-thought/app/mind_enrichment.py` | No allow-list widen unless a field is missing; already has work-shape keys |
| Tests under hub / curiosity / thought | Fixture: coloring present → role teach contains lines; coloring absent → teach unchanged; Juniper origin still cannot force hire labels into chat |

## Non-goals

- Python auto-MERGE of `hire_cursor` or HelpRequest
- Attention-winner or thermal hire
- Keyword detectors on claim text
- Changing peer budget / Cursor invoker
- Replacing #2247 self-model / self-sense eval with hire logic
- Forcing `deep` rates up via prompt yelling

## Privacy / boundary

- Disclosure is Orion-origin curiosity/self-inquiry only (same guard as soft labels today).
- Juniper chat must not grow hire work-shape coloring (existing origin allow-list).
- Peer still read-only; self-inquiry still forbids peer-drafted LivedAnswer / SelfDefinition text.

## Trace that proves it worked

1. Unit: Mind coloring `{expected_depth, cross_cutting, foresight_note}` → role teach contains those strings in `extra_lines`.
2. Unit: no coloring → role teach identical to pre-patch (aside from unrelated resume preamble).
3. Live: one curiosity turn after deploy — Hub/Thought log or inspectable prompt slice shows disclosure lines; subsequent `:InvestigationRole.why` may cite them (soft — not required every time).
4. Live: still zero HelpRequest is **allowed**; success is “decision informed,” not “hire happened.”
5. Flag-off peer still means no Cursor job even if role says `hire_cursor`.

## Dangerous failure modes

| Risk | Mitigation |
| --- | --- |
| Disclosure steers Juniper chat | Origin guard; only curiosity/orion turn path |
| Empty theater (“depth: unknown”) every time | Still advisory; do not auto-hire; optional omit `unknown`-only lines if all three are unknown and foresight empty |
| Kickoff frozen without Mind → patch no-ops | Explicit turn/resume injection (timing constraint above) |
| Collide with #2247 in `curiosity_investigation.py` | Thin helper; touch only turn/resume disclosure call sites |
| Prompt bloat | Cap foresight to existing 240 chars; ≤5 short lines |

## Disable / rollback

- Feature flag preferred: e.g. `HUB_CURIOSITY_ROLE_TEACH_DISCLOSURE=true` (default on once shipped, or default off for one soak — **recommend default on** with flag-off escape).
- Revert the Hub wiring commit; teach templates remain valid with empty `extra_lines`.
- `HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED=false` still kills enqueue independently.

## Acceptance checks

1. Code path named: Mind coloring → `format_role_teach_disclosure` → `_role_and_help_section(..., extra_lines=...)` on the **turn** prompt path.
2. Fixture green for present/absent coloring; Juniper chat fixture unchanged.
3. Live note in PR: one post-deploy curiosity Mind run + prompt/log excerpt showing disclosure **or** explicit `UNVERIFIED` with why.
4. No new keyword cathedral; no auto-hire.
5. README blurb under curiosity hire / `stores_not_ready` sibling: role teach can show Mind work-shape when present.

## Recommended next patch

Single thin PR: disclosure helper + Hub turn/resume wiring + tests. Optionally bundle `mind_appraisal_text` persistence on `CuriosityTurnRequestV1` if it stays <~50 lines. Then implementation plan via writing-plans; **no code until Juniper approves this written spec.**

## Relationship to parent

Parent addendum required “Soft-nudge disclosure of Mind labels / progress when available.” This document is that unfinished Patch 4 remainder, grounded in 2026-09-19 live evidence that Mind labels exist and the role teach never sees them.
