# Door B endogenous outreach synthesis — design

**Date:** 2026-09-19  
**Status:** approved  
**Scope:** Endogenous tick compose (`build_outreach_prompt`) only. Door A (curiosity investigation → `build_outreach_composition_prompt`) already shipped (#2237).  

## Problem

Door B is what actually reaches Juniper most days. It peeks open priors + short curiosity evidence summaries (+ optional tension/daydream/camera), then instructs: “Say one thing… in your own voice” and “Speaking with feeling… is always fine.” Recent Orion-only outreaches are labeled “the last thing the two of you said” and used for tone. Result: atmospheric mush that does not land the content that justified the interrupt, and burns the shared daily cap before Door A can speak.

Novelty (#2251) stops *repeating the same IDs*. It does not make the message be *about* the unused content.

## Goal

Same contract as Door A, adapted to what Door B actually has:

1. **Thinking thread** — synthesize from the talkable lanes that made this tick fire (open priors, curiosity summaries, and daydream/tension facts when present) into one clear thread.  
2. **Why share** — why bringing that thread to Juniper now.

Not free-float poetry. Not hop-by-hop recap of investigation (Door B does not own a full hop list unless we later enrich). Exact `PASS` stays.

## Choke point

`services/orion-hub/scripts/endogenous_outreach.py` → `build_outreach_prompt`

## Changes (thin)

1. **Rewrite closing instructions** to require both layers when talkable content is present (`open_prior_previews` / `curiosity_summaries` / `daydream`).  
2. **Recent turns:** if every turn is Orion (or prompt-empty outreach history), label as Orion’s recent unprompted notes — not “the two of you.” Do not describe that block as tone fuel for poetry.  
3. **Structural tests** on `build_outreach_prompt` output (both layers present; Orion-only history label; PASS retained).  
4. **Non-goals this patch:** fetching investigation hop notes into Door B; changing daily cap / quiet hours; changing Door A; novelty gate logic.

## Optional follow-up (not this patch)

When `curiosity_content_ids` are present, load the latest investigation hop notes and feed them like Door A — true hop synthesis on the endogenous door. Only after the prompt contract lands.

## Acceptance

1. Prompt with priors + curiosity requires synthesize-from-lanes + why-share.  
2. Orion-only recent turns are not titled as mutual conversation.  
3. Existing grounding / closed-vocab / PASS behavior preserved.  
4. Focused endogenous_outreach tests pass.

## Rollback

Revert `build_outreach_prompt` instruction/label changes + tests.
