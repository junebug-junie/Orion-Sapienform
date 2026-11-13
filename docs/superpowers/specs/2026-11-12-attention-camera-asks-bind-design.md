# Attention patch 4: camera asks obey the market (design)

Date: 2026-11-12. Arc: attention as the want-ledger competition (handoff A, `docs/superpowers/handoffs/2026-11-04-handoff-A-attention-and-ladder-demolition.md`). Binding order (decided default 1): Prime, curiosity, reading (all live), **camera asks**, then outreach (already binds: `services/orion-hub/scripts/outreach_bind.py`), journals, painting/dreams, play.

## In one paragraph

A camera ask is a card that asks Juniper a question about something the camera saw: "Is this <name>?", "Are these the same person?", "Whose vehicle is this?". Today Orion opens these cards under five fixed daily caps. Five minutes later a shadow want is posted for each card that is still open, and the market funds 89 of 90. **Binding as-is would change nothing, and the value it would bind on points the wrong way.** The value is "how many times the camera saw this individual". The asks with the highest value are the vehicle asks, and Juniper dismisses 88% of those with "don't know". The asks she answers (97%) have half the value. So this design has two patches: first make the ask's value count whether Juniper can answer it, then make the ask bid **before** the card is shown, and obey the answer.

## Arsonist summary

- **The shadow want arrives too late to matter.** 158 of 248 asks in 14 days had no want at all. They were answered or closed before the 5-minute loop saw them open.
- **The value is backwards.** Of asks with a want, dismissed asks had a mean net of 0.92 and answered asks 0.49. "Seen 86 times over 5 days" is a high count because a parked or daily vehicle is seen often, and that is exactly the vehicle she cannot name.
- **Nothing can defer an ask today.** A `vision.ask` want has no lanes and no `juniper_msgs` cost, so it is never priced. The only deferral is `sated`, and with the strict-below rank only the smallest past event ranks 0.
- **Juniper's own asks are mixed in.** "Who is this? (named on a walkway story)" rows (97 in 14 days, about 7 a day) are cards *she* makes when naming someone on a walkway story. She answers 100% of them within a minute. They must never bid.
- **So binding without patch 4a would be ceremony,** and binding on today's value would favour the asks she throws away.

## Current architecture (live numbers, 2026-11-12)

### Who opens camera asks

All go through one insert, `_insert_ask` (`services/orion-sql-writer/app/vision_individuals.py:2172`). The selection is a fresh query every tick, so an ask that is not opened stays a candidate. History blocks re-asking: open, answered or dismissed asks block for good; expired or "don't know" asks block for `VISION_ASK_COOLDOWN_DAYS` (30).

| Opener | Kind | Daily cap (env, default) | Order when the cap is tight |
|---|---|---|---|
| `_open_face_asks` (`:2211`), every 60 s | same-person pairs, then face "who is this?" | `VISION_CONFIRM_ASK_DAILY_CAP` 6 (handoff same-asks also count here) | cosine desc, then `face_n` desc |
| `_open_sighting_asks` (`:2317`) | vehicle / individual "whose is this?" | `ORION_ASK_DAILY_CAP` 2 | moving sightings desc |
| `_open_person_confirm_asks` (`:2387`) | "Is this X? best guess" | `VISION_PERSON_CONFIRM_ASK_DAILY_CAP` 6 + `..._PRIOR_...` 2 | non-prior first, score desc |
| `vision_handoff_loop.py:917` `step_asks`, every 15 s | door then walkway "same person?" | `VISION_HANDOFF_SAME_ASK_DAILY_CAP` 2 | appearance score desc |
| `vision_office_sessions.py:684` | office "who is in the office?" | `VISION_OFFICE_ASK_DAILY_CAP` 4 | — |

Cards expire after `VISION_ASK_EXPIRY_DAYS` (7). Hub lists open cards (`services/orion-hub/scripts/ask_routes.py:402`). An answer goes on `orion:ask:answered` to `services/orion-substrate-runtime/app/ask_answered_listener.py` and to sql-writer `_apply_answers` (`:1987`), which applies labels.

### Who answers, 14 days (Orion-made asks only)

- About 10.8 asks a day. 79% answered, 21% dismissed. Median time to answer is 11.7 h.
- By type, over 30 days:

| Type | Asks | Answered | Dismissed |
|---|---|---|---|
| "Is this X? best guess" (confirm) | 137 | 97% | 3% |
| "Are these the same person?" | 46 | 96% | 2% |
| "I have seen this same vehicle N times" | 43 | 12% | **88%** (`dont_know`) |

### The market today

- `config/physiology/families.yaml` `vision.ask`: tier 2, window 86,400 s, `lanes: []`, ungraded (no act-ledger actor).
- Producer `services/orion-sql-writer/app/ask_event_wants.py` (every 300 s, after open). Salience: the individual's moving-sighting count, as a strict-below percentile of 30 days of camera asks (`orion/physiology/event_wants.py:34`).
- 14 days: funded 89 (`net_positive`), exploration floor 1, deferred 0.
- Ungraded, so `rank_value` (`orion/physiology/appetite.py:236`) ignores any producer `outcome_value`: rank rests on salience alone.

### The pattern that already exists

- `hub.outreach` is an act (`actor hub.outreach, verb say`, 250 in 30 days) that the horizon resolver closes graded: did Juniper reply. That is a world grade (section 5 of the patch-3 spec).
- Binding: `orion/physiology/binding.py` `bind_verdict` and `BINDING_PRODUCERS`; `outreach_bind.py` is the Hub example, `RunBidder.bind` the durable-runs one.

## Proposal

### 4a. The ask's value counts whether Juniper can answer it (no behaviour change)

Make `vision.ask` a **graded** family whose grade is Juniper's answer. This uses the outreach pattern and needs no new score.

1. **Each Orion-made ask opens an act.** In `_insert_ask`, emit the grammar open event for `actor vision.ask`, `verb ask`, `object_ref orion_ask:<ask_id>`, `correlation_id` = the ask id, horizon = the ask's `expires_at`. The act ledger opens it (`orion/grammar/act_ledger.py:353` `open_act`).
2. **Juniper's answer closes it with a grade.** Answered with a name, yes, no, or same/different: `outcome_value 1`. Dismissed, or "don't know": `0`. Expired unanswered: the resolver closes it `0` (her silence for 7 days is an answer, as for outreach). Withdrawn by Orion: `expired`, not a grade. The close rides the existing answer path (`_apply_answers` already sees every answer).
3. **The want carries an expected outcome.** `ask_event_wants.plan_asks` sets `outcome_value` = the answered share of the same ask **rule** (confirm, same-pair, face, vehicle, office) over 30 days, with a Laplace prior (`(answered + 1) / (closed + 2)`). Once the family has 30 graded closes (it will within 3 days), `rank_value` ranks `salience x outcome`.
   - Vehicle asks: about 0.4 of today's rank (salience high, outcome about 0.13).
   - Confirm asks: about 0.97 of today's rank.
4. **Back-fill 30 days** of acts from `orion_ask` (413 rows, under the section-14 backfill protocol, snapshot to `/tmp/camera-ask-acts-backfill/`), so the family is graded on day one.
5. **Juniper's own cards open no act and post no want.** Exclude rows made by the walkway story-naming path (today they are found only by their question text "named on a walkway story"; 4a adds an explicit `source_ref` marker at that insert so the filter is not a text match).

Metric gate (CLAUDE.md 0A), short:
- Provenance: `orion_ask.status` and `answer_detail.dont_know`, written by Hub `ask_routes.py:587/:600`.
- Independence: salience counts sightings; the outcome counts her answers. Different sources.
- Theory: an ask is worth what its answer resolves, times the chance of an answer. A card she cannot answer resolves nothing.
- Live check: per-rule answered share is 0.12 to 0.97 (not flat). It can return to rest: a rule with no recent asks falls back to the prior 0.5.
- Existing mechanism: graded families and the outreach resolver already exist. This adds no new score.
- Reversible: drop the `outcome_value` and the family goes back to ungraded ranking.

### 4b. The ask bids before it opens, and obeys (flag `ORION_PHYSIOLOGY_BID_CAMERA_ASKS`)

1. **One bid site.** A small `ask_bind(candidate) -> BindVerdict` in sql-writer, called by every opener just before `_insert_ask`. It builds the event want for the candidate (`event_ref orion_ask_candidate:<rule>:<subject>`, a deterministic want id per candidate, salience and `outcome_value` as in 4a). Producer: `orion-sql-writer.ask_events.binding`, added to `BINDING_PRODUCERS`. It uses the synchronous `orion.physiology.client.bid` that sql-writer already has (0.5 s timeout; no answer means admit, as `bind_verdict` does).
2. **Obey `bind_verdict`.** Funded: open the card. Held (`sated`, or a Prime charge): do not open it this tick. The candidate comes back next tick on its own; a small map keyed by want id keeps its `retry_at`, so it is not re-bid every 60 s.
3. **At the window end (24 h held): put it on the 30-day cooldown, do not open it.** This is the one place the design differs from the "never dropped" rule of runs. An unopened ask that waits a day is not lost: it becomes a candidate again after the same 30-day cooldown an expired ask gets today. **This is missing question 2 below.**
4. **The caps stay.** They remain a hard ceiling for now. Under the rung rule (decided default 5), each cap retires in its own PR after two weeks of binding.
5. **The 5-minute shadow loop stops posting** for asks that bid at open (one want per ask, no duplicate). It keeps posting only while the flag is `shadow`.
6. **Ships `on`** (standing rule), with `shadow` as the off switch.

## Missing questions (each with a recommended default)

1. **Should an ask cost Juniper's attention?** Today a card is "answered when she likes", so it has no `juniper_msgs` cost (`families.yaml:232`). Charging it would put cards and outreach messages on one daily line. **Default: no cost in v1.** The grade (4a) already makes cards she can't answer rank low. Revisit if open cards pile up.
2. **A held ask at its window end: open it anyway, or cool it down for 30 days?** **Default: cool down.** Opening it anyway would only delay the vehicle cards by a day.
3. **Expired unanswered = grade 0?** **Default: yes**, as outreach treats no reply. If you leave cards open on purpose for later, say no; then expiry is `expired`, not a grade.
4. **Vehicle asks: let the market starve them, or retire the rule outright?** **Default: let the market decide.** At an expected outcome of about 0.13 they will mostly be sated, and the few with a fresh, rare vehicle can still win.

## Proposed schema / API changes

- No new table or column. New act-ledger actor `vision.ask` (verb `ask`), closed by producer (answer) and resolver (expiry).
- `WantV1.outcome_value` set by the ask producer (field exists).
- New producer id `orion-sql-writer.ask_events.binding` in `orion/physiology/binding.py`.
- `families.yaml` `vision.ask`: comment updated (graded, binds); no new keys.
- New env `ORION_PHYSIOLOGY_BID_CAMERA_ASKS=on` (orion-sql-writer `.env_example`, settings, compose).

## Files likely to touch

- `services/orion-sql-writer/app/vision_individuals.py` (`_insert_ask`, the three openers), `vision_handoff_loop.py`, `vision_office_sessions.py`
- `services/orion-sql-writer/app/ask_event_wants.py` (outcome per rule; stop posting when binding)
- new `services/orion-sql-writer/app/ask_bind.py`
- the act close on answer: `_apply_answers` (`vision_individuals.py:1987`)
- `orion/physiology/binding.py`, `config/physiology/families.yaml`
- `services/orion-sql-writer/{settings.py,.env_example,docker-compose.yml}`
- `services/orion-durable-runs/evals/event_wants_replay.py` (the replay lane below)
- tests in `services/orion-sql-writer/tests/`

## Non-goals

- Retiring any daily cap (each gets its own PR after two weeks of binding).
- A cross-opener value order (one queue for all openers). The caps are per opener today; that order comes when the caps retire.
- Pricing cards in Juniper's attention (question 1).
- Changing what Hub shows or how answers apply labels.
- Story-person asks (`vision_story_person`), which are Juniper's labels.

## Acceptance checks

1. **4a live:** a `vision.ask` act opens for each Orion-made ask and closes on answer; `grammar_acts` shows `outcome_source` producer/resolver; family reads `graded` within 3 days of the back-fill.
2. **4a value moves the right way:** over the next 7 days, mean net of dismissed asks is **below** that of answered asks (today 0.92 vs 0.49).
3. **Replay gate before 4b ships** (live, 30 days, Orion-made asks, with 4a's value and 4b's rule):
   - at least 70% of vehicle asks held to the window end;
   - at most 10% of confirm and same-person asks held at all;
   - answered asks per day fall by at most 10%.
4. **4b live week:** `want_ledger` rows for `orion-sql-writer.ask_events.binding`; Juniper-made cards have no want; held-to-cooldown count per rule; answered asks per day within 10% of the week before.
5. **Off switch:** `shadow` restores today's behaviour (open first, shadow want after).

## Recommended next patch

**Patch 4a: graded asks (no behaviour change).**
1. Open an act per Orion-made ask, close it on Juniper's answer, the resolver closes expiry.
2. Mark Juniper's own story-naming cards at insert, and exclude them.
3. Expected outcome per ask rule on the want.
4. Back-fill 30 days.
5. Add the replay lane for gate 3.

Verify: a new ask opens a `vision.ask` act, and her answer closes it with outcome 1.

Then **4b** (bid before open), after the replay gate passes.

## Proposal-mode checklist

- Capability change: Orion stops opening cards it predicts Juniper can't answer.
- Data touched: `orion_ask` (read; one marker on Juniper's story cards), `grammar_acts` (new actor), `want_ledger`.
- Privacy boundary: none new. Names already live in `orion_ask`.
- Trace that proves it: `grammar_acts` closes for `vision.ask`; `want_ledger` binding rows; per-rule held counts.
- Dangerous failure: a bug holds every card, and Orion stops learning faces. The replay gate and the answered-per-day check catch it; `shadow` restores today's behaviour.
- Rollback: flag `shadow` (4b); drop `outcome_value` (4a).
