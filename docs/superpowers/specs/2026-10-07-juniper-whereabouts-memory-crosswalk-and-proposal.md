# Juniper's whereabouts: memory crosswalk and proposal

Status: PROPOSAL (proposal mode, CLAUDE.md 0A: changes what Orion remembers about Juniper). Nothing here is built.
Date: 2026-10-07. Evidence taken live from `conjourney` and code on main @ bbc51703a.

The repo is public. Following the Stage-2 referent spec (line 22), Juniper's real whereabouts are never quoted here: "a travel city" stands in for the real one.

## Arsonist summary

Orion has no place to keep "where Juniper is right now." Juniper said "I'm in a travel city until Wednesday" on 10-05. On 10-06, 44 hours later, Orion asked whether she'd make it back to the basement before bed.

Live evidence for that turn (`3af6a31d…`): the travel city appears **zero times** in everything Orion saw (cognition trace, mind run, harness trace). What it did see:
- the fixed `Place: home_location=Ogden… Orion physical_location=…basement…` line, which is in every prompt;
- seven recalled old messages, including home cable-running work and "I updated your situation prompt so you stay focused on your physical location."

The next turn (`d96a6c3b…`) knew only because Juniper repeated it. Orion still invented a food detail ("deep dish, I assume") that Juniper never said: `deep dish` appears in no chat table except Orion's own reply.

Of the 21 memory primitives below, none stores "Juniper is at X from A until B" **and** reaches the prompt on every turn. The ones with expiry don't reach chat. The ones that reach chat have no expiry, or depend on recall happening to surface them.

## Current architecture: the crosswalk

The question each column answers:
- **Same-turn write?** Can it be written during the turn the fact is said, so it is there next turn?
- **Every turn?** Does it reach the prompt on every turn, without depending on a recall ranking?
- **Expiry?** Can it end on its own?
- **Keyed to Juniper?** Is it about a subject, or free text?

| # | Primitive | Holds | Same-turn write | Every turn | Expiry | Keyed to Juniper | Fit for whereabouts |
|---|---|---|---|---|---|---|---|
| 1 | `chat_history_log` (+ `chat_message`) | raw turns | yes | last 8 turns only (`HARNESS_RECENT_TURNS_MAX`) | no | no | **where the fact lives today, and why it is lost** |
| 2 | sql_timeline | recall view over #1 | — | recall-ranked | no | no | same problem as #1 |
| 3 | `memory_cards` | curated facts, `still_true[]`, `time_horizon` jsonb | via extractor, review-gated (`pending_review`) | always-inject subset via recall (`cards_adapter.py:86`) | weak (free jsonb) | anchors, semi | review gate is too slow; no hard end time |
| 4 | `memory_crystallizations` | consolidated claims, superseded status | async, governor | recall-ranked (`active_packet.py:121`) | supersession only | `subject` text | async + ranked; spec #2440 says the auto path saves raw chat junk |
| 5 | graphiti episodes / temporal | `valid_from`/`valid_to` | async | via #4 | yes | entity | dropped from Stage 2 (referent spec, decision 5) |
| 6 | `episode_memory` | distilled statements; `purpose=about_juniper`, `expires_at`, `supersedes_memory_id` | no: post-episode distill run | **no chat consumer** (SHADOW, Stages 1-3) | **yes** | via `episode_memory_referent` | right shape, wrong timing and not live in chat until Stage 4 |
| 7 | `referent_alias` / projection | entity names; `valid_until` for descriptors | consolidation | no chat consumer yet | descriptors, 90 d | entity | future home for the *place* as a referent, not the stay |
| 8 | `substrate_turn_referent` | per-turn surprise closure | yes | reverie only | no | no | no |
| 9 | `chat_stance_belief_log` | per-turn belief snapshot | yes | self-study only | no | no | no |
| 10 | `evidence_units` | generic evidence index | yes | no chat consumer | no | no | no |
| 11-12 | journals / pageindex | journal text | async | verb-invoked | no | no | no |
| 13 | self-concept / self-knowledge | Orion's self-model | async | felt-state hydrate | version | `self:*` | about Orion, not Juniper |
| 14 | `juniper_affective_state_log` | affect history | yes | self-study | no | Juniper | history only |
| 15 | Redis `orion:juniper_affect:latest` | latest affect | yes | **yes** ("Juniper's affect: …") | **TTL** | **Juniper** | **right delivery pattern**; Redis-only, holds no history |
| 16 | collapse_mirror | Juniper's manual entries | manual | recall disabled | no | no | dormant |
| 17 | social memory | room participant continuity | yes | social-room path | no | participant | different surface |
| 18 | substrate episode summaries | episode summaries | async | thought grounding | no | no | no |
| 19 | Falkor `orion_recall` | turn/entity graph | async | recall-ranked | no | entity | ranked, so same problem as #1 |
| 20-21 | substrate concepts, anchor stance | concept nodes | async | recall-ranked | no | concept | no |
| — | Hub `PresenceContextStore` | per-session presence, `notes`, `expires_at` | yes (`/api/presence`) | yes, but `notes` is never rendered | TTL, but `context.py:861` overwrites it with now+4h | session, in memory | almost: lost on restart, per-session, notes dropped |
| — | mind `semantic_synthesis` | per-turn claims (one labeled "current location") | yes | stance only | no | no | **natural trigger**: it already noticed; nothing keeps it |

Code pointers:
- prompt path: `orion/hub/turn_orchestrator.py:952` → situation brief `:776` → `orion/harness/prefix.py:208`;
- Place line: `orion/situational/context.py:2136-2149`;
- schema: `orion/schemas/situation.py:108` (`PlaceContextV1`, whose `source` literal already allows `"manual"`);
- affect pattern: `orion/situational/juniper_affect_state.py:67,207`;
- mind claims: `services/orion-mind/app/synthesis.py:171-294`, schema `orion/mind/synthesis_v1.py`.

**Verdict:** this calls for a new primitive. The nearest fits fail on things that matter here:
- **episode_memory (#6):** it has the right fields, but is written hours later by a batch run and has no chat reader until Stage 4.
- **affect key (#15):** right delivery, but Redis-only. An "until Wednesday" fact must survive a Redis flush, and the earlier stays should be auditable.
- **presence store:** in memory, per session, and its notes are dropped.

The new primitive copies #15's delivery and #6's field names, so it can fold into either later without a translation layer.

## Missing questions

1. **Who may write it?** Proposal: two writers. Juniper's own words in chat (extracted the same turn), and a manual Hub control. Orion's own guesses ("you're probably home by now") never write.
2. **When it has no end time** ("I'm in a travel city"), what then? Proposal: a default TTL of 3 days, and Orion is told the fact is "unconfirmed after <date>" rather than silently treating it as true or dropping it.
3. **The laptop webcam (UNVERIFIED that it fired these turns).** The `carbon` webcam travels with Juniper, but the brief renders it as "Room (seen …)" next to the home Place line. While away, should it be labeled as the laptop camera? Proposal: yes, in the same patch. It's a one-line render change and part of the same confusion.
4. **Privacy.** Whereabouts are sensitive. They stay in Postgres and the prompt, are never written to journals/self-study or anything that leaves the host, and are never quoted in repo docs or test fixtures (fixtures use placeholder cities).

## Proposed schema / API changes

**Event** (registered in `orion/schemas/registry.py` and `orion/bus/channels.yaml`):

```text
JuniperWhereaboutsV1
  whereabouts_id   uuid
  place_text       str          # Juniper's words, e.g. "<travel city>"; free text, no place taxonomy
  status           away | home  # "home" is the explicit "I'm back"
  valid_from       datetime
  valid_until      datetime | None
  until_confirmed  bool         # False when valid_until is the default TTL, not Juniper's own words
  source           chat_extracted | manual
  correlation_id   str          # the turn she said it in
  supersedes_id    uuid | None
```

Channel: `orion:juniper:whereabouts` (publish only on change).

**Storage:** table `juniper_whereabouts` (sql-writer, append-only, one row per event).
- The "current" row is the latest one where `valid_until > now()` and nothing supersedes it.
- Additive migration.
- Rollback: flag off, then drop the table.

**Writer: same-turn extraction.**
- mind's synthesis already emits a "current location" claim with evidence refs (`synthesis.py`), so no word lists are needed. Extend that claim with optional structured fields (`place_text`, `until_text`, `status`), filled by the same LLM call that already produces the claim.
- A small reducer in cortex-exec turns a claim whose evidence ref is `current_turn` (Juniper said it) into a `JuniperWhereaboutsV1` event.
- Dates are resolved deterministically in code, never by the model: "until Wednesday" → the next Wednesday 23:59 in Juniper's timezone.
- Claims sourced from recall or inference never write.

**Writer: manual.** `POST/DELETE /api/presence/whereabouts` on the hub, plus a small control in the Hub presence panel.

**Reader.** `_build_place_context` reads the current row (cached under the existing situation cache, with the row id added to `_situation_cache_key`). The up-front Place line, which the budget cap never trims, becomes:

```text
Place: home_location=Ogden, Utah; Orion physical_location=<cabinet>; Juniper is away: <place_text> until <date> (she said so <n>h ago).
```

`PlaceContextV1` gains an optional `requestor_whereabouts` sub-model. This is additive. The model is `extra="forbid"`, so consumers deploy first, per [[additive-schema-fields-are-a-consumer-first-migration-on-forbid-models]].

**Laptop camera label.** While `status=away`, the perception line for the `carbon` stream renders as "Laptop camera (with Juniper)", not "Room".

**Expiry trace.** When a row lapses, emit nothing new. The Place line just stops showing it. Once that happens the debug surface shows `whereabouts: none (last: lapsed <date>)`, so "Orion forgot" and "Orion was never told" are distinguishable.

## Files likely to touch

- `orion/schemas/situation.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`
- `orion/mind/synthesis_v1.py`, `services/orion-mind/app/synthesis.py` (structured fields on the existing claim)
- `services/orion-cortex-exec/app/` (new reducer `whereabouts_reducer.py`, plus settings flag)
- `services/orion-sql-writer/app/models/juniper_whereabouts.py`, plus worker route and migration
- `orion/situational/context.py` (read, render, cache key, camera label)
- `services/orion-hub/scripts/api_routes.py`, plus the presence panel template/JS
- `.env_example` for cortex-exec, hub and sql-writer (`ORION_WHEREABOUTS_ENABLED=true`, `ORION_WHEREABOUTS_DEFAULT_TTL_HOURS=72`), then sync local `.env`

## Non-goals

- No place taxonomy, geocoding, or location tracking from devices or IPs. Juniper's words only.
- No general "user current-state" framework. Whereabouts is the one fact with a live failure behind it. A second one earns its own slot when it has its own incident.
- Not a replacement for episode_memory or referents. At Stage 4, a lapsed stay can be distilled into an `episode_memory` "happened" row, and `place_text` can gain a `referent_id` once Stage 2 resolves places. Both are seams, not work in this patch.
- No change to recall ranking.

## Acceptance checks

1. **Regression (gate test):** replay the 10-05 → 10-06 shape with a placeholder city. Turn 1 says "in <city> till Wednesday"; turn 2, 44h later, is "just chilling." The turn-2 situation fragment contains "away: <city> until Wed". Before the fix it doesn't, and the test fails.
2. **Expiry:** after `valid_until`, the line is gone and the debug surface says "lapsed."
3. **"I'm back"** writes `status=home`, supersedes the away row, and the line clears.
4. **No inference writes:** a recall-sourced or Orion-inferred location claim produces no event (test).
5. **Date math:** "until Wednesday" said on a Sunday, a Wednesday, and across a timezone boundary resolves deterministically (test).
6. **Live smoke:** say "I'm in <city> until tomorrow" in the hub. A `juniper_whereabouts` row appears with that turn's correlation_id. The next turn's harness trace shows the Place line.
7. **Eval:** a 6-scenario eval in `services/orion-cortex-exec/evals/`, run with flag on vs off. It checks whether the reply implies Juniper is home when she's away. This is the actual failure, scored by a fixed rubric.

## Proposal-mode checklist

- **Capability change:** Orion keeps Juniper's stated whereabouts across turns until they expire.
- **Data touched:** a new table, an event, the situation Place line, and an optional hub control.
- **Privacy boundary:** host-local Postgres and the prompt only. Not journaled, not in self-study, never quoted in repo artifacts.
- **Proof it worked:** a `juniper_whereabouts` row carrying the turn's correlation_id, plus the Place line in the next harness trace.
- **Dangerous failure:** a wrong or stale location presented as fact (for example, Orion telling someone Juniper is away when she's home). Mitigations: only Juniper's own words write, everything expires, `until_confirmed=false` is rendered as unconfirmed, and "I'm back" clears it.
- **Disable / roll back:** `ORION_WHEREABOUTS_ENABLED=false` stops both writer and reader, and the Place line reverts to today's. The table can be dropped.

## Recommended next patch

One PR, flag on (per the always-ship-flags-on rule):
1. schema + registry + channel + table;
2. the reducer from mind's existing claim, plus deterministic date resolution;
3. the Place-line render, cache key and laptop-camera label;
4. the regression test from acceptance check 1, plus the eval.

The hub manual control can follow in a second PR if the first gets large.
