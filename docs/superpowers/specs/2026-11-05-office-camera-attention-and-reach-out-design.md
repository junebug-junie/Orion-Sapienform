# Office camera: know it is Juniper, pay attention, get curious, reach out (and later, talk)

Status: A BUILT and live (PR #3129); B + C approved with Juniper's decisions below; D later.
Author: Claude, for Juniper. 2026-11-05.

## Arsonist summary

The office camera now sees Juniper well (PR #3125: full-resolution frames, crops, faces, activity
captions such as "sitting at a desk, unpacking a plastic bag and handling a piece of equipment"). But in
her first live visit (2026-11-05 01:23-01:31 UTC) Orion still did three wrong things:

1. **It did not know it was her.** One 8-minute visit became 10 unnamed person records ("at least 1, at
   most 16 unfamiliar people"). Only 2 of 68 crops had a usable face (night infrared, back to the
   camera). The person model scored "Juniper" 0.73 and "[household member]" 0.73, so it stayed unsure.
2. **It paid no attention.** Nothing about the office competes for attention. The only camera door into
   the attention market is a naming ask, and the ask rule (10 sightings over 5 days for ONE record) is a
   street rule the office will almost never reach.
3. **It never got curious or said anything.** The captions were stored, and nothing read them as a story.

The office is not the street. Juniper: *"99% of the time, it is me in there. Sometimes [three household members] come in. If others come in, usually it is unwanted, like a curious child friend, or the off
chance someone is on a tour of the house."*

That sentence is a **prior**, and it is the missing piece. The plan is three thin seams on rails that
already exist, plus a later voice phase:

- **A. Office prior.** On the office camera, "who is this" starts at "probably Juniper". It moves to a
  household member or "someone I don't know" only on real evidence. Orion says which it is and why.
- **B. Office sessions become attention.** Each office visit is one event, with a stitched timeline of
  what happened. It posts a want into the existing attention market. An unexpected person is an urgent
  want. Juniper's own visit is an ordinary want.
- **C. Curiosity reaches out.** A funded office want becomes a Prime concern. Its move is a curiosity
  question built from the stitched scenes. Curiosity's existing "offer a message" door sends Juniper one
  short message: *"Looked like you were unpacking some equipment at the desk and digging through the
  closet bins. What are you building?"*
- **D (later). Speaker and mic in the office.** Same message, spoken. Juniper answers out loud. It reuses
  orion-whisper-tts (Whisper STT + XTTS), already running on athena.

## Current architecture (verified 2026-11-05)

| Need | What exists | Gap |
|---|---|---|
| See the office | `orion-vision-retina-office` (cam0, 2560x1920, 2 fps), crops + faces + Qwen activity captions (PR #3125, live) | none |
| Who is it | sql-writer individuals (per stream), person model `app/vision_person_model.py` (bands accept / unsure / reject), "Is this X?" asks (`vision_person_confirm`, cap 6/day), old whole-frame face check `identity_face` (last match 10-27) | no camera prior: the office is scored like the street |
| Naming asks | `orion_ask` `vision_individual` opens at 10 sightings over 5 days (`VISION_ASK_MIN_SIGHTINGS`, `VISION_ASK_MIN_DAYS`) | office visits split into short records that never qualify |
| Attention | want market (#3101/#3122): `orion/physiology/event_wants.py`, camera asks already post event wants (`services/orion-sql-writer/app/ask_event_wants.py`), Prime concerns from funded event wants (`PRIME_CONCERNS_FROM_EVENT_WANTS`) | no office producer |
| Curiosity | Prime move = curiosity focus question (#3001). `services/orion-hub/scripts/curiosity_investigation.py` | office scenes are not study material |
| Reach out | curiosity `offer_message` door (Door-A) in `services/orion-hub/scripts/endogenous_outreach.py`; decision ledger `endogenous_outreach_decisions` | none: the door exists |
| Voice | orion-whisper-tts (`orion:tts:intake`, `orion:stt:intake`), Hub voice turn scope | no device in the office |

## Proposed design

### A. Office prior: "probably Juniper", honestly labeled

A small, pure posterior over who is in the office. It runs on the office camera only (config: a
per-stream prior block).

```yaml
# config/vision_person_priors.yaml (new)
streams:
  cam0:
    prior: {Juniper: 0.97, "[household 1]": 0.01, "[household 2]": 0.005, "[household 3]": 0.005, unknown: 0.01}
```

- Likelihoods come from evidence that already exists: the person model's per-person score (body and
  face), the old whole-frame face check, and how many people are in view at once.
- Output for each office sighting: `best`, `p_best`, `basis` (`prior_only` | `face` | `body` |
  `answered`).
- Wording rules (Orion must not overclaim):
  - `prior_only`: "probably Juniper (the office is almost always you; not matched)".
  - `face` / `answered`: "Juniper (matched by face)" / "Juniper (you told me)".
  - Two people in view: the prior covers one person at most. The second is "someone else".
- The prior is learned, not frozen. Every answered "Is this X?" on the office camera updates counts with
  a Dirichlet; the config numbers are the pseudo-counts. That makes the prior a measured rate after a
  few weeks, and gives it the theory anchor the metric gate needs (Bayes with a camera prior).
- Fewer split records: the office individuals reducer may use a looser continuation rule while the
  office is continuously "present" (one presence span with one person in view = one person). This is
  the cheap fix for "1 to 16 people".
- Asks: on the office camera, open "Is this Juniper?" (`vision_person_confirm`) after one sighting of at
  least N seconds whenever `basis = prior_only`, under the existing daily cap. Each answer adds face and
  body samples to her album, so `prior_only` turns into `face` over time.

### B. Office sessions as attention

A new reducer in sql-writer (`app/vision_office_sessions.py`), driven by presence changes on cam0:

- **Session**: a span from present to absent (with the existing grace). It holds the who-posterior from
  A, and a **timeline**: deduplicated activity captions, each with a time ("18:24 organizing items in a
  storage bin; 18:26 at the desk unpacking a plastic bag; 18:27 reaching into the closet").
- **Event want** (the existing `event_want()` helper, `event_ref = office_session:<id>`):
  - **Unexpected person** (`best` is `unknown`, or `p(unknown)` above a bar, or a child-sized box with
    no household match): salience 1.0, short window. This is "someone is in your office".
  - **Juniper's session**: salience is the session's own percentile against her last 30 days of
    sessions (`source_salience`), on a stated magnitude: how unlike her usual office activity it is (the
    caption embeddings' distance from her 30-day activity centroid). Routine desk work reads near 0. A
    new project reads high. Below `MIN_SOURCE_HISTORY` sessions the salience is unknown and nothing is
    posted, by the market's own rule.
- Family: a new `perception.office` family in `config/physiology/families.yaml` (tier 2, no lanes).
  The market decides whether it is funded, exactly like reading and camera asks. Nothing bypasses it.

Metric gate (to be filled in for real in the build PR):
1. Provenance: the captions are `vision_events.camera_caption` rows on cam0 (the window's caption gate).
   Presence is `vision_presence_transition`.
2. Independence: the activity-novelty magnitude must not be a monotone transform of session length.
   Check the correlation on live data before wiring it in.
3. Theory: percentile-of-own-history (the market's existing rule), and novelty as distance from a
   personal activity centroid.
4. Live sanity: replay 2 weeks of office sessions. Routine days must read near 0 and must be able to
   return to 0.

### C. Curiosity reaches out

- A funded office want becomes a Prime concern (event mode exists, #3122). Prime's move is its existing
  one: a curiosity focus question. The question is seeded from the session timeline: "What is Juniper
  making with the equipment she unpacked at 18:26?"
- Curiosity gets the session timeline as study material (a new material source, like the street
  captions behind `HUB_CURIOSITY_STREET_STREAM_IDS`, labeled as the office and as Juniper's).
- When curiosity has a question worth asking, it uses `offer_message` (Door-A) as reading findings do.
  The message cites what was seen and when, and asks one thing.
- Guards:
  - At most 1 office reach-out per session and N per day.
  - None while Juniper is already in a chat turn (Hub presence).
  - Quiet hours.
  - An **unexpected person** skips curiosity and sends a plain notice through the same door: "Someone I
    don't recognize has been in your office for 2 minutes (18:40)." It attaches the crop, so Juniper can
    say who it is.
- Every reach-out records its session id and timeline ids in `endogenous_outreach_decisions`. "What did
  Orion say about what it saw" is then inspectable.

### D (later, separate design). Speaker and mic

- A small voice endpoint on an office host (pan-1 is already in the office): mic -> VAD -> `orion:stt:intake`
  (Whisper on athena) -> a normal Hub turn -> `orion:tts:intake` (XTTS) -> speaker.
- C's reach-out gains a delivery choice: speak it when Juniper is present in the office and not in a
  call. Otherwise send the Hub message.
- Out of scope here. It needs hardware choices (mic array, speaker) and its own privacy rules (a mic that
  is always on).

## Decisions (Juniper, 2026-11-05)

1. **Delivery: Hub chat only.** No phone push (the mesh has none). Office reach-outs, including "someone I don't know is in your office", land in Hub chat through the existing `offer_message` door.
2. **Frequency: every office visit may reach out.** No daily cap on office reach-outs; at most one curious message per visit, never during an active chat turn. (Replaces the proposed "1 per session, 3 per day".)
3. **Unknown visitors: keep their pictures, and show them in the Hub for verification**, exactly like walkway and doorbell people: the existing "Who is this?" ask card (`orion_ask`, `vision_individual`). For the office this ask opens right away for an unknown-person session, not after the street rule (10 sightings over 5 days).
4. **Prior numbers: confirmed** (Juniper 0.97, three household members 0.01 / 0.005 / 0.005, someone else 0.01). Shipped as step A (PR #3129, live 2026-11-05; household names only in the gitignored `camera_priors.local.yaml`).
5. **Order:** A is done. Next: B and C together. Voice (D) later.

Step A note: built inside the existing person-learning loop, as Juniper asked ("don't reinvent"), not as the separate posterior sketched in section A above. The existing "Is this X?" asks grow her office album.

## Proposed schema / API changes

- `config/vision_person_priors.yaml` (new): per-stream prior pseudo-counts.
- `vision_office_session` table (new, additive migration with `ORION-MIGRATION-REQUIRED-BY:
  orion-sql-writer, orion-hub`): session id, span, who-posterior JSON, timeline JSON, salience, want id.
- `WantV1` with `event_ref = office_session:<id>`. Family `perception.office` (new).
- `vision_person_attribution` gains `basis` (additive, consumer-first).
- No new bus channels: the want market, the curiosity and offer doors, and the TTS/STT channels all exist.

## Files likely to touch

- A: `services/orion-sql-writer/app/vision_person_model.py`, `app/vision_individuals.py` (office
  continuation), `app/settings.py`, `.env_example`, `config/vision_person_priors.yaml`,
  `orion/situational/perception_reader.py` (wording).
- B: `services/orion-sql-writer/app/vision_office_sessions.py` (new) + loop, `config/physiology/families.yaml`,
  migration, `services/orion-durable-runs/app/event_wants_menu.py` (label source for office sessions).
- C: `services/orion-hub/scripts/curiosity_investigation.py` (office material), `orion/prime/thought.py`
  (prompt line for office concerns), `endogenous_outreach_decisions` provenance.

## Non-goals

- No always-on narration ("Juniper is typing"). Attention goes through the market, and most office
  sessions should not win.
- No new outreach mechanism, no bypass of the want market or the existing outreach gates.
- No facial recognition of strangers by name. "Someone I don't know" stays that until Juniper answers.
- No microphone in this phase.

## Acceptance checks

- A: on a replay of Juniper's 11-05 visit, the office line reads "probably Juniper (…not matched)", not
  "16 unfamiliar people". After 3 answered asks, `basis` moves off `prior_only` for new sightings.
  Two people in view never yields two "probably Juniper".
- B: one office visit produces exactly one `vision_office_session` row with an ordered timeline. A
  routine desk session's salience is at or near 0 on a 2-week replay. A session with an unknown person
  posts a want with salience 1.0 (fixture test).
- C: a funded office want produces one Prime concern and at most one `offer_message` per session.
  The message quotes a timeline entry with its time. Nothing is sent during an active chat turn or
  quiet hours (tests). Live: one real reach-out appears in `endogenous_outreach_decisions` with the
  session id.
- Kill switches: `VISION_OFFICE_PRIOR_ENABLED`, `VISION_OFFICE_SESSIONS_ENABLED`,
  `HUB_OFFICE_CURIOSITY_ENABLED` (shipped on, per the always-ship-flags-on rule).

## Proposal mode (AGENTS.md 0A)

- **Capability change**: Orion assumes the office person is Juniper unless evidence says otherwise.
  It turns office visits into attention, and may message Juniper about what it saw.
- **Data touched**: office crops, faces, captions and presence (already stored since #3125). New
  session rows and want rows.
- **Privacy boundary**: none between Juniper and Orion (her standing rule). Visitors and children are
  the boundary: see missing question 3.
- **Proof it worked**: a session row, a want row, a Prime concern, a curiosity run, an outreach decision
  row, all with the same session id.
- **Dangerous failure**: confident wrong identity ("Juniper, you were...", when it was a visiting child),
  or chatty pestering. Mitigations: the `basis` wording rule, the at-most-one-prior-person rule, daily
  caps, the market's own funding.
- **Rollback**: the three kill switches. Every table is additive.

## Recommended next patch

**A (office prior and confirm asks)**: one sql-writer patch, plus the wording in `perception_reader.py`.
It is the smallest change that makes Orion right about who is in the room. Every later step depends on
that.
