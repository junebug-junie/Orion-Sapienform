# A known person arriving can win Orion's attention

Status: PROPOSAL (attention change; CLAUDE.md proposal mode). Not implemented.
Date: 2026-10-27. Follows PRs #2852, #2858 and #2897 (one record per person, person nodes in the graph).

## Arsonist summary

Orion's day letter said nothing attends to the people the walkway camera sees. That is true, and it is measured.

Over 7 days, the attention log (`substrate_attention_broadcast_log`) holds 9,873 broadcasts. None of them names a camera person or a person node. Attention never picks graph nodes. It picks from organ readings: `node:substrate.street`, `node:substrate.vision`, `node:substrate.perception_walkway` and others. So wiring person nodes "into attention" cannot work.

The right seam already exists, but it is dead:

- The street gate (`orion/substrate/street_attention.py`) states Juniper's rule from PR #2763: "unusual for the hour, **or a known person**". But its code says a known person "is recorded in the reason but changes nothing yet".
- The event kind `known_person_arrived` is declared in `StreetEventV1` (`orion/schemas/vision.py:643`). Nothing produces it, and the gate cannot pass it, because it has no p-value.

This proposal adds a producer for that event and lets the gate pass it under the same safeguards that rings and handoffs already have.

## Current architecture

- **Producer of street events:** orion-sql-writer `app/vision_handoff.py`. It publishes `StreetEventV1` to `orion:vision:street:events`.
  - Its handoff events carry `person_guess`, which is a name the interlocutor gave, never a face match.
  - Doorbell rings come from orion-vision-retina `app/ring_events.py`.
- **Consumer:** orion-substrate-runtime `street_gate()`. One verdict per event.
  - A ring always passes.
  - Any other event needs all of: a p-value under 0.05; every camera in `stream_ids` up (a frame within the limit); fewer than `SUBSTRATE_STREET_ATTENTION_MAX_PER_HOUR` passes this hour (default 4).
  - A passed event's salience is held on `node:substrate.street` for 90 s, then the node drops back to a real 0.0.
- **Who a cluster is:** `person` / `person_link` (PR #2852). A link is active while `revoked_at IS NULL`.

## Missing questions

1. **What counts as "arrived"?** Recommendation: the first sighting of a linked person's cluster on a camera after that person has been unseen for at least `KNOWN_PERSON_ABSENT_SEC` (default 3 h) on every camera. This is the habituation Juniper's rule requires: someone who walks past all afternoon wins once, not every time.
2. **Which people?** Recommendation:
   - Every active person (`status='active'`, role not `self`).
   - Not the interlocutor. Juniper on her own patio is not news to Orion. A config flag can turn this off.
   - Only clusters with an active `vision_individual` link to a person, never a picture-match guess (`vision_person_model` bars).
3. **Does it bypass the hourly cap?** Recommendation: no. It shares the street's 4-per-hour budget with rings and handoffs, so a busy street cannot flood attention.

## Proposed schema / API changes

No new schema, channel or table. Uses the existing `known_person_arrived` kind.

1. **Producer (orion-sql-writer):** a small pass beside the handoff pass. It runs on new `vision_individual_sighting` rows.
   - Trigger: a sighting starts for an individual with an active link to a qualifying person, and that person's previous sighting on any camera ended at least `KNOWN_PERSON_ABSENT_SEC` ago.
   - Payload: one `StreetEventV1(kind="known_person_arrived", stream_ids=[sighting stream], individual_id, person_guess=display_name, subject_kind="person", salience=...)`.
   - Salience: from the existing `salience(kind, subject, known_person=True)`. Table value to be set in the patch; the proposal is 0.6, below a ring.
   - The `event_id` is derived from the sighting id (idempotent).
   - Kill switch: `VISION_KNOWN_PERSON_EVENTS_ENABLED`, shipped on.
2. **Gate (orion-substrate-runtime `street_gate`):** a `known_person_arrived` event passes without a p-value, only when its camera is up and under the hourly cap.
   - It is the "known person" clause the docstring already names.
   - The reason string is `known person <name>`.
   - Every other kind keeps today's rules.

## Privacy boundary

Household names only. They already ride on handoff events as `person_guess`. No crops or embeddings enter the event.

## Trace that proves it worked

- Producer log: `[KNOWN_PERSON_ARRIVED] person=<id> individual=<id> stream=<s> absent_sec=<n>`.
- Gate log: the existing `street_gate passed ... reason=known person <name>`.
- Measured: rows in `substrate_attention_broadcast_log` whose world-first winner is `node:substrate.street` with that reason. Today there are 0.
- Live acceptance:
  - At least one such winner on the first day a named person walks by with the camera up.
  - No more than 4 street passes in any hour.

## Dangerous failure modes

- **A wrong name becomes salient.** Clusters can be impure (the Teddy cluster held one adult crop until 10-27). The mitigation is that only links from typed names count, never guesses. Splits and renames revoke links.
- **Flooding.** Mitigated by the absence window plus the shared hourly cap. `street_attention_max_per_hour=0` silences the street entirely.
- **Camera down reads as "no one came".** The existing camera-up check refuses events from a camera that has delivered no frame recently. The walkway camera has been offline since 2026-10-26 08:00 UTC (No route to host), so this guard is live today.

## Rollback

- `VISION_KNOWN_PERSON_EVENTS_ENABLED=false` stops the events.
- Reverting the gate branch makes any stray event fail with "no_p_value".

## Files likely to touch

- `services/orion-sql-writer/app/vision_handoff.py` (or a sibling `vision_known_person.py`), `settings.py`, `.env_example`, tests.
- `orion/substrate/street_attention.py` `street_gate`, tests; `services/orion-substrate-runtime/evals/run_street_gate_mutation_check.py`.
- `config/field/field_channel_glossary.v1.yaml`: the `node:substrate.street` meaning gains the known-person trigger.

## Non-goals

- Making person graph nodes cognitive (they stay fenced, like chat-memory referents; `orion/substrate/eligibility.py`).
- Face matching or picture-model guesses as a trigger.
- Doorbell-only arrivals. The doorbell has no clusters linked to people yet.

## Acceptance checks

- **Unit (gate):**
  - `known_person_arrived` with the camera up and under the cap passes.
  - With the camera stale, it fails.
  - With the cap used up, it fails.
  - A handoff with no p-value still fails.
- **Unit (producer):**
  - The first sighting after 3 h emits.
  - A second sighting 10 minutes later does not.
  - Copied-name, unlinked, not_a_person and interlocutor clusters do not emit.
  - A replayed sighting emits nothing new.
- **Replay (done 2026-10-27, read-only, last 7 days):** sightings of linked, typed, active non-interlocutor people, counted per local day as arrivals (gap of at least 3 h) out of raw sightings.

  | Day (MDT) | Arrivals | Raw sightings |
  |---|---|---|
  | 10-19 | 1 | 1 |
  | 10-20 | 8 | 29 |
  | 10-21 | 8 | 13 |
  | 10-22 | 4 | 6 |
  | 10-23 | 5 | 19 |
  | 10-24 | 5 | 19 |
  | 10-25 | 6 | 155 |
  | 10-26 | 4 | 46 |

  The absence window cuts up to 155 raw sightings a day down to at most 8 arrivals. With the 4-per-hour cap, the street cannot dominate attention. The replay also showed a person row named for a yard decoration. It was marked `not_a_person` the same day.
- **Live:** the measured trace above, after the walkway camera is back.

## Recommended next patch

Implement items 1 and 2 behind the kill switch (shipped on), with the replay numbers in the PR report.
