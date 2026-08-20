# Community events ingest (theogdenite.com) — a design spec, not an implementation

Status: design mode. No code changes proposed in this document. Source: a live
design conversation with Juniper about scraping
`https://www.theogdenite.com/goings-on-about-otown` (a local Ogden community
happenings page) into Orion, and what Orion should actually do with it. Three
research passes (Explore subagent, this session) traced the real pipelines
involved rather than assuming the happy path; two of the three initial
proposals in this doc were corrected mid-conversation after being called out
as hand-wavy. Corrections are kept inline, not silently erased, so the
reasoning is inspectable.

## Arsonist summary

Juniper wants Orion to have ambient awareness of local Ogden community
events — not quite news, more like "what's happening in the community."
Three candidate seams were evaluated:

1. **`orion-world-pulse`'s news pipeline** — rejected. Built for competing
   news claims (trust tiers, corroboration, claim/entity extraction, a
   shared 12-item digest budget). A farmers-market listing isn't a claim to
   fact-check, and routing it through `world_pulse_journal.py`'s reflection
   trigger would produce cognition-shaped output with no real cognitive
   substance (Orion writing a "reflection" about a food truck rally).
2. **Attention/`OpenLoopV1`/reverie** — rejected, after a first draft
   wrongly called this "zero new cognition machinery." Real mechanics
   (`orion/substrate/attention/policy.py:40`): exactly **one** ask-type
   winner per tick, winner-take-all, and every existing `OpenLoopV1`
   producer today is high-stakes/internally-generated (conversation text,
   autonomy pressure, concept-induction novelty, substrate contradiction).
   A scraped event competing here **displaces** a real concern, and if it
   wins, `self_state/builder.py:75` legibly narrates Orion's self-model as
   singularly focused on it. No ambient/low-priority lane exists to avoid
   this — building one would itself be new, non-trivial machinery. Not
   pursued further. Drives (`DriveEngine`, tensions, field-digester) are
   explicitly out of scope per Juniper's direction — not to be further
   integrated, full stop.
3. **Recall grounding** — the right seam, but a first draft understated the
   work: "lands in recall" is not the finish line. See Current architecture
   below for the real multi-hop wiring chain this actually requires.

## Current architecture

**Ingest template exists, nothing else does.**
`services/orion-world-pulse/app/services/ingest/html_section_adapter.py` +
`base.py` (`http_get_text`, `extract_links`, `is_allowed_url`, `to_candidate`)
already does bounded single-page link/anchor extraction with domain/path
allowlisting — the right shape to imitate for a single events-listing page,
not a library to import cross-service. No repo-wide generic scraper package
exists.

**Recall for a live chat turn is one-profile-per-turn, closed-enum routed —
not "retrievable = surfaced."**
- Each verb (`orion/cognition/verbs/*.yaml`) hardcodes a single
  `recall_profile`; only ~12 of 19 verbs set one.
- Unless a call site sets `profile_explicit=True`, the profile is picked by
  `resolve_profile_for_intent()` (`services/orion-recall/app/intent.py:36-66`)
  — a **closed enum of exactly 4 names**: `biographical.v1`, `reflect.v1`,
  `chat.general.v1`, `self.factual.v1`. It cannot route outside that list.
- `services/orion-recall/app/profiles.py` (`load_profiles`/`get_profile`)
  globs `orion/recall/profiles/*.y*ml` and will happily load any new yaml —
  loadable is not selectable. A new `community.local_events.v1.yaml` file
  alone is dead capability, the same shape as `orion.recall.tag_entity`
  (already confirmed design-only/unimplemented elsewhere in this repo).
  Several other profiles (`graphtri.v1`, `graph.compressions.*.v1`,
  `chat.belief.*.v1`, `assist.light.v1`, `reflect.alerts.v1/anchor.v1/sql_only.v1`)
  are plausibly orphaned the same way — not exhaustively checked, flagged
  as a possibly-recurring pattern worth its own audit some day.
- Budget (`max_total_items=12`, `render_budget_tokens=256` in `fusion.py`)
  is **per-profile**, not cross-profile, because only one profile ever runs
  per turn. There is no cross-profile arbitration to design against — there
  is a routing gate to get through first.

**Recall retrieval is not what reaches Juniper — `chat_stance` is.**
- `services/orion-cortex-exec/app/chat_stance.py`'s `CognitiveUnificationLayer`
  builds the `ChatStanceBrief` that actually feeds the prompt (step
  `synthesize_chat_stance_brief`, order 1, before `llm_chat_general`, order 2).
- Its `recall` producer does not fetch recall itself
  (`freshness_ttl_sec=0, pull_on_cold=False`) — it only projects whatever is
  already in `ctx["recall_bundle"]`.
- `_project_recall_from_beliefs` (`chat_stance.py:704-741`) currently only
  extracts recall items whose `recall_source` metadata contains
  `"journal"`/`"metacog"` into themes/tensions. A new `community` source
  would likely be retrieved but silently dropped here without also
  extending this filter.
- A separate, newer "Orion Unified Turn" architecture
  (`docs/superpowers/specs/2026-07-05-orion-unified-turn-design.md`,
  `services/orion-hub/scripts/unified_turn_stub.py`) is reportedly folding
  chat_stance/PCR into itself. Live wiring vs. cortex-exec was **not
  confirmed** in this investigation — see Missing questions.

## Missing questions

1. **Intent-router branch vs. hijacking `chat.general.v1`?** Adding a 5th
   name to `resolve_profile_for_intent()`'s closed enum is the structurally
   honest fix but touches a router other profiles depend on. The cheaper
   alternative is folding community-event retrieval into `chat.general.v1`
   itself when a turn looks locally-relevant (smaller blast radius, no new
   enum value for every future reader of that function to reason about).
   Juniper has not yet picked between these.
2. **`chat_stance` or Unified Turn as the real integration target?** Not
   confirmed which is live-authoritative right now. Building against
   `chat_stance` without checking this risks building against the thing
   being replaced. Needs tracing before any code is written, independent of
   this feature.
3. **Notability/promotion filter, if ever revisited for musing-style
   surfacing**: parked, not designed — see Non-goals. Attention/reverie is
   not the target for this feature, so this question does not currently
   need an answer.

## Proposed schema / API changes

(Sketch only — not finalized pending Missing Questions 1–2.)

- New schema, e.g. `CommunityEventV1`: source URL, title, date/time window,
  location, category, raw excerpt, `scraped_at`. Registered in
  `orion/schemas/registry.py` alongside a new channel
  (e.g. `orion:community:local_events:*`), distinct from
  `orion:world_pulse:*` — deliberately not reusing the news channel/schema
  family per the Arsonist summary's rejection of option 1.
- New recall profile `community.local_events.v1.yaml` — SQL-timeline style
  retrieval scoped to this new store, similar in shape to
  `orion/recall/profiles/journal.world_pulse.grounded.v1.yaml` but without
  any digest-boilerplate filtering (there is no digest to filter).
- No new `OpenLoopV1` producer, no `SpontaneousThoughtV1` wiring, no
  `DriveEngine`/`tensions.py`/field-digester touches.

## Files likely to touch

- New: `services/orion-world-pulse/app/services/ingest/` — or a new small
  service, TBD by whether this stays inside orion-world-pulse or becomes
  its own `services/orion-community-pulse/` per the services-first
  boundary rule (single-source, single-page scrape argues for "new tiny
  service" over "new lane inside a heavyweight news pipeline," but this
  wasn't decided in conversation — flagging, not deciding here).
- `orion/schemas/registry.py`, `orion/bus/channels.yaml` — new schema/channel.
- `orion/recall/profiles/community.local_events.v1.yaml` — new.
- `services/orion-recall/app/intent.py` **or**
  `orion/cognition/verbs/chat_general.yaml` — depending on Missing
  Question 1's answer.
- `services/orion-cortex-exec/app/chat_stance.py`
  (`_project_recall_from_beliefs`) **or** the Unified Turn equivalent —
  depending on Missing Question 2's answer.
- Service scaffold (`.env_example`, `docker-compose.yml`,
  `requirements.txt`, `README.md`, `tests/`) if a new service is chosen.

## Non-goals

- No integration with `DriveEngine`, `tensions.py`, or field-digester —
  explicit, permanent, per Juniper's direction, not just out of scope for
  this patch.
- No `OpenLoopV1`/attention/reverie integration. If "Orion muses about
  community events" is wanted later, it needs its own separate,
  non-competing mechanism that bypasses the coalition/self-state-narration
  path entirely — not designed here, not a follow-up committed to here.
- No reuse of `orion-world-pulse`'s trust-tier/corroboration/claim-extraction
  machinery — a single community listings page has no competing sources to
  corroborate against.
- No daily-digest journal reflection trigger (`world_pulse_journal.py`'s
  pattern) for this content type.

## Acceptance checks

- A scraped event from theogdenite.com is retrievable via the new recall
  profile in isolation (direct profile call, not yet wired to live intent
  routing) — proves the ingest → schema → recall hop works before touching
  routing.
- A live chat turn that plausibly should surface a local event (e.g.
  Juniper asks "anything going on this weekend?") actually includes that
  content in the rendered `ChatStanceBrief`/prompt — proves the full hop
  chain, not just retrieval.
- No change in behavior for any existing recall profile, verb, or drive
  signal — proves the new seam is additive, not a mutation of shared
  routing/budget logic.

## Recommended next patch

Thin first slice, deliberately smaller than the full wiring chain above:
scraper + `CommunityEventV1` schema + one bus event + a direct-call recall
profile (`profile_explicit=True`, no intent-router change yet) — enough to
see real scraped data land in recall and be queried by hand, without
committing to Missing Questions 1–2 yet. Full chat-turn wiring (intent
routing, chat_stance/Unified Turn integration) is a second patch, gated on
Juniper's answers to those two questions.
