# Chat stance arc cites become `experience_use` rows (integration patch 7d) -- design

Status: PROPOSAL. Juniper said yes to writing it on 2026-11-25. Nothing here is built.
Part of #3429. Evidence: the patch 7c replay, https://github.com/junebug-junie/Orion-Sapienform/issues/3429#issuecomment-7161391775.
Parent spec: `docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md` (Patch 7, "Retired").

## Arsonist summary

The old "was this used?" measure and the new one disagree (Jaccard 0.04) because the one place the old
measure looks, the chat stance citing an arc of Orion's day, never writes to the new table. Fix that one
gap: when the stance cites an arc, store the experiences behind that arc as uses by `chat`. A read-only
replay of the same week (below) shows this alone makes the new measure reproduce the old one, 51 of 51
used acts, Jaccard 1.00. Then the bid weight can switch tables, and only after that does the old loop stop.

Three things the brief had in a slightly different shape:

- Switch the weight **after** the replay passes, not before it. Switching first would zero two binding
  families' use rate (painting 0.18 -> 0.00, retro leisure 0.60 -> 0.00) if the chat site is late or broken.
- The new table links **more** families than the old one (world pulse reading, Prime). Switching tables
  must not silently give them a new weight source: keep the old family set at the switch.
- The 3-cite cap must not apply to the events behind an arc. At cap 3, 15 of 51 used acts are lost.
  Cap the expansion at 12 events per decision instead (lost nothing on the replay week).

## Current architecture

- **Old measure.** `services/orion-sql-writer/app/act_downstream_use.py`, a loop every 900 s
  (`ACT_DOWNSTREAM_USE_INTERVAL_SEC`, started at `services/orion-sql-writer/app/main.py:1180`). Per
  `grammar_acts` act: find its output row (direct ref, painting chain, chat row, retro session), then a
  `thought_decision` whose `day_citations.cited_valid[]` names `arc:<id>` whose **current**
  `arc_json.evidence_refs[]` holds that row, within 72 h. Writes `act_downstream_use`.
- **The chat stance.** `services/orion-thought/app/bus_listener.py:611-666`. It reads the day frame
  (`GET /temporal-self/frame`, `app/temporal_self_cue.py::build_stance_cue`), shows the model up to 3
  arcs, and `orion/thought/stance_react.py::gate_day_citations` keeps only arc cites it showed
  (`day_citations.cited_valid`). It never reads `GET /temporal-self/now` and never posts to
  `POST /temporal-self/now/use`. The day cue was shown on 1,425 of 1,466 stance decisions in the week.
- **New measure.** `experience_shown` (what a reader was shown) and `experience_use` (its valid cites),
  written by orion-durable-runs (`app/main.py:1384`, `orion/temporal_self/now.py::record_use`, max 3 per
  shown row, `ExperienceUseResultV1.uses` max_length 3). Live citing organs, last 7 days: reverie 668,
  curiosity 33, dream 13, prime 9, play 1. `chat` 0. There is no `chat_stance` organ; the `chat` organ in
  `config/integration/organs.yaml` has `reads_now: null` and an exemption ("reads the Prime block").
  The Prime block's now lines are only logged (`orion/hub/turn_orchestrator.py::_log_prime_citations`), never stored.
- **The live consumer.** Body economy weight source (a): `services/orion-durable-runs/app/physiology_market.py`
  `USES_SQL` reads `act_downstream_use` (30 days) into `orion/physiology/weights.py::evidence_for`
  (`use_rate` = used / linked, closed windows). The weight multiplies value rank in `choice.py` for
  every binding bid (curiosity, reading, dream, cooling, outreach, camera asks, experiments, retro watch).
- **Other `act_downstream_use` readers.** `scripts/analysis/{_body_economy_data,replay_body_economy,
  body_economy_shadow_vs_real,sample_body_economy_bids}.py`, `services/orion-durable-runs/evals/
  {want_market_reliability_replay,consent_gate_fixture}.py`. The rest are comments.

## The replay this design rests on (read-only, 2026-11-25)

SELECT only. Week = acts started 2026-11-16 00:00Z to 2026-11-23 00:00Z, closed windows; stance decisions
2026-11-16 to 2026-11-26 (so every act's 72 h window is covered).

1. **Arc evidence maps to events.** The 94 arc-citing decisions cite 110 arcs, 1,339 evidence refs.
   282 refs (165 distinct events) match a `temporal_self_event` exactly on `source_table || ':' || source_ref`.
   The other 1,057 are binding arcs' `physiology:<axis>:...:keepalive:...` refs (737) and attention arcs'
   `substrate_attention_broadcast_log` refs (277), which are not experiences. No act output lives there, so
   nothing the weight needs is lost.
2. **Every act output the old measure linked is an event.** All 133 linked acts' `output_ref` match a
   `temporal_self_event` row (painting 83, chat reply 34, retro leisure 10, dream 8).
3. **Simulated chat site vs old measure** (cites = events behind the cited arcs, round robin across arcs,
   newest first; use counts when after the act started and after the output, within the 72 h window):

   | expansion cap per decision | used acts found | overlap with old 51 | Jaccard | lost |
   |---|---|---|---|---|
   | none | 51 | 51 | 1.00 | - |
   | 12 | 51 | 51 | 1.00 | - |
   | 6 | 45 | 45 | 0.88 | 6 chat reply |
   | 3 | 36 | 36 | 0.71 | 15 chat reply |

   Caveat: this uses the arcs' current evidence, like the old measure. The built site stores events at
   decision time, which is more correct; the step-3 rule below accounts for the difference.
4. **Weight (a) per family, same week** (used / acts with an output event):

   | family (tier) | old | new, today | new + chat site | new + chat site, not own organ |
   |---|---|---|---|---|
   | thought.painting (2) | 15/83 = 0.18 | 0/83 = 0.00 | 15/83 = 0.18 | 0.18 |
   | hub.retro_leisure (2) | 6/10 = 0.60 | 0/10 = 0.00 | 6/10 = 0.60 | 0.60 |
   | dream.cycle (1) | 2/8 = 0.25 | 1/8 = 0.12 | 3/8 = 0.38 | 0.38 |
   | hub.chat_reply (0, never bids) | 28/34 = 0.82 | 4/34 = 0.12 | 28/34 = 0.82 | 4/34 = 0.12 |
   | hub.world_pulse_read | no source | 13/306 = 0.04 | 0.04 | 0.04 |
   | prime | no source | 12/62 = 0.19 | 0.19 | 0.19 |

   "new, today" is what a switch would read if made now. The two bottom rows link because their act row
   itself is an event (`temporal_self_event.source_table = 'grammar_acts'`, kinds `reading_turn`,
   `prime_thought`). With n = 306 against kappa 30, reading's 0.04 would dominate its target.

## Decisions (recommended)

1. **Chat is the citing organ.** `citing_organ = 'chat'`. No new `chat_stance` organ: the stance is the
   chat turn's decision step, and an organ name must belong to one producer set (organs.yaml rule). The
   Prime-block read stays as it is (logged, not stored); this patch does not touch it.
2. **One arc read = one `experience_shown` row.** Per stance decision whose day cue showed arcs, the
   server writes a shown row: `reader_organ='chat'`, `reader_run_ref='thought_decision:<correlation_id>'`,
   `event_ids` = events behind the **shown** arcs, resolved at that moment, `advances_cursor=false`,
   closed at once. Then `record_use` runs over the events behind the **cited** arcs. "Only a shown id
   counts" holds, because a cited arc is always a shown arc (`gate_day_citations` already enforces that).
   A decision that cites nothing still writes the closed shown row ("read and cited nothing").
3. **What counts as used: only `temporal_self_event` rows**, matched exactly on
   `source_table || ':' || source_ref` against the arc's `evidence_refs`. Not `related_refs` (those name
   inputs, not the experience), not fuzzy time matches for physiology refs.
4. **Caps.** No 3-arc cap: the cue shows up to 7 arcs (60-day max of `day_citations.shown`) and the stance
   cited 4-5 arcs in 10 of 143 citing decisions (2026-11-16 to 11-26); a 3-arc cap would drop real cites the gate
   already checked. The one cap is on the expansion: at most 12 events per decision (`ARC_USE_MAX_EVENTS`),
   round robin across cited arcs in cite order, newest first. Over-cap events are counted
   (`rejected_over_cap`) and logged, as today. The 3-cite cap for now readers does not change.
5. **Order: site -> replay -> weight switch -> kill.** Weight (a) switches only after the step-3 replay
   passes, and keeps the old family set (families with an old link path). Widening (a) to reading and
   Prime is a separate decision with its own replay. The old loop stops 7 days after the switch; the
   table is frozen, never dropped or deleted.

## Missing questions (defaults in brackets)

1. Does a chat cite of chat's own turn count for weight (a)? [Yes, for weight (a) only: it keeps the old
   meaning, and only `hub.chat_reply` is affected, which is Tier 0 and never bids. The integration metric
   still excludes self-use, as `use_rate` does today.]
2. Do arc-expanded uses count in the integration metric the same as a direct cite? [Yes, they count
   (the parent spec said the arc path becomes a site), but `citing_ref` carries the arc
   (`thought_decision:<corr>|arc:<id>`) and the coverage report shows "via arc" as its own line, so one arc
   cite is visible as one decision, not 12.]
3. A column (`experience_use.via_ref`) instead of encoding the arc in `citing_ref`? [No column. No migration,
   cheapest to reverse. Revisit if a second arc-like path appears.]
4. Should weight (a) gain reading and Prime once the switch lands? [Not in this arc. Ticket it with the
   numbers above; reading at 0.04 over 306 acts would pull its weight down hard.]
5. Expansion cap 12? [Yes. It lost nothing on the replay week; 6 lost 6 acts.]

## Proposed schema / API changes

- **Added:** `POST /temporal-self/arcs/use` on orion-durable-runs (HTTP only, no bus channel). Body
  `ArcCitesV1`: `reader_organ` (only `chat` accepted for now), `reader_run_ref`, `shown_arc_ids`
  (max 16), `cited_arc_ids` (each must be in `shown_arc_ids`, else 422). Response `ArcUseResultV1`:
  `shown_id`, `citing_organ`, `uses` (max 12, `ExperienceUseRowV1`), `arcs_without_events`,
  `rejected_over_cap`. Both models in `orion/schemas/temporal_self.py`, registered in `orion/schemas/registry.py`.
- **Behavior changed:** none for existing routes. `record_use` is reused with `max_cites=ARC_USE_MAX_EVENTS`.
  No table change.
- **Weight switch (later patch):** `physiology_market.USES_SQL` returns the same `UseRecord` rows from
  `grammar_acts` + the four old output link paths + `temporal_self_event` + `experience_use`, filtered to
  the old family set. `UseRecord` and `evidence_for` do not change.

## Files likely to touch

Patch 7d (this design's first patch):

- `orion/schemas/temporal_self.py`, `orion/schemas/registry.py`: `ArcCitesV1`, `ArcUseResultV1`, `ARC_USE_MAX_EVENTS`.
- `orion/temporal_self/now.py`: pure `arc_events(arcs, events)` (evidence refs -> event ids, round robin order).
- `services/orion-durable-runs/app/main.py`, `app/temporal_self_store.py`: the route; one read of the arcs'
  evidence and the matching events; reuse the shown-row insert and `record_cites`.
- `services/orion-thought/app/temporal_self_cue.py`: `post_arc_use(...)`, bounded, never raises.
- `services/orion-thought/app/bus_listener.py`: call it after `apply_stance_react_pipeline` when the cue showed arcs.
- `services/orion-thought/app/settings.py`, `.env_example`, local `.env`: `TEMPORAL_SELF_ARC_USE_ENABLED=true`.
- `config/integration/organs.yaml`: `chat.exempt_read` reason rewritten (below).
- `orion/integration/coverage.py`, `scripts/report_integration_coverage.py`: the "via arc" line.
- Tests in `orion/temporal_self/tests/`, `services/orion-durable-runs/tests/`, `services/orion-thought/tests/`.
- `orion/temporal_self/evals/run_arc_use_replay.py`: the step-3 replay as a script (the rule below).

Weight switch patch: `services/orion-durable-runs/app/physiology_market.py`, the two evals that read
`act_downstream_use`, `orion/physiology/weights.py` docstring. Kill patch: `services/orion-sql-writer/app/main.py`,
`app/act_downstream_use.py` (deleted), its settings key and `.env_example` line, `scripts/analysis/*` readers
(read the frozen table only for dates before the freeze).

## Integration gate

Chat emits already (`chat_turn`). For the read side, chat does **not** read `GET /temporal-self/now`, and
this patch does not pretend it does: `reads_now` stays `null` and `check_integration_seam.py` needs no change.
The exemption reason becomes: "the stance reads the day frame's arcs, not the now; its arc cites are recorded
as experience_use through POST /temporal-self/arcs/use (patch 7d). The Prime block's now lines are logged only."
A reviewer reads it. Making chat a real now reader is a separate question (it would add a second cue to the stance).

## Metric quality gate (this re-sources weight (a))

1. **Provenance.** `experience_use` rows with `citing_organ='chat'` come only from the new route, which
   calls `record_use` against a shown row it wrote in the same request. Arc cites come from
   `gate_day_citations` (code, not the model's claim).
2. **Independence.** Same quantity as `act_downstream_use` (a later decision cited the output), not a new
   signal; it replaces it. The other `experience_use` writers (reverie, curiosity, dream, prime, play) are
   different decision sites and add uses the old measure could not see (dream.cycle 2 -> 3).
3. **Theory anchor.** Same as the old measure and parent spec A4: an output is used when a later decision
   names it, checked against what that decision was shown.
4. **Live data.** Replay above: non-degenerate per-family rates (0.18, 0.60, 0.38). Rest state: no chat
   means no chat uses, a true zero, same known limit as the old measure ("gated on Juniper chatting").
5. **Existing mechanism.** Reuses `record_use`, `experience_shown`, `experience_use`, `day_citations`,
   `temporal_self_arc`. No new table.
6. **Reversibility.** Flag off stops new rows. Rows are tagged (`citing_organ='chat'`, `citing_ref` prefix
   `thought_decision:`), so a reader can exclude them without deleting anything. They age out with the
   30-day `experience_use` retention.

## Proposal mode

- **Capability change.** Orion's chat decisions start to count as using the experiences behind the parts
  of the day they cite. Painting, retro play and dream keep the use credit that today only the old table holds.
- **Data touched.** New rows in `experience_shown` (~1,400/week) and `experience_use` (~200-400/week).
  Reads of `temporal_self_arc` and `temporal_self_event`. No change to `thought_decision`.
- **Privacy boundary.** None new. The rows hold ids already in Orion's own tables; no message text.
- **Trace that proves it worked.** Log line `arc_use corr=<id> shown_arcs=N cited_arcs=M uses=K over_cap=J`
  in orion-thought, the route's own log in durable-runs, and
  `SELECT count(*) FROM experience_use WHERE citing_organ='chat' AND cited_at > now() - interval '1 day'` > 0
  on a day Juniper chats.
- **Dangerous failure mode.** Switching weight (a) before the chat site is live and proven: painting and
  retro leisure read 0 use and lose bids, silently, at up to 0.1 weight per day. Second: widening (a) to
  reading by accident (0.04 over 306 acts). Both are blocked by decision 5.
- **Disable / roll back.** `TEMPORAL_SELF_ARC_USE_ENABLED=false` in orion-thought (stops new rows; the
  new route should also return 404 when `TEMPORAL_SELF_NOW_ENABLED` is off, like the now routes). The weight switch carries
  `ORION_PHYSIOLOGY_USE_SOURCE=experience_use|act_downstream_use` until the kill patch, which removes the
  flag and the old option (kill means kill). The frozen table is never dropped or deleted.

## Step-3 replay rule (fixed now)

Run `orion/temporal_self/evals/run_arc_use_replay.py` on the first full UTC week (Monday 00:00Z to Monday
00:00Z) that starts after patch 7d is live, after every act's 72 h window has closed. The old loop is still
running then.

- A = acts with `act_downstream_use.used` true, started in the week, `window_closed`.
- B = acts started in the week whose output event (old four link paths, exact `source_table:source_ref`
  match) has an `experience_use` row with `citing_organ='chat'`, `cited_at` at or after both the act's
  start and the event's `occurred_at`, and at or before the act's `window_end`.
- **Pass** when all hold:
  1. |A| >= 20 (else the week is too quiet; take the next week).
  2. Jaccard(A, B) >= 0.90, and per family among painting, retro leisure, dream cycle: |B| >= |A| - 1.
  3. Every act in A - B or B - A is **explained** by arc rewriting: its output event is absent (A - B) or
     present (B - A) in the `experience_shown.event_ids` of the decisions in its window. An act lost to
     `rejected_over_cap` is **not** explained.
- Fail: nothing switches; post the diff on #3429.

## Non-goals

- Making the chat stance read `GET /temporal-self/now`.
- Storing the Prime block's now cites.
- Widening weight (a) to new families.
- Event links for binding or attention arcs (physiology keepalive refs, broadcast log refs).
- Any DROP, DELETE or TRUNCATE of `act_downstream_use`.

## Acceptance checks

- Unit: the pure resolver maps evidence refs to event ids exactly, ignores non-matching refs, orders round
  robin newest first, and stops at 12.
- Unit: the route refuses a cited arc that was not shown (422), writes one closed shown row with
  `advances_cursor=false`, and is idempotent per (shown_id, event_id).
- Unit: the stance posts once per decision with a shown cue, never when the cue was empty, and a post
  failure never fails the stance.
- Gates: `check_integration_seam.py`, schema registry check, env parity, `services/orion-thought/tests`,
  `services/orion-durable-runs/tests`.
- Eval: the replay script, run on the 2026-11-16 week now, must print the Jaccard 1.00 above (cap 12)
  with the current-arc caveat.
- Live: chat rows in `experience_use` the first day Juniper chats after deploy; the coverage report shows
  chat with cites and a "via arc" count.

## Recommended next patch

Patch 7d: the route, the stance post, the schema pair, the organs.yaml reason, the replay script. No change
to the bid market, no change to the sql-writer loop.
