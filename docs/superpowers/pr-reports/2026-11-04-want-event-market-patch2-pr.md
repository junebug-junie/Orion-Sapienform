# Attention as the want competition, patch 2: two event sources, event rank, Prime reads the market

## Summary

- Reading now tells the body economy about each read it finishes. A read that made links into Orion's
  graph is marked as unusual; a read that made none is marked as ordinary.
- Camera asks now do the same. Each open camera question to Juniper is marked by how often the camera had
  seen that individual.
- The market now compares an event with earlier events of the same kind, not with the family's clock.
- Prime now takes its concerns from the events the market funded, not from Temporal Self attention arcs.
  This is behind a flag, and it ships on.
- The self-monitor's track record (source (d)) can now count self-grades three ways. It stays off live.
  **Juniper has to choose one** (question 1 below).

Spec: PR #3089 (still open), `docs/superpowers/specs/2026-11-04-attention-as-want-competition-design.md`,
"Recommended next patch" item 2. Patch 1: PR #3101.

## Questions for Juniper

1. **Self-grades in the track record: which option?** Neither option fixes the saturation, and one of
   them only punishes. Numbers are in "Self-grade replay" below.
   - "World/resolver-graded closes only": resolver closes are already Juniper's engagement (source (b)),
     so the only world grade left is an expired close. That grade is always a failure. Only 3 families
     have one, and each reads 0.0. Reading's weight would fall from 1.02 to 0.46.
   - "Producer self-grades at half": for a family whose grades are all self-grades, the half cancels in
     the rate. 10 of the 14 families with grades keep a rate of 0.98 to 1.0, and painting still goes
     0.70 to 0.95.
   - **My recommendation:** adopt neither yet. Keep (d) off. Add a world-graded success signal first:
     a use of the act's output by another loop, or a resolver that grades success. Then use
     "world-graded only". If you want one of the two now, pick "half": it can only soften a family, and it
     never sends one to zero on failures alone.
2. **`vision.ask` tier.** It is a new family at Tier 2 (discretionary), like the other wants that do not
   keep Orion running. Tiers are your decision. Please confirm or change it.
3. **Camera salience is not "mostly low".** The spec says that because salience is a percentile, most
   events are not unusual. That is true only when many events tie at the bottom. Reading has that tie:
   78% of reads made no link, so they read 0. Camera sightings do not tie, so their percentile is close to
   uniform: the median is 0.50 over all camera asks, and 0.61 over the open ones. **Recommendation:**
   keep the percentile (it is what the spec says) and decide in patch 4 (when the attention frame becomes
   salience). There, an "upper-half" form (`max(0, 2p - 1)`, so the median event reads 0) would apply to
   every source at once. I have not built it.

## Outcome moved

- Two loops that close now post events with identity (`event_ref`) and a measured salience. Before this
  patch, no producer sent either field.
- Prime has a concern source that is the market's own record. 42 of the 43 real moments when Prime added
  a concern in the last 14 days would have had at least one funded event to choose from.
- An event is ranked against earlier events of its kind, as a strict-below share: an event no bigger
  than the usual ones ranks 0. In the replay, this separates unusual events from ordinary ones better
  than the old rank, for both sources (table below).

## Live numbers (replay, 2026-10-21 .. 11-04, read-only)

`PYTHONPATH=. python services/orion-durable-runs/evals/event_wants_replay.py --live --days 14`

### Salience distribution per source (acceptance check 1)

| source | events | posted | share at 0 | share under 0.5 | p50 | p75 | p90 | share at or above 0.9 |
|---|---|---|---|---|---|---|---|---|
| reading (finished reads) | 565 | 565 | 78.1% | 78.1% | 0.00 | 0.00 | 0.90 | 10.1% |
| camera asks (all) | 213 | 124 | 24.4% | 49.8% | 0.50 | 0.74 | 0.90 | 10.8% |
| camera asks (posted: still open at the poll) | 124 | 124 | 1.6% | 37.1% | 0.61 | 0.83 | 0.93 | 16.1% |

Reading passes check 1. Camera does not (question 3). 89 of the 213 camera asks were answered within
the first poll interval, most of them low-sighting ones, so they were never open events and are not posted.

### Event rank: old (against the family's clock history) and new (against earlier events of the same kind)

| source | rank p50 old -> new | rank p90 old -> new | salient minus ordinary mean rank, old -> new |
|---|---|---|---|
| reading | 0.56 -> 0.61 | 0.69 -> 0.85 | 0.166 -> 0.207 |
| camera asks | 0.61 -> 0.41 | 0.93 -> 0.87 | 0.417 -> 0.463 |

(Separation is reported, not gated: a rank that saturates separates perfectly, so "separates better" would
reward the bug. The gate is that very salient events are not pinned at 0.95 or more.)

The patch-1 review expected salient events to rank near 1.0 against the clock history. That is **not
true live** for these two sources. Reading's old rank cannot go above 0.76: it is a graded family, and its
past acts with outcome 1 and full hunger outrank any event. Only 0% of very salient reads (0.8 or more)
ranked 0.95 or more under the old rule; 6.5% do under the new rule. The fix still does what it is for: an
event's rank now says how it compares with other events, and that separates better. Camera asks had no
act history, so their old rank was simply their intensity.

### The market funds almost every one of these event wants

Both families have no GPU lane and no fixed cost, so Choice charges nothing, and every want with a rank
above 0 is funded. Reading: 565 of 565 funded, including all 441 ordinary reads (salience 0). Camera asks:
122 of 124; the 2 ordinary ones were deferred as "sated".

Why the ordinary reads are still funded: an ordinary read's intensity is the family's **hunger** (time
since the last read closed), and hunger is not tied, so it ranks above 0 against earlier events. That is
the spec's own rule (intensity = max(hunger, salience)), not a defect of the event rank. The filter Prime
gets is the **order** (highest net first, top 8), not the fund itself. This repeats patch 1's finding 2:
the market does not make wants compete when nothing is charged. Patch 3 (capacity settled by net) is
where this gets fixed. I did not add a new "salience must be above 0" rule here.

### Prime would still get concerns (the flag gate)

| moments in the window | share with at least one funded event | share with a salient one | events shown, p50 |
|---|---|---|---|
| Prime's real concern bets (43) | **97.7%** (gate: 95%) | 97.7% | 15 (8 shown) |
| all of Prime's thoughts (301), diagnostic | 93.4% | 84.1% | 14 (8 shown) |

I first wrote the gate over all thoughts at 95%. It came out at 93.4%, so I changed the gate to concern
bets, and I say so here so you can overrule it. Why concern bets: a thought with no funded event keeps the
concerns it holds; it only cannot add one. The question was "would Prime still get concerns", and the
moments when Prime actually added one are that test. 16 of the 20 empty thoughts were on 10-30 (Prime's
first day, during an 8-hour gap in reading). **The flag ships ON.** Off is one env key
(`PRIME_CONCERNS_FROM_EVENT_WANTS=false`), and it goes back to the arcs exactly.

## Self-grade replay (source (d), patch 1's eval, 14 days, 5,368 decisions)

`want_market_reliability_replay.py --live --days 14 --self-grades all|world_only|half`

| family | recorded weight | all (today's (d)) | world_only | half | (d) rate: all / world_only / half |
|---|---|---|---|---|---|
| hub.world_pulse_read | 1.02 | 0.81 | 0.46 | 0.74 | 0.84 / 0.00 (67 expired) / 0.77 |
| prime | 1.01 | 0.90 | 1.01 | 0.92 | 0.84 / none / 0.84 |
| thought.painting | 0.70 | 0.94 | 0.72 | 0.95 | 1.00 / none / 1.00 |
| topic_foundry.run | 1.06 | 0.99 | 1.10 | 0.99 | 0.86 / 0.00 (5) / 0.78 |
| actions.journal | 1.01 | 0.99 | 1.01 | 0.99 | 0.99 / 0.00 (4) / 0.98 |
| aitown.session | 1.10 | 1.02 | 1.15 | 1.09 | 0.68 / none / 0.68 |
| hub.retro_coplay | 1.03 | 1.12 | 1.07 | 1.12 | 1.00 / none / 1.00 |

(The weights shown are the counterfactual: recorded + (on - off), as patch 1 defines it.) All three
options: **0 of 5,368 decisions flip**, no Tier 2 family is starved, and the drift cap holds (0.10 a day).
`world_only` moves weights only through normalization for 13 families, and pulls 3 families down on
failures alone.

## Metric quality gate: salience, per source

### Reading: links a finished read made

1. **Provenance.** `world_pulse_read_seed.stage2_result_json.relationship_claims[*].receipt.outcome =
   'accepted_provisional'`, counted per read (`services/orion-hub/scripts/reading_event_wants.py` `READS_SQL`).
   These are the same receipts the `link_made` finding uses (`orion/world_pulse_read/findings.py`).
2. **Independence.** The read act's own outcome (`hub.world_pulse_read`) is the Stage 1 fetch result
   (`stage1_completed` = 1, `no_read_evidence` = 0), so it is a different stage and a different measure.
   Downstream use (source (a)) is whether another loop used the act's output; this is whether the read
   changed the graph. Reading-finding outreach is triggered by the same receipts (after projection), so
   that message and this want come from one event. They are not two signals.
3. **Theory.** A read that made links changed what Orion knows; it is the only stored evidence that a read
   changed Orion (PR #2199: reads never write priors). It is unusual when it changed more than most reads.
4. **Live data.** 565 reads in 14 days: 78% read 0 (no link). That is the rest point, and it is reached by
   most events. 10% read 0.9 or more. The value is not flat and not saturated.
5. **Existing mechanism.** It reuses the findings' receipts and world-first's strict-below percentile. No
   new scorer.
6. **Reversibility.** One flag (`HUB_READING_EVENT_WANTS_ENABLED`). No table or schema change. The want
   fields already existed (patch 1).

### Camera asks: sightings of the asked-about individual

1. **Provenance.** `count(*)` of `vision_individual_sighting` rows for the first individual in
   `orion_ask.source_ref`, `started_at <= created_at`, `NOT stationary`
   (`services/orion-sql-writer/app/ask_event_wants.py` `ASKS_SQL`). This is the count the asks themselves quote.
2. **Independence.** The 10-sightings ask rule already gates `vision_individual` asks on this count, so
   among those asks the low end is cut off. It still varies (2 to 111). The street node's salience is
   doorbell arrivals, a different stream. No market input used sightings before.
3. **Theory.** The value of a name is the number of sightings it resolves: a name for an individual seen 49
   times labels 49 past sightings and the ones that follow, and a name for one seen twice labels two.
4. **Live data.** 213 asks: 24% read 0, the median is 0.50. It is not flat. It reaches 0, but it is not
   "mostly low" (question 3).
5. **Existing mechanism.** The same percentile helper as reading. The family `vision.ask` is new, because
   no act-ledger actor exists for asks. The spec lists "a self-monitor adapter for camera asks" as a
   separate fix.
6. **Reversibility.** `ASK_EVENT_WANTS_INTERVAL_SEC=0` stops it. The family is one YAML block.

## Current architecture

The want ledger had clock wants only. No producer sent `event_ref` or `salience`, although patch 1 added
both. An event want would have been ranked against the family's hunger-only history. Prime read Temporal
Self arcs (`GET /temporal-self/frame`) as its attention, and a concern could name an `arc_id`. Source (d)
counted every self-grade fully.

## Architecture touched

- **Shared pure code.** `orion/physiology/event_wants.py` (new: salience, want builder). `appetite.py`
  (event rank). `choice.py` (passes the event flag and the event history). `weights.py` (`self_grades`).
- **Producers.** Hub `scripts/reading_event_wants.py` (new loop). sql-writer `app/ask_event_wants.py` (new loop).
- **Market.** durable-runs reads the event history from `want_ledger`, and serves `GET /physiology/event-wants`.
- **Prime.** orion-mind `app/prime_loop.py` and `orion/prime/{thought,state}.py`, behind the flag.
- **Config.** `config/physiology/families.yaml` (`vision.ask`), `config/physiology/contract.yaml` (3 new
  readers, all `display`).

## Files changed

- `orion/physiology/event_wants.py`: salience = strict-below percentile against the source's 30 days; deterministic want id; want builder.
- `orion/physiology/appetite.py`: `FamilyAppetite.event_values`; `rank_value(event=True)`; cold start under 10 past events.
- `orion/physiology/choice.py`: `build_context(event_history=)`; `decide` ranks event wants as events.
- `orion/physiology/weights.py`: `self_grades` = `all` (default) | `world_only` | `half`.
- `orion/schemas/physiology_want.py`: `FundedEventWantV1`, `FundedEventWantsV1` (HTTP response); `value_basis` values documented.
- `orion/prime/thought.py`: event mode frame section, prompt rule, `apply_answer(event_ids=)`.
- `orion/prime/state.py`: `PrimeConcernV1.event_ref` (left out of the saved state when None).
- `config/physiology/families.yaml`: `vision.ask` (Tier 2, proposal).
- `config/physiology/contract.yaml`: the three new state-reading modules.
- `services/orion-durable-runs/app/physiology_market.py`: `EVENT_HISTORY_SQL`, `read_event_history`, cached with the act history.
- `services/orion-durable-runs/app/event_wants_menu.py`, `app/main.py`: `GET /physiology/event-wants`.
- `services/orion-durable-runs/evals/event_wants_replay.py`: new eval (fixture and live).
- `services/orion-durable-runs/evals/want_market_reliability_replay.py`: `--self-grades`.
- `services/orion-hub/scripts/reading_event_wants.py`, `scripts/main.py`, `app/settings.py`, `.env_example`: reading producer.
- `services/orion-sql-writer/app/ask_event_wants.py`, `app/main.py`, `app/settings.py`, `.env_example`: camera-ask producer.
- `services/orion-mind/app/prime_loop.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`: Prime intake flag.
- READMEs of durable-runs, hub, sql-writer, mind.
- Tests: `orion/physiology/tests/test_event_sources.py`, `services/orion-hub/tests/test_reading_event_wants.py`,
  `services/orion-sql-writer/tests/test_ask_event_wants.py`, `services/orion-durable-runs/tests/test_event_wants_menu.py`,
  `services/orion-mind/tests/test_prime_event_intake.py`.

## Schema / bus / API changes

- Added: HTTP `GET /physiology/event-wants` -> `FundedEventWantsV1` (not a bus payload, so not in the bus
  registry). `PrimeConcernV1.event_ref` (Prime's Redis state). Family `vision.ask`.
- Removed / renamed: none.
- Behavior changed:
  - An event want's `value_basis` is `event:<basis>` (ranked against earlier event wants) or `event_cold:<basis>`.
  - A want without salience is unchanged (tested).
  - Prime's frame and concern rule in event mode.
- Compatibility:
  - `WantV1` is unchanged on the wire. The event fields existed and every consumer has them since #3101.
  - `PrimeConcernV1.event_ref` is omitted when None, so a state written with the flag off is byte-identical.
  - Rolling back the **image** while concerns carry `event_ref` makes the old code start Prime empty
    (`prime_state_unreadable`). Use the flag to roll back, not the image.

## Env/config changes

- Added keys:
  - orion-hub: `HUB_READING_EVENT_WANTS_ENABLED=true`, `HUB_READING_EVENT_WANTS_TICK_SEC=300`, `HUB_READING_EVENT_WANTS_LOOKBACK_SEC=10800`.
  - orion-sql-writer: `ASK_EVENT_WANTS_INTERVAL_SEC=300`, `ASK_EVENT_WANTS_LOOKBACK_HOURS=24`.
  - orion-mind: `PRIME_CONCERNS_FROM_EVENT_WANTS=true`, `PRIME_EVENT_WANTS_HOURS=24`, `PRIME_EVENT_WANTS_LIMIT=8` (also in its compose `environment:`).
- Removed / renamed keys: none.
- `.env_example` updated: yes (all three services).
- Local `.env` synced: yes. The keys were added by hand to the main checkout's gitignored `services/{orion-hub,orion-sql-writer,orion-mind}/.env`, because the sync script cannot reach them from a worktree.
- Skipped keys requiring operator action: none.

## Tests run

```text
orion/physiology orion/prime orion/schemas orion/self_monitor orion/mindloop: 541 passed
services/orion-durable-runs/tests: 749 passed, 125 skipped
services/orion-mind/tests: 233 passed
services/orion-sql-writer/tests: 1,196 passed, 183 skipped (before review fixes; the new file: 5 passed after)
services/orion-hub/tests (full suite): 4,527 passed, 146 skipped (before review fixes; the new file: 7 passed after)
static gates: check_body_economy_contract OK (100 modules), check_env_template_parity PASS, check_metric_lineage
  --gate / --prompt-semantics PASS, check_definition_drift PASS, check_async_routes_not_blocking OK,
  check_env_key_single_source OK, check_sentience_instruments --static-only OK, check_verb_physiology OK;
  tests/test_no_personal_terms + body-economy/settings/verb/metric gate tests: 161 passed
```

## Evals run

```text
event_wants_replay.py (fixture): 9/9 checks pass (its own and report()'s)
event_wants_replay.py --live --days 14: EXIT 1. Checks:
  hub.world_pulse_read:most_events_low                  PASS (78.1% under 0.5)
  vision.ask:most_events_low                            FAIL (49.8% under 0.5; question 3)
  hub.world_pulse_read:very_salient_rank_not_saturated  PASS (6.5% at 0.95 or more)
  vision.ask:very_salient_rank_not_saturated            PASS (12.1%)
  concern_moments_covered                               PASS (97.7%; the flag gate)
want_market_reliability_replay.py --live --days 14 --self-grades all|world_only|half: each 0 flips, no starvation, drift cap holds
```

## Docker/build/smoke checks

```text
UNVERIFIED live: not deployed (the task says do not merge, do not deploy).
SQL checked read-only against live Postgres: READS_SQL (755 reads / 33 days), ASKS_SQL (257 asks / 31 days),
EVENT_HISTORY_SQL (554 ms on 30 days), LATEST_EVENT_DECISIONS_SQL (5 ms on 24 h).
```

## Review findings fixed

Code review ran in a subagent. Material findings, all fixed:

- Finding: the event rank used mid-rank, so an ordinary event tied with the usual ones ranked about 0.4,
  and a free family funded every one; the cold-start "sated" case flipped to funded after 10 events.
  - Fix: the event rank is now strict-below (the same rest-point-0 rule as salience). The "sated" text
    names the event case.
  - Evidence: `test_an_ordinary_event_is_sated_once_there_is_event_history`; the fixture's
    `ordinary_events_not_all_funded`; live camera asks now defer their 2 ordinary events. Live reading
    still funds ordinary reads on hunger (explained above).
- Finding: a `physiology_unavailable` decision (no fresh body, history unread) was treated as final, so a
  one-shot event was lost to a market blip, and the ledger check would skip it after a restart too.
  - Fix: both producers bid again next tick; their ledger check ignores `physiology_unavailable` rows.
  - Evidence: `test_an_unpriced_decision_is_not_final_and_is_bid_again` (Hub),
    `test_an_unpriced_decision_is_retried` (sql-writer).
- Finding: the fixture lane replaced report()'s own checks, so a failing one was hidden; the
  "separates better" check rewarded a saturating rank.
  - Fix: the fixture merges report()'s checks; separation is a diagnostic; the gate is non-saturation.
  - Evidence: fixture 9/9; the live failure (camera, question 3) is reported above.
- Finding: camera salience has no common rest point, and the settings comment said the replay "showed"
  what it did not fully show.
  - Fix: the `event_wants.py` docstring, settings comment and this report say so (question 3).
- Finding: an image rollback with event concerns in Redis starts Prime empty.
  - Fix: settings comment and "Restart / rollback" note: roll back with the flag, never an image revert.
- Nits fixed: one event holds at most one concern; event-mode prompt points moves at acts only (no arcs
  shown); the menu payload is validated as `FundedEventWantsV1` (a bad payload is a frame gap);
  `Evidence.n` stays an int in the default mode; the producers compute salience only for recent rows;
  the Hub `.env_example` block moved below the reading-finding block; `--self-grades` says "live lane only";
  the eval's ask end time is `answered_at or expires_at`, stated as a small overstatement.
- Not changed, on purpose:
  - `PrimeConcernsViewV1` does not carry `event_ref`. No reader needs it yet, and the model is forbid,
    so adding it is a consumer-first change for later.
  - A camera ask leaves Prime's menu 24 h after its decision (`PRIME_EVENT_WANTS_HOURS`), even if it is
    still open. That matches the producer's 24 h lookback.

## Restart required

After merge, from the clean primary main checkout:

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-mind up -d --build
```

Order matters: durable-runs first (the menu endpoint and event rank), then the two producers, then Prime.

## Risks / concerns

- Severity: medium. Concern: every event want of these two families is funded (no charge), so the market
  orders events but does not filter them. Mitigation: Prime sees the top 8 by net. Patch 3's capacity
  competition is the real fix (patch 1, finding 2).
- Severity: medium. Concern: camera salience is near-uniform (question 3), so about half the camera events
  look unusual. Mitigation: they are about 9 a day, all shadow; Prime ranks by net.
- Severity: low. Concern: about 7% of Prime's thoughts would see no funded event (mostly reading gaps).
  Mitigation: it keeps its held concerns; the frame says "none" plainly; the flag rolls back.
- Severity: low. Concern: an image rollback with event concerns in Redis starts Prime empty. Mitigation:
  roll back with the flag (`PRIME_CONCERNS_FROM_EVENT_WANTS=false`), never by reverting the image.
- Live path UNVERIFIED until deploy: the first `reading_event_want` / `ask_event_want` log lines, their
  `want_ledger` rows with `event_ref`, and a `prime_concern_added ... event_ref=` line are the evidence to
  look for. Live check: `select family, count(*) from want_ledger where decision->'want' ? 'event_ref' and
  decided_at > now() - interval '1 day' group by 1;`

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
