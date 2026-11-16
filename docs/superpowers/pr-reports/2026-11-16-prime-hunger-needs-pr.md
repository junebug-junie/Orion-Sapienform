# Prime hunger: needs with content wake Prime; the 6-hour quiet floor is deleted (integration patch 5)

## Summary

- Prime now thinks when something specific pulls at it, not on a 6-hour timer. Three **needs** each have a drive
  (0 to 1, computed by code) and a content line with a ref: **horizon** (a held concern, promise or unheld funded
  event want nears its end), **surprise** (another organ lived something improbable that nobody has used yet) and
  **connection** (Juniper has been silent longer than her own usual gap between conversations).
  (`orion/prime/hunger.py`)
- The strongest damped need wakes Prime at 0.5. Anti-swamp guards: a thought that changes nothing halves the needs
  it was shown (unless it answered that need), recovering over the need's own horizon; at most 24 thoughts in any
  24 h, after which only a real change wakes Prime; the same content line wakes Prime again only after its drive
  rose 0.1 from its lowest point since. All persisted in Prime's state.
- `QUIET_FLOOR` and the "quiet: nothing has woken me" wake are **deleted** (kill means kill; a git-grep test
  enforces it). The flag off means "changes only", not the floor.
- **The per-thought trace** (stable, for the daily roundup and the 14-day check): a need-woken thought's act close
  note starts `woke: need:<name> <ref>` (see "Trace contract" below).
- New read-only durable-runs `GET /temporal-self/hunger?organ=prime` (surprise and connection inputs) and an
  `include=` pin on `GET /temporal-self/now`, so a need's experience is in the shown row and can be cited.
- A live replay eval replaces the floor's wake replay: same 10 days, 2.9-3.5 thoughts a day instead of 5.0, no
  thought woken by time alone, every held concern still thought about within 18 h.

## Outcome moved

Before: 36 of Prime's 50 thoughts in 10 days (11-06..11-16) woke only because 6 hours passed, and most did nothing.
After (replayed on the same live data, upper bound since every eligible tick counts as a thought):

| policy | thoughts/day | woken by a change | by connection | by horizon | by time alone |
| --- | --- | --- | --- | --- | --- |
| live timer | 5.0 | 14 | - | - | 36 |
| idle (Prime never acts; damping on) | 2.9 | 14 | 12 | 3 | 0 |
| busy (Prime always acts; no damping) | 3.5 | 14 | 20 | 1 | 0 |

The real success metric (spec): at least 2 meaningful moves a day over 14 days with the empty-thought share at or
below 79%. That is a live 14-day check after deploy (UNVERIFIED; #3427 stays open for it).

## Current architecture (before)

`orion/prime/state.py` `wants_to_think` returned True only with a pending change; `observe` added a "quiet" change
after `QUIET_FLOOR` (6 h) without a thought. The market's "hunger" for Prime was a clock. Nothing told Prime what to
think about.

## Architecture touched

- `orion/prime/hunger.py` (new, pure): drives, guards, frame lines, trace.
- `orion/prime/state.py`: `QUIET_FLOOR` and the quiet wake deleted; `PrimeStateV1.needs` (`PrimeNeedsStateV1`:
  halvings, fired lines, thought starts), omitted from the saved state while empty; `wants_to_think(..., need_wake)`.
- `orion/prime/thought.py`: frame section WHAT IS PULLING AT YOU and its system-prompt paragraph.
- `services/orion-mind/app/prime_loop.py`: reads hunger inputs every `PRIME_HUNGER_READ_SEC`, computes needs every
  tick, need wake, bid `salience` (no `event_ref`), the trace, guard 1 at the thought's close, guards 4/5 at the open.
- `services/orion-durable-runs`: `GET /temporal-self/hunger` (read-only) and `include=` on `GET /temporal-self/now`
  (`ExperienceStore.by_id`, `hunger_inputs`; `build_now(pinned=...)` in `orion/temporal_self/now.py`).
- `orion/schemas/temporal_self.py` + registry: `TemporalHungerInputsV1`, `HungerSurpriseV1`, `HungerJuniperV1`.

## Trace contract (read by patch 4's daily roundup)

- Where: `grammar_acts.outcome_note` of Prime's thought acts (`actor = 'prime'`, `verb = 'infer'`). The same string
  is the `prime_thought` experience's label in `temporal_self_event`.
- Shape: `woke: need:<name> <ref>`, then ` (also: <changes>)` if a change was also pending, then
  ` | <what the thought did>`. `<name>` is one of `horizon`, `surprise`, `connection`. `<ref>` (at most 120 chars)
  never holds a space, `;`, `|` or a parenthesis: `prime_concern:<16 hex>`, an event want's `event_ref`,
  `experience:<event_id>` (surprise) or `chat_history_log:<id>` (connection: the message, stable for one silence).
- Parser: `orion.prime.hunger.parse_woke_need(note) -> (name, ref) | None` (regex `WOKE_NEED_RE`). None = the thought
  woke on a change (`woke: <changes>`, unchanged format).
- SQL: `substring(outcome_note from '^woke: need:(horizon|surprise|connection) ')`.
- Also in the log lines `prime_need_wake need=<name> ref=<ref>` and `prime_thought ... need=<name>:<ref>`.
- Patch 4's roundup is its consumer; the contract is posted on #3426.

## Metric quality gate (each drive)

**horizon**
1. Provenance: `PrimeConcernV1.last_evidence_at` / `opened_at` (written by `apply_answer` keep/add, `fold_turn_signals`),
   `concern_ttl`, `intention.horizon_for`; event wants' `decided_at`/`expires_at` from `GET /physiology/event-wants`.
2. Independence: Prime's own state and the market's funded list; no other need reads them.
3. Anchor: the spec's own definition (elapsed / horizon, counted from 0.5) and the Prime design's homeostatic drive:
   the need is reduced by the act that answers it (keep, drop, take up).
4. Live: 2026-11-16 09:36Z Prime held 3 concerns, all kept at 07:48 (1.8 h of 24 h; the promise 28 h of 72 h):
   drive 0. It returns to 0 every time Prime keeps a concern, which live Prime does at every thought ("kept 3").
5. Existing mechanism: replaces `QUIET_FLOOR` (deleted). Event wants: the floor was their only guaranteed reader.
6. Reversible: pure function; no schema default outside Prime's own state.

**surprise**
1. Provenance: `temporal_self_event.surprise_nats` (chronicle insert, `stored_payload_surprise`): forecast
   exceedances' `-ln P(observed)` and act-ledger outcome surprisal.
2. Independence: not Juniper's chat, not Prime's own state. Partial overlap with now salience for
   `prediction_surprise` (its salience magnitude is the same `surprise_nats`), so the need adds the "nobody used it"
   and "more unlikely than not" conditions, not a new measurement.
3. Anchor: surprisal (Shannon): `1 - exp(-nats)` = 1 - P(observed); the ln 2 floor means the outcome had P <= 1/2.
   Relative to its own source's 30-day p90, so a source that is always surprised is not always hungry.
4. Live (7 days to 11-16): 286 rows with surprise; `prediction_surprise` 224 rows all exactly 0.0123 nats
   (forecast 0%, not crossed: degenerate at its floor, so nothing is strictly above its p90); 5 rows above their
   own p90 (action_outcome up to 0.22 nats, reading_turn 0.00078), all under ln 2. **Drive 0 everywhere: the rest
   point is real and reachable**. A percentile rule alone would have fired on 10% of rows forever; the ln 2 floor
   is what lets it read calm. It has not fired once in 7 days: UNVERIFIED that it ever fires live.
5. Existing: `experience_use` (uncited check), MIN_SOURCE_HISTORY 20 (salience rule).
6. Reversible: one function and one read-only query.

**connection**
1. Provenance: `chat_history_log` rows with `source = 'hub_orion'` and a non-blank prompt (Hub's own chat).
2. Independence: no other need reads Juniper's chat. The "she asked and got no answer" half was not built:
   `chat_history_log` has 0 blank responses in 30 days across all sources (rows are written with the reply), so it
   cannot observe an unanswered prompt.
3. Anchor: her own empirical gap distribution (the same strictly-smaller share as salience): drive 0.5 at her median
   gap between conversations. Gaps under 2 h are pauses inside one conversation: live 145 of 185 gaps were under 2 h,
   the rest spread 2-61 h (bimodal).
4. Live: 40 gaps between conversations in 30 days, median 13.2 h. 09:36Z she last wrote 13.6 h ago: drive 0.525.
   It is 0 within 2 h of any message (rest point), and it rises in 1/40 steps, which is why guard 5 needs a 0.1 rise.
5. Existing: the chronicle's `chat_turn` experience (used for the citable ref).
6. Reversible: one function, one read-only query.

Not built (no producer exists; no placeholder shipped): `coherence` (#3523), `learning` (#3524), `project`
(comment on #3428, patch 6).

## Files changed

- `orion/prime/hunger.py`: new, the needs and guards.
- `orion/prime/state.py`: floor deleted, needs state, `wants_to_think(need_wake)`.
- `orion/prime/thought.py`: frame section and prompt paragraph.
- `services/orion-mind/app/prime_loop.py`: wiring (read, compute, wake, bid, trace, guards).
- `services/orion-mind/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: 4 keys, docs.
- `services/orion-durable-runs/app/main.py`, `app/temporal_self_store.py`, `README.md`: endpoint, pins, reads.
- `orion/temporal_self/now.py`: `build_now(pinned=)`, `NOW_MAX_PINNED`.
- `orion/schemas/temporal_self.py`, `orion/schemas/registry.py`: hunger inputs contract.
- `orion/prime/evals/run_prime_hunger_replay.py` + live fixture: new eval; `run_prime_wake_replay.py` + its fixture
  deleted (it measured the floor).
- `.github/workflows/orion-prime-tests.yml`: runs the new eval instead of the deleted one.
- `config/physiology/families.yaml`, `config/integration/organs.yaml`: comments (no new family, kind or channel).
- Tests: `orion/prime/tests/test_prime_hunger.py`, `services/orion-mind/tests/test_prime_hunger_loop.py`,
  `services/orion-durable-runs/tests/test_prime_hunger_inputs.py`, `orion/temporal_self/tests/test_now.py`; floor
  tests replaced in `test_prime_retries.py`, `test_prime_promise_concerns.py`, `test_prime_evals.py`; replay golden
  re-recorded (`test_prime_replay.py` docstring: 13 parts changed, 0 after mapping the quiet line and `need=-`).

## Schema / bus / API changes

- Added: `TemporalHungerInputsV1`, `HungerSurpriseV1`, `HungerJuniperV1` (registered; HTTP only, no channel);
  `GET /temporal-self/hunger?organ=<id>`; `include=` on `GET /temporal-self/now`; `PrimeStateV1.needs`.
- Removed: `orion.prime.state.QUIET_FLOOR`; the "quiet" pending change.
- Behavior changed: when a need wakes Prime, its bid carries `salience` = the need's damped drive (the market ranks
  it as an event want, intensity = max(clock, need)). No `event_ref`, so Prime's bid never appears in
  `GET /physiology/event-wants`. A change-woken bid without a need is a plain clock want, as before.
- Compatibility: `PrimeStateV1` is `extra="forbid"`; Hub (`orion/hub/turn_orchestrator.py`), the harness
  governor's Claude tools (`orion/claude_tools/readers.py`) and orion-mind's consent responder read
  `orion:prime:state` through `parse_prime_redis`: rebuild Hub and orion-harness-governor before orion-mind. `needs` is omitted while empty, so the state is byte-identical until Prime first
  thinks with hunger on. Rollback of orion-mind: drop the `needs` key from `orion:prime:state` first.
  durable-runs before orion-mind (else the hunger read 404s: surprise and connection are then unknown and shown as
  a gap; horizon still works). No migration (reads existing tables).

## Env/config changes

- Added keys (orion-mind): `PRIME_HUNGER_NEEDS_ENABLED=true`, `PRIME_HUNGER_THRESHOLD=0.5`,
  `PRIME_HUNGER_DAILY_BUDGET=24`, `PRIME_HUNGER_READ_SEC=300`.
- `.env_example` updated: yes. Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (all 4
  present in the primary checkout's `services/orion-mind/.env`). Skipped keys: none.

## Tests run

```text
pytest orion/prime orion/temporal_self -q              -> 596 passed
orion-mind: pytest tests -q                            -> 420 passed
orion-durable-runs: pytest tests -q                    -> 1003 passed, 149 skipped
pytest orion/mindloop orion/temporal_self test_replica_contracts -> 353 passed
static gates (settings defaults, schema registry, schema skew, personal terms, env sync) -> 89 passed; 7 passed
scripts/check_integration_seam.py                      -> OK (16 organs)
```

## Evals run

```text
run_prime_hunger_replay.py  6/6 PASS (report above; budget_holds max 4 need wakes/24 h on live data;
                            concerns_get_thought 18.0 h; never_falling_bounded 1 thought in 7 days;
                            ever_new_bounded: with no halving the budget alone binds at exactly 24/24 h every
                            day (refractory alone would allow 288); with halving 4/24 h; rest_point 0)
run_prime_frame_eval.py, run_prime_block_eval.py, run_intention_take_on_replay -> rc 0
```

Known replay biases: change wakes are the live non-floor wakes (from the old regime); held concerns come from the
live notes' `kept N`; funded event wants are not replayed (no history of them was pulled).

## Docker/build/smoke checks

```text
Live SQL of the new endpoint, read-only against the live DB: 82 ms (surprises), 4 ms (Juniper), 9 ms (event id).
Not deployed (task: do not merge or deploy). Live path UNVERIFIED.
```

## Review findings fixed

Reviewed by an independent subagent against `origin/main`. All material findings fixed:

- Finding (blocker): Prime's bid carried `event_ref prime_need:...`; `GET /physiology/event-wants` lists every want
  with an `event_ref`, so Prime's own funded bid came back as an unheld event, became a horizon need at 0.5 after
  22 minutes, with a new nested ref each time (guard 5 never held): a self-fed loop capped only by the budget.
  - Fix: the bid sends `salience` only, and only when a need woke Prime (also fixes the nit that any weak need moved
    every bid into event ranking). Backstop: `event_horizon_needs` skips family `prime` and the `prime_need:` prefix.
  - Evidence: `test_prime_own_thought_bid_is_never_an_event_need`, loop test asserts `want.event_ref is None`.
- Finding: the mixed wake note `woke: need:<name> <ref>; <changes>` made the parsed ref `...;`.
  - Fix: mixed form is `(also: <changes>)`; the ref group excludes space, `;`, `|`, parentheses.
  - Evidence: mixed-case tests in `test_prime_hunger.py` and `test_prime_hunger_loop.py`.
- Finding: guard 5 re-armed when a line merely left view (a surprise cut by the read's cap of 10; every event line
  when the event-wants read failed; the connection ref changing once the chat turn was ingested).
  - Fix: `track_troughs(events_read=, hunger_read=)`; surprises never re-arm by absence (fixed surprisal); the
    connection ref is the chat message, with the citable experience carried separately (`NeedV1.experience`).
  - Evidence: `test_guard5_rearms_from_the_trough_but_an_unread_source_never_rearms`, connection test.
- Finding: the waking need could be missing from the frame's top 5 and from the now pins; `include` was cut by
  characters.
  - Fix: `shown_needs(needs, woke)` puts it first; `include` joins whole ids up to `NOW_MAX_PINNED`.
  - Evidence: `test_the_need_that_woke_is_always_shown_first`.
- Finding: spec said "off restores the floor", contradicting "QUIET_FLOOR is deleted".
  - Fix: spec rollback line amended in this PR (off = changes only; a held concern can fade unthought).
- Finding: `ever_new_bounded` never exercised the budget (halving bound first); `no_floor_wakes` could not fail.
  - Fix: ever-new also runs with no halving and must hit exactly 24/24 h; the tautological check is dropped.
- Finding: rollout note named only Hub. Fix: governor added (above, and the restart line).
- Nits fixed: the trace prefix is bounded (ref <= 120, changes <= 100 chars) so the move stays inside the 500-char
  note; the frame says "could not be read" only when no read is in use; the trace contract posted on #3426.
- Not fixed, by design: the connection need's "asked and got no answer" half (no table can observe it; see gate).

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build && scripts/safe_docker_build.sh orion-harness-governor up -d --build && scripts/safe_docker_build.sh orion-durable-runs up -d --build && scripts/safe_docker_build.sh orion-mind up -d --build
```

Order matters: Hub and the governor (read Prime state), then durable-runs (serves the hunger read), then orion-mind. From the clean
primary `main` checkout after merge.

## Risks / concerns

- Severity: medium. Concern: the bid's new `salience` makes Prime an event want in the market; once Prime has 20 past
  event wants it ranks each need against them (strict below), which could defer Prime more often than today
  (funded 6/6 live). Mitigation: watch `prime_bid_not_funded` for a week; the flag turns it off.
- Severity: medium. Concern: surprise has not fired once in 7 days live (prediction_surprise is constant at its
  floor). It is correct at rest but unproven in motion. Mitigation: listed UNVERIFIED; the replay planted cases
  test the mechanism.
- Severity: low. Concern: the flag off now means changes only, so a held concern can fade with no thought (spec
  amended). Mitigation: ships on; restoring the floor is a revert.
- Severity: low. Concern: connection may wake Prime 1-2 times a day while Juniper is away; bounded by guard 5's
  0.1 rise, damping and the budget.
- Issues: opened #3523 (coherence need), #3524 (learning need); commented #3428 (project need). #3427 stays open
  for the 14-day live check (`Refs #3427`, not `Fixes`).

## PR link

(filled after push)
