# Binding arcs: Orion sees every body limit that binds (felt state, body economy Organ 5)

## Summary

- Until now Orion sensed heat only at the 34 °C emergency reflex (strain). Between the 31 °C soft line and 34 °C, the hot gate refuses GPU work and the heat price binds, but Orion's day record and chat cue said nothing. The same was true for full GPU lanes, Juniper's message cap and the electricity allowance.
- The physiology step (orion-durable-runs) now writes a `body_binding` row when a constraint's price starts binding, changes how it binds, keeps binding (every 30 min), or stops binding.
- The day story (Temporal Self chronicle) turns those rows into one `binding` arc per constraint. Several arcs can be open at once.
- Orion sees the arcs in the chat stance cue. With one constraint binding, the cue reads "Binding now: cabinet running hot (32.4 °C, line 31 °C) since 14:02 [arc:…]". With more, it reads "Binding now: cabinet running hot since 14:02 [arc:…]; my agent GPU lane full since 13:10 [arc:…] (out of my hands: Claude limit reached [arc:…])". The arcs are also in the Day letter, recent arcs and the citation truth check.
- This is awareness only. No price, gate, strain or arousal rule changed.
- Spec: `docs/superpowers/specs/2026-10-23-organism-body-economy-design.md`, Organ 5 "Where Orion sees it" (`binding:<constraint>`), acceptance A7 (the cue clause carries `[arc:]` tags that `day_citations` can cite).

## Outcome moved

The replay below runs the shipped code over the last 7 days of real `physiology_state`. It gives 89 binding episodes that Orion would have seen. Before this patch Orion saw 0 of them (only strain episodes reached the cue).

## Current architecture

- `orion/physiology/budget.py` `price_constraint` gives every constraint a `bind_state` each ~120 s tick: `slack`, `binding`, `exogenous_bind`, `stuck_at_ceiling` or `no_reading`. It is stored in `physiology_state`.
- The strain arc (#2817, #2860) is the only body state in the chronicle, and it follows the categorical arousal level.

## Architecture touched

- **orion-durable-runs**
  - `app/body_binding.py` (new, pure): the per-constraint rule and the row shape.
  - `temporal_self_graph._binding_step`: runs in the physiology node after `price_body`. Carry key `binding_episodes` is in `CARRIED_KEYS`.
  - `regulation_store.record_binding`: the insert.
  - `temporal_self_sources`: `SourceQuery("body_binding")`. The flag excludes the kind.
- **orion/temporal_self**
  - The adapter **keeps the row's own `event_id`** (a mutation test reproduces the #2817 runaway: 34 copies instead of 2).
  - Reducer `_fold_binding`.
  - Readers: stance cue, Day letter, recent arcs, citation truth.
- **orion-dream:** `binding` is in `ARC_EXCLUDED_KINDS`.

## The rule

"Bound" means `bind_state` is `binding`, `exogenous_bind` or `stuck_at_ceiling`:

- `binding`: forecast use is over the budget.
- `exogenous_bind`: over the line even with Orion's share at 0; the price is clamped.
- `stuck_at_ceiling`: the price is pinned at its ceiling for more than one horizon.

This is the same set the body report already prints as BINDING (`orion/physiology/report_text.py` `BIND_WORDS`). `slack` is not bound. `no_reading` is unknown.

| Row | When |
|---|---|
| onset | bound continuously for 240 s (anti-flap) |
| change | the bound state changed, at most once per 600 s |
| keepalive | every 1,800 s while bound |
| release | slack continuously for 240 s; the arc ends at the first slack reading |

- `no_reading` writes nothing and restarts any pending timer. An open arc closes by the reducer's 4,500 s stale timeout at its last row (`source_stale`), the same way strain does.
- A gap of more than 3 ticks between readings (3 × `TEMPORAL_SELF_TICK_SEC`) restarts a pending run.
- If an open episode goes unread (`no_reading` or a gap) and then reads bound again, the producer writes a keepalive marked `after_gap` (rate-limited 600 s). The reducer then starts a new segment, so the unread stretch is never counted as binding time.
- A release after a gap closes the arc at its last bound row (`source_stale`), because the real end is unknown.
- An episode whose last row is 4,500 s old starts fresh.
- A failed write keeps the old carry, so the next tick retries.
- Event id: `body_binding:<constraint>:<episode start>:<row>:<written at>`.

## Cue wording

- Named constraints are ranked by price / ceiling, at most 2; any more read "and N more".
- Numbers (level and line) appear only when one constraint is named.
- Energy's standing bind is named only when nothing else binds.
- Position: after company and play, before threads and sleep.
- Over the 400-char cap, the clause first shrinks to one named constraint, then drops.
- `exogenous_bind` constraints trail in "(out of my hands: …)".
- **Deviation from the brief:** energy's `exogenous_bind` is named last, never "out of my hands". Juniper ruled on 10-30 that it is the always-on body, not the room (`report_text.py` `ENERGY_EXOGENOUS_WORDS`).
- Every phrase says only what `bind_state` means. Energy and the Juniper cap bind on a forecast pace:
  - energy: "on pace to go over this month's electricity allowance"
  - energy, exogenous: "my always-on body alone on pace to reach …"
  - Juniper: "messages to Juniper over today's pace", until the cap is actually reached
- An exogenous lane is a seat with no live slot: "my <role> GPU lane has no live slot".
- With no arc open, an arc that closed in the last 6 h reads:
  - after a release: "Was binding 14:02 to 14:30: cabinet running hot [arc:…]"
  - after a stale close: "Last read binding at 14:30: …", with the same split in the Day letter. `ArcBindingV1.released` tells the two apart.

## Metric gate record

1. **Provenance:** `orion/physiology/budget.py` `price_constraint` → `PhysiologyPriceV1.bind_state`, written per tick to `physiology_state`.
2. **Independence:** none claimed. This is a rendering of the prices: a thresholded, debounced view of `bind_state`. It is not new signal. Heat mirrors the cabinet input strain uses; the other constraints are not in strain.
3. **Theory:** a Lagrange multiplier above zero means the constraint is active (binding). This patch shows which constraints are active.
4. **Live data:** last 7 days, read-only, 4,767 ticks, 10-27 08:11Z to 11-03 08:09Z (final run after the review fixes). Every constraint returns to slack except energy (see Risks).

| constraint | share of ticks bound | naive onsets | rows | arcs (closed how) |
|---|---|---|---|---|
| heat | 4.7% | 2 | 1 onset, 4 change, 13 keepalive, 1 release | 1 (process_ended) |
| lane:agent | 26% | 42 | 34 onset, 2 change, 63 keepalive, 34 release | 34 (process_ended) |
| lane:chat | 13% | 27 | 19 onset, 33 keepalive, 19 release | 19 (18 process_ended, 1 source_stale after an unread gap) |
| lane:agent-gpu2 | 5.8% | 26 | 19 onset, 7 keepalive, 19 release | 19 |
| lane:diffusion | 2% | 6 | 5 onset, 3 keepalive, 5 release | 5 |
| attention:juniper | 54% | 9 | 9 onset, 173 keepalive, 9 release | 9 |
| energy (exogenous) | 86% | 2 | 2 onset, 279 keepalive | 2 (1 source_stale across a no_reading stretch, 1 open) |

Total: 754 rows (~108 a day) and 89 arcs. The anti-flap rule cut naive onsets by about 25% on the lanes.

5. **Existing mechanism:** the strain arc is the template, and the body report wording is reused. No new price or state.
6. **Reversibility:** set `TEMPORAL_SELF_BINDING_ARCS_ENABLED=false`. That stops the rows, and open arcs stale-close. The rows are ordinary `temporal_self_event` rows and age out with retention.

## Files changed

- `orion/schemas/temporal_self.py`, `orion/schemas/registry.py`: source kind `body_binding`, arc kind `binding`, `ArcBindingV1`, `binding` on the arc and on its summary.
- `orion/temporal_self/{sources,arcs,frame,stance_cue,day_block,recent_arcs,citation_truth}.py`, README.
- `orion/temporal_self/tests/test_binding_arcs.py` (new, 21 tests), `evals/export_fixture_day.sql`.
- `services/orion-durable-runs/app/{body_binding (new),temporal_self_graph,temporal_self_driver,regulation_store,temporal_self_sources,main,settings}.py`.
- `services/orion-durable-runs/{.env_example,docker-compose.yml,README.md}`.
- `services/orion-durable-runs/tests/test_body_binding.py` (new, 17 tests incl. a Postgres round trip).
- `services/orion-durable-runs/evals/{body_binding_replay.py (new),temporal_self_fixture_db.py}`.
- `services/orion-dream/app/replay.py` and its test.
- `orion/inner_state_registry.py`: physiology.state.v1 changes from REHEARSAL to SHADOW and lists this reader. `config/metrics/metric_definitions.lock.json` re-locked after merging origin/main.
- `.github/workflows/orion-durable-runs-tests.yml`: the eval runs in CI.

## Schema / bus / API changes

- **Added:** `TemporalSourceKind` `body_binding`, `ArcKind` `binding`, `ArcBindingV1`, optional `binding` on `TemporalSelfArcV1` and `ArcSummaryV1`.
- **Removed / renamed:** none.
- **Compatibility:** these models forbid unknown fields, and every stored arc and frame now carries `binding`.
  - **Consumer-first deploy:** orion-thought and orion-hub before orion-durable-runs. Otherwise the stance cue reads `invalid_frame` and goes dark.
  - orion-dream reads columns, not the models.
- No bus channel. No migration: same table and columns.

## Env/config changes

- **Added:** `TEMPORAL_SELF_BINDING_ARCS_ENABLED=true` (orion-durable-runs), plus the compose passthrough.
- **Local `.env` synced:** yes, with `python scripts/sync_local_env_from_example.py` (+TEMPORAL_SELF_BINDING_ARCS_ENABLED='true').
- **Skipped keys:** none.

## Tests run

```text
orion/temporal_self orion/physiology orion/regulation           462 passed (before fixes); orion/temporal_self 272 incl. binding (after)
services/orion-durable-runs/tests (disposable Postgres 16)       760 passed, 1 skipped (after review fixes)
binding tests (orion/temporal_self + durable-runs, after fixes)   21 + 17 passed
orion-thought stance/recent-arcs/personality/self-readings        73 passed, 3 skipped
orion-hub temporal-self routes, day, curiosity fact, strain       77 passed
orion-dream test_replay_arc_candidates                            20 passed
orion/thought                                                     125 passed
tests/test_metric_definition_drift.py (after re-lock)             63 passed
Mutation checks: adapter makes a new id -> Postgres test fails (34 copies vs 2);
                 onset at 0 s -> body_binding_replay fixture fails
```

## Evals run

```text
python -P services/orion-durable-runs/evals/body_binding_replay.py             fixture passed
python -P services/orion-durable-runs/evals/body_binding_replay.py --live --days 7   passed (numbers above)
python orion/temporal_self/evals/run_arc_precision_eval.py                      passed
services/orion-durable-runs/evals/temporal_self_live_replay.py --step-sec 600   passed
orion/temporal_self/evals/run_play_cue_eval.py                                  PASS (6 of 6)
orion/temporal_self/evals/run_day_citation_truth_eval.py                        truth_rate 1.000
```

## Docker/build/smoke checks

```text
Not built here. Gates: check_inner_state_registry OK, check_definition_drift --gate PASS,
check_metric_lineage --gate PASS, check_env_template_parity PASS, check_env_key_single_source OK,
check_service_env_compose_parity orion-durable-runs OK, check_personal_terms OK,
check_body_economy_contract OK, check_sentience_instruments --static-only OK, git diff --check clean.
```

## Review findings fixed

The code review ran in a subagent and found no HIGH issues. Every finding is fixed.

- **Finding (MEDIUM):** a `no_reading` stretch shorter than the stale timeout was counted as binding time (dwell accrued across the gap).
  - **Fix:** the producer marks the next row `after_gap`. The reducer starts a new segment, and a release after a gap closes at the last bound row (`source_stale`).
  - **Evidence:** `test_an_unread_stretch_is_never_counted_as_binding_time`, `test_bound_again_after_an_unread_stretch_writes_a_flagged_keepalive_at_once`.
- **Finding (MEDIUM):** the binding clause (~250 chars) could push company, play, threads and sleep out of the 400-char cue.
  - **Fix:** numbers only when one constraint is named; energy's standing bind only when nothing else binds; the clause moved below company and play; over the cap it shrinks to one named constraint before it drops.
  - **Evidence:** `test_binding_fits_the_default_cap_beside_strain_and_play` (strain, play and binding together, 0 clauses dropped); `run_play_cue_eval` still 6 of 6.
- **Finding (MEDIUM):** some wording claimed more than the data shows.
  - **Problem:** energy and Juniper bind on a forecast pace, not on spend. An exogenous lane is a seat with no live slot. A stale close read like a real release.
  - **Fix:** pace wording; "has no live slot"; `ArcBindingV1.released`, so a stale close reads "Last read binding at …" (cue and Day letter).
  - **Evidence:** `test_energy_alone_is_named_as_my_always_on_body`, `test_juniper_words_say_pace_until_the_cap_is_reached`, `test_a_stale_closed_arc_reads_last_read_not_was_binding`.
- **Finding (LOW):** a release just after midnight could end before the carried arc began.
  - **Fix:** the end is clamped to the open segment's start.
  - **Evidence:** `test_a_release_just_after_midnight_never_ends_before_the_carried_arc_began`.
- **Finding (LOW):** a malformed open carry (no `began_at`) asserted on every tick, forever.
  - **Fix:** it resets, the same way a stale episode does.
  - **Evidence:** `test_a_malformed_open_carry_starts_over_instead_of_failing_every_tick`.
- **Finding (LOW):** the reading-gap limit was hardcoded at 360 s.
  - **Fix:** `binding_max_gap_sec` = 3 × `TEMPORAL_SELF_TICK_SEC`, the same rule arousal uses.
  - **Evidence:** `test_the_gap_limit_follows_the_tick_length`.
- **Finding (LOW):** the citation phrase missed "Binding now (out of my hands …".
  - **Fix:** the pattern is now `binding now ?[:(]`, plus `last read binding at`.
  - **Evidence:** `test_citation_phrases_cover_out_of_hands_only_and_last_read`.
- **Finding (NIT):** a vanished constraint stayed in the carry forever.
  - **Fix:** a stale open entry with no reading resets and is pruned.
  - **Evidence:** `test_a_vanished_constraint_leaves_the_carry_once_stale`.
- **Finding (NIT):** a start more than 6 days old was printed as an ambiguous weekday.
  - **Fix:** it is printed as a date ("Nov 01 00:00").

## Restart required

Consumer-first, from the clean primary `main` checkout after merge:

```bash
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-dream up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns

- **Severity: medium. Energy is a near-permanent line in the cue.**
  - **Concern:** energy read bound on 86% of ticks over the 7 days. Orion's always-on body alone exceeds the $100 monthly allowance, so the energy arc stays open almost all month. Every cue then carries an energy item.
  - **What the cue shows when energy is the only open arc:** "Binding now: my always-on body alone on pace to reach this month's electricity allowance since Sun 23:00 [arc:…]." This is about 130 chars, from the live replay at the end of the window. On most turns this month, that is the whole binding clause.
  - **What happens when other constraints bind too:** energy never takes a named slot while anything else binds. It is counted in "and N more".
  - **Cost:** it takes cue space that threads and sleep could use under the 400-char cap.
  - **Mitigation / follow-up:** if it proves to be noise, either name a standing energy bind only once a day, or move it to the Day letter only. Juniper's call. It is true, and per the 10-30 ruling it is Orion's own body, so it is not hidden here.
- **Severity: medium. Deploy order.**
  - **Concern:** deploying durable-runs before thought and hub darkens the cue.
  - **Mitigation:** the restart order above.
- **Severity: low. Short cap.**
  - **Concern:** on a busy turn (strain, a play session and an active conversation), threads and sleep drop before binding does.
  - **Mitigation:** binding shrinks to one named constraint first.
- **Severity: low. Midnight.**
  - **Concern:** the past-tense clause only sees today's arcs. An arc that ended at 23:30 is not named after midnight. Strain has the same limit.
- **UNVERIFIED until deploy.** Live checks:

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "SELECT event_id, occurred_at, payload_json->>'row' AS row, payload_json->>'bind_state' AS bind_state FROM temporal_self_event WHERE source_kind = 'body_binding' ORDER BY occurred_at DESC LIMIT 5"
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "SELECT arc_id, subject_ref, status, began_at, arc_json->'binding'->>'bind_state' AS bind_state FROM temporal_self_arc WHERE kind = 'binding' ORDER BY updated_at DESC LIMIT 5"
curl -s localhost:8124/temporal-self/frame | jq '.active_by_kind | with_entries(select(.key | startswith("binding:")))'
```

Expect a first `onset` row within about 4 min of the durable-runs restart for any constraint already bound (energy at least). Expect an open `binding` arc, and "Binding now:" in a live turn's stance cue (`cockpit_turn_sighting`).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3067

🤖 Generated with [Claude Code](https://claude.com/claude-code)
