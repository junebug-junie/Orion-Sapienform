# Attention patch 4a: camera asks become graded acts

## Summary

Orion now keeps score of its camera questions. Every card Orion opens to ask Juniper about something the camera saw ("Is this Teddy?", "Same person?", "Whose vehicle is this?") becomes one act on the act ledger, and Juniper's reply is its grade. Cards still open exactly as before: same openers, same caps, same timing.

- Each camera ask Orion makes opens a `vision.ask` act. The verb is `ask`, the object is `orion_ask:<id>`, the correlation id is the ask id, and the horizon is the ask's expiry. sql-writer opens it at the start of each act-resolver tick (`app/ask_acts.py`).
- The act ledger's resolver grades it from the ask row (`orion/grammar/camera_ask_acts.py`, new signal `juniper_answered`):
  - Any answer (a name, yes, no, same, different) is 1.
  - A dismissal or "I don't know" is 0.
  - No answer by the time the card expires is 0. Juniper confirmed this on PR #3213 (question 3).
  - A card Orion withdraws closes as `expired` with an `abandoned:` note. This happens when a split undo removes the person, or when a name typed elsewhere makes the card moot. It counts as a grade nowhere: not in the market, not in the self-monitor (`failed_to_execute`), and not in the effect posterior.
- The ask's shadow want now carries `outcome_value`. This is how often Juniper answers asks of the same kind (rule) over 30 days, with a prior: (answered + 1) / (closed + 2). Vehicle asks score 0.13 and confirm asks 0.97.
- Juniper's own cards are excluded from every act and every ask statistic. These are the names she types on a walkway story.
  - Hub now marks them with the evidence ref `rule:juniper_story_naming`. Older rows are matched by `answer_detail.story_id`. Neither check reads the question text.
  - The marker goes in `evidence_refs`, not `source_ref`. `source_ref` is the individual id that the label-apply path, the open-ask unique index, the re-ask dedupe and split undo all read, so a marker there would break them.
  - Today the marker is belt and braces: every story card also carries `story_id`. Each clause is tested on its own.
- New back-fill script for 30 days of acts (dry run by default). New replay lane for 4b's acceptance gate 3.

## Outcome moved

The `vision.ask` family can now learn which questions Juniper can answer. Today the market ranks camera asks backwards. Live, 30 days: dismissed asks had a mean net of **0.931** and answered asks **0.472** (want_ledger joined to orion_ask, n=22 / 78). The highest-ranked asks are the vehicle cards she cannot answer. Once the family has 30 graded closes (on day one after the back-fill), the market ranks `salience x expected answer` instead of salience alone.

## Current architecture

- Five openers write cards to `orion_ask`: three in `app/vision_individuals.py`, plus `vision_handoff_loop.py` and `vision_office_sessions.py`.
- `ask_event_wants.py` posted a shadow `vision.ask` want 5 minutes after a card opened. Its rank rested on salience alone (the moving-sighting count), because the family had no acts and so was "ungraded".
- Juniper's own story-naming cards (`vision_individual` rows, inserted already answered) were in the salience history.

## Architecture touched

- `orion/schemas/grammar.py`: new `ActSignal` value `juniper_answered` (additive).
- `orion/grammar/camera_ask_acts.py` (new): the rule, the grade, the Juniper-card filter (SQL and Python), the per-rule Laplace prior, and the act shape. One copy, shared by the ledger, sql-writer, Hub's test and the replay.
- `orion/grammar/act_ledger.py`: the resolver closes `juniper_answered` acts from the `orion_ask` row. Adds `resolve_ask_acts` (used by the back-fill) and two summary counters.
- sql-writer: `app/ask_acts.py` (new; opens acts), `app/act_resolver_loop.py` (calls it before each resolve), `app/ask_event_wants.py` (outcome per rule; Juniper cards excluded), settings, env, compose, README.
- Hub `scripts/motion_story_routes.py`: the name action stamps the marker.
- `config/physiology/families.yaml`: comment only (the family is now graded).

## Files changed

- `orion/grammar/camera_ask_acts.py`: new shared rules, grade, filter and act shape.
- `orion/grammar/act_ledger.py`: resolver branch for `juniper_answered`, plus `resolve_ask_acts`.
- `orion/schemas/grammar.py`: `ActSignal` adds `juniper_answered`.
- `orion/physiology/event_wants.py`: `event_want(..., outcome_value=)`.
- `orion/physiology/appetite.py`: comment corrected (one event family is graded now).
- `services/orion-sql-writer/app/ask_acts.py`: new; opens the acts.
- `services/orion-sql-writer/app/act_resolver_loop.py`: opens ask acts at the start of each tick.
- `services/orion-sql-writer/app/ask_event_wants.py`: per-rule `outcome_value`; Juniper cards excluded.
- `services/orion-sql-writer/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: `SQL_WRITER_ACT_CAMERA_ASKS_ENABLED`.
- `services/orion-sql-writer/scripts/backfill_camera_ask_acts.py`: new back-fill (section 14).
- `services/orion-sql-writer/evals/camera_ask_acts_mutations.py`: new mutation check.
- `services/orion-sql-writer/tests/test_camera_ask_acts.py`, `test_camera_ask_acts_integration.py`, `test_ask_event_wants.py`, `conftest.py`: tests.
- `services/orion-hub/scripts/motion_story_routes.py`, `services/orion-hub/tests/test_motion_story_routes.py`: the marker.
- `services/orion-durable-runs/evals/event_wants_replay.py`: `--camera-asks` lane (4b gate).
- `config/physiology/families.yaml`: comment.
- `.github/workflows/orion-sql-writer-tests.yml`, `orion-durable-runs-tests.yml`: new integration test, mutation check and replay fixture in CI.

## Schema / bus / API changes

- Added: `ActSignal` value `juniper_answered`. New act-ledger actor `vision.ask` (verb `ask`).
- Removed: none.
- Renamed: none.
- Behaviour changed:
  - The resolver now also selects open `juniper_answered` acts every tick (about 10 open at a time) and reads one `orion_ask` row per act.
  - The `vision.ask` want carries `outcome_value`.
- Compatibility notes:
  - The new signal is never published on the bus. sql-writer applies the opens straight to `grammar_acts` with `open_act`, and the closes come from the resolver. No other service parses these acts.
  - `orion-sql-writer` was already a cataloged `orion:grammar:event` producer, so the catalog gate passes unchanged.

## Env/config changes

- Added keys: `SQL_WRITER_ACT_CAMERA_ASKS_ENABLED=true` (orion-sql-writer).
- Removed / renamed keys: none.
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (`+SQL_WRITER_ACT_CAMERA_ASKS_ENABLED='true'`).
- Skipped keys requiring operator action: none.

## Metric gate (CLAUDE.md 0A), checked against live data 2026-11-13

1. **Provenance.**
   - The grade comes from `orion_ask.status`, `answer_detail.dont_know` and `answered_at`.
   - Hub `ask_routes.py` writes them in `_close_ask` for `/answer` and `/dismiss`, which takes `{reason: dont_know}`.
   - The writer tick (`vision_individuals.py`) sets `expired` once `expires_at` passes. A split undo (`vision_split.py`) withdraws an ask before then.
2. **Independence.** Salience counts camera sightings. The outcome counts Juniper's replies. They come from different sources and different producers. They are anti-correlated, not redundant: the most-seen individuals are the vehicles she cannot name.
3. **Theory.** An ask is worth what its answer resolves, times the chance of an answer. A card she cannot answer resolves nothing. Outreach uses the same rule ("did Juniper reply", resolver-graded).
4. **Live data, 30 days, Orion-made asks only (n=248). Each rule's rate is answered / (answered + dismissed), and the Laplace value is (answered + 1) / (closed + 2):**

   | Rule | Answered | Dismissed | Other | Laplace value |
   |---|---|---|---|---|
   | confirm | 141 | 4 | — | 0.966 |
   | same | 41 | 0 | 5 open | 0.977 |
   | face | 5 | 0 | — | 0.857 |
   | office | 1 | 0 | — | 0.667 |
   | sighting (person) | 4 | 1 | 1 withdrawn | 0.714 |
   | vehicle | 5 | 40 (all "don't know") | — | 0.128 |

   - The value is not flat: it runs from 0.13 to 0.98.
   - It has a real rest point. A rule with no closes in 30 days returns to the prior 0.5. It does not pin at 0 or 1.
   - Time to an answer: median 3.7 h, p90 20.6 h, max 43 h. That is well inside the 7-day horizon, so the "unanswered at expiry" 0 is rare today: 0 of 248 asks.
5. **Existing mechanism.** Graded families, the resolver and the `outcome_value` field all existed. This adds no new score, only a signal and a reader.
6. **Reversible.** `SQL_WRITER_ACT_CAMERA_ASKS_ENABLED=false` stops new acts and drops `outcome_value` from the want. The back-fill report prints rollback SQL for its rows.

Corrections to the spec's numbers:
- There are 248 Orion-made camera asks in 30 days, not 413. The 413 seems to have included Juniper's 181 story-naming cards.
- Same-person asks: 41 answered and 0 dismissed. The spec's "2% dismissed" counted the 5 asks that are still open.

## Back-fill dry run (READ-ONLY, run 2026-11-13 00:52 UTC)

```text
python services/orion-sql-writer/scripts/backfill_camera_ask_acts.py --dsn postgresql://postgres:postgres@localhost:55432/conjourney
targets 248  by rule: confirm 145, same 46, vehicle 45, sighting 6, face 5, office 1
by status: answered 197, dismissed 45, withdrawn 1, open 5
would grade 1: 197   would grade 0: 45   would close withdrawn (not a grade): 1   stays open: 5
expected outcome by rule: confirm 0.966, same 0.977, face 0.857, office 0.667, sighting 0.714, vehicle 0.128
files: /tmp/camera-ask-acts-backfill/{dry_run.json, plan.csv, snapshot.jsonl}
```

Apply. Juniper runs this AFTER sql-writer is deployed. The live loop opens the last 24 h itself, and the back-fill skips any ask that already has an act:

```bash
cd /mnt/scripts/Orion-Sapienform && .venv/bin/python services/orion-sql-writer/scripts/backfill_camera_ask_acts.py --dsn postgresql://postgres:postgres@localhost:55432/conjourney --apply
```

The run writes `progress.log`, `report.md` (with rollback SQL) and `before_after.csv` to `/tmp/camera-ask-acts-backfill/`. It writes about 243 `grammar_acts` closes and the same number of `substrate_action_outcomes` rows.

## Replay lane for 4b (acceptance gate 3), live, 30 days

```text
PYTHONPATH=. .venv/bin/python services/orion-durable-runs/evals/event_wants_replay.py --camera-asks --live --days 30
asks 228 (20 earlier asks had too little history for a salience)
vehicle: held to window end 95.2% (gate >= 70%)        PASS
confirm: held at all 4.1%; same: 0%  (gate <= 10%)     PASS
answered per day 6.0 -> 5.77 under 4b, loss 3.9% (gate <= 10%)  PASS
first-bid intensity from hunger (not salience): 100%
also: face 2/2 and office 1/1 asks held to the end (all were answered) -- the lost answers
```

This is evidence for 4b, which is not built. 4b still waits on open question 2 below.

## Tests run

```text
cd services/orion-sql-writer && PYTHONPATH=../..:.:tests pytest -q tests/test_camera_ask_acts.py tests/test_ask_event_wants.py
  -> 27 passed
ORION_SQL_WRITER_TEST_DATABASE_URI=<throwaway pg16> pytest -q -m integration tests/test_act_ledger_integration.py \
  tests/test_act_reading_cost_integration.py tests/test_reading_energy_closure_integration.py \
  tests/test_act_workspace_binding_integration.py tests/test_camera_ask_acts_integration.py
  -> 57 passed (lifecycle: ask opens act -> answer closes 1; don't know closes 0; expiry closes 0;
     withdrawn closes expired; Juniper cards (marker, legacy, marker-only) open nothing; back-fill dry run reads only)
PYTHONPATH=.:services/orion-sql-writer pytest -q orion/grammar/tests orion/physiology/tests services/orion-sql-writer/tests \
  tests/test_act_ledger_contract.py tests/test_grammar_event_producer_catalog.py
  (+ orion/self_monitor) -> 1752 passed, 218 skipped (DB-only)
cd services/orion-hub && ORION_MEMORY_EPISODE_TEST_DATABASE_URL=<throwaway pg16> pytest -q tests/test_motion_story_routes.py
  -> 26 passed
python scripts/check_env_template_parity.py -> 0
```

## Mutation check

`services/orion-sql-writer/evals/camera_ask_acts_mutations.py` (also in CI). Twelve mutants, twelve killed:
- Each clause of the Juniper-card SQL filter (marker; legacy `story_id`).
- Each clause of the Python twin.
- answered -> 0; dismissed -> 1.
- Expiry not graded; open past expiry left open; withdrawn graded; stamped withdrawal graded; moot card counted as answered.
- The Laplace prior removed.

Each SQL mutant is killed by the Postgres lifecycle test, which seeds a marker-only card and a legacy card. A string check does not count as a kill.

## Evals run

```text
event_wants_replay.py --camera-asks            (fixture, CI)  -> all checks pass
event_wants_replay.py --camera-asks --live --days 30          -> gate 3 passes (numbers above)
event_wants_replay.py                           (fixture, unchanged lane) -> pass
```

## Docker/build/smoke checks

```text
No image build was run. The change is Python in already-copied paths (orion/, services/orion-sql-writer/app, scripts/),
no new dependency, and one new env key with a default in settings and compose.
Live path: UNVERIFIED until deploy. After deploy, check:
  docker logs orion-athena-sql-writer 2>&1 | grep -E "ASK_ACTS|ask_act_opened|ACT_RESOLVER"
  docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select outcome_source, outcome_value, count(*) from grammar_acts where actor='vision.ask' group by 1,2"
```

## Review findings fixed

The code review ran in a subagent with an explicit target (`git diff origin/main...HEAD`). It found no blockers.

- Finding: withdrawn and silent asks could only be told apart in real time. A back-fill or an outage would grade a split-undo withdrawal as Juniper's silence.
  - Fix: split undo now stamps `answered_at` when it withdraws an ask (`vision_split.py`). `ask_grade` reads that stamp. The clock sweep leaves it NULL.
  - Evidence: a unit test grades a stamped withdrawal 20 days later and still gets `expired`. New mutant "stamped withdrawal graded" is killed. EXPLAIN of the new UPDATE ran read-only on live.
- Finding: a withdrawal was "not a grade" only to appetite. It still wrote a 0 into the effect posterior, and the self-monitor counted it as refuted.
  - Fix: the note now starts `abandoned:` (self-monitor reads `failed_to_execute`), and `close_act` skips the outcome row and the posterior for `juniper_answered` + `expired`.
  - Evidence: the integration test asserts 0 outcome rows and posterior n 0 after a withdrawal. A unit test checks `act_outcome`.
- Finding: an ask with an unknown status would stay in the resolver's due set forever.
  - Fix: past horizon + 120 s it closes `expired` with note `abandoned: unknown_status:<s>`.
  - Evidence: unit test.
- Finding: the back-fill could report a false anomaly when the live resolver graded some acts first.
  - Fix: anomalies are counted from the final `grammar_acts` rows.
  - Evidence: the back-fill integration test.
- Finding: `_one_close` returned True even on `already_closed`, so the counters over-counted.
  - Fix: it returns `close_act(...).closed`.
- Finding: a card closed as moot by a name typed elsewhere (`answer_detail.resolved_by`) graded as Juniper's answer.
  - Fix: it now closes withdrawn (`expired`). 1 live row.
  - Evidence: unit test plus the mutant "moot card counts as answered", killed.
- Finding: the hunger basis change was unstated.
  - Fix: stated in families.yaml and under Risks below, and measured. The replay now reports `first_bid_intensity_from_hunger_share`.
- Findings (nits):
  - The settings comment for the flag-off case was wrong.
  - The 24 h catch-up limit was undocumented (now in the README and the settings comment).
  - The conftest comment was stale.
  - The mutation harness's in-place rewrite was undocumented (now in its docstring).
  - The older replay lane counted Juniper's cards (now filtered).
  - There was no pin test for the `rule:face`/`rule:office_session` refs (added).
  - The producer's `ASKS_SQL` never ran against Postgres (now it does).
  - Fix: all fixed.
- Not fixed: no late-answer integration case. Hub refuses an answer once `expires_at` has passed (`CLOSE_ASK_SQL`), and live has 0 answers later than their expiry. The existing `close_act` late rule is covered by `test_act_ledger_integration.py`.
- Note: the back-fill's closes are historically dated `substrate_action_outcomes` rows. Temporal Self's source reader sees them like any other act outcome.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull && scripts/safe_docker_build.sh orion-sql-writer up -d --build
cd /mnt/scripts/Orion-Sapienform && scripts/safe_docker_build.sh orion-hub up -d --build
```

The post-pull rebuild (`scripts/rebuild_services_from_git_diff.sh`) does both automatically. No migration.

## Risks / concerns

- Severity: medium. Juniper should decide this one.
  - Concern: once asks are acts, the family's hunger counts from its last close, not from each want's open time. This is the market's standard rule for every family with acts, reading included. In the live replay, 100% of first bids took their intensity from hunger, not from salience. So the sightings count no longer shapes a camera ask's rank: rank is about `hunger x expected answer`, and only the rule separates asks.
  - That still moves value the right way: vehicles are held 95% in the replay and confirm asks 4%. But it is not the spec's `salience x outcome`.
  - Mitigation, if unwanted: a small appetite change that keeps event-want hunger from `opened_at` for this family, with a test. I did not make it, because it changes the market's rule.

- Severity: medium.
  - Concern: this is "no behaviour change" for cards only. The shadow `vision.ask` want's rank does change once the family is graded. Prime reads funded event wants as concerns (`PRIME_CONCERNS_FROM_EVENT_WANTS`). So vehicle asks will often become `sated` and stop showing up as Prime concerns. That is the intended effect of 4a (acceptance check 2), but Prime will see it.
  - Mitigation: `SQL_WRITER_ACT_CAMERA_ASKS_ENABLED=false`.
- Severity: low.
  - Concern: a card Orion withdraws closes `expired`. That is not a grade for the market, but `close_act` still writes its 0 into the `vision.ask` effect posterior (L10), as it does for every expired close. That is 1 row in 30 days.
- Severity: low.
  - Concern: a withdrawal that the resolver first sees after the card's expiry looks the same as a time-out. It would be graded 0. This needs the resolver to be down for days.
- Severity: low.
  - Concern: `juniper_answered` closes do not feed the family-weight engagement source (b). That source counts only `juniper_replied`. Counting them would change tier-2 weights, which is a behaviour change outside 4a.
- Open question for 4b, UNDECIDED (Juniper): question 2 of PR #3213, a held ask at its window end (open it anyway, or cool it down for 30 days). Nothing in 4a depends on it.
- Juniper's answers already applied: question 1, no attention cost for asks (no `juniper_msgs`). Question 4, vehicles: the market decides, and the rule is not retired.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3277

🤖 Generated with [Claude Code](https://claude.com/claude-code)
