# Attention patch 4b: camera asks bid before the card opens, and obey the market

Design: PR #3213 (`docs/superpowers/specs/2026-11-12-attention-camera-asks-bind-design.md`, section 4b). Decisions: #3285 (a held ask keeps bidding, no deadline) and #3286 (rank stays family hunger x expected answer). Follows patch 4a (#3277).

## Summary

Orion now asks the body economy before it shows Juniper a camera question, and it does what the market says. The market is the system that decides which of Orion's wants get effort. Before this patch, the cards opened under fixed daily caps, and a shadow want arrived 5 minutes later that nothing obeyed.

- Every camera opener opens a card only through one bid site, `open_camera_ask` (`services/orion-sql-writer/app/ask_bind.py`). This covers same-person, face, the slow "seen N times" rule, "Is this <name>?", door/walkway handoff, and office.
- **Funded:** the card opens, stamped `want:<want_id>`.
- **Held:** no card. The candidate bids again later, after 600 s, doubling up to 3 h. There is no deadline. A held candidate that its opener stops proposing just drops; it was never shown.
- **No answer** (0.5 s timeout or market down): the card opens, as before.
- The 5-minute shadow loop posts no second want for a card that bid at open.
- The daily caps stay as a hard ceiling. Juniper's own walkway-story cards never bid.
- Flag `ORION_PHYSIOLOGY_BID_CAMERA_ASKS=on`. The value `shadow` restores the behaviour before 4b.

## Outcome moved

Orion stops showing Juniper the vehicle cards she cannot answer. Live replay, 30 days, with this rule:
- 95% of vehicle asks are held and never open within 7 days. Juniper dismissed 88% of these as "don't know".
- Confirm and same-person asks are almost never held: 4% and 0%. The 6 held confirm asks all opened 11 h later.
- Answered asks per day fall from 6.6 to 6.4, a loss of 3.0%.

## Current architecture

- Five openers inserted cards straight into `orion_ask` via `_insert_ask`, under per-opener daily caps.
- `ask_event_wants.py` posted a shadow `vision.ask` want about 5 minutes after a card opened. Nothing obeyed it. Prime read the funded ones as concerns.
- Patch 4a made each ask a graded act and put the rule's answered share on the want as `outcome_value`.

## Architecture touched

- sql-writer: a new bid site, all five openers, the shadow loop, settings, env, compose and README.
- `orion/physiology/binding.py`: new binding producer `orion-sql-writer.ask_events.binding`. The market now treats these bids as obeyed, so a Prime regulate charge applies (vision.ask declares no receptor today).
- `orion/grammar/camera_ask_acts.py`: the candidate key, the `want:` stamp and the backoff rule, shared with the replay.
- durable-runs: Prime's concern menu drops a funded ask want whose card never opened. The replay lane now models no-deadline holds.
- `config/physiology/families.yaml`: comment only.

## Files changed

- `services/orion-sql-writer/app/ask_bind.py` (new): `ask_bind`, `open_camera_ask`, `in_backoff`, the backoff map and a 12-bids-per-minute budget.
- `services/orion-sql-writer/app/vision_individuals.py`, `vision_handoff_loop.py`, `vision_office_sessions.py`: open through `open_camera_ask`. The slow rule checks `in_backoff` before it reads pictures. Person-confirm marks a prior name only after a real open.
- `services/orion-sql-writer/app/ask_event_wants.py`: skips a stamped card. It also skips an unstamped card when the binding producer's `want_ledger` row names it (`BID_AT_OPEN_SQL`).
- `services/orion-sql-writer/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: the flag.
- `orion/grammar/camera_ask_acts.py`: `candidate_key`, `rebid_after_sec`, `bid_want_ref`, constants.
- `orion/physiology/binding.py`: the producer is added to `BINDING_PRODUCERS`.
- `services/orion-durable-runs/app/event_wants_menu.py`: a missing card row counts as gone.
- `services/orion-durable-runs/evals/event_wants_replay.py`: the no-deadline replay, with a new check that vehicle asks stay unopened.
- `config/physiology/families.yaml`: comment.
- Tests: `services/orion-sql-writer/tests/test_ask_bind.py` (new), `test_ask_bind_integration.py` (new, Postgres), `conftest.py`, and `services/orion-durable-runs/tests/test_event_wants_menu.py`.
- `services/orion-sql-writer/evals/ask_bind_mutations.py` (new): 13 mutants in two lanes.
- `.github/workflows/orion-sql-writer-tests.yml`, `orion-durable-runs-tests.yml`: the new integration test and both mutation lanes.

## Schema / bus / API changes

- Added: want producer `orion-sql-writer.ask_events.binding`. Evidence ref `want:<want_id>` on cards that bid at open.
- Removed: none.
- Renamed: none.
- Behaviour changed:
  - Want ids are keyed by candidate (`camera_ask_candidate:<rule>:<kind>:<subject>`), not by ask id. `event_ref` stays `orion_ask:<ask_id>`, the card the bid would open, so Prime's menu can label and expire it.
  - Prime's menu treats a funded `orion_ask:` ref with no card row as gone.
- Compatibility notes:
  - No new channel or schema. The `want_ledger` rows reach sql-writer the same way as before.
  - The Hub card UI does not read `evidence_refs` (checked), so the stamp is invisible to Juniper.

## Env/config changes

- Added keys: `ORION_PHYSIOLOGY_BID_CAMERA_ASKS=on` (orion-sql-writer).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (`ORION_PHYSIOLOGY_BID_CAMERA_ASKS=on` in `services/orion-sql-writer/.env`).
- Skipped keys requiring operator action: none.

## Metric gate (CLAUDE.md 0A)

This patch wires no new metric. It obeys two values patch 4a already gated: salience and the per-rule expected answer. The one new number is the backoff cap.
- 1 h and 3 h open the same asks with the same delays in the live replay.
- 3 h makes 8.6 re-bids per held candidate per day, against 24.2 at 1 h.
- About 100 vehicle candidates are live at once, and `want_ledger` grows about 800 rows a day today.

## Replay gate (spec acceptance check 3), live, 30 days, read-only, no-deadline rule

```text
PYTHONPATH=. .venv/bin/python services/orion-durable-runs/evals/event_wants_replay.py --camera-asks --live --days 30
as_of 2026-11-13T20:32Z   asks 244 (Orion-made, Juniper's story cards excluded)
rule      asks  held_at_all  opened_later  never_opened_7d  later_delay_h  answered  expected_answer
confirm   153   3.9%         6             0%               11.2           148       0.949
same       40   0%           0             0%               -               40       0.963
vehicle    44   95.5%        0             95.5%            -                4       0.267
face        2   100%         1             50%              44.2             2       0.817
sighting    4   50%          0             50%              -                3       0.763
office      1   100%         0             100%             -                1       0.5
answered/day 6.6 -> 6.4 under 4b (loss 3.0%; lost: vehicle 3, face 1, office 1, sighting 1)
re-bids per held candidate-day 8.55; first-bid intensity from hunger 100%
checks: vehicle held >= 70% PASS (95.5%); vehicle never opened in 7 d >= 70% PASS (95.5%);
        confirm+same held <= 10% PASS (3.9% / 0%); answered/day loss <= 10% PASS (3.0%)
```

How "held" is read here: under the no-deadline rule, a held ask bids again later and may open later. It is not dropped at a window end. The replay follows each held ask for 7 days (the card's own life) with the live backoff. An answered ask not opened by then counts as a lost answer.

Stated approximations:
- A held ask is assumed to stay a candidate for the whole 7 days.
- An ask that opens later is assumed answered as it really was.
- The daily caps are not replayed.
- 14 vehicle asks were followed less than 7 days because the data ended.

## Tests run

```text
services/orion-sql-writer: pytest tests -> 1297 passed, 226 skipped
services/orion-durable-runs: pytest tests -> 931 passed, 144 skipped
orion/grammar/tests orion/physiology/tests tests/test_act_ledger_contract.py tests/test_grammar_event_producer_catalog.py -> 399 passed
ORION_SQL_WRITER_TEST_DATABASE_URI=<throwaway pg16> pytest -m integration (act ledger x5 + test_ask_bind_integration) -> 61 passed, 0 skipped
services/orion-hub/tests/test_outreach_bind.py services/orion-dream/tests/test_physiology_bind.py -> 55 passed
python scripts/check_env_template_parity.py -> PASS
```

Lifecycle coverage:
- Unit, decisions shaped from a live `want_ledger` row:
  - A funded ask opens with the stamp.
  - A held ask does not open. The backoff holds across 60 s ticks. A re-bid uses the same want and funds and opens later.
  - Ten days of sated answers never open the card (no deadline), at about 8 bids a day.
  - Timeout, market down and an exception all admit unstamped.
  - `shadow` does not bid. A Juniper card does not bid.
  - The bid budget holds. `in_backoff` reads the backoff correctly.
  - The shadow loop skips stamped cards and late-decided cards.
  - A static guard checks that only `ask_bind.py` calls `_insert_ask`, and that no raw camera-kind insert exists elsewhere.
- Postgres: the real `_open_sighting_asks` holds two vehicles and opens the person card.
  - On the next tick it does not re-bid the vehicles.
  - A later re-bid funds the vehicle, with the same want id and the first-seen `opened_at`, and opens it.
  - The daily cap still stops the third card.
  - `shadow` opens two cards first with no bid.
  - The shadow loop's SQL: one want per card, whether the card is stamped, decided late, or neither.

## Mutation check

`ORION_SQL_WRITER_TEST_DATABASE_URI=... python services/orion-sql-writer/evals/ask_bind_mutations.py`: 13 of 13 killed (also in CI, two lanes):
- held opens anyway
- held branch admits
- a deadline returns
- backoff ignored
- shadow loop posts a second want
- no stamp on a funded card
- stamp without a market answer
- bid budget removed
- shadow loop ignores a late-decided bid
- Juniper card bids
- shadow mode bids
- handoff opener skips the bid
- menu shows a card that never opened

## Evals run

```text
event_wants_replay.py --camera-asks            (fixture, CI)  -> all checks pass
event_wants_replay.py --camera-asks --live --days 30          -> gate passes (numbers above)
camera_ask_acts_mutations.py (4a, unchanged)                   -> all killed
```

## Docker/build/smoke checks

```text
docker compose ... -f services/orion-sql-writer/docker-compose.yml config -> ORION_PHYSIOLOGY_BID_CAMERA_ASKS: "on"
No image build: Python in already-copied paths, no new dependency.
Live path: UNVERIFIED until deploy. After deploy, check:
  docker logs orion-athena-sql-writer 2>&1 | grep camera_ask_bid
  psql ... -c "select outcome, reason, count(*) from want_ledger where producer='orion-sql-writer.ask_events.binding' group by 1,2"
  psql ... -c "select count(*) from orion_ask where evidence_refs::text like '%\"want:%' and created_at > now()-interval '1 day'"
```

## Review findings fixed

The code review ran in a subagent with an explicit target (`git diff origin/main...HEAD`). It found no blockers.

- Finding: a client timeout is not "the market did not decide". The market can record a funded row after the client gives up. The card then opens unstamped, and the shadow loop posts a second want for it.
  - Fix: the shadow loop also skips a card whose `orion_ask:<id>` is named by a binding-producer row (`BID_AT_OPEN_SQL`).
  - Evidence: a unit test and the Postgres test. Mutant "shadow loop ignores a late-decided bid" is killed.
- Finding: the bids run inside the opener's transaction, and a held candidate no longer stops the loop. After a restart (empty backoff map), one tick could bid about 130 candidates at up to 0.5 s each. Each candidate in backoff also paid for picture reads every tick.
  - Fix: at most 12 fresh bids per minute per process. A candidate over the budget waits one minute with no backoff step. The slow rule calls `in_backoff` before it reads pictures and passes the kind it already has.
  - Evidence: unit tests. Mutant "bid budget removed" is killed.
- Finding: the gate's "vehicle held" counted only the first bid. "Held once, opened 10 minutes later" would pass.
  - Fix: a new check, vehicle never opened in 7 days >= 70%.
  - Evidence: live value 95.5%.
- Finding: the replay left out the market's own wait.
  - Fix: it passes the sated wait that `bind_verdict` uses (600 s). Every replayed hold is sated, because the replay asserts that vision.ask has no charge. The numbers are unchanged.
- Finding: the static guard missed raw `INSERT INTO orion_ask` for camera kinds.
  - Fix: guard added. `person_record.py` (person_fact cards only) is pinned by name.
- Nits fixed with comments, not code:
  - The menu reads a card as gone while its opener's transaction is still open.
  - The cache refresh race.
  - The shared `same` key between the face and handoff openers.
  - `HELD_FORGET` is measured from `retry_at`.
- Nit fixed in code: the held log line now carries `history_n`, so a drift toward "no history, admit unbid" shows.
- Correction to the review: it said no held candidate was ever funded later in the replay. The rerun shows 6 confirm asks (after 11.2 h) and 1 face ask (after 44.2 h) opened later. Vehicles: 0.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull && scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

The post-pull rebuild (`scripts/rebuild_services_from_git_diff.sh`) does this automatically. durable-runs changed only in the menu read: `cd /mnt/scripts/Orion-Sapienform && scripts/safe_docker_build.sh orion-durable-runs up -d --build`. No migration.

## Risks / concerns

- Severity: medium.
  - Concern: rare rules are starved. A rule with few graded asks sits at its prior: office 1/1 is 0.667, below the family mean of about 0.82. It is held, so it is never graded, so it never moves. Office, face and people asks account for 3 of the 6 lost answers.
  - Mitigation: this is the market's rule (#3286), so it needs Juniper's choice. Filed #3350 (needs-juniper).
- Severity: low.
  - Concern: `want_ledger` has no retention. Held candidates add about 8.6 rows per candidate per day, and about 100 vehicle candidates are live, so about 850 rows a day on top of about 800 today. No held vehicle opened in the replay.
  - Mitigation: filed #3349.
- Severity: low.
  - Concern: the backoff map is in process memory. A restart re-bids every held candidate once, and the 12-per-minute budget spreads that over about 11 minutes.
- Severity: low.
  - Concern: Prime will see fewer camera concerns, because held vehicle asks are no longer funded wants. This is intended.
- Issues opened: #3349, #3350. Closed: none.

## PR link

(filled after `gh pr create`)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
