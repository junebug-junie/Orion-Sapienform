## Summary

Attention patch 3c: **reading obeys the market.** It follows the spec `docs/superpowers/specs/2026-11-07-attention-patch3-market-decides-design.md`, section 4, decided 2026-11-10.

- With `ORION_PHYSIOLOGY_BID_READING=on`, a read's stage 1 bids at its first seat request and obeys the answer, the same way curiosity does.
  - Funded, or a capacity deferral worth its price (`queued_at_pool`): it asks the GPU pool.
  - Deferred on price or sated: it is held, then bids again. After 2,900 s from its request it asks the pool anyway. It is never dropped.
- Stage 2 never bids. It carries stage 1's net (patch 3b).
- New obeyed producer: `orion-durable-runs.admission.reading.binding`, in `BINDING_PRODUCERS`.
- **Ships `shadow`** (no live change). Decided item 4 says a family binds only for a deliberate live week with an off switch. Switching to `on` is Juniper's call. In shadow, patch 3b's value-only bid stays exactly as it is.

## Outcome moved

- Reading becomes the third family to choose by value whether it runs now, after Prime and curiosity.
- **Replay C** (the acceptance 5 gate, live, last 14 days): PASS.
  - Read starts: 773.
  - Verdicts: `queued_at_pool` 499, funded 274.
  - **Held at all: 0. Held over 1 h: 0. Held to the 2,900 s end: 0.**
  - Reading's net: p10 0.636, median 0.798, p90 0.823.
  - Reading's share of Tier 2 acts started (7 d): **45.9%**. The floor is 5%.
  - So binding reading would not starve it or slow it. The live week checks this for real.

## Current architecture

- Reading (`reading.turn` durable runs) asked the pool for a seat with no market answer.
- Since 3b it makes a value-only bid at submit: logged, carrying its net to the pool, never obeyed.
- Curiosity investigate is the existing binding pattern: `RunBidder.bind` and `_economy_gate`.

## Architecture touched

- durable-runs:
  - `app/run_bids.py`: `reading_binds`, the reading branch in `bind`, `READING_BINDING_PRODUCER`, `READING_HOLD_END_SEC`.
  - `app/admission_runtime.py`: `make_bidder`, so the flag wiring is a single tested function.
  - `app/settings.py`, `.env_example`, `docker-compose.yml`: the flag.
- `orion/physiology/binding.py`: the producer joins `BINDING_PRODUCERS`.

## Files changed

- `services/orion-durable-runs/app/run_bids.py`: reading stage 1 binds when the flag is on.
- `services/orion-durable-runs/app/admission_runtime.py`: `make_bidder` builds the bid site from settings.
- `services/orion-durable-runs/app/settings.py`, `.env_example`, `docker-compose.yml`: `ORION_PHYSIOLOGY_BID_READING` (default `shadow`).
- `orion/physiology/binding.py`: the new producer is obeyed.
- `services/orion-durable-runs/tests/test_reading_binding.py` (new): 10 tests, including the runtime path and the flag wiring.

## Schema / bus / API changes

- Added: producer `orion-durable-runs.admission.reading.binding` (`want_ledger` rows once `on`).
- Removed: none.
- Renamed: none.
- Behavior changed: none while `shadow`. When `on`:
  - reading stage 1 is held on price or sated deferrals;
  - Prime's regulate charges would apply to this producer, but `hub.world_pulse_read` declares no receptors, so none do.
- Compatibility notes: none. No schema change.

## Env/config changes

- Added keys: `ORION_PHYSIOLOGY_BID_READING=shadow` (orion-durable-runs).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes, plus a compose fallback.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (`+ORION_PHYSIOLOGY_BID_READING='shadow'`).
- Skipped keys requiring operator action: none.
- **Flag default note:** the standing rule is "ship flags on". This flag ships `shadow` because Juniper's decided item 4 (2026-11-10) requires a deliberate live week. The spec says it "ships `shadow` until its live week starts".

## Tests run

```text
services/orion-durable-runs: pytest tests                     859 passed, 130 skipped
tests/test_reading_binding.py                                 10 passed
orion/physiology/tests                                        309 passed
scripts/check_env_template_parity.py                          PASS
scripts/check_env_key_single_source.py                        OK
git diff --check                                              clean
```

## Evals run

```text
services/orion-durable-runs/evals/reading_binding_replay.py --live --days 14: PASS
  773 read starts; queued_at_pool 499, funded 274; held 0; held > 1 h 0; held to hold end 0 (0.0%)
  reading net p10 0.636 p50 0.798 p90 0.823; reading share of Tier 2 acts started (7 d) 0.459 (floor 0.05)
  (replay C models exactly this rule: producer ...reading.binding, hold end 2,900 s)
```

## Docker/build/smoke checks

```text
No image build. Python only. Live path: none while shadow.
```

When Juniper starts the live week (`ORION_PHYSIOLOGY_BID_READING=on`, then restart durable-runs), run these daily checks from the spec:

```sql
SELECT outcome, reason, count(*) FROM want_ledger
 WHERE producer = 'orion-durable-runs.admission.reading.binding' AND decided_at > '<on time>' GROUP BY 1, 2;
-- reads started per day (expect 15 to 130)
SELECT date_trunc('day', opened_at), count(*) FROM grammar_acts
 WHERE actor = 'hub.world_pulse_read' AND opened_at > '<on time>' GROUP BY 1;
-- held more than 1 h: run.economy_held followed by run.economy_admitted more than 1 h later, at or under 5%
```

```bash
docker logs orion-athena-durable-runs 2>&1 | grep "physiology_bind run=reading-" | tail
```

## Review findings fixed

Code review ran in a subagent on `git diff origin/main...HEAD`. It found 0 blockers, 3 should-fix and 1 nit.

- **Finding: the 2,900 s hold end's stated reason was false.** The spec says it keeps a held read under Hub's 3,500 s claim reclaim. But Hub releases its claim on every tick while the run waits (`ReadingPending` leads to `release_claim`), so the reclaim never fires on a held read. Verified in `world_pulse_read_pipeline.py` and `queue.py`.
  - Fix: kept the decided value, and stated the real effect in code, docs and the test name. Held reads occupy Hub's 3 in-flight slots, so the cap bounds how long they can pause all reading: about 48 min, against the family's 3 h window. Raised for Juniper to re-decide (see Risks).
- **Finding: held reads occupy Hub's in-flight slots.**
  - Not changed. The spec chose this as the back-pressure ("while reads are held, Hub starts no new ones"). It is stated in Risks.
- **Finding: no test covered the runtime path or the flag wiring.**
  - Fix: `make_bidder` (the flag wiring) is tested for on, shadow and a typo. A runtime test covers a held read: no pool request while held, a re-bid on wake, and a pool request carrying its net. It also shows the hold end is fixed by the stored `requested_at`.
- **Finding (nit): the docs said "consent holds".** Reading carries no consent request.
  - Fix: removed from the docs.
- **Checked by the reviewer and fine:**
  - no duplicate attempts while held (the binding is reused, and `ReadingPending` costs no attempt);
  - the reservation lifecycle;
  - re-bids honour `retry_at`;
  - `BINDING_PRODUCERS` has no side effects for reading.

## Restart required

```text
No live change while shadow. The post-pull rebuild picks up orion-durable-runs.
Start the live week (Juniper): set ORION_PHYSIOLOGY_BID_READING=on in services/orion-durable-runs/.env, then
  scripts/safe_docker_build.sh orion-durable-runs up -d --build
Off switch: set it back to shadow, then restart the same way.
```

## Risks / concerns

- Severity: medium. **A decision for Juniper (decided default 6).**
  - Concern: 2,900 s was derived from Hub's reclaim, which turns out not to apply. The real trade-off:
    - a held read occupies one of Hub's 3 reading slots;
    - 3 held reads pause all new reads until one is admitted;
    - a short cap (2,900 s, about 48 min) limits that pause but overrides the market sooner;
    - the family window (10,800 s, 3 h) obeys the market longer but can pause reading for hours.
  - Recommendation: keep 2,900 s for the live week. Replay C held 0 reads at all, so the cap is unlikely to bind. Revisit it with live held-time data.
- Severity: low.
  - Concern: a newer, more valuable seed cannot bid while 3 held reads fill Hub's slots.
  - Mitigation: replay C shows 0 holds. The live-week check on held reads over 1 h (5% or less) catches it.
- Severity: low.
  - Concern: a retried attempt (after a failure) gets a new run id and a new 2,900 s window.
  - Mitigation: attempts are bounded by the read's max attempts.

## PR link

<link>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
