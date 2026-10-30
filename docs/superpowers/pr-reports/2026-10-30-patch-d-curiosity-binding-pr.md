# Patch D: curiosity investigate obeys its bid

## Summary

- Curiosity investigate runs now ask the body economy before they ask the GPU pool for a seat, and they obey the answer. This is the first durable-run family whose bid binds (Prime was the first bid of any kind).
- The rule: funded runs go to the pool. A run deferred because the lane is full, but still worth its price, also goes to the pool, because the pool's own queue is the queue. A run deferred on price, with no hunger, or with a value at or below zero is held and bids again later.
- A held run is never dropped. It bids again at the market's `retry_after_sec` (clamped to 2 minutes .. 1 hour). After 6 hours (the family's deferrable window) it goes to the pool anyway.
- No answer from the market (down, stale body state, history unread, timeout) means the run goes to the pool as it does today. This patch removes no other gate, so "no answer" falls back to the old gates. It never falls back to "funded by the economy", and it never falls back to "held forever".
- Every verdict is a run event (`run.economy_held` / `run.economy_admitted`) that carries the market's own reason: net, charge, binding constraint, lane and the `why` sentence. This is the channel later "slow down" signals from Prime will act through.
- Flag `ORION_PHYSIOLOGY_BID_CURIOSITY_INVESTIGATE=on`. Set it to `shadow` and restart for an instant rollback to the C3a shadow bid.

## Outcome moved

### The live evidence, before any change (7 days to 2026-10-30 18:38 UTC)

Live shadow decisions in `want_ledger` (the bid site has been live since 10-28):

| | bids | funded | deferred | ran anyway | agreement |
|---|---|---|---|---|---|
| durable.curiosity_investigate | 21 | 3 | 18 (all `capacity_reserved`) | 21 of 21 | 14% (3 / 21) |

Binding constraints of the 18 deferrals: `lane:agent` 12, `lane:agent-gpu2` 6. Median wait from bid to the act opening: 660 s.

Replay of all 46 real curiosity admissions over the full 7 days (live Choice code against the `physiology_state` row in force at each admission): 3 funded, 43 deferred (all `capacity_reserved`: `lane:agent` 36, `lane:agent-gpu2` 7). All 46 ran. Agreement 7% (3 / 45 in the table). Median wait to open 3,318 s.

So every disagreement is the same one: the market says "the lane is full", and the run then waits in the pool's queue and runs. The disagreement is about a queue, not about a veto.

### Why the lane always reads full

- Curiosity's measured cost (p90 GPU slot-seconds) is longer than one hour, so the market caps it at one whole lane window. On the 1-slot agent lane it fits only when that lane was idle for the whole past hour. On the 2-slot agent-gpu2 lane it fits only when the lane was at most half busy.
- The agent lane was idle for a whole hour in 0-89 of about 720 two-minute ticks per day. Agent-gpu2 had room for one curiosity run in 0-235 ticks per day.
- Over the 7 days, curiosity itself used 55% of the agent lane's slot time, self-sense eval 23%, and self-inquiry 20%. Three Tier 2 durable families fill about 98% of the lane.
- Curiosity's cost is measured (there is history in `grammar_acts.cost_json`), so it is NOT hit by the "unmeasured family is priced at one whole window" problem Prime had. It is priced at one window because its runs really are that long.

### The stop check: what "every deferral holds" would have done

Before shipping, the obvious design (hold on every deferral, capacity included) was replayed with re-bids, and it failed the stop condition:

| rule | runs held | held to the 6 h limit | median extra delay past the real start |
|---|---|---|---|
| hold on capacity, real lane levels | 41 / 46 | 32 | 4.6 h |
| hold on capacity, curiosity's own use removed from the lanes (optimistic) | 41 / 46 | 15 | 2.0 h |
| **shipped: capacity deferral worth its price goes to the pool queue** | **2 / 46** | **0** | **one run, 766 s** |

"Hold on capacity" would delay almost every curiosity run by hours. Most runs would be released only by the 6-hour timer, not by the market. Its lanes would also be given to self-inquiry and self-sense eval, which still bid in shadow. That would silence curiosity in favour of its siblings, not slow the body down. So it was not shipped.

The shipped rule treats a capacity deferral as queueing at the GPU pool. The pool already holds a run until a real slot frees, so nothing runs past real capacity. The rule holds only when the market says the run is not worth its price now. In the 7-day replay that happened twice: both times the family had no hunger yet on a full lane (net 0.0, right after the previous curiosity run closed). Those two runs were held for 1 and 2 hours. One of them started 766 s later than it really did.

**What this means today.** Lane prices on the agent lanes are near 0, and energy, heat and Claude are not priced yet. So binding changes almost nothing today. It holds only on "no hunger yet". When prices rise (a hot cabinet, Prime asking the body to slow down), held runs become real: a price deferral holds the run and records why.

## Current architecture

- `services/orion-durable-runs/app/run_bids.py` (Patch C3a) placed a SHADOW bid at submit for every durable want family. The bid ran as a background task, reserved from the act's open, and was never obeyed.
- `AdmissionRuntime.register` asked the GPU pool for a hold right away. The run then waited in the pool's queue.
- Hub's curiosity scheduler keeps its own `arousal_strained` veto (`services/orion-hub/scripts/curiosity_investigation.py`).

## Architecture touched

- `run_bids.py`: new pure rule `bind_verdict`, `hold_end` and `binding_family`, and new `RunBidder.bind` (awaited, never raises). `on_submit` skips the shadow bid for a run whose bid binds, so each run has one bid. `on_grant` and `on_terminal` now follow the run when the shadow flag is off but binding is on.
- `admission_runtime.py`: `register` calls `_economy_gate` only for a run that has never asked the pool (`hold_seq == 0` and no request id). A held run returns `waiting_resource` with `hold.retry_at` and `hold_seq` still 0. The existing `_hold_ready` backoff wakes the run at `retry_at`, and `resource_request` bids again. A run that has asked the pool once is never gated again (takebacks and requeues are the pool's business).
- The market itself (`orion/physiology/choice.py`, `physiology_market.py`) is unchanged.

## Files changed

- `services/orion-durable-runs/app/run_bids.py`: binding rule and binding bid.
- `services/orion-durable-runs/app/admission_runtime.py`: the gate in `register`, and the run events.
- `services/orion-durable-runs/app/settings.py`, `.env_example`, `docker-compose.yml`: the flag, shipped `on`.
- `config/physiology/contract.yaml`: `run_bids.py` added to `protected_paths` (the design says to add each producer's bid site when its Patch D lands). It is not a `readers` row, because it reads no regulation or price state itself. It obeys a decision, and the checker sees no read. The replay script is registered as a `display` reader.
- `scripts/analysis/body_economy_binding_replay.py`: a new read-only replay. It re-bids each real run of a binding family until it would be admitted, with three rules side by side.
- `services/orion-durable-runs/evals/curiosity_binding_replay.py`: the eval (a fixture lane runs in CI, and a `--live` lane runs on the live database).
- `services/orion-durable-runs/tests/test_run_bids_binding.py`, `scripts/analysis/tests/test_body_economy_binding_replay.py`: tests.
- `.github/workflows/orion-durable-runs-tests.yml`: runs the eval and the replay tests.

## Schema / bus / API changes

- Added: run events `run.economy_held` and `run.economy_admitted` on the existing `durable_resource_events` / `ResourceEventV1` (`event` is a free string; detail is a dict). There is no new schema or channel.
- Removed: none.
- Renamed: none.
- Behavior changed: curiosity investigate runs bid at their first `resource_request` instead of at submit, and they obey. Their bids still publish to `want_ledger` with `shadow=true`, because `WantDecisionV1.shadow` is `Literal[True]`. Prime's binding bids have the same limitation.
- Compatibility notes: widening `shadow` to `bool` is a consumer-first migration (sql-writer validates with `WantDecisionV1`), so it is left for a follow-up. The run events and the `obeyed=true` log line are the record of obedience until then.

## Env/config changes

- Added keys: `ORION_PHYSIOLOGY_BID_CURIOSITY_INVESTIGATE=on` (orion-durable-runs).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. `services/orion-durable-runs/.env` in the primary checkout now has `ORION_PHYSIOLOGY_BID_CURIOSITY_INVESTIGATE=on`. The key is covered by the existing `ORION_PHYSIOLOGY_` prefix.
- skipped keys requiring operator action: none.

## Tests run

```text
cd services/orion-durable-runs && ORION_ADMISSION_TEST_DSN=<disposable postgres:16> pytest tests -q
725 passed, 1 skipped   (includes the 2 real-Postgres binding tests: held run never asks the pool,
                          records why, runs once funded; flag shadow leaves admission as before)
pytest scripts/analysis/tests/test_body_economy_binding_replay.py scripts/analysis/tests/test_body_economy_shadow_vs_real.py
7 passed
python scripts/check_body_economy_contract.py
body-economy contract: OK (91 state-reading modules, all registered)
orion-static-gates.yml steps, run locally: STATIC_GATES_RESULT
```

## Evals run

```text
python services/orion-durable-runs/evals/curiosity_binding_replay.py            (fixture, CI)
fixture: 6/6 streams as expected; saturated lane: shipped holds 0 s, not-shipped holds 21600 s
PASS
python services/orion-durable-runs/evals/curiosity_binding_replay.py --live --days 7
live: 46 runs; shipped rule held 2 (1 longer than 1 h)
PASS   (gate: at most 10% of runs held longer than 1 h; none held past the window)
```

## Docker/build/smoke checks

```text
No image build run: the change is Python only, and the Dockerfile already copies app/ and config/physiology.
Not deployed (task rule).
```

## Review findings fixed

REVIEW_FINDINGS

## Restart required

After merge, from the clean primary `main` checkout synced with `origin/main`:

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Rollback without a code change: set `ORION_PHYSIOLOGY_BID_CURIOSITY_INVESTIGATE=shadow` in `services/orion-durable-runs/.env`, then run the same command.

## Risks / concerns

- Severity: medium. Concern: the live path is UNVERIFIED. No curiosity run has bid in binding mode on the live rail yet. Mitigation: after the deploy, check `docker logs orion-athena-durable-runs | grep physiology_bind` for one `verdict=` line per curiosity run. Also check `select event, payload->'detail'->>'verdict' from durable_resource_events where event like 'run.economy_%' order by generated_at desc limit 20;`.
- Severity: medium. Concern: the old Hub curiosity veto (`arousal_strained`) is still in place. Mitigation: this follows the design rule. An old gate is deleted only after its reader has run `on` for 7 days with the acceptance checks green. That deletion is the next PR.
- Severity: low. Concern: `want_ledger` rows for binding bids still say `shadow=true`, so the shadow-vs-real report cannot tell obeyed bids apart. Mitigation: join to the `run.economy_*` events by `decision_id`. A follow-up can widen `WantDecisionV1.shadow` consumer-first.
- Severity: low. Concern: the capacity line prices a curiosity run at a whole lane hour. The binding rule works around this by sending capacity deferrals to the pool queue, but the market's capacity check stays a weak model for long durable runs. Mitigation: placement and real-slot capacity are Patch E.

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
