# Attention as the want competition, patch 1: event wants and the self-monitor as evidence (pure)

## Summary

- A want can now say which event it is about (`event_ref`) and how unusual that event is (`salience`).
- An unusual event can win before its family's clock is due: intensity = max(hunger, salience).
- The market's learned family weights can read a fourth kind of evidence. This is the self-monitor's
  track record: the share of a family's acts that succeeded. It is off unless the caller asks.
- A 14-day replay over the real want ledger shows what that evidence would have changed.
- No live caller changes. With nothing new switched on, every decision is the same as today.

Spec: PR #3089, `docs/superpowers/specs/2026-11-04-attention-as-want-competition-design.md`.
Juniper's answers to its open questions were recorded there on 11-04.

## Outcome moved

The market can now carry events with identity and weigh families by whether their acts work. Both
are prerequisites for patch 2 (the first event sources) and patch 3 (binding curiosity and reading).

Replay, live, 14 days (2026-10-27..11-04, 5,171 decisions):

| family | recorded weight | with track record | track record (graded closes) |
|---|---|---|---|
| prime | 1.01 | 0.88 | 0.83 (248) |
| hub.world_pulse_read | 1.02 | 0.81 | 0.84 (692) |
| aitown.session | 1.10 | 1.00 | 0.68 (19) |
| topic_foundry.run | 1.06 | 0.97 | 0.88 (41) |
| thought.painting | 0.73 | 0.96 | 1.00 (116) |
| dream.cycle | 0.98 | 1.05 | 1.00 (23) |
| hub.outreach, hub.retro_leisure | unchanged | unchanged | none (Juniper-settled, or no graded closes) |

**0 of 5,171 decisions flip.** Only 172 weight-dependent decisions had any price charge, and all of
them stayed positive. No Tier 2 family drops under the 5% floor. Weights move at most 0.1 a day.

## Findings worth Juniper's attention

1. **The track record is nearly saturated.** 9 of 13 families read 0.98 to 1.0, because most closes
   are graded by the doer itself (`outcome_source = producer`). Averaged with downstream use, a
   self-graded 1.0 pulls a family up even when its output is rarely used: painting goes 0.73 → 0.96,
   although few of its outputs get used. **Recommendation:** keep it off live (as shipped). Before
   patch 3 turns it on, count only closes graded by the world or a resolver, or weight producer
   self-grades at half. Both are one-line changes in `reliability_grades`.
2. **Capacity is first come, first served, not a competition.** The real contention is
   `capacity_reserved`: 910 Prime bids and 838 outreach bids were deferred on capacity in 14 days.
   That check never looks at net value. So a higher-value want cannot push out a lower-value
   reservation, and no weight change can alter those outcomes. "Attention = competition" needs
   capacity to be settled by net. **Recommendation:** make this part of patch 3's design. A funded
   reservation that has not opened yet can be displaced by a want whose net beats it by a margin. It
   already lists `displaced`/`displaced_by`.
3. **Old weights can't be rebuilt exactly from today's tables.** Use windows that were open then are
   closed now. Closes arrived later. On 10-30 the family list gained `prime`, which changed the
   normalization. Recomputed "off" weights match the recorded weight within 0.02 for only 41% of
   decisions. So the replay's counterfactual is recorded weight + (on − off) on the same history.
   It does not use a recomputed weight. This is stated in the eval's docstring.

## Metric quality gate: source (d), self-monitor reliability

1. **Provenance.** `grammar_acts.outcome_value / outcome_source / outcome_note`, graded by
   `orion/self_monitor/adapters.py:act_outcome` (the same function the self-monitor uses).
2. **Independence.**
   - Source (b), Juniper engagement, is the resolver closes. Those are excluded from (d), so
     Juniper's reply is not counted twice.
   - Source (a), downstream use, is a different measure over the same acts. Each act now counts
     once toward `n`.
   - Appetite already uses outcome magnitude, but only as a rank *within* a family. Its
     cross-family effect is new.
3. **Theory.** A family's success rate is the expected return of funding it: the probability that
   the effort yields what it set out to make. The weight is that return's multiplier.
4. **Live data.** It is not flat: it ranges 0.68–1.0 across families. It can read low: prime
   0.83, pulled down by `prime.predict` at about 5%. But it is near-saturated for self-graded
   families (finding 1).
5. **Existing mechanism.** This reuses the self-monitor's grading and the weights' shrinkage, drift
   and rest rules. Nothing new was invented.
6. **Reversibility.** It is a keyword argument, default off. There is no schema or table change.

## Current architecture

The want ledger (`orion/physiology/choice.py`) bids clock-driven wants:
`net = family_weight × value_rank − Σ price × cost`. Intensity came only from hunger (time since
the family's last close). Family weights had two evidence sources: downstream use and Juniper
engagement. Wants carried no event identity.

## Architecture touched

- **Shared contract.** `WantV1` gains two optional fields. They are left out of the wire when unset,
  so consumers on the old schema (`extra=forbid`) keep parsing while services roll forward. Those
  consumers are sql-writer, Hub, dream, actions, cortex-orch, mind and durable-runs.
- **Pure organs.** Appetite, Choice and Weights in `orion/physiology/`.
- **Self-monitor.** `_act_outcome` is now public as `act_outcome`. It is the same function.

## Files changed

- `orion/schemas/physiology_want.py`: `event_ref`, `salience`, and a serializer that omits them when unset.
- `orion/physiology/appetite.py`: `want_intensity` = max(hunger, salience); `ActRecord.outcome_note`.
- `orion/physiology/choice.py`: uses `want_intensity`; `why` names the event salience when it wins.
- `orion/physiology/weights.py`: source (d) (`reliability_grades`, `reliability=` flag); `n` counts distinct acts.
- `orion/self_monitor/adapters.py`: `act_outcome` public.
- `orion/physiology/tests/test_event_wants.py`: 15 tests.
- `services/orion-durable-runs/evals/want_market_reliability_replay.py`: fixture and live replay.

## Schema / bus / API changes

- Added: `WantV1.event_ref` (str ≤ 500), `WantV1.salience` (0..1). Both are optional.
- Removed / renamed: none (`affordance` from the spec dropped: the want already names its act).
- Behavior changed: none for wants without salience. A want with salience can be funded on salience
  alone. No producer sends one yet.
- Compatibility: serialized wants are byte-identical when the new fields are unset. A test checks
  an old `extra=forbid` model against a real decision payload.

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: no.
- Local `.env` sync: not needed.

## Tests run

```text
pytest orion/physiology orion/self_monitor orion/schemas + durable-runs physiology tests: 476 passed, 2 skipped
orion/physiology/tests/test_event_wants.py: 15 passed
service suites touching WantV1 (sql-writer, durable-runs, dream, actions, cortex-orch, hub): 10/65/30/7/24/26 passed
tests/test_act_ledger_contract.py: 20 passed
```

## Evals run

```text
want_market_reliability_replay.py (fixture): all 5 checks pass
want_market_reliability_replay.py --live --days 14: no_tier2_family_starved, within_drift_cap pass; 0 flips / 5,171
```

## Docker/build/smoke checks

```text
None: no service code or runtime path changes. Live callers do not pass reliability=True and no producer sends salience.
```

## Review findings fixed

REVIEW_PLACEHOLDER

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low. Concern: a producer could start sending `salience` before its consumers are
  rebuilt. Mitigation: patch 2 adds the first producers, after this lands everywhere.
- Severity: medium (for patch 3). Concern: the track record is saturated by self-grades (finding 1).
  Mitigation: it stays off until that is fixed.

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
