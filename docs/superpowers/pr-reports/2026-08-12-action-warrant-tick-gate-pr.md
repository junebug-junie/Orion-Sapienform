# Gate the proposal arena on `action_warrant` — O1 moves

Date: 2026-08-12
Branch: `feat/action-warrant-signal` (follow-on to PR #1567, which shipped the signal)

## Summary

- Wires `action_warrant` in as a **tick-level gate** in `orion/proposals/builder.py`. The warrant
  decides **whether** this tick's state justifies acting; the existing per-template scoring still
  decides **which**.
- **O1 moves.** Measured over 10,542 real ticks through the real builder.
- Retracts this repo's own **Patch B** recommendation, which merged to main in PR #1554 and is wrong.
- Two test fixtures corrected: they claimed to be "loaded" fields while carrying no precision
  baseline at all.

## Outcome moved

| | before (measured live) | after (replay, 10,542 real ticks) |
| --- | --- | --- |
| dispatched per tick | exactly 5, always | mean **1.841**, range 0–5 |
| coefficient of variation | 0.022 | **1.310** (60x) |
| ticks with zero dispatch | 0.00% | **63.19%** |
| ticks at the ceiling | 100.00% | 36.81%, bounded by `max_dispatches_per_tick` |

O1 in its own words: the budget *rises and falls with real internal pressure* ✓, is *not a flat
per-cycle allowance* ✓, *with a demonstrated, verified ceiling* ✓.

## Why this worked when five deletions did not

`min_priority: 0.10` was a cut on an absolute pressure whose measured floor was **0.3035**. It could
never bind, and never did. A threshold is only meaningful on a scale that has a rest point.
`action_warrant` is a combined tail probability, so **0.5 is a median normal day for this machine by
construction** — the threshold becomes a decision about tolerance rather than a guess about scale.

## Architecture touched

`build_proposal_frame` gains one gate, applied **after** `external_candidates` merge — deliberately.
Reverie and cognitive-hop producers write `priority_score` straight from their own salience,
bypassing `proposal_priority()`, so no per-candidate threshold could gate them consistently. A
tick-level gate is producer-agnostic by construction, which **dissolves that scale collision for
gating purposes**.

A closed gate records *why* (`warranted` / `below_threshold` / `no_live_dimensions`) on the frame and
**suppresses rather than drops** candidates. "Orion was calm" and "the signal broke" must not look
identical from the outside.

## Files changed

- `orion/proposals/builder.py` — the gate
- `orion/proposals/policy.py` — `thresholds.action_warrant_min`, with its definitional provenance
- `config/proposals/proposal_policy.v1.yaml` — `action_warrant_min: 0.50` + provenance block
- `orion/schemas/proposal_frame.py` — `action_warrant`, `action_warrant_dimensions`,
  `action_warrant_gate` (all optional, backward compatible)
- `tests/test_proposal_frame_builder.py` — 5 new gate tests; fixture corrected
- `tests/test_proposal_transport_readonly_candidates.py` — fixture corrected
- `docs/superpowers/specs/2026-08-11-proposal-arena-rate-coupling-design.md` — Patch B retraction

## Schema / bus / API changes

- **Added:** three optional fields on `ProposalFrameV1`. `None`/`[]` defaults mean frames persisted
  before this load unchanged; `action_warrant=None` is deliberately distinguishable from a real 0.0.
- Removed / renamed: none. No table, column, or channel change (`proposal_frame_json` is a blob).

## Env/config changes

- Added keys: `thresholds.action_warrant_min` in `config/proposals/proposal_policy.v1.yaml` (**not**
  an env key)
- Added / removed / renamed **env** keys: none. `.env_example` untouched, no `.env` sync needed.

## The Patch B retraction

This document previously recommended Patch B as "the one that can actually move O1". Replay of 5,276
real ticks through the live scoring functions:

| variant | % ticks proposing | avg candidates | avg dispatched |
| --- | --- | --- | --- |
| current | 100.00% | 12.00 | 5.00 |
| **B only** | **100.00%** | **12.00** | **5.00** |
| C only | 100.00% | 7.83 | 5.00 |
| B + C | 98.92% | 2.83 | 2.83 |

`base_priority` (0.20–0.42) clears `min_priority` (0.10) on its own, with urgency **0.0** and
confidence **0.0**. Patch B is a no-op, and **C was its prerequisite, not its follow-up** — the
dependency was inverted. B+C together would also have re-created the monoculture they were meant to
remove (the two `resource_pressure` templates capture 96.6% of ticks): the *third* inversion of the
same monoculture.

## Tests run

```text
$ PYTHONPATH=. pytest tests/test_proposal_frame_builder.py tests/test_proposal_policy_loader.py \
    tests/test_proposal_transport_readonly_candidates.py tests/test_proposal_scoring.py \
    tests/test_action_warrant.py tests/test_feedback_extractors.py \
    tests/test_field_channel_glossary.py tests/test_execution_dispatch_envelopes.py \
    services/orion-field-digester/tests
256 passed in 5.80s
```

Two fixtures carried **no** `dimension_precision_*` state, so the gate correctly read them as "cannot
tell" and returned empty frames. They passed before only because nothing had ever asked the field
whether its state warranted acting — `base_priority` carried every candidate over `min_priority`
regardless of what the field said. Both now carry real elevated baselines, matching what they claim
to be.

Pre-existing, not caused here: `pytest tests/` as a whole fails collection with 32–33 errors on clean
`main` (an `app.*` package-name collision across services).

## Evals run

```text
$ PYTHONPATH=. POSTGRES_URI=... python orion/field/evals/run_action_warrant_eval.py
{ "verdict": "PASS", "scored_ticks": 10548, "rest_fraction": 0.629124,
  "fire_fraction": 0.017254, "median": 0.40812, "median_live_dimensions": 4.0 }
```

Plus the O1 replay above, run through `build_proposal_frame` itself rather than a copy of the logic.

## Docker/build/smoke checks

```text
Not run -- no Dockerfile, compose, dependency, port, or health-check change.
```

## Restart required

```bash
scripts/safe_docker_build.sh orion-proposal-runtime up -d --build
```

Only the proposal runtime evaluates the gate. The field-digester is unchanged by this patch.

**This is the deploy where Orion's behaviour actually changes.** Expect ~63% of ticks to dispatch
nothing. That is the intended outcome, not an outage — verify with
`SELECT action_warrant_gate, count(*) ... FROM substrate_proposal_frames` rather than by absence of
dispatches.

## Risks / concerns

- **Severity: medium.** If `action_warrant` ever fails, the gate closes and Orion goes quiet. Chosen
  as the safe direction, but it is a real single point of failure. Mitigated by
  `action_warrant_gate` recording `no_live_dimensions` distinctly and by the eval's `NEVER_ON`
  detector. A watchdog on sustained `no_live_dimensions` is worth adding.
- **Severity: low.** The budget is bimodal (0 or 5, nothing between) because the gate is binary and a
  warranted tick still clears the full slate. "Rises and falls" is satisfied; a graduated budget is
  the natural next patch.
- **Severity: low.** The reverie/metacog scale collision is dissolved for *gating* but remains for
  *ranking* within a warranted tick — those candidates still enter at a flat 0.75.

## PR link

_to be filled on open_
