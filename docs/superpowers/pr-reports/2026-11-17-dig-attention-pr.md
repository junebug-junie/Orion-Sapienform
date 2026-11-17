## Summary

Orion and any session can now ask "what did I pay attention to, and was it fair to the world?" with one read-only command: `python -P -m orion.dig attention [24h|7d] [<end>]`. Fixes #3600.

- The command judges each source against its own activity. It never compares raw winner counts across sources, because the world fires rarely by nature.
- It prints five measures per target: won when eligible (and won when contested), eligible against its own activity, why it was not attended, how often nothing won (rest), and drift against the kept frames before the window.
- Every printed line is a rate with its numerator and denominator, or a ref. A test fails on any bare count.
- One read of the frames, about 5-7 s live. Each query has a 25 s statement timeout.
- New: a live eval (7 checks), a README row, an "Ask Orion" catalog entry, and a read-only grant script for Orion's sandbox role.

## Outcome moved

Before: each attention question was rebuilt by hand from `substrate_attention_frames`, and the first try on 11-17 used raw winner counts (the wrong measure). After: one command gives the relative rates, with numbers that match the issue's hand-built baseline.

## Current architecture

World-first attention (`orion/attention/world_first.py`) writes one frame about every 2.1 s. `dominant_targets` holds the eligible targets in rank order. `suppressed_targets` holds the others, each with its verdict. Frames are pruned after 72 h. Event counts live in `temporal_self_event`. No dig command read either.

## Architecture touched

- `orion/dig/attention.py` (new): window parsing, three read-only queries, rates, render, the bare-count checker.
- `orion/dig/__main__.py`: `cmd_attention`, routing, one HELP line.
- No bus, schema, env, or service change. It writes no experience, so the integration gate does not apply.

## Files changed

- `orion/dig/attention.py`: the command.
- `orion/dig/__main__.py`: wiring.
- `orion/dig/tests/test_attention.py`, `orion/dig/tests/attention_2026-11-17_baseline.json`: 38 tests on the real rows of the issue's baseline window (captured read-only; those frames are pruned soon).
- `orion/dig/evals/run_attention_live_eval.py`: live eval E1-E7.
- `orion/dig/README.md`: command row.
- `services/orion-hub/static/catalog/orion-abilities.json`: catalog entry (the catalog test requires one per dig command).
- `scripts/sql/2026-11-17_grant_orion_readonly_attention_frames.sql`: GRANT SELECT for Orion's read-only role. Not a migration; applied by hand.
- `.github/workflows/dig-kit-tests.yml`: also runs on `orion/attention/**` and `orion/schemas/temporal_self.py`.

## How each measure is computed

1. **Won when eligible**: frames where the target is ranked first, out of frames where it is eligible. Also: of the frames with more than one eligible target, how often it won.
2. **Eligible against its own activity**:
   - Event sources, from an explicit map: `world:chat` uses Juniper's `chat_turn` events with the 15 min chat rate window; `world:reading` uses `reading_turn` with the 15 min reading window; `node:substrate.street` uses `street_event` with the 300 s orienting window. Windows come from `world_first`. Expected frames = seconds lit by the events (overlapping windows count once) divided by the frame interval.
   - Every other source: eligible share of the frames where it was judged, against the 10% top-decile design rate (`1 - DEFAULT_BAND_CUTS[1]`). Its kind (continuous, event-written, percentile) comes from `world_first`'s semantic-layer functions. Nothing is hard-coded.
3. **Not attended**: each verdict is put in a cause class (quiet for itself, usual for itself, absent, insufficient history, past the orienting window, not sustained, below its band floor, ...). SQL and Python use the same table. Eval E6 checks that they agree on real reasons.
4. **Rest**: frames with no eligible target.
5. **Drift**: the relative change of each rate against the kept frames before the window. The prior rates, with their numerators and denominators, print in their own block.

## Live output (24 h to 2026-11-17T08:23Z, the issue's baseline window)

Read as `postgres` inside a READ ONLY transaction, because the grant is not applied yet:

```text
attention 24h to 2026-11-17T08:23Z, read-only, relative rates only (read in 9.0 s)
  every frame, about 2.1 s a frame; drift = relative change vs the 37h of kept frames before it (1 in 4 sampled)
rest (no eligible target): 70% (28,643 of 40,836) of frames; drift +13% (was 62% (9,192 of 14,867))
contested (more than one eligible): 7.5% (3,047 of 40,836) of frames; drift -35% (was 11.4% (1,697 of 14,867))
won = ranked first in the frames it was eligible (contested: of those with more than one eligible); eligible against its own activity:
  node:substrate.perception [percentile]: won 91% (3,971 of 4,368), contested won 64% (721 of 1,118); eligible 10.7% (4,368 of 40,836) vs design 10.0%; drift won +28%, eligible +15%
  node:substrate.execution [event-written]: won 68% (2,173 of 3,176), contested won 11.3% (128 of 1,131); eligible 7.8% (3,176 of 40,836) vs design 10.0%; drift won -11%, eligible +69%
  node:prediction.room [percentile]: won 69% (2,154 of 3,143), contested won 45% (822 of 1,811); eligible 7.7% (3,143 of 40,836) vs design 10.0%; drift won +6%, eligible -36%
  node:prediction.body [percentile]: won 65% (1,805 of 2,770), contested won 42% (705 of 1,670); eligible 6.8% (2,770 of 40,836) vs design 10.0%; drift won +6%, eligible -43%
  world:chat [event]: won 100% (834 of 834), contested won 100% (281 of 281); eligible 100% (834 of 830) of expected, 2 chat_turn x 900 s, overlaps merged [Juniper's turns]; drift won +8%, eligible -2%
  node:substrate.perception_walkway [percentile]: won 96% (419 of 436), contested won 93% (210 of 227); eligible 1.1% (436 of 40,836) vs design 10.0%; drift won -4%, eligible +15%
  world:reading [event]: won 100% (426 of 426), contested won n/a (0 of 0); eligible 3.5% (426 of 12,033) of expected, 39 reading_turn x 900 s, overlaps merged [loose]; drift won +6%, eligible -95%
  node:substrate.biometrics [continuous]: won 88% (247 of 280), contested won 83% (163 of 196); eligible 0.7% (280 of 40,836) vs design 10.0%; drift won +22%, eligible -37%
  node:substrate.strain [event-written]: won 48% (68 of 143), contested won 18.5% (17 of 92); eligible 0.4% (143 of 40,836) vs design 10.0%; drift won n/a, eligible new (prior 0%)
  node:substrate.street [event]: won 100% (96 of 96), contested won n/a (0 of 0); eligible 12.1% (96 of 792) of expected, 7 street_event x 300 s, overlaps merged; drift won n/a, eligible new (prior 0%)
  node:substrate.cabinet [percentile]: never eligible; eligible 0.0% (0 of 40,836) vs design 10.0%; drift won n/a, eligible -100%
  never eligible here or before (eligible / frames it was judged): node:prediction.chat_turn 0.0% (0 of 40,836), node:substrate.harness_closure n/a (0 of 0)
kinds: [event] eligible frames vs its own events x its news window / frame interval ([loose]: world:reading is lit by accepted reading links, not every reading); [percentile] eligible share of frames vs the top-decile design; [continuous] same, the dwell rule keeps it below; [event-written] same, no clean event source, the orienting window keeps it below
not attended, top causes (share of the frames it was judged and not attended):
...
```

Against the issue's baseline:

| Target | Issue | This run |
|---|---|---|
| chat won when eligible | 100% (834) | 100% (834 of 834) |
| reading | 100% (426) | 100% (426 of 426) |
| street | 100% (96) | 100% (96 of 96) |
| walkway | 96% | 96% (419 of 436) |
| perception | 91% | 91% (3,971 of 4,368) |
| body prediction | 65% | 65% (1,805 of 2,770) |
| room | 68% | 69% (2,154 of 3,143); one frame at the window edge differs |
| execution | 68% | 68% (2,173 of 3,176) |
| strain | 48% | 48% (68 of 143) |
| rest | 70% (28,641 of 40,835) | 70% (28,643 of 40,836) |
| perception eligible | 10.7% | 10.7% (4,368 of 40,836) |

Three eligibility numbers differ from the issue on purpose:

- **Chat: 100% (834 of 830 expected), not about 74%.** The issue counted all 8 `chat_turn` rows. 6 of them are Orion's outreach, and outreach does not make chat eligible (`world_first._CHAT_TURN_WHERE` counts Juniper's turns only). Chat's window is also 15 min, not 300 s.
- **Reading: 3.5%, not about 8%.** The 15 min window is used, overlaps are merged, and reading is marked `[loose]`, because what makes reading eligible is accepted reading links, not finished readings (#3653).
- **Street: 12.1% (96 of 792), not about 10%.** Overlaps are merged, and an event just before the window counts for the part of the window it lights.

Perception is classed `[percentile]`, not "continuous" as the issue says. The semantic layer applies the dwell rule only to body sources (`node_prediction_error_continuous`). The 10% design comparison is the same either way.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none. Behavior changed: none outside dig.

## Env/config changes

- No keys added, removed, or renamed. No `.env_example` changed. Local `.env` sync is not needed.

## Tests run

```text
pytest orion/dig/tests services/orion-hub/tests/test_orion_abilities_catalog.py -q -p no:cacheprovider
311 passed
```

## Evals run

```text
PYTHONPATH=. python orion/dig/evals/run_attention_live_eval.py --role postgres
eval: read in 5.1-7.3 s; PASS (E1-E7)   (E1 <30 s, E2 no bare counts, E3 not empty, E4 frame total, E5 event counts,
                                          E6 SQL/Python cause parity, E7 merged lit seconds vs Python reference)
PYTHONPATH=. python orion/dig/evals/run_attention_live_eval.py        (default role orion_readonly)
UNREAD: needs GRANT SELECT ON substrate_attention_frames TO orion_readonly   (expected until #3651)
```

## Docker/build/smoke checks

```text
No service or image change. Live smoke = the eval above (read-only, against orion-athena-sql-db).
```

## Review findings fixed

- Finding: a target judged only before the window printed "attended in every frame it was judged" (false).
  - Fix: it now prints "not judged in this window".
  - Evidence: `test_a_target_judged_only_before_the_window_is_not_called_attended`.
- Finding: the bare-count checker let `world:chat=834` and `834/40846` through.
  - Fix: ref and path patterns are anchored.
  - Evidence: both strings are now must-flag cases.
- Finding: expected frames for event sources counted overlapping windows twice and ignored window edges.
  - Fix: SQL now sums the union of the event windows, each cut at the next event and clipped to the window. Events just before the window are included.
  - Evidence: `test_lit_seconds_merges_overlapping_windows_and_counts_window_edges`; live eval E7 checks the SQL against the Python reference.
- Finding: the SQL cause CASE and the Python `classify()` could differ: `_` is a LIKE wildcard, and a reason without `' -- '` was handled differently.
  - Fix: `_` is escaped with `ESCAPE '!'`, and `verdict_of` matches the SQL.
  - Evidence: live eval E6 compares them on real reasons.
- Finding: drift had no prior numerator or denominator.
  - Fix: a "prior rates" block prints them, and the rest and contested lines carry `(was ...)`.
- Finding: the eligible share used all frames, not the frames where the target was judged.
  - Fix: the denominator is now the frames where it was judged.
- Nits fixed:
  - `abs(hashtext())` could overflow, so it was dropped.
  - 99.6% no longer rounds up to "100%".
  - The 30 s figure is documented as typical: each of the 3 queries is capped at 25 s.
  - The chat 98% vs 74% gap is explained in the test and in this report.

## Restart required

```text
No restart required.
```

Before Orion can run the command from its sandbox, Juniper applies one grant (one line, from the host, after merge):

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < scripts/sql/2026-11-17_grant_orion_readonly_attention_frames.sql
```

## Risks / concerns

- Severity: medium. Concern: until the grant is applied, Orion's own `dig attention` stops with a "needs GRANT" line. Mitigation: #3651 (needs-juniper, verify-live).
- Severity: medium. Concern: drift cannot reach 7 days, because frames are pruned after 72 h. For a 24 h window, the prior is at most about 2 days, and a `7d` window has no prior. The header says this. Mitigation: #3652 (daily rollup design).
- Severity: low. Concern: `world:reading` has no clean event source, so its eligibility is marked `[loose]`. Mitigation: #3653.
- Severity: low. Concern: on a host at load about 50, one live run hit the 25 s statement timeout. It says so instead of hanging. Normal runs take 5-7 s.
- Issues opened: #3651, #3652, #3653. Issues closed: #3600 (by this PR).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3654

🤖 Generated with [Claude Code](https://claude.com/claude-code)
