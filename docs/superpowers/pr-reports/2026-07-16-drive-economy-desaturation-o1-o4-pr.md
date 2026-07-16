# PR report: drive-economy desaturation, round 1 (O1 + O4) + retrospective reconciliation

Branch: `fix/drive-economy-desaturation`
Series: follows the 2026-07-15 saturation deep dive (see the motor-nerve PR reports and
`orion/autonomy/drives_and_autonomy_retrospective.md` §5a, updated in this branch). Scope
approved by Juniper: O1 + O4 now; O3 (predictive re-grounding) and O2 (event-rate
normalization) are named follow-ups, deliberately not in this patch.

## Summary

- **O1**: `derive_pressure_competition_tensions` is now signal-only — `drive_impacts={}`.
  The event still fires (kind, magnitude=spread, related_nodes, provenance) so observers
  keep the competition signal; it no longer feeds anything derived from itself.
- **O4**: `measure_autonomy_gate.py` gains a `SATURATED` verdict — a completed measurement
  that says the co-activation bar was met by an always-on monoculture, not an economy.
  Precedence: UNMEASURABLE > SATURATED > NO-GO/GO, with the genuine low-coactivation NO-GO
  short-circuiting first.
- Retrospective §5 reconciled with the parallel motor-nerve stream (§5a addendum): the
  actuator and router exist; the signal is the bottleneck.

## Outcome moved

The dominant-drive monoculture mechanism is dead. Live-verified pathology this fixes:
`dominant_drive=relational` in 96% of 1,727 audited ticks and 1,643 byte-identical audit
summaries ("orion pressure concentrates on relational") — and since PR #1085, dominance is
what chat stance (`stance_react.j2`) and Mind consume. The gate can also never again bless
this state as GO.

## The diagnosis correction worth reading (found during O1's consumer trace)

The 2026-07-15 deep dive framed the competition tension as pressure-feedback incest. The
trace showed the impacts never actually re-entered `DriveEngine.update()` in-process (the
worker doesn't consume its own tension channel; the channel's only live consumers are
persistence sinks). What the impacts demonstrably poisoned, every tick, is
`compute_tick_attribution`/`dominant_drive_from_attribution` — magnitude ~0.96 × weights
0.9/0.75 dominated attribution, which is precisely the 96%-relational / identical-summary
symptom. Two mechanisms, now correctly attributed:

1. **Dominance/narrative poisoning** (the competition tension) — fixed by O1.
2. **Pressure pinning** (~13 substrate events/min vs decay τ=1800s ≈ 0.3% decay between
   events; three drives at median 0.975–0.986; `predictive` starved dead at 0.016) — NOT
   fixed here; that is O2/O3. **Expect the gate to read SATURATED (via `all_active_frac`)
   until they land — that is the instrument working.**

O1 additionally guarantees zero pressure fold on every present-or-future consumer path
(`DriveEngine.update`, `compute_tick_attribution`, autonomy's `_fold_tension_into_pressures`,
wildcard channel consumers), so the incest class is closed regardless of future wiring.

## Files changed

- `orion/spark/concept_induction/tensions.py`: `drive_impacts={}` + load-bearing
  do-not-re-add comment; docstring updated
- `orion/spark/concept_induction/tests/test_pressure_tensions.py`: assertions flipped to
  signal-only; alias-path regression test
- `orion/spark/concept_induction/tests/test_drives_leaky.py`: two regression tests —
  competition tension contributes exactly zero pressure change; competition-only tick ==
  pure decay
- `scripts/analysis/measure_autonomy_gate.py`: `SATURATED` + three visible constants
  (`SATURATION_DOMINANT_SHARE=0.90`, `SATURATION_MIN_ACTIVE=5`,
  `SATURATION_ALL_ACTIVE_FRAC=0.75`), `DriveStats.all_active_frac/dominant_counts/
  top_dominant_share`, pure `parse_dominant_rows`/`apply_dominant_counts`, second
  degrade-safe Postgres query (failure keeps the histogram result), report renders the new
  stats and explains SATURATED; exit-code comment states SATURATED is not a pass
- `orion/autonomy/drives_and_autonomy_retrospective.md`: §5 bullets annotated, §5a status
  addendum (two-mechanism diagnosis, O1–O4 map), §6 item 2 closed

## Consumer trace (all empty-impacts paths verified by the implementing agent)

`DriveEngine.update` (zero iterations, tick still decays), `drive_attribution` (zero
contribution; falls through cleanly), `orion/autonomy/reducer.py` (never receives this
kind), `autonomy/summary.py` (computes from pressures directly), `tension_ratelimit`
(safe signature), `substrate/adapters/autonomy.py` (empty metadata, no crash; live call
passes no tension items), divergence audit / goals / dossier / rdf (zero references).

## Schema / bus / API changes

- None on the wire: `tension.drive_competition.v1` keeps its schema; `drive_impacts` was
  always an arbitrary dict and `{}` is valid. Gate report format gains fields; verdict
  string vocabulary gains `SATURATED`.

## Env/config changes

- None. No new keys, no flag flips.

## Tests run

```text
pytest orion/spark/concept_induction/tests \
       scripts/analysis/tests/test_measure_autonomy_gate.py \
       tests/test_autonomy_summary.py -q
  → 190 passed (one combined invocation)

Also run standalone by the implementing agents:
  concept_induction suite 142 passed; gate suite 44 passed (31 pre-existing + 13 new);
  autonomy summary + substrate materialization 12 passed.
```

## Evals run

```text
No eval harness for these surfaces (tracked as issue #1066). The post-deploy live
re-measurement below is the behavioral check.
```

## Review findings fixed

Review verdict: **approve — no CRITICAL/HIGH/MEDIUM findings.** The headline risk
(quiet ticks minting a NEW alphabetical monoculture once the competition tension left
attribution) was checked and cleared: all-zero attribution degrades to an honest
`dominant_drive=None`/`summary=None`, and the max-pressure fallback in `audit.py` that
would have re-minted `relational` from the still-pinned pressures is unreachable from the
live call site. No code anywhere parses the verdict strings, so `SATURATED` breaks no
consumer.

- Finding: (LOW, the one fix the reviewer recommended) the histogram and dominance
  queries ran with no shared upper bound on an autocommit connection — rows landing
  between the two statements would count in `dominant_counts` but not the share
  denominator, letting `top_dominant_share` drift past 1.0 (fail-safe direction, but
  breaks the ≤1 contract).
  - Fix: `window_end` captured once, both queries bounded `>= start AND < end`; a test
    asserts both statements carry the byte-identical upper bound.
- Finding: (LOW, judgment call, accepted + softened) SATURATED returns before the
  resource-pressure check, so a reader could infer "fix saturation → GO" when pressure
  would still fail.
  - Fix: when the pressure bar is also unmet, the SATURATED explanation now appends
    "resolving saturation alone would read NO-GO, not GO."
- Accepted without change: tie-break step 2 inert when the competition tension is lead
  (pre-existing alphabetical fallback, only on exact float ties, and the old tie bias
  was itself part of the removed loop); `SATURATION_MIN_ACTIVE=5` as a global constant
  (correct for the actual 6-drive deployment; deriving from drive count is a
  someday-refactor).

## Restart required

```bash
# concept-induction picks up the signal-only competition tension (after merge)
docker compose --env-file .env --env-file services/orion-spark-concept-induction/.env \
  -f services/orion-spark-concept-induction/docker-compose.yml up -d --build

# The gate script is not a service; no restart. Post-deploy verification (clean window,
# after the bus-core situation settles):
psql -h localhost -p 55432 -U postgres -d conjourney -c \
  "SELECT dominant_drive, count(*) FROM drive_audits WHERE created_at > now() - interval '6 hours' GROUP BY 1 ORDER BY 2 DESC;"
python scripts/analysis/measure_autonomy_gate.py --window-hours 6
# Success criteria: dominant_drive churns (no single drive >90%), summaries diversify,
# gate reads SATURATED (not GO) while pressure pinning persists pre-O2/O3.
```

## Risks / concerns

- Severity: MEDIUM. Concern: with the competition tension out of attribution, quiet ticks
  could fall through to a new degenerate dominant (alphabetical tie-break). Mitigation:
  flagged to the review pass explicitly; most ticks carry real extracted tensions with
  non-zero impacts; post-deploy verification watches exactly this.
- Severity: LOW. Concern: SATURATED thresholds (0.90/5/0.75) are first-pass calibrations.
  Mitigation: visible module constants, trivially tunable; the live shape (0.96/0.74) sits
  comfortably past both.

## PR link

(filled after push)
