# PR report — give the regime panel its authoritative write timestamps (R2 follow-through)

Branch: `fix/hub-regime-timestamps`

## Summary

- `channel_regime()` has always accepted an `updated_at` parameter with a docstring saying *"supply it whenever available — it makes `refresh_state` authoritative instead of inferred."* **The Hub never supplied it.**
- `build_channel_series()` computed the merge, discarded the provenance, and returned values only — so all 38 channels on the one surface anyone reads fell back to value-ratio inference, which is blind in the subnormal range.
- Threads the **merge winner's** write timestamp through, using R1's provenance dict, plus the window's start time.
- Rewrites `_refresh_from_timestamps` around the question actually being asked, after two earlier rules were both wrong.
- Adds a coverage floor so a *negative* verdict cannot be asserted from a sparsely-stamped series.

## Outcome moved

Measured across **four consecutive 1-hour windows** (~1750 ticks each):

| window | rows | timestamp path | verdicts changed |
|---|---|---|---|
| −1h..0h | 1747 | 25/38 | 3 |
| −2h..−1h | 1752 | 22/38 | 12 |
| −3h..−2h | 1745 | 24/38 | 12 |
| −4h..−3h | 1752 | 22/38 | 12 |

**22–25 of 38 on the timestamp path, 3–12 verdicts changed.** Quoted as a range on purpose — see the retraction below.

Changes go in both directions. `stream_backlog_pressure` was reported `static` while a producer really was writing it (86 distinct stamps in-window); several channels move from `static` ("nothing changed") to `no_write_in_window` ("nobody wrote"), which is a different and more useful claim.

## Architecture touched

`build_channel_series()` now returns `(series, stamps, tick_times, unparsable)`. The merged value is *one* source's reading and R1's provenance dict names which, so the honest timestamp is that winner's own `node_vector_updated_at` entry — not the newest across all sources, which would credit the merged value with freshness no contributor had.

## Three rules were tried here; the first two were wrong

1. **`len(set(stamps)) > 1` — "are the stamps different."** Correct for a single-source series. Wrong the moment a *merged* channel arrives: the winner changes between ticks, so two nodes each frozen at their own old write time yield two distinct stamps and read as a write that never happened.

2. **"Does the newest stamp advance."** Fixed that and introduced worse. A producer that wrote **exactly once** in the window is structurally undetectable, because one distinct stamp can never show an advance. Confirmed live: `execution_friction` carried **2 stamped samples of 437 (0.5%)**, both at `23:58:29` — a real write, inside the window — and this rule reported `no_write_in_window` *and flagged it authoritative*.

   Same shape as CLAUDE.md 0A's metric-gate item 4, inverted: structurally incapable of reading one write. A confidently wrong verdict is worse than the blindness the timestamp path exists to replace.

3. **Compare the newest stamp against the window's start.** That is the question, needs no second stamp, and cannot be fooled by winner churn.

The two directions carry **different evidential bars**, deliberately:

- `producer_written` needs **one** stamp inside the window — a single real observation of a write is proof a write happened.
- `no_write_in_window` needs stamps on at least `MIN_STAMP_COVERAGE` (0.5) of samples — absence cannot be concluded from 0.5% of a series, because the other 99.5% are unstamped, not silent. Below the floor it returns `unknown` and the caller falls back.

## Files changed

- `orion/field/regime.py`: rewritten `_refresh_from_timestamps`, `MIN_STAMP_COVERAGE`, `window_start` parameter, `_aware_utc` on all comparisons.
- `services/orion-hub/scripts/field_channel_glossary_routes.py`: `stamps` + `tick_times` threaded through; `_winning_write_time()`.
- `services/orion-hub/tests/…` and `tests/test_field_regime.py`: 54 passing.

## Schema / bus / API changes

None. The `/health` response gains no new fields; existing `regime.refresh_state` / `refresh_evidence` values become correct more often.

## Env/config changes

None.

## Tests run

```
pytest tests/test_field_regime.py \
       services/orion-hub/tests/test_field_channel_glossary_routes.py -q
    54 passed, 1 pre-existing failure
```

Pre-existing failure is `test_channels_endpoint_returns_35_entries` (38 vs 35), red on main and parked by Juniper. Untouched.

## Evals run

Mutation testing, 12 mutants across both files:

```
KILLED 12/12
```

Two notes on getting there. The first run reported 10/12 with `flip the comparison` surviving — that was **a bug in my harness**, not a test gap: it detected kills by substring-matching `"1 failed"`, which also matches `"11 failed"`. Fixed to parse the count numerically, and the mutant was in fact killed. The one genuine survivor was `drop _aware_utc on stamps`, now covered.

## Review findings fixed

Code review found a BLOCKER, two MAJORs and three SHOULDs. It was right on all of them, and the BLOCKER contradicted this patch's own headline claim.

- **BLOCKER — the patch shipped a confidently wrong verdict on the two channels it was written to fix.** `channel_regime` promoted to `refresh_evidence="timestamp"` if *any* non-None stamp existed, so `execution_friction` (0.5% coverage) and `egress_confidence_deficit` (3.9%) got authoritative `no_write_in_window` while their newest stamps were **inside** the window.
  - **Fix:** rule 3 above, plus the coverage floor for the negative direction.
  - **Evidence:** verified independently — `execution_friction` 2/437 stamped, newest `23:58:29`, in-window. `test_sparse_in_window_stamp_is_producer_written_not_absence` regresses it end to end.
  - My commit message said *"`execution_friction` was claiming `producer_written` with no write in the window at all."* **That was backwards**, and it had been written into three places on disk. Retracted.

- **MAJOR — the advance rule had zero test coverage.** Reverting `_refresh_from_timestamps` to the pre-commit implementation passed all 48 tests byte-identically. Half the commit, with a 12-line docstring justifying itself, was unverified.
  - **Fix:** `test_two_frozen_sources_at_different_times_is_not_a_write`, asserted in **both** winner orderings.

- **MAJOR — the advance loop was order-dependent and its `max()` line was dead.** `newest = max(newest, stamp)` was unreachable, making the function `any(s > real[0])` — so two frozen sources read `producer_written` whenever the older-stamped one won first. A coin flip. Moot under rule 3, and the both-orderings test pins it.

- **SHOULD — the quoted measurements did not reproduce and were window-sensitive.** "26 of 38 / exactly 7" was one window stated as a property. Re-measured across four windows and quoted as a range, in the report and both docstrings.

- **SHOULD — `_winning_write_time`'s capability claim was wrong in principle.** `FieldStateV1.capability_provenance` documents itself as *"the edge source_id (a node_id like `node:atlas`)"*, so it can collide with a real node key by design; a bare lookup would attribute a node-vector write time to a capability-vector value. Zero live collisions (775 occurrences, 400 ticks) — an observation, not the guarantee I claimed. Now an explicit `source_id not in state.node_vectors` guard.

- **SHOULD — mixed naive/aware datetimes raise `TypeError` and would 500 the panel.** Not reachable today (400 rows sampled, 0 naive), but this patch is what makes the comparison path reachable at all. Routed through `_aware_utc`; regression test covers both directions.

- **Also flagged and fixed:** no PR report file (this document).

- **Confirmed clean by review, not taken on my word:** stamp/series alignment (zero length mismatches across 1754 rows × 38 channels), the `_regime_for` length guard, and no regression to `_refresh_from_timestamps`'s original single-source contract.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build
```

Hub-only; the digester and bus are untouched.

## Risks / concerns

- **Severity: low.** `MIN_STAMP_COVERAGE = 0.5` is declared, not derived. It is the threshold at which a negative verdict is withheld; too high and sparse-but-honest series lose the authoritative path, too low and the BLOCKER returns. Chosen as a majority rule and stated as declared.
- **Severity: low.** 13–16 of 38 channels still use value-ratio inference (capability-sourced or below the coverage floor) and keep its documented subnormal blind spot. Reduced, not eliminated.

## PR link

<pending>
