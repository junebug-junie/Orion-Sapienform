# Persist juniper_affective_state (and make the counts actually summable)

## Summary

- `orion:substrate:juniper_affective_state` was the **last channel in the PR #1491
  four-domain family still on `consumer_services: []`**. On a Redis pub/sub bus that is
  not "shadow write", it is deletion — `PUBSUB NUMSUB` returned **0**. Three days of real
  readings exist nowhere.
- New `JuniperAffectiveStateSQL` → `juniper_affective_state_log`, first real consumer.
- Persists the **raw counts**, not just the rate, so the 15-minute window is no longer
  baked in.
- Review then caught that summing those counts was **already wrong**, on the only two
  rows the table had. Fixed with a `cold_start` flag.
- Live and verified end-to-end: `PUBSUB NUMSUB` now returns 1, and real rows are landing
  with values matching the producer's own log line field for field.

## Outcome moved

The signal stopped evaporating. Its readings had been corrected only hours earlier by
PR #1629 (−21.4%, after finding 45.6% of all counted swearing was a model quoting Juniper
back at itself in `/compact` summaries) — and every corrected reading since that merge was
also being dropped. Fixing a measurement and then discarding it is the worse of the two
failures.

That closes the four-domain family: `structural_mass`, `dev_economics`,
`doc_semantic_drift`, and now this. **No channel in it drops events any more.**

## Why raw counts, not just the rate

`swear_frequency` is computed over a fixed 15-minute window, but real typing is bursty.
Measured over the live 31-day corpus: **78% of windows are empty** (`swear_frequency`
NULL), and of the remainder **69% read exactly 0.0** — only ~6.7% of all windows carry a
non-zero value, and a short window turns one swear into a spike (a real window reads 0.077
off 13 words).

Storing `swear_count`/`word_count`/`message_count` means a consumer can re-derive the rate
over an hour, a day, or a session by summing — which is how `aggregate_scores()` weights
correctly across messages of very different lengths. A stored ratio cannot be
re-aggregated without reintroducing the short-message bias.

`swear_frequency` stays nullable. NULL means `word_count == 0` — Juniper typed nothing —
which is a different fact from a calm window. NOT NULL with a 0.0 default would merge the
two and drag every downstream average toward zero.

## The bug review found: summing was already wrong

The producer seeds `last_until = now - cold_start_lookback_sec` on every start
(`affective_state.py:117`). Live config is **3600s against a 900s poll**, so the first tick
after any restart emits a one-hour window re-covering up to four windows a previous run
already published. Ordinary ticks tile exactly and sum cleanly; that one does not.

Proven on real rows, not argued:

```
window_since         span_s  msgs  words  cold_start
2026-08-14 00:28:23    3600    26   5913  t
2026-08-14 00:51:44    3600    28   5153  t
2026-08-14 01:11:54     913     9   1756  f
```

```sql
select sum(word_count) filter (where not cold_start) correct, sum(word_count) naive
  from juniper_affective_state_log;
-- correct: 1756    naive: 12822     (+630%)

select count(*) from juniper_affective_state_log a join juniper_affective_state_log b
  on a.event_id < b.event_id and a.window_since < b.window_until and b.window_since < a.window_until
 where not a.cold_start and not b.cold_start;
-- 0   <- no overlaps survive the filter
```

Deploying this branch *guarantees* a producer restart — it edits a schema
`orion-cocreation-signals` imports — so it would have minted a fresh overlap row on the
way in. It did, twice, which is why two of the three rows above are flagged.

**Fix:** mark the tick. `cold_start` rides the wire and the row, indexed, and the model
docstring now says **SUM ONLY WHERE `cold_start IS FALSE`** instead of promising
unqualified summability. Cold-start rows are kept rather than dropped — they are the only
record of a span crossing a restart, so they are what fills a genuine downtime gap. They
are simply not summable alongside the tiling rows.

The flag clears on a **successful** publish, not merely on having ticked: a failed publish
parks the cursor, so the retry covers an even wider range that still overlaps a previous
run's rows.

## `event_id` keyed on both window bounds

Derived rather than uuid4, so a replay or backfill upserts instead of duplicating. **Both**
bounds rather than `window_since` alone: windows tile contiguously, so `window_since` looks
unique until a restart produces an overlapping window. Those are real observations over
different spans and must not collapse onto one row.

Honest correction to an earlier draft of this reasoning: the motivation is *not* bus
redelivery. `OrionBusAsync.publish()` is pub/sub — no ack, no retry, no redelivery. The key
is still right (it is free, and it makes replay/backfill safe), but the transport cannot
produce the scenario originally cited.

## Files changed

- `orion/schemas/affective_state.py`: `event_id` (derived), `cold_start`.
- `orion/bus/channels.yaml`: `consumer_services: ["orion-sql-writer"]`; corrected a
  now-stale sibling comment claiming this was still the last unconsumed channel.
- `services/orion-sql-writer/app/models/juniper_affective_state.py`: new.
- `services/orion-sql-writer/app/{models/__init__,worker,settings}.py`: wiring.
- `services/orion-sql-writer/.env_example`: subscribe list + route map.
- `services/orion-cocreation-signals/app/producers/affective_state.py`: cold-start marking.
- tests: 17 shape/contract + 2 producer-loop; 4 stub signatures updated.

## Schema / bus / API changes

- Added: `JuniperAffectiveStateV1.event_id`, `.cold_start` (both defaulted — an event
  predating them validates unchanged).
- Behavior changed: `orion:substrate:juniper_affective_state` now has a consumer.
- Compatibility: no existing consumer of this schema exists. Verified: only two code sites
  construct or validate it, and `scripts/replay_juniper_affective_state.py` never builds
  the model at all.

## Env/config changes

- Added to subscribe list: `orion:substrate:juniper_affective_state`.
- `.env_example` updated: yes.
- **Live `.env` hand-edited**: yes, required. `effective_subscribe_channels` is
  env-**REPLACED**, not merged, so an operator `.env` predating this patch leaves the
  writer permanently unsubscribed with no error and rows silently never appear (confirmed
  live with `dev_economics`, 2026-08-12). Verified after editing: `Settings()` against the
  real file resolves the route to `JuniperAffectiveStateSQL` and reports the channel
  subscribed.
- Live `.env` route map deliberately **not** edited: `Settings.route_map` is
  `{**DEFAULT_ROUTE_MAP, **overrides}` — a genuine merge — so the code default supplies it.
  Confirmed by review, including that `.env_example`'s map is a strict superset of the live
  one with zero value conflicts, so a future `sync_local_env_from_example.py` is
  non-destructive.

## Manual migration applied

This repo has no migration system, and `create_all()` creates missing **tables** but not
missing **columns** — so `cold_start` silently did not exist on the already-created table.

```sql
ALTER TABLE juniper_affective_state_log
  ADD COLUMN IF NOT EXISTS cold_start BOOLEAN NOT NULL DEFAULT FALSE;
CREATE INDEX IF NOT EXISTS ix_juniper_affective_state_log_cold_start
  ON juniper_affective_state_log (cold_start);
UPDATE juniper_affective_state_log SET cold_start = TRUE
 WHERE extract(epoch from (window_until - window_since)) > 1800;   -- 1 row
```

The backfill is not cosmetic: leaving both rows FALSE would assert the overlapping row is
summable, which is the bug itself.

## Tests run

```text
services/orion-sql-writer/tests/test_juniper_affective_state_sql_shape.py  -> 17 passed
services/orion-cocreation-signals/tests                                    -> 86 passed
```

Every new test mutation-verified against the specific wrong implementation it exists to
catch:

```text
key on window_since only            -> 2 failed
clobber a wire-supplied event_id    -> 1 failed
uuid4 instead of derived            -> 2 failed
swear_frequency NOT NULL            -> 1 failed
drop the raw count columns          -> 2 failed
drop channel from code default      -> 1 failed
drop route from DEFAULT_ROUTE_MAP   -> 2 failed
drop window_until index             -> 1 failed
remove channel from .env_example    -> 1 failed
never flag cold_start               -> 2 failed
flag every tick                     -> 1 failed
clear the flag unconditionally      -> 1 failed
no-op control                       -> all passed
```

**Known gate limitation:** `pytest services/orion-sql-writer/tests` (the command
CLAUDE.md §11/§17 prescribes) aborts at collection on `test_dream_model_constraints.py` and
runs zero tests. Pre-existing and identical on `main`; **fixed by PR #1632**, which is open
on a separate branch. Until that merges, the 17 shape tests reproduce only when the file is
named directly or `--ignore`d.

## Evals run

```text
None. services/orion-sql-writer/evals/ does not exist.
```

Consistent with the `DocSemanticDriftSQL` precedent this patch mirrors, stated rather than
implied.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-sql-writer         build / up -d   -> Started
scripts/safe_docker_build.sh orion-cocreation-signals build / up -d   -> Started
redis-cli PUBSUB NUMSUB orion:substrate:juniper_affective_state       -> 1   (was 0)
```

End-to-end, producer log vs persisted row:

```text
producer  01:27:19  message_count=9 word_count=1756 swear_count=0 swear_frequency=0.0
row                 9 | 1756 | 0 | 0    window 01:11:54 -> 01:27:07   cold_start=f
```

Field for field. Review independently rescored the cold-start window from the real
transcripts and got `(26, 5913, 2, 0.00033823778116015557)` against a stored
`26 | 5913 | 2 | 0.00033823778116015557` — exact.

## Review findings fixed

- **Finding (high): `SUM(counts)` was wrong — the table's entire justification.** 100% of
  rows affected, +34.7% on the first pair.
  - Fix: `cold_start` flag, indexed; docstring corrected from an unqualified summability
    promise to `SUM ONLY WHERE cold_start IS FALSE`.
  - Evidence: reproduced independently before accepting; now 630% naive inflation across 3
    rows, and 0 overlapping pairs survive the filter.
- **Finding (high, operational): a concurrent session redeployed sql-writer from a branch
  without this commit,** so the durable `.env` subscription outlived the code and ticks
  landed in `bus_fallback_log`.
  - Confirmed: 1 event, preserved as JSON — it took `_write_fallback`, **not** the
    `evidence_units` catch-all, so nothing was lost.
  - Fix: redeployed both services from this worktree. Will recur on any sql-writer deploy
    from another branch until this merges.
- **Finding (medium): the prescribed gate command runs zero tests.** Pre-existing; see
  "Known gate limitation" above.
- **Finding (low-medium): no test constructs the empty window** (`word_count=0`,
  `swear_frequency=None`) — 78% of real windows, and the case the nullable column exists
  for. Verified by hand (pydantic accepts it; `merge()` writes NULL) but still unguarded.
  **Not fixed — disclosed.**
- **Finding (low): `event_id` is not canonicalized across timezone representations.** Not
  reachable from the current producer (always `datetime.now(timezone.utc)`), but a backfill
  using local time would duplicate. **Not fixed — disclosed.**
- **Finding (low): "a redelivered event upserts" overstates pub/sub.** Corrected above.

**Confirmed clean by review**, each by running code: end-to-end persistence; `event_id`
stability across old/new-schema producers including the `microsecond == 0` case; it really
is an upsert (`sess.merge()`, `created_at` preserved); no table/column drift; no existing
consumer broken; env correct and complete and the route-map merge reasoning holds; the
sql-writer suite not stomped (+14 = exactly the new tests, same failure set); and **no
unfailable test** across 10 mutations.

## Restart required

```text
Already deployed as part of this patch.
```

## Risks / concerns

- Severity: medium
- Concern: `cold_start` correctly marks *unsummable* rows but does not make them usable. A
  consumer wanting a true hourly figure across a restart must reconcile the overlap itself.
- Mitigation: the raw window bounds are stored, so the overlap is computable. Clamping the
  cold-start window to the last persisted `window_until` would remove the problem entirely
  — deliberately not done here, since it couples the producer to the consumer's table.

---

- Severity: low
- Concern: `COCREATION_SIGNALS_AFFECTIVE_STATE_COLD_START_LOOKBACK_SEC=3600` against a 900s
  poll means every restart mints an unsummable row, and this repo redeploys often.
- Mitigation: setting the lookback equal to the poll interval would nearly eliminate them,
  at the cost of losing downtime recovery. Left as-is; the flag makes the trade visible
  rather than silent.

## PR link

<!-- filled in on push -->
