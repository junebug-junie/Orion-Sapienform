## Summary

- The heartbeat's "smear" reading (how entangled the far end of the lattice is compared to the near end) divided by a near end that had almost no entanglement left, so it reported values up to 1.46 million. Its own docstring says a dead near end should read as absent, not huge.
- Smear and `smeared` are now `null` when the near end carries less than a tenth of the far end's entanglement (`SMEAR_DEAD_RATIO = 10`, `services/orion-heartbeat/app/substrate/proprioception.py`).
- The 10 comes from the live data: the trough of the 7-day smear distribution (derivation below and in the code comment).
- The old absolute floor (1e-6 bits) stays, only as a 0/0 guard for when both ends are dead.
- Fixes bug #2 from the self-calibration spec (PR #2528, "Real bugs found").

## Outcome moved

About 5.7% of persisted self-model rows over the last 7 days (around 830 of 14,668) carried a heartbeat smear between 10 and 1.46e6. Each consumer took those as "hugely smeared", and the hub smear chart scaled its y-axis to them, which flattened the real readings (around 2-3) to a line. After this patch those ticks read as absent.

## Current architecture

`compute_h1_ensemble()` (`reconstruction.py`) averages the 8 trajectories' 9-cut entanglement entropy profiles. `profile_smear()` computes near = mean(cuts 0,1) and far = mean(cuts 7,8) and returns far/near. Before this patch it returned absent only when near < 1e-6 bits. The result goes out on `GET /h1`. The substrate runtime copies it into `AttentionSelfModelV1.heartbeat_smear`/`heartbeat_smeared`, which persist to `substrate_attention_self_model.self_model_json`. The hub attention-organ tab renders the live value and its history.

## Architecture touched

Only the `orion-heartbeat` producer. No schema, bus, env or consumer change.

## Files changed

- `services/orion-heartbeat/app/substrate/proprioception.py`: adds `SMEAR_DEAD_RATIO` with its derivation, adds a relative dead-near-end check, and updates the docstring.
- `services/orion-heartbeat/tests/test_proprioception.py`: adds 3 regression tests.
- `services/orion-heartbeat/README.md`: documents when `/h1` `smear`/`smeared` are null.
- `docs/superpowers/pr-reports/2026-10-10-heartbeat-smear-dead-near-end-pr.md`: this report.

## Live evidence (bug re-verified before fixing)

There was one bounded read-only query against `substrate_attention_self_model`, plus two follow-up histogram views of the same 7 days of rows:

```text
timeout 25 docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "SET statement_timeout='15s';
  with r as (select (self_model_json->>'heartbeat_smear')::float s from substrate_attention_self_model
  where generated_at > now() - interval '7 days' and self_model_json ? 'heartbeat_smear') select count(*), count(s), min(s),
  percentile_cont(array[0.05,0.25,0.5,0.75,0.9,0.95,0.99]) within group (order by s), max(s),
  count(*) filter (where s>=1), count(*) filter (where s>=100), count(*) filter (where s>=1e4) from r"

rows=15235 non_null=14668 min=0.9449
p5=2.016 p25=2.121 p50=2.619 p75=2.865 p90=3.183 p95=50.74 p99=12482.95 max=1459596.4
>=1: 14636   >=100: 529   >=1e4: 166
```

The log10 quarter-decade histogram of the same rows (bin lower edge, count) is clearly bimodal:

```text
-0.25:32  0:104  0.25:12983  0.5:697  0.75:19  | 1.0:9 |  1.25:36  1.5:77  1.75:180  2:65 ... 4:48  4.5:37  5:12  5.5:9  6:5
```

A finer view of 3..33 shows the alive body ending at 4.536, nothing between 4.536 and 5.118, and then a sparse, continuous tail.

## Floor derivation

- The near-end entropy is not persisted or exposed anywhere (it exists only in-process), so the floor is expressed relative to the far end. Since smear = far/near, "near < far/K" is exactly "smear > K", and the persisted smear distribution fixes K directly.
- Alive population: 94% of rows fall in 0.56..4.54 (median 2.62). Collapsed population: about 830 rows from about 18 up to 1.46e6, meaning near is about far/1e6. The least-populated quarter-decade bin between the two is [10, 17.8), with 9 rows (0.06%). Its lower edge, 10, is the cut.
- Theory check: far is at most log2(min(PHYS_DIM, BOND_DIM)) = 2 bits, so a smear of 1e4 implies near ≤ 2e-4 bits. That is a near-pure boundary site, which is dead by any reading. Without the relative check, such a site still clears the 1e-6 floor.
- The cut is scale-free on purpose: absolute entropy moves with BOND_DIM/PHYS_DIM, but the ratio does not. Re-derive it with the same query if the lattice dimensions or the dissipation change.
- The old `NEAR_FLOOR = 1e-6` is kept, but only as the 0/0 guard. It is documented as not being the honesty fix.

## Consumers checked (absent must not read as calm 0 or as huge)

- `orion/substrate/attention_self_model.py` `_heartbeat_h1_fields`: accepts only finite numbers ≥ 0, so None becomes `heartbeat_smear=None`. `smeared` is accepted only when it is a bool. Existing tests (`test_attention_self_model.py:878,923`) already pin None. No change needed.
- `orion/schemas/attention_self_model.py`: `heartbeat_smear: float | None`. No change needed.
- `services/orion-hub/scripts/attention_organ_routes.py`: passes values through unchanged, and only passes `heartbeat_smeared` when it is a bool. No change needed.
- `services/orion-hub/static/js/attention-organ.js`: `num()` renders null as "—", the history chart filters out null points, and the `smeared` label renders only on `=== true/false`. No change needed. The chart's y-max is no longer blown out by 1e6 points.
- `services/orion-equilibrium-service` flow/insight metacog gates: listed in the metric lock as consumers of the whole self-model row, but they read only `prediction_error_confidence`, never smear.
- `lattice_probe.score_kick` uses the same 1e-6 pattern (and returns +inf). It is the pre-registered offline probe, not a live reading, so it is left unchanged and out of scope.

## Metric semantic layer lineage (checked before calling this a bug)

Ran `.venv/bin/python scripts/check_metric_lineage.py` (`--metric <token>`, `--drift`, `--unwritten`) on 2026-10-10 against main, and read the matching `orion/inner_state_registry.py` and `config/field/field_channel_glossary.v1.yaml` entries. Nothing in the semantic layer marks this behaviour as designed.

- `--metric heartbeat_smear` resolves to `metric://inner_state/orion-substrate-runtime/attention_self_model.v1#heartbeat_smear`, a scalar field on `AttentionSelfModelV1` produced by orion-substrate-runtime. There is no glossary entry, and the registry note says nothing about a valid range or about huge ratios being intended.
- Declared consumers are inherited from the whole `attention_self_model.v1` signal: the equilibrium flow and insight metacog gates. Both read `prediction_error_confidence` only, never the smear, so null does not reach them.
- Discovered blast radius (non-test, 3 sites): `services/orion-hub/scripts/attention_organ_routes.py:240` and `:592`. Both pass the value through, and the hub JS renders null as "—". That matches the consumers checked above.
- `WRITTEN BY: 0`, because the field is written by a whole-model construction the scan cannot see. Producer traced by hand: `proprioception.py` to `_heartbeat_h1_fields`.
- `--drift`: the declared flow/insight gates are not among the discovered consumers, which is consistent with the gates never reading this field.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: `/h1` `smear`/`smeared` (and therefore `AttentionSelfModelV1.heartbeat_smear`/`heartbeat_smeared`) are null when far/near > 10, where before they were huge.
- Compatibility notes: the field was already nullable, and every consumer handles null.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no env change)
- skipped keys requiring operator action: none

## Tests run

```text
PYTHONPATH=<worktree> .venv/bin/python -m pytest services/orion-heartbeat/tests -q   -> 150 passed
Mutation check: reverted the condition to `if near < NEAR_FLOOR:` -> 3 new tests FAILED (20 passed); restored -> 23 passed
python scripts/check_definition_drift.py --gate   -> PASS (no re-lock: the resolved metric definition did not change; this is producer behavior)
python scripts/check_metric_lineage.py --gate     -> PASS
python scripts/check_inner_state_registry.py      -> OK
```

No CI workflow runs the `orion-heartbeat` tests. The static gates above are the CI gates that touch this metric.

## Evals run

```text
None for this patch. services/orion-heartbeat/evals exists but does not cover proprioception smear.
The live-distribution query above is the evidence for the cut.
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-heartbeat build   -> Image orion-heartbeat-heartbeat Built
(built from this worktree with temporary .env symlinks, which were removed afterwards; not brought up)
```

## Review findings fixed

- Finding (should): two phrases in the derivation comment overstated the data. The alive body was described as "0.56..4.54", but the live minimum is 0.94 and the body has a sparse shoulder that reaches 9.5. The comment also said "nothing between 4.54 and 5.12", but 1 row sits there.
  - Fix: reworded the comment to "0.94..5.5 with a sparse shoulder to 9.5 (19 rows)". It now also says the trough is shallow (19 / 9 / 35 rows), so 10 should be read as an order-of-magnitude edge rather than a precise one.
  - Evidence: the reviewer re-ran the live query independently and confirmed that [10, 17.8) is the least-populated bin.
- Finding (nit): a NaN or negative entropy made both comparisons false, so the function returned `(nan, False)`. The `smeared=False` value then leaked downstream and read as "local".
  - Fix: rewrote the check fail-closed as `if not (near >= NEAR_FLOOR and 0.0 <= far <= SMEAR_DEAD_RATIO * near)`.
  - Evidence: added `test_profile_smear_nan_or_negative_is_absent_not_local`. A re-run of the mutation check (reverting to the old condition) fails 4 tests, and the restored code passes 24/24. The full service suite passes 151.
- Finding (should, pre-existing, not fixed here): `smeared` is a saturated constant, because every live reading is at least 0.94, above `SMEAR_MIN = 0.5`. Recorded under Risks.
- Finding (nit, not fixed): when both ends are tiny but above 1e-6 (for example near = 2e-6, far = 1e-6), the function still returns a reading. This was not seen in the live data. The 1e-6 floor is documented as only the 0/0 guard.
- Finding (nit, not fixed): old rows with smear between 1e4 and 1e6 stay in the database, so the hub history chart's y-max stays stretched for any window that includes rows from before the deploy, until those rows age out. This is cosmetic, needs no backfill, and is not a regression.

## Restart required

After merge, from the primary checkout on main:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && scripts/safe_docker_build.sh orion-heartbeat up -d --build
```

## Risks / concerns

- Severity: low
  - Concern: the 10 cut comes from 7 days of data under the current lattice dimensions and dissipation settings.
  - Mitigation: the code comment names the query to re-derive it, and the cut is scale-free.
- Severity: low (side finding, not fixed here)
  - Concern: over these 7 days `smeared` was never false. The minimum smear was 0.94 against `SMEAR_MIN = 0.5`, so the boolean is saturated.
  - Mitigation: report it for the self-calibration work. It is a separate metric-gate question.
- Severity: low
  - Concern: the local `orion-heartbeat-heartbeat` image tag now holds this branch's build. The running container is unchanged, but a restart before merge and a rebuild from main would run this code.
  - Mitigation: deploy with the one-liner above, which rebuilds from main.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2571

🤖 Generated with [Claude Code](https://claude.com/claude-code)
