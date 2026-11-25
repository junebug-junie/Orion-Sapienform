## Summary

- The "will Juniper message Orion in the next hour?" forecast now knows whether she had the Hub open and in view just before the hour. Fixes #3578
- Two new inputs: Hub in view in the last 15 min (`hub_lt15m`), and 15-60 min ago but not since (`hub_15_60m`). Both read only minutes that started before the hour.
- Juniper's rule ("looking at the Hub means reachable") is the starting belief: +1.0 and +0.5 log-odds, strength 1. The data moves them; nothing drops them.
- A third input, `hub_unknown`, marks hours before the sensor existed (before 2026-11-11 03:22Z). Those hours still train every other rule but do not teach "not looking".
- The two booleans (and `hub_unknown`) are stored in each forecast's `features` jsonb. No migration.
- The replay eval gains `--no-hub` for a before/after run, and a `hub_covered_only` block.

## Outcome moved

The first sign of skill on cold hours ("will she come back after 3 quiet hours?"). Same origins, with vs without Hub, over the sensor's 14 full days:

- all hours: -0.028 log loss/day (se 0.021, P(better) 0.90); last 7 days -0.054 (P 0.97)
- cold hours: -0.013 (se 0.011, P 0.87); last 7 days -0.027 (P 0.95)

Whole-history replay (118 days, only 14 with the sensor): all 0.1124 -> 0.1096, cold 0.0708 -> 0.0700; cold vs base P 0.52 -> 0.65.

Fitted at 2026-11-25 06:00Z: Hub < 15 min +1.29, Hub 15-60 min +0.68, hub_unknown -0.34.

## Current architecture

`app/reachable.py` fits a MAP logistic regression each hour from data before the origin. Inputs were the message-recency buckets, Juniper's clock rules, and Claude Code typing (with a coverage mask). `hub_attention_minute` (orion-hub #3179) had no reader in orion-prediction.

## Architecture touched

orion-prediction only. New read-only queries on `hub_attention_minute`. No bus, schema-registry, or env change.

## Files changed

- `services/orion-prediction/app/reachable.py`: three new inputs and priors, `features(..., hub_ns, hub_since_ns)`, `forecast(..., hub, hub_since)`, tick reads the Hub minutes; docstring.
- `services/orion-prediction/app/store.py`: `juniper_hub_minutes(start, end)`, `juniper_hub_since()`.
- `services/orion-prediction/evals/run_reachable_replay_eval.py`: Hub threading, `--no-hub`, `hub_covered_only`.
- `services/orion-prediction/tests/test_reachable.py`: no-leakage, window edges, hub_unknown, cold-hour lift, tick stores the Hub booleans; fake store updated.
- `services/orion-prediction/README.md`, `docs/superpowers/specs/2026-11-10-juniper-reachable-forecast-design.md`: evidence, metric gate, replay.

## Schema / bus / API changes

- Added: three keys in `prediction_reachable_forecast.features` jsonb (`hub_lt15m`, `hub_15_60m`, `hub_unknown`).
- Removed: none
- Renamed: none
- Behavior changed: the served p now uses the Hub inputs.
- Compatibility notes: old rows simply lack the keys; `GET /reachable` shape unchanged otherwise.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
orion-prediction image: python -m pytest tests -q   -> 595 passed (before and after review fixes)
scripts/check_metric_lineage.py --gate            -> PASS
```

## Evals run

```text
evals/run_reachable_replay_eval.py [--no-hub] on live data (read-only, orion-athena-prediction network)
paired hub vs no-hub replay over sensor-covered origins (scratch script) -- numbers above and in the spec
```

## Docker/build/smoke checks

```text
Tests and replays ran in the orion-prediction-prediction image. No rebuild or deploy in this PR.
```

## Review findings fixed

- Finding: the doc's "with vs without Hub" table could not be reproduced by the shipped eval (`--no-hub` set every row unknown, so its covered block was empty).
  - Fix: the replay now computes both arms on the same origins itself (`hub_paired`, from the first full Denver day after the sensor), with the daily gap via `_gap`.
  - Evidence: re-run on live data gives all -0.028 (se 0.021, P 0.90), cold -0.013 (se 0.011, P 0.87); docs now quote those.
- Finding (minor): `hub_unknown` is an era flag and also absorbs base-rate drift.
  - Fix: said so in the docstring. Harmless: a live origin is always after the sensor started.
- Finding (minor): "first input with cold-hour skill" overstated P 0.87 over 14 days.
  - Fix: now "first sign of cold-hour skill".
- Finding (minor): `_ns_td` truncated to whole seconds.
  - Fix: exact integer division by 1 microsecond.
- Finding (minor): a broken tracker with the Hub up reads as "not looking" with no flag.
  - Fix: filed #3926 (follow-up).
- No leakage or window off-by-one found; the reviewer checked both.

## Restart required

```bash
scripts/safe_docker_build.sh orion-prediction up -d --build   # from clean primary main after merge
```

## Risks / concerns

- Severity: low. Concern: 14 days of sensor data; the gain is concentrated in the second week. Mitigation: the priors carry the rule; the grade keeps reporting P(better) as data grows.
- Severity: low. Concern: a Hub outage reads as "not looking" and lowers p. Mitigation: documented as honest (she cannot look at a Hub that is down); no consumer acts on this forecast yet.
- Live path: UNVERIFIED until deployed (first live row with `features.hub_lt15m`).
- Severity: low. Concern: live tick sees a minute's row only if written before the issue time (rows land at second :30). UNVERIFIED; check `last_at` vs `issued_at` on a few live rows after deploy.
- Issues opened: #3926 (silent tracker detection). Issues closed: #3578.

## PR link

TBD
