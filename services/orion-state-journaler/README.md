# orion-state-journaler

Heartbeat-only as of 2026-10-10.

This service used to roll up `orion:spark:state:snapshot` (valence, arousal,
coherence, novelty) plus equilibrium distress into the Postgres table
`spark_state_rollups`. The spark channel lost its only real producer when
`orion-spark-introspector` was deleted on 2026-07-28, so from then on every
rollup row was 0.0 with `pct_missing=1.0`. The rollup, both bus
subscriptions, the Postgres writer and their env keys were removed.

What is left:

- the standard `SystemHealthV1` heartbeat on `orion:system:health`, so
  equilibrium's expected-services check keeps seeing the container;
- `GET /rollups` returns `410 Gone` so no caller mistakes the frozen table for
  live state.

`spark_state_rollups` was not dropped. It is frozen history: last real spark
row 2026-07-28 07:09 UTC, last row of any kind written by the final deployed
rollup build.

Follow-up candidate: retire the container outright (remove `state-journaler`
from `EQUILIBRIUM_EXPECTED_SERVICES` and the `state_journaler` organ in
`orion/signals/registry.py`, which needs a metric-lock refresh).

Tests: `PYTHONPATH=.:services/orion-state-journaler python -m pytest -q services/orion-state-journaler/tests`
