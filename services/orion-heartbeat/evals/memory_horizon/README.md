# Heartbeat memory-horizon replay (2026-11-08)

Question: what does heartbeat's `mean_ratio` actually carry? Reproducible signal or dice,
event counts or sequence, and how far back does it remember?

Method: replay 2.5 h of real routed atoms (14,510; biometrics 67%, bus 30%, cortex-exec 2%,
hub 24 events, cortex-orch 0) through the shipped `EnsembleSubstrate` (N=8), decay/reheat
every 2 s, H1 every 30 s, reheat probability held at 0.0076 (bus history is not stored).
Five arms: seeds 42/142/242 on the real stream; organ labels shuffled; event content
shuffled across the real timestamps.

Run (throwaway containers, never the live service):

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "copy (select extract(epoch from emitted_at), source_service, event_json->'atom'->>'atom_type', event_json->'atom'->>'confidence', event_json->'atom'->>'salience', event_json->'atom'->>'uncertainty' from grammar_events where event_json->>'event_kind'='atom_emitted' and source_service in ('orion-hub','orion-biometrics','orion-cortex-exec','orion-bus','orion-cortex-orch') and emitted_at between now()-interval '3 hours' and now()-interval '30 minutes' order by emitted_at) to stdout with csv" > $W/atoms.csv
docker run -d --name hb_real_42 -e PYTHONPATH=/app -e OPENBLAS_NUM_THREADS=1 -v $W:/w -w /app --entrypoint python orion-heartbeat-heartbeat /w/replay.py real 42
# ... real 142, real 242, organ_shuffle 42, order_shuffle 42 (~30 min each, run in parallel)
docker run --rm -v $W:/w --entrypoint python orion-heartbeat-heartbeat /w/analyze.py
docker run --rm -v $W:/w --entrypoint python orion-heartbeat-heartbeat /w/analyze2.py
```

Result (2026-11-08 window 02:17-04:47 UTC):

| Check | Result |
|---|---|
| mean_ratio, seed 42 vs 142 / 242 | r = +0.91 / +0.91 (reproducible) |
| verdict class agreement across seeds | 69-70% |
| verdict agreement, real vs organ-shuffle / order-shuffle | 67% / 68% (verdict no better than shuffled input) |
| mean_ratio, real vs organ-shuffle / order-shuffle | r = +0.13 / +0.03 (depends on which organ, in what order) |
| leaky per-organ event counts (tau 60-900 s) | R^2 0.37-0.40 |
| organ + atom-type mix of the last 1 / 3 / 10 / 30 events (held-out) | R^2 0.79 / 0.88 / 0.60 / -0.03 |
| seed-to-seed ceiling | R^2 0.85 |

Reading: the reproducible part of mean_ratio is almost entirely the mix of the last ~3
events (about 1-2 s of traffic). It is a short-memory sequence meter, not a count meter
and not an integrator. The verdict class is noise.
