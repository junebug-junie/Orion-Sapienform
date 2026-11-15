# orion-lane-door

A small proxy in front of each model server. It writes down every model call that did not come through
orion-llm-gateway. In `shadow` mode (the default) it refuses nothing.

Spec: `docs/superpowers/specs/2026-11-12-leased-evals-and-server-refusal-design.md` (Phase 2).
Runbook (deploy, check, roll back): `docs/runbooks/2026-11-14-lane-door.md`.

## What it does

- The gateway adds two headers to every call it sends to a model server under a GPU pool grant:
  `X-Orion-Lane-Key` (a secret only the gateway and the doors hold) and `X-Orion-Lease-Id` (the grant).
- The door checks the key: `ok`, `missing` or `bad`. It removes both headers and passes the call to the
  engine byte for byte. Streams pass as they arrive. If the caller hangs up, the door closes the engine
  connection, so the engine stops generating.
- It writes one JSON line per model call to `<LANE_DOOR_LOG_HOST_DIR>/<lane>-YYYY-MM-DD.jsonl`:
  time, lane, source address, method, path, key state, lease id, status, duration. It never writes the
  prompt or the key. An unkeyed call also goes to the container log (`lane_door_unkeyed ...`).
- `GET /door/stats` returns counters since start and a hash of the key (`key_hash`, 16 hex characters),
  never the key itself. The gateway compares that hash with its own and logs `lane_door_key_mismatch`.
- No-model reads always pass, with or without a key, and are only counted:
  `GET /health /props /slots /v1/models /models /metrics /orion/server-info`, `POST /tokenize /detokenize /apply-template`.
  The GPU pool's probes and the lane controller's ready check use these.
- It asks the GPU pool nothing per call. A pool lookup would sit on the chat path. The lease ids in the
  log are joined to `gpu_pool_events` later, by `scripts/lane_door_report.py`.

- `Expect: 100-continue` is answered by the door itself; an engine's interim 1xx is never relayed as the
  reply. A caller that sends no complete request head in 30 s is dropped.
- A caller on circe's own host reaches the door through docker's proxy and shows as the bridge gateway
  address (172.x.0.1), not its real address.
- When the engine is down the door answers 502 `lane_door_upstream_unreachable`. The gateway counts that
  as an upstream HTTP error, where without a door it saw a connect error (telemetry only; nothing keys on it).

Modes (`LANE_DOOR_MODE`, per lane `LANE_DOOR_MODE_<LANE>`): `off` (pure pass-through, nothing counted),
`shadow` (count and log, refuse nothing).

## Cost on the call path

Measured with `evals/measure_latency.py` (a new connection per call, as the gateway does):

| case | added time, p50 | p99 |
|---|---|---|
| local engine stand-in, POST chat, keyed and logged | 0.1 ms | 0.6 ms |
| local stand-in, first byte of a stream | 0.9 ms | 1.4 ms |
| live circe fast engine, `GET /props` (door on athena) | 0.8 ms | 1.1 ms |
| live circe chat engine (Strata), `GET /props` | 0.5 ms | 0.7 ms |

## Layout

- `app/door.py`: the proxy (stdlib asyncio, one request per connection, `Connection: close`).
- `app/settings.py`: environment settings and the key hash.
- `app/main.py`: `python -m app.main`.
- `docker-compose.yml`: one service per lane. Profile `circe`: chat 8011, agent 8015, agent-gpu2 8016,
  fast 8013. Profile `hecate`: agent-deep 8021.
- `evals/measure_latency.py`: the latency measurement above.
- `tests/test_door.py`: the proxy against a real socket engine stand-in.

Stdlib only: `requirements.txt` lists nothing to install.

## Tests

```bash
cd services/orion-lane-door && python -m pytest tests -q
python evals/measure_latency.py --n 400                       # local stand-in
python evals/measure_latency.py --n 300 --engine 100.112.254.99:8013   # live engine, GET /props only
```
