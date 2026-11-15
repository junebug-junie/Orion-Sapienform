# orion-lane-door

A small proxy in front of each model server. It writes down every model call that did not come through
orion-llm-gateway. In `shadow` mode (the default) it refuses nothing.

Spec: `docs/superpowers/specs/2026-11-12-leased-evals-and-server-refusal-design.md` (Phase 2).
Runbook (check, roll back, Phase 3): `docs/runbooks/2026-11-14-lane-door.md`.

## What it does

- The gateway adds one header to every call it sends to a model server under a GPU pool grant:
  `X-Orion-Lease-Id` (the grant). There is no shared secret and nothing to configure.
- The door reads it: `leased` (a lease id), `unleased` (none) or `invalid` (not a lease id). It removes the
  header and passes the call to the engine byte for byte. Streams pass as they arrive. If the caller hangs
  up, the door closes the engine connection, so the engine stops generating.
- It writes one JSON line per model call to `<LANE_DOOR_LOG_HOST_DIR>/<lane>-YYYY-MM-DD.jsonl`:
  time, lane, source address, method, path, lease state, lease id, status, duration. It never writes the
  prompt. An unleased call also goes to the container log (`lane_door_unleased ...`).
- `GET /door/stats` returns counters since start, with unleased calls by source and by path.
- No-model reads always pass, with or without a lease id, and are only counted:
  `GET /health /props /slots /v1/models /models /metrics /orion/server-info`, `POST /tokenize /detokenize /apply-template`.
  The GPU pool's probes and the lane controller's ready check use these.
- It asks the GPU pool nothing per call. A pool lookup would sit on the chat path. The lease ids in the
  log are joined to `gpu_pool_events` later, by `scripts/lane_door_report.py`: that is what catches a
  made-up lease id.

What it is for: catching ACCIDENTAL bypass, an eval or script of Orion's that calls a model server
directly instead of the gateway. It is not a fence against a deliberate caller. Any caller can send a
lease-id header, and on athena the Hub holds the docker socket, so a secret would not stop a local caller
either. circe's port gate (`scripts/ops/circe_llm_port_gate.sh`) keeps every host but athena off the ports.

## How it takes the lane port

The doors run on the host network. Each engine publishes only on loopback (`127.0.0.1:180xx`, the compose
default of its `*_ENGINE_PUBLISH`). A door binds its lane port (8011...) only once its engine answers on
that loopback port. Until then it waits and logs `lane_door_waiting` once a minute; its health check reads
healthy while it waits. So:

- the door never takes a port that an engine still publishes (an engine not moved yet, or restarting
  after a crash, keeps its port);
- the deploy order does not matter;
- agent-gpu2's door waits while the pool has that seat unloaded, and takes 8016 at the next load.

Deploy is automatic. On circe, the post-pull rebuild starts the four circe doors (`include_services_circe.txt`).
`scripts/pre_rebuild.sh` creates `.env` from `.env_example` if it is missing. `scripts/post_rebuild.sh`
runs `scripts/move_engines_behind_doors.py`. That recreates each running chat/agent/fast engine still on its
public lane port, so it comes back on loopback. Each later deploy is a no-op for the engines. On hecate,
`services/orion-vllm-host/scripts/hecate-deep.sh up` starts the agent-deep door after the engine, and
`stop`/`down` stops it first.

On a stop (a door redeploy) the door stops taking new calls and lets calls in flight finish, up to
`LANE_DOOR_DRAIN_SEC` (100 s). New calls during that window are refused at connect (about the door's
restart time, plus the drain).

Other details:

- `Expect: 100-continue` is answered by the door itself; an engine's interim 1xx is never relayed as the
  reply. A caller that sends no complete request head in 30 s is dropped.
- Host network: the logged source is the caller's real address (athena's tailnet address, or a circe
  container's own bridge address).
- When the engine is down after the door took the port, the door answers 502
  `lane_door_upstream_unreachable`. The pool sees a down seat, as without a door.

Modes (`LANE_DOOR_MODE`, per lane `LANE_DOOR_MODE_<LANE>`): `off` (pure pass-through, nothing counted),
`shadow` (count and log, refuse nothing), `enforce` (Phase 3: a model call without a lease id gets
401 `lane_door_unleased` and never reaches the engine; exempt reads still pass).

## Cost on the call path

Measured with `evals/measure_latency.py` (a new connection per call, as the gateway does):

| case | added time, p50 | p99 |
|---|---|---|
| local engine stand-in, POST chat, leased and logged | 0.3 ms | 0.4 ms |
| local stand-in, first byte of a stream | 0.8 ms | 1.4 ms |
| live circe fast engine, `GET /props` (door on athena; 2026-11-14) | 0.8 ms | 1.1 ms |
| live circe chat engine (Strata), `GET /props` (2026-11-14) | 0.5 ms | 0.7 ms |

## Layout

- `app/door.py`: the proxy (stdlib asyncio, one request per connection, `Connection: close`), and
  `take_port` / `drain` / `serve`.
- `app/settings.py`: environment settings.
- `app/health.py`: the container health check.
- `app/main.py`: `python -m app.main`.
- `docker-compose.yml`: one service per lane, host network. No profile (circe): chat 8011, agent 8015,
  agent-gpu2 8016, fast 8013. Profile `hecate`: agent-deep 8021.
- `scripts/pre_rebuild.sh`, `scripts/post_rebuild.sh`, `scripts/move_engines_behind_doors.py`: the deploy
  hooks above.
- `evals/measure_latency.py`: the latency measurement above.
- `tests/test_door.py`: the proxy against a real socket engine stand-in. `tests/test_move_engines.py`:
  the engine-move hook, and the loopback ports the doors and engines agree on.

Stdlib only: `requirements.txt` lists nothing to install.

## Tests

```bash
cd services/orion-lane-door && python -m pytest tests -q
python evals/measure_latency.py --n 400                       # local stand-in
python evals/measure_latency.py --n 300 --engine 100.112.254.99:8013   # live engine, GET /props only
```
