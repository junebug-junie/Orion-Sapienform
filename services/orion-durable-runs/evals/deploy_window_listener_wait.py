"""Periodic eval: the 2026-10-31 deploy-window reading losses, replayed. Disposable Postgres DSN only.

On 2026-10-31, 16 reading runs failed ``turn_exception:RpcNoListener: ... request published to
nobody`` in four bursts, each a Hub redeploy: the reading turn RPC went out while no Hub was
subscribed, and the durable run went terminal ``failed``. This replays one such window: 16 reading
runs whose Hub turn is unheard (``RpcPublishedToNobody``) while "Hub is down", through the real
durable-runs runtime, the real in-process GPU pool, the real client + codec and real Postgres
checkpoints -- once with the listener wait OFF (the incident) and once ON (the fix).

Checks (the fix):
  no_run_failed        -- zero run.failed during or after the deploy window
  all_completed        -- every run completes once Hub is back
  no_attempt_spent     -- every run ends with attempt == 1 (its one real turn)
  waits_visible        -- every run unheard in the window has >=1 run.listener_wait event
  waits_bounded        -- no run waited more than DURABLE_RUNS_LISTENER_WAIT_MAX_TRIES times
  no_turn_during_wait  -- Hub turns sent while down <= runs * max_tries (no retry storm)
Control (wait OFF): incident_reproduced -- every run unheard in the window fails, as on 10-31.

In-flight window (2026-11-03: 3 readings failed ``RpcNoListener ... while awaiting its reply`` in
Hub redeploys at 18:12 and 18:26 UTC): Hub received each run's turn, then went down mid-turn; any
re-send while Hub is still down is published to nobody. Checks:
  inflight_no_run_failed / inflight_all_completed / inflight_no_attempt_spent -- as above
  inflight_resends_visible -- every run lost mid-turn has a run.listener_wait with case lost_in_flight
  inflight_resends_bounded -- at most reading_graph.IN_FLIGHT_RESENDS (2) such re-sends per run
Control (reading's opt-in set to 0): inflight_incident_reproduced -- every run fails as on 11-03.

Prints inspectable JSON; exits non-zero on any failed check. No model inference or production bus.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

SERVICE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SERVICE.parents[1]), str(SERVICE), str(SERVICE / "tests")]

import app.reading_graph as reading_graph  # noqa: E402
from orion.core.bus.async_service import RpcNoListener, RpcPublishedToNobody  # noqa: E402
from orion.schemas.reading_turn import ReadingTurnResultV1  # noqa: E402
from pool_fixture import InProcessPool  # noqa: E402
from test_admission_runtime_postgres import Runner, with_database  # noqa: E402
from test_listener_wait_postgres import AWAITING, NOBODY, reading_request, reading_runtime  # noqa: E402

RUNS = 16                 # the 2026-10-31 count
DEPLOY_WINDOW_SEC = 1.0   # Hub unsubscribed
BASE_SEC, MAX_SEC, MAX_TRIES = 0.1, 0.4, 6   # the live 60 s / 300 s / 6, scaled down
TICK_SEC = 0.02


class DeployHub(Runner):
    """A Hub reading turn that nobody hears while ``down``. ``in_flight``: Hub received each run's
    first turn and went down mid-turn ("while awaiting its reply"); later sends while down reach nobody."""

    def __init__(self, saver, in_flight: bool = False):
        super().__init__(saver)
        self.down = True
        self.in_flight = in_flight
        self.lost: dict[str, int] = {}
        self.unheard: dict[str, int] = {}
        self.answered: dict[str, int] = {}

    async def _run_reading_turn(self, request):
        if self.down and self.in_flight and request.run_id not in self.lost:
            self.lost[request.run_id] = 1
            raise RpcNoListener(AWAITING)
        if self.down:
            self.unheard[request.run_id] = self.unheard.get(request.run_id, 0) + 1
            raise RpcPublishedToNobody(NOBODY)
        self.answered[request.run_id] = self.answered.get(request.run_id, 0) + 1
        return ReadingTurnResultV1(run_id=request.run_id, correlation_id=request.correlation_id, ok=True,
                                   text="A grounded reading.", source_fetches=[])


async def replay(pool, saver, store, *, max_tries: int, in_flight: bool = False) -> dict:
    gpu = await InProcessPool().boot()
    hub = DeployHub(saver, in_flight)
    rt = reading_runtime(pool, saver, store, gpu, hub, DURABLE_RUNS_LISTENER_WAIT_MAX_TRIES=max_tries,
                         DURABLE_RUNS_LISTENER_WAIT_BASE_SEC=BASE_SEC, DURABLE_RUNS_LISTENER_WAIT_MAX_SEC=MAX_SEC,
                         DURABLE_RUNS_HOLD_STATUS_POLL_SEC=0.05)
    tag = uuid4().hex[:6]
    ids = []
    for i in range(RUNS):
        req = reading_request(f"deploy-{tag}-{i:02d}")
        ids.append(req.run_id)
        await rt.submit(req)

    async def drive_all():
        for run_id in ids:
            row = await store.get_run(run_id)
            if row and not row.get("terminal"):
                await rt._drive(row)

    started = time.monotonic()
    while time.monotonic() - started < DEPLOY_WINDOW_SEC:
        await drive_all()
        await asyncio.sleep(TICK_SEC)
    hub.down = False
    for _ in range(300):
        await drive_all()
        if all([(await store.get_run(r))["terminal"] for r in ids]):
            break
        await asyncio.sleep(TICK_SEC)
    finals = {}
    for r in ids:
        snap = await rt._graph_for("reading.turn").aget_state(rt.config(r))
        finals[r] = dict(snap.values)
    await rt.close()
    histories = {r: await store.history(r) for r in ids}
    return {
        "terminals": {r: (await store.get_run(r))["terminal"] for r in ids},
        "attempts": {r: int(v.get("attempt") or 0) for r, v in finals.items()},
        "waits": {r: [e["detail"]["waits"] for e in h if e["event"] == "run.listener_wait"]
                  for r, h in histories.items()},
        "in_flight_waits": {r: [e["detail"]["waits"] for e in h if e["event"] == "run.listener_wait"
                                and e["detail"].get("case") == "lost_in_flight"] for r, h in histories.items()},
        "failed": {r: [e["detail"].get("error") for e in h if e["event"] == "run.failed"]
                   for r, h in histories.items()},
        "unheard": dict(hub.unheard), "answered": dict(hub.answered), "lost": dict(hub.lost),
    }


async def main() -> int:
    if not os.environ.get("ORION_ADMISSION_TEST_DSN"):
        print("ORION_ADMISSION_TEST_DSN (a disposable database) is required", file=sys.stderr)
        return 2
    results: dict = {}

    async def run(pool, saver, store):
        results["off"] = await replay(pool, saver, store, max_tries=0)
        results["on"] = await replay(pool, saver, store, max_tries=MAX_TRIES)
        results["inflight"] = await replay(pool, saver, store, max_tries=MAX_TRIES, in_flight=True)
        opted_in, reading_graph.IN_FLIGHT_RESENDS = reading_graph.IN_FLIGHT_RESENDS, 0
        try:
            results["inflight_off"] = await replay(pool, saver, store, max_tries=MAX_TRIES, in_flight=True)
        finally:
            reading_graph.IN_FLIGHT_RESENDS = opted_in

    await asyncio.wait_for(with_database(run), 180)
    off, on = results["off"], results["on"]
    fl, fl_off = results["inflight"], results["inflight_off"]
    lost = [r for r, n in fl["lost"].items() if n]
    lost_off = [r for r, n in fl_off["lost"].items() if n]
    unheard_on = [r for r, n in on["unheard"].items() if n]
    unheard_off = [r for r, n in off["unheard"].items() if n]
    checks = {
        "incident_reproduced": bool(unheard_off) and all(off["terminals"][r] == "failed" for r in unheard_off),
        "no_run_failed": not any(on["failed"].values()),
        "all_completed": all(t == "completed" for t in on["terminals"].values()),
        "no_attempt_spent": all(a == 1 for a in on["attempts"].values()),
        "waits_visible": bool(unheard_on) and all(on["waits"][r] for r in unheard_on),
        "waits_bounded": all(len(w) <= MAX_TRIES for w in on["waits"].values()),
        "no_turn_during_wait": sum(on["unheard"].values()) <= RUNS * MAX_TRIES,
        "inflight_incident_reproduced": bool(lost_off) and all(
            fl_off["terminals"][r] == "failed" and any("while awaiting its reply" in str(e) for e in fl_off["failed"][r])
            for r in lost_off),
        "inflight_no_run_failed": not any(fl["failed"].values()),
        "inflight_all_completed": all(t == "completed" for t in fl["terminals"].values()),
        "inflight_no_attempt_spent": all(a == 1 for a in fl["attempts"].values()),
        "inflight_resends_visible": len(lost) == RUNS and all(fl["in_flight_waits"][r] for r in lost),
        "inflight_resends_bounded": all(len(w) <= reading_graph.IN_FLIGHT_RESENDS for w in fl["in_flight_waits"].values()),
    }
    report = {
        "runs": RUNS, "deploy_window_sec": DEPLOY_WINDOW_SEC,
        "wait_off": {"failed": sum(t == "failed" for t in off["terminals"].values()),
                     "completed": sum(t == "completed" for t in off["terminals"].values()),
                     "sample_error": next((e[0] for e in off["failed"].values() if e), None)},
        "wait_on": {"failed": sum(t == "failed" for t in on["terminals"].values()),
                    "completed": sum(t == "completed" for t in on["terminals"].values()),
                    "runs_unheard": len(unheard_on), "turns_unheard": sum(on["unheard"].values()),
                    "max_waits_one_run": max((len(w) for w in on["waits"].values()), default=0)},
        "in_flight": {"failed": sum(t == "failed" for t in fl["terminals"].values()),
                      "completed": sum(t == "completed" for t in fl["terminals"].values()),
                      "runs_lost_mid_turn": len(lost),
                      "max_in_flight_resends_one_run": max((len(w) for w in fl["in_flight_waits"].values()), default=0),
                      "opt_in_off_failed": sum(t == "failed" for t in fl_off["terminals"].values()),
                      "opt_in_off_sample_error": next((e[0] for e in fl_off["failed"].values() if e), None)},
        "checks": checks,
    }
    failed = [name for name, ok in checks.items() if not ok]
    report["verdict"] = "FAIL" if failed else "PASS"
    report["failed_checks"] = failed
    print(json.dumps(report, indent=2, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
