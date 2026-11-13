"""Graphiti full-engine trial (spec 2026-11-13 core memory pages, Patch 0): go/no-go.

Feeds Orion's episode memories (and the 44 high-salience stance messages) through Graphiti's real
``add_episode`` (entity resolution, facts as timed edges, contradicted facts invalidated) on the
local agent lane, into the FalkorDB graph ``graphiti_trial`` ONLY. Nothing reads that graph.

Runs inside the adapter image (graphiti-core 0.19 is installed there), never inside the live
adapter container:

    docker run --rm --network app-net --user 1000:1000 -e PYTHONDONTWRITEBYTECODE=1 \
      -v <worktree>/services/orion-graphiti-adapter/scripts:/trial:ro -v <worktree>/orion:/app/orion:ro \
      -v /tmp/graphiti-trial:/tmp/graphiti-trial \
      orion-graphiti-adapter-graphiti-adapter python /trial/graphiti_trial.py run --limit 3

LLM seat: one GPU pool HOLD (work class ``agent``, priority background, holder
``graphiti-trial:<run>``) for the whole run. Every Graphiti call goes to llm-gateway's OpenAI
passthrough (route ``agent``) with the hold ref in ``X-Orion-Gpu-Lease``; the gateway attaches it
to the hold's seat. A recalled hold is released between episodes and taken again.

Resumable: an episode whose Episodic node (name = <origin>:<source row id>) is already in the
graph is skipped. Progress: /tmp/graphiti-trial/progress.log; per-episode cost: results.jsonl.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
import uuid as uuidlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphiti_trial_clients import (  # noqa: E402
    CallStats, HeaderBox, build_cross_encoder, build_embedder, build_llm_client,
)

logger = logging.getLogger("graphiti_trial")

TRIAL_GRAPH = "graphiti_trial"
# The live graphs this trial must never touch. Checked before and after a run (node counts).
PROTECTED_GRAPHS = ("graphiti_temporal", "orion_worldview")
GROUP_ID = "orion_trial"
OUT_DIR = Path(os.environ.get("GRAPHITI_TRIAL_OUT", "/tmp/graphiti-trial"))

# The rows the go/no-go checks are about. Any episode whose text matches is always in a sample.
KEY_ROW_RE = re.compile(r"chicago|ogden|\bcoco\b|amanda|\bwife\b|\bcats?\b|\bdogs?\b", re.IGNORECASE)

EPISODE_SQL = """
select memory_id::text as id, purpose, voice, statement as body,
       coalesce(occurred_at, created_at) as at, status, (occurred_at is null) as at_from_created
from episode_memory where status in ('active', 'superseded')
"""
STANCE_SQL = """
select crystallization_id::text as id, summary as body, created_at as at
from memory_crystallizations where kind = 'stance' and status = 'active' and salience >= 1
"""


@dataclass(frozen=True)
class TrialEpisode:
    uuid: str
    name: str
    body: str
    source: str            # graphiti EpisodeType value: "text" | "message"
    source_description: str
    reference_time: datetime
    origin: str            # "episode_memory" | "stance"

    def key_row(self) -> bool:
        return bool(KEY_ROW_RE.search(self.body))


def episodes_from_rows(memory_rows: list[dict[str, Any]], stance_rows: list[dict[str, Any]]) -> list[TrialEpisode]:
    """Both sources in one timeline, oldest first (ties broken by id, so the order is stable)."""
    out: list[TrialEpisode] = []
    for r in memory_rows:
        body = str(r["body"] or "").strip()
        if not body:
            continue
        out.append(TrialEpisode(
            uuid=str(r["id"]), name=f"episode_memory:{r['id']}", body=body,
            source="text",
            source_description=(f"Orion's own episode memory, first person (purpose={r['purpose']}, "
                                f"voice={r['voice']}, status={r['status']})"),
            reference_time=_utc(r["at"]), origin="episode_memory"))
    for r in stance_rows:
        body = str(r["body"] or "").strip()
        if not body:
            continue
        out.append(TrialEpisode(
            uuid=str(r["id"]), name=f"stance:{r['id']}", body=f"Juniper: {body}", source="message",
            source_description="Juniper's own chat message to Orion (verbatim, high-salience stance)",
            reference_time=_utc(r["at"]), origin="stance"))
    out.sort(key=lambda e: (e.reference_time, e.uuid))
    return out


def select_sample(episodes: list[TrialEpisode], size: int, seed: int = 20261113) -> list[TrialEpisode]:
    """Every key row plus a seeded random fill up to ``size``, back in time order."""
    keys = [e for e in episodes if e.key_row()]
    rest = [e for e in episodes if not e.key_row()]
    fill = random.Random(seed).sample(rest, k=max(0, min(len(rest), size - len(keys))))
    chosen = {e.uuid for e in keys} | {e.uuid for e in fill}
    return [e for e in episodes if e.uuid in chosen]


def _utc(value: Any) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def fmt_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"


def progress_line(*, done: int, total: int, skipped: int, errors: int, elapsed: float, processed: int,
                  title: str, anomalies: str = "-") -> str:
    rate = processed / elapsed if elapsed > 0 and processed else 0.0
    remaining = total - done
    eta = fmt_eta(remaining / rate) if rate > 0 else "unknown"
    pct = 100.0 * done / total if total else 100.0
    return (f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} graphiti_trial {title} | "
            f"{pct:5.1f}% | ETA {eta} | {done}/{total} (skipped {skipped}) | "
            f"{rate * 3600:.1f} ep/h | errors {errors} | anomalies {anomalies}")


# --- inputs -----------------------------------------------------------------------------------

async def load_episodes(postgres_uri: str) -> list[TrialEpisode]:
    import asyncpg  # noqa: PLC0415
    conn = await asyncpg.connect(postgres_uri)
    try:
        mem = [dict(r) for r in await conn.fetch(EPISODE_SQL)]
        stance = [dict(r) for r in await conn.fetch(STANCE_SQL)]
    finally:
        await conn.close()
    return episodes_from_rows(mem, stance)


# --- graph guards -----------------------------------------------------------------------------

async def graph_node_counts(falkor_host: str, falkor_port: int) -> dict[str, int]:
    from falkordb.asyncio import FalkorDB  # noqa: PLC0415
    db = FalkorDB(host=falkor_host, port=falkor_port)
    names = set(await db.list_graphs())
    out: dict[str, int] = {}
    for g in PROTECTED_GRAPHS:
        if g in names:
            res = await db.select_graph(g).ro_query("MATCH (n) RETURN count(n)")
            out[g] = int(res.result_set[0][0])
    return out


async def ingested_names(driver: Any) -> set[str]:
    """Episode names already in the graph. add_episode(uuid=...) LOADS an existing episode rather
    than naming a new one, so the source row id rides in the episode name instead."""
    rows, _, _ = await driver.execute_query(
        "MATCH (e:Episodic) WHERE e.group_id = $g RETURN e.name AS name", g=GROUP_ID)
    return {r["name"] for r in rows or []}


# --- pool hold --------------------------------------------------------------------------------

async def cancel_lease(bus: Any, lease_id: str, *, source: str) -> None:
    from orion.gpu_pool.client import lease_rpc  # noqa: PLC0415
    from orion.schemas.gpu_pool import GpuLeaseRequestV1  # noqa: PLC0415
    try:
        await lease_rpc(bus, GpuLeaseRequestV1(verb="cancel", lease_id=lease_id), source=source, timeout_sec=5)
        logger.info("hold cancelled lease=%s", lease_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("hold cancel failed lease=%s: %s", lease_id, exc)


class TrialHold:
    """One GPU pool hold for the run; heartbeats in the background; re-taken after a recall."""

    def __init__(self, bus: Any, run_id: str, header_box: HeaderBox, *, min_ctx_tokens: int):
        self.bus, self.run_id, self.box = bus, run_id, header_box
        self.holder = f"graphiti-trial:{run_id}"
        self.min_ctx = min_ctx_tokens
        self.lease_id: str | None = None
        self.role: str | None = None
        self.granted_at: float | None = None
        self.lease_seconds = 0.0
        self.attempt = 0
        self.recalled = asyncio.Event()
        self.lost = asyncio.Event()
        self._beat: asyncio.Task | None = None

    async def acquire(self, wait_sec: float = 3600.0) -> None:
        from orion.gpu_pool.client import acquire_hold, hold_ref, lease_status  # noqa: PLC0415
        from orion.llm.resource_lease import encode_gpu_lease_header  # noqa: PLC0415
        self.attempt += 1
        reply = await acquire_hold(self.bus, holder=self.holder, work_class="agent",
                                   request_id=f"{self.run_id}:{self.attempt}", priority="background",
                                   min_ctx_tokens=self.min_ctx, source=self.holder)
        deadline = time.monotonic() + wait_sec
        last_log = 0.0
        try:
            while reply.status not in ("granted", "recall"):
                if reply.status in ("unavailable", "unknown_lease") or time.monotonic() > deadline:
                    raise RuntimeError(f"hold_not_granted:{reply.status}:{reply.reason}")
                if time.monotonic() - last_log > 300:
                    _log_progress(f"hold waiting status={reply.status} lease={reply.lease_id}")
                    last_log = time.monotonic()
                await asyncio.sleep(10)
                reply = await lease_status(self.bus, reply.lease_id, source=self.holder)
        except BaseException:
            # Never leave a queued hold behind: the pool would grant it to nobody.
            if reply.lease_id:
                await cancel_lease(self.bus, reply.lease_id, source=self.holder)
            raise
        ref = hold_ref(reply, self.holder)
        self.lease_id, self.role = ref.lease_id, ref.role
        self.box.set_hold(encode_gpu_lease_header(ref))
        self.granted_at = time.monotonic()
        self.recalled.clear()
        self.lost.clear()
        self._beat = asyncio.create_task(self._heartbeat())
        logger.info("hold granted lease=%s role=%s generation=%s", ref.lease_id, ref.role, ref.generation)

    async def _heartbeat(self) -> None:
        from orion.gpu_pool.client import heartbeat_lease  # noqa: PLC0415
        while True:
            await asyncio.sleep(25)
            try:
                reply = await heartbeat_lease(self.bus, self.lease_id, source=self.holder)
            except Exception as exc:  # noqa: BLE001 -- TTL is 90 s; one missed beat is survivable
                logger.warning("hold heartbeat failed: %s", exc)
                continue
            if reply.status == "recall":
                self.recalled.set()
            elif reply.status != "granted":
                logger.warning("hold lost status=%s reason=%s", reply.status, reply.reason)
                self.lost.set()
                return

    async def release(self) -> None:
        from orion.gpu_pool.client import release_lease  # noqa: PLC0415
        if self._beat:
            self._beat.cancel()
            self._beat = None
        if self.granted_at is not None:
            self.lease_seconds += time.monotonic() - self.granted_at
            self.granted_at = None
        if self.lease_id:
            try:
                await release_lease(self.bus, self.lease_id, source=self.holder, outcome="ok")
            except Exception as exc:  # noqa: BLE001
                logger.warning("hold release failed: %s", exc)
        self.lease_id = None
        self.box.set_hold(None)

    def held_seconds(self) -> float:
        live = time.monotonic() - self.granted_at if self.granted_at is not None else 0.0
        return self.lease_seconds + live


# --- run --------------------------------------------------------------------------------------

def _log_progress(line: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "progress.log").open("a") as fh:
        fh.write(line + "\n")
    logger.info(line)


async def run(args: argparse.Namespace) -> int:
    if args.graph != TRIAL_GRAPH:
        raise SystemExit(f"refusing graph {args.graph!r}: this trial writes only {TRIAL_GRAPH!r}")
    from graphiti_core import Graphiti  # noqa: PLC0415
    from graphiti_core.driver.falkordb_driver import FalkorDriver  # noqa: PLC0415
    from graphiti_core.nodes import EpisodeType  # noqa: PLC0415
    from graphiti_core.utils.maintenance.graph_data_operations import (  # noqa: PLC0415
        build_indices_and_constraints,
    )
    from orion.core.bus.async_service import OrionBusAsync  # noqa: PLC0415

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    episodes = await load_episodes(args.postgres_uri)
    total_source = len(episodes)
    if args.sample:
        episodes = select_sample(episodes, args.sample)
    if args.limit:
        episodes = episodes[: args.limit]
    (OUT_DIR / "plan.json").write_text(json.dumps(
        {"source_total": total_source, "planned": len(episodes), "sample": args.sample, "limit": args.limit,
         "key_rows": sum(e.key_row() for e in episodes),
         "uuids": [e.uuid for e in episodes]}, indent=1))

    host, port = args.falkor_host, args.falkor_port
    guard_before = await graph_node_counts(host, port)
    driver = FalkorDriver(host=host, port=port, database=TRIAL_GRAPH)
    stats = CallStats()
    box = HeaderBox()
    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=build_llm_client(base_url=args.llm_base_url, model=args.llm_route, header_box=box,
                                    stats=stats, max_tokens=args.max_tokens),
        embedder=build_embedder(args.embed_url),
        cross_encoder=build_cross_encoder(),
        max_coroutines=args.max_coroutines,
    )
    await build_indices_and_constraints(driver)
    done_names = await ingested_names(driver)

    bus = OrionBusAsync(args.bus_url)
    await bus.connect()
    run_id = args.run_id or uuidlib.uuid4().hex[:12]
    hold = TrialHold(bus, run_id, box, min_ctx_tokens=args.min_ctx)
    await hold.acquire()

    total = len(episodes)
    done = skipped = errors = processed = 0
    started = time.monotonic()
    results = (OUT_DIR / "results.jsonl").open("a")
    try:
        for ep in episodes:
            if ep.name in done_names:
                done += 1
                skipped += 1
                continue
            if hold.recalled.is_set() or hold.lost.is_set():
                _log_progress(f"hold {'recalled' if hold.recalled.is_set() else 'lost'}: releasing and re-acquiring")
                await hold.release()
                await hold.acquire()
            before = stats.snapshot()
            held_before = hold.held_seconds()
            t0 = time.monotonic()
            row: dict[str, Any] = {"uuid": ep.uuid, "origin": ep.origin, "reference_time": ep.reference_time.isoformat(),
                                   "key_row": ep.key_row(), "run_id": run_id, "lease_id": hold.lease_id,
                                   "role": hold.role}
            try:
                res = await asyncio.wait_for(graphiti.add_episode(
                    name=ep.name, episode_body=ep.body, source_description=ep.source_description,
                    reference_time=ep.reference_time, source=EpisodeType(ep.source), group_id=GROUP_ID),
                    timeout=args.episode_timeout)
                row.update(ok=True, nodes=len(res.nodes), edges=len(res.edges))
            except Exception as exc:  # noqa: BLE001 -- one bad episode is a data point, not the end
                errors += 1
                row.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:500]}")
            row.update(wall_sec=round(time.monotonic() - t0, 2),
                       lease_sec=round(hold.held_seconds() - held_before, 2), llm=stats.diff(before))
            results.write(json.dumps(row) + "\n")
            results.flush()
            done += 1
            processed += 1
            anomalies = "-"
            if not row["ok"]:
                anomalies = row["error"][:120]
            elif row.get("edges", 0) == 0:
                anomalies = "no_edges"
            _log_progress(progress_line(done=done, total=total, skipped=skipped, errors=errors,
                                        elapsed=time.monotonic() - started, processed=processed,
                                        title=ep.name, anomalies=anomalies))
            if args.max_errors and errors >= args.max_errors:
                _log_progress(f"stopping: {errors} errors reached --max-errors")
                break
    finally:
        results.close()
        await hold.release()
        await bus.close()
        guard_after = await graph_node_counts(host, port)
        changed = {g: (guard_before.get(g), guard_after.get(g)) for g in PROTECTED_GRAPHS
                   if guard_before.get(g) != guard_after.get(g)}
        summary = {"run_id": run_id, "done": done, "skipped": skipped, "errors": errors, "processed": processed,
                   "wall_sec": round(time.monotonic() - started, 1), "lease_sec": round(hold.lease_seconds, 1),
                   "llm": stats.snapshot(), "llm_error_samples": stats.errors[:20],
                   "protected_graphs_before": guard_before, "protected_graphs_after": guard_after,
                   "protected_graphs_changed": changed}
        (OUT_DIR / f"run-{run_id}.json").write_text(json.dumps(summary, indent=1))
        _log_progress(f"run {run_id} finished: {json.dumps({k: summary[k] for k in ('done', 'errors', 'wall_sec', 'lease_sec')})}"
                      f" protected_graphs_changed={changed or 'none'}")
    return 0 if errors == 0 else 2


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--graph", default=TRIAL_GRAPH)
    r.add_argument("--limit", type=int, default=0, help="first N planned episodes (0 = all)")
    r.add_argument("--sample", type=int, default=0, help="key rows + seeded random fill to N (0 = all)")
    r.add_argument("--run-id", default="")
    r.add_argument("--postgres-uri", default=os.environ.get(
        "POSTGRES_URI", "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"))
    r.add_argument("--falkor-host", default="orion-athena-falkordb")
    r.add_argument("--falkor-port", type=int, default=6379)
    r.add_argument("--embed-url", default=os.environ.get(
        "CRYSTALLIZER_EMBED_HOST_URL", "http://orion-athena-vector-host:8320/embedding"))
    r.add_argument("--llm-base-url", default="http://orion-llm-gateway:8210/v1")
    r.add_argument("--llm-route", default="agent")
    r.add_argument("--bus-url", default=os.environ.get("ORION_BUS_URL", "redis://100.92.216.81:6379/0"))
    r.add_argument("--min-ctx", type=int, default=16384)
    r.add_argument("--max-tokens", type=int, default=8192)
    r.add_argument("--max-coroutines", type=int, default=4)
    r.add_argument("--episode-timeout", type=float, default=1800.0)
    r.add_argument("--max-errors", type=int, default=0, help="stop after N failed episodes (0 = never)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    for noisy in ("httpx", "orion.bus.async", "orion.bus"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return asyncio.run(_run_with_sigterm(args))


async def _run_with_sigterm(args: argparse.Namespace) -> int:
    """docker stop sends SIGTERM: cancel the run task so the finally blocks release the hold."""
    import signal  # noqa: PLC0415
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        return await run(args)
    except asyncio.CancelledError:
        _log_progress("stopped by SIGTERM (hold released)")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
