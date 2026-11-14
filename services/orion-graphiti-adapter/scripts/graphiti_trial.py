"""Graphiti full-engine trial (spec 2026-11-13 core memory pages, Patch 0): go/no-go.

Feeds Orion's episode memories (and the 44 high-salience stance messages) through Graphiti's real
``add_episode`` (entity resolution, facts as timed edges, contradicted facts invalidated) on the
local agent lane, into the FalkorDB graph ``graphiti_trial`` ONLY. Nothing reads that graph.

Runs inside the adapter image (graphiti-core 0.19 is installed there), never inside the live
adapter container:

    docker run --rm --stop-timeout 60 --network app-net --user 1000:1000 -e PYTHONDONTWRITEBYTECODE=1 \
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
import contextlib
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
GROUP_ID = "orion_trial"
OUT_DIR = Path(os.environ.get("GRAPHITI_TRIAL_OUT", "/tmp/graphiti-trial"))



def key_row_re(terms: dict[str, str] | None) -> re.Pattern:
    """The rows the go/no-go checks are about; any episode that matches is always in a sample. The
    names come from the gitignored terms file (personal terms stay out of the public repo)."""
    words = [r"\bcats?\b", r"\bdogs?\b"]
    for k in ("home_city", "trip_city", "trip_hotel", "spouse", "spouse_alias", "pet"):
        if terms and str(terms.get(k) or "").strip():
            words.append(r"\b" + re.escape(str(terms[k]).strip()) + r"\b")
    return re.compile("|".join(words), re.IGNORECASE)


KEY_ROW_RE = key_row_re(None)

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
            uuid=str(r["id"]), name=f"stance:{r['id']}", body=f"Juniper (to Orion): {body}", source="message",
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


def plan_episodes(episodes: list[TrialEpisode], names: list[str]) -> list[TrialEpisode]:
    """Exactly the planned episodes, in time order; refuses a plan naming rows that are gone."""
    wanted = set(names)
    out = [e for e in episodes if e.name in wanted]
    missing = wanted - {e.name for e in out}
    if missing:
        raise SystemExit(f"plan names {len(missing)} episodes no longer in the source: {sorted(missing)[:5]}")
    return out


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

async def isolation_snapshot(falkor_host: str, falkor_port: int) -> dict[str, Any]:
    """Graph names, plus how many nodes carry the trial's group_id in every OTHER graph. Live
    graphs (graphiti_temporal, orion_worldview) change during a run on their own, so node counts
    prove nothing; a trial write anywhere else would show up here as a non-zero count."""
    from falkordb.asyncio import FalkorDB  # noqa: PLC0415
    db = FalkorDB(host=falkor_host, port=falkor_port)
    try:
        names = sorted(await db.list_graphs())
        leaked: dict[str, int] = {}
        for g in names:
            if g == TRIAL_GRAPH:
                continue
            res = await db.select_graph(g).ro_query("MATCH (n) WHERE n.group_id = $g RETURN count(n)",
                                                    params={"g": GROUP_ID})
            n = int(res.result_set[0][0])
            if n:
                leaked[g] = n
        return {"graphs": names, "trial_group_nodes_elsewhere": leaked}
    finally:
        with contextlib.suppress(Exception):
            await db.connection.aclose()


def isolation_verdict(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    new = sorted(set(after["graphs"]) - set(before["graphs"]) - {TRIAL_GRAPH})
    return {"new_graphs_other_than_trial": new, "trial_group_nodes_elsewhere": after["trial_group_nodes_elsewhere"],
            "isolated": not new and not after["trial_group_nodes_elsewhere"]}


async def graph_episode_names(driver: Any) -> dict[str, str]:
    """name -> uuid of episodes in the graph. add_episode(uuid=...) LOADS an existing episode rather
    than naming a new one, so the source row id rides in the episode name instead."""
    rows, _, _ = await driver.execute_query(
        "MATCH (e:Episodic) WHERE e.group_id = $g RETURN e.name AS name, e.uuid AS uuid", g=GROUP_ID)
    return {r["name"]: r["uuid"] for r in rows or []}


def ok_names_from_results(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("ok") and r.get("name"):
                out.add(r["name"])
    return out


def resume_plan(graph_names: dict[str, str], ok_names: set[str]) -> tuple[set[str], list[str]]:
    """(done, partial uuids to remove). Done = in the graph AND recorded ok. An episode in the graph
    without an ok row was cut off mid-write (FalkorDB bulk save is not one transaction)."""
    done = set(graph_names) & ok_names
    partial = [uuid for name, uuid in graph_names.items() if name not in ok_names]
    return done, partial


# --- pool hold --------------------------------------------------------------------------------

async def cancel_lease(bus: Any, lease_id: str, *, source: str) -> None:
    from orion.gpu_pool.client import lease_rpc  # noqa: PLC0415
    from orion.schemas.gpu_pool import GpuLeaseRequestV1  # noqa: PLC0415
    try:
        await asyncio.shield(lease_rpc(bus, GpuLeaseRequestV1(verb="cancel", lease_id=lease_id),
                                       source=source, timeout_sec=5))
        logger.info("hold cancelled lease=%s", lease_id)
    except BaseException as exc:  # noqa: BLE001
        logger.warning("hold cancel failed lease=%s: %r", lease_id, exc)


HOLD_FILE = "hold.json"


class TrialHold:
    """One GPU pool hold for the run; heartbeats in the background; re-taken after a recall.

    The live lease id is written to OUT_DIR/hold.json while held, so a run killed hard (SIGKILL)
    leaves a record the next run cancels before asking again (``cancel_leftover``)."""

    TERMINAL = ("unavailable", "unknown_lease", "ok")

    def __init__(self, bus: Any, run_id: str, header_box: HeaderBox, *, min_ctx_tokens: int,
                 out_dir: Path | None = None):
        self.bus, self.run_id, self.box = bus, run_id, header_box
        self.holder = f"graphiti-trial:{run_id}"
        self.min_ctx = min_ctx_tokens
        self.out_dir = out_dir or OUT_DIR
        self.lease_id: str | None = None
        self.role: str | None = None
        self.granted_at: float | None = None
        self.lease_seconds = 0.0
        self.attempt = 0
        self.recalled = asyncio.Event()
        self.lost = asyncio.Event()
        self._beat: asyncio.Task | None = None

    def _record(self, lease_id: str | None) -> None:
        f = self.out_dir / HOLD_FILE
        if lease_id:
            f.write_text(json.dumps({"lease_id": lease_id, "holder": self.holder}))
        elif f.exists():
            f.unlink()

    async def cancel_leftover(self) -> None:
        f = self.out_dir / HOLD_FILE
        if f.exists():
            old = json.loads(f.read_text())
            _log_progress(f"cancelling leftover hold from a killed run: {old}")
            await cancel_lease(self.bus, old["lease_id"], source=old.get("holder") or self.holder)
            f.unlink()

    async def acquire(self, wait_sec: float = 4 * 3600.0) -> None:
        from orion.gpu_pool.client import acquire_hold, hold_ref, lease_status  # noqa: PLC0415
        from orion.llm.resource_lease import encode_gpu_lease_header  # noqa: PLC0415
        self.attempt += 1
        request_id = f"{self.run_id}:{self.attempt}"
        lease_id: str | None = None

        async def ask():
            return await acquire_hold(self.bus, holder=self.holder, work_class="agent", request_id=request_id,
                                      priority="background", min_ctx_tokens=self.min_ctx, source=self.holder)
        try:
            reply = await ask()
            lease_id = reply.lease_id
            self._record(lease_id)
            deadline = time.monotonic() + wait_sec
            last_log = 0.0
            while reply.status not in ("granted", "recall"):
                if reply.status in self.TERMINAL or time.monotonic() > deadline:
                    raise RuntimeError(f"hold_not_granted:{reply.status}:{reply.reason}")
                if time.monotonic() - last_log > 300:
                    _log_progress(f"hold waiting status={reply.status} lease={reply.lease_id}")
                    last_log = time.monotonic()
                await asyncio.sleep(10)
                reply = await lease_status(self.bus, lease_id, source=self.holder)
            ref = hold_ref(reply, self.holder)
            header = encode_gpu_lease_header(ref)
        except BaseException:
            # Never leave a hold behind: retryable holds re-queue and get granted to nobody.
            if lease_id is None:
                with contextlib.suppress(BaseException):  # idempotent on request_id: learn the id
                    lease_id = (await asyncio.shield(ask())).lease_id
            if lease_id:
                await cancel_lease(self.bus, lease_id, source=self.holder)
            self._record(None)
            raise
        self.lease_id, self.role = ref.lease_id, ref.role
        self.box.set_hold(header)
        self.granted_at = time.monotonic()
        self.recalled.clear()
        self.lost.clear()
        self._beat = asyncio.create_task(self._heartbeat())
        _log_progress(f"hold granted lease={ref.lease_id} role={ref.role} generation={ref.generation}")

    async def _heartbeat(self, every: float = 25.0, max_misses: int = 3) -> None:
        from orion.gpu_pool.client import heartbeat_lease  # noqa: PLC0415
        misses = 0
        while True:
            await asyncio.sleep(every)
            try:
                reply = await heartbeat_lease(self.bus, self.lease_id, source=self.holder)
            except Exception as exc:  # noqa: BLE001 -- TTL is 90 s: 3 misses at 25 s is the limit
                misses += 1
                logger.warning("hold heartbeat failed (%d/%d): %s", misses, max_misses, exc)
                if misses >= max_misses:
                    self.lost.set()
                    return
                continue
            misses = 0
            if reply.status == "recall":
                self.recalled.set()
            elif reply.status != "granted":
                logger.warning("hold lost status=%s reason=%s", reply.status, reply.reason)
                self.lost.set()
                return

    def usable(self) -> bool:
        return not (self.recalled.is_set() or self.lost.is_set())

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
                await asyncio.shield(release_lease(self.bus, self.lease_id, source=self.holder, outcome="ok"))
                self._record(None)
            except BaseException as exc:  # noqa: BLE001 -- hold.json stays: the next run cancels it
                logger.warning("hold release failed: %r", exc)
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
    if args.plan_file:
        # A resumed sample must be the SAME sample: new rows land in episode_memory every day, which
        # changes the seeded random fill, so a re-run of --sample would quietly pick other rows.
        episodes = plan_episodes(episodes, json.loads(Path(args.plan_file).read_text())["names"])
    elif args.sample:
        global KEY_ROW_RE
        from graphiti_trial_score import load_terms  # noqa: PLC0415
        KEY_ROW_RE = key_row_re(load_terms())
        episodes = select_sample(episodes, args.sample)
    if args.limit:
        episodes = episodes[: args.limit]
    run_id = args.run_id or uuidlib.uuid4().hex[:12]
    (OUT_DIR / f"plan-{run_id}.json").write_text(json.dumps(
        {"source_total": total_source, "planned": len(episodes), "sample": args.sample, "limit": args.limit,
         "key_rows": sum(e.key_row() for e in episodes),
         "names": [e.name for e in episodes]}, indent=1))

    host, port = args.falkor_host, args.falkor_port
    iso_before = await isolation_snapshot(host, port)
    driver = FalkorDriver(host=host, port=port, database=TRIAL_GRAPH)
    stats = CallStats()
    box = HeaderBox()
    bus = OrionBusAsync(args.bus_url)
    hold = TrialHold(bus, run_id, box, min_ctx_tokens=args.min_ctx)
    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=build_llm_client(base_url=args.llm_base_url, model=args.llm_route, header_box=box,
                                    stats=stats, max_tokens=args.max_tokens, hold_usable=hold.usable),
        embedder=build_embedder(args.embed_url),
        cross_encoder=build_cross_encoder(),
        max_coroutines=args.max_coroutines,
    )
    await build_indices_and_constraints(driver)
    done_names, partial = resume_plan(await graph_episode_names(driver), ok_names_from_results(OUT_DIR / "results.jsonl"))
    for uuid in partial:
        _log_progress(f"removing partial episode {uuid} (in graph, no ok result row)")
        await graphiti.remove_episode(uuid)

    total = len(episodes)
    done = skipped = errors = processed = 0
    started = time.monotonic()
    work_sec = 0.0  # time spent in episodes only: queue waits for the hold are not ingest rate
    results = (OUT_DIR / "results.jsonl").open("a")
    await bus.connect()
    try:
        await hold.cancel_leftover()
        await hold.acquire()
        for ep in episodes:
            if ep.name in done_names:
                done += 1
                skipped += 1
                continue
            if not hold.usable():
                _log_progress(f"hold {'recalled' if hold.recalled.is_set() else 'lost'}: releasing and re-acquiring")
                await hold.release()
                await hold.acquire()
            before = stats.snapshot()
            held_before = hold.held_seconds()
            t0 = time.monotonic()
            row: dict[str, Any] = {"uuid": ep.uuid, "name": ep.name, "origin": ep.origin,
                                   "reference_time": ep.reference_time.isoformat(), "key_row": ep.key_row(),
                                   "run_id": run_id, "lease_id": hold.lease_id, "role": hold.role}
            try:
                res = await asyncio.wait_for(graphiti.add_episode(
                    name=ep.name, episode_body=ep.body, source_description=ep.source_description,
                    reference_time=ep.reference_time, source=EpisodeType(ep.source), group_id=GROUP_ID),
                    timeout=args.episode_timeout)
                row.update(ok=True, nodes=len(res.nodes), edges=len(res.edges))
            except Exception as exc:  # noqa: BLE001 -- one bad episode is a data point, not the end
                errors += 1
                row.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:500]}")
                left = (await graph_episode_names(driver)).get(ep.name)
                if left:  # cut off mid-write: drop the half-saved episode so a retry starts clean
                    with contextlib.suppress(Exception):
                        await graphiti.remove_episode(left)
                        row["partial_removed"] = True
            work_sec += time.monotonic() - t0
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
                                        elapsed=work_sec, processed=processed,
                                        title=ep.name, anomalies=anomalies))
            if args.max_errors and errors >= args.max_errors:
                _log_progress(f"stopping: {errors} errors reached --max-errors")
                break
    finally:
        results.close()
        await hold.release()
        with contextlib.suppress(BaseException):
            await bus.close()
        iso = isolation_verdict(iso_before, await isolation_snapshot(host, port))
        summary = {"run_id": run_id, "done": done, "skipped": skipped, "errors": errors, "processed": processed,
                   "wall_sec": round(time.monotonic() - started, 1), "lease_sec": round(hold.lease_seconds, 1),
                   "llm": stats.snapshot(), "llm_error_samples": stats.errors[:20], "isolation": iso}
        (OUT_DIR / f"run-{run_id}.json").write_text(json.dumps(summary, indent=1))
        _log_progress(f"run {run_id} finished: {json.dumps({k: summary[k] for k in ('done', 'errors', 'wall_sec', 'lease_sec')})}"
                      f" isolation={iso}")
    return 0 if errors == 0 else 2


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--graph", default=TRIAL_GRAPH)
    r.add_argument("--limit", type=int, default=0, help="first N planned episodes (0 = all)")
    r.add_argument("--sample", type=int, default=0, help="key rows + seeded random fill to N (0 = all)")
    r.add_argument("--run-id", default="")
    r.add_argument("--plan-file", default="", help="re-run exactly the episodes named in a plan-<run>.json")
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
    # Under the pool's 600 s recall grace (config/gpu_pool.yaml hold_clawback_grace_sec), so a
    # recalled hold is let go between episodes, never aborted under one.
    r.add_argument("--episode-timeout", type=float, default=540.0)
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
