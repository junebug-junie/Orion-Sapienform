"""Replay the disk rule (ladder retirement PR 4) over REAL disk and prune history, tick by tick.

Runs the shipped ``app.image_prune.ImagePruneRule`` (and through it ``orion.hardware_watch.rules.
image_prune_verdict``) every 30 s on a simulated clock. Only the I/O is swapped:

- **The disk** is athena's ``orion_biometrics_summary.pressures.disk_capacity`` (orion-biometrics'
  ``collect_disk_capacity``: ``shutil.disk_usage`` on the host's /mnt/docker, every ~30 s). That is
  the max over five mounts; the docker volume is the max on every row checked (the others sit near
  30 %), and it matched the skill's own ``disk_before.used_pct`` within 0.047 points on all 173
  dispatch runs since 10-08 (mean 0.003). A reading older than 300 s reads as unreadable.
- **The skill's answer** comes from a model of the dangling-image count built from dispatch's real
  runs (``substrate_dispatch_results``, kind ``maintain``): each run's ``dangling_before`` and, after a
  real prune, ``dangling_after``. New dangling images between two real runs (``before_i`` minus the
  count the previous run left) are spread evenly over that gap. The replay keeps its own count: it
  grows by those arrivals and drops to 0 when the replay prunes. An ask prunes when that count is
  >= 100 (the skill's ``IMAGE_PRUNE_MIN_DANGLING``) and the disk is >= 70 %, else it declines
  ``declined_nothing_to_reclaim`` -- the skill's own two-part gate.

``--export`` needs ``substrate_dispatch_results``, which the retired dispatch rung wrote (retired
2026-11-07, ladder retirement PR 6). That table was dropped on 2026-11-10, and the prune runs since
then are recorded nowhere else (the disk rule's answers go only to the hardware-watch log and its
/health; no act-ledger or want_ledger row). So ``--export`` reads the runs from a database restored
from the pre-drop backup, named with ``--dispatch-db``; the disk history still comes from the live
database. Against a database without the table it stops with exit code 2, says how to restore it, and
leaves the committed fixture untouched. The committed fixture keeps the replay and its CI gates working.

What it does not model: restarts (each one delays the next ask by the 10 min start delay and resets
the cooldown, so live asks can be a little more often), RPC failures, Juniper's hand prunes.

Usage (from services/orion-hardware-watch):
  PYTHONPATH=../..:. python evals/run_image_prune_replay_eval.py             # committed fixture + gates
  PYTHONPATH=../..:. python evals/run_image_prune_replay_eval.py --export --dispatch-db dispatch_restore
                                                                  # refresh it (athena, docker, restored runs)
  PYTHONPATH=../..:. python evals/run_image_prune_replay_eval.py --sweep     # cooldown calibration
"""
from __future__ import annotations

import argparse
import asyncio
import bisect
import gzip
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVICE = HERE.parent
ROOT = SERVICE.parent.parent
for p in (str(ROOT), str(SERVICE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from orion.hardware_watch.rules import DiskReading, ImagePruneRuleConfig  # noqa: E402

from app.image_prune import ImagePruneRule  # noqa: E402

FIXTURE = HERE / "fixtures" / "image_prune_replay.json.gz"
START = "2026-10-22T00:00:00+00:00"     # 14 days, the spec's audit window
END = "2026-11-05T00:00:00+00:00"
SEED_DAYS = 30                          # the last dispatch run before START seeds the dangling count
TICK_SEC = 30.0
# A reading older than this reads as unreadable. Live, the rule reads the disk itself every tick, so
# only a gap in the biometrics record can make it stale here. 300 s plus one 5-min fixture bucket.
STALE_SEC = 600.0
MIN_DANGLING = 100                      # services/orion-cortex-exec/app/verb_adapters.py IMAGE_PRUNE_MIN_DANGLING
DISK_BUCKET_SEC = 300                   # the fixture keeps the max reading per 5 min (ticks hold it)


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace(" ", "T").replace("+00", "+00:00").replace("+00:00:00", "+00:00"))


LIVE_DB = "conjourney"
DISPATCH_TABLE = "substrate_dispatch_results"
DISPATCH_BACKUP = "/mnt/storage-warm/orion-backups/retired-proposal-policy-dispatch-rungs.dump"


class DispatchHistoryMissing(RuntimeError):
    """The prune-run table is not in the database ``--export`` was pointed at."""


def missing_history_message(db: str) -> str:
    return (f"{DISPATCH_TABLE} is not in database {db!r}: it was dropped on 2026-11-10 (the dispatch rung "
            "retired 2026-11-07), and the prune runs are recorded nowhere else. To export, restore it from "
            f"{DISPATCH_BACKUP} into a scratch database and pass --dispatch-db, e.g.:\n"
            "  docker exec orion-athena-sql-db createdb -U postgres dispatch_restore\n"
            "  docker exec -i orion-athena-sql-db pg_restore -U postgres -d dispatch_restore "
            f"-t {DISPATCH_TABLE} < {DISPATCH_BACKUP}\n"
            "The committed fixture was left as it is.")


def _psql(sql: str, db: str = LIVE_DB) -> list[list[str]]:
    out = subprocess.run(["docker", "exec", "orion-athena-sql-db", "psql", "-U", "postgres", "-d", db,
                          "-AtF", "\t", "-c", sql], check=True, capture_output=True, text=True).stdout
    return [line.split("\t") for line in out.splitlines() if line.strip()]


def _require_dispatch_history(db: str) -> None:
    present = _psql(f"select to_regclass('{DISPATCH_TABLE}') is not null", db)
    if not present or present[0][0] != "t":
        raise DispatchHistoryMissing(missing_history_message(db))


def export(start: str = START, end: str = END, dispatch_db: str = LIVE_DB) -> dict:
    _require_dispatch_history(dispatch_db)   # before any work: a missing table must not write a fixture
    seed = (_ts(start) - timedelta(days=SEED_DAYS)).isoformat()
    disk_rows = _psql(
        "select timestamp, pressures->>'disk_capacity' from orion_biometrics_summary "
        f"where node='athena' and timestamp >= '{start[:10]}' and timestamp < '{end[:10]}' "
        "and pressures->>'disk_capacity' is not null")
    # Per 5-min bucket: the max reading, stamped at the bucket's LAST reading, so a gap between two
    # stamps is a real gap in the record (a stamp at the max's own time could invent a 600 s gap).
    buckets: dict[int, tuple[float, float]] = {}
    for ts, v in disk_rows:
        t = _ts(ts).timestamp()
        b = int(t // DISK_BUCKET_SEC)
        pct = round(100.0 * float(v), 3)
        if b in buckets:
            t = max(t, buckets[b][0])
            pct = max(pct, buckets[b][1])
        buckets[b] = (t, pct)
    disk = [[round(t, 1), pct] for t, pct in sorted(buckets.values())]
    runs = _psql(
        "select created_at, result_json->'structured_result'->>'decision', "
        "result_json->'structured_result'->'dangling_before'->>'dangling_count', "
        "result_json->'structured_result'->'dangling_after'->>'dangling_count', "
        "result_json->'structured_result'->'disk_before'->>'used_pct', "
        "result_json->'structured_result'->>'images_removed' "
        "from substrate_dispatch_results where dispatch_kind='maintain' "
        f"and dispatch_id like '%prune_dangling_images%' and created_at >= '{seed}' and created_at < '{end}' "
        "order by created_at", dispatch_db)
    if not runs:   # an empty restore (schema only, interrupted COPY) must not become an empty fixture
        raise DispatchHistoryMissing(f"{DISPATCH_TABLE} in database {dispatch_db!r} has no prune runs in "
                                     f"{seed} .. {end}; the committed fixture was left as it is")
    obs = []
    for created, decision, before, after, used, removed in runs:
        if not before:
            continue
        obs.append({"t": round(_ts(created).timestamp(), 1), "decision": decision, "before": int(before),
                    "after": int(after) if after else None, "disk_pct": float(used) if used else None,
                    "images_removed": int(removed) if removed else None})
    return {"window": [start, end], "disk_bucket_sec": DISK_BUCKET_SEC, "disk": disk, "runs": obs,
            "exported_at": datetime.now(timezone.utc).isoformat()}


def load() -> dict:
    with gzip.open(FIXTURE, "rt") as f:
        return json.load(f)


class DanglingModel:
    """New dangling images per real gap, spread evenly; the replay's own count from them."""

    def __init__(self, runs: list[dict]):
        self.segments: list[tuple[float, float, float]] = []   # (t0, t1, arrivals)
        self.marks: list[tuple[float, float]] = []             # (t, count the real run left)
        prev_left = None
        prev_t = None
        for r in runs:
            if prev_t is not None:
                arrivals = max(0, r["before"] - prev_left)
                if r["t"] > prev_t and arrivals:
                    self.segments.append((prev_t, r["t"], float(arrivals)))
            left = r["after"] if (r["decision"] == "pruned" and r["after"] is not None) else r["before"]
            prev_left, prev_t = left, r["t"]
            self.marks.append((r["t"], float(left)))
        self._mark_ts = [m[0] for m in self.marks]

    def real_count(self, t: float) -> float | None:
        """The real record's dangling count at ``t`` (the last real run's leftover plus arrivals since)."""
        i = bisect.bisect_right(self._mark_ts, t) - 1
        if i < 0:
            return None
        t0, left = self.marks[i]
        return left + self.arrivals(t0, t)

    def arrivals(self, a: float, b: float) -> float:
        total = 0.0
        for t0, t1, n in self.segments:
            lo, hi = max(a, t0), min(b, t1)
            if hi > lo:
                total += n * (hi - lo) / (t1 - t0)
        return total


def replay(data: dict, *, cooldown_sec: float = ImagePruneRuleConfig().cooldown_sec, mutate=None) -> dict:
    start, end = (_ts(x).timestamp() for x in data["window"])
    disk = data["disk"]
    if mutate:
        disk = mutate(disk)
    dts = [d[0] for d in disk]
    model = DanglingModel(data["runs"])
    sim_t = {"t": start}
    state = {"count": 0.0, "since": start}
    # Seed: the dangling count the real record left at START. With no real run before START, the
    # first run's count is taken as already there at START (it was there when that run measured it;
    # a little optimistic for the replay, disclosed in the PR).
    before_start = [r for r in data["runs"] if r["t"] <= start]
    if before_start:
        last = before_start[-1]
        left = last["after"] if (last["decision"] == "pruned" and last["after"] is not None) else last["before"]
        state["count"] = float(left) + model.arrivals(last["t"], start)
    elif data["runs"]:
        first = data["runs"][0]
        state["count"], state["since"] = float(first["before"]), max(start, first["t"])
    asks: list[dict] = []
    backlog = {"replay": 0.0, "real": 0.0, "n": 0}

    def clock() -> datetime:
        return datetime.fromtimestamp(sim_t["t"], tz=timezone.utc)

    def read_disk(_path: str) -> DiskReading:
        i = bisect.bisect_right(dts, sim_t["t"]) - 1
        if i < 0 or sim_t["t"] - dts[i] > STALE_SEC:
            return DiskReading(None, "stale")
        return DiskReading(disk[i][1])

    async def ask(attempt_id: str) -> dict:
        now = sim_t["t"]
        state["count"] += model.arrivals(state["since"], now)
        state["since"] = now
        reading = read_disk("")
        n = int(state["count"])
        if reading.used_pct is not None and reading.used_pct >= 70.0 and n >= MIN_DANGLING:
            decision, removed = "pruned", n
            state["count"] = 0.0
        else:
            decision, removed = "declined_nothing_to_reclaim", 0
        asks.append({"t": now, "disk_pct": reading.used_pct, "dangling": n, "decision": decision,
                     "removed": removed})
        body = {"decision": decision, "dangling_before": {"available": True, "dangling_count": n},
                "disk_before": {"used_pct": reading.used_pct}, "images_removed": removed}
        return {"ok": True, "result": {"status": "success", "final_text": json.dumps(body)}}

    async def run() -> ImagePruneRule:
        # The process started long before the window: no start delay inside it.
        sim_t["t"] = start - 3600.0
        rule = ImagePruneRule(enabled=True, mount_path="/hostfs/docker",
                              cfg=ImagePruneRuleConfig(cooldown_sec=cooldown_sec), ask=ask,
                              read_disk=read_disk, clock=clock)
        sim_t["t"] = start
        while sim_t["t"] < end:
            await rule.tick()
            if rule._in_flight is not None:
                await asyncio.gather(rule._in_flight)
            real_now = model.real_count(sim_t["t"])
            if real_now is not None:
                backlog["replay"] += state["count"] + model.arrivals(state["since"], sim_t["t"])
                backlog["real"] += real_now
                backlog["n"] += 1
            sim_t["t"] += TICK_SEC
        return rule

    rule = asyncio.run(run())
    real = [r for r in data["runs"] if start <= r["t"] < end]
    real_prunes = [r for r in real if r["decision"] == "pruned"]
    sim_prunes = [a for a in asks if a["decision"] == "pruned"]
    days = (end - start) / 86400.0
    gaps = [b["t"] - a["t"] for a, b in zip(asks, asks[1:])]
    # For each real prune: was there a replay prune within one cooldown before or after it?
    matched = sum(1 for r in real_prunes if any(abs(a["t"] - r["t"]) <= cooldown_sec for a in sim_prunes))
    return {
        "cooldown_sec": cooldown_sec,
        "days": round(days, 2),
        "ticks": sum(rule.verdicts.values()),
        "verdicts": dict(rule.verdicts),
        "asks": len(asks),
        "asks_per_day": round(len(asks) / days, 2),
        "replay_prunes": len(sim_prunes),
        "replay_declines": len(asks) - len(sim_prunes),
        "replay_images_removed": sum(a["removed"] for a in sim_prunes),
        "real_runs": len(real),
        "real_prunes": len(real_prunes),
        "real_declines": sum(1 for r in real if r["decision"] == "declined_nothing_to_reclaim"),
        "real_images_removed": sum((r["before"] - r["after"]) if r["after"] is not None else 0
                                   for r in real_prunes),
        "real_prunes_matched_within_one_cooldown": matched,
        "min_gap_between_asks_sec": min(gaps) if gaps else None,
        "disk_min_pct": min(d[1] for d in disk if start <= d[0] < end),
        "disk_max_pct": max(d[1] for d in disk if start <= d[0] < end),
        "max_dangling_at_an_ask": max((a["dangling"] for a in asks), default=0),
        # Time-weighted mean dangling images left on the host, every tick (lower = pruned sooner).
        "mean_backlog_replay": round(backlog["replay"] / backlog["n"], 1) if backlog["n"] else None,
        "mean_backlog_real": round(backlog["real"] / backlog["n"], 1) if backlog["n"] else None,
        "asks_detail": asks,
    }


def _dip(disk, t0, t1, pct):
    return [[t, pct if t0 <= t < t1 else v] for t, v in disk]


def gates(data: dict, r: dict | None = None) -> list[tuple[str, bool, str]]:
    r = r or replay(data)
    start = _ts(data["window"][0]).timestamp()
    out = [
        ("asks only on a readable disk at/above 70 %",
         all(a["disk_pct"] is not None and a["disk_pct"] >= 70.0 for a in r["asks_detail"]), ""),
        ("never two asks inside the cooldown",
         r["min_gap_between_asks_sec"] is None or r["min_gap_between_asks_sec"] >= r["cooldown_sec"],
         f"min gap {r['min_gap_between_asks_sec']} s"),
        ("at most one ask per cooldown (2 a day at 12 h)",
         r["asks_per_day"] <= 86400.0 / r["cooldown_sec"] + 1e-9, f"{r['asks_per_day']}/day"),
        ("the replay removes at least 90 % of the images the real record removed (keeps up)",
         r["replay_images_removed"] >= 0.9 * r["real_images_removed"],
         f"replay {r['replay_images_removed']} vs real {r['real_images_removed']}"),
    ]
    out.append(("the host carries no more dangling images on average than under dispatch",
                r["mean_backlog_replay"] is not None and r["mean_backlog_replay"] <= r["mean_backlog_real"],
                f"mean backlog replay {r['mean_backlog_replay']} vs real {r['mean_backlog_real']}"))
    # Fault injection: the disk below the floor for a whole day -> no ask that day.
    day0, day1 = start + 3 * 86400, start + 4 * 86400
    low = replay(data, mutate=lambda d: _dip(d, day0, day1, 69.0))
    in_dip = [a for a in low["asks_detail"] if day0 <= a["t"] < day1]
    out.append(("disk at 69 % for a day -> 0 asks that day", not in_dip, f"{len(in_dip)} asks"))
    # Fault injection: the disk reading goes silent for 2 h -> no ask while it is silent.
    s0, s1 = start + 5 * 86400, start + 5 * 86400 + 7200
    gap = replay(data, mutate=lambda d: [x for x in d if not (s0 <= x[0] < s1)])
    in_gap = [a for a in gap["asks_detail"] if s0 + STALE_SEC < a["t"] < s1]
    out.append(("replay check: a 2 h gap in the disk record -> 0 asks in it (live reads the disk itself)",
                not in_gap and
                gap["verdicts"].get("disk_unreadable", 0) > 0,
                f"{len(in_gap)} asks, {gap['verdicts'].get('disk_unreadable', 0)} unreadable ticks"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true", help="refresh the fixture from live Postgres")
    ap.add_argument("--sweep", action="store_true", help="replay at several cooldowns")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--dispatch-db", default=LIVE_DB,
                    help="--export: database holding substrate_dispatch_results (restored from the pre-drop backup)")
    args = ap.parse_args()
    if args.dispatch_db != LIVE_DB and not args.export:
        ap.error("--dispatch-db needs --export")
    if args.export:
        try:
            data = export(dispatch_db=args.dispatch_db)
        except DispatchHistoryMissing as exc:
            print(f"cannot export: {exc}", file=sys.stderr)
            return 2
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        tmp = FIXTURE.with_name(FIXTURE.name + ".tmp")   # a failed write never leaves a broken fixture
        with gzip.open(tmp, "wt") as f:
            json.dump(data, f, separators=(",", ":"))
        os.replace(tmp, FIXTURE)
        print(f"wrote {FIXTURE}: {len(data['disk'])} disk buckets, {len(data['runs'])} real prune-skill runs")
        return 0
    data = load()
    if args.sweep:
        for c in (3600, 3 * 3600, 6 * 3600, 12 * 3600, 24 * 3600, 48 * 3600):
            r = replay(data, cooldown_sec=float(c))
            print(f"cooldown {c / 3600:>4.0f} h: asks {r['asks']:>4} ({r['asks_per_day']}/day)  "
                  f"prunes {r['replay_prunes']:>3}  declines {r['replay_declines']:>4}  "
                  f"images {r['replay_images_removed']:>5}  mean backlog {r['mean_backlog_replay']:>6} "
                  f"(real {r['mean_backlog_real']})  real prunes matched "
                  f"{r['real_prunes_matched_within_one_cooldown']}/{r['real_prunes']}")
        return 0
    r = replay(data)
    summary = {k: v for k, v in r.items() if k != "asks_detail"}
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"window {data['window'][0]} .. {data['window'][1]} ({r['days']} days, {r['ticks']} ticks)")
        print(f"disk {r['disk_min_pct']}..{r['disk_max_pct']} % (floor 70 %): verdicts {r['verdicts']}")
        print(f"replay: {r['asks']} asks ({r['asks_per_day']}/day) -> {r['replay_prunes']} prunes, "
              f"{r['replay_declines']} declined, {r['replay_images_removed']} images removed")
        print(f"real (dispatch): {r['real_runs']} runs -> {r['real_prunes']} prunes, {r['real_declines']} declined, "
              f"{r['real_images_removed']} images removed")
        print(f"mean dangling backlog: replay {r['mean_backlog_replay']}, real {r['mean_backlog_real']}")
        print(f"real prunes with a replay prune within one cooldown: "
              f"{r['real_prunes_matched_within_one_cooldown']}/{r['real_prunes']}")
    failed = 0
    for name, ok, detail in gates(data, r):
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}{('  (' + detail + ')') if detail else ''}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
