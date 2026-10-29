#!/usr/bin/env python3
"""Find post-pull rebuild runs that died without saying so.

scripts/rebuild_services_from_git_diff.sh writes one line per step to
~/.orion/post_pull_rebuild.log: a RUN line naming every service it will
rebuild, START/END around each compose call, and DONE at the end. Its trap
writes ABORT when the run is interrupted. SIGKILL cannot be trapped: a killed
run just stops, with a START and no END (2026-10-29: pid 3250904 died inside
`orion-harness-governor up -d --build`; the harness worker stayed stopped for
~9.5 h and hub, thought and cortex-orch were never rebuilt, with nothing in the
log to say so).

This reads the log and reports every run that has a RUN line, no DONE/ABORT,
and no live process -- naming the step in flight, the services never started,
and the command that finishes them.

  --mark           append a DIED line for each one, so it is reported once
                   (the rebuild script runs this at the start of every run)
  --session-start  print a Claude Code SessionStart JSON payload, or nothing
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

LINE = re.compile(r"^(?P<ts>\S+) pid=(?P<pid>\d+) head=(?P<head>\S+) (?P<rest>.*)$")
RUN = re.compile(r"^RUN base=\S* services=(?P<services>.*?)recreate=(?P<recreate>.*)$")
CLOSED = re.compile(r"^DIED run=(?P<run>\S+)")


@dataclass
class Run:
    key: str  # "<RUN ts>/<pid>": a pid alone is reused across days
    pid: int
    started_at: str
    head: str
    services: list[str]
    recreate: list[str]
    started: list[str] = field(default_factory=list)  # services with a START line, in order
    in_flight: str | None = None  # the step whose START has no END yet
    closed: bool = False

    def unfinished(self) -> list[str]:
        """The service in flight first, then every service never started."""
        current = self.in_flight.split()[0] if self.in_flight else None
        out = [current] if current else []
        return out + [s for s in self.services if s not in self.started and s not in out]

    def finish_command(self) -> str:
        todo = self.unfinished()
        # Same quoting as the rebuild script's own ABORT-FINISH line; service names are
        # [a-z0-9_-] (the script refuses anything else), so single quotes are safe.
        cmd = f"scripts/rebuild_services_from_git_diff.sh --services '{' '.join(todo)}'"
        recreate = [s for s in self.recreate if s in todo]
        if recreate:
            cmd += f" --recreate '{' '.join(recreate)}'"
        return cmd


def parse(lines) -> list[Run]:
    runs: dict[str, Run] = {}
    open_by_pid: dict[int, Run] = {}
    closed_keys: set[str] = set()
    for raw in lines:
        m = LINE.match(raw.rstrip("\n"))
        if not m:
            continue
        pid, rest = int(m["pid"]), m["rest"]
        run_m = RUN.match(rest)
        if run_m:
            run = Run(
                key=f"{m['ts']}/{pid}",
                pid=pid,
                started_at=m["ts"],
                head=m["head"],
                services=run_m["services"].split(),
                recreate=run_m["recreate"].split(),
            )
            runs[run.key] = run
            open_by_pid[pid] = run
            continue
        closed = CLOSED.match(rest)
        if closed:
            closed_keys.add(closed["run"])
            continue
        run = open_by_pid.get(pid)
        if run is None:
            continue
        if rest.startswith("START "):
            run.in_flight = rest[len("START "):]
            svc = run.in_flight.split()[0] if run.in_flight.split() else ""
            if svc and svc not in run.started:
                run.started.append(svc)
        elif rest.startswith("END "):
            run.in_flight = None
        elif rest.startswith("DONE") or rest.startswith("ABORT"):
            run.closed = True
            del open_by_pid[pid]
    for key in closed_keys:
        if key in runs:
            runs[key].closed = True
    return list(runs.values())


def _alive(pid: int) -> bool:
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    return b"rebuild_services_from_git_diff" in cmdline


def _age_hours(ts: str, now: float) -> float:
    try:
        then = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return float("inf")
    return (now - then) / 3600.0


def died_runs(lines, *, now: float, max_age_hours: float, alive=_alive) -> list[Run]:
    return [
        r
        for r in parse(lines)
        if not r.closed and _age_hours(r.started_at, now) <= max_age_hours and not alive(r.pid)
    ]


def report(run: Run) -> str:
    step = f"`{run.in_flight}`" if run.in_flight else "between steps"
    return (
        f"Post-pull rebuild pid={run.pid} (started {run.started_at}, head {run.head}) died with no END line, "
        f"during {step}. Not rebuilt: {' '.join(run.unfinished()) or 'nothing'}. "
        f"A container of the step in flight may be stopped or left 'Created' (check `docker ps -a`). "
        f"Finish it from the primary main checkout: {run.finish_command()}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_log = os.environ.get("ORION_REBUILD_LOG") or str(Path.home() / ".orion" / "post_pull_rebuild.log")
    parser.add_argument("--log", default=default_log)
    parser.add_argument("--max-age-hours", type=float, default=72.0)
    parser.add_argument("--mark", action="store_true")
    parser.add_argument("--session-start", action="store_true")
    args = parser.parse_args(argv)
    try:
        with open(args.log, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return 0
    now = time.time()
    died = died_runs(lines, now=now, max_age_hours=args.max_age_hours)
    if not died:
        return 0
    texts = [report(r) for r in died]
    if args.session_start:
        payload = {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "\n".join(texts)}}
        sys.stdout.write(json.dumps(payload))
        return 0
    for text in texts:
        print(f"check_rebuild_log: {text}", file=sys.stderr)
    if args.mark:
        stamp = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            with open(args.log, "a", encoding="utf-8") as fh:
                for r in died:
                    fh.write(
                        f"{stamp} pid={os.getpid()} head=- DIED run={r.key} "
                        f"in_flight={(r.in_flight or 'none').replace(' ', '_')} unfinished={' '.join(r.unfinished())}\n"
                    )
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
