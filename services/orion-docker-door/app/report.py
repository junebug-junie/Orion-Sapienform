"""What broke during a Docker-door canary window: read the door's JSONL record and summarize it.

Run on the host (stdlib only):

    python3 services/orion-docker-door/app/report.py --since 2026-11-26T15:00Z --until 2026-11-26T19:00Z

"Broke" here means the door refused a call a turn made. Calls that passed are counted by kind, so
the report also shows what Orion did through the door while it was live.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

DEFAULT_DIR = Path("/mnt/storage-warm/orion-record")


def parse_ts(text: str) -> datetime:
    text = text.strip().replace("Z", "+00:00")
    ts = datetime.fromisoformat(text)
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def read_entries(folder: Path, since: datetime, until: datetime) -> Iterator[dict[str, Any]]:
    day = since.date()
    while day <= until.date():
        path = folder / f"docker-{day:%Y-%m-%d}.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    entry = json.loads(line)
                    ts = parse_ts(entry["ts"])
                except (ValueError, KeyError):
                    continue
                if since <= ts < until:
                    yield entry
        day += timedelta(days=1)


def summarize(entries: Iterable[dict[str, Any]]) -> str:
    entries = list(entries)
    allowed = [e for e in entries if e.get("decision") == "allow"]
    refused = [e for e in entries if e.get("decision") != "allow"]
    lines = [f"Calls through the door: {len(entries)} ({len(allowed)} passed, {len(refused)} refused)"]
    if not entries:
        lines.append("Nothing used the door in this window. Check that the canary window was set "
                     "(ORION_DOCKER_DOOR_CANARY_WINDOWS in the governor) and that turns ran.")
        return "\n".join(lines)
    lines.append("")
    lines.append("Passed, by kind:")
    for (reason, method), n in Counter((e.get("reason"), e.get("method")) for e in allowed).most_common():
        lines.append(f"  {n:5d}  {method} {reason}")
    execs = Counter(e.get("container") for e in allowed if e.get("reason") == "exec" and "exec_cmd" in e)
    if execs:
        lines.append("")
        lines.append("Exec targets:")
        for name, n in execs.most_common():
            lines.append(f"  {n:5d}  {name}")
    lines.append("")
    lines.append("Refused (what broke):" if refused else "Refused: none.")
    for e in refused:
        detail = e.get("container") or ""
        if e.get("exec_cmd"):
            detail += f" cmd={e['exec_cmd']}"
        lines.append(f"  {e.get('ts', '')[:19]}  {e.get('method')} {e.get('path')}  [{e.get('reason')}] {detail}".rstrip())
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", required=True, help="UTC ISO time, e.g. 2026-11-26T15:00Z")
    ap.add_argument("--until", default=None, help="UTC ISO time (default: now)")
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    args = ap.parse_args(argv)
    since = parse_ts(args.since)
    until = parse_ts(args.until) if args.until else datetime.now(timezone.utc)
    print(summarize(read_entries(Path(args.dir), since, until)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
