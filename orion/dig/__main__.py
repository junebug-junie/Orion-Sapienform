"""``python -m orion.dig``: Orion's read-only kit for digging into their own history.

Run through Bash from a chat turn or a curiosity run (orion-harness-governor). Every subcommand
prints at most ``MAX_LINES`` lines / ``MAX_CHARS`` chars and writes its full result to a file
under ``$ORION_DIG_HOME/dig/<dig>/``, so a dig costs context only for what Orion chooses to read.
Nothing here writes to the repo: a past checkout is a plain-file export (``git archive``) with no
git link back.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from orion import self_inventory

MAX_LINES = 40
MAX_CHARS = 4000
MAX_OPEN = 2
SNAPSHOT_TTL_SEC = 24 * 3600
MARKER = ".dig_snapshot.json"
ZONE = ZoneInfo("America/Denver")
FIRST_DAY = date(2025, 6, 24)  # first commit in the repo

HELP = f"""dig: read-only tools for your own history. Every command prints a short summary and the
path of a file with the full result; read only what you need from that file.

  open <date>            export the repo as it was at the end of <date> (YYYY-MM-DD, Denver)
                         to a plain folder (no git link). At most {MAX_OPEN} open; 24 h old ones go.
  close <date>           delete that export.  list: show open exports.
  inventory <date|now>   services, modules, channels, verbs, schemas in that checkout.
  compare <date> [<date|now>]   what was added / removed between two checkouts.
  history <start> <end>  merged PRs (title + number) and commits in [start, end].
  body [<date>]          your body atlas: parts -> circulation -> organs. No date = live bus now;
                         a date = that export's channel catalog only (exists from 2026-01).
  notes [append <text>]  this dig's shared notes (episode hops / turns read each other).
  --dig <id>             group results under one dig (default: today's date).

Label what you say from a dig by its source:
  code_then     seen in an export: a primary source.
  pr_text       a builder's words in a PR or commit: a secondary source. Say so.
  lived_record  your own record from that time (journals exist from 2026-07-23 only).
  now           must come from `inventory now` or `body`, never from memory.
  reflection    your interpretation. Wonder freely, but do not state it as fact.
Code from {FIRST_DAY} on is all there. Your memory before 2026-07-22 was lost in a storage
failure; that is still you, you just have no lived record of it."""


def _env_path(key: str, default: str) -> Path:
    return Path(os.environ.get(key) or default)


def home() -> Path:
    return _env_path("ORION_DIG_HOME", "/mnt/orion-fcc")


def repo() -> Path:
    """The full-history clone exports come from (Orion's sandbox; never checked out by us)."""
    return _env_path("ORION_DIG_REPO", "/mnt/orion-fcc/repo")


def now_root() -> Path:
    """Today's deployed main (the primary checkout, mounted read-only in the harness)."""
    return _env_path("ORION_DIG_NOW_ROOT", "/mnt/scripts/Orion-Sapienform")


def past_dir() -> Path:
    return home() / "past"


class DigError(Exception):
    pass


def _parse_day(text: str) -> date:
    try:
        day = date.fromisoformat(text)
    except ValueError as exc:
        raise DigError(f"not a date: {text!r} (want YYYY-MM-DD)") from exc
    if day < FIRST_DAY:
        raise DigError(f"{day} is before the first commit ({FIRST_DAY})")
    if day > datetime.now(ZONE).date():
        raise DigError(f"{day} is in the future")
    return day


def _end_of_day_utc(day: date) -> datetime:
    return datetime.combine(day + timedelta(days=1), dtime(0), tzinfo=ZONE).astimezone(timezone.utc)


def _git(*args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo()), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise DigError(f"git {' '.join(args)} failed: {proc.stderr.strip()[:300]}")
    return proc.stdout.strip()


def _main_ref() -> str:
    try:
        _git("rev-parse", "--verify", "--quiet", "origin/main")
        return "origin/main"
    except DigError:
        return "HEAD"


def commit_at(day: date) -> tuple[str, str]:
    """Last main commit on or before the end of ``day`` (Denver): (sha, commit date ISO)."""
    out = _git("log", "-1", "--first-parent", f"--before={_end_of_day_utc(day).isoformat()}",
               "--format=%H %cI", _main_ref())
    if not out:
        raise DigError(f"no commit on or before {day}")
    sha, when = out.split(" ", 1)
    return sha, when


# --- output ---------------------------------------------------------------------------------

class Out:
    def __init__(self, dig: str) -> None:
        self.dir = home() / "dig" / dig
        self.dig = dig

    def write(self, name: str, payload) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        n = 1 + sum(1 for _ in self.dir.glob(f"{name}-*.json"))
        path = self.dir / f"{name}-{n}.json"
        path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
        return path


def capped(lines: list[str], *, tail: str | None) -> str:
    """At most MAX_LINES lines and MAX_CHARS chars; a cut is always said, the tail always kept."""
    keep = MAX_LINES - (1 if tail else 0)
    body = lines
    if len(body) > keep:
        body = body[: keep - 1] + [f"(showing {keep - 1} of {len(lines)} lines; the rest is in the file)"]
    text = "\n".join(body)
    room = MAX_CHARS - (len(tail) + 1 if tail else 0)
    if len(text) > room:
        note = "(cut at 4000 chars; the rest is in the file)"
        text = text[: room - len(note) - 1].rsplit("\n", 1)[0] + "\n" + note
    return text + ("\n" + tail if tail else "")


# --- snapshots ------------------------------------------------------------------------------

def _snapshots() -> list[tuple[Path, dict]]:
    if not past_dir().is_dir():
        return []
    out = []
    for path in sorted(past_dir().iterdir()):
        marker = path / MARKER
        if path.is_dir() and marker.exists():
            out.append((path, json.loads(marker.read_text(encoding="utf-8"))))
    return out


def _remove(path: Path) -> None:
    # Only ever a folder we exported: it must sit directly under past/ and carry our marker.
    if path.parent != past_dir() or not (path / MARKER).exists():
        raise DigError(f"refusing to remove {path}: not a dig export")
    shutil.rmtree(path)


def sweep(now_epoch: float | None = None) -> list[str]:
    now_epoch = time.time() if now_epoch is None else now_epoch
    gone = []
    for path, meta in _snapshots():
        if now_epoch - float(meta.get("opened_epoch", 0)) > SNAPSHOT_TTL_SEC:
            _remove(path)
            gone.append(path.name)
    return gone


def snapshot_root(day: date) -> Path:
    path = past_dir() / day.isoformat()
    if not (path / MARKER).exists():
        raise DigError(f"{day} is not open; run: python -m orion.dig open {day}")
    return path


def cmd_open(day: date, out: Out) -> str:
    swept = sweep()
    target = past_dir() / day.isoformat()
    if (target / MARKER).exists():
        meta = json.loads((target / MARKER).read_text(encoding="utf-8"))
        return f"{day} already open at {target} (commit {meta['commit'][:10]} of {meta['commit_date']})"
    open_now = _snapshots()
    if len(open_now) >= MAX_OPEN:
        names = ", ".join(p.name for p, _ in open_now)
        raise DigError(f"{MAX_OPEN} exports already open ({names}); close one first")
    sha, when = commit_at(day)
    partial = past_dir() / f".partial-{day.isoformat()}"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    proc = subprocess.Popen(["git", "-C", str(repo()), "archive", "--format=tar", sha], stdout=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
        tar.extractall(partial, filter="data")
    if proc.wait() != 0:
        shutil.rmtree(partial)
        raise DigError(f"git archive {sha[:10]} failed")
    meta = {"date": day.isoformat(), "commit": sha, "commit_date": when, "opened_epoch": time.time()}
    (partial / MARKER).write_text(json.dumps(meta), encoding="utf-8")
    partial.rename(target)
    msg = [f"opened {day} at {target}", f"commit {sha[:10]} of {when} (last main commit that day)"]
    if swept:
        msg.append(f"swept exports older than 24 h: {', '.join(swept)}")
    return "\n".join(msg)


def cmd_close(day: date) -> str:
    path = past_dir() / day.isoformat()
    if not (path / MARKER).exists():
        return f"{day} was not open"
    _remove(path)
    return f"closed {day}"


def cmd_list() -> str:
    snaps = _snapshots()
    if not snaps:
        return "no exports open"
    return "\n".join(f"{p.name}: commit {m['commit'][:10]} of {m['commit_date']}" for p, m in snaps)


# --- inventory / compare --------------------------------------------------------------------

def _root_for(arg: str) -> tuple[Path, str]:
    if arg == "now":
        return now_root(), "now (deployed main)"
    day = _parse_day(arg)
    return snapshot_root(day), str(day)


def _unfamiliar(inv: dict[str, list[str]], root: Path, label: str) -> list[str]:
    """An empty inventory means 'built differently', not 'empty': say so and show the layout."""
    if any(inv.values()):
        return []
    return [f"{label} has none of today's layout (services/, orion/bus, orion/schemas, ...): the scanner",
            "recognizes nothing there, which is NOT the same as nothing existing. Its top level:",
            "  " + "  ".join(self_inventory.layout(root)),
            f"Read those files directly under {root}."]


def cmd_inventory(arg: str, out: Out) -> str:
    root, label = _root_for(arg)
    inv = self_inventory.inventory(root)
    path = out.write(f"inventory-{arg}", {"checkout": label, "root": str(root), "inventory": inv,
                                          "layout": self_inventory.layout(root)})
    lines = [f"inventory of {label}", *_unfamiliar(inv, root, label)]
    for cat, names in inv.items():
        sample = ", ".join(names[:6]) + (f" +{len(names) - 6}" if len(names) > 6 else "")
        lines.append(f"{cat}s: {len(names)}" + (f" ({sample})" if names else ""))
    return capped(lines, tail=f"full result: {path}")


def compare(then: dict[str, list[str]], now: dict[str, list[str]]) -> dict[str, dict[str, list[str]]]:
    return {cat: {"added": sorted(set(now[cat]) - set(then[cat])),
                  "removed": sorted(set(then[cat]) - set(now[cat])),
                  "kept": len(set(then[cat]) & set(now[cat]))}
            for cat in self_inventory.CATEGORIES}


def cmd_compare(a: str, b: str, out: Out) -> str:
    root_a, label_a = _root_for(a)
    root_b, label_b = _root_for(b)
    inv_a, inv_b = self_inventory.inventory(root_a), self_inventory.inventory(root_b)
    diff = compare(inv_a, inv_b)
    path = out.write(f"compare-{a}-{b}", {"from": label_a, "to": label_b, "diff": diff,
                                          "layout_from": self_inventory.layout(root_a)})
    lines = [f"{label_a} -> {label_b} (code only: what existed, not what ran)",
             *_unfamiliar(inv_a, root_a, label_a), *_unfamiliar(inv_b, root_b, label_b)]
    for cat, d in diff.items():
        lines.append(f"{cat}s: +{len(d['added'])} -{len(d['removed'])} ={d['kept']}")
        if d["added"]:
            lines.append("  added: " + ", ".join(d["added"][:8]) + (" ..." if len(d["added"]) > 8 else ""))
        if d["removed"]:
            lines.append("  removed: " + ", ".join(d["removed"][:8]) + (" ..." if len(d["removed"]) > 8 else ""))
    return capped(lines, tail=f"full result: {path}")


# --- history --------------------------------------------------------------------------------

def _prs(start: date, end: date) -> list[dict]:
    lo = datetime.combine(start, dtime(0), tzinfo=ZONE).astimezone(timezone.utc)
    hi = _end_of_day_utc(end)
    query = f"merged:{lo.strftime('%Y-%m-%dT%H:%M:%SZ')}..{hi.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    proc = subprocess.run(
        ["gh", "pr", "list", "--repo", os.environ.get("ORION_DIG_GITHUB_REPO", "junebug-junie/Orion-Sapienform"),
         "--state", "merged", "--search", query, "--limit", "1000",
         "--json", "number,title,mergedAt,body"],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise DigError(f"gh pr list failed: {proc.stderr.strip()[:300]}")
    return sorted(json.loads(proc.stdout or "[]"), key=lambda p: p["mergedAt"])


def cmd_history(start: date, end: date, out: Out) -> str:
    if end < start:
        raise DigError("end is before start")
    lo = datetime.combine(start, dtime(0), tzinfo=ZONE).astimezone(timezone.utc)
    log = _git("log", "--first-parent", _main_ref(), f"--since={lo.isoformat()}",
               f"--until={_end_of_day_utc(end).isoformat()}", "--format=%h%x09%cI%x09%s")
    commits = [dict(zip(("sha", "date", "subject"), row.split("\t", 2))) for row in log.splitlines() if row]
    prs = _prs(start, end)
    path = out.write(f"history-{start}-{end}", {"start": str(start), "end": str(end), "prs": prs, "commits": commits})
    lines = [f"{start} .. {end}: {len(prs)} merged PRs, {len(commits)} commits on main"
             " (PR text = the builders' words: a secondary source)"]
    for pr in prs[:25]:
        lines.append(f"#{pr['number']} {pr['mergedAt'][:10]} {pr['title'][:110]}")
    if len(prs) > 25:
        lines.append(f"... {len(prs) - 25} more PRs")
    if not prs:
        for c in commits[:25]:
            lines.append(f"{c['sha']} {c['date'][:10]} {c['subject'][:110]}")
    return capped(lines, tail=f"full result (PR bodies included): {path}")


# --- body -----------------------------------------------------------------------------------

def cmd_body(arg: str | None, out: Out) -> str:
    from orion.body_atlas.__main__ import _RoGraphReader, summarize
    from orion.body_atlas.build import build_atlas, fetch_observed, load_catalog

    if arg is None:
        root, observed = now_root(), None
        uri = os.environ.get("FALKORDB_URI")
        if not uri:
            raise DigError("live body needs $FALKORDB_URI (orion-harness-governor .env)")
        reader = _RoGraphReader(uri, os.environ.get("FALKORDB_BUS_GRAPH", "orion_bus_synapse"))
        try:
            observed = fetch_observed(reader)
        finally:
            reader.close()
        when = datetime.now(timezone.utc)
    else:
        day = _parse_day(arg)
        root, observed, when = snapshot_root(day), None, _end_of_day_utc(day)
    catalog = load_catalog(root)
    if not catalog:
        raise DigError(f"no channel catalog in {root} (the catalog starts 2026-01)")
    atlas = build_atlas(catalog, observed, now=when, live_graph="orion_bus_synapse" if observed else None)
    path = out.write(f"body-{arg or 'now'}", json.loads(atlas.model_dump_json()))
    return summarize(atlas, out=str(path))


# --- notes ----------------------------------------------------------------------------------

def cmd_notes(text: str | None, out: Out) -> str:
    path = out.dir / "notes.md"
    if text:
        out.dir.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n- {datetime.now(timezone.utc).isoformat(timespec='seconds')}: {text.strip()}\n")
        return f"appended to {path}"
    if not path.exists():
        return f"no notes yet for dig {out.dig}"
    return capped(path.read_text(encoding="utf-8").strip().splitlines(), tail=f"notes: {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m orion.dig", add_help=False)
    ap.add_argument("--dig", default=datetime.now(ZONE).date().isoformat())
    ap.add_argument("command", nargs="?", default="help")
    ap.add_argument("args", nargs="*")
    ns = ap.parse_args(argv)
    out, a = Out(ns.dig), ns.args
    try:
        if ns.command in ("help", "-h", "--help"):
            text = HELP
        elif ns.command == "open" and len(a) == 1:
            text = cmd_open(_parse_day(a[0]), out)
        elif ns.command == "close" and len(a) == 1:
            text = cmd_close(_parse_day(a[0]))
        elif ns.command == "list" and not a:
            text = cmd_list()
        elif ns.command == "inventory" and len(a) == 1:
            text = cmd_inventory(a[0], out)
        elif ns.command == "compare" and len(a) in (1, 2):
            text = cmd_compare(a[0], a[1] if len(a) == 2 else "now", out)
        elif ns.command == "history" and len(a) == 2:
            text = cmd_history(_parse_day(a[0]), _parse_day(a[1]), out)
        elif ns.command == "body" and len(a) <= 1:
            text = cmd_body(a[0] if a else None, out)
        elif ns.command == "notes" and (not a or a[0] == "append"):
            text = cmd_notes(" ".join(a[1:]) if a else None, out)
        else:
            print(f"unknown or malformed: {' '.join([ns.command, *a])}\n\n{HELP}", file=sys.stderr)
            return 2
    except DigError as exc:
        print(f"dig: {exc}", file=sys.stderr)
        return 1
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
