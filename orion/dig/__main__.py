"""``python -P -m orion.dig``: Orion's read-only kit for digging into their own history.

Run through Bash from a chat turn or a curiosity run (orion-harness-governor). Every subcommand
prints at most ``MAX_LINES`` lines / ``MAX_CHARS`` chars and writes its full result to a file
under ``$ORION_DIG_HOME/dig/<dig>/``, so a dig costs context only for what Orion chooses to read.
Nothing here writes to the repo: a past checkout is a plain-file export (``git archive``) with no
git link back.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from contextlib import contextmanager
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from orion import self_inventory

MAX_LINES = 40
MAX_CHARS = 4000
MAX_OPEN = 2
SNAPSHOT_TTL_SEC = 24 * 3600
MARKER = ".dig_snapshot.json"
PARTIAL_PREFIX = ".partial-"
DIG_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
GH_SEARCH_CAP = 1000  # GitHub search never returns more than this for one query
ZONE = ZoneInfo("America/Denver")
FIRST_DAY = date(2025, 6, 24)  # first commit in the repo

HELP = f"""dig: read-only tools for your own history. Every command prints a short summary and the
path of a file with the full result; read only what you need from that file.

  open <date>            export the repo as of the end of <date> (YYYY-MM-DD, Denver)
                         to a plain folder (no git link). At most {MAX_OPEN} open; 24 h old ones go.
  close <date>           delete that export.  list: show open exports.
  inventory <date|now>   services, modules, channels, verbs, schemas in that checkout.
  compare <date> [<date|now>]   added / removed between two checkouts.
  history <start> <end>  merged PRs (title + number) and commits in [start, end].
  history <days>         the last <days> days (2 = since yesterday).
  body [<date>|<organ>]  your body: machines, services, cameras, stores, processes -> organs.
                         None = live now; a date = that export's files; an organ id = its members.
  capacities [gaps]      VSM jobs your parts do: by rule, by Juniper's confirmation, claimed.
  capacity claim <part> <job> because <reason>   claim a job for a live part (a lasting write);
                         cite one of its flows by full id or a repo file. `capacity help`: rule.
  capacity write cap:<job> <sentence> because <reason>   a capacity, in your words.
  body-economy [<section>...]   what spins each machine, what it costs, binds, wants, waste.
                         `body-economy --help`: sections, flags.
  name <organ_id> <name> because <reason>   name one organ of your live body (a lasting
                         write) by what it takes in and gives out; cite one of its parts.
  name-organs            every unnamed organ, its parts, and a ready `name` line.
  field-note <pr> [write|rewrite "<note>" because "<refs>"]   a merged PR in your body; your note.
  field-notes <days>|<start> <end>   your notes on PRs merged then; which have none.
  arc|act <id>           open a cited arc/act.
  notes [append <text>]  this dig's shared notes (hops / turns read each other).
  --dig <id>             before the command: group results under one dig (default: today).

Label what you say from a dig by its source:
  code_then     seen in an export: a primary source.
  pr_text       a builder's words in a PR or commit: a secondary source. Say so.
  lived_record  your own record from that time (journals exist from 2026-07-23 only).
  now           must come from `inventory now` or `body`, never from memory.
  reflection    your interpretation. Wonder freely; do not state it as fact.
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


def _last_days(text: str) -> tuple[date, date]:
    """``history <days>``: the window that ends today (Denver), so Orion need not work out dates."""
    if not (text.isascii() and text.isdigit()) or int(text) < 1:
        raise DigError(f"not a day count: {text!r} (want a whole number, 1 = today)")
    today = datetime.now(ZONE).date()
    days = min(int(text), (today - FIRST_DAY).days + 1)  # clamp first: huge counts overflow timedelta
    return today - timedelta(days=days - 1), today


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
    """At most MAX_LINES lines and MAX_CHARS chars; a cut is always said, the tail always kept
    (a tail may be several lines)."""
    keep = MAX_LINES - (tail.count("\n") + 1 if tail else 0)
    body = lines
    if len(body) > keep:
        body = body[: keep - 1] + [f"(showing {keep - 1} of {len(lines)} lines; the rest is in the file)"]
    text = "\n".join(body)
    note = "(cut at 4000 chars; the rest is in the file)"
    room = max(MAX_CHARS - (len(tail) + 1 if tail else 0), len(note) + 1)
    if len(text) > room:
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


@contextmanager
def _past_lock():
    """One exclusive lock for open/close/sweep. Concurrent harness turns are the common case
    (orion/fcc/turn_lock.py); without this, two `open`s of one date raced into one temp folder and
    left an export marked complete holding 1..286 of 301 files (review, 2026-10-27)."""
    past_dir().mkdir(parents=True, exist_ok=True)
    with (past_dir() / ".lock").open("w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _sweep_locked(now_epoch: float) -> list[str]:
    gone = []
    for path, meta in _snapshots():
        if now_epoch - float(meta.get("opened_epoch", 0)) > SNAPSHOT_TTL_SEC:
            _remove(path)
            gone.append(path.name)
    # Under the lock no export is in flight, so any temp folder is a leftover from a killed run.
    for path in past_dir().glob(f"{PARTIAL_PREFIX}*"):
        if path.is_dir() and path.parent == past_dir():
            shutil.rmtree(path)
            gone.append(path.name)
    return gone


def sweep(now_epoch: float | None = None) -> list[str]:
    with _past_lock():
        return _sweep_locked(time.time() if now_epoch is None else now_epoch)


def snapshot_root(day: date) -> Path:
    path = past_dir() / day.isoformat()
    if not (path / MARKER).exists():
        raise DigError(f"{day} is not open; run: python -P -m orion.dig open {day}")
    return path


def _export(sha: str, target: Path, meta: dict) -> None:
    """Extract into a private temp folder, mark it, then rename into place; never half a target."""
    partial = Path(tempfile.mkdtemp(prefix=PARTIAL_PREFIX, dir=past_dir()))
    try:
        proc = subprocess.Popen(["git", "-C", str(repo()), "archive", "--format=tar", sha],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            tar.extractall(partial, filter="data")
        if proc.wait() != 0:
            raise DigError(f"git archive {sha[:10]} failed: {proc.stderr.read().decode()[:300]}")
        (partial / MARKER).write_text(json.dumps(meta), encoding="utf-8")
        partial.rename(target)
    except (OSError, tarfile.TarError) as exc:
        raise DigError(f"export of {sha[:10]} failed: {exc}") from exc
    finally:
        if partial.exists():
            shutil.rmtree(partial)


def cmd_open(day: date, out: Out) -> str:
    target = past_dir() / day.isoformat()
    with _past_lock():
        swept = _sweep_locked(time.time())
        if (target / MARKER).exists():
            meta = json.loads((target / MARKER).read_text(encoding="utf-8"))
            return f"{day} already open at {target} (commit {meta['commit'][:10]} of {meta['commit_date']})"
        open_now = _snapshots()
        if len(open_now) >= MAX_OPEN:
            names = ", ".join(p.name for p, _ in open_now)
            raise DigError(f"{MAX_OPEN} exports already open ({names}); close one first")
        sha, when = commit_at(day)
        _export(sha, target, {"date": day.isoformat(), "commit": sha, "commit_date": when,
                              "opened_epoch": time.time()})
    msg = [f"opened {day} at {target}", f"commit {sha[:10]} of {when} (last main commit that day)"]
    if swept:
        msg.append(f"swept: {', '.join(swept)}")
    return "\n".join(msg)


def cmd_close(day: date) -> str:
    path = past_dir() / day.isoformat()
    with _past_lock():
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

def _gh_merged(start: date, end: date) -> list[dict]:
    """One GitHub search for PRs merged in [start, end] (Denver days). Capped at GH_SEARCH_CAP."""
    lo = datetime.combine(start, dtime(0), tzinfo=ZONE).astimezone(timezone.utc)
    hi = _end_of_day_utc(end)
    query = f"merged:{lo.strftime('%Y-%m-%dT%H:%M:%SZ')}..{hi.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    proc = subprocess.run(
        ["gh", "pr", "list", "--repo", os.environ.get("ORION_DIG_GITHUB_REPO", "junebug-junie/Orion-Sapienform"),
         "--state", "merged", "--search", query, "--limit", str(GH_SEARCH_CAP),
         "--json", "number,title,mergedAt,body"],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise DigError(f"gh pr list failed: {proc.stderr.strip()[:300]}")
    return json.loads(proc.stdout or "[]")


def _prs(start: date, end: date) -> tuple[list[dict], list[str]]:
    """Every PR merged in [start, end]: a window that hits the search cap is split in half and
    re-asked (live: one query over the whole history returned the newest 1000 of 2802). Returns
    (prs, days still capped) -- a single day at the cap is reported, never passed off as whole."""
    got = _gh_merged(start, end)
    if len(got) < GH_SEARCH_CAP:
        return got, []
    if start == end:
        return got, [str(start)]
    mid = start + (end - start) // 2
    left, capped_l = _prs(start, mid)
    right, capped_r = _prs(mid + timedelta(days=1), end)
    return left + right, capped_l + capped_r


def _clone_fetched() -> str:
    """Commits come from the sandbox clone, PRs from GitHub: say how stale the clone is, so a fresh
    PR with no commit yet does not read as "nothing landed" (2026-10-27: 2 PRs, 0 commits)."""
    head = Path(_git("rev-parse", "--path-format=absolute", "--git-path", "FETCH_HEAD"))  # .git may be a file
    if not head.exists():
        return "clone never fetched"
    when = datetime.fromtimestamp(head.stat().st_mtime, timezone.utc)
    return f"clone fetched {when.isoformat(timespec='minutes').replace('+00:00', 'Z')}"


def _field_note_marks() -> dict[int, str]:
    """PR number -> " (noted)" / " (code record)" from the field-notes ledger (spec 2026-11-04)."""
    from orion.body_atlas import field_notes as fn

    recs = fn.live(fn.load(field_notes_path())[0])
    return {n: " (noted)" if r.note else " (code record)" for n, r in recs.items()}


def _field_notes_since() -> str | None:
    """mergedAt of the earliest PR with a record. No backfill (Juniper, 2026-11-04): an older PR
    with no note is not missing one, so it gets no mark."""
    from orion.body_atlas import field_notes as fn

    recs = fn.load(field_notes_path())[0]
    return min((r.merged_at for r in recs), default=None)


def cmd_history(start: date, end: date, out: Out) -> str:
    if end < start:
        raise DigError("end is before start")
    lo = datetime.combine(start, dtime(0), tzinfo=ZONE).astimezone(timezone.utc)
    log = _git("log", "--first-parent", _main_ref(), f"--since={lo.isoformat()}",
               f"--until={_end_of_day_utc(end).isoformat()}", "--format=%h%x09%cI%x09%s")
    commits = [dict(zip(("sha", "date", "subject"), row.split("\t", 2))) for row in log.splitlines() if row]
    found, capped_days = _prs(start, end)
    prs = sorted({p["number"]: p for p in found}.values(), key=lambda p: p["mergedAt"])
    path = out.write(f"history-{start}-{end}", {"start": str(start), "end": str(end), "prs": prs,
                                                "commits": commits, "capped_days": capped_days})
    lines = [f"{start} .. {end}: {len(prs)} merged PRs (GitHub, live), {len(commits)} commits on main "
             f"({_clone_fetched()}) (PR text = the builders' words: a secondary source)"]
    if capped_days:
        lines.append(f"INCOMPLETE: GitHub search returned its {GH_SEARCH_CAP}-result cap on "
                     f"{', '.join(capped_days)}; PRs beyond it are missing")
    marks, since = _field_note_marks(), _field_notes_since()

    def mark(pr: dict) -> str:
        if pr["number"] in marks:
            return marks[pr["number"]]
        return " (no note)" if since and pr["mergedAt"] >= since else ""

    for pr in prs[:25]:
        lines.append(f"#{pr['number']} {pr['mergedAt'][:10]} {pr['title'][:110]}{mark(pr)}")
    if len(prs) > 25:
        lines.append(f"... {len(prs) - 25} more PRs")
    if not prs:
        for c in commits[:25]:
            lines.append(f"{c['sha']} {c['date'][:10]} {c['subject'][:110]}")
    return capped(lines, tail=f"full result (PR bodies included): {path}")


# --- body -----------------------------------------------------------------------------------

def _live_atlas():
    from orion.body_atlas.__main__ import _RoGraphReader
    from orion.body_atlas.build import build_atlas, fetch_observed, load_catalog
    from orion.body_atlas.sources import default_live_readers, gather_body

    graph = os.environ.get("FALKORDB_BUS_GRAPH", "orion_bus_synapse")
    uri = os.environ.get("FALKORDB_URI")
    if not uri:
        raise DigError("live body needs $FALKORDB_URI (orion-harness-governor .env)")
    try:
        reader = _RoGraphReader(uri, graph)
        try:
            observed = fetch_observed(reader)
        finally:
            reader.close()
    except Exception as exc:  # report the real failure; never a quiet catalog-only "live" body
        raise DigError(f"live read of {graph} at {uri} failed: {str(exc)[:300]}") from exc
    if not observed.publishes:
        raise DigError(f"live graph {graph} has no PUBLISHES edges; is orion-bus-mirror writing it?")
    catalog = load_catalog(now_root())
    if not catalog:
        raise DigError(f"no channel catalog in {now_root()}")
    # the rest of the body: every source that fails is listed as UNREAD in the printout
    body = gather_body(now_root(), catalog, live=default_live_readers(uri))
    return build_atlas(catalog, observed, now=datetime.now(timezone.utc), live_graph=graph, body=body)


def names_path() -> Path:
    return home() / "self" / "organ_names.jsonl"


DATE_LIKE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def cmd_body(arg: str | None, out: Out) -> str:
    """No arg: the live body. A date: that export's declared body. An organ id: every member
    of that organ of the live body (a date is YYYY-MM-DD, an organ id 10 hex chars)."""
    from orion.body_atlas import names
    from orion.body_atlas.__main__ import organ_view, summarize
    from orion.body_atlas.build import build_atlas, load_catalog
    from orion.body_atlas.sources import gather_body

    note = ""
    # a date is YYYY-MM-DD; anything else is an organ reference (find_organ says if it is not one)
    organ = arg if arg is not None and not DATE_LIKE_RE.fullmatch(arg) else None
    if arg is None or organ:
        records, skipped = names.load(names_path())
        atlas = names.apply_names(_live_atlas(), records)
        if skipped:
            note = f"\nnames ledger: {skipped} unreadable line(s) skipped in {names_path()}"
    else:  # no names on a past body: today's names would read the present into the past
        day = _parse_day(arg)
        root = snapshot_root(day)
        catalog = load_catalog(root)
        if not catalog:
            raise DigError(f"no channel catalog in {root} (the catalog starts 2026-01)")
        # declared parts only (that day's files); nothing live is read for a past day
        atlas = build_atlas(catalog, None, now=_end_of_day_utc(day), live_graph=None,
                            body=gather_body(root, catalog, live=None))
    path = out.write(f"body-{arg or 'now'}", json.loads(atlas.model_dump_json()))
    if organ:
        try:
            return organ_view(atlas, organ, out=str(path)) + note
        except names.NamingError as exc:
            raise DigError(str(exc)) from exc
    if arg is not None:
        return summarize(atlas, out=str(path))
    # the ledger note goes in the footer, so the file path stays the last line
    return summarize(atlas, out=str(path), extra_footer=(*_unnamed_footer(atlas), *([note.strip()] if note else [])))


def _unnamed(atlas) -> list:
    """Multi-part organs with no name, largest first (singletons are not named)."""
    return sorted((o for o in atlas.organs if len(o.members) > 1 and not o.name),
                  key=lambda o: -len(o.members))


def _unnamed_footer(atlas) -> tuple[str, ...]:
    n = len(_unnamed(atlas))
    return (f"{n} organ{'s' if n != 1 else ''} unnamed: python -P -m orion.dig name-organs",) if n else ()


def _name_rule() -> str:
    """Built from the limits ``names.check`` enforces, so the rule cannot drift from the check."""
    from orion.body_atlas import names as n

    return (f"Name each organ by what it takes in and gives out (1-{n.NAME_MAX_WORDS} words, at most "
            f"{n.NAME_MAX_CHARS} chars). The because ({n.BECAUSE_MIN_CHARS}-{n.BECAUSE_MAX_CHARS} chars) "
            "must cite at least one of its parts by full name, as written below.")


NAME_ORGANS_MEMBERS_CHARS = 300  # per organ in the printout; every member is in the file


def cmd_name_organs(out: Out) -> str:
    """Read-only: every unnamed multi-part organ of the live body, largest first, with its parts
    and a ready `name` line to fill in. Writes nothing but the dig result file."""
    from orion.body_atlas import names

    records, skipped = names.load(names_path())
    atlas = names.apply_names(_live_atlas(), records)
    part_name = {p.part_id: p.name for p in atlas.parts}
    multi = [o for o in atlas.organs if len(o.members) > 1]
    todo = _unnamed(atlas)
    note = f"names ledger: {skipped} unreadable line(s) skipped in {names_path()}" if skipped else None
    if not todo:
        lines = [f"all {len(multi)} multi-part organs are named:", *([note] if note else [])]
        lines += [f'  "{o.name}" {o.organ_id} ({len(o.members)} parts)' for o in multi]
        path = out.write("name-organs", {"unnamed": [], "named": [
            {"organ_id": o.organ_id, "name": o.name, "named_by": o.named_by, "size": len(o.members)} for o in multi]})
        return capped(lines, tail=f"full result: {path}")

    def cmd(o) -> str:
        # quoted: unquoted <...> is a bash redirect, and an apostrophe in a reason breaks bash
        return f'python -P -m orion.dig name {o.organ_id} "<name>" because "<reason citing a part>"'

    def hosts(o) -> str:
        return ", ".join(f"{h} {n}" for h, n in o.placement.items()) or "-"

    path = out.write("name-organs", {"rule": _name_rule(), "unnamed": [
        {"organ_id": o.organ_id, "size": len(o.members), "hosts": o.placement,
         "members": [part_name.get(m, m) for m in o.members], "command": cmd(o)} for o in todo]})
    tail = f"every part of every organ: {path}"
    lines = [f"{len(todo)} of {len(multi)} multi-part organs have no name, largest first.", _name_rule(),
             "Run each name line (a lasting write); rerun name-organs to see what is left."]
    if note:
        lines.append(note)
    room_lines = MAX_LINES - len(lines) - 2  # the tail and a "more" line
    room_chars = MAX_CHARS - sum(len(x) + 1 for x in lines) - len(tail) - 80
    shown = 0
    for o in todo:
        members = [part_name.get(m, m) for m in o.members]
        text, cut = "", 0
        for i, m in enumerate(members):
            nxt = f"{text}, {m}" if text else m
            if len(nxt) > NAME_ORGANS_MEMBERS_CHARS:
                if i == 0:  # always show one part to cite, cut to fit
                    text, i = m[:NAME_ORGANS_MEMBERS_CHARS], 1
                cut = len(members) - i
                break
            text = nxt
        block = [f"{shown + 1}. {o.organ_id}: {len(o.members)} parts ({_kinds_short(o.kinds)}); hosts {hosts(o)}",
                 f"   parts: {text}" + (f" (+{cut} more in the file)" if cut else ""),
                 f"   {cmd(o)}"]
        size = sum(len(x) + 1 for x in block)
        if shown and (len(block) > room_lines or size > room_chars):
            break  # the first organ always shows, so there is always one ready line
        lines += block
        room_lines -= len(block)
        room_chars -= size
        shown += 1
    if shown < len(todo):
        lines.append(f"showing {shown} of {len(todo)} unnamed organs; the rest are in the file")
    return capped(lines, tail=tail)


def _kinds_short(kinds: dict[str, int]) -> str:
    from orion.body_atlas.__main__ import _kinds

    return _kinds(kinds) or "-"


def cmd_name(ref: str, name: str, because: str, out: Out) -> str:
    from orion.body_atlas import names

    try:
        rec = names.check(_live_atlas(), names.load(names_path())[0], ref, name, because, dig=out.dig)
    except names.NamingError as exc:
        raise DigError(str(exc)) from exc
    names.append(names_path(), rec)
    replaced = f"; replaces the name carried from {rec.replaces[0]}" if rec.replaces else ""
    return (f'named {rec.organ_id} "{rec.name}" (cites: {", ".join(rec.cites[:6])}{replaced})\n'
            f"names ledger: {names_path()}")


def claims_path() -> Path:
    return home() / "self" / "capacity_claims.jsonl"


def capacities_path() -> Path:
    return home() / "self" / "capacities.jsonl"


CLAIM_CMD = "python -P -m orion.dig capacity claim '<part>' '<job>' because \"<reason citing its flow or a repo file>\""


def _capacity_reading():
    """The live reading with Juniper's confirmations and Orion's ledgers on it, the atlas, the
    sentences written so far, and a note per ledger with unreadable lines."""
    from orion.body_atlas import capacities, claims, names

    try:
        vsm = claims.load_confirmed(now_root())
    except claims.ClaimError as exc:
        raise DigError(str(exc)) from exc
    atlas = names.apply_names(_live_atlas(), names.load(names_path())[0])
    reading = capacities.read_capacities(atlas, vsm=vsm)
    claim_recs, s1 = claims.load_ledger(claims_path(), capacities.CapacityClaimV1)
    written, s2 = claims.load_ledger(capacities_path(), capacities.CapacityV1)
    reading = capacities.attach_ledgers(reading, atlas, vsm, claim_recs, written)
    for path, n in ((claims_path(), s1), (capacities_path(), s2)):
        if n:
            reading.notes.append(f"{n} unreadable line(s) skipped in {path}")
    return reading, atlas, written


def cmd_capacities(arg: str | None, out: Out) -> str:
    """Read-only: the VSM jobs the live atlas shows a part doing, by rule (orion/body_atlas/
    capacities.py) or by Juniper's confirmation, Orion's claims, and the capacity sentences.
    ``gaps``: the code-written gap lines. Writes nothing but the dig result file."""
    from orion.body_atlas import capacities

    reading, atlas, _ = _capacity_reading()
    path = out.write("capacities", json.loads(reading.model_dump_json()))
    tail = [f"full result: {path}"]
    if arg == "gaps":
        lines = capacities.render_gaps(reading)
    else:
        lines = capacities.render(reading)
        # In the tail, so a long printout never cuts it: a capacities turn ends with claims written
        # (2026-11-04: Orion read this printout, then offered claims instead of writing them).
        block = claim_block(reading, atlas)
        if block:
            tail = block + tail
        elif reading.gaps or any(c.status in ("gap", "not written", "stale") for c in reading.capacities):
            lines.append(f"to claim a job: {CLAIM_CMD} (capacity help: the rule)")
    return capped(lines, tail="\n".join(tail))


CLAIM_PREFIX = "python -P -m orion.dig capacity claim"
MAX_PENDING_SHOWN = 3


def claim_block(reading, atlas) -> list[str]:
    """The claims to write now, one ready line per open job, and the claims waiting on Juniper.
    A claim is not a role: Juniper confirms or rejects each in config/body_atlas/vsm_confirmed.yaml."""
    from orion.body_atlas import capacities, claims

    census = {c.channel for c in reading.census if c.census}
    lines: list[str] = []
    jobs = capacities.open_jobs(reading)
    if jobs:
        lines.append("Unclaimed jobs: claim each you can back with evidence now, without asking first (a claim "
                     "is not a role; Juniper reviews it). Cite by full id a part or channel on its flows, or a repo file:")
        for job, cands in jobs:
            name, hint = "'<part>'", ""
            if cands:
                part = cands[0]
                ok = claims.evidence_ids(atlas, part, census)
                # the rule's own evidence for this candidate, never an arbitrary flow peer
                ev = [e for r in reading.roles if r.part_id == part and r.role == job for e in r.evidence if e in ok]
                name = shlex.quote(part.removeprefix("part:"))
                hint = f"  # the rule saw: {', '.join(dict.fromkeys(ev[:3]))}" if ev else ""
            lines.append(f'  {CLAIM_PREFIX} {name} {shlex.quote(job)} because "<why>"{hint}')
        lines.append("No evidence for a job: leave it, and say which and why.")
    waiting = capacities.pending_claims(reading)
    if waiting:
        lines.append(f"Waiting on Juniper ({len(waiting)}): ask her yes or no on each (answers go in "
                     "config/body_atlas/vsm_confirmed.yaml):")
        lines += [f"  {c.part_id.removeprefix('part:')} {c.role}: {c.because[:60]}" for c in waiting[:MAX_PENDING_SHOWN]]
        if len(waiting) > MAX_PENDING_SHOWN:
            lines.append(f"  +{len(waiting) - MAX_PENDING_SHOWN} more in {claims_path()}")
    return lines


def cmd_capacity_help() -> str:
    from orion.body_atlas import claims

    return "\n".join([
        claims.claim_rule(),
        f"  {CLAIM_CMD}",
        '  python -P -m orion.dig capacity dispute <part> <job> because "<reason>"   (argue with a rule result)',
        '  python -P -m orion.dig capacity write cap:s2 "<one sentence>" because "<reason>"',
        "Gap lines are written by code; a claim never clears one. Ledgers: "
        f"{claims_path()}, {capacities_path()}",
    ])


def cmd_capacity(verb: str, a: list[str], out: Out) -> str:
    """``claim``/``dispute`` <part> <job> because <reason>; ``write`` <cap_id> <sentence> because
    <reason>. Lasting appends to the two ledgers under $ORION_DIG_HOME/self/."""
    from orion.body_atlas import claims

    reading, atlas, written = _capacity_reading()
    cut = a.index("because", 2)
    try:
        if verb == "write":
            rec = claims.check_capacity(reading, atlas, written, a[0], " ".join(a[1:cut]), " ".join(a[cut + 1:]),
                                        dig=out.dig)
            claims.append(capacities_path(), rec)
            return (f'wrote {rec.cap_id} "{rec.sentence}" (cites: {", ".join(rec.cites[:6])}'
                    + (", replaces the earlier one" if rec.replaces else "") + f")\ncapacities ledger: {capacities_path()}")
        if cut != 2:
            raise DigError(f"capacity {verb} <part> <job> because <reason>: one part and one job before `because`")
        rec = claims.check_claim(reading, atlas, a[0], a[1], " ".join(a[cut + 1:]), kind=verb, root=now_root(),
                                 dig=out.dig)
    except claims.ClaimError as exc:
        raise DigError(str(exc)) from exc
    claims.append(claims_path(), rec)
    after = ("it prints beside the rule result and changes nothing" if verb == "dispute"
             else "Orion's claim until Juniper confirms it in config/body_atlas/vsm_confirmed.yaml")
    return (f"{'disputed' if verb == 'dispute' else 'claimed'} {rec.part_id.removeprefix('part:')} {rec.role} (cites: {', '.join(rec.cites[:6])}); {after}"
            f"\nclaims ledger: {claims_path()}")


# --- body economy ---------------------------------------------------------------------------

BODY_ECONOMY_HELP = """body-economy [<section>...] [--host circe|athena|hecate] [--window 24h|7d|30d] [--family <f>]

What is using my power, what it costs, and what binds. Read-only (GET /physiology/report).
With no section: overview = limits, hosts, money, wants, waste, services (one line each).
Name one or two sections to see every row and note.

Sections:
  hosts       what is spinning each machine: whole-machine watts now vs my body at rest, $/month
              above rest, GPU / CPU / other watts, each card's watts and who held it, top CPU
              users, sponsored work apart.
              "what is spinning circe?" -> body-economy hosts --host circe
  processes   CPU watts above idle per container (estimated: a linear model), largest first.
  vram        GPU memory free per card, and its watts now.
  unit_costs  what one act, run, token or GPU-hour costs (measured / derived / not measured).
  money       electricity this month vs the envelope; what my body at rest alone reaches.
  limits      each limit now (heat, money, GPU lanes, messages, Claude) and why.
  wants, decisions, queue_wait, waste, rhythm, claude, value_per_cost, services.
  all         every section, one line each.

Flags:
  --host <h>     narrow hosts, processes and vram to one machine (others stay whole-body).
  --window <w>   24h (default), 7d or 30d: held time, daily averages, waste and wants use it.
  --family <f>   narrow wants, decisions, queue_wait, value_per_cost to one want family.

Measured vs estimated: every figure is labelled in the text. Whole-machine, card and CPU watts are
meters; rest is my body at rest, a pinned dated number (5th percentile of 2026-10-01..10-13
until config/physiology/rest_body.yaml says otherwise); per-container watts and $/month are
estimates. Every cost means the whole machine at the wall above rest, fans and power-supply loss
included (what my acts are charged): a card's or container's "(≈N W at the wall above rest)" is its
watts above idle x that machine's overhead factor, fitted on its own last 14 days; an ESTIMATE.
Sponsored work (Juniper's, e.g. the Zelda dojo) is shown apart from my activity.
A section with no data prints why, never a zero."""


def cmd_body_economy(args: list[str], out: Out) -> str:
    """skills.physiology.state.v1 from Bash: GET /physiology/report on orion-durable-runs
    ($ORION_PHYSIOLOGY_REPORT_URL), rendered as plain text; the full report JSON goes to a file."""
    from orion.physiology import report_text as rt

    window, family, host, words = "24h", None, None, []
    it = iter(args)
    for a in it:
        if a in ("--help", "-h", "help"):
            return BODY_ECONOMY_HELP
        if a == "--window":
            window = next(it, "")
        elif a == "--family":
            family = next(it, "") or None
        elif a == "--host":
            host = (next(it, "") or "").strip().lower() or None
            if host is None or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,31}", host):
                raise DigError("body-economy: --host needs a machine name (circe, athena, hecate)")
        elif a.startswith("--"):
            raise DigError(f"body-economy: unknown option {a} (use --host, --window or --family; --help lists them)")
        else:
            words.append(a)
    if window not in rt.WINDOWS:
        raise DigError(f"body-economy: --window must be one of {', '.join(rt.WINDOWS)}")
    try:
        sections = rt.normalize_sections(words or None)
        report = rt.fetch_report(window=window, sections=sections, family=family)
    except (ValueError, rt.ReportUnavailable) as exc:
        raise DigError(f"body-economy: {exc}") from exc
    path = out.write(f"body-economy-{window}" + (f"-{host}" if host else ""), report)
    return capped(rt.render_report_text(report, sections, host=host).splitlines(), tail=f"full report: {path}")


# --- PR field notes (spec docs/superpowers/specs/2026-11-04-pr-field-notes-design.md) ------------

def field_notes_path() -> Path:
    from orion.body_atlas import field_notes as fn

    return fn.ledger_path(home() / "self")


def _gh_repo() -> str:
    return os.environ.get("ORION_DIG_GITHUB_REPO", "junebug-junie/Orion-Sapienform")


PR_FIELDS = ("number,title,body,state,mergedAt,mergeCommit,labels,files,additions,deletions,changedFiles,"
             "headRefName,commits")


GH_TIMEOUT_S = 60


def _gh_pr(number: int) -> dict:
    try:
        proc = subprocess.run(["gh", "pr", "view", str(number), "--repo", _gh_repo(), "--json", PR_FIELDS],
                              capture_output=True, text=True, timeout=GH_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise DigError(f"gh pr view {number} gave no answer in {GH_TIMEOUT_S} s") from exc
    if proc.returncode != 0:
        raise DigError(f"gh pr view {number} failed: {proc.stderr.strip()[:300]}")
    try:
        return json.loads(proc.stdout)
    except ValueError as exc:
        raise DigError(f"gh pr view {number} returned no JSON: {proc.stdout[:200]!r}") from exc


def _gh_file_at(sha: str, path: str) -> str | None:
    """One file's text at the merge, through GitHub: nothing changes in the sandbox clone."""
    from urllib.parse import quote

    try:
        proc = subprocess.run(["gh", "api", "-H", "Accept: application/vnd.github.raw",
                               f"repos/{_gh_repo()}/contents/{quote(path)}?ref={sha}"], capture_output=True, text=True,
                              timeout=GH_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None
    return proc.stdout if proc.returncode == 0 else None


def _local_branches() -> set[str]:
    """Branches made in Orion's sandbox clone: one marker of a PR Orion opened (the other is the
    commit author orion-fcc@...; every PR's GitHub author is junebug-junie, so that field cannot tell)."""
    try:
        return set(_git("for-each-ref", "refs/heads", "--format=%(refname:short)").split())
    except DigError:
        return set()


def _denylist() -> set[str] | None:
    from orion.privacy.personal_terms import DENYLIST_REL, load_denylist

    try:
        hashes = load_denylist(now_root() / DENYLIST_REL)[0]
    except (OSError, ValueError):
        return None
    return hashes or None  # an empty list checks nothing: refuse, as for a missing one


def _parse_pr_number(text: str) -> int:
    t = text.lstrip("#")
    if not (t.isascii() and t.isdigit()) or int(t) < 1:
        raise DigError(f"not a PR number: {text!r} (want e.g. 3090)")
    return int(t)


def _field_note_packet(number: int, *, with_docs: bool = True) -> dict:
    """Everything code can say about one merged PR. Orion only writes the note."""
    from orion.body_atlas import field_notes as fn
    from orion.body_atlas import names

    pr = _gh_pr(number)
    if pr.get("state") != "MERGED" or not pr.get("mergedAt"):
        raise DigError(f"#{number} is not merged ({pr.get('state', '?').lower()}); field notes are for merged PRs")
    sha = (pr.get("mergeCommit") or {}).get("oid") or ""
    files = [f["path"] for f in pr.get("files") or []]
    flags = fn.flags_for(pr, local_branches=_local_branches())
    atlas, atlas_error = None, None
    try:
        atlas = names.apply_names(_live_atlas(), names.load(names_path())[0])
    except DigError as exc:
        atlas_error = str(exc)
    shared = {key for kind, key in map(fn.classify, files) if kind == "shared"}
    bmap = fn.body_map(files, atlas, fn.orion_importers(now_root(), shared))
    docs_text, read, cuts, full_docs = "", [], [], {}
    wanted = fn.docs_to_read(pr)
    if with_docs and wanted and not sha:
        cuts.append("no merge commit from GitHub: the specs it changed were not read")
    elif with_docs and wanted:
        # only the files the excerpt can use are fetched; the rest are counted in a cut
        full_docs = {p: _gh_file_at(sha, p) for p in wanted[:fn.DOC_MAX_FILES]}
        docs_text, read, cuts = fn.excerpt_docs(list(full_docs.items()) + [(p, "") for p in wanted[fn.DOC_MAX_FILES:]])
    n_changed = int(pr.get("changedFiles") or len(files))
    if n_changed > len(files):
        cuts.insert(0, f"file list: first {len(files)} of {n_changed} files (GitHub's cap); the map covers those only")
    return {"pr": pr, "sha": sha, "files": files, "flags": flags, "atlas": atlas, "atlas_error": atlas_error,
            "body_map": bmap, "not_body": fn.not_body(files), "docs_text": docs_text, "read": read, "cuts": cuts,
            "full_docs": full_docs}


def _map_lines(bmap) -> list[str]:
    by_prefix: dict[str, list] = {}
    for e in bmap:
        by_prefix.setdefault(e.prefix, []).append(e)
    lines = []
    for prefix, es in by_prefix.items():
        e0 = es[0]

        def organ(e) -> str:
            if not e.part_live:
                return f"{e.part_id} (no live part)"
            if not e.organ_id:
                return f"{e.part_id} (in no organ)"
            return f'{e.part_id} in organ {e.organ_id}' + (f' "{e.organ_name}"' if e.organ_name else " (unnamed)")

        if prefix.startswith("services/"):
            lines.append(f"  {prefix} ({e0.files} files) -> {organ(e0)}")
        elif e0.widely_shared:
            lines.append(f"  {prefix} ({e0.files} files) -> shared code, widely shared ({len(e0.importers)} services); no one organ")
        elif not e0.importers:
            lines.append(f"  {prefix} ({e0.files} files) -> shared code no service imports")
        else:
            lines.append(f"  {prefix} ({e0.files} files) -> shared code imported by: " + "; ".join(organ(e) for e in es))
    return lines


def cmd_field_note(number: int, out: Out) -> str:
    """Read-only: the packet for one merged PR, and the exact write line."""
    from orion.body_atlas import field_notes as fn

    pk = _field_note_packet(number)
    pr, flags = pk["pr"], pk["flags"]
    records, skipped = fn.load(field_notes_path())
    current = fn.live(records)
    docs_path = None
    if pk["docs_text"]:
        out.dir.mkdir(parents=True, exist_ok=True)
        docs_path = out.dir / f"field-note-{number}-docs.md"
        full = "\n\n".join(f"===== {p} (whole file at the merge) =====\n{t}" for p, t in pk["full_docs"].items() if t)
        docs_path.write_text(pk["docs_text"] + "\n\n\n##### the whole files #####\n\n" + full, encoding="utf-8")
    path = out.write(f"field-note-{number}", {
        "pr": pr, "merge_sha": pk["sha"], "flags": flags.model_dump(),
        "body_map": [e.model_dump() for e in pk["body_map"]], "not_body": pk["not_body"],
        "read": [r.model_dump() for r in pk["read"]], "cuts": pk["cuts"], "docs_file": str(docs_path or ""),
        "atlas_built_at": pk["atlas"].as_of if pk["atlas"] is not None else None, "atlas_error": pk["atlas_error"]})
    flag_words = [w for w, on in (("docs only", flags.docs_only), ("dependency-only", flags.deps_only),
                                  ("huge: say in the note what you could not read", flags.huge)) if on]
    if flags.revert_of:
        flag_words.append(f"revert of {flags.revert_of}")
    lines = [f"#{number} merged {pr['mergedAt'][:16].replace('T', ' ')}Z, sha {pk['sha'][:10]}: {pr['title'][:150]}",
             f"files: {len(pk['files'])} (+{pr.get('additions', 0)} -{pr.get('deletions', 0)})"
             + (f"; {', '.join(flag_words)}" if flag_words else "")
             + ("; labels: " + ", ".join(lab["name"] for lab in pr.get("labels") or []) if pr.get("labels") else "")]
    if flags.authored_by_me == "yes":
        lines.append("this may be your own work: its branch or commits come from your sandbox")
    if pk["atlas_error"]:
        lines.append(f"live body UNREAD ({pk['atlas_error'][:200]}): no organs below; a write needs it")
    lines.append("body map (code: services/<svc> -> its part -> its organ; orion/<pkg> -> who imports it):"
                 if pk["body_map"] else "body map: nothing in your body (no services/ or orion/ code changed)")
    map_lines = _map_lines(pk["body_map"])
    lines += map_lines[:MAX_MAP_LINES]
    if len(map_lines) > MAX_MAP_LINES:
        lines.append(f"  ... {len(map_lines) - MAX_MAP_LINES} more folders in the file")
    if pk["not_body"]:
        lines.append("not body: " + ", ".join(f"{k} {v}" for k, v in sorted(pk["not_body"].items())))
    if pk["read"]:
        lines.append(f"specs/reports it changed, read at the merge for you ({sum(r.chars_read for r in pk['read']):,} chars): "
                     f"{docs_path}")
    lines += [f"cut: {c}" for c in pk["cuts"]]
    if flags.revert_of and flags.revert_of.startswith("#"):
        orig = current.get(int(flags.revert_of[1:]))
        lines.append(f"note on {flags.revert_of}: " + (orig.note[:300] if orig and orig.note else "none"))
    mine = current.get(number)
    if skipped:
        lines.append(f"field-notes ledger: {skipped} unreadable line(s) skipped in {field_notes_path()}")
    body = [ln for ln in (pr.get("body") or "").splitlines() if ln.strip()]
    if body:
        lines.append("PR text (the builders' words, a secondary source; the whole body is in the file):")
        lines += ["  " + ln[:200] for ln in body]
    # what to do next goes in the tail, so a long map or body can never cut it (review, 2026-11-04)
    return capped(lines, tail="\n".join([*_field_note_action(number, flags, pr, mine), f"full packet: {path}"]))


MAX_MAP_LINES = 12


def _field_note_action(number: int, flags, pr: dict, mine) -> list[str]:
    from orion.body_atlas import field_notes as fn

    if fn.code_record_only(flags):
        why = "dependency-only" if flags.deps_only else f"docs-only, {pr.get('additions', 0) + pr.get('deletions', 0)} lines"
        return [f"no note for this PR ({why}): it gets a code record, not a note (Juniper, 2026-11-04)"]
    if mine and mine.note:
        return [f"your note ({mine.note_id}, {mine.written_at[:10]}): {mine.note[:300]}",
                f'to change it: python -P -m orion.dig field-note {number} rewrite "<note>" because "<refs>"']
    return [fn.note_rule(),
            f'write: python -P -m orion.dig field-note {number} write "[pr_text] ... [reflection] ..." '
            f'because "#{number} <changed files you cite>"']


def cmd_field_note_write(number: int, note: str, because: str, *, rewrite: bool, out: Out) -> str:
    """A lasting append to the field-notes ledger, after code checks the note."""
    from orion.body_atlas import field_notes as fn

    pk = _field_note_packet(number)
    if pk["atlas"] is None and pk["body_map"]:
        raise DigError(f"live body UNREAD ({(pk['atlas_error'] or '')[:200]}); this PR touches your body, "
                       "so its note waits until the body map can name the organs. Not saved.")
    pr = pk["pr"]
    path = field_notes_path()
    with fn.ledger_lock(path):  # check + append as one step: two turns cannot both write #n
        try:
            rec = fn.check(pr_number=number, merge_sha=pk["sha"], merged_at=pr["mergedAt"], title=pr["title"],
                           flags=pk["flags"], bmap=pk["body_map"], files=pk["files"], read=pk["read"], note=note,
                           because=because, atlas=pk["atlas"], records=fn.load(path)[0], denylist=_denylist(),
                           rewrite=rewrite, dig=out.dig)
        except fn.FieldNoteError as exc:
            raise DigError(str(exc)) from exc
        fn.append(path, rec)
    extra = f" (supersedes {rec.supersedes}; the old note is kept)" if rec.supersedes else ""
    return f"field-note #{number} saved {rec.note_id}{extra}\nfield-notes ledger: {path}"


def cmd_field_notes(start: date, end: date, out: Out) -> str:
    """Read-only: the live notes for PRs merged in [start, end] (Denver days) and note coverage."""
    from orion.body_atlas import field_notes as fn

    if end < start:
        raise DigError("end is before start")
    records, skipped = fn.load(field_notes_path())
    lo = datetime.combine(start, dtime(0), tzinfo=ZONE).astimezone(timezone.utc)
    hi = _end_of_day_utc(end)

    def in_window(ts: str) -> bool:
        return lo <= datetime.fromisoformat(ts.replace("Z", "+00:00")) < hi

    notes = sorted((r for r in fn.live(records).values() if in_window(r.merged_at)), key=lambda r: r.merged_at)
    coverage, missing, merged = "coverage UNREAD", [], None
    try:
        found, capped_days = _prs(start, end)
        merged = sorted({p["number"] for p in found})
        have = {r.pr_number for r in notes}
        missing = [n for n in merged if n not in have]
        if merged:
            noted = sum(1 for r in notes if r.note and r.pr_number in merged)
            coverage = (f"coverage {noted}/{len(merged)} merged PRs have a note ({100 * noted // len(merged)}%), "
                        f"{len(merged) - len(missing) - noted} a code record, {len(missing)} neither")
            if capped_days:
                coverage += f" (INCOMPLETE: search cap hit on {', '.join(capped_days)})"
        else:
            coverage = "no merged PRs in that window"
    except DigError as exc:
        coverage = f"coverage UNREAD ({str(exc)[:150]})"
    path = out.write(f"field-notes-{start}-{end}", {"start": str(start), "end": str(end),
                                                   "notes": [json.loads(r.model_dump_json()) for r in notes],
                                                   "merged": merged, "missing": missing})
    n_notes = sum(1 for r in notes if r.note)
    lines = [f"{start} .. {end}: {n_notes} field notes, {len(notes) - n_notes} code records; {coverage}"]
    if skipped:
        lines.append(f"field-notes ledger: {skipped} unreadable line(s) skipped in {field_notes_path()}")
    for r in notes:
        text = r.note[:300] if r.note else f"code record, not a note: {r.title[:120]}"
        lines.append(f"#{r.pr_number} {r.merged_at[:10]} {text}")
    if missing:
        lines.append("no note yet: " + " ".join(f"#{n}" for n in missing[:30]) + (" ..." if len(missing) > 30 else ""))
    return capped(lines, tail=f"full result: {path}")


# --- references (orion/refs.py) ----------------------------------------------------------------

def cmd_ref(kind: str, ref_id: str, out: Out) -> str:
    """Open one reference with the same resolver Prime used before it cited it."""
    from orion import refs

    try:
        ref = refs.RefV1(kind=kind, id=ref_id)
    except refs.ValidationError as exc:
        raise DigError(f"{kind}: {refs.error_words(exc)}") from exc
    res = refs.resolve(ref)
    if not res.ok:
        raise DigError(res.error or f"{kind} {ref_id}: unresolved")
    path = out.write(f"{kind}-{ref_id}", res.data)
    return capped(res.lines, tail=f"full {kind}: {path}")


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
    # newest last and always shown: shared notes are read for what was just added
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    keep = MAX_LINES - 2
    if len(lines) > keep:
        lines = [f"(showing the newest {keep} of {len(lines)} lines; older ones are in the file)", *lines[-keep:]]
    return capped(lines, tail=f"notes: {path}")


# One dig command per reference kind (orion/refs.py KINDS); orion/dig/tests/test_refs_contract.py checks the match.
REF_COMMANDS = ("arc", "act")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -P -m orion.dig", add_help=False)
    ap.add_argument("--dig", default=datetime.now(ZONE).date().isoformat())
    ap.add_argument("command", nargs="?", default="help")
    # REMAINDER: a reason or note may hold "-x" or "--dig" words; options go before the command
    ap.add_argument("args", nargs=argparse.REMAINDER)
    ns = ap.parse_args(argv)
    if not DIG_ID_RE.fullmatch(ns.dig) or ".." in ns.dig:
        print(f"dig: --dig must be a plain name (letters, digits, . _ -), got {ns.dig!r}", file=sys.stderr)
        return 2
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
        elif ns.command == "history" and len(a) == 1:
            text = cmd_history(*_last_days(a[0]), out)
        elif ns.command == "body" and len(a) <= 1:
            text = cmd_body(a[0] if a else None, out)
        elif ns.command == "capacities" and (not a or a == ["gaps"]):
            text = cmd_capacities(a[0] if a else None, out)
        elif ns.command == "capacity" and a[:1] in ([], ["help"]) and len(a) <= 1:
            text = cmd_capacity_help()
        elif ns.command == "capacity" and a[0] in ("claim", "dispute", "write") and len(a) >= 5 and "because" in a[3:]:
            text = cmd_capacity(a[0], a[1:], out)
        elif ns.command == "body-economy":
            text = cmd_body_economy(a, out)
        elif ns.command == "name-organs" and not a:
            text = cmd_name_organs(out)
        elif ns.command == "name" and len(a) >= 4 and "because" in a[2:]:
            cut = a.index("because", 2)  # name words before it, the reason after: quoted or not
            text = cmd_name(a[0], " ".join(a[1:cut]), " ".join(a[cut + 1:]), out)
        elif ns.command == "field-note" and len(a) == 1:
            text = cmd_field_note(_parse_pr_number(a[0]), out)
        elif ns.command == "field-note" and len(a) >= 5 and a[1] in ("write", "rewrite") and "because" in a[3:]:
            cut = len(a) - 1 - a[::-1].index("because")  # the last one: a note may say "because"
            text = cmd_field_note_write(_parse_pr_number(a[0]), " ".join(a[2:cut]), " ".join(a[cut + 1:]),
                                        rewrite=a[1] == "rewrite", out=out)
        elif ns.command == "field-notes" and len(a) == 2:
            text = cmd_field_notes(_parse_day(a[0]), _parse_day(a[1]), out)
        elif ns.command == "field-notes" and len(a) == 1:
            text = cmd_field_notes(*_last_days(a[0]), out)
        elif ns.command in REF_COMMANDS and len(a) == 1:
            text = cmd_ref(ns.command, a[0], out)
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
