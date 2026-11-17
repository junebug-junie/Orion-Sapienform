#!/usr/bin/env python3
"""Fill ``retro_sitting`` from the Retro evidence journals: a baseline, not a memory.

Game company design, section 1 ("History for day one"). The journals hold who pressed what and when.
This replays them through the SAME tracker the live gateway uses (orion/retro/sitting.py: a sitting is
counted by changes of the button mask), so the baseline and the live count come from one instrument.

It writes ``retro_sitting`` rows with ``backfilled = true`` and NOTHING else: no ``retro_play_sessions``
row, no Temporal Self event, no memory card. Orion did not live those sittings.

Only rows the journal itself attributes to a human are counted. Journals from before controller
attribution (2026-10-23) carry no ``controller`` field; their play is not claimed.

One known difference from the live counter: the journal records the button mask but not the analog
stick (no ``applied`` row in any journal has an ``axes`` field). The live counter also counts a stick
push (N64). So stick-only play in an N64 journal would be undercounted here. Checked 2026-11-17: the
journals hold one N64 session (Mario Kart 64) and it has no input row at all, so nothing is lost today.

It never plans a sitting the live recorder may also hold: the plan stops at ``--until``. By default
that is the start of the first live row in ``retro_sitting`` (read through --psql, a SELECT), or,
when there is none or the database cannot be read, 5 minutes before now, so a game still being
played is not written as a half sitting.

Dry run (default): writes the plan and the report under --out and touches no database.

    python3 scripts/backfill_retro_sittings.py

Apply (a production write; run it only on Juniper's word, after manual_migration_retro_watch_v1.sql):

    python3 scripts/backfill_retro_sittings.py --apply

Files under --out (default /tmp/retro-sitting-backfill): progress.log, plan.csv, inserts.sql,
report.md, before_after.csv. Watch it: tail -f /tmp/retro-sitting-backfill/progress.log
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orion.retro.sitting import GAP_SEC_DEFAULT, MIN_SEC_DEFAULT, Sitting, sittings_from_journal  # noqa: E402

DEFAULT_SERVICE = Path.home() / ".local/share/orion/retro/service"
DEFAULT_PSQL = "docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1"
COLUMNS = ("sitting_id", "retro_session_id", "game_id", "game_title", "started_at", "last_input_at", "ended_at",
           "end_reason", "mask_changes", "saves", "loads", "pauses", "player_stated", "backfilled")


def sitting_id(s: Sitting) -> str:
    """Deterministic, so a second run inserts nothing new."""
    return "backfill-" + hashlib.sha1(f"{s.retro_session_id}:{s.started_at:.3f}".encode()).hexdigest()[:20]


def titles(catalog: Path) -> dict[str, str]:
    try:
        data = json.loads(catalog.read_text())
    except (OSError, ValueError):
        return {}
    games = data if isinstance(data, list) else data.get("games", []) if isinstance(data, dict) else []
    return {str(g["game_id"]): str(g.get("title") or g["game_id"]) for g in games if isinstance(g, dict) and g.get("game_id")}


def journal_counts(path: Path) -> tuple[str | None, int, int]:
    """(game_id, human rows, rows with no controller field) of one journal: the counter check."""
    game, human, unattributed = None, 0, 0
    with path.open("rb") as fh:
        for line in fh:
            if b'"kind":"applied"' in line:
                if b'"controller":"human"' in line:
                    human += 1
                elif b'"controller"' not in line:
                    unattributed += 1
            elif game is None and b'"kind":"start"' in line:
                try:
                    game = json.loads(line).get("game_id")
                except ValueError:
                    pass
    return game, human, unattributed


def iso(epoch: float | None) -> str | None:
    return None if epoch is None else datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def sql_literal(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return "'" + str(v).replace("'", "''") + "'"


def first_live_start(psql_cmd: str) -> float | None:
    """Epoch start of the first sitting the live recorder stored, read with one SELECT. None when
    there is none or the database cannot be read."""
    try:
        proc = subprocess.run(shlex.split(psql_cmd) + ["-At"], capture_output=True, text=True, timeout=30,
                              input="SELECT extract(epoch FROM min(started_at)) FROM retro_sitting WHERE NOT backfilled;")
        return float(proc.stdout.strip()) if proc.returncode == 0 and proc.stdout.strip() else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def row_of(s: Sitting, title_of: dict[str, str]) -> dict:
    return {"sitting_id": sitting_id(s), "retro_session_id": s.retro_session_id, "game_id": s.game_id,
            "game_title": title_of.get(s.game_id or "", s.game_id), "started_at": iso(s.started_at),
            "last_input_at": iso(s.last_input_at), "ended_at": iso(s.ended_at), "end_reason": s.end_reason,
            "mask_changes": s.mask_changes, "saves": s.saves, "loads": s.loads, "pauses": s.pauses,
            "player_stated": None, "backfilled": True}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evidence-dir", type=Path, default=DEFAULT_SERVICE / "evidence")
    ap.add_argument("--catalog", type=Path, default=DEFAULT_SERVICE / "catalog/catalog.json")
    ap.add_argument("--days", type=float, default=30.0, help="how far back a sitting may start")
    ap.add_argument("--gap-sec", type=float, default=GAP_SEC_DEFAULT)
    ap.add_argument("--min-sec", type=float, default=MIN_SEC_DEFAULT, help="reported only; every sitting is stored")
    ap.add_argument("--out", type=Path, default=Path("/tmp/retro-sitting-backfill"))
    ap.add_argument("--psql", default=DEFAULT_PSQL, help="command that reads SQL on stdin")
    ap.add_argument("--apply", action="store_true", help="write to the database (default: dry run)")
    ap.add_argument("--now", default=None, help="ISO time to count --days back from (tests)")
    ap.add_argument("--until", default=None, help="ISO time: a sitting with input at or after it is not planned "
                                                  "(default: the first live sitting's start, else now minus the gap)")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "progress.log").open("a")

    def say(text: str) -> None:
        line = f"{datetime.now(timezone.utc).strftime('%H:%M:%SZ')} {text}"
        print(line, flush=True); log.write(line + "\n"); log.flush()

    now = datetime.fromisoformat(args.now) if args.now else datetime.now(timezone.utc)
    since = (now - timedelta(days=args.days)).timestamp()
    if args.until:
        until, until_why = datetime.fromisoformat(args.until).timestamp(), "--until"
    else:
        live = first_live_start(args.psql)
        open_edge = now.timestamp() - args.gap_sec
        until, until_why = ((live, "the first live sitting") if live is not None and live < open_edge
                            else (open_edge, "now minus the gap (no live sitting was read)" if live is None
                                  else "now minus the gap"))
    journals = sorted(args.evidence_dir.glob("*.jsonl"))
    title_of = titles(args.catalog)
    say(f"backfill start mode={'APPLY' if args.apply else 'dry-run'} journals={len(journals)} since={iso(since)} "
        f"until={iso(until)} ({until_why})")

    rows: list[dict] = []
    per_game: dict[str, dict] = defaultdict(lambda: {"journals": 0, "human_rows": 0, "unattributed_rows": 0,
                                                     "mask_changes": 0, "sittings": 0, "minutes": 0.0, "long": 0})
    t0, errors, after_until = time.time(), 0, 0
    for i, path in enumerate(journals, 1):
        try:
            game, human, unattributed = journal_counts(path)
            g = per_game[game or "unknown"]
            g["journals"] += 1; g["human_rows"] += human; g["unattributed_rows"] += unattributed
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for s in sittings_from_journal(fh, gap_sec=args.gap_sec):
                    g["mask_changes"] += s.mask_changes
                    if s.started_at < since:
                        continue
                    if s.last_input_at >= until:      # the live recorder's, or still being played
                        after_until += 1
                        continue
                    g["sittings"] += 1; g["minutes"] += s.duration_sec / 60.0
                    g["long"] += int(s.duration_sec >= args.min_sec)
                    rows.append(row_of(s, title_of))
        except Exception as exc:  # noqa: BLE001 -- one bad journal must not stop the rest
            errors += 1
            say(f"journal failed file={path.name} error={type(exc).__name__}: {exc}")
        done = i / len(journals) if journals else 1.0
        rate = i / max(1e-6, time.time() - t0)
        say(f"journals {i}/{len(journals)} ({done:.0%}) eta={max(0.0, (len(journals) - i) / rate):.0f}s "
            f"rate={rate:.1f}/s sittings={len(rows)} errors={errors}")

    rows.sort(key=lambda r: r["started_at"])
    with (args.out / "plan.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS); w.writeheader(); w.writerows(rows)
    values = ",\n".join("(" + ", ".join(sql_literal(r[c]) for c in COLUMNS) + ")" for r in rows)
    sql = (f"INSERT INTO retro_sitting ({', '.join(COLUMNS)}) VALUES\n{values}\n"
           "ON CONFLICT (sitting_id) DO NOTHING;\n") if rows else "-- no sittings in the window\n"
    (args.out / "inserts.sql").write_text(sql)

    before = after = None
    verdict = "DRY RUN: nothing was written"
    if args.apply and rows:
        def psql(text: str) -> str:
            proc = subprocess.run(shlex.split(args.psql) + ["-At"], input=text, capture_output=True, text=True, timeout=120)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip()[:400])
            return proc.stdout.strip()
        try:
            before = int(psql("SELECT count(*) FROM retro_sitting;"))
            psql(sql)
            after = int(psql("SELECT count(*) FROM retro_sitting;"))
            verdict = f"APPLIED: {after - before} new rows ({len(rows)} planned; the rest were already there)"
        except Exception as exc:  # noqa: BLE001
            errors += 1
            verdict = f"FAILED: {type(exc).__name__}: {exc}"
        say(verdict)
    with (args.out / "before_after.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["table", "rows_before", "rows_after", "planned"]); w.writerow(["retro_sitting", before, after, len(rows)])

    lines = [f"# Retro sitting backfill", "", f"- Verdict: {verdict}", f"- Mode: {'apply' if args.apply else 'dry run'}",
             f"- Window: sittings that started after {iso(since)} and had no input at or after {iso(until)} ({until_why})",
             f"- Left out because they reach past that time: {after_until}",
             f"- Journals read: {len(journals)}; errors: {errors}",
             f"- Sittings planned: {len(rows)}; of those {sum(1 for r in rows if _sec(r) >= args.min_sec)} lasted {int(args.min_sec)} s or more",
             f"- Minutes of play in the window: {sum(_sec(r) for r in rows) / 60.0:.0f}",
             "- Written: `retro_sitting` only. No play row, no Temporal Self event, no memory.", "",
             "## The counter, per game", "",
             "A sitting is counted by changes of the button mask. `rows per change` is how many journal rows one real",
             "press or release stands for; `unattributed` rows are from before the journal named its controller and are not claimed.",
             "The journal holds no analog stick, so stick-only play (N64) would be undercounted here; the live counter counts it.", "",
             "| game | journals | human rows | mask changes | rows per change | unattributed rows | sittings in window | minutes | long sittings |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for game, g in sorted(per_game.items(), key=lambda kv: -kv[1]["human_rows"]):
        ratio = f"{g['human_rows'] / g['mask_changes']:.1f}" if g["mask_changes"] else "-"
        lines.append(f"| {game} | {g['journals']} | {g['human_rows']} | {g['mask_changes']} | {ratio} | "
                     f"{g['unattributed_rows']} | {g['sittings']} | {g['minutes']:.0f} | {g['long']} |")
    lines += ["", "## Longest sittings", ""]
    for r in sorted(rows, key=_sec, reverse=True)[:10]:
        lines.append(f"- {r['started_at'][:16]}Z {r['game_title']}: {_sec(r) / 60.0:.0f} min, {r['mask_changes']} mask changes, "
                     f"{r['saves']} saves, {r['loads']} loads, ended {r['end_reason']}")
    lines += ["", f"Files: {args.out}/progress.log, plan.csv, inserts.sql, report.md, before_after.csv", ""]
    (args.out / "report.md").write_text("\n".join(lines))
    say(f"backfill done verdict={verdict!r} report={args.out / 'report.md'}")
    return 1 if errors else 0


def _sec(r: dict) -> float:
    return (datetime.fromisoformat(r["ended_at"]) - datetime.fromisoformat(r["started_at"])).total_seconds()


if __name__ == "__main__":
    raise SystemExit(main())
