#!/usr/bin/env python3
"""Live check of game company patch 3 (Orion's lines while a person plays), after deploy. Read-only.

    python3 scripts/analysis/check_retro_lines_live.py [--hours 24] [--line-id <id>]

Reads the live database through ``docker exec orion-athena-sql-db psql`` (SELECT only) and checks the spec's patch 3
acceptance items that a database can show:

1. LEASE PRIORITY (spec: "lines do not ship at interactive"). For each line turn, every ``gpu_pool_events`` row of
   its GPU hold and of the calls that carried its correlation id. FAIL if any is ``interactive``. A line with no
   pool event found is reported UNVERIFIED (not a pass).
2. Every delivered line has a ``retro_line`` row naming the looks it was shown and a chat history row (its turn
   trace, by correlation id).
3. The note error rate: contradicted / checkable, beside the share of shown notes that were checkable.
4. The share of funded line turns that ended in a pass, and seconds by stage (stance, motor, finalize).
5. Guests: chat-box rows with source ``retro_guest`` and their Temporal Self events (speaker must not be juniper).
6. ``retro_look`` events: each label starts "Saw on the game screen:" and names a checked_holds look.

Exit 0 = no FAIL (UNVERIFIED items are printed, never counted as passes). Exit 1 = a FAIL.
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys

CONTAINER = "orion-athena-sql-db"
DB = "conjourney"


def q(sql: str) -> list[list[str]]:
    out = subprocess.run(["docker", "exec", CONTAINER, "psql", "-U", "postgres", "-d", DB, "-AtF", "\t",
                          "-v", "ON_ERROR_STOP=1", "-c", "BEGIN READ ONLY; " + sql + "; COMMIT;"],
                         capture_output=True, text=True, check=True).stdout
    return [line.split("\t") for line in out.splitlines() if line and line not in ("BEGIN", "COMMIT")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--line-id")
    ap.add_argument("--container", default=CONTAINER, help="the Postgres container (default the live one)")
    ap.add_argument("--db", default=DB)
    a = ap.parse_args()
    globals().update(CONTAINER=a.container, DB=a.db)
    where = (f"line_id = '{a.line_id}'" if a.line_id else
             f"decided_at >= now() - make_interval(secs => {float(a.hours) * 3600.0})")
    fails: list[str] = []
    unverified: list[str] = []
    if q("SELECT to_regclass('public.retro_line') IS NOT NULL")[0][0] != "t":
        print("UNVERIFIED: table retro_line does not exist (manual_migration_retro_lines_v1.sql not applied yet)")
        return 0
    lines = q(f"SELECT line_id, correlation_id, coalesce(hold_lease_id, ''), outcome, coalesce(array_length(look_ids, 1), 0), "
              f"looks_shown, looks_checkable, looks_contradicted, coalesce(stance_sec::text, ''), "
              f"coalesce(motor_sec::text, ''), coalesce(finalize_sec::text, ''), decided_at, "
              f"coalesce(finished_at, now()) FROM retro_line WHERE {where} ORDER BY decided_at")
    print(f"line turns: {len(lines)}")
    if not lines:
        print("UNVERIFIED: no line turn in the window")
        return 0
    outcomes: dict[str, int] = {}
    shown = checkable = contradicted = 0
    stages: dict[str, list[float]] = {"stance": [], "motor": [], "finalize": []}
    for (line_id, corr, hold, outcome, nlooks, s_shown, s_check, s_contra, st, mo, fi, t0, t1) in lines:
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        shown, checkable, contradicted = shown + int(s_shown), checkable + int(s_check), contradicted + int(s_contra)
        for k, v in (("stance", st), ("motor", mo), ("finalize", fi)):
            if v:
                stages[k].append(float(v))
        if outcome in ("no_frames",):
            continue
        events = q(f"SELECT event, coalesce(priority, ''), coalesce(holder, ''), coalesce(work_class, '') "
                   f"FROM gpu_pool_events WHERE turn_correlation_id = '{corr}'"
                   + (f" OR lease_id = '{hold}'" if hold else ""))
        prios = sorted({e[1] for e in events if e[1]})
        print(f"  {line_id} outcome={outcome} looks={nlooks} pool_events={len(events)} priorities={prios}")
        if "interactive" in prios:
            fails.append(f"line {line_id}: a GPU pool event at interactive ({events})")
        if not events and outcome != "failed":
            unverified.append(f"line {line_id}: no gpu_pool_events found for its hold or correlation id")
        if outcome == "delivered":
            if int(nlooks) == 0:
                unverified.append(f"line {line_id}: delivered with no look ids (no note since the last line?)")
            hist = q(f"SELECT count(*) FROM chat_history_log WHERE correlation_id = '{corr}'")
            if int(hist[0][0]) == 0:
                fails.append(f"line {line_id}: delivered but no chat_history_log row")
    funded = sum(outcomes.get(k, 0) for k in ("delivered", "passed"))
    print(f"outcomes: {outcomes}")
    print(f"pass share of funded turns that reached an answer: "
          f"{(outcomes.get('passed', 0) / funded):.0%}" if funded else "pass share: no answered turn")
    print(f"notes shown {shown}, checkable {checkable} ({(checkable / shown):.0%} of shown)" if shown else "notes shown 0")
    print(f"note error rate: {contradicted}/{checkable} = {(contradicted / checkable):.0%}" if checkable
          else "note error rate: nothing checkable yet (UNVERIFIED)")
    for k, v in stages.items():
        print(f"{k} seconds: median {statistics.median(v):.1f} over {len(v)}" if v else f"{k} seconds: none")

    guests = q("SELECT count(*) FROM chat_history_log WHERE source = 'retro_guest' "
               f"AND created_at >= now() - make_interval(secs => {float(a.hours) * 3600.0})")
    g_events = q("SELECT count(*), count(*) FILTER (WHERE payload_json->>'speaker' = 'juniper' OR subject_ref IS NOT NULL) "
                 "FROM temporal_self_event WHERE source_kind = 'chat_turn' AND payload_json->>'source' = 'retro_guest'")
    print(f"guest chat-box rows: {guests[0][0]}; their temporal events {g_events[0][0]}, read as Juniper {g_events[0][1]}")
    if int(g_events[0][1]):
        fails.append("a guest's chat-box turn was read as Juniper in temporal_self_event")
    looks = q("SELECT count(*), count(*) FILTER (WHERE e.label NOT LIKE 'Saw on the game screen:%'), "
              "count(*) FILTER (WHERE w.verdict IS DISTINCT FROM 'checked_holds') "
              "FROM temporal_self_event e LEFT JOIN retro_watch_look w ON w.look_id = e.source_ref "
              "WHERE e.source_kind = 'retro_look'")
    print(f"retro_look events: {looks[0][0]}, bad label {looks[0][1]}, not checked_holds {looks[0][2]}")
    if int(looks[0][1]) or int(looks[0][2]):
        fails.append("a retro_look event without the 'Saw on the game screen:' label or for an unsettled/contradicted look")
    for u in unverified:
        print("UNVERIFIED:", u)
    for f in fails:
        print("FAIL:", f)
    print("verdict:", "FAIL" if fails else "PASS (see UNVERIFIED lines)" if unverified else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
