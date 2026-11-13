"""Eval: does the chat stance cue carry today's Retro session, truthfully, on real frames?

Orion plays Zelda, section F, acceptance check 6 ("Same-day, the Temporal Self cue renders it").
Each fixture session (fixtures/retro_play_sessions.json, the exact row shape Hub writes) goes through
the real path -- source adapter -> reducer -> frame summary -- and is placed into the two real frames
from 2026-10-11 (tests/patch4_rows: a crowded "talking" frame and a "busy" one). Then the real
``project_stance_cue`` renders it.

Gated (exit 1): the play clause is shown, and is exactly "Played <label> (<start> to <end>) [arc:<id>]."
with local times (label: orion.temporal_self.sources.retro_play_label), cites an arc that is in the frame, says "together" only for a played-together session and
"solo" only for a solo one, and the cue stays under its cap.
A frame already at ARCS_IN_FRAME_CAP loses its oldest arc to the session, the same pin rule
``build_frame`` applies to today's latest play arc.

    python orion/temporal_self/evals/run_play_cue_eval.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from orion.schemas.temporal_self import ARCS_IN_FRAME_CAP, STANCE_CUE_MAX_CHARS  # noqa: E402
from orion.temporal_self import sources as S  # noqa: E402
from orion.temporal_self.arcs import advance_clock, fold, initial_state  # noqa: E402
from orion.temporal_self.day import DEFAULT_TZ, as_utc  # noqa: E402
from orion.temporal_self.frame import summarize  # noqa: E402
from orion.temporal_self.stance_cue import project_stance_cue  # noqa: E402
from orion.temporal_self.tests.patch4_fixtures import BUSY_NOW, TALKING_NOW, frame  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "retro_play_sessions.json"


def play_summary(row: dict, now: datetime):
    state = advance_clock(fold(initial_state(), events=S.adapt("retro_play", [row])), now)
    (arc,) = [a for a in state.arcs.values() if a.kind == "play"]
    return summarize(arc)


def main() -> int:
    rows = json.loads(FIXTURE.read_text())["rows"]
    tz = ZoneInfo(DEFAULT_TZ)
    failures: list[str] = []
    shown = total = 0
    for name, now in (("talking", TALKING_NOW), ("busy", BUSY_NOW)):
        for row in rows:
            if as_utc(row["ended_at"]) > now:
                continue
            total += 1
            arc = play_summary(row, now)
            f = frame(name)
            listed = list(f["arcs_today"])
            if len(listed) >= ARCS_IN_FRAME_CAP:
                listed = sorted(listed, key=lambda a: a["began_at"])[1:]
            f["arcs_today"] = sorted(listed + [arc.model_dump(mode="json")], key=lambda a: (a["began_at"], a["arc_id"]))
            cue = project_stance_cue(f, now)
            began = as_utc(row["started_at"]).astimezone(tz).strftime("%H:%M")
            ended = as_utc(row["ended_at"]).astimezone(tz).strftime("%H:%M")
            want = f"Played {S.retro_play_label(row)} ({began} to {ended}) [arc:{arc.arc_id}]."
            case = f"{name}/{row['session_id']}"
            if len(cue.text) > STANCE_CUE_MAX_CHARS:
                failures.append(f"{case}: cue over cap ({len(cue.text)})")
            frame_ids = {a["arc_id"] for a in f["arcs_today"]} | {
                (f.get(k) or {}).get("arc_id") for k in ("active_arc", "previous_arc")}
            if not set(re.findall(r"\[arc:([0-9a-f]{16})\]", cue.text)) <= frame_ids:
                failures.append(f"{case}: cites an arc not in the frame")
            if "Played " in cue.text:
                if want not in cue.text:
                    failures.append(f"{case}: clause is not exact: {cue.text!r}")
                if ("together" in want) != bool(row["played_together"]):
                    failures.append(f"{case}: together claim does not match the control share")
                if ("solo" in want) != (row["kind"] == "leisure" and not row["played_together"]):
                    failures.append(f"{case}: solo claim does not match the session kind")
                shown += 1
                print(f"SHOWN   {case}: ...{cue.text[-150:]}")
            else:
                failures.append(f"{case}: play clause not shown (cue {len(cue.text)} chars, "
                                f"{cue.clauses_dropped} clause(s) dropped, skip={cue.skip_reason})")
                print(f"DROPPED {case}")
    print(f"\nplay clause survived the {STANCE_CUE_MAX_CHARS}-char cap in {shown} of {total} renders")
    for msg in failures:
        print("FAIL", msg)
    print("PASS" if not failures else f"FAIL ({len(failures)})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
