"""Audit every heartbeat input against live grammar data.

Answers, for each (source_service, semantic_role) on orion:grammar:event:
  - how often it fires, and whether it fires on a fixed clock (hourly CV);
  - which atom fields really vary (distinct values);
  - which input rule routing.py applies (measured / presence / skip);
  - for measured inputs: does the level vary, and can it read calm on a quiet
    night (07-13 UTC = 00-06 Mountain)?

It flags rules that the data contradicts, so a drifting emitter shows up here
instead of silently gating the substrate:
  - MEASURED_FLAT: a measured input whose level never varies;
  - MEASURED_NO_REST: a measured input on a fixed clock whose night p05 level
    is above 0.05 (a structural floor -- it cannot read calm). Event-driven
    inputs read calm by going silent, so this check does not apply to them;
  - PRESENCE_CLOCKWORK: a presence input that fires on a fixed clock (hourly
    CV < 0.35), so even its timing carries nothing;
  - UNLISTED_ROLE: a role on a routed service that no ROLE_RULES entry names.

Input: the same 7-column CSV as replay_overhaul_before_after.py, over as long a
window as grammar_events keeps (about 3 days):

    docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "copy (
      select extract(epoch from emitted_at), source_service,
             event_json->'atom'->>'atom_type', event_json->'atom'->>'semantic_role',
             event_json->'atom'->>'confidence', event_json->'atom'->>'salience',
             event_json->'atom'->>'uncertainty'
      from grammar_events where event_kind='atom_emitted' order by emitted_at)
      to stdout with csv" > grammar_all.csv
    python services/orion-heartbeat/evals/audit_heartbeat_inputs.py --csv grammar_all.csv

Pure Python, no quimb. Exit code 1 if any flag fires, so it can run as a gate.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_SERVICE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SERVICE_ROOT.parents[1]
for _p in (str(_REPO_ROOT), str(_SERVICE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.substrate.routing import (  # noqa: E402
    ORGAN_SITE_MAP,
    SELF_SOURCE_SERVICE,
    SITE_ORGAN_MAP,
    UNROUTED_PRODUCERS,
    input_rule,
)

NIGHT_UTC_HOURS = range(7, 13)  # 00:00-06:00 Mountain
REST_LEVEL = 0.02
NO_REST_P05 = 0.05
# Fixed-cadence emitters (bus, biometrics, window observers) measure 0.21-0.31
# over 3 days (partial edge hours, restarts); event-driven ones 0.6+.
CLOCKWORK_CV = 0.35


def _f(value: str):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def audit(rows: list[list[str]]) -> list[dict]:
    groups: dict[tuple[str, str], list[list[str]]] = defaultdict(list)
    for row in rows:
        groups[(row[1], row[3])].append(row)
    t0 = float(rows[0][0])
    t1 = float(rows[-1][0])
    days = max((t1 - t0) / 86400.0, 1e-6)
    hours_total = max(1, int((t1 - t0) // 3600) + 1)
    out = []
    for (service, role), grp in sorted(groups.items()):
        hourly = defaultdict(int)
        for r in grp:
            hourly[int((float(r[0]) - t0) // 3600)] += 1
        counts = [hourly.get(h, 0) for h in range(hours_total)]
        mean_h = statistics.fmean(counts)
        cv = (statistics.pstdev(counts) / mean_h) if mean_h > 0 else None
        distinct = {
            name: len({r[idx] for r in grp if r[idx] not in ("", None)})
            for name, idx in (("confidence", 4), ("salience", 5), ("uncertainty", 6))
        }
        rec = {
            "service": service,
            "role": role,
            "per_day": round(len(grp) / days),
            "hourly_cv": None if cv is None else round(cv, 2),
            "distinct": distinct,
            "seat": None,
            "rule": None,
            "listed": None,
            "reason": None,
            "level_p50_day": None,
            "level_p05_night": None,
            "rest_share_night": None,
            "flags": [],
        }
        if service == SELF_SOURCE_SERVICE:
            rec["rule"] = "self"
            rec["reason"] = "heartbeat's own verdict atoms; skipped before routing"
        elif service not in ORGAN_SITE_MAP:
            rec["rule"] = "unrouted"
            rec["reason"] = UNROUTED_PRODUCERS.get(service, "NOT NAMED in UNROUTED_PRODUCERS")
            if service not in UNROUTED_PRODUCERS:
                rec["flags"].append("UNNAMED_PRODUCER")
        else:
            rule, listed = input_rule(service, role)
            rec["seat"] = SITE_ORGAN_MAP[ORGAN_SITE_MAP[service]]
            rec["rule"] = rule.kind
            rec["listed"] = listed
            rec["reason"] = rule.reason
            if not listed:
                rec["flags"].append("UNLISTED_ROLE")
            if rule.kind == "presence" and cv is not None and cv < CLOCKWORK_CV:
                rec["flags"].append("PRESENCE_CLOCKWORK")
            if rule.kind == "measured":
                day, night = [], []
                for r in grp:
                    atom = {
                        "atom_type": r[2],
                        "semantic_role": r[3],
                        "confidence": _f(r[4]),
                        "salience": _f(r[5]),
                        "uncertainty": _f(r[6]),
                    }
                    level = rule.level_fn(atom)
                    if level is None:
                        continue
                    hour = datetime.fromtimestamp(float(r[0]), timezone.utc).hour
                    (night if hour in NIGHT_UTC_HOURS else day).append(level)
                levels = day + night
                rec["level_p50_day"] = None if not day else round(_pct(day, 0.5), 3)
                rec["level_p05_night"] = None if not night else round(_pct(night, 0.05), 3)
                rec["rest_share_night"] = (
                    None if not night else round(sum(1 for v in night if v <= REST_LEVEL) / len(night), 3)
                )
                if len({round(v, 6) for v in levels}) <= 1:
                    rec["flags"].append("MEASURED_FLAT")
                clockwork = cv is not None and cv < CLOCKWORK_CV
                if clockwork and night and _pct(night, 0.05) > NO_REST_P05:
                    rec["flags"].append("MEASURED_NO_REST")
        out.append(rec)
    return out


def to_markdown(records: list[dict]) -> str:
    lines = [
        "| service | role | per day | hourly CV | distinct c/s/u | seat | rule | night p05 level | night rest share | flags |",
        "| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | --- |",
    ]
    for r in records:
        d = r["distinct"]
        lines.append(
            f"| {r['service']} | {r['role']} | {r['per_day']} | {r['hourly_cv']} | "
            f"{d['confidence']}/{d['salience']}/{d['uncertainty']} | {r['seat'] or '-'} | {r['rule']} | "
            f"{'' if r['level_p05_night'] is None else r['level_p05_night']} | "
            f"{'' if r['rest_share_night'] is None else r['rest_share_night']} | {' '.join(r['flags'])} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", required=True)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(argv)
    with open(args.csv, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.reader(fh) if r and r[0]]
    records = audit(rows)
    print(to_markdown(records))
    if args.json_out:
        import json

        Path(args.json_out).write_text(json.dumps(records, indent=2), encoding="utf-8")
    flagged = [r for r in records if r["flags"]]
    if flagged:
        print(f"\n{len(flagged)} input(s) flagged", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
