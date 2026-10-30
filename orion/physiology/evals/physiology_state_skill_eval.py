"""Eval: skills.physiology.state.v1 on fixture data, end to end through the path Orion uses.

Runs ``python -P -m orion.dig body-economy`` as a subprocess (the exact Bash command a chat turn
or curiosity run uses) against a local HTTP server that serves fixture reports, then checks the
text Orion would read:

* grounded   -- every dollar figure in the text is a value in the report JSON;
* headline   -- "Binding now" names exactly the limits whose bind_state binds;
* honest     -- every asked section shows content or its ``missing`` reason, never silence;
* compact    -- the overview fits the dig cap (40 lines, 4000 chars);
* no shells  -- no "None", "nan" or raw JSON in the text;
* failure    -- an unreachable report gives a plain reason and exit 1, never an empty success;
* hosts      -- ``body-economy hosts --host circe`` answers "what is spinning circe?": its first line
                is circe's own watts now vs rest from the report, its $/month is the report's,
                no other machine appears, and it says what is measured and what is estimated;
                the overview carries one "what is spinning" line per machine;
* overhead   -- (live 2026-10-30T04:37Z fixture with the overhead fields) every card with an at-the-wall
                estimate shows it as "(≈N W at the wall above rest, est.)", the host's factor and how it was found
                are said, and the text calls the wall figures ESTIMATES (they are derived, not measured);
* brief      -- the dig line in every chat/curiosity prompt points at ``body-economy hosts`` for
                "what is using my power" and stays under its cap.

Scenarios: the live 24 h report captured 2026-10-30 (real data, with ``hosts``), an all-empty report,
and a report whose lease-event reads failed. Run: ``PYTHONPATH=. python orion/physiology/evals/physiology_state_skill_eval.py``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "physiology_report_live_2026-10-30_24h.json"
OVERHEAD_FIXTURE = FIXTURE.with_name("physiology_report_live_2026-10-30_overhead_24h.json")
LIVE = ("live", "live_overhead")
SECTIONS = ("limits", "wants", "rhythm", "money", "unit_costs", "waste", "queue_wait", "decisions",
            "claude", "vram", "value_per_cost", "services", "processes", "hosts")
OVERVIEW = ("limits", "hosts", "money", "wants", "waste", "services")
BINDING = ("binding", "exogenous_bind", "stuck_at_ceiling")


def scenarios() -> dict[str, dict]:
    live = json.loads(FIXTURE.read_text())
    empty = {"schema_version": "physiology.report.v1", "generated_at": live["generated_at"], "window": live["window"],
             "missing": [{"section": s, "reason": f"no {s} in the window (eval)"} for s in SECTIONS]}
    failed = json.loads(json.dumps(live))
    for s in ("waste", "queue_wait", "rhythm"):
        failed[s] = {"failed_lease_s_by_lane": {}, "top": []} if s == "waste" else []
        failed["missing"].append({"section": s, "reason": "read failed: OperationalError: canceling statement"})
    return {"live": live, "live_overhead": json.loads(OVERHEAD_FIXTURE.read_text()), "empty": empty,
            "failed_reads": failed}


class _Server:
    def __init__(self, reports: dict[str, dict]):
        self.reports = reports
        self.current = "live"
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                body = json.dumps(outer.reports[outer.current]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"


def run_dig(url: str, home: str, *args: str) -> tuple[int, str, str]:
    env = {**os.environ, "ORION_PHYSIOLOGY_REPORT_URL": url, "ORION_DIG_HOME": home,
           "PYTHONPATH": str(ROOT)}
    p = subprocess.run([sys.executable, "-P", "-m", "orion.dig", "body-economy", *args], cwd=home, env=env,
                       capture_output=True, text=True, timeout=60)
    return p.returncode, p.stdout, p.stderr


def _numbers(obj) -> set[str]:
    out: set[str] = set()
    if isinstance(obj, dict):
        for v in obj.values():
            out |= _numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= _numbers(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.add(f"{obj:.2f}")
    elif isinstance(obj, str):
        out |= set(re.findall(r"\$([0-9][0-9,]*\.[0-9]{2})\b", obj))
    return {n.replace(",", "") for n in out}


def check(name: str, report: dict, text: str, sections: tuple[str, ...]) -> list[str]:
    fails: list[str] = []
    body = text.rsplit("\nfull report:", 1)[0]
    known = _numbers(report)
    for usd in re.findall(r"\$([0-9][0-9,]*\.[0-9]{2})\b", body):
        if usd.replace(",", "") not in known:
            fails.append(f"{name}: ${usd} is not a value in the report")
    if "limits" in sections and report.get("limits"):
        want = sorted(l["label"] for l in report["limits"] if (l.get("now") or {}).get("bind_state") in BINDING)
        m = re.search(r"Binding now: (.*?)\.(?: |\n)", body)
        got = sorted(x.strip() for x in m.group(1).split(", ")) if m else []
        if want and got != want:
            fails.append(f"{name}: headline names {got}, report binds {want}")
    for s in sections:
        reasons = [m["reason"] for m in report.get("missing") or [] if m["section"] == s]
        has_body = bool(report.get(s)) and report.get(s) not in ({"failed_lease_s_by_lane": {}, "top": []},
                                                                  {"families": [], "hourly": []})
        if not has_body and not any(r in body for r in reasons):
            fails.append(f"{name}: section {s} is empty and its reason is not shown")
    for bad in ("None", "nan", '{"'):
        if bad in body:
            fails.append(f"{name}: text contains {bad!r}")
    return fails


def _host_head(h: dict) -> str:
    return f"{h['host']} draws {h['chassis_w_now']:.0f} W now vs {h['chassis_w_rest']:.0f} W at rest"


def check_hosts_circe(report: dict, text: str) -> list[str]:
    """"What is spinning circe?" in a couple of lines, grounded in the report."""
    fails: list[str] = []
    body = text.rsplit("\nfull report:", 1)[0]
    circe = next((h for h in report.get("hosts") or [] if h["host"] == "circe"), None)
    if circe is None:
        return ["hosts: the live fixture has no circe row"]
    lines = body.splitlines()
    if len(lines) < 3 or not lines[2].startswith(_host_head(circe)):
        fails.append(f"hosts --host circe: line 3 is not circe's headline {_host_head(circe)!r}: {lines[2:3]}")
    usd = circe.get("usd_per_month_at_this_rate")
    if usd is not None and f"about ${usd:,.0f}/month at this rate" not in body:
        fails.append(f"hosts --host circe: $/month above rest ${usd:,.0f} not shown")
    for other in report.get("hosts") or []:
        if other["host"] != "circe" and _host_head(other) in body:
            fails.append(f"hosts --host circe: {other['host']} shown too")
    for c in circe.get("cards") or []:
        if c.get("watts") is not None and f"- {c['card']} {c['watts']:.0f} W" not in body:
            fails.append(f"hosts --host circe: card {c['card']} missing")
        top = (c.get("top_holders") or [{}])[0]
        if top.get("label") and top["label"] not in body:
            fails.append(f"hosts --host circe: {c['card']}'s top holder {top['label']} missing")
    if "measured:" not in body or "Estimated:" not in body:
        fails.append("hosts --host circe: does not say what is measured vs estimated")
    if "machine circe" not in lines[0]:
        fails.append("hosts --host circe: the first line does not say the view is narrowed to circe")
    return fails


def check_overhead_circe(report: dict, text: str) -> list[str]:
    """Component watts are put on the whole-machine footing, and the text says they are estimates."""
    body = text.rsplit("\nfull report:", 1)[0]
    circe = next((h for h in report.get("hosts") or [] if h["host"] == "circe"), {})
    if circe.get("overhead_factor") is None:
        return []
    fails = []
    for c in circe.get("cards") or []:
        if c.get("watts_at_wall_est") is not None and f"(≈{c['watts_at_wall_est']:.0f} W at the wall above rest, est.)" not in body:
            fails.append(f"overhead: card {c['card']} has no at-the-wall estimate in the text")
    if f"draws about {circe['overhead_factor']:.2f} W at the wall above rest" not in body:
        fails.append("overhead: circe's factor is not said")
    if "ESTIMATES" not in body or (circe.get("overhead_basis") or "")[:40] not in body:
        fails.append("overhead: the text does not say the wall figures are estimates, or how the factor was found")
    return fails


def check_overview_hosts(report: dict, text: str) -> list[str]:
    body = text.rsplit("\nfull report:", 1)[0]
    return [f"overview: no 'what is spinning' line for {h['host']}" for h in report.get("hosts") or []
            if h.get("chassis_w_now") is not None and h.get("chassis_w_rest") is not None
            and f"- {_host_head(h)}" not in body]


def check_brief() -> list[str]:
    sys.path.insert(0, str(ROOT))
    from orion.dig.brief import DIG_BRIEF_LINE, MAX_BRIEF_CHARS

    fails = []
    if "orion.dig body-economy hosts" not in DIG_BRIEF_LINE or "power" not in DIG_BRIEF_LINE:
        fails.append("brief: the dig line does not point 'what is using my power' at body-economy hosts")
    if len(DIG_BRIEF_LINE) > MAX_BRIEF_CHARS:
        fails.append(f"brief: {len(DIG_BRIEF_LINE)} chars, over {MAX_BRIEF_CHARS}")
    return fails


def main() -> int:
    reports = scenarios()
    srv = _Server(reports)
    results: list[dict] = []
    fails: list[str] = []
    with tempfile.TemporaryDirectory() as home:
        for name in reports:
            srv.current = name
            for args, secs in (((), OVERVIEW), (("all", "--window", "7d"), SECTIONS)):
                code, out, err = run_dig(srv.url, home, *args)
                f = [] if code == 0 else [f"{name}: exit {code}: {err.strip()[:200]}"]
                if code == 0:
                    f += check(name, reports[name], out, secs)
                    if not args and (len(out.splitlines()) > 40 or len(out) > 4000):
                        f.append(f"{name}: overview over the dig cap ({len(out.splitlines())} lines, {len(out)} chars)")
                results.append({"scenario": name, "args": list(args), "exit": code, "lines": len(out.splitlines()),
                                "chars": len(out), "fails": f})
                fails += f
                if name in LIVE and not args and code == 0:
                    f2 = check_overview_hosts(reports[name], out)
                    results.append({"scenario": name, "args": ["overview:hosts"], "fails": f2})
                    fails += f2
            if name in LIVE:
                code, out, err = run_dig(srv.url, home, "hosts", "--host", "circe")
                f = [f"hosts --host circe: exit {code}: {err.strip()[:200]}"] if code else \
                    check(name, reports[name], out, ("hosts",)) + check_hosts_circe(reports[name], out) + \
                    check_overhead_circe(reports[name], out)
                results.append({"scenario": name, "args": ["hosts", "--host", "circe"], "exit": code,
                                "lines": len(out.splitlines()), "chars": len(out), "fails": f})
                fails += f
        srv.srv.shutdown()
        code, out, err = run_dig("http://127.0.0.1:9", home)
        unreachable_ok = code == 1 and "unreachable" in err and not out.strip()
        results.append({"scenario": "unreachable", "exit": code, "fails": [] if unreachable_ok else [err[:200]]})
        if not unreachable_ok:
            fails.append("unreachable: no plain failure")
    f = check_brief()
    results.append({"scenario": "dig_brief_line", "fails": f})
    fails += f
    print(json.dumps({"eval": "physiology_state_skill", "passed": not fails, "cases": results}, indent=1))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
