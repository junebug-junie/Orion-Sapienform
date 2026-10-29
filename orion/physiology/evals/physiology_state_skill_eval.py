"""Eval: skills.physiology.state.v1 on fixture data, end to end through the path Orion uses.

Runs ``python -P -m orion.dig body-economy`` as a subprocess (the exact Bash command a chat turn
or curiosity run uses) against a local HTTP server that serves fixture reports, then checks the
text Orion would read:

* grounded   -- every dollar figure in the text is a value in the report JSON;
* headline   -- "Binding now" names exactly the limits whose bind_state binds;
* honest     -- every asked section shows content or its ``missing`` reason, never silence;
* compact    -- the overview fits the dig cap (40 lines, 4000 chars);
* no shells  -- no "None", "nan" or raw JSON in the text;
* failure    -- an unreachable report gives a plain reason and exit 1, never an empty success.

Scenarios: the live 24 h report captured 2026-10-29 (real data), an all-empty report, and a report
whose lease-event reads failed. Run: ``PYTHONPATH=. python orion/physiology/evals/physiology_state_skill_eval.py``.
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
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "physiology_report_live_2026-10-29_24h.json"
SECTIONS = ("limits", "wants", "rhythm", "money", "unit_costs", "waste", "queue_wait", "decisions",
            "claude", "vram", "value_per_cost", "services", "processes")
BINDING = ("binding", "exogenous_bind", "stuck_at_ceiling")


def scenarios() -> dict[str, dict]:
    live = json.loads(FIXTURE.read_text())
    empty = {"schema_version": "physiology.report.v1", "generated_at": live["generated_at"], "window": live["window"],
             "missing": [{"section": s, "reason": f"no {s} in the window (eval)"} for s in SECTIONS]}
    failed = json.loads(json.dumps(live))
    for s in ("waste", "queue_wait", "rhythm"):
        failed[s] = {"failed_lease_s_by_lane": {}, "top": []} if s == "waste" else []
        failed["missing"].append({"section": s, "reason": "read failed: OperationalError: canceling statement"})
    return {"live": live, "empty": empty, "failed_reads": failed}


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


def main() -> int:
    reports = scenarios()
    srv = _Server(reports)
    results: list[dict] = []
    fails: list[str] = []
    with tempfile.TemporaryDirectory() as home:
        for name in reports:
            srv.current = name
            for args, secs in (((), ("limits", "money", "wants", "waste", "services")), (("all", "--window", "7d"), SECTIONS)):
                code, out, err = run_dig(srv.url, home, *args)
                f = [] if code == 0 else [f"{name}: exit {code}: {err.strip()[:200]}"]
                if code == 0:
                    f += check(name, reports[name], out, secs)
                    if not args and (len(out.splitlines()) > 40 or len(out) > 4000):
                        f.append(f"{name}: overview over the dig cap ({len(out.splitlines())} lines, {len(out)} chars)")
                results.append({"scenario": name, "args": list(args), "exit": code, "lines": len(out.splitlines()),
                                "chars": len(out), "fails": f})
                fails += f
        srv.srv.shutdown()
        code, out, err = run_dig("http://127.0.0.1:9", home)
        unreachable_ok = code == 1 and "unreachable" in err and not out.strip()
        results.append({"scenario": "unreachable", "exit": code, "fails": [] if unreachable_ok else [err[:200]]})
        if not unreachable_ok:
            fails.append("unreachable: no plain failure")
    print(json.dumps({"eval": "physiology_state_skill", "passed": not fails, "cases": results}, indent=1))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
