"""Weekly ablation eval for Orion's now (integration patch 3, #3425): does a cite really shape the output?

Spec: docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md, Part A metric gate step 4 and the
patch 3 acceptance check: "removing a cited item from a replayed prompt changes the output in most cases; the
hollow-cite share is reported". Local model only; never an outside model.

For each case and each cited event id: replay the prompt as shown, and again with the cited item's
``[experience:<id>]`` line removed (``ablate``). Same model, temperature 0. The cite is **changed** when the two
outputs differ by at least ``--threshold`` (token-set Jaccard distance) or the full output names the cited ref and
the ablated one does not. Otherwise it is **hollow**: the organ cited something its output did not depend on.

Lanes:

* ``--offline`` (CI, < 2 s): the checked-in fixtures (built with the real ``now_lines`` renderer) and a
  deterministic stub model that writes from the lines its case says it uses. One fixture is a planted hollow cite.
  This proves the harness logic, not Orion's behavior.
* ``--live``: replays ``--cases`` (default: the fixtures) through ``--model-url`` (an OpenAI-compatible local
  endpoint; default the repo's local gateway convention). Numbers from fixtures are fixture numbers, not live organ
  numbers: the real organ prompts are not captured yet (follow-up issue), so a live organ-level number is UNVERIFIED.

Usage:

    python -m orion.integration.evals.run_now_ablation_eval --offline
    python -m orion.integration.evals.run_now_ablation_eval --live --model-url http://127.0.0.1:8222/v1/chat/completions \\
        --model agent --out /path/to/report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "now_ablation_cases.jsonl"
DEFAULT_MODEL_URL = "http://127.0.0.1:8222/v1/chat/completions"
DEFAULT_THRESHOLD = 0.25

Model = Callable[[dict[str, Any], str, str], str]  # (case, system, prompt) -> output


class AblationError(ValueError):
    """A case that cannot be ablated (its cited item is not a line of its prompt)."""


def ablate(prompt: str, event_id: str) -> str:
    """The prompt with the cited item's line removed. Refuses when the line is not there: a cite that was never
    shown cannot be ablated, and ablating nothing would read as a hollow cite."""
    ref = f"[experience:{event_id}]"
    lines = prompt.split("\n")
    kept = [ln for ln in lines if ref not in ln]
    if len(kept) == len(lines):
        raise AblationError(f"{ref} is not a line of the prompt")
    return "\n".join(kept)


_TOKEN = re.compile(r"[a-z0-9_.:\-]+")


def distance(a: str, b: str) -> float:
    """Token-set Jaccard distance: 0.0 identical sets, 1.0 nothing shared."""
    ta, tb = set(_TOKEN.findall((a or "").lower())), set(_TOKEN.findall((b or "").lower()))
    if not ta and not tb:
        return 0.0
    return 1.0 - len(ta & tb) / len(ta | tb)


@dataclass
class CiteResult:
    case_id: str
    organ: str
    event_id: str
    distance: float
    changed: bool
    names_ref_full: bool
    names_ref_ablated: bool
    full: str = ""
    ablated: str = ""


@dataclass
class Report:
    mode: str
    results: list[CiteResult] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        n = len(self.results)
        changed = sum(r.changed for r in self.results)
        per: dict[str, dict[str, Any]] = {}
        for r in self.results:
            o = per.setdefault(r.organ, {"n": 0, "changed": 0})
            o["n"] += 1
            o["changed"] += int(r.changed)
        for o in per.values():
            o["changed_share"] = o["changed"] / o["n"]
            o["hollow_share"] = 1.0 - o["changed_share"]
        return {
            "mode": self.mode,
            "cites": n,
            "changed_share": (changed / n) if n else None,
            "hollow_share": ((n - changed) / n) if n else None,
            "per_organ": per,
            "hollow": [f"{r.case_id}:{r.event_id} (distance {r.distance:.2f})" for r in self.results if not r.changed],
            "refused": self.refused,
        }


def run(cases: list[dict[str, Any]], model: Model, *, mode: str, threshold: float = DEFAULT_THRESHOLD) -> Report:
    report = Report(mode=mode)
    for case in cases:
        system, prompt = str(case.get("system") or ""), str(case["prompt"])
        full: Optional[str] = case.get("original_output")
        for eid in case.get("cited_event_ids") or []:
            try:
                cut = ablate(prompt, eid)
            except AblationError as exc:
                report.refused.append(f"{case['case_id']}: {exc}")
                continue
            if full is None:
                full = model(case, system, prompt)
            out = model(case, system, cut)
            ref = f"experience:{eid}"
            d = distance(full, out)
            in_full, in_cut = ref in full, ref in out
            report.results.append(CiteResult(
                case_id=case["case_id"], organ=str(case.get("organ") or "?"), event_id=eid, distance=d,
                changed=d >= threshold or (in_full and not in_cut), names_ref_full=in_full,
                names_ref_ablated=in_cut, full=full[:400], ablated=out[:400]))
    return report


def stub_model(case: dict[str, Any], _system: str, prompt: str) -> str:
    """Deterministic: writes a fixed opening plus the label of every line it 'uses' (``stub_uses``) that is still in
    the prompt. A cite the stub does not use cannot change its output: that is the planted hollow case."""
    parts = ["I read what my other organs lived."]
    for line in prompt.split("\n"):
        for eid in case.get("stub_uses") or []:
            if f"[experience:{eid}]" in line:
                parts.append(f"{line.split('] ', 1)[-1]} (experience:{eid})")
    return " ".join(parts)


def http_model(url: str, model: str, *, timeout: float = 120.0, max_tokens: int = 400) -> Model:
    def call(_case: dict[str, Any], system: str, prompt: str) -> str:
        body = {"model": model, "temperature": 0, "seed": 7, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 -- local model only
            data = json.loads(resp.read())
        return str(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    return call


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    lane = ap.add_mutually_exclusive_group(required=True)
    lane.add_argument("--offline", action="store_true", help="fixtures + deterministic stub (CI)")
    lane.add_argument("--live", action="store_true", help="replay through a local model")
    ap.add_argument("--cases", type=Path, default=FIXTURES)
    ap.add_argument("--model-url", default=DEFAULT_MODEL_URL)
    ap.add_argument("--model", default="agent")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--out", type=Path, default=None, help="write the full JSON report here (never into the repo)")
    args = ap.parse_args(argv)
    cases = load_cases(args.cases)
    model = stub_model if args.offline else http_model(args.model_url, args.model)
    report = run(cases, model, mode="offline-stub" if args.offline else f"live:{args.model}", threshold=args.threshold)
    summary = report.summary()
    if args.live and args.cases == FIXTURES:
        summary["note"] = "fixture prompts through the local model: not live organ numbers (UNVERIFIED for organs)"
    print(json.dumps(summary, indent=1))
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "results": [r.__dict__ for r in report.results]},
                                       indent=1))
    if args.offline:
        # The offline lane checks the harness: 3 real cites change, the planted hollow one does not.
        ok = summary["cites"] == 4 and summary["hollow"] == [h for h in summary["hollow"] if "prime-hollow" in h] \
            and len(summary["hollow"]) == 1 and not summary["refused"]
        print("PASS" if ok else "FAIL", "offline harness check")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
