"""Weekly ablation eval for Orion's now (integration patch 3, #3425): does a cite really shape the output?

Spec: docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md, Part A metric gate step 4 and the
patch 3 acceptance check: "removing a cited item from a replayed prompt changes the output in most cases; the
hollow-cite share is reported". Local model only; never an outside model.

For each case and each cited event id: replay the prompt as shown, and again with the cited item's
``[experience:<id>]`` line removed (``ablate``). Every merged renderer shows one item per line with that token: Prime's
frame (``orion.prime.thought._now_lines``), reverie's block (``temporal_self_cue.render_now_block``) and curiosity's
kickoff (``orion.curiosity.now_section.format_now_section``). Same model, temperature 0. The cite is **changed** only when the CONTENT of the two outputs
differs by at least ``--threshold`` (token-set Jaccard distance). Before the distance is taken, the cited id is
stripped from both outputs in every shape the organs cite in: Prime's ``{"kind": "experience", "id": "<id>"}`` in its
refs, reverie's ``cites`` entry, curiosity's ``[experience:<id>]``. The ablated prompt cannot show the id, so "the
full output names it and the ablated one does not" is true by construction; it is reported (``names_ref_*``) and
never counted as change. Otherwise the cite is **hollow**: the organ cited something its output did not depend on.

Never a fake number:

* An empty output (a model whose thinking used its whole budget returns no content) is **refused**, not scored: an
  empty pair reads distance 0, which would be hollow.
* ``--live`` replays the full prompt twice per cite and records that **noise** distance, because a local server at
  temperature 0 is not bit-deterministic. A cite counts as changed only at ``max(threshold, noise + NOISE_MARGIN)``.
  A stored ``original_output`` (made with other sampling) is not used as the baseline in that mode.

Lanes:

* ``--offline`` (CI, < 2 s): the checked-in fixtures (built with the merged renderers by
  ``now_ablation_fixtures.py``; a test fails when a renderer changes and they were not rebuilt) and a
  deterministic stub model that writes from the lines its case says it uses. One fixture is a planted hollow cite.
  This proves the harness logic, not Orion's behavior.
* ``--live``: replays ``--cases`` (default: the fixtures) through ``--model-url`` (an OpenAI-compatible local
  endpoint; default the repo's local gateway convention). Numbers from fixtures are fixture numbers, not live organ
  numbers: the organs do not keep the prompt they cited from yet (#3516), so a live organ-level number is UNVERIFIED.

Periodic and run by hand (weekly, per the spec): nothing schedules it yet. Running it over real cited runs is #3516.

Usage:

    python -m orion.integration.evals.run_now_ablation_eval --offline
    python -m orion.integration.evals.run_now_ablation_eval --live --model-url http://127.0.0.1:8210/v1/chat/completions \\
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

from orion.integration.evals.now_ablation_fixtures import FIXTURES

DEFAULT_MODEL_URL = "http://127.0.0.1:8210/v1/chat/completions"
DEFAULT_THRESHOLD = 0.25
NOISE_MARGIN = 0.10  # live: a cite must move the output this much past two replays of the same prompt

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


# Words; inner . : _ - kept (ids, numbers like 32.4), trailing punctuation dropped ("warm." == "warm").
_TOKEN = re.compile(r"[a-z0-9]+(?:[_.:\-][a-z0-9]+)*")
_ID_CHARS = r"A-Za-z0-9_.:+\-"


def _id_re(event_id: str) -> str:
    """The id as a whole token: ``a:1`` does not match inside ``a:10``. It may follow the ``experience:`` prefix (the
    colon is an id character, so the plain boundary alone would refuse ``experience:a:1``)."""
    return rf"(?:(?<=experience:)|(?<![{_ID_CHARS}])){re.escape(event_id)}(?![{_ID_CHARS}])"


def names(text: str, event_id: str) -> bool:
    return re.search(_id_re(event_id), text or "") is not None


def strip_cite(text: str, event_id: str) -> str:
    """``text`` without the cited id in any shape an organ cites in, so only content is compared."""
    out = re.sub(r'\{\s*"kind"\s*:\s*"experience"\s*,\s*"id"\s*:\s*"' + re.escape(event_id) + r'"\s*\}\s*,?',
                 " ", text or "")
    out = re.sub(r"\[?\s*experience:\s*" + _id_re(event_id) + r"\s*\]?", " ", out)
    return re.sub(_id_re(event_id), " ", out)


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
    threshold: float
    noise: Optional[float]
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
            "noise": [round(r.noise, 3) for r in self.results if r.noise is not None],
            "refused": self.refused,
        }


def run(cases: list[dict[str, Any]], model: Model, *, mode: str, threshold: float = DEFAULT_THRESHOLD,
        measure_noise: bool = False) -> Report:
    """Score every cite. ``measure_noise``: replay the full prompt twice and raise the bar by the replay noise."""
    report = Report(mode=mode)
    for case in cases:
        system, prompt = str(case.get("system") or ""), str(case["prompt"])
        stored: Optional[str] = None if measure_noise else case.get("original_output")
        for eid in case.get("cited_event_ids") or []:
            where = f"{case['case_id']}:{eid}"
            try:
                cut = ablate(prompt, eid)
            except AblationError as exc:
                report.refused.append(f"{where}: {exc}")
                continue
            full = stored if stored is not None else model(case, system, prompt)
            out = model(case, system, cut)
            again = model(case, system, prompt) if measure_noise else None
            if not (full or "").strip() or not (out or "").strip() or (again is not None and not again.strip()):
                report.refused.append(f"{where}: empty output (not scored: it would read as hollow)")
                continue
            d = distance(strip_cite(full, eid), strip_cite(out, eid))
            noise = distance(strip_cite(full, eid), strip_cite(again, eid)) if again is not None else None
            bar = max(threshold, noise + NOISE_MARGIN) if noise is not None else threshold
            report.results.append(CiteResult(
                case_id=case["case_id"], organ=str(case.get("organ") or "?"), event_id=eid, distance=d,
                threshold=bar, noise=noise, changed=d >= bar, names_ref_full=names(full, eid),
                names_ref_ablated=names(out, eid), full=full[:400], ablated=out[:400]))
    return report


def stub_model(case: dict[str, Any], _system: str, prompt: str) -> str:
    """Deterministic: writes a fixed opening plus the label of every line it 'uses' (``stub_uses``) that is still in
    the prompt. A cite the stub does not use cannot change its output: that is the planted hollow case."""
    parts = ["I read what my other organs lived."]
    for line in prompt.split("\n"):
        for eid in case.get("stub_uses") or []:
            if f"[experience:{eid}]" in line:
                parts.append(line.split("] ", 1)[-1])  # the label only: change must come from content
    return " ".join(parts)


def http_model(url: str, model: str, *, timeout: float = 120.0, max_tokens: int = 800) -> Model:
    def call(_case: dict[str, Any], system: str, prompt: str) -> str:
        # Thinking off: on these lanes a thinking model can spend max_tokens before writing any content.
        body = {"model": model, "temperature": 0, "seed": 7, "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
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
    report = run(cases, model, mode="offline-stub" if args.offline else f"live:{args.model}", threshold=args.threshold,
                 measure_noise=args.live)
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
