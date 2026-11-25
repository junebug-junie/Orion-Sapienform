#!/usr/bin/env python3
"""Gate: a committed eval or script that makes a model call goes through the lease helper or the gateway.

Why (#3921, 2026-11-25): the lane door in front of each circe model server counts calls that carry no
lease id. Two committed scripts sent such calls to the agent seat (:8015): an eval whose docstring said
"run under an operator lease" but took a bare ``--url`` and sent no header, and a judge eval that took
``name=base_url`` lanes. Neither named circe's address in code, so the port gate
(scripts/check_circe_worker_refs.py, address literals only) never saw them. Once the door enforces,
those calls get 401.

What this checks. Scope: every ``.py``/``.sh`` file under an ``evals/``, ``scripts/`` or ``bench/``
directory (operator tooling: the code a person or agent runs by hand), minus tests (``tests/``,
``test_*.py``), hidden directories and the port gate's zones (the pool, the gateway, the lane door, the
workers themselves). A file in scope "makes a model call" when it holds a model-call path literal
(``/v1/chat/completions``, ``/v1/completions``, ``/v1/messages`` or llama.cpp's quoted ``"/completion"``).
Each such file must be one of:

1. a LEASED caller: it imports ``orion.evals.leased_llm`` (Python) or runs ``-m orion.evals.leased_llm``
   (shell). The helper takes a GPU pool operator hold and calls only the gateway, under that hold.
2. a GATEWAY caller, named below with a reason: it posts to orion-llm-gateway by route (the gateway
   leases each call itself), or to a local stub, or makes no HTTP call at all.

Anything else fails: a new model-calling script must be classified by a reviewer. A file that names a
circe lane port (``:8011``, ``:8015`` ...; ports read from config/gpu_pool.yaml the same way the port
gate reads them) and makes a model call must be a leased caller; a GATEWAY_CALLERS entry cannot cover
it. An entry whose file is gone, no longer makes a model call, or is now leased is stale and fails too.

Usage:
    python3 scripts/check_leased_lane_callers.py           # gate (exit 1 on fail)
    python3 scripts/check_leased_lane_callers.py --json
    python3 scripts/check_leased_lane_callers.py --root DIR
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
from dataclasses import asdict, dataclass

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import check_circe_worker_refs as port_gate  # noqa: E402

RULE = (
    "a committed eval/script that calls a model must use orion/evals/leased_llm.py "
    "(operator_lane_sync / OperatorLanes / gateway_call, or `python -m orion.evals.leased_llm chat|hold`), "
    "or call orion-llm-gateway by route and be named in GATEWAY_CALLERS with a reason"
)

SCOPE_DIRS = frozenset({"evals", "scripts", "bench"})
SUFFIXES = (".py", ".sh")
_EXCLUDED_DIRS = {"tests", "docs", "graphify-out", "venv", ".venv", "node_modules", "__pycache__",
                  "site-packages"}

# The helper itself and this gate name the paths and the import they look for.
OWNERS = frozenset({"orion/evals/leased_llm.py", "scripts/check_leased_lane_callers.py"})

MODEL_CALL_RE = re.compile(r"""/v1/chat/completions|/v1/completions|/v1/messages|["']/completion["'?]""")
LEASED_RE = re.compile(r"""^\s*(?:from\s+orion\.evals\.leased_llm\s+import\b|import\s+orion\.evals\.leased_llm\b)"""
                       r"""|-m\s+orion\.evals\.leased_llm\b""", re.MULTILINE)

# Model-calling files that are NOT leased callers, each with the reason it needs no operator hold.
# Keys are repo-relative paths. Reviewed one by one 2026-11-25 (#3921).
GATEWAY_CALLERS: dict[str, str] = {
    "orion/integration/evals/run_now_ablation_eval.py": "--model-url defaults to the gateway's OpenAI port (127.0.0.1:8222), by route",
    "orion/thought/evals/stance_task_boundary.py": "posts to --gateway-url (orion-llm-gateway) by route",
    "scripts/analysis/eval_personality_naming_grounding.py": "posts to orion-llm-gateway (:8210) by route",
    "scripts/core_page_dry_run.py": "posts to orion-llm-gateway (--gateway, :8210) by route",
    "scripts/memory_supersede_eval.py": "posts to orion-llm-gateway (--gateway, :8210) by route",
    "scripts/memory_supersede_pass.py": "posts to orion-llm-gateway (--gateway, :8210) by route",
    "scripts/probe_fcc_spawn_timing.py": "points the FCC subprocess at a local stub server; no model is called",
    "scripts/probe_fcc_warm_pool.py": "points the FCC relay at a local stub server; no model is called",
    "services/orion-ai-town/scripts/smoke_llm_rail.sh": "curls orion-llm-gateway (:8210) by route",
    "services/orion-durable-runs/evals/run_dream_day_eval.py": "posts to orion-llm-gateway (--gateway, :8210) by route",
    "services/orion-fcc/evals/compaction_replay.py": "posts to llm-gateway:8210/v1/messages by route",
    "services/orion-hub/evals/run_reading_link_targets_replay.py": "--llm-url is the gateway (:8210) by route",
    "services/orion-hub/evals/run_reading_resonance_eval.py": "--judge-url is the gateway's OpenAI port (:8222) by route",
    "services/orion-vllm-host/evals/check_deep_protocol.py": "calls the deep-budget handler in process; no HTTP",
}


@dataclass(frozen=True)
class Finding:
    path: str
    problem: str   # "unclassified" | "names_lane_port"
    detail: str


def _in_scope(rel: pathlib.PurePosixPath) -> bool:
    if rel.suffix not in SUFFIXES or rel.name.startswith("test_"):
        return False
    if not any(part in SCOPE_DIRS for part in rel.parts[:-1]):
        return False
    return rel.as_posix() not in OWNERS and port_gate._zone_for(rel.as_posix()) is None


def iter_scope(root: pathlib.Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDED_DIRS and not d.startswith(".")]
        for name in filenames:
            path = pathlib.Path(dirpath) / name
            rel = pathlib.PurePosixPath(path.relative_to(root).as_posix())
            if _in_scope(rel):
                yield path, rel.as_posix()


def lane_port_re(ports: frozenset[int]) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w.]):({'|'.join(str(p) for p in sorted(ports))})(?!\d)")


def classify_text(rel: str, text: str, port_re: re.Pattern[str]) -> tuple[str, str]:
    """(kind, detail): kind is "none" (no model call), "leased", "gateway" (allow-listed) or a problem."""
    m = MODEL_CALL_RE.search(text)
    if m is None:
        return "none", ""
    if LEASED_RE.search(text):
        return "leased", ""
    port = port_re.search(text)
    if port is not None:
        line = text.count("\n", 0, port.start()) + 1
        return "names_lane_port", f"line {line} names lane port {port.group(0)} and the file sends {m.group(0)} unleased"
    if rel in GATEWAY_CALLERS:
        return "gateway", GATEWAY_CALLERS[rel]
    line = text.count("\n", 0, m.start()) + 1
    return "unclassified", f"line {line}: {m.group(0)}"


def run(root: pathlib.Path) -> dict:
    ident = port_gate.load_identity(root)
    port_re = lane_port_re(ident.ports)
    findings: list[Finding] = []
    kinds: dict[str, str] = {}
    for path, rel in iter_scope(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        kind, detail = classify_text(rel, text, port_re)
        kinds[rel] = kind
        if kind in ("unclassified", "names_lane_port"):
            findings.append(Finding(rel, kind, detail))
    stale = [rel for rel in GATEWAY_CALLERS if kinds.get(rel) != "gateway"
             and not any(f.path == rel for f in findings)]
    return {
        "ok": not findings and not stale,
        "ports": sorted(ident.ports),
        "findings": [asdict(f) for f in findings],
        "stale_gateway_callers": {rel: kinds.get(rel, "missing or out of scope") for rel in stale},
        "leased": sorted(r for r, k in kinds.items() if k == "leased"),
        "gateway": sorted(r for r, k in kinds.items() if k == "gateway"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=pathlib.Path, default=REPO)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    out = run(args.root.resolve())
    if args.json:
        print(json.dumps(out, indent=2))
        return 0 if out["ok"] else 1
    if out["findings"]:
        print("leased lane caller gate: FAIL")
        print(f"\n  {RULE}\n")
        for f in out["findings"]:
            print(f"    {f['path']} [{f['problem']}] {f['detail']}")
    if out["stale_gateway_callers"]:
        print("leased lane caller gate: STALE GATEWAY_CALLERS entries (remove them):")
        for rel, kind in out["stale_gateway_callers"].items():
            print(f"    {rel} (now: {kind})")
    if not out["ok"]:
        return 1
    print(f"leased lane caller gate: PASS ({len(out['leased'])} leased caller(s), "
          f"{len(out['gateway'])} gateway caller(s); lane ports {','.join(map(str, out['ports']))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
