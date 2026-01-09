from __future__ import annotations

import ast
import sys
from pathlib import Path

from scripts.platform._common import find_repo_root, iter_files, read_text, relpath, service_guess_from_path, write_json


def _extract_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/platform/audit_spine.py <RUN_DIR>")
        return 2

    run_dir = Path(sys.argv[1]).resolve()
    repo_root = find_repo_root(run_dir)

    verb_runtime_usages = []
    verb_request_emitters = []
    verb_request_consumers = []
    verb_result_emitters = []
    verb_result_consumers = []
    spine_violations = []

    for p in iter_files(repo_root, (".py",)):
        src = read_text(p)
        # VerbRuntime usage
        if "VerbRuntime" in src:
            try:
                t = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(t):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    txt = ast.get_source_segment(src, node) or ""
                    if "VerbRuntime" in txt:
                        verb_runtime_usages.append({"file": relpath(repo_root, p), "line": getattr(node, "lineno", 0), "context": txt.strip()})
                if isinstance(node, ast.Name) and node.id == "VerbRuntime":
                    verb_runtime_usages.append({"file": relpath(repo_root, p), "line": getattr(node, "lineno", 0), "context": "usage"})

        # channel publish/subscribe sites
        try:
            t2 = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(t2):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                method = node.func.attr
                if method in {"publish", "subscribe", "psubscribe"} and node.args:
                    ch = _extract_str(node.args[0])
                    if ch is None:
                        continue
                    rec = {"file": relpath(repo_root, p), "line": getattr(node, "lineno", 0), "service_guess": service_guess_from_path(p)}
                    if ch == "orion:verb:request":
                        (verb_request_emitters if method == "publish" else verb_request_consumers).append(rec)
                    if ch == "orion:verb:result":
                        (verb_result_emitters if method == "publish" else verb_result_consumers).append(rec)

    # Spine checks
    for u in verb_runtime_usages:
        f = u["file"]
        if "services/orion-cortex-exec" not in f and "orion/core/verbs" not in f:
            spine_violations.append({"type": "VerbRuntimeOutsideExec", **u, "rationale": "VerbRuntime should be used only by cortex-exec"})

    for e in verb_request_emitters:
        if "services/orion-cortex-orch" not in e["file"] and "/tests/" not in e["file"]:
            spine_violations.append({"type": "VerbRequestEmitterOutsideOrch", **e, "rationale": "Only cortex-orch should emit orion:verb:request in planned flows"})

    audit = {
        "verb_runtime_usages": verb_runtime_usages,
        "verb_request_emitters": verb_request_emitters,
        "verb_request_consumers": verb_request_consumers,
        "verb_result_emitters": verb_result_emitters,
        "verb_result_consumers": verb_result_consumers,
        "spine_violations": spine_violations,
        "summary": {
            "verb_runtime_usages": len(verb_runtime_usages),
            "verb_request_emitters": len(verb_request_emitters),
            "spine_violations": len(spine_violations),
        },
    }

    write_json(run_dir / "reports" / "spine_audit.json", audit)
    print(f"Wrote spine audit to {run_dir / 'reports'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
