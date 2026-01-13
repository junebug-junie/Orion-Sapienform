from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any, Dict, List

from scripts.platform._common import (
    find_repo_root,
    iter_files,
    read_text,
    relpath,
    write_json,
)


def _extract_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def build_pydantic_class_index(repo_root: Path) -> Dict[str, List[str]]:
    """Index BaseModel subclasses by class name -> [path:line]."""
    idx: Dict[str, List[str]] = {}
    for p in iter_files(repo_root, (".py",)):
        src = read_text(p)
        try:
            t = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if isinstance(node, ast.ClassDef):
                bases = []
                for b in node.bases:
                    if isinstance(b, ast.Name):
                        bases.append(b.id)
                    elif isinstance(b, ast.Attribute):
                        bases.append(b.attr)
                if "BaseModel" in bases:
                    loc = f"{relpath(repo_root, p)}:{getattr(node, 'lineno', 0)}"
                    idx.setdefault(node.name, []).append(loc)
    return idx


def scan_schema_sites(repo_root: Path) -> Dict[str, Any]:
    envelope_sites: List[Dict[str, Any]] = []
    schema_id_sites: List[Dict[str, Any]] = []
    non_schema_envelopes: List[Dict[str, Any]] = []

    for p in iter_files(repo_root, (".py",)):
        src = read_text(p)
        hits_env = []
        hits_sid = []
        try:
            t = ast.parse(src)
        except SyntaxError:
            continue

        for node in ast.walk(t):
            if isinstance(node, ast.Call):
                # detect BaseEnvelope(...) constructor-like call
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name in {"BaseEnvelope", "Envelope"}:
                    hits_env.append({"line": getattr(node, "lineno", 0), "text": "BaseEnvelope("})
                    # check schema_id kw
                    sid = None
                    for kw in node.keywords or []:
                        if kw.arg == "schema_id":
                            sid = _extract_str(kw.value)
                            if sid is not None:
                                hits_sid.append({"line": getattr(node, "lineno", 0), "text": f"schema_id=\"{sid}\""})
                            else:
                                hits_sid.append({"line": getattr(node, "lineno", 0), "text": "schema_id=<dynamic>"})
                    if sid is None:
                        non_schema_envelopes.append({"file": relpath(repo_root, p), "line": getattr(node, "lineno", 0), "note": "BaseEnvelope without schema_id"})

        if hits_env:
            envelope_sites.append({"file": relpath(repo_root, p), "hits": hits_env})
        if hits_sid:
            schema_id_sites.append({"file": relpath(repo_root, p), "hits": hits_sid})

    return {
        "envelope_sites": envelope_sites,
        "schema_id_sites": schema_id_sites,
        "non_schema_envelopes": non_schema_envelopes,
        "summary": {
            "envelope_files": len(envelope_sites),
            "schema_id_files": len(schema_id_sites),
            "non_schema_envelope_sites": len(non_schema_envelopes),
        },
        "notes": [
            "Automated scan is static and conservative.",
            "Model resolution is best-effort by class-name match unless a registry exists.",
        ],
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/platform/audit_schemas.py <RUN_DIR>")
        return 2

    run_dir = Path(sys.argv[1]).resolve()
    repo_root = find_repo_root(run_dir)

    class_idx = build_pydantic_class_index(repo_root)
    scan = scan_schema_sites(repo_root)

    # Extract schema_id values
    schema_ids: Dict[str, Dict[str, Any]] = {}
    for item in scan["schema_id_sites"]:
        f = item["file"]
        for h in item["hits"]:
            txt = h["text"]
            if txt.startswith("schema_id=\"") and txt.endswith("\""):
                sid = txt[len("schema_id=\"") : -1]
            else:
                sid = "<dynamic>"
            e = schema_ids.setdefault(sid, {"schema_id": sid, "files": set(), "model_resolves": False, "model_locations": [], "notes": []})
            e["files"].add(f"{f}:{h['line']}")

    inventory = []
    unresolved = []
    for sid, e in sorted(schema_ids.items()):
        if sid != "<dynamic>" and sid in class_idx:
            e["model_resolves"] = True
            e["model_locations"] = class_idx[sid]
        elif sid != "<dynamic>":
            unresolved.append(sid)
        inventory.append(
            {
                "schema_id": sid,
                "files": sorted(list(e["files"])),
                "model_resolves": bool(e["model_resolves"]),
                "model_locations": e["model_locations"],
                "notes": e["notes"],
            }
        )

    drift = {
        "unresolved_schema_ids": unresolved,
        "missing_schema_id_sites": scan["non_schema_envelopes"],
        "summary": scan["summary"],
        "notes": scan["notes"],
    }

    write_json(run_dir / "reports" / "schema_inventory.json", inventory)
    write_json(run_dir / "reports" / "schema_drift.json", drift)

    print(f"Wrote schema artifacts to {run_dir / 'reports'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
