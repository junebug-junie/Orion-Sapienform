from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from scripts.platform._common import find_repo_root, iter_files, read_text, relpath, write_json

RAW_REDIS_PATTERNS = [
    r"\bimport\s+redis\b",
    r"\bfrom\s+redis\b",
    r"redis\.asyncio",
    r"\bRedis\(",
    r"\.pubsub\(",
]

RAW_PUBSUB_CALLS = [
    r"\.publish\(",
    r"\.subscribe\(",
    r"\.psubscribe\(",
]


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/platform/audit_antipatterns.py <RUN_DIR>")
        return 2

    run_dir = Path(sys.argv[1]).resolve()
    repo_root = find_repo_root(run_dir)

    findings: List[Dict[str, Any]] = []

    for p in iter_files(repo_root, (".py",)):
        src = read_text(p)
        f = relpath(repo_root, p)

        # raw redis usage
        for pat in RAW_REDIS_PATTERNS:
            if re.search(pat, src):
                findings.append({"type": "raw_redis_import_or_client", "file": f, "detail": pat, "severity": "major"})
                break

        # raw pubsub calls (heuristic; may include OrionBus internals)
        for pat in RAW_PUBSUB_CALLS:
            if re.search(pat, src) and "OrionBus" not in src:
                findings.append({"type": "raw_pubsub_call_suspect", "file": f, "detail": pat, "severity": "minor"})
                break

        # non-envelope publish heuristic
        if re.search(r"publish\([^,]+,\s*json\.dumps\(", src):
            findings.append({"type": "publish_json_dumps", "file": f, "detail": "publish(..., json.dumps(...))", "severity": "major"})

    write_json(run_dir / "reports" / "antipatterns.json", {"findings": findings, "summary": {"count": len(findings)}})
    print(f"Wrote antipatterns to {run_dir / 'reports'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
