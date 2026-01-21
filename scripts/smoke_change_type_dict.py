#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

scripts_dir = Path(__file__).resolve().parent
repo_root = scripts_dir.parent
if str(scripts_dir) in sys.path:
    sys.path.remove(str(scripts_dir))
sys.path.insert(0, str(repo_root))

from orion.schemas.collapse_mirror import CollapseMirrorEntryV2


def main() -> int:
    payload = {
        "observer": "orion",
        "trigger": "smoke",
        "type": "flow",
        "emergent_entity": "system",
        "summary": "Smoke test",
        "mantra": "steady",
        "change_type": {
            "change_type": "stabilizing",
            "stabilizing": 1.0,
            "deadband": 0.0,
            "biometrics": {"gpu": 0.9},
        },
    }

    try:
        entry = CollapseMirrorEntryV2.model_validate(payload)
    except Exception as exc:
        print(f"Validation failed: {exc}")
        return 1

    if entry.change_type != "stabilizing":
        print(f"Unexpected change_type: {entry.change_type}")
        return 1

    scores = entry.change_type_scores
    if scores.get("stabilizing") != 1.0 or scores.get("deadband") != 0.0:
        print(f"Unexpected change_type_scores: {scores}")
        return 1

    telemetry = entry.state_snapshot.telemetry
    if "change_type_meta" in telemetry:
        meta = telemetry["change_type_meta"]
        if not isinstance(meta, dict) or "biometrics" not in meta:
            print(f"Unexpected change_type_meta: {meta}")
            return 1
    elif "change_type_meta_keys" in telemetry:
        keys = telemetry["change_type_meta_keys"]
        if "biometrics" not in keys:
            print(f"Unexpected change_type_meta_keys: {keys}")
            return 1
    else:
        print(f"Missing change_type metadata in telemetry: {telemetry}")
        return 1

    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
