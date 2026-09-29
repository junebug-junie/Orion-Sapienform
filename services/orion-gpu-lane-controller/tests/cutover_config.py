"""The stage-5.3 rollback shape of config/gpu_pool.yaml, derived from the committed file.

5.3 cut agent-gpu2 over from the stage-4 bridge verbs to the generic launch executor
(docs/runbooks/2026-09-29-gpu-pool-stage5-3-cutover.md). Its rollback is the reverse edit: put
``load: gpu2/agent, unload: gpu2/restore`` back and drop ``launch.profiles`` (a bridged seat cannot
take a profile). The bridge stays in the controller until 5.6, so the bridge tests run against this
rollback shape -- which is exactly what proves the rollback still works.
"""
from __future__ import annotations

import copy

import yaml

from test_api import REPO_ROOT


def live_config() -> dict:
    return yaml.safe_load((REPO_ROOT / "config" / "gpu_pool.yaml").read_text())


def rollback_config(data: dict | None = None) -> dict:
    data = copy.deepcopy(live_config() if data is None else data)
    seat = data["roles"]["agent-gpu2"]
    seat["swap"]["load"], seat["swap"]["unload"] = "gpu2/agent", "gpu2/restore"
    seat["launch"].pop("profiles", None)
    return data


def write_rollback_config(path) -> None:
    path.write_text(yaml.safe_dump(rollback_config(), sort_keys=False))
