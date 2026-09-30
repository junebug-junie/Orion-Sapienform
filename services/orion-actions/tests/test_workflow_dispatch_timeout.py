"""The scheduled-workflow RPC wait must outlast a full compactor pass.

Before: _dispatch_scheduled_workflow waited ACTIONS_EXEC_TIMEOUT_SECONDS (420s)
while one github_compactor_pass could take fetch (300s) + a 660s digest call,
so a still-running pass was recorded failed and retried.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from orion.cognition.compactor.constants import COMPACTOR_DIGEST_TOTAL_BUDGET_SEC
from orion.cognition.github_compactor.constants import GITHUB_FETCH_ORCH_RPC_TIMEOUT_SEC

SERVICE = Path(__file__).resolve().parents[1]


def _default(alias: str) -> float:
    text = (SERVICE / "app" / "settings.py").read_text(encoding="utf-8")
    m = re.search(r"Field\(([\d.]+), alias=\"" + alias + r"\"\)", text)
    assert m, alias
    return float(m.group(1))


def _env_example(key: str) -> float:
    for line in (SERVICE / ".env_example").read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            return float(line.split("=", 1)[1])
    raise AssertionError(key)


def test_workflow_dispatch_timeout_exceeds_compactor_worst_case() -> None:
    worst = GITHUB_FETCH_ORCH_RPC_TIMEOUT_SEC + COMPACTOR_DIGEST_TOTAL_BUDGET_SEC
    assert _default("ACTIONS_WORKFLOW_DISPATCH_TIMEOUT_SECONDS") > worst
    assert _env_example("ACTIONS_WORKFLOW_DISPATCH_TIMEOUT_SECONDS") > worst


def test_scheduled_workflow_dispatch_uses_workflow_timeout() -> None:
    tree = ast.parse((SERVICE / "app" / "main.py").read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_dispatch_scheduled_workflow"
    )
    src = ast.unparse(fn)
    assert "scheduled_workflow_dispatch_timeout_sec(entry.workflow_id)" in src
    main_src = (SERVICE / "app" / "main.py").read_text(encoding="utf-8")
    # Only compactor passes get the long wait; claims are not reaped mid-dispatch.
    assert 'LONG_RUNNING_SCHEDULED_WORKFLOWS = frozenset({"github_compactor_pass", "chat_history_compactor_pass"})' in main_src
    assert "claim_ttl_seconds=int(max(300.0, settings.actions_workflow_dispatch_timeout_seconds + 60.0))" in main_src
