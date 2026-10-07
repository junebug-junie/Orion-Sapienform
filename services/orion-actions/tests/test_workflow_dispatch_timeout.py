"""The scheduled-workflow RPC wait covers the synchronous part of a compactor dispatch only.

History: the wait was 420s (too short: fetch + a 660s digest call), then 3600s while the whole
map-reduce digest ran inside the RPC (PR #2422), blocking the serial scheduler for up to an hour.
Now the digest runs as a ``compactor.digest`` durable run and orch replies ``accepted`` once it is
registered, so the wait only needs the fetch plus the durable receipts.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from orion.cognition.compactor.constants import COMPACTOR_MAX_RUN_GENERATIONS
from orion.cognition.github_compactor.constants import GITHUB_FETCH_ORCH_RPC_TIMEOUT_SEC

SERVICE = Path(__file__).resolve().parents[1]
RECEIPT_TIMEOUT_SEC = 10.0   # CORTEX_DURABLE_RECEIPT_TIMEOUT_SEC default (cortex-orch)


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


def test_workflow_dispatch_timeout_covers_fetch_and_receipts_but_not_the_digest() -> None:
    worst_sync = GITHUB_FETCH_ORCH_RPC_TIMEOUT_SEC + COMPACTOR_MAX_RUN_GENERATIONS * RECEIPT_TIMEOUT_SEC
    for value in (_default("ACTIONS_WORKFLOW_DISPATCH_TIMEOUT_SECONDS"),
                  _env_example("ACTIONS_WORKFLOW_DISPATCH_TIMEOUT_SECONDS")):
        assert value > worst_sync
        # Not sized to the digest any more: it must not block the serial scheduler for an hour.
        assert value <= 900


def test_scheduled_workflow_dispatch_uses_workflow_timeout() -> None:
    tree = ast.parse((SERVICE / "app" / "main.py").read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_dispatch_scheduled_workflow"
    )
    src = ast.unparse(fn)
    assert "scheduled_workflow_dispatch_timeout_sec(entry.workflow_id)" in src
    main_src = (SERVICE / "app" / "main.py").read_text(encoding="utf-8")
    assert 'LONG_RUNNING_SCHEDULED_WORKFLOWS = frozenset({"github_compactor_pass", "chat_history_compactor_pass"})' in main_src
    assert "claim_ttl_seconds=int(max(300.0, settings.actions_workflow_dispatch_timeout_seconds + 60.0))" in main_src
