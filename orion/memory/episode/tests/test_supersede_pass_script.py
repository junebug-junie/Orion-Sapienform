"""scripts/memory_supersede_pass.py: the ids-only manifest Orion reads never carries statements."""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "memory_supersede_pass.py"
spec = importlib.util.spec_from_file_location("memory_supersede_pass", SCRIPT)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


class FakeConn:
    def __init__(self, rows):
        self.rows = rows

    async def execute(self, sql, params):
        rows = [r for r in self.rows if r["memory_id"] in params[0]]

        class Cur:
            async def fetchall(self_inner):
                return rows

        return Cur()


def test_manifest_has_ids_and_statuses_only(tmp_path):
    secret = "Juniper told me a private thing about Biscuit."
    proposed = [{"old_id": "o1", "new_id": "n1", "reason": "r", "old_statement": secret}]
    conn = FakeConn([{"memory_id": "o1", "status": "active", "confirmation_state": "auto",
                      "supersedes_memory_id": None, "superseded_by": None}])
    path = tmp_path / "m.json"
    asyncio.run(P.write_manifest(conn, path, mode="dry_run", proposed=proposed, applied=None,
                                 watch=P.parse_watch(["pet_row=o1", "gone=missing-id", "bad"])))
    text = path.read_text()
    assert secret not in text and "Biscuit" not in text
    m = json.loads(text)
    assert m["proposed"] == [{"old_id": "o1", "new_id": "n1"}] and m["applied"] == []
    assert m["watched"][0] == {"label": "pet_row", "memory_id": "o1", "status": "active",
                               "confirmation_state": "auto", "supersedes_memory_id": None,
                               "superseded_by": None, "proposed_superseded_by": "n1"}
    assert m["watched"][1]["status"] == "missing"
    assert "superseded|<old_id>|<new_id>" in m["pair_handled_key"]
    # An apply keeps the proposed list and records what it applied.
    asyncio.run(P.write_manifest(conn, path, mode="apply", proposed=proposed,
                                 applied=[{"old_id": "o1", "new_id": "n1", "applied_at": "t"}], watch=[]))
    assert json.loads(path.read_text())["applied"][0]["old_id"] == "o1"
