"""Stage 0A: a failed Graphiti write is visible, not just a warning log line.

Live before this fix: the Hub (network_mode: host) could not resolve the
adapter's container name, every approval logged
`graphiti_sync_failed ... Temporary failure in name resolution`, and the
approve response looked like a success with no Graphiti episode.

Covers the three layers: the adapter result carries the error, the projector
puts it in `errors` (which the approve route already returns as
`projection.errors`), and the Hub UI turns it into a red status line. Plus the
config fix itself: the Hub's Graphiti URL must be reachable from host
networking.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pytest
import yaml

from orion.memory.crystallization.projection_graphiti import GraphitiAdapter
from orion.memory.crystallization.projector import ProjectionConfig, project_crystallization
from orion.memory.crystallization.proposer import propose
from orion.memory.crystallization.schemas import (
    CrystallizationEvidenceRefV1,
    MemoryCrystallizationProposeRequestV1,
)

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HUB_ROOT.parents[1]


def _active():
    row = propose(
        MemoryCrystallizationProposeRequestV1(
            kind="semantic",
            subject="Austin",
            summary="Headed to Austin and will fly back on Wednesday.",
            scope=["project:orion"],
            evidence=[CrystallizationEvidenceRefV1(source_kind="chat_turn", source_id="corr-a")],
            proposed_by="test",
        )
    )
    row.status = "active"
    return row


def _fail_dns(monkeypatch):
    class _Client:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, *a, **kw):
            raise httpx.ConnectError("[Errno -3] Temporary failure in name resolution")

    monkeypatch.setattr("orion.memory.crystallization.projection_graphiti.httpx.Client", _Client)


def test_adapter_result_carries_the_failure(monkeypatch):
    _fail_dns(monkeypatch)
    result = GraphitiAdapter(enabled=True, url="http://orion-athena-graphiti-adapter:8000").sync_crystallization(_active())
    assert result.episode_ids == []
    assert result.error and "name resolution" in result.error


@pytest.mark.asyncio
async def test_projector_reports_graphiti_failure_in_errors(monkeypatch):
    _fail_dns(monkeypatch)
    cfg = ProjectionConfig(graphiti_enabled=True, graphiti_url="http://orion-athena-graphiti-adapter:8000")
    _row, result = await project_crystallization(
        None, None, _active(), actor="test", config=cfg, project_card=False, project_chroma=False
    )
    assert any(e.startswith("graphiti_sync_failed:") and "name resolution" in e for e in result.errors)
    assert "name resolution" in result.graphiti["error"]
    assert result.graphiti["episode_ids"] == []


@pytest.mark.asyncio
async def test_projector_reports_missing_url_instead_of_silently_skipping():
    cfg = ProjectionConfig(graphiti_enabled=True, graphiti_url="")
    _row, result = await project_crystallization(
        None, None, _active(), actor="test", config=cfg, project_card=False, project_chroma=False
    )
    assert "graphiti_sync_failed:graphiti_adapter_url_missing" in result.errors


def _env_example_value(key: str) -> str:
    for line in (HUB_ROOT / ".env_example").read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    raise AssertionError(f"{key} missing from orion-hub/.env_example")


def test_hub_graphiti_url_resolves_from_host_network_mode():
    compose = yaml.safe_load((HUB_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    hub = next(iter(compose["services"].values()))
    assert hub.get("network_mode") == "host", "re-check this test if the Hub leaves host networking"

    url = urlparse(_env_example_value("GRAPHITI_ADAPTER_URL"))
    # Host networking has no Docker DNS: the host must be loopback or an IP,
    # never a container/service name.
    assert url.hostname in {"127.0.0.1", "localhost"} or re.fullmatch(r"[\d.]+", url.hostname or ""), url

    adapter_compose = (REPO_ROOT / "services" / "orion-graphiti-adapter" / "docker-compose.yml").read_text(
        encoding="utf-8"
    )
    published = re.search(r'"\$\{PORT:-(\d+)\}:8000"', adapter_compose)
    assert published, "adapter compose no longer publishes ${PORT:-NNNN}:8000"
    assert url.port == int(published.group(1))


_NODE_PROBE = r"""
const fs = require("fs"), vm = require("vm");
const src = fs.readFileSync(process.argv[1], "utf8");
const ctx = { window: { location: { pathname: "/", origin: "http://hub" } }, console };
vm.createContext(ctx);
vm.runInContext(src, ctx);
const f = ctx.window.OrionMemoryCrystallizationUI.graphitiFailureNote;
console.log(JSON.stringify({
  approveFail: f({ projection: { errors: ["graphiti_sync_failed:ConnectError: Temporary failure in name resolution"] } }),
  syncFail: f({ errors: ["graphiti_projection_failed:boom"] }),
  ok: f({ projection: { errors: [], graphiti: { episode_ids: ["e1"] } } }),
  otherError: f({ projection: { errors: ["chroma_projection_failed:x"] } }),
  nothing: f(null),
}));
"""


def test_hub_ui_turns_graphiti_failure_into_a_status_message():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not on PATH -- graphitiFailureNote was NOT executed")
    ui = HUB_ROOT / "static" / "js" / "memory-crystallization-ui.js"
    out = subprocess.run(
        [node, "-e", _NODE_PROBE, str(ui)], capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, out.stderr
    notes = json.loads(out.stdout.strip().splitlines()[-1])
    assert "Graphiti write failed" in notes["approveFail"] and "name resolution" in notes["approveFail"]
    assert "Graphiti write failed" in notes["syncFail"]
    assert notes["ok"] == "" and notes["otherError"] == "" and notes["nothing"] == ""
    # Shown after the inbox reload rewrites the status line, as an error.
    src = ui.read_text(encoding="utf-8")
    assert "if (failureNote) setStatus(statusEl" in src
