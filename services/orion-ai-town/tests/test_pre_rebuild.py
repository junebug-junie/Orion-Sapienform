"""Gate: scripts/pre_rebuild.sh deploys the town from scratch, every time.

The live town ran 2026-08-30 code until 2026-10-23: the post-pull rebuild
never applied the patches or pushed the Convex functions.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_SERVICE = Path(__file__).resolve().parents[1]


def _upstream_source() -> str | None:
    if os.environ.get("AITOWN_UPSTREAM_GIT"):
        return os.environ["AITOWN_UPSTREAM_GIT"]
    if (_SERVICE / "upstream" / ".git").exists():
        return str(_SERVICE / "upstream")
    return "https://github.com/a16z-infra/ai-town.git" if os.environ.get("CI") else None


@pytest.fixture
def town(tmp_path):
    src = _upstream_source()
    if not src:
        if os.environ.get("AITOWN_REQUIRE_TOOLS") == "1":
            pytest.fail("no upstream clone available")
        pytest.skip("no upstream clone available (set AITOWN_UPSTREAM_GIT)")
    svc = tmp_path / "orion-ai-town"
    for name in ("patches", "scripts", "cards"):
        shutil.copytree(_SERVICE / name, svc / name)
    shutil.copy2(_SERVICE / ".env_example", svc / ".env_example")
    shutil.copy2(svc / "cards" / "town_cards.yaml", svc / "cards" / "town_cards.local.yaml")
    subprocess.run(["git", "clone", "-q", src, str(svc / "upstream")], check=True)
    return svc


def _run(svc: Path, tmp_path: Path, **extra: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(
        AITOWN_CONVEX_PUSH=f"echo pushed >> {tmp_path / 'pushes'}",
        AITOWN_BACKUP_ROOT=str(tmp_path / "backups"),
        **extra,
    )
    return subprocess.run(
        ["bash", str(svc / "scripts" / "pre_rebuild.sh")], env=env, capture_output=True, text=True
    )


def test_deploys_and_redeploys_on_a_patched_tree(town, tmp_path):
    first = _run(town, tmp_path)
    assert first.returncode == 0, first.stdout + first.stderr
    conv = (town / "upstream" / "convex" / "agent" / "conversation.ts").read_text(encoding="utf-8")
    assert "function isPartnerEcho" in conv
    # Re-run on the now-patched tree: the bare apply script refuses this.
    (town / "upstream" / "convex" / "agent" / "conversation.ts").write_text(conv + "\n// hand edit\n")
    second = _run(town, tmp_path)
    assert second.returncode == 0, second.stdout + second.stderr
    assert (tmp_path / "pushes").read_text().count("pushed") == 2
    backups = sorted((tmp_path / "backups").glob("aitown-upstream-backup-*"))
    assert backups and "// hand edit" in (backups[-1] / "tracked.diff").read_text()


def test_refuses_without_local_cards(town, tmp_path):
    (town / "cards" / "town_cards.local.yaml").unlink()
    run = _run(town, tmp_path)
    assert run.returncode == 1
    assert "town_cards.local.yaml missing" in run.stderr
    assert not (tmp_path / "pushes").exists()
