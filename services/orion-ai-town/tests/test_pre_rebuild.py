"""Gate: pre_rebuild.sh rebuilds upstream/ from the pin, every time;
post_rebuild.sh pushes the functions.

The live town ran 2026-08-30 code until 2026-10-23: the post-pull rebuild
never applied the patches or pushed the Convex functions.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_SERVICE = Path(__file__).resolve().parents[1]
_REF = re.search(
    r"^AITOWN_UPSTREAM_REF=([0-9a-f]{40})$", (_SERVICE / ".env_example").read_text(), re.M
).group(1)


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
    for name in ("patches", "scripts", "cards", "overlay"):
        shutil.copytree(_SERVICE / name, svc / name)
    shutil.copy2(_SERVICE / ".env_example", svc / ".env_example")
    shutil.copy2(svc / "cards" / "town_cards.yaml", svc / "cards" / "town_cards.local.yaml")
    subprocess.run(["git", "clone", "-q", src, str(svc / "upstream")], check=True)
    subprocess.run(["git", "-C", str(svc / "upstream"), "checkout", "-q", "--detach", _REF], check=True)
    return svc


def _run(svc: Path, tmp_path: Path, script: str = "pre_rebuild.sh") -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.pop("AITOWN_ALLOW_EXAMPLE_CARDS", None)
    env.update(
        AITOWN_BACKUP_ROOT=str(tmp_path / "backups"),
        # Records each convex call with its cwd, and whether the sim overlay
        # was already in place (codegen must see the patched tree).
        AITOWN_TEST_CONVEX=(
            f"echo \"$(pwd) sim=$(test -f convex/sim/tick.ts && echo yes || echo no) $*\" >> {tmp_path / 'convex_calls'}"
        ),
    )
    return subprocess.run(["bash", str(svc / "scripts" / script)], env=env, capture_output=True, text=True)


def _calls(tmp_path: Path) -> list[str]:
    path = tmp_path / "convex_calls"
    return path.read_text().splitlines() if path.exists() else []


def _tree(svc: Path) -> dict[str, str]:
    up = svc / "upstream"
    files = subprocess.run(
        ["git", "-C", str(up), "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return {f: (up / f).read_text(encoding="utf-8", errors="replace") for f in files if (up / f).is_file()}


def test_rebuilds_a_hand_edited_tree_to_the_fresh_chain(town, tmp_path):
    first = _run(town, tmp_path)
    assert first.returncode == 0, first.stdout + first.stderr
    fresh = _tree(town)
    assert "function isPartnerEcho" in fresh["convex/agent/conversation.ts"]
    # The bare apply script refuses a second run on a patched tree.
    bare = subprocess.run(["bash", str(town / "scripts" / "apply_upstream_patches.sh")], capture_output=True, text=True)
    assert bare.returncode != 0
    conv = town / "upstream" / "convex" / "agent" / "conversation.ts"
    conv.write_text(conv.read_text() + "\n// staged hand edit\n")
    subprocess.run(["git", "-C", str(town / "upstream"), "add", "convex/agent/conversation.ts"], check=True)
    hub = town / "upstream" / "src" / "lib" / "hubEmbed.ts"
    hub.write_text(hub.read_text() + "\n// untracked hand edit\n")
    # Overlay files are reset too, and a file no longer in overlay/ is removed.
    core = town / "upstream" / "convex" / "sim" / "core.ts"
    core.write_text(core.read_text() + "\n// overlay hand edit\n")
    (town / "upstream" / "convex" / "sim" / "dropped.ts").write_text("export const stale = 1;\n")
    second = _run(town, tmp_path)
    assert second.returncode == 0, second.stdout + second.stderr
    assert _tree(town) == fresh
    assert "convex/sim/core.ts" in fresh and "convex/sim/dropped.ts" not in fresh
    backup = sorted((tmp_path / "backups").glob("aitown-upstream-backup-*"))[-1]
    assert "// staged hand edit" in (backup / "tracked.diff").read_text()
    assert "// untracked hand edit" in (backup / "untracked" / "src" / "lib" / "hubEmbed.ts").read_text()
    # pre runs codegen on the patched tree each time and never pushes.
    up = town / "upstream"
    assert _calls(tmp_path) == [f"{up} sim=yes codegen --typecheck disable"] * 2


def test_refuses_without_local_cards(town, tmp_path):
    (town / "cards" / "town_cards.local.yaml").unlink()
    run = _run(town, tmp_path)
    assert run.returncode == 1
    assert "town_cards.local.yaml missing" in run.stderr


def test_refuses_when_upstream_head_is_not_the_pin(town, tmp_path):
    subprocess.run(
        ["git", "-C", str(town / "upstream"), "-c", "user.email=t@example.com", "-c", "user.name=T",
         "commit", "-q", "--allow-empty", "-m", "past the pin"],
        check=True,
    )
    run = _run(town, tmp_path)
    assert run.returncode == 1
    assert "is not the pin" in run.stderr


def test_post_rebuild_pushes_from_upstream(town, tmp_path):
    run = _run(town, tmp_path, "post_rebuild.sh")
    assert run.returncode == 0, run.stderr
    assert "NOT running convex dev --once" in run.stderr
    assert _calls(tmp_path) == [f"{town / 'upstream'} sim=no dev --once"]


def test_seed_render_failure_leaves_no_example_seed(town, tmp_path):
    local = town / "cards" / "town_cards.local.yaml"
    local.write_text(local.read_text().replace("      wage: 10\n", "", 1))
    run = _run(town, tmp_path)
    assert run.returncode == 1
    assert "missing wage" in run.stderr and "removed convex/sim/seed.ts" in run.stderr
    sim = town / "upstream" / "convex" / "sim"
    assert (sim / "core.ts").is_file()
    assert not (sim / "seed.ts").exists() and not (sim / "seed.ts.tmp").exists()
    assert _calls(tmp_path) == []  # no codegen on a failed chain


def test_codegen_failure_fails_pre_rebuild(town, tmp_path):
    env = os.environ.copy()
    env.update(AITOWN_BACKUP_ROOT=str(tmp_path / "backups"), AITOWN_TEST_CONVEX="exit 4")
    run = subprocess.run(["bash", str(town / "scripts" / "pre_rebuild.sh")], env=env, capture_output=True, text=True)
    assert run.returncode == 4
    assert "codegen done" not in run.stdout
