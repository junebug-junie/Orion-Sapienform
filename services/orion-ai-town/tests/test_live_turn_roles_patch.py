"""Gate: live-turn speaker roles + partner-echo check (c:578946, 2026-09-17).

The behavior test runs the patch's own TypeScript helpers under Node >= 22.6
(type stripping). The chain test applies every patch to the pinned upstream.
Both skip when their tool is missing; CI runs them in aitown-patch-chain.yml.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_SERVICE = Path(__file__).resolve().parents[1]
_PATCH = _SERVICE / "patches" / "orion-live-turn-roles.patch"
_APPLY = _SERVICE / "scripts" / "apply_upstream_patches.sh"
_FIXTURE = _SERVICE / "tests" / "fixtures" / "aitown_echo_c578946.json"
_UPSTREAM_URL = "https://github.com/a16z-infra/ai-town.git"


def _apply_order() -> list[str]:
    return [
        line.strip().strip('",')
        for line in _APPLY.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith('"orion-')
    ]


def _added(patch: str) -> str:
    return "\n".join(
        line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")
    )


def test_live_turn_roles_applies_after_self_repeat():
    order = _apply_order()
    assert order.index("orion-npc-no-self-repeat.patch") < order.index("orion-live-turn-roles.patch")


def test_history_uses_speaker_roles_not_name_prefix():
    patch = _PATCH.read_text(encoding="utf-8")
    added = _added(patch)
    assert "role: message.author === player.id ? 'assistant' : 'user'" in added
    assert "content: message.text," in added
    removed = [line for line in patch.splitlines() if line.startswith("-")]
    assert any("content: `${author.name} to ${recipient.name}: ${message.text}`" in l for l in removed)
    # The "Name to Name:" user cue is gone from continue, leave and the retry.
    assert any("llmMessages.push({ role: 'user', content: lastPrompt });" in l for l in removed)
    assert "{ role: 'user', content: lastPrompt }" not in added
    assert added.count("llmMessages.push(...replyCue(priorMessages, otherPlayer.name));") == 2


def test_echo_joins_self_repeat_gate():
    added = _added(_PATCH.read_text(encoding="utf-8"))
    assert "isSelfRepeat(line, ownLines) || isPartnerEcho(line, partnerLine)" in added
    assert "if (repeats(response)) {" in added
    # Retry nudge rides in the system turn, not as a partner (user) turn.
    assert "...llmMessages.slice(1)," in added
    assert "partnerLine.trim().endsWith('?')" in added
    assert "repeats(response) || isSelfRepeat(response, [firstAttempt])" in added


def _skip(reason: str) -> None:
    # CI sets AITOWN_REQUIRE_TOOLS=1 so a missing tool fails instead of going green.
    if os.environ.get("AITOWN_REQUIRE_TOOLS") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def _node_with_strip_types() -> str | None:
    node = os.environ.get("AITOWN_TEST_NODE") or shutil.which("node")
    if not node:
        return None
    out = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip()
    m = re.match(r"v(\d+)\.(\d+)", out)
    if not m or (int(m[1]), int(m[2])) < (22, 6):
        return None
    return node


def _helper_source() -> str:
    """normalizeTownLine (self-repeat patch) + echo helpers (this patch), as shipped."""
    self_repeat = _added((_SERVICE / "patches" / "orion-npc-no-self-repeat.patch").read_text(encoding="utf-8"))
    roles = _added(_PATCH.read_text(encoding="utf-8"))
    normalize = re.search(r"function normalizeTownLine\(.*?\n}\n", self_repeat, re.S)
    echo = re.search(r"const PARTNER_ECHO_MIN_OVERLAP.*?\nfunction isPartnerEcho\(.*?\n}\n", roles, re.S)
    assert normalize and echo, "helpers not found in patches"
    return normalize.group(0) + echo.group(0)


def test_partner_echo_flags_real_transcript(tmp_path):
    node = _node_with_strip_types()
    if not node:
        _skip("needs node >= 22.6 (set AITOWN_TEST_NODE)")
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    turns = fixture["turns"] + fixture["boundary"]
    script = tmp_path / "echo.ts"
    script.write_text(
        _helper_source()
        + "\nconst turns = JSON.parse(process.argv[2]);\n"
        + "console.log(JSON.stringify(turns.map((t: any) => isPartnerEcho(t.reply, t.partner_line))));\n",
        encoding="utf-8",
    )
    out = subprocess.run(
        [node, "--experimental-strip-types", "--no-warnings", str(script), json.dumps(turns)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    got = json.loads(out)
    assert got == [t["echo"] for t in turns]
    # The five word-for-word copies in c:578946; boundary rows pin 0.7 and the
    # question exemption (0.99 or 0.6 would fail them).
    assert sum(got[: len(fixture["turns"])]) == 5


def _upstream_source() -> str | None:
    if os.environ.get("AITOWN_UPSTREAM_GIT"):
        return os.environ["AITOWN_UPSTREAM_GIT"]
    local = _SERVICE / "upstream"
    if (local / ".git").exists():
        return str(local)
    return _UPSTREAM_URL if os.environ.get("CI") else None


def test_patch_chain_applies_to_pinned_upstream(tmp_path):
    src = _upstream_source()
    if not src or not shutil.which("git"):
        _skip("no upstream clone available (set AITOWN_UPSTREAM_GIT)")
    readme = (_SERVICE / "README.md").read_text(encoding="utf-8")
    ref = re.search(r"AITOWN_UPSTREAM_REF=([0-9a-f]{40})", readme).group(1)
    work = tmp_path / "town"
    work.mkdir()
    subprocess.run(["git", "clone", "-q", src, str(work / "upstream")], check=True)
    subprocess.run(["git", "-C", str(work / "upstream"), "checkout", "-q", ref], check=True)
    for name in ("patches", "scripts", "cards"):
        shutil.copytree(_SERVICE / name, work / name)
    run = subprocess.run(
        ["bash", str(work / "scripts" / "apply_upstream_patches.sh")], capture_output=True, text=True
    )
    assert run.returncode == 0, run.stdout + run.stderr
    for name in _apply_order():
        assert f"Applied {name}" in run.stdout, name
