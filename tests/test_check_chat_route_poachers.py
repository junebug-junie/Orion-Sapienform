"""Tests for scripts/check_chat_route_poachers.py.

The gate keeps non-Hub-chat code off the reserved `chat` LLM route. These tests
build a tiny fake tree with one allowed hit and one disallowed hit and check
the classifier's three outputs: unallowed, allowed, stale-allow-keys.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import check_chat_route_poachers as gate  # noqa: E402


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fake_tree(tmp_path: Path) -> Path:
    # Allowed: a vision caller that genuinely needs the multimodal worker.
    _write(
        tmp_path,
        "services/orion-vision/app/settings.py",
        'FOVEAL_LLM_ROUTE: str = "chat"\n',
    )
    # Disallowed: a background classifier defaulting onto chat.
    _write(
        tmp_path,
        "services/orion-thing/app/worker.py",
        "async def classify(bus):\n"
        "    payload = ChatRequestPayload(\n"
        '        route="chat",\n'
        "        messages=[],\n"
        "    )\n",
    )
    # Not a route: a turn-source label. Must NOT be flagged.
    _write(
        tmp_path,
        "orion/hub/activity.py",
        'def turn_requested(source):\n    return str(source or "chat")\n',
    )
    # Multi-line def header: the hit must be attributed to the def, not <module>.
    _write(
        tmp_path,
        "orion/compactor/run.py",
        "async def run_digest(\n"
        "    *,\n"
        "    req,\n"
        ") -> str:\n"
        '    for route in ("chat", "quick"):\n'
        "        pass\n",
    )
    # Tests and a real .env are never scanned.
    _write(tmp_path, "services/orion-thing/tests/test_x.py", 'route="chat"\n')
    _write(tmp_path, "services/orion-thing/.env", "THING_LLM_ROUTE=chat\n")
    _write(tmp_path, "services/orion-thing/.env_example", "THING_LLM_ROUTE=chat\nTHING_OTHER=chat\n")
    return tmp_path


def test_classify_splits_allowed_disallowed_and_stale(tmp_path: Path) -> None:
    root = _fake_tree(tmp_path)
    hits = gate.scan_tree(root)
    keys = sorted(h.key for h in hits)
    assert keys == [
        "orion/compactor/run.py:run_digest",
        "services/orion-thing/.env_example:THING_LLM_ROUTE",
        "services/orion-thing/app/worker.py:classify",
        "services/orion-vision/app/settings.py:<module>",
    ]

    allow = {
        "services/orion-vision/**:*": "multimodal worker requirement",
        "orion/compactor/run.py:run_digest": "long digest; needs durable admission",
        "services/orion-gone/app/old.py:*": "this file was deleted; entry is stale",
    }
    unallowed, allowed, stale = gate.classify(hits, allow)
    assert sorted(h.key for h in unallowed) == [
        "services/orion-thing/.env_example:THING_LLM_ROUTE",
        "services/orion-thing/app/worker.py:classify",
    ]
    assert sorted(h.key for h, _ in allowed) == [
        "orion/compactor/run.py:run_digest",
        "services/orion-vision/app/settings.py:<module>",
    ]
    assert stale == ["services/orion-gone/app/old.py:*"]


def test_main_fails_on_disallowed_hit_and_quotes_the_line(tmp_path: Path, capsys, monkeypatch) -> None:
    root = _fake_tree(tmp_path)
    monkeypatch.setattr(gate, "ALLOW", {"services/orion-vision/**:*": "ok", "orion/compactor/run.py:run_digest": "ok"})
    rc = gate.main(["--root", str(root)])
    out = capsys.readouterr().out
    assert rc == 1
    assert 'route="chat"' in out
    assert "chat is Juniper's reserved Hub lane" in out


def test_main_fails_on_stale_allow_entry(tmp_path: Path, capsys, monkeypatch) -> None:
    root = _fake_tree(tmp_path)
    monkeypatch.setattr(
        gate,
        "ALLOW",
        {
            "services/orion-vision/**:*": "ok",
            "orion/compactor/run.py:run_digest": "ok",
            "services/orion-thing/**:*": "ok",
            "services/orion-gone/app/old.py:*": "stale",
        },
    )
    rc = gate.main(["--root", str(root)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "STALE" in out
    assert "services/orion-gone/app/old.py:*" in out


def test_main_passes_when_every_hit_is_allowed(tmp_path: Path, monkeypatch) -> None:
    root = _fake_tree(tmp_path)
    monkeypatch.setattr(
        gate,
        "ALLOW",
        {
            "services/orion-vision/**:*": "ok",
            "orion/compactor/run.py:run_digest": "ok",
            "services/orion-thing/**:*": "ok",
        },
    )
    assert gate.main(["--root", str(root)]) == 0


def test_comment_lines_and_trailing_comments_are_ignored(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "orion/x.py",
        '# route="chat" in a comment\n'
        'route = pick()  # never or "chat" here\n',
    )
    assert gate.scan_tree(tmp_path) == []


def test_main_passes_against_the_real_tree() -> None:
    """The shipped allow-list must match the shipped tree exactly: no poacher, no stale entry."""
    assert gate.main([]) == 0
