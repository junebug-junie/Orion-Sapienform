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


def _new_shape_tree(tmp_path: Path) -> Path:
    # A compose default: a set env var beats the pydantic default.
    _write(
        tmp_path,
        "services/orion-thing/docker-compose.yml",
        "services:\n  thing:\n    environment:\n"
        "      - THING_LLM_ROUTE=${THING_LLM_ROUTE:-chat}\n"
        "      - THING_LLM_LANE=${THING_LLM_LANE:-chat}\n"
        "      # - THING_OLD_ROUTE=${THING_OLD_ROUTE:-chat}\n"
        "      - THING_TIMEOUT=${THING_TIMEOUT:-5}\n",
    )
    # Dict literal and .get default, both quote styles.
    _write(
        tmp_path,
        "orion/thing/dispatch.py",
        "def build_payload(opts):\n"
        "    body = {'route': 'chat', 'messages': []}\n"
        "    other = {\"llm_route\": \"chat\"}\n"
        "    picked = opts.get('LLM_ROUTE', 'chat')\n"
        "    lane = opts.get('llm_lane', 'chat')\n"
        "    return body, other, picked, lane\n",
    )
    # Quoted env value and trailing comment.
    _write(
        tmp_path,
        "services/orion-thing/.env_example",
        'THING_LLM_ROUTE="chat"  # reserved lane\n'
        "THING_LLM_PROFILE='chat'\n"
        "THING_LLM_LANE=chat\n"
        "THING_ROUTE_NAME=chatter\n",
    )
    return tmp_path


def test_new_shapes_compose_dict_get_and_quoted_env(tmp_path: Path) -> None:
    root = _new_shape_tree(tmp_path)
    hits = gate.scan_tree(root)
    got = sorted((h.key, h.shape) for h in hits)
    assert got == [
        ("orion/thing/dispatch.py:build_payload", "dict_literal"),
        ("orion/thing/dispatch.py:build_payload", "dict_literal"),
        ("orion/thing/dispatch.py:build_payload", "get_default"),
        ("services/orion-thing/.env_example:THING_LLM_PROFILE", "env_default"),
        ("services/orion-thing/.env_example:THING_LLM_ROUTE", "env_default"),
        ("services/orion-thing/docker-compose.yml:THING_LLM_ROUTE", "compose_default"),
    ]
    # The lane-class axis (LLM_LANE, llm_lane) is deliberately not a hit.
    assert not any("LANE" in h.scope or "lane" in h.line.split("=")[0] for h in hits if h.shape != "dict_literal")


def test_compose_default_fails_unless_allow_listed(tmp_path: Path, capsys, monkeypatch) -> None:
    root = _new_shape_tree(tmp_path)
    monkeypatch.setattr(gate, "ALLOW", {"orion/thing/dispatch.py:*": "ok", "services/orion-thing/.env_example:*": "ok"})
    rc = gate.main(["--root", str(root)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "THING_LLM_ROUTE=${THING_LLM_ROUTE:-chat}" in out
    monkeypatch.setattr(
        gate,
        "ALLOW",
        {
            "orion/thing/dispatch.py:*": "ok",
            "services/orion-thing/.env_example:*": "ok",
            "services/orion-thing/docker-compose.yml:THING_LLM_ROUTE": "ok",
        },
    )
    assert gate.main(["--root", str(root)]) == 0


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
