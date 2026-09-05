"""The live `.env` scan must actually fire -- it passes today only because
nothing is currently broken, and a gate that has never been shown to fail is
indistinguishable from one that cannot.

Context: `check_service_hostname_refs.py` has always read `.env_example` only.
`.env` is gitignored so CI cannot see it, but `.env` is what the container
reads. A hand-edit to `http://orion-notify:7140` therefore passes CI green,
deploys successfully, and silently breaks notify -- reported as roughly twenty
PRs of recurring repair. Verified live 2026-09-05 from inside a container:
`notify` and `orion-athena-notify` resolve, `orion-notify` resolves nowhere.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_MOD = Path(__file__).resolve().parents[1] / "check_service_hostname_refs.py"
_spec = importlib.util.spec_from_file_location("check_service_hostname_refs", _MOD)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


@pytest.fixture
def fake_services(tmp_path, monkeypatch):
    """A services/ tree with one service whose compose key differs from its dirname."""
    services = tmp_path / "services"
    (services / "orion-notify").mkdir(parents=True)
    (services / "orion-notify" / "docker-compose.yml").write_text(
        "services:\n  notify:\n    image: x\n", encoding="utf-8"
    )
    (services / "orion-caller").mkdir(parents=True)
    (services / "orion-caller" / "docker-compose.yml").write_text(
        "services:\n  caller:\n    image: x\n", encoding="utf-8"
    )
    monkeypatch.setattr(mod, "_SERVICES_DIR", services)
    monkeypatch.setattr(mod, "_REPO_ROOT", tmp_path)
    # The live .env resolver deliberately points at the PRIMARY CHECKOUT, not at
    # this module's location -- redirect it too, or these fixtures are ignored and
    # the tests silently exercise the real repo instead.
    monkeypatch.setattr(mod, "_live_env_root", lambda: tmp_path)
    return services


def _keys():
    return {"orion-notify": "notify", "orion-caller": "caller"}


def test_bad_hostname_in_live_env_is_caught_only_with_the_flag(fake_services):
    """The whole point: this exact edit is invisible to CI."""
    caller = fake_services / "orion-caller"
    (caller / ".env_example").write_text("NOTIFY_URL=http://notify:7140\n", encoding="utf-8")
    (caller / ".env").write_text("NOTIFY_URL=http://orion-notify:7140\n", encoding="utf-8")

    ci = mod._find_violations(_keys())
    assert ci == [], "the committed template is correct, so CI must stay green"

    deploy = mod._find_violations(_keys(), include_live_env=True)
    assert len(deploy) == 1
    assert deploy[0]["hostname_used"] == "orion-notify"
    assert deploy[0]["real_compose_service_key"] == "notify"
    assert deploy[0]["file"].endswith(".env")


def test_correct_live_env_passes(fake_services):
    caller = fake_services / "orion-caller"
    (caller / ".env_example").write_text("NOTIFY_URL=http://notify:7140\n", encoding="utf-8")
    (caller / ".env").write_text("NOTIFY_URL=http://notify:7140\n", encoding="utf-8")
    assert mod._find_violations(_keys(), include_live_env=True) == []


def test_container_name_form_is_not_flagged(fake_services):
    """`orion-athena-notify` RESOLVES (container_name is also a network alias), so
    flagging it would be a false positive on 19 live references. Fragile, not
    broken -- a separate portability question, deliberately out of scope."""
    caller = fake_services / "orion-caller"
    (caller / ".env_example").write_text("N=http://notify:7140\n", encoding="utf-8")
    (caller / ".env").write_text("N=http://orion-athena-notify:7140\n", encoding="utf-8")
    assert mod._find_violations(_keys(), include_live_env=True) == []


def test_commented_out_reference_is_ignored(fake_services):
    """All four surviving `orion-notify` strings in this repo are COMMENTS recording
    the 2026-07-28 fix. Flagging those would make the gate cry wolf forever."""
    caller = fake_services / "orion-caller"
    (caller / ".env_example").write_text("N=http://notify:7140\n", encoding="utf-8")
    (caller / ".env").write_text(
        "# 2026-07-28: was http://orion-notify:7140 -- never resolved\n"
        "N=http://notify:7140\n",
        encoding="utf-8",
    )
    assert mod._find_violations(_keys(), include_live_env=True) == []


def test_only_service_restricts_the_scan(fake_services):
    """Deploy-time use scans just the service being brought up -- it must not block
    a deploy on some unrelated service's file."""
    for name in ("orion-caller", "orion-notify"):
        d = fake_services / name
        (d / ".env_example").write_text("N=http://notify:7140\n", encoding="utf-8")
        (d / ".env").write_text("N=http://orion-notify:7140\n", encoding="utf-8")

    both = mod._find_violations(_keys(), include_live_env=True)
    assert len(both) == 2
    one = mod._find_violations(_keys(), include_live_env=True, only_service="orion-caller")
    assert len(one) == 1 and "orion-caller" in one[0]["file"]


def test_live_env_is_read_from_the_primary_checkout_not_this_worktree():
    """The bug that nearly shipped: a gate that scans nothing and reports OK.

    `_REPO_ROOT` is `Path(__file__).parents[1]`, which inside a linked worktree is
    the WORKTREE. `.env` is gitignored, so it exists only in the primary checkout
    -- resolving it relative to this file made `--include-live-env` find no files
    at all and exit 0. Caught 2026-09-05 by injecting a real bad hostname into a
    live .env and watching the gate pass clean.

    This asserts the resolution target, not merely that some path is returned:
    the previous version also returned a path, it was just the wrong one.
    """
    root = mod._live_env_root()
    assert (root / "services").is_dir(), f"{root} has no services/ dir"
    envs = list(root.glob("services/*/.env"))
    assert envs, (
        f"no live .env found under {root} -- the scan would be vacuous and would "
        "report OK no matter what any .env contained"
    )


def test_scan_reads_env_from_the_live_root_not_beside_the_template(tmp_path, monkeypatch):
    """Reproduces the worktree shape exactly, and pins the CALL SITE.

    The test above asserts `_live_env_root()` returns a sane path -- but a mutation
    that reverts the call site to `service_dir / ".env"` still passed it, because
    the helper itself was untouched. That is the vacuous-gate bug wearing a green
    suite, so this pins the behaviour instead of the helper:

      templates live in the worktree  (services/ has .env_example, NO .env)
      the live .env lives elsewhere   (the primary checkout)

    Scanning beside the template finds nothing and reports OK. Only reading from
    the live root finds the bad hostname.
    """
    worktree = tmp_path / "worktree"
    primary = tmp_path / "primary"
    for base in (worktree, primary):
        (base / "services" / "orion-caller").mkdir(parents=True)
    (worktree / "services" / "orion-caller" / "docker-compose.yml").write_text(
        "services:\n  caller:\n    image: x\n", encoding="utf-8"
    )
    # Template is correct and sits in the worktree. No .env here -- gitignored.
    (worktree / "services" / "orion-caller" / ".env_example").write_text(
        "NOTIFY_URL=http://notify:7140\n", encoding="utf-8"
    )
    # The file the container actually reads, only in the primary checkout, is wrong.
    (primary / "services" / "orion-caller" / ".env").write_text(
        "NOTIFY_URL=http://orion-notify:7140\n", encoding="utf-8"
    )

    monkeypatch.setattr(mod, "_SERVICES_DIR", worktree / "services")
    monkeypatch.setattr(mod, "_REPO_ROOT", worktree)
    monkeypatch.setattr(mod, "_live_env_root", lambda: primary)

    # orion-notify MUST be in the key map or the host is unrecognised and skipped --
    # without it this test fails for the wrong reason and reads as a caught mutation.
    found = mod._find_violations(
        {"orion-caller": "caller", "orion-notify": "notify"}, include_live_env=True
    )
    assert len(found) == 1, (
        "the live .env was not read from the primary checkout -- this is the "
        "vacuous-scan regression, which reports OK no matter what .env contains"
    )
    assert found[0]["hostname_used"] == "orion-notify"
