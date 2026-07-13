"""Regression coverage for the chat-lane duplicate-exec fix.

Bug: both the ``legacy`` and ``chat`` orion-cortex-exec containers used to
register a ``Hunter`` listener on the shared ``orion:verb:request`` broadcast
channel (Redis pub/sub -> every subscriber gets every message), so every
chat-lane verb request (``chat_general``/``chat_quick``) was independently
executed twice -- once by each container, including a real duplicate LLM
gateway call.

Fix: ``app/main.py`` only wires ``verb_listener`` for lanes in
``{"chat", ""}``. ``"legacy"`` was removed from that set. The ``legacy``
container's separate, correct job -- serving direct RPC traffic (e.g.
``stance_react`` from orion-thought) via the bare
``orion:cortex:exec:request`` channel through the unrelated ``Rabbit``-backed
``svc``/``handle()`` registration -- is untouched by this test and by the
fix (see ``services/orion-cortex-exec/app/main.py`` lines ~884-889).

``verb_listener``/``_lane`` are computed once at module import time from
``settings.exec_lane``, so each case below needs a *fresh* import of
``app.main`` (and its transitive ``app.settings``) with the env var set
beforehand. This mirrors the guarded-fresh-import pattern already used by
``test_chat_stance_shared_spine.py`` / ``test_mind_handoff_shortcut.py`` in
this directory (via ``_exec_import_guard.ensure_orion_cortex_exec_app``),
extended with an explicit purge-before-reimport since those two tests only
needed a *clean* app import, not a *lane-varying* one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_guard = Path(__file__).resolve().parent / "_exec_import_guard.py"
_spec = importlib.util.spec_from_file_location("_exec_guard_boot_lane_wiring", _guard)
assert _spec and _spec.loader
_guard_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_guard_mod)

ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from orion.core.bus.bus_service_chassis import Hunter  # noqa: E402


def _fresh_exec_main(monkeypatch: pytest.MonkeyPatch, exec_lane: str | None):
    """Re-import ``app.main`` from scratch with ``EXEC_LANE`` set as given.

    ``exec_lane=None`` means the env var is left unset (exercises
    ``settings.py``'s field default, which is ``"legacy"`` -- see
    ``services/orion-cortex-exec/app/settings.py:28``).
    """
    if exec_lane is None:
        monkeypatch.delenv("EXEC_LANE", raising=False)
    else:
        monkeypatch.setenv("EXEC_LANE", exec_lane)

    app_mod = sys.modules.get("app")
    app_loc = (getattr(app_mod, "__file__", "") or "").replace("\\", "/")
    if app_mod is None or "/orion-cortex-exec/" not in app_loc:
        # Not yet imported this session, or a sibling service's `app`
        # currently owns the top-level name -- do the full purge + sys.path
        # fixup this repo's tests already use for that case.
        _guard_mod.ensure_orion_cortex_exec_app()
    else:
        # `app` already correctly resolves to this service. Only drop
        # app.main/app.settings so the lane-dependent module-level wiring
        # under test re-executes against the env just set. Deliberately do
        # NOT purge the rest of the app.* tree: submodules such as
        # app.verb_adapters register verb triggers into the process-global
        # orion.core.verbs.registry singleton at import time, and
        # re-running that registration raises
        # "ValueError: Verb already registered: legacy.plan".
        sys.modules.pop("app.main", None)
        sys.modules.pop("app.settings", None)

    import app.main as exec_main

    return exec_main


# (input EXEC_LANE, expected resolved `_lane`, expect a live verb_listener, case note)
_LANE_CASES = [
    pytest.param("chat", "chat", True, id="chat-enabled"),
    pytest.param(
        "",
        "chat",
        True,
        # settings.exec_lane == "" is falsy, so main.py's own
        # `str(settings.exec_lane or "chat")` fallback maps it to "chat".
        id="explicit-empty-falls-back-to-chat-enabled",
    ),
    pytest.param(
        "legacy",
        "legacy",
        False,
        # The bug this test suite guards against: "legacy" used to be in the
        # enabled set too, so both legacy and chat independently executed
        # every chat-lane verb request. Must now be disabled.
        id="legacy-disabled",
    ),
    pytest.param(
        None,
        "legacy",
        False,
        # settings.py's field default for EXEC_LANE is "legacy" (matching
        # docker-compose.yml's `EXEC_LANE: ${EXEC_LANE:-legacy}` for the
        # legacy container), NOT "chat" -- unset behaves identically to
        # explicit EXEC_LANE=legacy and must not (re-)register the
        # broadcast listener.
        id="unset-defaults-to-legacy-disabled",
    ),
    pytest.param(
        "spark",
        "spark",
        False,
        # Regression check: spark/background were already excluded before
        # this fix (only "legacy" was removed from the enabled set) and
        # must remain so.
        id="spark-already-disabled",
    ),
    pytest.param("background", "background", False, id="background-already-disabled"),
]


@pytest.mark.parametrize("exec_lane, expected_lane, expect_listener", _LANE_CASES)
def test_verb_listener_lane_wiring(
    monkeypatch: pytest.MonkeyPatch,
    exec_lane: str | None,
    expected_lane: str,
    expect_listener: bool,
) -> None:
    exec_main = _fresh_exec_main(monkeypatch, exec_lane)
    assert exec_main._lane == expected_lane
    if expect_listener:
        assert exec_main.verb_listener is not None
        assert isinstance(exec_main.verb_listener, Hunter)
    else:
        assert exec_main.verb_listener is None
