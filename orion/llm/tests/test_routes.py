from __future__ import annotations

from orion.llm.routes import (
    ACCEPTED_LLM_ROUTES,
    BACKGROUND_LLM_ROUTES,
    LLM_ROUTE_DISPLAY_ORDER,
    SYSTEM_LLM_ROUTES,
    normalize_llm_route,
)


def test_harness_is_accepted_and_displayed() -> None:
    # `harness` (2026-08-20): the FCC/Claude Code CLI harness's own route, split off `chat`
    # for the same reason `agent` was split off `chat` on 2026-08-14 -- see the
    # ACCEPTED_LLM_ROUTES docstring in orion/llm/routes.py. Regression guard: this module
    # raises at import if a route is accepted but absent from the display order, so this test
    # mostly documents the contract rather than catching drift the import-time check wouldn't
    # already catch -- but an explicit assertion here means a future rename/removal of
    # `harness` fails a fast, readable test instead of only a RuntimeError at collection time.
    assert "harness" in ACCEPTED_LLM_ROUTES
    assert "harness" in LLM_ROUTE_DISPLAY_ORDER


def test_harness_is_not_a_valid_general_caller_override() -> None:
    # Review caught this live 2026-08-20: `harness` being in ACCEPTED_LLM_ROUTES made it a valid
    # override at normalize_llm_route()'s three real call sites (orion-actions,
    # orion-cortex-exec, Hub's cortex_request_builder.py) even though SYSTEM_LLM_ROUTES already
    # said it should never be a general caller's choice -- that exclusion only reached Hub's UI
    # picker, not the actual server-side acceptance boundary. `harness` must resolve the same
    # way an unrecognized route does here: None, so callers fall through to their own
    # verb-based default rather than actually dispatching onto the FCC-reserved lane.
    assert normalize_llm_route("harness") is None
    assert normalize_llm_route("HARNESS") is None
    # This does not affect the real production path: anthropic_passthrough.resolve_anthropic_route
    # (what ~/.fcc/.env's MODEL=llamacpp/harness actually resolves through) never calls
    # normalize_llm_route -- it looks the route table up directly.


def test_harness_is_not_a_background_route() -> None:
    # FCC turns are interactive (Juniper is waiting on them), not a yielding lane -- unlike
    # `quick_background`, a caller choosing `harness` should get normal foreground admission,
    # not slot-slack waiting. If `harness` is ever given a `priority: "background"` route-table
    # entry, it must also be added here deliberately, not fall in by accident.
    assert "harness" not in BACKGROUND_LLM_ROUTES


def test_harness_is_a_system_route_hidden_from_the_human_picker() -> None:
    # Regression for the actual bug: an earlier version of this module claimed in a comment that
    # `harness` was "not human-interactive" with nothing that enforced it. Hub's picker filters
    # on `priority`, not on this module directly, so the claim was inert -- `harness` would have
    # been offered as an ordinary choosable Compute lane. `system` is deliberately a different
    # value from `background`: it must dispatch immediately (no slot-slack wait), just never be
    # a human's turn.
    assert "harness" in SYSTEM_LLM_ROUTES
    assert "harness" not in BACKGROUND_LLM_ROUTES


def test_metacog_background_is_accepted_background_and_displayed() -> None:
    # Second pilot of the `quick_background` pattern (2026-08-29): shares metacog's upstream
    # (circe-worker-2/GPU5) but yields for /slots slack. Nothing routes onto it by default --
    # see ORION_REVERIE_METACOG_BACKGROUND_ENABLED in services/orion-thought/.env_example.
    assert "metacog_background" in ACCEPTED_LLM_ROUTES
    assert "metacog_background" in LLM_ROUTE_DISPLAY_ORDER
    assert "metacog_background" in BACKGROUND_LLM_ROUTES
    assert "metacog_background" not in SYSTEM_LLM_ROUTES


def test_metacog_background_is_a_valid_caller_override() -> None:
    # Unlike `harness`, this is a real caller-selectable override -- reverie.py's
    # `_metacog_route()` sets it explicitly once its own flag is on.
    assert normalize_llm_route("metacog_background") == "metacog_background"
    assert normalize_llm_route("METACOG_BACKGROUND") == "metacog_background"


def test_system_and_background_routes_are_mutually_exclusive() -> None:
    # orion/llm/routes.py raises at import if this is ever violated; this test exists so the
    # invariant shows up in a normal test run too, not only as an import-time RuntimeError.
    assert not (SYSTEM_LLM_ROUTES & BACKGROUND_LLM_ROUTES)
    assert SYSTEM_LLM_ROUTES <= ACCEPTED_LLM_ROUTES


def test_display_order_still_matches_accepted_routes() -> None:
    # orion/llm/routes.py already asserts this at import time; this test exists so the
    # invariant shows up in a normal test run's output too, not only as an import-time
    # RuntimeError from an unrelated test file that happened to import the module first.
    assert set(LLM_ROUTE_DISPLAY_ORDER) == set(ACCEPTED_LLM_ROUTES)
    assert len(LLM_ROUTE_DISPLAY_ORDER) == len(set(LLM_ROUTE_DISPLAY_ORDER))
