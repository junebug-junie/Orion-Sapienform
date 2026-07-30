"""`trigger_kind` -> `CollapseMirrorEntryV2.type` mapping in _fallback_metacog_draft.

Missing Question 6 of docs/superpowers/specs/2026-07-28-collapse-mirror-
generative-triggers-design.md: before this, the type was guessed *only* from phi
bands and ignored `trigger_kind` entirely, so `"epiphany"` was unreachable dead
code no branch could produce.

This function is not fallback-only despite its name -- the successful-LLM-draft
path seeds its `base_entry` from it too, and the draft prompt explicitly forbids
the LLM from emitting `type`. So it is the single construction site deciding
every published entry's `type`, which is why this mapping is worth a gate test.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

EXEC_ROOT = Path(__file__).resolve().parents[1]


def _load_executor_module():
    app_dir = EXEC_ROOT / "app"
    executor_path = app_dir / "executor.py"
    package_name = "orion_cortex_exec_trigger_kind_type"
    app_package_name = f"{package_name}.app"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(app_dir.parent)]
        sys.modules[package_name] = pkg
    if app_package_name not in sys.modules:
        pkg = types.ModuleType(app_package_name)
        pkg.__path__ = [str(app_dir)]
        sys.modules[app_package_name] = pkg
    spec = importlib.util.spec_from_file_location(
        f"{app_package_name}.executor", executor_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Phi bands that the *unchanged* fallback heuristic maps to each of its three
# reachable outputs, so a trigger_kind override can be proven to win over each.
_BANDS_TURBULENCE = {"coherence_band": "low", "novelty_band": "low"}
_BANDS_FLOW = {"energy_band": "high", "coherence_band": "high", "novelty_band": "low"}
_BANDS_IDLE = {"energy_band": "low", "coherence_band": "medium", "novelty_band": "low"}


def _draft(module, *, trigger_kind: str, phi_hint: dict | None = None):
    return module._fallback_metacog_draft(
        {
            "trigger": {"trigger_kind": trigger_kind, "reason": "test", "zen_state": "zen"},
            "phi_hint": dict(phi_hint or _BANDS_IDLE),
        }
    )


# ===========================================================================
# New behavior: trigger_kind wins
# ===========================================================================


def test_insight_maps_to_epiphany() -> None:
    module = _load_executor_module()
    entry = _draft(module, trigger_kind="insight")
    assert entry.type == "epiphany"
    # Acceptance Check 4: the entry's change_type must follow from that type via
    # DEFAULT_CHANGE_TYPE_BY_ENTRY_TYPE, not fall through to context_shift.
    assert entry.change_type == "reorientation"


def test_flow_maps_to_flow() -> None:
    module = _load_executor_module()
    entry = _draft(module, trigger_kind="flow")
    assert entry.type == "flow"
    assert entry.change_type == "stabilizing"


def test_insight_beats_every_phi_band_guess() -> None:
    """trigger_kind is consulted FIRST, so bands that would otherwise produce
    turbulence/flow/idle must not override it."""
    module = _load_executor_module()
    for bands in (_BANDS_TURBULENCE, _BANDS_FLOW, _BANDS_IDLE):
        entry = _draft(module, trigger_kind="insight", phi_hint=bands)
        assert entry.type == "epiphany", bands


def test_flow_kind_beats_turbulence_bands() -> None:
    module = _load_executor_module()
    entry = _draft(module, trigger_kind="flow", phi_hint=_BANDS_TURBULENCE)
    assert entry.type == "flow"


# ===========================================================================
# Regression: every other trigger_kind keeps the unchanged phi-band behavior
# ===========================================================================


def test_other_kinds_still_use_phi_band_guess() -> None:
    module = _load_executor_module()
    # Same three band combinations, now with a kind that has no direct mapping.
    assert _draft(module, trigger_kind="baseline", phi_hint=_BANDS_TURBULENCE).type == "turbulence"
    assert _draft(module, trigger_kind="baseline", phi_hint=_BANDS_FLOW).type == "flow"
    assert _draft(module, trigger_kind="baseline", phi_hint=_BANDS_IDLE).type == "idle"


def test_rupture_shaped_kinds_are_untouched() -> None:
    """The kinds that existed before this patch must be completely unaffected."""
    module = _load_executor_module()
    for kind in ("chat_turn", "transport", "relational", "telemetry_anomaly", "pulse"):
        assert _draft(module, trigger_kind=kind, phi_hint=_BANDS_TURBULENCE).type == "turbulence"
        assert _draft(module, trigger_kind=kind, phi_hint=_BANDS_IDLE).type == "idle"


def test_unknown_trigger_kind_falls_through() -> None:
    module = _load_executor_module()
    entry = _draft(module, trigger_kind="some_future_kind", phi_hint=_BANDS_IDLE)
    assert entry.type == "idle"


def test_trigger_kind_is_still_recorded_on_the_entry() -> None:
    """The mapping must not swallow the trigger identity -- `trigger` still
    carries the kind so an orion_metacog row stays traceable to its cause."""
    module = _load_executor_module()
    entry = _draft(module, trigger_kind="insight")
    assert entry.trigger == "insight"
    assert "insight" in entry.summary
