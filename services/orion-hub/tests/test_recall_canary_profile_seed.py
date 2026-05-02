from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

hub_scripts_pkg = HUB_ROOT / "scripts" / "__init__.py"
if (
    "scripts" not in sys.modules
    or not str(getattr(sys.modules.get("scripts"), "__file__", "")).startswith(str(HUB_ROOT))
):
    spec = importlib.util.spec_from_file_location(
        "scripts",
        str(hub_scripts_pkg),
        submodule_search_locations=[str(HUB_ROOT / "scripts")],
    )
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules["scripts"] = module
        spec.loader.exec_module(module)

seed_spec = importlib.util.spec_from_file_location(
    "seed_recall_canary_profile",
    str(HUB_ROOT / "scripts" / "seed_recall_canary_profile.py"),
)
if seed_spec is None or seed_spec.loader is None:
    raise RuntimeError("failed to load seed_recall_canary_profile module")
seed_module = importlib.util.module_from_spec(seed_spec)
seed_spec.loader.exec_module(seed_module)

from scripts import api_routes
from orion.substrate.mutation_queue import SubstrateMutationStore


def test_seed_default_recall_canary_profile_is_idempotent() -> None:
    store = SubstrateMutationStore()
    first, created_first = seed_module.seed_recall_canary_profile(store=store)
    second, created_second = seed_module.seed_recall_canary_profile(store=store)
    assert created_first is True
    assert created_second is False
    assert first.profile_id == second.profile_id == "recall_v2_shadow_default"
    listed = store.list_recall_strategy_profiles(limit=20)
    assert len([row for row in listed if row.get("profile_id") == "recall_v2_shadow_default"]) == 1


def test_seeded_profile_appears_in_canary_status_and_stays_bounded(monkeypatch) -> None:
    store = SubstrateMutationStore()
    profile, _ = seed_module.seed_recall_canary_profile(store=store)
    monkeypatch.setattr(api_routes, "SUBSTRATE_MUTATION_STORE", store)
    payload = api_routes.api_substrate_recall_canary_status(limit=20)
    data = payload["data"]
    row = next(item for item in data["available_profiles"] if item["profile_id"] == profile.profile_id)
    assert data["default_canary_profile_id"] == profile.profile_id
    assert data["production_recall_mode"] == "v1"
    assert data["recall_live_apply_enabled"] is False
    assert row["production_default"] is False
    assert row["live_apply_enabled"] is False
    assert row["status"] == "shadow_canary_review_only"
    saved = store.get_recall_strategy_profile(profile.profile_id)
    assert saved is not None
    assert saved.status == "staged"


def test_seeding_does_not_trigger_execute_once_or_promote(monkeypatch) -> None:
    calls = {"mutation": 0, "review": 0}
    monkeypatch.setattr(api_routes, "_execute_substrate_mutation_cycle", lambda *a, **k: calls.__setitem__("mutation", calls["mutation"] + 1))
    monkeypatch.setattr(api_routes, "_execute_substrate_review_cycle", lambda *a, **k: calls.__setitem__("review", calls["review"] + 1))
    store = SubstrateMutationStore()
    profile, _ = seed_module.seed_recall_canary_profile(store=store)
    assert profile.status == "staged"
    assert calls == {"mutation": 0, "review": 0}


def test_seed_script_and_docs_keep_unsafe_actions_absent() -> None:
    readme = (HUB_ROOT / "README.md").read_text(encoding="utf-8")
    script_text = (HUB_ROOT / "scripts" / "seed_recall_canary_profile.py").read_text(encoding="utf-8")
    forbidden = [
        "Promote to Production",
        "Apply Recall Patch",
        "Apply Recall Profile",
        "Enable Recall V2",
        "Switch Production Recall",
        "Auto Promote",
    ]
    merged = readme + "\n" + script_text
    for token in forbidden:
        assert token not in merged
