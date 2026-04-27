from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


HUB_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = HUB_ROOT / "scripts" / "run_recall_canary_battle.py"
FIXTURE_PATH = HUB_ROOT / "tests" / "fixtures" / "recall_canary" / "orion_memory_battle_cases.json"
README_PATH = HUB_ROOT / "README.md"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_recall_canary_battle", str(SCRIPT_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load run_recall_canary_battle module")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("run_recall_canary_battle", module)
    spec.loader.exec_module(module)
    return module


def test_battle_fixture_has_required_fields() -> None:
    rows = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert isinstance(rows, list)
    assert len(rows) >= 20
    required = {
        "id",
        "query",
        "expected_project_or_domain",
        "expected_anchor_terms",
        "expected_time_sensitivity",
        "notes",
    }
    for row in rows:
        assert required.issubset(set(row.keys()))


class _FakeClient:
    def __init__(self, *, status_payload: dict, query_payload: dict | None = None):
        self.status_payload = status_payload
        self.query_payload = query_payload or {
            "data": {
                "canary_run_id": "run-1",
                "selected_profile": {"profile_id": "recall_v2_shadow_default"},
                "production_recall_mode": "v1",
                "comparison": {"v1_latency_ms": 10, "v2_latency_ms": 8},
            }
        }
        self.calls: list[tuple[str, str, dict | None]] = []

    def get_status(self) -> dict:
        self.calls.append(("GET", "/api/substrate/recall-canary/status", None))
        return self.status_payload

    def post_query(self, *, query_text: str, profile_id: str) -> dict:
        self.calls.append(
            ("POST", "/api/substrate/recall-canary/query", {"query_text": query_text, "profile_id": profile_id})
        )
        return self.query_payload


def _status_payload(default_profile_id: str = "recall_v2_shadow_default") -> dict:
    return {
        "data": {
            "available_profiles": [
                {
                    "profile_id": default_profile_id,
                    "label": "Recall V2 Shadow Default",
                    "status": "shadow_canary_review_only",
                }
            ],
            "default_canary_profile_id": default_profile_id,
            "production_recall_mode": "v1",
            "recall_live_apply_enabled": False,
            "judgment_counts": {"v2_better": 0},
            "failure_mode_counts": {},
        }
    }


def test_battle_runner_uses_default_profile_when_omitted(tmp_path: Path) -> None:
    module = _load_module()
    cases = module.load_battle_fixture(FIXTURE_PATH)[:2]
    client = _FakeClient(status_payload=_status_payload(default_profile_id="profile-default"))
    summary = module.run_battle(
        client=client,
        fixture_cases=cases,
        requested_profile_id=None,
        output_path=tmp_path / "battle.jsonl",
    )
    assert summary.selected_profile_id == "profile-default"
    posts = [call for call in client.calls if call[0] == "POST"]
    assert len(posts) == 2
    assert all(call[2]["profile_id"] == "profile-default" for call in posts)


def test_battle_runner_rejects_invalid_profile_before_posting(tmp_path: Path) -> None:
    module = _load_module()
    cases = module.load_battle_fixture(FIXTURE_PATH)[:1]
    client = _FakeClient(status_payload=_status_payload(default_profile_id="profile-default"))
    with pytest.raises(ValueError):
        module.run_battle(
            client=client,
            fixture_cases=cases,
            requested_profile_id="invalid-profile",
            output_path=tmp_path / "battle.jsonl",
        )
    posts = [call for call in client.calls if call[0] == "POST"]
    assert posts == []


def test_battle_runner_never_calls_judgment_or_review_or_execute_once(tmp_path: Path) -> None:
    module = _load_module()
    cases = module.load_battle_fixture(FIXTURE_PATH)[:3]
    client = _FakeClient(status_payload=_status_payload())
    summary = module.run_battle(
        client=client,
        fixture_cases=cases,
        requested_profile_id=None,
        output_path=tmp_path / "battle.jsonl",
    )
    assert summary.production_recall_mode == "v1"
    touched_paths = [call[1] for call in client.calls]
    assert all("/judgment" not in path for path in touched_paths)
    assert all("/create-review-artifact" not in path for path in touched_paths)
    assert all("execute-once" not in path for path in touched_paths)
    assert all(path in {"/api/substrate/recall-canary/status", "/api/substrate/recall-canary/query"} for path in touched_paths)


def test_readme_mentions_recall_v2_battle_harness() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    assert "Recall V2 Battle Test Harness" in readme
    assert "run_recall_canary_battle.py" in readme


def test_operator_token_loader_prefers_explicit_then_env(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", "env-token")
    assert module._load_operator_token("explicit-token") == "explicit-token"
    assert module._load_operator_token(None) == "env-token"
