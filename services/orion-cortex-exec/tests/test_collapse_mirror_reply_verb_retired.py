from __future__ import annotations

from pathlib import Path


def test_respond_to_juniper_collapse_mirror_verb_yaml_removed() -> None:
    root = Path(__file__).resolve().parents[3]
    verb_path = root / "orion" / "cognition" / "verbs" / "actions.respond_to_juniper_collapse_mirror.v1.yaml"
    assert not verb_path.exists()


def test_verb_adapter_symbol_removed() -> None:
    import app.verb_adapters as adapters

    assert not hasattr(adapters, "RespondToJuniperCollapseMirrorVerb")
