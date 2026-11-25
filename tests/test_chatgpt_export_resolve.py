"""resolve_export_path must find conversations in old and sharded ChatGPT exports."""

import json
import zipfile

import pytest

from orion.importers.chatgpt_export import load_conversations, resolve_export_path


def _conv(cid: str) -> dict:
    return {"id": cid, "mapping": {}}


def _write_zip(path, files: dict) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, json.dumps(data))


def test_sharded_zip_merges_all_shards_and_ignores_lookalikes(tmp_path):
    # The 2026-11 export: shards plus look-alike files that end in conversations.json.
    export = tmp_path / "export.zip"
    _write_zip(
        export,
        {
            "sectioned_conversations.json": [],
            "shared_conversations.json": [{"id": "shared-not-a-conv"}],
            "conversations-001.json": [_conv("c"), _conv("d")],
            "conversations-000.json": [_conv("a"), _conv("b")],
        },
    )
    ids = [c["id"] for c in load_conversations(resolve_export_path(export))]
    assert ids == ["a", "b", "c", "d"]


def test_single_file_zip_still_works(tmp_path):
    export = tmp_path / "export.zip"
    _write_zip(
        export,
        {"shared_conversations.json": [], "conversations.json": [_conv("x")]},
    )
    ids = [c["id"] for c in load_conversations(resolve_export_path(export))]
    assert ids == ["x"]


def test_sharded_directory_merges(tmp_path):
    (tmp_path / "conversations-000.json").write_text(json.dumps([_conv("a")]))
    (tmp_path / "conversations-001.json").write_text(json.dumps([_conv("b")]))
    (tmp_path / "sectioned_conversations.json").write_text("[]")
    ids = [c["id"] for c in load_conversations(resolve_export_path(tmp_path))]
    assert ids == ["a", "b"]


def test_zip_with_only_lookalikes_raises(tmp_path):
    export = tmp_path / "export.zip"
    _write_zip(export, {"sectioned_conversations.json": [], "shared_conversations.json": []})
    with pytest.raises(FileNotFoundError):
        resolve_export_path(export)
