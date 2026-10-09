"""Internal document sources: pure refs, allowlisted capture, snapshot evidence."""
from __future__ import annotations

import hashlib
import os

import pytest

from orion.schemas.reading import ReadingRequestedV1, SourceFetchEvidenceV1
from orion.world_pulse_read.documents import (
    SNAPSHOT_TOOL,
    DocumentPolicy,
    DocumentSourceError,
    document_ref,
    normalize_document_ref,
    parse_document_ref,
    read_document,
    unversioned_ref,
)
from orion.world_pulse_read.read_evidence import document_snapshot_evidence, source_read_evidence
from orion.world_pulse_read.urls import normalize_reading_source

SHA = "b" * 64


def _policy(root, **over) -> DocumentPolicy:
    values = {"roots": str(root), "extensions": None, "max_bytes": 4096, **over}
    return DocumentPolicy.from_values(**values)


def test_refs_are_pure_and_round_trip() -> None:
    assert normalize_document_ref("/mnt/x/docs/a b.md") == "file:///mnt/x/docs/a%20b.md"
    assert normalize_document_ref("file:///mnt/x/./docs/spec.md") == "file:///mnt/x/docs/spec.md"
    assert parse_document_ref(f"file:///mnt/x/spec.md?sha256={SHA}") == ("/mnt/x/spec.md", SHA)
    assert unversioned_ref(f"file:///mnt/x/spec.md?sha256={SHA}") == "file:///mnt/x/spec.md"
    assert normalize_reading_source("/mnt/x/spec.md") == "file:///mnt/x/spec.md"
    assert normalize_reading_source("https://Example.org/a#frag") == "https://example.org/a"


@pytest.mark.parametrize("bad", [
    "relative/spec.md", "/mnt/x/../etc/passwd", "file://host/mnt/x.md", "file:///mnt/x.md?sha256=nothex",
    "file:///mnt/x.md?other=1", "file:///mnt/x.md#frag", "/mnt/x\nspec.md", "/mnt/x/sp?ec.md",
])
def test_bad_refs_are_refused(bad: str) -> None:
    with pytest.raises(DocumentSourceError):
        normalize_document_ref(bad)


def test_capture_reads_exact_bytes_under_an_allowed_root(tmp_path) -> None:
    body = "# Spec\n\nOrion reads internal documents.\n".encode()
    (tmp_path / "spec.md").write_bytes(body)
    doc = read_document(str(tmp_path / "spec.md"), _policy(tmp_path))
    assert doc.sha256 == hashlib.sha256(body).hexdigest()
    assert doc.text == body.decode()
    assert doc.ref == document_ref(os.path.realpath(tmp_path / "spec.md"), doc.sha256)


@pytest.mark.parametrize("name,content,code", [
    ("big.md", b"x" * 5000, "document_too_large"),
    ("bin.md", b"abc\x00def", "document_not_text"),
    ("latin.md", "caf\xe9".encode("latin-1"), "document_not_text"),
    ("empty.md", b"  \n\t", "document_empty"),
    ("script.py", b"print('hi')", "document_type_not_allowed"),
    (".env.md", b"SECRET=1", "document_path_denied"),
])
def test_capture_refuses_what_is_not_a_readable_text_document(tmp_path, name, content, code) -> None:
    (tmp_path / name).write_bytes(content)
    with pytest.raises(DocumentSourceError, match=code):
        read_document(str(tmp_path / name), _policy(tmp_path))


def test_capture_refuses_paths_outside_roots_symlink_escapes_and_git(tmp_path) -> None:
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir(), outside.mkdir()
    (outside / "secret.md").write_text("private")
    (root / "link.md").symlink_to(outside / "secret.md")
    (root / ".git").mkdir()
    (root / ".git" / "notes.md").write_text("git internals")
    (root / "dir.md").mkdir()
    policy = _policy(root)
    with pytest.raises(DocumentSourceError, match="document_outside_allowed_roots"):
        read_document(str(outside / "secret.md"), policy)
    with pytest.raises(DocumentSourceError, match="document_outside_allowed_roots"):
        read_document(str(root / "link.md"), policy)
    with pytest.raises(DocumentSourceError, match="document_path_denied"):
        read_document(str(root / ".git" / "notes.md"), policy)
    with pytest.raises(DocumentSourceError, match="document_not_found"):
        read_document(str(root / "missing.md"), policy)
    with pytest.raises(DocumentSourceError, match="document_not_a_file"):
        read_document(str(root / "dir.md"), policy)


def test_sibling_directory_with_shared_prefix_is_not_inside_root(tmp_path) -> None:
    (tmp_path / "Orion").mkdir()
    (tmp_path / "Orion-other").mkdir()
    (tmp_path / "Orion-other" / "a.md").write_text("not in root")
    with pytest.raises(DocumentSourceError, match="document_outside_allowed_roots"):
        read_document(str(tmp_path / "Orion-other" / "a.md"), _policy(tmp_path / "Orion"))


def test_open_judges_the_file_it_opened_not_the_path_it_checked(tmp_path) -> None:
    # Each case is what the checked path could be swapped to before open().
    from orion.world_pulse_read.documents import _read_checked_file

    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir(), outside.mkdir()
    (outside / "secret.md").write_text("private")
    os.mkfifo(root / "pipe.md")
    (root / "final.md").symlink_to(outside / "secret.md")
    (root / "docs").symlink_to(outside)
    with pytest.raises(DocumentSourceError, match="document_not_a_file"):
        _read_checked_file(str(root / "pipe.md"), 100)  # returns; never blocks on the FIFO
    with pytest.raises(DocumentSourceError, match="document_changed_during_read"):
        _read_checked_file(str(root / "final.md"), 100)
    with pytest.raises(DocumentSourceError, match="document_changed_during_read"):
        _read_checked_file(str(root / "docs" / "secret.md"), 100)


def test_lookups_normalize_to_the_stored_form(tmp_path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "spec.md").write_text("text")
    (tmp_path / "alias").symlink_to(tmp_path / "real")
    stored = read_document(str(tmp_path / "real" / "spec.md"), _policy(tmp_path)).ref
    assert normalize_document_ref(str(tmp_path / "alias" / "spec.md")) == unversioned_ref(stored)
    assert normalize_document_ref("//mnt/x/spec.md") == "file:///mnt/x/spec.md"


def test_empty_roots_disable_document_reading(tmp_path) -> None:
    (tmp_path / "a.md").write_text("text")
    with pytest.raises(DocumentSourceError, match="document_reading_disabled"):
        read_document(str(tmp_path / "a.md"), _policy(tmp_path, roots=""))
    assert DocumentPolicy.from_env({}).roots  # unset env keeps the defaults


def test_request_schema_accepts_documents_and_keeps_http_normalization() -> None:
    doc = ReadingRequestedV1(url=f"file:///mnt/x/spec.md?sha256={SHA}", requested_by="juniper",
                             invocation_context="operator")
    assert doc.url == f"file:///mnt/x/spec.md?sha256={SHA}"
    web = ReadingRequestedV1(url="https://example.org", requested_by="juniper", invocation_context="operator")
    assert web.url == "https://example.org/"
    with pytest.raises(ValueError):
        ReadingRequestedV1(url="file://host/x.md", requested_by="juniper", invocation_context="operator")


def test_only_hubs_own_snapshot_record_proves_a_document_read() -> None:
    seed_url = f"file:///mnt/x/spec.md?sha256={SHA}"
    own = document_snapshot_evidence(seed_url, content_sha256=SHA, content_chars=900)
    assert source_read_evidence(seed_url, [own]) == [own]
    # A model tool call naming the same path is not a read of the snapshot.
    model_fetch = SourceFetchEvidenceV1(url=seed_url, tool_name="WebFetch", content_chars=900)
    assert source_read_evidence(seed_url, [model_fetch]) == []
    # A snapshot of different bytes is not this seed's read.
    other = document_snapshot_evidence(seed_url, content_sha256="c" * 64, content_chars=900)
    assert source_read_evidence(seed_url, [other]) == []
    assert own.tool_name == SNAPSHOT_TOOL


def test_web_evidence_serializes_without_the_document_field() -> None:
    web = SourceFetchEvidenceV1(url="https://ex.com/a", tool_name="WebFetch", content_chars=10)
    assert web.model_dump(mode="json") == {"url": "https://ex.com/a", "tool_name": "WebFetch", "content_chars": 10}
