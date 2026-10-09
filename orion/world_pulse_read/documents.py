"""Internal documents as reading sources: allowlisted paths, captured once.

A document request names an absolute path on a filesystem Hub already mounts.
Hub reads it at acceptance, stores the exact text content-addressed in
``reading_document_snapshot`` and rewrites the source to
``file:///abs/path?sha256=<hex>``. The reader only ever sees that snapshot,
never the live file: a durable turn can wait hours for GPU and the first bound
prompt is authoritative (durable.py), so the file may have moved on by then.

Same path + same bytes dedups like a re-requested URL (already_read); an edited
file has a new sha256 and is a new read.
"""
from __future__ import annotations

import errno
import hashlib
import os
import posixpath
import re
import stat
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import quote, unquote, urlsplit

DOCUMENT_SCHEME = "file"
# tool_name on SourceFetchEvidenceV1 for a Hub-captured snapshot. Distinct from
# every harness fetch tool, so a model tool call can never be mistaken for it.
SNAPSHOT_TOOL = "orion_document_snapshot"

ROOTS_ENV = "HUB_READING_DOCUMENT_ROOTS"
EXTENSIONS_ENV = "HUB_READING_DOCUMENT_EXTENSIONS"
MAX_BYTES_ENV = "HUB_READING_DOCUMENT_MAX_BYTES"

DEFAULT_ROOTS = ("/mnt/scripts/Orion-Sapienform", "/mnt/orion-fcc/repo")
DEFAULT_EXTENSIONS = (".md", ".markdown", ".txt", ".rst", ".adoc")
# The whole snapshot rides in the Stage 1 prompt, which is one `claude -p`
# argv string (Linux MAX_ARG_STRLEN = 131072 bytes) on the `agent` lane
# (32768-token window). Larger documents are refused, never truncated: a
# partial read must not be recorded as a read.
DEFAULT_MAX_BYTES = 49152

_SHA = re.compile(r"^[0-9a-f]{64}$")
_DENY_COMPONENTS = frozenset({".git", ".ssh", ".gnupg", ".aws", ".docker"})
_DENY_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".kdbx")
_DENY_PREFIXES = (".env", "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa")

ENSURE_SNAPSHOT_SQL = """
CREATE TABLE IF NOT EXISTS reading_document_snapshot (
    sha256 text PRIMARY KEY CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    content text NOT NULL,
    content_chars integer NOT NULL CHECK (content_chars > 0),
    first_source text NOT NULL,
    captured_at timestamptz NOT NULL DEFAULT now()
);
"""


class DocumentSourceError(ValueError):
    """A short, operator-facing policy code (``str(exc)``); never file content."""


@dataclass(frozen=True)
class DocumentPolicy:
    roots: tuple[str, ...]
    extensions: frozenset[str]
    max_bytes: int

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "DocumentPolicy":
        env = os.environ if env is None else env
        try:
            max_bytes = int(env.get(MAX_BYTES_ENV) or DEFAULT_MAX_BYTES)
        except ValueError:
            max_bytes = DEFAULT_MAX_BYTES
        return cls.from_values(
            roots=env.get(ROOTS_ENV), extensions=env.get(EXTENSIONS_ENV), max_bytes=max_bytes,
        )

    @classmethod
    def from_values(
        cls, *, roots: str | None, extensions: str | None, max_bytes: int,
    ) -> "DocumentPolicy":
        """``roots=None`` means the defaults; an empty string disables reading."""
        roots = DEFAULT_ROOTS if roots is None else tuple(
            r.strip() for r in roots.split(",") if r.strip()
        )
        extensions = DEFAULT_EXTENSIONS if not (extensions or "").strip() else tuple(
            e.strip().lower() for e in extensions.split(",") if e.strip()
        )
        return cls(
            roots=tuple(os.path.realpath(r) for r in roots if os.path.isabs(r)),
            extensions=frozenset(e if e.startswith(".") else f".{e}" for e in extensions),
            max_bytes=max(1, max_bytes),
        )


@dataclass(frozen=True)
class CapturedDocument:
    path: str
    sha256: str
    text: str

    @property
    def ref(self) -> str:
        return document_ref(self.path, self.sha256)


def is_document_ref(value: Any) -> bool:
    raw = str(value or "").strip()
    return raw.startswith("/") or raw.lower().startswith(f"{DOCUMENT_SCHEME}:")


def document_ref(path: str, sha256: str | None = None) -> str:
    ref = f"{DOCUMENT_SCHEME}://{quote(path, safe='/-._~+@')}"
    return f"{ref}?sha256={sha256}" if sha256 else ref


def parse_document_ref(value: Any) -> tuple[str, str | None]:
    """``(absolute_path, sha256 | None)``. Pure: never touches the filesystem."""
    raw = str(value or "").strip()
    if not raw or any(ord(c) < 32 or ord(c) == 127 for c in raw) or "\\" in raw:
        raise DocumentSourceError("invalid_document_path")
    if raw.startswith("/"):
        path, sha = raw, None
    else:
        parts = urlsplit(raw)
        if parts.scheme.lower() != DOCUMENT_SCHEME or parts.netloc or parts.fragment:
            raise DocumentSourceError("invalid_document_path")
        sha = None
        if parts.query:
            key, _, value = parts.query.partition("=")
            if key != "sha256" or not _SHA.match(value):
                raise DocumentSourceError("invalid_document_path")
            sha = value
        path = unquote(parts.path)
    if "?" in path or "#" in path or "\x00" in path:
        raise DocumentSourceError("invalid_document_path")
    if not path.startswith("/") or any(seg == ".." for seg in path.split("/")):
        raise DocumentSourceError("invalid_document_path")
    # POSIX normpath keeps a leading "//"; Hub stores the single-slash form.
    return "/" + posixpath.normpath(path).lstrip("/"), sha


def normalize_document_ref(value: Any) -> str:
    """The ref as Hub stores it: symlinks resolved where this host sees the path."""
    path, sha = parse_document_ref(value)
    return document_ref(os.path.realpath(path), sha)


def unversioned_ref(value: Any) -> str:
    path, _ = parse_document_ref(value)
    return document_ref(path)


def _denied(path: str) -> bool:
    parts = [p for p in path.split("/") if p]
    if any(p in _DENY_COMPONENTS for p in parts):
        return True
    name = parts[-1].lower() if parts else ""
    return name.startswith(_DENY_PREFIXES) or name.endswith(_DENY_SUFFIXES)


def _within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


def check_document_path(value: Any, policy: DocumentPolicy) -> str:
    """Policy checks on the path alone; returns it with symlinks resolved.

    Blocking (resolving symlinks touches the filesystem); call via a thread.
    """
    if not policy.roots:
        raise DocumentSourceError("document_reading_disabled")
    requested, _ = parse_document_ref(value)
    real = os.path.realpath(requested)
    # Symlinks are resolved first: a link inside a root must not reach outside it.
    if not any(_within(real, root) for root in policy.roots):
        raise DocumentSourceError("document_outside_allowed_roots")
    if _denied(requested) or _denied(real):
        raise DocumentSourceError("document_path_denied")
    if os.path.splitext(real)[1].lower() not in policy.extensions:
        raise DocumentSourceError("document_type_not_allowed")
    return real


def read_document(value: Any, policy: DocumentPolicy) -> CapturedDocument:
    """Read one allowlisted text document. Blocking; call via a thread."""
    real = check_document_path(value, policy)
    if not os.path.exists(real):
        raise DocumentSourceError("document_not_found")
    if not os.path.isfile(real):
        raise DocumentSourceError("document_not_a_file")
    return capture_text(real, _read_checked_file(real, policy.max_bytes + 1), policy)


def _read_checked_file(real: str, limit: int) -> bytes:
    # The path can be swapped between the checks above and open(): a directory
    # for a symlink out of the roots, the file for a FIFO. So open without
    # following a final symlink or blocking, then judge the file actually opened.
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(real, flags)
    except FileNotFoundError as exc:
        raise DocumentSourceError("document_not_found") from exc
    except OSError as exc:
        code = "document_changed_during_read" if exc.errno == errno.ELOOP else "document_unreadable"
        raise DocumentSourceError(code) from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise DocumentSourceError("document_not_a_file")
        if _opened_path(fd, real) != real:
            raise DocumentSourceError("document_changed_during_read")
        chunks: list[bytes] = []
        remaining = limit
        while remaining > 0:
            chunk = os.read(fd, min(remaining, 65536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    except OSError as exc:
        raise DocumentSourceError("document_unreadable") from exc
    finally:
        os.close(fd)


def _opened_path(fd: int, fallback: str) -> str:
    # Linux only; elsewhere the pre-open checks stand alone.
    try:
        return os.readlink(f"/proc/self/fd/{fd}")
    except OSError:
        return fallback


def capture_text(path: str, data: bytes, policy: DocumentPolicy) -> CapturedDocument:
    if len(data) > policy.max_bytes:
        raise DocumentSourceError("document_too_large")
    if b"\x00" in data:
        raise DocumentSourceError("document_not_text")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentSourceError("document_not_text") from exc
    if not text.strip():
        raise DocumentSourceError("document_empty")
    return CapturedDocument(path=path, sha256=hashlib.sha256(data).hexdigest(), text=text)


async def store_snapshot(conn: Any, doc: CapturedDocument) -> None:
    await conn.execute(
        """INSERT INTO reading_document_snapshot (sha256, content, content_chars, first_source)
           VALUES ($1, $2, $3, $4) ON CONFLICT (sha256) DO NOTHING""",
        doc.sha256, doc.text, len(doc.text), doc.ref,
    )


async def load_snapshot(conn: Any, sha256: str) -> str | None:
    return await conn.fetchval(
        "SELECT content FROM reading_document_snapshot WHERE sha256 = $1", sha256,
    )
