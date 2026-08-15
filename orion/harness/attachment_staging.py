"""Stage chat attachments into the FCC sandbox so Orion's own model can look at them.

Why a copy rather than a path into the attachment store:

The ``claude`` process for a unified turn runs in the **harness-governor**
container, and that container does not mount ``/mnt/orion-chat-attachments`` --
verified live 2026-08-15, it is not there at all. Only the Hub mounts both the
attachment store and the sandbox. So the Hub is the one process that can move
bytes from one to the other, and handing the governor a store path would produce
a file-not-found on the side that actually needs to read it.

Why outside the repo checkout:

``orion/fcc/sandbox_sync.py`` manages ``<sandbox>/repo`` with git and will rescue
a dirty worktree onto a branch. Dropping turn images inside that checkout would
make every image-bearing turn look like uncommitted work and trip the rescue
path. Staging lands in ``<sandbox>/attachments/<correlation_id>/`` instead --
a sibling of ``repo`` that sync never touches.

Reading from there works: claude-code resolves absolute paths outside the
project directory under the harness's permission mode, and its Read tool sniffs
image content rather than trusting the extension (both verified live
2026-08-15 -- a ``.bin`` file outside the workspace was read correctly as an
image by the chat lane).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

logger = logging.getLogger("orion.harness.attachment-staging")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Mirrors the Hub attachment store's accepted set.
_MIME_SUFFIX = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

DEFAULT_SANDBOX_ROOT = "/mnt/orion-fcc"
DEFAULT_STORE_DIR = "/mnt/orion-chat-attachments"
STAGING_DIRNAME = "attachments"


@dataclass(frozen=True)
class StagedAttachment:
    path: str
    mime: str
    sha256: str
    filename: str | None = None


def _default_max_items() -> int:
    try:
        return int(os.environ.get("HUB_CHAT_ATTACHMENT_MAX_PER_TURN", "4") or 4)
    except (TypeError, ValueError):
        return 4


def sandbox_root() -> Path:
    return Path(os.environ.get("HARNESS_ATTACHMENT_SANDBOX_ROOT", DEFAULT_SANDBOX_ROOT))


def store_dir() -> Path:
    return Path(os.environ.get("HUB_CHAT_ATTACHMENT_DIR", DEFAULT_STORE_DIR))


def staging_dir_for(correlation_id: str, *, root: Path | None = None) -> Path:
    # correlation_id reaches us from a request payload, so it is sanitised before
    # it becomes a directory name -- it is the only caller-influenced path
    # component here.
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(correlation_id or "unknown"))[:64]
    # The substitution keeps "." and "-", so a correlation_id of exactly "." or
    # ".." survives it intact -- and ".." would make this return the sandbox
    # ROOT, which prune_staging() then rmtree's, taking the git-managed repo
    # checkout with it. Both call sites pass a uuid4 today; this is the guard
    # that keeps that an implementation detail rather than a load-bearing
    # assumption.
    if not safe or safe.strip(".") == "":
        safe = "unknown"
    return (root or sandbox_root()) / STAGING_DIRNAME / safe


def _safe_display_name(raw: Any) -> str | None:
    """Sanitise a client-declared filename before it can reach a prompt.

    This value is fully client-controlled -- the browser round-trips the ref --
    and ``describe_for_prompt`` interpolates it into the instruction block of a
    motor running under ``--permission-mode bypassPermissions``. Unbounded
    free text there is a prompt-injection surface, so: newlines collapsed,
    length capped, and anything exotic dropped.
    """
    text = str(raw or "").strip()
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^A-Za-z0-9 ._\-]", "", text)
    text = text.strip()[:64]
    return text or None


def _sniffed_mime(src_dir: Path, sha: str) -> str:
    """Read the store's own sniffed type, written beside the bytes.

    chat_attachments.py deliberately does not trust the declared Content-Type
    ("The declared MIME is not trusted") and records what the magic bytes
    actually say. Staging must use that answer, not the client's -- otherwise a
    payload claiming ``filename: "x.sh"`` lands real image bytes under a ``.sh``
    name inside the sandbox that Orion's own Bash tool can enumerate.
    """
    try:
        return (src_dir / f"{sha}.mime").read_text(encoding="utf-8").strip().lower()
    except OSError:
        return ""


def _suffix_for(mime: str) -> str:
    """Extension from the STORE-sniffed mime only. Never from client metadata."""
    return _MIME_SUFFIX.get(str(mime or "").strip().lower(), ".img")


def stage_attachments(
    attachments: Sequence[Any],
    *,
    correlation_id: str,
    source_dir: Path | None = None,
    root: Path | None = None,
    max_items: int | None = None,
) -> list[StagedAttachment]:
    """Copy each attachment out of the store and into the sandbox.

    Accepts anything with ``sha256``/``mime``/``bytes``/``filename`` attributes
    (``AttachmentRefV1``) or the equivalent mapping keys.

    Best-effort per attachment: one unreadable image is logged and skipped
    rather than failing the turn. The caller decides what an empty result means
    -- and it should say something, because a turn that silently loses an image
    is exactly the failure this whole feature exists to prevent.
    """
    if not attachments:
        return []

    # Bounded because `attachments` arrives from a client payload: the browser
    # round-trips the refs, so a scripted caller could otherwise make one turn
    # stage an unbounded number of files into the sandbox.
    limit = max_items if max_items is not None else _default_max_items()
    if limit <= 0:
        # 0 is the intuitive way to say "no images"; treating it as unlimited
        # would remove the very bound the comment above claims to enforce.
        logger.warning("attachment staging disabled (max_items=%s) corr=%s", limit, correlation_id)
        return []
    if len(attachments) > limit:
        logger.warning(
            "turn carried %s attachments; staging only %s corr=%s",
            len(attachments), limit, correlation_id,
        )
        attachments = list(attachments)[:limit]

    src_dir = source_dir or store_dir()
    target_dir = staging_dir_for(correlation_id, root=root)
    staged: list[StagedAttachment] = []

    for item in attachments:
        get = item.get if isinstance(item, dict) else (lambda k, d=None: getattr(item, k, d))
        sha = str(get("sha256", "") or "").strip().lower()
        if not _SHA256_RE.match(sha):
            logger.warning("skipping attachment with non-hex sha256 corr=%s", correlation_id)
            continue
        filename = _safe_display_name(get("filename", None))

        source = src_dir / f"{sha}.bin"
        if not source.exists():
            logger.warning(
                "attachment %s missing from store at %s corr=%s", sha[:12], source, correlation_id
            )
            continue

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            mime = _sniffed_mime(src_dir, sha)
            target = target_dir / f"{sha}{_suffix_for(mime)}"
            if not target.exists():
                # Copy to a temp name then rename, so a reader can never observe
                # a half-written image.
                tmp = target.with_suffix(target.suffix + f".{os.getpid()}.part")
                shutil.copyfile(source, tmp)
                tmp.replace(target)
            staged.append(
                StagedAttachment(
                    path=str(target),
                    mime=mime,
                    sha256=sha,
                    filename=str(filename) if filename else None,
                )
            )
        except OSError as exc:
            logger.warning("failed staging attachment %s corr=%s: %s", sha[:12], correlation_id, exc)

    if staged:
        logger.info(
            "staged %s attachment(s) for corr=%s into %s", len(staged), correlation_id, target_dir
        )
    return staged


def prune_staging(correlation_id: str, *, root: Path | None = None) -> None:
    """Drop a turn's staged images. Never raises -- cleanup must not fail a turn.

    ``ignore_errors=True`` keeps a cleanup problem from taking down a turn, but
    it also swallows the answer, so the result is checked afterwards and a
    survivor is logged. A prune that quietly does nothing would leak the sandbox
    a directory per turn with no signal that it was happening.
    """
    target = staging_dir_for(correlation_id, root=root)
    try:
        shutil.rmtree(target, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001
        logger.debug("staging prune raised corr=%s: %s", correlation_id, exc)
    if target.exists():
        logger.warning(
            "staging prune left %s in place corr=%s (permissions?)", target, correlation_id
        )


def describe_for_prompt(attachments: Iterable[Any]) -> str:
    """Render the prompt block that tells Orion the images exist and where.

    Without this the images sit on disk, perfectly readable, and nothing ever
    mentions them -- which is precisely how they were being dropped before.
    """
    lines: list[str] = []
    for item in attachments:
        get = item.get if isinstance(item, dict) else (lambda k, d=None: getattr(item, k, d))
        path = str(get("path", "") or "").strip()
        if not path:
            continue
        name = get("filename", None)
        label = f" (sent as {name})" if name else ""
        lines.append(f"- {path}{label}")
    if not lines:
        return ""
    plural = "images" if len(lines) > 1 else "an image"
    return (
        f"\n\nJuniper attached {plural} to this message. Use your Read tool on "
        f"the path(s) below to look at {'them' if len(lines) > 1 else 'it'} "
        f"before you answer -- you can see {'them' if len(lines) > 1 else 'it'} "
        f"directly, so do not guess or ask for a description:\n" + "\n".join(lines)
    )
