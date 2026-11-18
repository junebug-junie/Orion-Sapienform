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

Two kinds of attachment are staged. ``kind: "image"`` (the default) is a file a person chose to upload, read
from Hub's chat attachment store. ``kind: "percept"`` is a still of a game screen (game company patch 3,
docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md sections 2 and 5), read over HTTP
from orion-percept-store, which keeps a frame one hour after its LAST post. A percept ref carries
``frames_expire_at``: past it the frame is gone by design and staging answers "expired" from the ref without
asking the store; before it a missing frame is a fault and is logged as lost. Staged percepts carry
``source = "game_screen"`` so the prompt names them as frames of the game screen, never as something Juniper
attached.

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
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

logger = logging.getLogger("orion.harness.attachment-staging")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Mirrors the Hub attachment store's accepted set.
_MIME_SUFFIX = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

GAME_SCREEN = "game_screen"          # StagedAttachment.source of a percept (a still of a game screen)
PERCEPT_FETCH_TIMEOUT_SEC = 10.0

DEFAULT_SANDBOX_ROOT = "/mnt/orion-fcc"
DEFAULT_STORE_DIR = "/mnt/orion-chat-attachments"
STAGING_DIRNAME = "attachments"


@dataclass(frozen=True)
class StagedAttachment:
    path: str
    mime: str
    sha256: str
    filename: str | None = None
    source: str | None = None          # None: a person's upload; GAME_SCREEN: a still of a game screen


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


def percept_store_url() -> str:
    return str(os.environ.get("PERCEPT_STORE_BASE_URL", "") or "").strip().rstrip("/")


def _percept_token() -> str:
    return str(os.environ.get("PERCEPT_STORE_TOKEN", "") or "").strip()


def fetch_percept(sha: str, *, base_url: str | None = None, timeout_sec: float = PERCEPT_FETCH_TIMEOUT_SEC) -> bytes | None:
    """The bytes of one percept from orion-percept-store, or None when the store answers 404 (gone or never
    there). Any other failure raises OSError. urllib: no new dependency in the harness package."""
    base = (base_url or percept_store_url()).rstrip("/")
    if not base:
        raise OSError("percept_store_unconfigured")
    req = urllib.request.Request(f"{base}/{sha}", method="GET")
    token = _percept_token()
    if token:
        req.add_header("X-Orion-Percept-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # noqa: S310 -- internal service URL
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise OSError(f"percept_http_{exc.code}") from exc


def _as_utc(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def percept_state(expires_at: Any, now: Optional[datetime] = None) -> str:
    """``expired`` when the ref's ``frames_expire_at`` is past (the frame is gone by design: do not ask the
    store), ``live`` when it is in the future, ``unknown`` when the ref carries no readable time (ask the
    store; a 404 is then reported as lost)."""
    when = _as_utc(expires_at)
    if when is None:
        return "unknown"
    return "expired" if (now or datetime.now(timezone.utc)) >= when else "live"


def _stage_percept(get: Callable[..., Any], sha: str, target_dir: Path, correlation_id: str,
                   fetch: Callable[[str], bytes | None], now: Optional[datetime]) -> StagedAttachment | None:
    state = percept_state(get("frames_expire_at", None), now)
    if state == "expired":
        logger.info("percept %s expired by design (frames_expire_at passed) corr=%s", sha[:12], correlation_id)
        return None
    mime = str(get("mime", "") or "image/png").strip().lower()
    if mime not in _MIME_SUFFIX:
        mime = "image/png"
    target = target_dir / f"{sha}{_suffix_for(mime)}"
    if not target.exists():
        data = fetch(sha)
        if data is None:
            # Before its time a missing frame is a fault, never "there was nothing to see".
            logger.error("percept %s lost: store has no frame before frames_expire_at corr=%s", sha[:12], correlation_id)
            return None
        target_dir.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + f".{os.getpid()}.part")
        tmp.write_bytes(data)
        tmp.replace(target)
    return StagedAttachment(path=str(target), mime=mime, sha256=sha, filename=None, source=GAME_SCREEN)


def stage_attachments(
    attachments: Sequence[Any],
    *,
    correlation_id: str,
    source_dir: Path | None = None,
    root: Path | None = None,
    max_items: int | None = None,
    fetch: Callable[[str], bytes | None] | None = None,
    now: Optional[datetime] = None,
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
        if str(get("kind", "") or "").strip().lower() == "percept":
            try:
                item_staged = _stage_percept(get, sha, target_dir, correlation_id, fetch or fetch_percept, now)
            except OSError as exc:
                logger.warning("failed staging percept %s corr=%s: %s", sha[:12], correlation_id, exc)
                item_staged = None
            if item_staged is not None:
                staged.append(item_staged)
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

    The sentence names the source (game company patch 3): frames of a game screen (``source ==
    GAME_SCREEN``) are introduced as such, with their real count, never as something Juniper attached.
    """
    uploads: list[str] = []
    frames: list[str] = []
    for item in attachments:
        get = item.get if isinstance(item, dict) else (lambda k, d=None: getattr(item, k, d))
        path = str(get("path", "") or "").strip()
        if not path:
            continue
        if get("source", None) == GAME_SCREEN:
            frames.append(f"- {path}")
            continue
        name = get("filename", None)
        label = f" (sent as {name})" if name else ""
        uploads.append(f"- {path}{label}")
    out = ""
    if uploads:
        plural = "images" if len(uploads) > 1 else "an image"
        out += (
            f"\n\nJuniper attached {plural} to this message. Use your Read tool on "
            f"the path(s) below to look at {'them' if len(uploads) > 1 else 'it'} "
            f"before you answer -- you can see {'them' if len(uploads) > 1 else 'it'} "
            f"directly, so do not guess or ask for a description:\n" + "\n".join(uploads)
        )
    if frames:
        n = len(frames)
        what = (f"These are {n} frames of the game screen from the last 30 seconds, oldest first"
                if n > 1 else "This is 1 frame of the game screen from the last 30 seconds")
        out += (
            f"\n\n{what}. Use your Read tool on the path(s) below to look at "
            f"{'them' if n > 1 else 'it'} before you answer -- you can see {'them' if n > 1 else 'it'} directly:\n"
            + "\n".join(frames)
        )
    return out
