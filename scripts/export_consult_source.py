#!/usr/bin/env python3
"""Export the deployed code for consult rooms: tracked files only, no ``.git``, no secrets.

Consult turns (orion-room-companion, room id ``consult:<id>``) let Claude read the code with
Read/Grep/Glob. They must not read anything else. Orion's FCC sandbox clone was the first choice and
is NOT safe: its ``.git/config`` remote URL carries a GitHub credential, and Orion writes notes and
scratch files into it during turns (found in review, 2026-11-11). So the companion mounts this
export instead.

``git archive <commit>`` writes only tracked files: no ``.git``, no gitignored ``.env``, no
``graphify-out``. Each commit goes into ``<dest>/<sha>``. ``<dest>/current`` is a relative symlink
swapped atomically, so a turn that is already running keeps its tree. The companion's working
directory is ``/consult-src/current``, resolved again at each turn. The newest two exports are kept.

    python3 scripts/export_consult_source.py [--repo .] [--ref HEAD] [--dest /mnt/scripts/orion-consult-src]

Prints one line: ``ok sha=<sha> created=<bool>`` or ``failed reason=<...>``. Exit 0 either way when
called with ``--never-fail`` (the post-pull rebuild does): a failed export never fails a deploy.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

KEEP = 2
FORBIDDEN_NAMES = (".git", ".env")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def find_forbidden(root: Path) -> list[str]:
    """Paths in an export that must never be there (a ``.git`` dir or a ``.env`` file)."""
    bad = []
    for path in root.rglob("*"):
        if path.name in FORBIDDEN_NAMES:
            bad.append(str(path.relative_to(root)))
    return bad


def export(repo: Path, ref: str, dest: Path) -> tuple[str, bool]:
    sha = _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / sha
    created = False
    if not target.is_dir():
        tmp = Path(tempfile.mkdtemp(prefix=f".tmp-{sha[:12]}-", dir=dest))
        try:
            with tempfile.TemporaryFile() as tar_file:
                subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", sha], check=True, stdout=tar_file)
                tar_file.seek(0)
                with tarfile.open(fileobj=tar_file) as tar:
                    tar.extractall(tmp, filter="data")
            bad = find_forbidden(tmp)
            if bad:
                raise RuntimeError(f"export holds forbidden paths: {bad[:5]}")
            os.chmod(tmp, 0o755)
            os.replace(tmp, target)
            created = True
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
    link_tmp = dest / f".current-{os.getpid()}"
    if link_tmp.is_symlink() or link_tmp.exists():
        link_tmp.unlink()
    link_tmp.symlink_to(sha)  # relative: resolves inside the container mount too
    os.replace(link_tmp, dest / "current")
    exports = sorted((p for p in dest.iterdir() if p.is_dir() and not p.is_symlink() and not p.name.startswith(".")),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    for old in exports[KEEP:]:
        if old.name != sha:
            shutil.rmtree(old, ignore_errors=True)
    return sha, created


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--dest", default=os.environ.get("ORION_CONSULT_SOURCE_DIR", "/mnt/scripts/orion-consult-src"))
    ap.add_argument("--never-fail", action="store_true")
    a = ap.parse_args(argv)
    try:
        sha, created = export(Path(a.repo), a.ref, Path(a.dest))
    except Exception as exc:  # noqa: BLE001 -- one line, the caller logs it
        print(f"failed reason={type(exc).__name__}: {str(exc)[:200]}")
        return 0 if a.never_fail else 1
    print(f"ok sha={sha[:12]} created={created}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
