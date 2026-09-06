"""No tracked `.gitignore` may contain a bare catch-all line.

graphify (0.9.15) applies a nested `.gitignore`'s unanchored patterns to the
whole remaining directory walk rather than that file's subtree. A single bare
`*` in `services/orion-hub/tests/e2e/artifacts/.gitignore` made its scanner
report 0 code files at repo root, which is what turned every incremental
graph refresh into a ~95% wipe since 2026-07-14 (root-caused 2026-09-06; see
scripts/check_graphify_scan_scope.py). `/*` means the same thing to git and
stays confined to its directory in graphify.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BARE_CATCHALL = {"*", "**", "**/*"}


def _tracked_gitignores() -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--", ".gitignore", "*/.gitignore", "**/.gitignore"],
        capture_output=True, text=True, check=True,
    ).stdout
    return sorted({REPO_ROOT / line for line in out.splitlines() if line.strip()})


def test_tracked_gitignores_are_found() -> None:
    assert len(_tracked_gitignores()) >= 2  # root plus at least one nested


def test_no_tracked_gitignore_has_a_bare_catchall_line() -> None:
    offenders = []
    for path in _tracked_gitignores():
        for raw in path.read_text(errors="ignore").splitlines():
            line = raw.split("#", 1)[0].strip()
            if line in BARE_CATCHALL:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {raw!r}")
    assert not offenders, (
        "bare catch-all ignore lines leak across graphify's whole scan; "
        "use the anchored form (`/*`) instead:\n  " + "\n  ".join(offenders)
    )


def test_the_artifacts_ignore_is_anchored() -> None:
    text = (REPO_ROOT / "services/orion-hub/tests/e2e/artifacts/.gitignore").read_text()
    assert text.splitlines()[0].strip() == "/*"
