#!/usr/bin/env python3
"""Gate: no service compose file may bind-mount a host path relatively.

Why this exists (real incident, 2026-08-29):

`scripts/safe_docker_build.sh` refuses to run `docker compose` from the
shared/primary checkout -- worktrees only. But a relative bind mount like
`../../config/biometrics` is resolved by docker compose against the file's
own directory AT `up` TIME and then baked into the container. So deploying
from a worktree silently pins that worktree as production infrastructure:
`orion-circe-biometrics` ended up mounting its node catalog out of
`/mnt/scripts/Orion-Sapienform-deploy-loop`, a throwaway tree, and deleting
that tree would have broken the container on its next restart.

Those two policies cannot both hold. Worktrees are disposable by design;
prod mounts must not be. The resolution is that host paths are absolute,
rooted at ${ORION_REPO_ROOT}, so the mount is identical no matter which tree
the deploy was driven from -- while a developer can still point a local run
at their own worktree by setting ORION_REPO_ROOT.

This is the deterministic gate for that rule, per CLAUDE.md section 4: the
fix for a repeated latent failure is a failing check, not a louder prompt.
"""
from __future__ import annotations

import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]

# A volume entry whose SOURCE side starts with ./ or ../ -- i.e. resolved
# against the compose file's directory rather than an absolute host path.
# Named volumes (no slash), absolute paths, and ${VAR}-rooted paths are fine.
_RELATIVE_SOURCE = re.compile(r"^\s*-\s*(\.{1,2}/[^:]*):")


def main() -> int:
    offenders: list[tuple[str, int, str]] = []
    composes = sorted(REPO.glob("services/*/docker-compose.yml"))

    for path in composes:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            m = _RELATIVE_SOURCE.match(line)
            if m:
                offenders.append((str(path.relative_to(REPO)), lineno, m.group(1)))

    if offenders:
        print("compose relative-mount gate: FAIL")
        print("")
        print("  These bind mounts resolve against the compose file's own directory,")
        print("  so deploying from a git worktree pins that worktree as production")
        print("  infrastructure -- and worktrees are disposable by design.")
        print("")
        for rel, lineno, src in offenders:
            print(f"    {rel}:{lineno}  {src}")
        print("")
        print("  Fix: root the host path at ${ORION_REPO_ROOT}, e.g.")
        print("    - ${ORION_REPO_ROOT:-/mnt/scripts/Orion-Sapienform}/config/foo:/app/config/foo:ro")
        return 1

    print(f"compose relative-mount gate: PASS ({len(composes)} compose files, 0 relative host mounts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
