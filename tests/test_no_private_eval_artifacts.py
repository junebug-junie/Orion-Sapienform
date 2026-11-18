"""The repo is public: eval artifacts that quote Juniper's memories and chats stay out of git.

2026-10-14: the reading-resonance labeling set (docs/superpowers/evals/*labeling-set*.{md,csv,html})
carried crystallized memories of conversations with Juniper verbatim and was committed on a
PR branch of this public repo. Those files are now written to /tmp only
(services/orion-hub/evals/run_reading_resonance_eval.py refuses a path inside the checkout),
.gitignore covers them, and this gate fails if any such file is tracked again.
"""
from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVALS_DIR = "docs/superpowers/evals/"
# Shapes of a per-item human-review artifact: it carries the stored texts being reviewed.
FORBIDDEN = ("*labeling-set*", "*-review.html", "*-decisions.csv")


def private_eval_files(paths: list[str]) -> list[str]:
    return sorted(p for p in paths if p.startswith(EVALS_DIR)
                  and any(fnmatch.fnmatch(Path(p).name, pat) for pat in FORBIDDEN))


def test_no_private_review_artifact_is_tracked():
    tracked = subprocess.run(["git", "ls-files", EVALS_DIR], cwd=REPO, capture_output=True, text=True,
                             check=True).stdout.split()
    bad = private_eval_files(tracked)
    assert not bad, f"private review artifacts tracked in a public repo (write them under /tmp): {bad}"


def test_pattern_catches_the_incident_files_and_spares_ordinary_evals():
    incident = [
        "docs/superpowers/evals/2026-10-13-resonance-labeling-set.md",
        "docs/superpowers/evals/2026-10-13-resonance-labeling-set.csv",
        "docs/superpowers/evals/2026-10-13-resonance-labeling-set.html",
        "docs/superpowers/evals/x-review.html",
        "docs/superpowers/evals/x-decisions.csv",
    ]
    assert private_eval_files(incident) == sorted(incident)
    assert private_eval_files(["docs/superpowers/evals/2026-10-01-recall-eval.md",
                               "docs/other/x-labeling-set.md"]) == []


def test_gitignore_covers_the_labeling_set():
    out = subprocess.run(["git", "check-ignore", "-q", "--no-index",
                          "docs/superpowers/evals/2026-10-13-resonance-labeling-set.html"], cwd=REPO)
    assert out.returncode == 0
