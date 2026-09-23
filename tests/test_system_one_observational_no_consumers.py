"""Guard: observational System One questions must not gain accidental consumers."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Files allowed to *mention* these question ids (schema, appraisal builder,
# eval/smoke, docs, tests). Behavioral consumers must not appear elsewhere.
OBSERVATIONAL = ("reverie_fit", "attention_interrupt", "deliberation_need")
ALLOWED_SUBSTRINGS = (
    "orion/schemas/system_one_appraisal.py",
    "orion/substrate/system_one_appraisal.py",
    "orion/substrate/system_one_access.py",
    "scripts/smoke_system_one_appraisal.py",
    "scripts/analysis/eval_system_one_appraisal.py",
    "tests/test_system_one",
    "tests/test_system_one_access.py",
    "services/orion-substrate-runtime/tests/test_",
    "services/orion-substrate-runtime/app/settings.py",
    "services/orion-substrate-runtime/README.md",
    "services/orion-kev/README.md",
    "docs/superpowers/",
)


def test_observational_system_one_questions_have_no_runtime_consumers():
    offenders: list[str] = []
    for path in REPO.rglob("*.py"):
        rel = path.relative_to(REPO).as_posix()
        if any(part in rel for part in ("graphify-out", ".git", "__pycache__", ".worktrees")):
            continue
        if any(allow in rel for allow in ALLOWED_SUBSTRINGS):
            continue
        # Appraisal producer + access helpers are the only runtime readers of
        # frame answers that may name all four questions.
        if rel.endswith("system_one_appraisal.py") or rel.endswith("system_one_access.py"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for question in OBSERVATIONAL:
            if question in text:
                # Curiosity gate module may list them only in comments about
                # non-promotion — still disallow outside allowlist.
                offenders.append(f"{rel}:{question}")
    assert offenders == [], f"unexpected observational System One consumers: {offenders}"
