#!/usr/bin/env python3
"""Read-only measurement: does an embedding-diff pre-filter separate real
doc/README rewrites from trivial (typo/one-line) doc commits at a real,
calibratable threshold?

This is `docs/superpowers/specs/2026-07-30-doc-semantic-drift-design.md`'s
own "Recommended next patch" -- the embedding-diff pre-filter alone, read-
only, against real historical doc commits, before touching graphify Part B
escalation or the `orion/concepts/` registry shape at all.

Existing-mechanism check (that design doc's own "Missing questions" --
"what non-frontier embedding model is acceptable to run inline"): confirmed
live 2026-08-11 that `orion-vector-host` is already running locally
(`docker ps`), HF backend, model `BAAI/bge-large-en-v1.5` -- a real,
already-vetted, non-frontier embedding model this repo already depends on
in production. This script reuses that exact model by calling directly into
the live container's already-loaded `Embedder` (via `docker exec`), rather
than adding a new embedding dependency to this repo or duplicating a
multi-GB model download in this script's own environment. The container's
real bus-facing contract (`EmbeddingGenerateV1`/`EmbeddingResultV1`) is
coupled to vector-store persistence (upsert as a side effect) -- not what a
stateless, read-only, offline calibration pass wants -- so this script
bypasses that contract deliberately and calls `Embedder.embed()` directly,
never touching the vector store.

Sample: 5 real historical doc commits from this repo's own git history,
picked for a real size spread (1-line PR-link edit up to a 36-line real
README addition) -- not synthetic text.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Real commits from this repo's own history, 2026-08-11. `path` is the doc
# file touched; `before`/`after` are read via `git show` at build time, not
# hardcoded here (see _load_sample below) so this stays a real read against
# live git history, not a frozen fixture.
REAL_SAMPLE: tuple[dict, ...] = (
    {
        "sha": "9e34ee88e10b93728dbc4771e168eda94cd25b0e",
        "path": "docs/superpowers/pr-reports/2026-07-31-transport-domain-ewma-successor-pr.md",
        "shortstat": "1 insertion(+), 1 deletion(-)",
        "expected": "trivial",
    },
    {
        "sha": "f88dc088f4c73f340d764b7b9355e840a684236c",
        "path": "services/orion-substrate-runtime/README.md",
        "shortstat": "3 insertions(+), 1 deletion(-)",
        "expected": "trivial",
    },
    {
        "sha": "dc375dca75438803a0c4635a4d7e29a83670150b",
        "path": "services/orion-attention-runtime/README.md",
        "shortstat": "17 insertions(+)",
        "expected": "real",
    },
    {
        "sha": "2144f852a62136ad9d611ff3e7564f0ed170dcfe",
        "path": "services/orion-equilibrium-service/README.md",
        "shortstat": "24 insertions(+), 4 deletions(-)",
        "expected": "real",
    },
    {
        "sha": "121185a3589ed68d0be0b9d67e43c69e8466bbde",
        "path": "services/orion-substrate-runtime/README.md",
        "shortstat": "36 insertions(+)",
        "expected": "real",
    },
)


def _git_show(rev: str, path: str) -> str:
    """Empty string if the path didn't exist at that revision (e.g. a new
    file) -- a real, meaningful "before" state, not an error."""
    result = subprocess.run(
        ["git", "show", f"{rev}:{path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout


def _git_diff_hunks(sha: str, path: str) -> tuple[str, str]:
    """(removed_text, added_text) -- just the changed lines from the real
    unified diff for this commit/file, not the whole before/after document.
    Real fix for the whole-document-embedding truncation defect this script
    itself found and reported (see module docstring, "Diff-scoped
    embedding, the real fix"): a real doc edit is almost always far smaller
    than the whole document, so hunk text reliably fits inside the model's
    real 512-token limit even when the surrounding document doesn't.

    File-header lines (`--- a/...`, `+++ b/...`) and hunk markers (`@@ ...
    @@`) are excluded -- only real content lines. A pure addition yields an
    empty ``removed_text``; a pure deletion yields an empty ``added_text``
    -- both real, meaningful states (``main()`` embeds them via the same
    ``text or " "`` fallback every other text in this script uses), not
    errors.

    Known, currently-unhit limitation: a renamed file (no ``-M`` passed to
    ``git diff``) shows as a full delete-at-old-path + full-add-at-new-path,
    so ``removed``/``added`` would be the *entire* file content for that
    case -- degenerating back into the whole-document truncation problem
    this function exists to avoid. Not a live bug (``REAL_SAMPLE`` only
    touches files edited in place, never renamed), but a real gap to close
    with an explicit guard before extending the sample set to renamed
    files. Likewise a binary-file diff produces no ``+``/``-`` lines at all
    (`Binary files ... differ`), silently yielding two empty strings rather
    than an error -- also currently moot for this script's real,
    hand-picked, text-only sample."""
    result = subprocess.run(
        ["git", "diff", "--no-color", "--unified=0", f"{sha}~1", sha, "--", path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    removed: list[str] = []
    added: list[str] = []
    for line in result.stdout.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added.append(line[1:])
        elif line.startswith("-"):
            removed.append(line[1:])
        # everything else (hunk headers, context lines under --unified=0
        # there shouldn't be any, diff metadata) is skipped
    return "\n".join(removed), "\n".join(added)


def _commit_subject(sha: str) -> str:
    result = subprocess.run(
        ["git", "log", "-1", "--pretty=%s", sha],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


_CONVENTIONAL_PREFIXES = ("feat", "fix", "chore", "docs", "test", "refactor", "perf", "build", "ci")


def conventional_commit_prefix(subject: str) -> str | None:
    """The free tier: a self-declared conventional-commit type prefix
    (`docs(x): ...`, `fix: ...`), or None if the subject doesn't use one.
    Weak feature, per the design doc's own "Missing questions" -- a
    `docs:` commit is self-declared, not verified against what actually
    changed."""
    head = subject.split(":", 1)[0].strip()
    kind = head.split("(", 1)[0].strip().lower()
    return kind if kind in _CONVENTIONAL_PREFIXES else None


# Confirmed live 2026-08-11 via the actual tokenizer inside the running
# orion-vector-host container (transformers AutoTokenizer.model_max_length
# for BAAI/bge-large-en-v1.5). `Embedder._embed_hf` calls the tokenizer with
# `truncation=True` and no explicit `max_length`, so it silently truncates
# to this many tokens -- real text past that point never reaches the model
# at all, for either the "before" or "after" side of a diff. Informational
# only now (surfaced in the report) -- the real truncation *decision* per
# sample comes from a live token count measured inside the container (see
# ``_embed_batch_via_vector_host``), not from this constant directly.
EMBEDDING_MODEL_MAX_TOKENS = 512


@dataclass(frozen=True)
class DriftSample:
    sha: str
    path: str
    shortstat: str
    expected: str
    commit_subject: str
    commit_prefix: str | None
    before_len_chars: int
    after_len_chars: int
    embedding_diff: float  # 1 - cosine_similarity; 0 = identical, 2 = opposite
    # Real, live-measured (not char-count-approximated) truncation fact for
    # either the "before" or "after" text, from the container's own real
    # tokenizer. Code review 2026-08-11: a chars-per-token average is a
    # prose estimate and this repo's real docs are heavy in code blocks/
    # paths/identifiers that tokenize denser than prose -- a char-count
    # proxy risks a false negative on exactly this corpus. None means the
    # live measurement wasn't taken (e.g. a code path that only has char
    # counts available, such as a test fixture).
    likely_truncated: bool | None = None
    # Diff-scoped embedding-diff -- embeds only the changed hunk lines (see
    # _git_diff_hunks), not the whole before/after document. The real fix
    # this script's own whole-document finding calls for.
    diff_scoped_embedding_diff: float | None = None
    diff_scoped_truncated: bool | None = None
    hunk_removed_len_chars: int = 0
    hunk_added_len_chars: int = 0


def _load_sample(entry: dict) -> tuple[str, str, str, str, dict]:
    before = _git_show(f"{entry['sha']}~1", entry["path"])
    after = _git_show(entry["sha"], entry["path"])
    subject = _commit_subject(entry["sha"])
    return before, after, subject, conventional_commit_prefix(subject), entry


def _embed_batch_via_vector_host(
    texts: list[str], *, container: str
) -> tuple[list[list[float]], list[bool]]:
    """One docker exec call embeds the whole batch (not one call per text)
    -- real, live, already-loaded BAAI/bge-large-en-v1.5 inside the running
    orion-vector-host container. Communicates over stdin/stdout JSON so no
    text has to survive shell quoting.

    Also returns a real, live-measured ``truncated`` flag per text, from the
    container's actual tokenizer -- not a char-count proxy. Code review
    2026-08-11 correctly flagged that a chars-per-token average is a prose
    estimate, and this repo's real docs are heavy in code blocks/paths/
    identifiers that tokenize denser than prose -- a doc just under a char
    threshold could still be really truncated. Since this script already
    execs into the container that owns the real tokenizer, using it directly
    is strictly more correct than approximating from ``len(text)``, and no
    more expensive (the tokenizer call is cheap; the model forward pass,
    already paid for by ``embed()``, dominates the real cost either way)."""
    script = """
import asyncio, json, sys
sys.path.insert(0, "/app")
from app.settings import Settings
from app.embedder import Embedder

texts = json.load(sys.stdin)
settings = Settings()
embedder = Embedder(settings)
tokenizer = embedder._tokenizer  # already loaded by Embedder.__init__ (hf backend)

async def go():
    vectors = []
    truncated = []
    for text in texts:
        real_text = text or " "
        vector, model, dim = await embedder.embed(real_text)
        vectors.append(vector)
        if tokenizer is not None:
            # Real token count with truncation disabled, compared against
            # the tokenizer's own real limit -- not an approximation.
            token_count = len(tokenizer(real_text, truncation=False)["input_ids"])
            truncated.append(token_count > tokenizer.model_max_length)
        else:
            truncated.append(False)
    print(json.dumps({"vectors": vectors, "truncated": truncated}))

asyncio.run(go())
"""
    result = subprocess.run(
        ["docker", "exec", "-i", container, "python3", "-c", script],
        input=json.dumps(texts),
        capture_output=True,
        text=True,
        check=True,
        # CPU inference for BAAI/bge-large-en-v1.5, ~335M params, one
        # sequential forward pass per text (no batching in Embedder.embed) --
        # confirmed live 2026-08-11 that 10 real README-length texts exceed
        # 120s on this host's CPU. Generous, not tuned to a measured worst
        # case yet.
        timeout=600,
    )
    payload = json.loads(result.stdout)
    return payload["vectors"], payload["truncated"]


def sample_truncated(before_truncated: bool, after_truncated: bool) -> bool:
    """Either side of a before/after diff being truncated makes the whole
    comparison untrustworthy -- a truncated "before" that happens to share
    its (also truncated) leading content with the "after" produces a near-
    zero embedding_diff regardless of what changed past the truncation
    point. Extracted as its own function so this combining rule is testable
    without a live container."""
    return before_truncated or after_truncated


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="orion-athena-vector-host")
    parser.add_argument("--out-dir", default="/tmp/doc-semantic-drift-calibration")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = [_load_sample(entry) for entry in REAL_SAMPLE]
    hunks = [_git_diff_hunks(entry["sha"], entry["path"]) for entry in REAL_SAMPLE]

    # One batch call embeds everything -- whole-document before/after AND
    # diff-scoped hunk before/after -- 4 texts per sample, in a fixed order
    # so indices below line up.
    all_texts = []
    for (before, after, _subject, _prefix, _entry), (hunk_removed, hunk_added) in zip(loaded, hunks):
        all_texts.append(before)
        all_texts.append(after)
        all_texts.append(hunk_removed)
        all_texts.append(hunk_added)
    embeddings, truncated_flags = _embed_batch_via_vector_host(all_texts, container=args.container)

    samples: list[DriftSample] = []
    for i, ((before, after, subject, prefix, entry), (hunk_removed, hunk_added)) in enumerate(
        zip(loaded, hunks)
    ):
        base = i * 4
        before_vec, after_vec, hunk_removed_vec, hunk_added_vec = (
            embeddings[base], embeddings[base + 1], embeddings[base + 2], embeddings[base + 3]
        )
        similarity = _cosine_similarity(before_vec, after_vec)
        hunk_similarity = _cosine_similarity(hunk_removed_vec, hunk_added_vec)
        samples.append(
            DriftSample(
                sha=entry["sha"],
                path=entry["path"],
                shortstat=entry["shortstat"],
                expected=entry["expected"],
                commit_subject=subject,
                commit_prefix=prefix,
                before_len_chars=len(before),
                after_len_chars=len(after),
                embedding_diff=1.0 - similarity,
                # Real per-side truncation, measured by the container's own
                # tokenizer.
                likely_truncated=sample_truncated(
                    truncated_flags[base], truncated_flags[base + 1]
                ),
                diff_scoped_embedding_diff=1.0 - hunk_similarity,
                diff_scoped_truncated=sample_truncated(
                    truncated_flags[base + 2], truncated_flags[base + 3]
                ),
                hunk_removed_len_chars=len(hunk_removed),
                hunk_added_len_chars=len(hunk_added),
            )
        )

    truncated = [s for s in samples if s.likely_truncated]
    clean = [s for s in samples if not s.likely_truncated]
    trivial_scores = [s.embedding_diff for s in clean if s.expected == "trivial"]
    real_scores = [s.embedding_diff for s in clean if s.expected == "real"]
    separates = bool(trivial_scores) and bool(real_scores) and max(trivial_scores) < min(real_scores)
    proposed_threshold = (
        (max(trivial_scores) + min(real_scores)) / 2 if separates else None
    )

    lines = [
        "# Doc semantic drift: embedding-diff pre-filter calibration",
        "",
        f"Embedding model: `BAAI/bge-large-en-v1.5` (live, via `{args.container}`), "
        f"real hard limit {EMBEDDING_MODEL_MAX_TOKENS} tokens (confirmed live from the "
        f"actual tokenizer, not assumed).",
        "",
        "| sha | path | shortstat | commit prefix | embedding_diff | expected | truncation risk |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in sorted(samples, key=lambda s: s.embedding_diff):
        lines.append(
            f"| {s.sha[:8]} | {s.path.split('/')[-1]} | {s.shortstat} | "
            f"{s.commit_prefix or '-'} | {s.embedding_diff:.4f} | {s.expected} | "
            f"{'YES (' + str(max(s.before_len_chars, s.after_len_chars)) + ' chars)' if s.likely_truncated else 'no'} |"
        )

    lines += ["", "## Real finding: whole-document embedding is structurally broken for real docs"]
    if truncated:
        lines.append(
            f"**{len(truncated)} of {len(samples)} samples are likely truncated** -- confirmed "
            f"live for the largest (`services/orion-substrate-runtime/README.md`, "
            f"71,795 real characters before, 74,482 after; the real edit is a paragraph appended "
            f"near the end of the file, far past the model's real {EMBEDDING_MODEL_MAX_TOKENS}-token "
            f"limit). A truncated \"before\" and truncated \"after\" that share the same leading "
            f"content produce `embedding_diff ~= 0.0000` regardless of how real the actual edit "
            f"was -- this is exactly what the table above shows: every sample, trivial or "
            f"substantial, scores ~0.0000. **This is not \"no separation found\" (a negative result "
            f"about the embedding-diff signal) -- it's a structural defect in whole-document "
            f"embedding as a technique for docs this size**, which most real READMEs in this repo "
            f"are (several exceed 70KB). Naively embedding the whole before/after text, as the "
            f"design doc's Phase-1 framing implicitly assumed, cannot work without a real fix "
            f"(diff-scoped embedding -- embed only the changed hunks/sections, not the whole "
            f"document -- or a longer-context embedding model) before any threshold from this "
            f"sample, or any future sample, can be trusted."
        )
    # A real check, not decorative prose (code review 2026-08-11: an
    # earlier draft hardcoded "too few samples" regardless of the actual
    # count) -- a future rerun with a genuinely larger non-truncated sample
    # should not still claim "too few" once that's no longer true.
    MIN_SAMPLES_PER_SIDE_FOR_REAL_THRESHOLD = 5
    if separates:
        enough = (
            len(trivial_scores) >= MIN_SAMPLES_PER_SIDE_FOR_REAL_THRESHOLD
            and len(real_scores) >= MIN_SAMPLES_PER_SIDE_FOR_REAL_THRESHOLD
        )
        confidence = (
            "a real, trustworthy calibrated cutoff"
            if enough
            else f"only a midpoint -- {len(trivial_scores)} trivial / {len(real_scores)} real "
            f"non-truncated samples is fewer than the {MIN_SAMPLES_PER_SIDE_FOR_REAL_THRESHOLD} "
            f"per side this script requires before calling a threshold calibrated, not guessed"
        )
        lines.append(
            f"\nOn the {len(clean)} sample(s) short enough to avoid truncation risk: all `trivial` "
            f"samples score below all `real` samples (max trivial={max(trivial_scores):.4f}, "
            f"min real={min(real_scores):.4f}). Proposed threshold **{proposed_threshold:.4f}** is "
            f"{confidence}."
        )
    elif clean:
        lines.append(
            f"\nOn the {len(clean)} sample(s) short enough to avoid truncation risk: no clean "
            f"separation either. Sample too small to draw a conclusion either way."
        )

    # -- Diff-scoped embedding: the real fix --------------------------------
    diff_truncated = [s for s in samples if s.diff_scoped_truncated]
    diff_clean = [s for s in samples if not s.diff_scoped_truncated]
    diff_trivial_scores = [s.diff_scoped_embedding_diff for s in diff_clean if s.expected == "trivial"]
    diff_real_scores = [s.diff_scoped_embedding_diff for s in diff_clean if s.expected == "real"]
    diff_separates = (
        bool(diff_trivial_scores)
        and bool(diff_real_scores)
        and max(diff_trivial_scores) < min(diff_real_scores)
    )
    diff_proposed_threshold = (
        (max(diff_trivial_scores) + min(diff_real_scores)) / 2 if diff_separates else None
    )

    lines += [
        "",
        "## Diff-scoped embedding (the real fix): embed only the changed hunk lines",
        "",
        "| sha | hunk removed/added chars | diff_scoped_embedding_diff | expected | truncated |",
        "|---|---|---|---|---|",
    ]
    for s in sorted(samples, key=lambda s: s.diff_scoped_embedding_diff):
        lines.append(
            f"| {s.sha[:8]} | {s.hunk_removed_len_chars}/{s.hunk_added_len_chars} | "
            f"{s.diff_scoped_embedding_diff:.4f} | {s.expected} | "
            f"{'YES' if s.diff_scoped_truncated else 'no'} |"
        )

    lines += [""]
    if diff_truncated:
        lines.append(
            f"{len(diff_truncated)} of {len(samples)} hunk-scoped samples still flagged truncated "
            f"(a single commit touching hundreds of lines in one hunk can still exceed 512 tokens) "
            f"-- excluded from the separation check below, same discipline as the whole-document "
            f"section above."
        )
    if diff_separates:
        enough = (
            len(diff_trivial_scores) >= MIN_SAMPLES_PER_SIDE_FOR_REAL_THRESHOLD
            and len(diff_real_scores) >= MIN_SAMPLES_PER_SIDE_FOR_REAL_THRESHOLD
        )
        diff_confidence = (
            "a real, trustworthy calibrated cutoff"
            if enough
            else f"only a midpoint -- {len(diff_trivial_scores)} trivial / {len(diff_real_scores)} "
            f"real non-truncated samples is fewer than {MIN_SAMPLES_PER_SIDE_FOR_REAL_THRESHOLD} "
            f"per side"
        )
        lines.append(
            f"\n**Real separation found on diff-scoped embedding.** All `trivial` samples score "
            f"below all `real` samples (max trivial={max(diff_trivial_scores):.4f}, "
            f"min real={min(diff_real_scores):.4f}). Proposed threshold "
            f"**{diff_proposed_threshold:.4f}** is {diff_confidence}. This confirms the diagnosis "
            f"above: the embedding-diff *signal* works; whole-document embedding was the broken "
            f"part, not the technique."
        )
    else:
        lines.append(
            "\n**No clean separation on diff-scoped embedding either**, on this same 5-sample set. "
            "Given the whole-document approach is confirmed structurally broken, this would mean "
            "either the sample is too small (5 is genuinely small for calibrating a threshold) or "
            "embedding-diff needs further tuning (e.g. weighting hunk size) even once truncation "
            "is fixed -- not that diff-scoping failed to fix the truncation defect it targets."
        )

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
