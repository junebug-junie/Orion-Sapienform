#!/usr/bin/env python3
"""Fail if any tracked file contains a personal term from the hashed denylist.

The repo is public. Names of people close to Juniper, her home place, and her
contact details must not be committed. The denylist stores only SHA-256 hashes of
the lowercased terms (config/privacy/personal_terms.sha256), so the list itself
never leaks them by being read. Note: a hash of a common first name can still be
guessed by hashing a name dictionary; the hashes stop casual reading and grep,
not a determined attacker.

Matching: every tracked text file is split into lowercase letter-only word
tokens; each token and each email address is hashed and looked up. Terms are
therefore single words (or emails); for a multi-word name add its distinctive
word. A hit prints the file, line, and the hash (never the term).

  python scripts/check_personal_terms.py            # gate (exit 1 on any hit)
  python scripts/check_personal_terms.py --add      # read terms from stdin, append hashes

Real values belong in gitignored local files: services/*/.env and
services/orion-ai-town/cards/town_cards.local.yaml.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DENYLIST = ROOT / "config" / "privacy" / "personal_terms.sha256"

# Third-party trees: not ours to edit, and full of unrelated common names.
SKIP_GLOBS = ("*/node_modules/*", "node_modules/*")
MAX_BYTES = 4_000_000

_WORD = re.compile(r"[a-z]+")
_EMAIL = re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}")


def term_hash(term: str) -> str:
    norm = term.strip().lower()
    if " " in norm:
        raise SystemExit(f"single words or emails only, got a phrase ({len(norm)} chars)")
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def load_denylist(path: Path | None = None) -> tuple[set[str], list[tuple[str, str]]]:
    """Returns (hashes, allow) where allow is [(hash, path_glob)]."""
    path = path or DENYLIST
    hashes: set[str] = set()
    allow: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if parts[0] == "allow" and len(parts) == 3:
            allow.append((parts[1], parts[2]))
        elif re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            hashes.add(parts[0])
        else:
            raise SystemExit(f"{path}: bad line: {raw!r}")
    return hashes, allow


_HASH_CACHE: dict[str, str] = {}


def _h(c: str) -> str:
    h = _HASH_CACHE.get(c)
    if h is None:
        h = _HASH_CACHE[c] = hashlib.sha256(c.encode("utf-8")).hexdigest()
    return h


def line_hits(line: str, hashes: set[str]) -> set[str]:
    low = line.lower()
    words = _WORD.findall(low)
    cands = set(words)
    cands.update(_EMAIL.findall(low))
    return {h for h in map(_h, cands) if h in hashes}


def tracked_files(root: Path = ROOT) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True
    ).stdout.decode("utf-8", "replace")
    return [p for p in out.split("\0") if p and not any(fnmatch(p, g) for g in SKIP_GLOBS)]


def scan(root: Path = ROOT, files: list[str] | None = None) -> list[tuple[str, int, str]]:
    hashes, allow = load_denylist()
    found: list[tuple[str, int, str]] = []
    for rel in files if files is not None else tracked_files(root):
        p = root / rel
        try:
            if not p.is_file() or p.stat().st_size > MAX_BYTES:
                continue
            data = p.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:8192]:
            continue  # binary
        text = data.decode("utf-8", "replace")
        if not line_hits(text, hashes):
            continue  # whole file clean: skip the per-line pass
        for n, line in enumerate(text.splitlines(), 1):
            for h in line_hits(line, hashes):
                if any(h == ah and fnmatch(rel, g) for ah, g in allow):
                    continue
                found.append((rel, n, h))
    return found


def add_terms(terms: list[str]) -> int:
    hashes, _ = load_denylist() if DENYLIST.exists() else (set(), [])
    new = sorted({term_hash(t) for t in terms if t.strip()} - hashes)
    with DENYLIST.open("a", encoding="utf-8") as fh:
        for h in new:
            fh.write(h + "\n")
    print(f"added {len(new)} hash(es)")
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["--add"]:
        return add_terms(sys.stdin.read().splitlines())
    hits = scan()
    for rel, n, h in hits:
        print(f"{rel}:{n}: personal term (sha256 {h[:12]}...)")
    if hits:
        print(
            f"\n{len(hits)} personal-term hit(s). Replace with a role placeholder "
            "([spouse], [boss], [home city]) in docs, or an obviously fake name in "
            "tests; real values go in gitignored local files. A genuine non-personal "
            "use (e.g. a library name) can be allowed with an 'allow <sha256> <path-glob>' "
            f"line in {DENYLIST.relative_to(ROOT)}.",
            file=sys.stderr,
        )
        return 1
    print("no personal terms in tracked files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
