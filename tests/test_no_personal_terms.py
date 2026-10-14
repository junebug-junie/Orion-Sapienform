"""Public-repo privacy gate: no tracked file may contain a hashed personal term."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "check_personal_terms", _ROOT / "scripts" / "check_personal_terms.py"
)
cpt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cpt)


def test_denylist_holds_hashes_only():
    for raw in cpt.DENYLIST.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if parts[0] == "allow":
            assert re.fullmatch(r"[0-9a-f]{64}", parts[1]), raw
        else:
            assert re.fullmatch(r"[0-9a-f]{64}", line), f"plaintext in denylist: {raw!r}"
    hashes, _ = cpt.load_denylist()
    assert len(hashes) >= 10


def test_line_hits_matches_words_and_emails_case_insensitively():
    hashes = {cpt.term_hash("zorblax"), cpt.term_hash("pat@example.org")}
    assert cpt.line_hits("Dinner with ZORBLAX tonight", hashes) == {cpt.term_hash("zorblax")}
    assert cpt.line_hits("mail Pat@Example.org", hashes) == {cpt.term_hash("pat@example.org")}
    assert cpt.line_hits("zorblaxian ideas", hashes) == set()  # whole words only


def test_line_hits_catches_camel_case_accents_and_email_variants():
    hashes = {cpt.term_hash("zorblax"), cpt.term_hash("pat@example.org")}
    z = {cpt.term_hash("zorblax")}
    assert cpt.line_hits("def zorblaxMemory(): pass", hashes) == z
    assert cpt.line_hits("class ZorblaxCard: ...", hashes) == z
    assert cpt.line_hits("memoryZorblax = 1", hashes) == z
    assert cpt.line_hits("Zörblax said", hashes) == z
    e = {cpt.term_hash("pat@example.org")}
    assert cpt.line_hits("x.pat@example.org", hashes) == e
    assert cpt.line_hits("mailto:pat%40example.org", hashes) == e


def test_term_hash_rejects_phrases():
    with pytest.raises(SystemExit):
        cpt.term_hash("two words")


def test_scan_reports_hit_and_honours_allow(tmp_path, monkeypatch):
    deny = tmp_path / "terms.sha256"
    h = cpt.term_hash("zorblax")
    deny.write_text(f"{h}\n", encoding="utf-8")
    monkeypatch.setattr(cpt, "DENYLIST", deny)
    (tmp_path / "a.md").write_text("ok\nmet Zorblax\n", encoding="utf-8")
    (tmp_path / "lib.py").write_text("import zorblax\n", encoding="utf-8")
    assert cpt.scan(tmp_path, ["a.md", "lib.py"]) == [("a.md", 2, h), ("lib.py", 1, h)]
    deny.write_text(f"{h}\nallow {h} lib.py\n", encoding="utf-8")
    assert cpt.scan(tmp_path, ["a.md", "lib.py"]) == [("a.md", 2, h)]


def test_no_tracked_file_contains_a_personal_term():
    hits = cpt.scan()
    assert not hits, "\n".join(f"{r}:{n}: sha256 {h[:12]}" for r, n, h in hits)


def test_redacted_examples_are_what_is_tracked():
    """Real values live only in gitignored local files."""
    import subprocess

    ignored = subprocess.run(
        ["git", "-C", str(_ROOT), "check-ignore", "-q",
         "services/orion-ai-town/cards/town_cards.local.yaml"],
    ).returncode
    assert ignored == 0
