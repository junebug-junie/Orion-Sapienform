"""Real personal data lives in gitignored *.local.* files; the committed
*.example.* files are redacted fixtures and the loaders read local first."""

import json
import subprocess
from pathlib import Path

import yaml

from orion.person_model.person import DEFAULT_ANSWERS_PATH, REPO_ROOT, local_first

EVALS = REPO_ROOT / "orion" / "person_model" / "evals"


def test_local_first_prefers_the_local_file(tmp_path):
    ex = tmp_path / "stated_answers.example.yaml"
    ex.write_text("people: {}\n", encoding="utf-8")
    assert local_first(ex) == ex
    local = tmp_path / "stated_answers.local.yaml"
    local.write_text("people: {a: 1}\n", encoding="utf-8")
    assert local_first(ex) == local
    plain = tmp_path / "people.yaml"
    assert local_first(plain) == plain


def test_local_files_are_gitignored():
    for rel in ("config/person_model/stated_answers.local.yaml",
                "orion/person_model/evals/cleanup_labels.local.jsonl",
                "orion/person_model/evals/mention_labels.local.jsonl"):
        assert subprocess.run(["git", "-C", str(REPO_ROOT), "check-ignore", "-q", rel]).returncode == 0, rel


def test_example_answers_have_the_shape_the_dry_run_reads():
    data = yaml.safe_load(DEFAULT_ANSWERS_PATH.read_text(encoding="utf-8"))
    for entry in data["people"].values():
        assert entry["home"]["value"] and entry["home"]["source"]


def test_example_cleanup_labels_load_through_the_eval():
    from orion.person_model.evals.run_linkage_eval import load_labels

    labels = load_labels(EVALS / "cleanup_labels.example.jsonl", {})
    assert labels and set(labels.values()) <= {"same", "facet", "junk", "different", "not_home", "distinct"}


def test_example_mention_spans_match_their_text():
    rows = [json.loads(x) for x in (EVALS / "mention_labels.example.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    assert rows
    for r in rows:
        assert [r["fixture_text"][a:b] for a, b in r["persons"]] == r["persons_text"]
        assert [r["fixture_text"][a:b] for a, b in r["relations"]] == r["relations_text"]
