"""The references contract (orion/refs.py): every kind has one resolver, one dig command and a test.

``SAMPLES`` is the per-kind test: a valid id and the durable-runs answer for it. Adding a kind to
``orion.refs.RefKind`` without a resolver spec, a dig command or a sample here fails
``test_every_kind_has_a_resolver_a_dig_command_and_a_sample``.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from orion import refs
from orion.dig import __main__ as dig

SAMPLES = {
    "arc": ("8d87adeca2ce06a5", {
        "arc_id": "8d87adeca2ce06a5", "day_id": "2026-10-30", "kind": "concern", "status": "open",
        "subject_ref": "open-loop-5cb6cd032923", "subject_label": "", "began_at": "2026-10-30T06:00:00Z",
        "last_seen_at": "2026-10-30T17:33:17Z", "attention_returns": 6, "segments": [{}, {}],
        "carried_from_arc_id": "997fd25d701e4067", "evidence_refs": ["attention_salience_trace:saltrace_1"],
    }, "/temporal-self/arcs/8d87adeca2ce06a5"),
    "act": ("act_42c0e5f55fbd5d881786ebd4", {
        "act_id": "act_42c0e5f55fbd5d881786ebd4", "actor": "prime", "verb": "ask", "level": "deliberate",
        "object_ref": "curiosity_focus:d9eb0763-60b8-51f9-903a-6ef5a1f5ccb0", "signal_id": "artifact_produced",
        "opened_at": "2026-10-30T09:26:26+00:00", "horizon_at": "2026-11-02T09:26:26+00:00",
        "closed_at": "2026-10-30T16:05:03+00:00", "outcome_value": 0, "outcome_source": "producer",
        "outcome_note": "control run 34d7345e36e7 died", "test_before": "holdback:control",
    }, "/acts/act_42c0e5f55fbd5d881786ebd4"),
}


def test_every_kind_has_a_resolver_a_dig_command_and_a_sample():
    kinds = set(refs.REF_KINDS)
    assert set(refs.KINDS) == kinds, "every RefKind needs a RefKindSpec in orion.refs.KINDS"
    assert set(dig.REF_COMMANDS) == kinds, "every RefKind needs a dig command (orion/dig/__main__.py REF_COMMANDS)"
    assert set(SAMPLES) == kinds, "every RefKind needs a sample in this test's SAMPLES"


@pytest.mark.parametrize("kind", sorted(SAMPLES))
def test_resolver_reads_the_one_path_and_renders(kind):
    ref_id, payload, path = SAMPLES[kind]
    seen = []

    def get_json(url):
        seen.append(url)
        return payload

    res = refs.resolve(refs.RefV1(kind=kind, id=ref_id), base_url="http://dr:8121/", get_json=get_json)
    assert res.ok and res.data == payload and seen == [f"http://dr:8121{path}"]
    assert ref_id in res.lines[0]
    assert refs.dig_command(res.ref) == f"python -P -m orion.dig {kind} {ref_id}"


@pytest.mark.parametrize("kind", sorted(SAMPLES))
def test_dig_command_opens_it_with_the_same_resolver(kind, monkeypatch, tmp_path, capsys):
    ref_id, payload, path = SAMPLES[kind]
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path))
    monkeypatch.setenv(refs.DURABLE_RUNS_URL_ENV, "http://dr:8121")
    urls = []

    def fake_urlopen(url, timeout):
        urls.append(url)
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(refs.urllib.request, "urlopen", fake_urlopen)
    assert dig.main([kind, ref_id]) == 0
    out = capsys.readouterr().out
    assert urls == [f"http://dr:8121{path}"] and ref_id in out and f"full {kind}:" in out
    saved = list((tmp_path / "dig").rglob(f"{kind}-{ref_id}-*.json"))
    assert len(saved) == 1 and json.loads(saved[0].read_text()) == payload


@pytest.mark.parametrize("kind", sorted(SAMPLES))
def test_a_missing_id_is_unresolved_never_fine(kind, capsys, monkeypatch, tmp_path):
    ref_id = SAMPLES[kind][0]

    def not_found(url):
        raise urllib.error.HTTPError(url, 404, "nf", {}, None)

    res = refs.resolve(refs.RefV1(kind=kind, id=ref_id), base_url="http://dr", get_json=not_found)
    assert not res.ok and res.error == f"{kind} {ref_id}: not found"

    def down(url):
        raise OSError("refused")

    res = refs.resolve(refs.RefV1(kind=kind, id=ref_id), base_url="http://dr", get_json=down)
    assert not res.ok and "could not read" in res.error

    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path))
    assert dig.main([kind, "not-an-id"]) == 1
    assert "must look like" in capsys.readouterr().err


def test_an_id_of_the_wrong_shape_never_reaches_a_url():
    for kind, bad in (("arc", "../acts/x"), ("arc", "act_42c0e5f55fbd5d881786ebd4"), ("act", "8d87adeca2ce06a5")):
        with pytest.raises(refs.ValidationError):
            refs.RefV1(kind=kind, id=bad)
    with pytest.raises(refs.ValidationError):
        refs.RefV1(kind="run", id="34d7345e36e7")


def test_parse_refs_reports_problems_in_words():
    ok, problems = refs.parse_refs([{"kind": "arc", "id": "8d87adeca2ce06a5"}, {"kind": "arc", "id": "8d87adeca2ce06a5"}])
    assert ok == [refs.RefV1(kind="arc", id="8d87adeca2ce06a5")] and problems == []
    ok, problems = refs.parse_refs([{"kind": "run", "id": "34d7345e36e7"}])
    assert ok == [] and "kinds: arc, act" in problems[0] and "http" not in problems[0]
    assert refs.parse_refs("arc 8d87") == ([], ["refs must be a list of {kind, id}"])


def test_id_tokens_in_text():
    arc = refs.RefV1(kind="arc", id="8d87adeca2ce06a5")
    text = ("Why does arc 8d87adeca2ce06a5 return, and what did run 34d7345e36e7, act act_42c0e5f55fbd5d881786ebd4 "
            "and focus d9eb0763-60b8-51f9-903a-6ef5a1f5ccb0 show? Not ids: 184.86 dollars, 202610301234, facade, "
            "concern_id_x, deadbeefcafe")
    assert refs.id_tokens_not_in(text, [arc]) == [
        "34d7345e36e7", "act_42c0e5f55fbd5d881786ebd4", "d9eb0763-60b8-51f9-903a-6ef5a1f5ccb0"]
    assert refs.id_tokens_not_in("What heats the cabinet after 14:00?", []) == []
