"""scripts/smoke_introspect.py: argument combos fail fast; a named dream id that comes back empty is degenerate."""
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "smoke_introspect.py"
_spec = importlib.util.spec_from_file_location("smoke_introspect", _SCRIPT)
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)


def _result(items=(), total=0):
    return {"items": list(items), "total_available": total, "as_of": "2026-09-28T12:00:00Z"}


def _dream(text="a story"):
    return {"id": "dream:19", "text": text, "extra": {}}


@pytest.mark.parametrize("argv", [
    ["--tool", "dreams", "--url", "https://example.org/a"],
    ["--tool", "dreams", "--query", "vision", "--dream-id", "dream:19"],
    ["--tool", "dreams", "--run-id", "r1"],
    ["--self-questions"],
    ["--tool", "curiosity", "--dream-id", "dream:19"],
    ["--tool", "curiosity", "--query", "x", "--run-id", "r1"],
    ["--tool", "curiosity", "--run-id", "r1", "--self-questions"],
])
def test_bad_dreams_arg_combos_exit_via_argparse(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        smoke.parse_args(argv)
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("argv, expected", [
    (["--url", "https://example.org/a"], {"url": "https://example.org/a"}),
    (["--query", "gpus", "--limit", "2"], {"query": "gpus", "limit": 2}),
    ([], {"limit": 3}),
    (["--tool", "dreams", "--dream-id", "dream:19"], {"dream_id": "dream:19"}),
    (["--tool", "dreams", "--query", "vision"], {"query": "vision", "limit": 3}),
    (["--tool", "curiosity"], {"limit": 3}),
    (["--tool", "curiosity", "--run-id", "71238de8f271"], {"run_id": "71238de8f271"}),
    (["--tool", "curiosity", "--query", "stance gate"], {"query": "stance gate", "limit": 3}),
    (["--tool", "curiosity", "--self-questions", "--limit", "2"], {"kind": "self_question", "limit": 2}),
])
def test_arguments_built_per_tool(argv, expected):
    _, arguments = smoke.parse_args(argv)
    assert arguments == expected


def test_named_dream_id_with_no_items_is_degenerate():
    args, _ = smoke.parse_args(["--tool", "dreams", "--dream-id", "dream:19"])
    code, message = smoke.verdict(_result(), args)
    assert code == 1 and "DEGENERATE: dream id not found" in message


def test_named_dream_id_found_is_ok():
    args, _ = smoke.parse_args(["--tool", "dreams", "--dream-id", "dream:19"])
    code, message = smoke.verdict(_result([_dream()], total=1), args)
    assert code == 0 and message.startswith("OK items=1")


def test_empty_dream_text_is_degenerate():
    args, _ = smoke.parse_args(["--tool", "dreams", "--dream-id", "dream:19"])
    code, message = smoke.verdict(_result([_dream(text="")], total=1), args)
    assert code == 1 and "empty text" in message


def test_empty_recent_window_is_degenerate():
    args, _ = smoke.parse_args(["--tool", "dreams"])
    code, _ = smoke.verdict(_result(), args)
    assert code == 1


def _run(status="unsettled", text="I concluded"):
    return {"id": "71238de8f271", "epistemic_status": status, "text": text, "extra": {}}


def test_curiosity_named_run_missing_or_hollow_is_degenerate():
    args, _ = smoke.parse_args(["--tool", "curiosity", "--run-id", "71238de8f271"])
    assert smoke.verdict(_result(), args)[0] == 1
    assert smoke.verdict(_result([_run(text="")], total=1), args)[0] == 1
    assert smoke.verdict(_result([_run()], total=1), args)[0] == 0
    recent, _ = smoke.parse_args(["--tool", "curiosity"])
    assert smoke.verdict(_result(), recent)[0] == 1
    assert smoke.verdict(_result([_run(status="record", text="World question run failed.")], total=1), recent)[0] == 0
