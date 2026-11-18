"""Merge Orion's lane defaults into a Strata run config (stdlib only; runs on the Strata image's Python).

Strata's setup rewrites only its own keys (setup.py SETUP_KEYS) and keeps the rest, so a key merged
here survives restarts and re-setup. Each lane's defaults match the llama.cpp lane it replaces
(strata/chat-defaults.json, strata/agent-defaults.json).

--context N sets the run config's --max-context, for a config seeded from another lane: Strata compares
that number with the launch's --context and runs setup again when they differ (strata_cli/launch/runner.py
desc_matches), which re-downloads llama.cpp and touches the shared model folder.

usage: strata_apply_defaults.py <strata-<model>.json> <defaults.json> [--context N]
"""
import json
import sys
from pathlib import Path


def merged(config: dict, defaults: dict, context: int | None = None) -> dict:
    out = {**config, **defaults}
    if context is not None:
        args = list(out.get("args") or [])
        if "--max-context" not in args[:-1]:
            raise ValueError("the run config's args have no --max-context to set")
        args[args.index("--max-context") + 1] = str(context)
        out["args"] = args
    return out


def main(argv: list[str]) -> int:
    context = None
    if len(argv) == 5 and argv[3] == "--context" and argv[4].isdigit():
        context, argv = int(argv[4]), argv[:3]
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    path, defaults = Path(argv[1]), json.loads(Path(argv[2]).read_text())
    if not path.is_file():
        print(f"{path} does not exist: start Strata once so its setup writes it", file=sys.stderr)
        return 1
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    try:
        new = merged(config, defaults, context)
    except ValueError as exc:
        print(f"{path}: {exc}", file=sys.stderr)
        return 1
    changed = sorted(k for k in new if config.get(k) != new[k])
    if changed:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(new, indent=1), encoding="utf-8")
        tmp.replace(path)
    print(f"{path}: {'set ' + ', '.join(changed) if changed else 'already current'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
