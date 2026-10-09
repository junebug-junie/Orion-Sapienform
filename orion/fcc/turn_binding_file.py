"""Per-turn tool binding for a long-lived (warm) FCC process.

A spawned ``claude -p`` gets its reading/introspect binding as a fixed env var
on each MCP server, because the servers live exactly one turn. A warm process
keeps its MCP servers across turns, so the binding has to change underneath
them: the warm slot owns one small file per server, the motor writes the
current turn's binding into it before the prompt goes in, and clears it when
the turn ends. The MCP server reads the file on every tool call.

An empty or missing file means "no turn is bound": the tool call fails rather
than attributing work to a previous turn.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Type, TypeVar

from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


class NoTurnBoundError(RuntimeError):
    """The warm process has no turn bound right now."""


def write_binding_file(path: Path, binding: Optional[BaseModel]) -> None:
    """Atomically replace the file with this turn's binding (``None`` clears it)."""
    path = Path(path)
    body = binding.model_dump_json() if binding is not None else ""
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(body)
    os.replace(tmp, path)


def read_binding_file(path: Path, model: Type[M]) -> M:
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        text = ""
    if not text:
        raise NoTurnBoundError("no turn is bound to this warm process; tool unavailable")
    return model.model_validate_json(text)


def binding_source(model: Type[M], *, env_key: str, file_env_key: str) -> Callable[[], M]:
    """How an MCP server gets its binding: re-read a file per call, or a fixed env value."""
    file_path = os.environ.get(file_env_key, "").strip()
    if file_path:
        return lambda: read_binding_file(Path(file_path), model)
    fixed = model.model_validate_json(os.environ[env_key])
    return lambda: fixed
