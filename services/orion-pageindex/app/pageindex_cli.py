from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


class PageIndexCliError(RuntimeError):
    pass


class PageIndexCli:
    def __init__(
        self,
        *,
        python_bin: str,
        repo_path: str,
        run_script: str,
        build_args: str,
        query_args: str,
        timeout_sec: int,
    ) -> None:
        self._python_bin = python_bin
        self._repo_path = Path(repo_path)
        self._run_script = run_script
        self._build_args = build_args
        self._query_args = query_args
        self._timeout = timeout_sec

    def installation_proof(self) -> dict[str, Any]:
        script_path = self._repo_path / self._run_script
        return {
            "repo_path": str(self._repo_path),
            "run_script": str(script_path),
            "repo_exists": self._repo_path.exists(),
            "run_script_exists": script_path.exists(),
        }

    def build(self, *, md_path: Path, artifact_dir: Path) -> dict[str, Any]:
        args = self._build_args.format(md_path=shlex.quote(str(md_path)), artifact_dir=shlex.quote(str(artifact_dir)))
        return self._run(args)

    def query(self, *, md_path: Path, artifact_dir: Path, query: str) -> dict[str, Any]:
        args = self._query_args.format(
            md_path=shlex.quote(str(md_path)),
            artifact_dir=shlex.quote(str(artifact_dir)),
            query=shlex.quote(query),
        )
        return self._run(args)

    def _run(self, formatted_args: str) -> dict[str, Any]:
        script_path = self._repo_path / self._run_script
        if not script_path.exists():
            raise PageIndexCliError(f"PageIndex run script missing: {script_path}")
        cmd = [self._python_bin, str(script_path), *shlex.split(formatted_args)]
        proc = subprocess.run(
            cmd,
            cwd=str(self._repo_path),
            capture_output=True,
            text=True,
            timeout=self._timeout,
            check=False,
        )
        if proc.returncode != 0:
            raise PageIndexCliError(proc.stderr.strip() or proc.stdout.strip() or f"exit={proc.returncode}")
        data = {
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "returncode": proc.returncode,
            "cmd": cmd,
        }
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    data["json"] = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        return data
