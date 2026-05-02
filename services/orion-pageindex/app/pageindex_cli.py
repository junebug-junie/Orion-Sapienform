from __future__ import annotations

import json
import re
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
        if "{query}" not in self._query_args:
            raise PageIndexCliError(
                "PageIndex query is not supported by current runner configuration; set PAGEINDEX_QUERY_ARGS with a {query} placeholder"
            )
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

    def query_local_tree(self, *, md_path: Path, query: str, top_k: int = 8) -> dict[str, Any]:
        result_file = self._repo_path / "results" / f"{md_path.stem}_structure.json"
        if not result_file.exists():
            raise PageIndexCliError(f"PageIndex tree artifact missing: {result_file}")
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PageIndexCliError(f"Invalid PageIndex tree artifact: {result_file}") from exc

        query_terms = self._tokenize(query)
        if not query_terms:
            return {"results": [], "source": "pageindex_local_tree"}

        flattened: list[dict[str, Any]] = []
        self._flatten_nodes(payload, flattened)

        scored: list[tuple[int, dict[str, Any]]] = []
        for node in flattened:
            text = " ".join(
                str(node.get(key) or "")
                for key in ("title", "summary", "text", "content", "description")
            ).lower()
            if not text:
                continue
            score = sum(1 for term in query_terms if term in text)
            if score > 0:
                scored.append((score, node))

        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, node in scored[:top_k]:
            results.append(
                {
                    "node_id": node.get("node_id"),
                    "heading": node.get("title"),
                    "excerpt": (node.get("summary") or node.get("text") or "")[:500],
                    "created_at": None,
                    "source_kind": "journal",
                    "provenance": {
                        "engine": "pageindex_local_tree",
                        "score": score,
                        "start_index": node.get("start_index"),
                        "end_index": node.get("end_index"),
                    },
                }
            )

        return {"results": results, "source": "pageindex_local_tree", "artifact_path": str(result_file)}

    def _flatten_nodes(self, node: Any, out: list[dict[str, Any]]) -> None:
        if isinstance(node, dict):
            if any(key in node for key in ("title", "summary", "text", "node_id")):
                out.append(node)
            children = node.get("nodes")
            if isinstance(children, list):
                for child in children:
                    self._flatten_nodes(child, out)
        elif isinstance(node, list):
            for child in node:
                self._flatten_nodes(child, out)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(token) > 2]
