from __future__ import annotations

import asyncio
import contextlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


_SAFE_SERVICE_NAME = re.compile(r"^[a-zA-Z0-9._-]+$")


@dataclass(frozen=True)
class ServiceLogConfig:
    name: str
    compose_file: Path
    service_env_file: Optional[Path]


def _repo_root() -> Path:
    env_root = os.getenv("ORION_REPO_ROOT", "").strip()
    if env_root:
        candidate = Path(env_root).resolve()
        if (candidate / "services").is_dir():
            return candidate

    module_path = Path(__file__).resolve()
    search_roots = [module_path.parent, *module_path.parents]
    for candidate in search_roots:
        if (candidate / "services").is_dir():
            return candidate

    cwd = Path.cwd().resolve()
    if (cwd / "services").is_dir():
        return cwd

    for fallback in (
        Path("/workspace/Orion-Sapienform"),
        Path("/workspace"),
        Path("/repo"),
    ):
        if (fallback / "services").is_dir():
            return fallback

    return module_path.parent


def resolve_repo_root() -> Path:
    return _repo_root()


def discover_loggable_services(repo_root: Optional[Path] = None) -> List[ServiceLogConfig]:
    root = repo_root or _repo_root()
    services_root = root / "services"
    if not services_root.exists() or not services_root.is_dir():
        return []

    discovered: List[ServiceLogConfig] = []
    for entry in sorted(services_root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        service_name = entry.name
        if not _SAFE_SERVICE_NAME.match(service_name):
            continue

        compose_file = entry / "docker-compose.yml"
        if not compose_file.is_file():
            continue

        service_env_file = entry / ".env"
        discovered.append(
            ServiceLogConfig(
                name=service_name,
                compose_file=compose_file,
                service_env_file=service_env_file if service_env_file.is_file() else None,
            )
        )

    return discovered


def build_compose_logs_command(config: ServiceLogConfig, repo_root: Optional[Path] = None) -> List[str]:
    root = repo_root or _repo_root()
    cmd: List[str] = ["docker", "compose", "--env-file", ".env"]

    if config.service_env_file is not None:
        cmd.extend(["--env-file", os.fspath(config.service_env_file.relative_to(root))])

    cmd.extend([
        "-f",
        os.fspath(config.compose_file.relative_to(root)),
        "logs",
        "-f",
        "--no-color",
        "--timestamps",
    ])
    return cmd


class ServiceLogProcess:
    def __init__(self, config: ServiceLogConfig, out_queue: asyncio.Queue, repo_root: Optional[Path] = None) -> None:
        self.config = config
        self._out_queue = out_queue
        self._repo_root = repo_root or _repo_root()
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        if self._proc and self._proc.returncode is None:
            return

        cmd = build_compose_logs_command(self.config, self._repo_root)
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self._repo_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._tasks = [
            asyncio.create_task(self._pump_stream(self._proc.stdout, "stdout")),
            asyncio.create_task(self._pump_stream(self._proc.stderr, "stderr")),
            asyncio.create_task(self._wait_for_exit()),
        ]

    async def stop(self) -> None:
        proc = self._proc
        if proc and proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()

        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        self._tasks.clear()
        self._proc = None

    async def _pump_stream(self, stream: Optional[asyncio.StreamReader], stream_name: str) -> None:
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip("\r\n")
            await self._out_queue.put({
                "service": self.config.name,
                "stream": stream_name,
                "line": text,
            })

    async def _wait_for_exit(self) -> None:
        if self._proc is None:
            return
        code = await self._proc.wait()
        await self._out_queue.put({
            "service": self.config.name,
            "stream": "process",
            "line": f"[process exited with code {code}]",
            "event": "service_exit",
            "exit_code": code,
        })


class ServiceLogSession:
    def __init__(self, repo_root: Optional[Path] = None) -> None:
        self._repo_root = repo_root or _repo_root()
        self._available: Dict[str, ServiceLogConfig] = {
            cfg.name: cfg for cfg in discover_loggable_services(self._repo_root)
        }
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=5000)
        self._active: Dict[str, ServiceLogProcess] = {}

    @property
    def available_services(self) -> List[str]:
        return sorted(self._available.keys())

    async def set_selected_services(self, names: Iterable[str]) -> List[str]:
        requested = [name for name in names if name in self._available]
        requested_set = set(requested)

        for name in list(self._active.keys()):
            if name in requested_set:
                continue
            proc = self._active.pop(name)
            await proc.stop()
            await self._queue.put({"event": "service_stopped", "service": name, "line": "[stream stopped]"})

        for name in requested:
            if name in self._active:
                continue
            proc = ServiceLogProcess(self._available[name], self._queue, self._repo_root)
            self._active[name] = proc
            await proc.start()
            await self._queue.put({"event": "service_started", "service": name, "line": "[stream started]"})

        return sorted(requested_set)

    async def next_event(self) -> Dict[str, object]:
        return await self._queue.get()

    async def close(self) -> None:
        for name in list(self._active.keys()):
            proc = self._active.pop(name)
            await proc.stop()
