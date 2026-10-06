"""Always-warm Claude Code processes for Juniper's chat replies (spec L5, option a).

Why: a spawned ``claude -p`` spends ~3.4 s (measured, PR #2509) before it can
talk to the model, almost all of it MCP servers starting. A warm process pays
that once. Each chat turn borrows an idle process, wipes its conversation with
``/clear`` (~0.02-0.13 s, MCP servers stay connected), sends the prompt as a
stream-json user message, and reads until the CLI's own ``result`` event.

What a warm process cannot do is change its environment, so nothing per-turn
lives there:

- model upstream, credential, GPU lease header, correlation id -> the
  governor-local relay (``fcc_warm_relay.py``) applies the CURRENT turn's
  values per request;
- the turn clock (``ORION_TURN_*``) -> rewritten into the slot's
  ``CLAUDE_ENV_FILE`` before every turn (the CLI re-sources it on every Bash
  call; verified on 2.1.291);
- ``--model`` and the per-lane context window -> part of the slot's signature;
  a turn that needs a different one spawns as today while the idle slot is
  respawned for it.

Any failure before the prompt goes in returns a reason instead of a turn, and
the motor falls back to a per-turn spawn. See
docs/superpowers/pr-reports/2026-10-06-fcc-chat-warm-pool-pr.md for the full
per-value table.
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import logging
import os
import shlex
import shutil
import signal
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Mapping, Optional, Tuple

from orion.harness.fcc_warm_relay import (
    RelayRegistry,
    RelayServer,
    RelayTurnBinding,
    RelayUpstream,
    summarize_binding,
)

logger = logging.getLogger("orion.harness.fcc_warm_pool")

# An idle slot unused this long gets a /clear round-trip from the health loop,
# so a hung process is found before a turn waits on it.
IDLE_PROBE_AFTER_SEC = 300.0


@dataclass
class WarmPoolConfig:
    size: int = 1
    max_turns: int = 50
    max_age_sec: float = 3600.0
    spawn_timeout_sec: float = 90.0
    clear_timeout_sec: float = 5.0
    health_interval_sec: float = 30.0
    relay_host: str = "127.0.0.1"
    relay_port: int = 7157
    warm_model_label: str = "MODEL_SONNET"
    claude_bin: str = "claude"
    workspace: str = "."
    fcc_server_url: str = "http://127.0.0.1:8080"
    auth_token: str = ""
    state_dir: Optional[str] = None
    stream_read_limit: int = 8 * 1024 * 1024
    graceful_stop_sec: float = 5.0
    respawn_backoff_sec: float = 5.0
    # Longer than any turn can legally run (HARNESS_FCC_TIMEOUT_SEC default 7200).
    busy_reclaim_sec: float = 3 * 3600.0

    @classmethod
    def from_runtime_env(cls, **overrides: Any) -> "WarmPoolConfig":
        """claude bin, workspace, FCC URL and token from the same sources
        ``runner.default_fcc_runner`` uses for a spawned turn."""
        from orion.harness.fcc_motor import expand_env_path, load_fcc_env, resolve_auth_token

        fcc_env = load_fcc_env(expand_env_path(os.environ.get("HARNESS_FCC_ENV_PATH", "~/.fcc/.env")))
        base: Dict[str, Any] = dict(
            claude_bin=os.environ.get("HARNESS_FCC_CLAUDE_BIN", "claude"),
            workspace=os.environ.get("HARNESS_FCC_WORKSPACE", os.getcwd()),
            fcc_server_url=os.environ.get(
                "HARNESS_FCC_SERVER_URL", os.environ.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:8080")
            ),
            auth_token=resolve_auth_token(fcc_env, override=os.environ.get("HARNESS_FCC_AUTH_TOKEN", "")),
        )
        base.update(overrides)
        return cls(**base)


def warm_signature(model_id: str, n_ctx: Optional[int], fcc_env: Mapping[str, str]) -> str:
    """Everything a warm process freezes at spawn, hashed.

    ``--model``, the lane window (``CLAUDE_CODE_AUTO_COMPACT_WINDOW`` and friends),
    ``~/.fcc/.env`` (MCP server config and curiosity credentials come from it)
    and the governor's own environment (every other flag the argv/env builders
    read). A turn only reuses a process whose signature matches exactly.
    """
    payload = {
        "model_id": model_id,
        "n_ctx": n_ctx,
        "fcc_env": sorted((str(k), str(v)) for k, v in fcc_env.items()),
        "environ": sorted(os.environ.items()),
    }
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()[:16]


def _user_line(text: str) -> bytes:
    return (json.dumps({"type": "user", "message": {"role": "user", "content": text}}) + "\n").encode()


def _parse(line: bytes) -> Dict[str, Any]:
    try:
        ev = json.loads(line.decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, ValueError):
        return {}
    return ev if isinstance(ev, dict) else {}


def _kill_group(proc: Any) -> None:
    """SIGKILL the process and its MCP children (it leads its own session).

    Only while the leader is alive: once it is reaped its pid -- and so the
    group id -- could belong to an unrelated process.
    """
    if proc is None or proc.returncode is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def write_turn_env_file(path: Path, values: Mapping[str, str]) -> None:
    """Rewrite the slot's CLAUDE_ENV_FILE with this turn's clock (truncates; absent keys stay absent)."""
    body = "".join(f"export {key}={shlex.quote(str(value))}\n" for key, value in values.items())
    tmp = path.with_suffix(".tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)


class _Slot:
    def __init__(self, idx: int, state_dir: Path) -> None:
        self.slot_id = f"s{idx}"
        # empty -> starting -> idle <-> busy ; retiring -> starting ; dead -> starting
        self.state = "empty"
        self.proc: Any = None
        self.signature: Optional[str] = None
        self.model_id: Optional[str] = None
        self.n_ctx: Optional[int] = None
        self.env_file = state_dir / f"{self.slot_id}.turn_env.sh"
        self.mcp_config_path: Optional[Path] = None
        self.turns = 0
        self.spawned_at = 0.0
        self.last_used = 0.0
        self.generation = 0
        self.stderr_tail: Deque[str] = collections.deque(maxlen=40)
        self.stderr_task: Optional[asyncio.Task] = None
        self.next_retry = 0.0
        self.last_error: Optional[str] = None
        self.busy_since = 0.0
        self.pgid: Optional[int] = None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None


class _Killable:
    """What the motor's cancel registry holds for a warm turn."""

    def __init__(self, turn: "WarmTurn") -> None:
        self._turn = turn

    def kill(self) -> None:
        self._turn.kill()


class WarmTurn:
    """One chat turn on a borrowed warm process. Ends at the CLI's `result` event."""

    def __init__(self, pool: "WarmPool", slot: _Slot, correlation_id: str) -> None:
        self.pool = pool
        self.slot = slot
        self.correlation_id = correlation_id
        self._proc = slot.proc
        self.pid = getattr(slot.proc, "pid", None)
        self.result_seen = False
        self.result_is_error = False
        self.eof = False
        self.killed = False
        self.released = False
        self.recycle_after = False
        self.started_at = time.monotonic()
        self.killable = _Killable(self)

    @property
    def died_unexpectedly(self) -> bool:
        return self.eof and not self.killed and not self.result_seen

    async def send_prompt(self, prompt: str) -> None:
        try:
            stdin = self._proc.stdin
            stdin.write(_user_line(prompt))
            await stdin.drain()
        except (BrokenPipeError, ConnectionResetError, RuntimeError, AttributeError):
            # Died between /clear and the prompt: read as EOF with nothing said,
            # so the motor retries the turn as a spawn instead of failing it.
            self.eof = True

    async def readline(self) -> bytes:
        if self.result_seen or self.eof:
            return b""
        line = await self._proc.stdout.readline()
        if not line:
            self.eof = True
            return b""
        if b'"tool_use"' in line and b"run_in_background" in line:
            # A background shell can outlive the turn inside a warm process (it
            # holds the slot secret); never hand this process to another turn.
            self.recycle_after = True
        if b'"result"' in line:
            ev = _parse(line)
            if ev.get("type") == "result":
                self.result_seen = True
                self.result_is_error = bool(ev.get("is_error"))
        return line

    def kill(self) -> None:
        self.killed = True
        _kill_group(self._proc)

    async def wait(self) -> int:
        if self.result_seen:
            return 1 if self.result_is_error else 0
        try:
            rc = await asyncio.wait_for(self._proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            _kill_group(self._proc)
            return -9
        # A warm process that ends mid-turn failed, whatever its exit code says.
        return rc if rc else 1

    def stderr_snippet(self) -> str:
        return "\n".join(self.slot.stderr_tail).strip()[-500:]

    async def release(self) -> None:
        if self.released:
            return
        self.released = True
        await self.pool._release(self)


class WarmPool:
    def __init__(self, config: WarmPoolConfig) -> None:
        self.config = config
        self.registry = RelayRegistry()
        self.relay: Optional[RelayServer] = None
        self._state_dir: Optional[Path] = None
        self._owns_state_dir = False
        self._slots: List[_Slot] = []
        self._wanted: Optional[Tuple[str, Optional[int]]] = None
        self._tasks: set[asyncio.Task] = set()
        self._health_task: Optional[asyncio.Task] = None
        self.running = False
        self._stopping = False
        self.counters: Dict[str, int] = collections.Counter()

    # ---- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        cfg = self.config
        if cfg.state_dir:
            self._state_dir = Path(cfg.state_dir)
            self._state_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._state_dir = Path(tempfile.mkdtemp(prefix="orion-fcc-warm-"))
            self._owns_state_dir = True
        from orion.harness.fcc_motor import default_relay_upstream

        self.relay = RelayServer(
            self.registry,
            default_relay_upstream(fcc_server_url=cfg.fcc_server_url, auth_token=cfg.auth_token),
            host=cfg.relay_host,
            port=cfg.relay_port,
        )
        try:
            await asyncio.to_thread(self.relay.start)
        except BaseException:
            await asyncio.to_thread(self.relay.stop)
            if self._owns_state_dir:
                shutil.rmtree(self._state_dir, ignore_errors=True)
            raise
        self._slots = [_Slot(i, self._state_dir) for i in range(max(1, int(cfg.size)))]
        self._wanted = self._initial_wanted()
        self.running = True
        if self._wanted is not None:
            for slot in self._slots:
                self._schedule_respawn(slot, reason="warm_up")
        self._health_task = asyncio.create_task(self._health_loop(), name="fcc-warm-pool-health")
        logger.info(
            "fcc_warm_pool_started size=%s relay=%s:%s warm_model=%s max_turns=%s max_age_sec=%s",
            len(self._slots), cfg.relay_host, self.relay.port,
            self._wanted[0] if self._wanted else None, cfg.max_turns, cfg.max_age_sec,
        )

    def _initial_wanted(self) -> Optional[Tuple[str, Optional[int]]]:
        from orion.harness.fcc_motor import expand_env_path, label_to_claude_model_id, load_fcc_env

        fcc_env = load_fcc_env(expand_env_path(os.environ.get("HARNESS_FCC_ENV_PATH", "~/.fcc/.env")))
        try:
            return label_to_claude_model_id(self.config.warm_model_label, fcc_env), None
        except ValueError as exc:
            # Nothing to warm yet; the first chat turn's demand sets it.
            logger.warning("fcc_warm_pool_no_warm_model label=%r error=%s", self.config.warm_model_label, exc)
            return None

    async def stop(self) -> None:
        self._stopping = True
        self.running = False
        if self._health_task is not None:
            self._health_task.cancel()
        for task in list(self._tasks):
            task.cancel()
        for task in [self._health_task, *self._tasks]:
            if task is None:
                continue
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        for slot in self._slots:
            await self._retire_proc(slot, graceful=True)
        if self.relay is not None:
            await asyncio.to_thread(self.relay.stop)
        if self._owns_state_dir and self._state_dir is not None:
            shutil.rmtree(self._state_dir, ignore_errors=True)
        logger.info("fcc_warm_pool_stopped counters=%s", dict(self.counters))

    # ---- turns -----------------------------------------------------------

    async def acquire(
        self,
        *,
        model_id: str,
        n_ctx: Optional[int],
        correlation_id: str,
        binding: RelayTurnBinding,
        turn_env: Mapping[str, str],
        fcc_env: Optional[Mapping[str, str]] = None,
    ) -> Tuple[Optional[WarmTurn], Optional[str]]:
        """(turn, None) on a hit, (None, reason) on a miss. Never raises on a pool fault it can name."""
        started = time.monotonic()
        if not self.running or self._stopping:
            return self._miss(correlation_id, "pool_stopped")
        if fcc_env is None:
            from orion.harness.fcc_motor import expand_env_path, load_fcc_env

            fcc_env = load_fcc_env(expand_env_path(os.environ.get("HARNESS_FCC_ENV_PATH", "~/.fcc/.env")))
        sig = warm_signature(model_id, n_ctx, fcc_env)
        # Keep warm whatever chat turns actually ask for.
        self._wanted = (model_id, n_ctx)

        slot: Optional[_Slot] = None
        # No await from here until the slot is marked busy: two concurrent turns
        # on the same event loop can never pick the same slot.
        for candidate in self._slots:
            if candidate.state != "idle":
                continue
            if not candidate.alive():
                self._schedule_respawn(candidate, reason="died_idle")
                continue
            if candidate.signature == sig:
                slot = candidate
                break
        if slot is None:
            mismatched = [s for s in self._slots if s.state == "idle"]
            if mismatched:
                self._schedule_respawn(mismatched[0], reason="signature_mismatch", graceful=True)
                return self._miss(correlation_id, "signature_mismatch")
            for s in self._slots:
                if s.state in ("empty", "dead") and time.monotonic() >= s.next_retry:
                    self._schedule_respawn(s, reason="demand")
            if any(s.state == "busy" for s in self._slots):
                return self._miss(correlation_id, "pool_busy")
            return self._miss(correlation_id, "pool_warming")
        slot.state = "busy"

        slot.stderr_tail.clear()
        try:
            write_turn_env_file(slot.env_file, turn_env)
            await self._clear(slot, timeout=self.config.clear_timeout_sec)
            self.registry.bind(slot.slot_id, binding)
        except asyncio.CancelledError:
            self._schedule_respawn(slot, reason="acquire_cancelled")
            raise
        except Exception as exc:  # noqa: BLE001 -- named in the reason; the turn falls back
            slot.last_error = f"clear_failed:{type(exc).__name__}: {exc}"
            self._schedule_respawn(slot, reason="clear_failed")
            return self._miss(correlation_id, f"clear_failed:{type(exc).__name__}")
        slot.turns += 1
        slot.last_used = time.monotonic()
        slot.busy_since = slot.last_used
        self.counters["hit"] += 1
        logger.info(
            "fcc_warm_pool_acquired corr=%s slot=%s gen=%s pid=%s turn=%s acquire_ms=%d",
            correlation_id, slot.slot_id, slot.generation, getattr(slot.proc, "pid", None),
            slot.turns, int((time.monotonic() - started) * 1000),
        )
        return WarmTurn(self, slot, correlation_id), None

    def _miss(self, correlation_id: str, reason: str) -> Tuple[None, str]:
        self.counters[f"miss:{reason.split(':', 1)[0]}"] += 1
        return None, reason

    async def _release(self, turn: WarmTurn) -> None:
        slot = turn.slot
        self.registry.unbind(slot.slot_id)
        if slot.proc is not turn._proc:
            return  # already replaced (e.g. pool stopped mid-turn)
        healthy = turn.result_seen and not turn.killed and slot.alive()
        if not healthy:
            reason = "killed" if turn.killed else ("process_died" if turn.eof else "turn_abandoned")
            logger.warning(
                "fcc_warm_pool_slot_discarded corr=%s slot=%s pid=%s reason=%s",
                turn.correlation_id, slot.slot_id, turn.pid, reason,
            )
            self._schedule_respawn(slot, reason=reason)
            return
        if turn.recycle_after:
            self._schedule_respawn(slot, reason="recycle_background_shell")
            return
        if slot.turns >= self.config.max_turns:
            self._schedule_respawn(slot, reason="recycle_max_turns", graceful=True)
            return
        if time.monotonic() - slot.spawned_at >= self.config.max_age_sec:
            self._schedule_respawn(slot, reason="recycle_max_age", graceful=True)
            return
        slot.state = "idle"

    # ---- process management ---------------------------------------------

    async def _clear(self, slot: _Slot, *, timeout: float) -> None:
        """Send /clear and wait for the CLI's reset -> init -> result. Raises on anything else."""
        proc = slot.proc
        if proc is None or proc.returncode is not None:
            raise RuntimeError("warm process is not running")
        proc.stdin.write(_user_line("/clear"))
        await proc.stdin.drain()
        saw_reset = False
        end = time.monotonic() + timeout
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("no /clear acknowledgement")
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
            if not line:
                raise RuntimeError("warm process exited during /clear")
            ev = _parse(line)
            etype = ev.get("type")
            if etype == "conversation_reset":
                saw_reset = True
            elif saw_reset and etype == "result":
                if ev.get("is_error"):
                    raise RuntimeError(f"/clear returned an error result: {str(ev.get('result'))[:200]}")
                return

    def _schedule_respawn(self, slot: _Slot, *, reason: str, graceful: bool = False) -> None:
        if self._stopping or slot.state in ("starting", "retiring"):
            return
        slot.state = "retiring" if slot.proc is not None else "starting"
        self.counters[f"respawn:{reason}"] += 1
        task = asyncio.create_task(self._respawn(slot, reason=reason, graceful=graceful))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _respawn(self, slot: _Slot, *, reason: str, graceful: bool) -> None:
        await self._retire_proc(slot, graceful=graceful)
        wanted = self._wanted
        if wanted is None or self._stopping:
            slot.state = "empty"
            return
        slot.state = "starting"
        try:
            await self._spawn(slot, model_id=wanted[0], n_ctx=wanted[1], reason=reason)
        except asyncio.CancelledError:
            await self._retire_proc(slot, graceful=False)
            slot.state = "empty"
            raise
        except Exception as exc:  # noqa: BLE001 -- slot goes dead; health loop retries
            slot.last_error = f"spawn_failed:{type(exc).__name__}: {exc}"
            logger.warning(
                "fcc_warm_pool_spawn_failed slot=%s reason=%s error=%r", slot.slot_id, reason, exc
            )
            await self._retire_proc(slot, graceful=False)
            slot.state = "dead"
            slot.next_retry = time.monotonic() + self.config.respawn_backoff_sec

    async def _spawn(self, slot: _Slot, *, model_id: str, n_ctx: Optional[int], reason: str) -> None:
        from orion.harness import fcc_motor as m

        assert self.relay is not None
        cfg = self.config
        fcc_env = m.load_fcc_env(m.expand_env_path(os.environ.get("HARNESS_FCC_ENV_PATH", "~/.fcc/.env")))
        sig = warm_signature(model_id, n_ctx, fcc_env)
        slot.generation += 1
        secret = self.registry.register_slot(slot.slot_id)
        env = m._build_subprocess_env(
            chat_reply=True,
            gpu_lease=None,
            n_ctx=n_ctx,
            fcc_server_url=self.relay.slot_base_url(slot.slot_id),
            auth_token=secret,
            fcc_env=fcc_env,
            turn_budget_sec=None,
            turn_deadline_epoch=None,
            turn_step_stall_sec=None,
        )
        # The relay supplies the real credential per turn.
        env.pop("ANTHROPIC_API_KEY", None)
        env["CLAUDE_ENV_FILE"] = str(slot.env_file)
        write_turn_env_file(slot.env_file, {})
        process_tag = f"fcc-warm-{slot.slot_id}-g{slot.generation}"
        mcp_config_path = m._maybe_render_mcp_config(correlation_id=process_tag)
        argv = m.build_claude_argv(
            claude_bin=cfg.claude_bin,
            model_id=model_id,
            prompt=None,
            correlation_id=process_tag,
            mcp_config_path=mcp_config_path,
        )
        started = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cfg.workspace,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=max(65536, int(cfg.stream_read_limit)),
                start_new_session=True,
            )
        except BaseException:
            if mcp_config_path is not None:
                from orion.fcc.mcp_config import cleanup_mcp_config

                cleanup_mcp_config(mcp_config_path)
            raise
        slot.proc = proc
        slot.pgid = proc.pid  # start_new_session: the leader's pid is the group id
        slot.mcp_config_path = mcp_config_path
        slot.stderr_tail.clear()
        slot.stderr_task = asyncio.create_task(self._drain_stderr(slot, proc))
        # The first /clear makes the CLI start (and wait for) its MCP servers, so
        # an idle slot is genuinely warm, not just forked.
        await self._clear(slot, timeout=cfg.spawn_timeout_sec)
        slot.signature = sig
        slot.model_id = model_id
        slot.n_ctx = n_ctx
        slot.turns = 0
        slot.spawned_at = time.monotonic()
        slot.last_used = slot.spawned_at
        slot.last_error = None
        slot.state = "idle"
        logger.info(
            "fcc_warm_pool_spawned slot=%s gen=%s pid=%s reason=%s model=%s n_ctx=%s sig=%s spawn_ms=%d",
            slot.slot_id, slot.generation, proc.pid, reason, model_id, n_ctx, sig,
            int((time.monotonic() - started) * 1000),
        )

    async def _retire_proc(self, slot: _Slot, *, graceful: bool) -> None:
        proc = slot.proc
        if proc is not None and proc.returncode is None:
            if graceful:
                try:
                    proc.stdin.close()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=self.config.graceful_stop_sec)
                except asyncio.TimeoutError:
                    _kill_group(proc)
            else:
                _kill_group(proc)
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("fcc_warm_pool_reap_timeout slot=%s pid=%s", slot.slot_id, proc.pid)
        if slot.pgid is not None:
            # Leader is reaped; sweep any MCP children left in its group. Done
            # immediately after reaping, so the group id cannot have been reused.
            try:
                os.killpg(slot.pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            slot.pgid = None
        if slot.stderr_task is not None:
            slot.stderr_task.cancel()
            slot.stderr_task = None
        if slot.mcp_config_path is not None:
            from orion.fcc.mcp_config import cleanup_mcp_config

            cleanup_mcp_config(slot.mcp_config_path)
            slot.mcp_config_path = None
        self.registry.forget_slot(slot.slot_id)
        slot.proc = None
        slot.signature = None

    async def _drain_stderr(self, slot: _Slot, proc: Any) -> None:
        # A long-lived process's stderr pipe must be read, or it fills and blocks the CLI.
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    return
                slot.stderr_tail.append(line.decode("utf-8", errors="replace").rstrip())
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            return

    async def _health_loop(self) -> None:
        while not self._stopping:
            await asyncio.sleep(self.config.health_interval_sec)
            await self.health_tick()

    async def health_tick(self) -> None:
        """Respawn dead slots, recycle old idle ones, and /clear-probe long-idle ones."""
        now = time.monotonic()
        for slot in self._slots:
            if slot.state == "idle" and not slot.alive():
                self._schedule_respawn(slot, reason="died_idle")
            elif slot.state == "idle" and now - slot.spawned_at >= self.config.max_age_sec:
                self._schedule_respawn(slot, reason="recycle_max_age", graceful=True)
            elif slot.state in ("empty", "dead") and self._wanted is not None and now >= slot.next_retry:
                self._schedule_respawn(slot, reason="health")
            elif (
                slot.state == "busy"
                and slot.busy_since
                and now - slot.busy_since >= self.config.busy_reclaim_sec
            ):
                # A turn that never released (should not happen; release is in a
                # finally). Reclaim rather than leave the pool permanently busy.
                logger.warning("fcc_warm_pool_busy_reclaimed slot=%s busy_sec=%.0f", slot.slot_id, now - slot.busy_since)
                self.registry.unbind(slot.slot_id)
                self._schedule_respawn(slot, reason="busy_reclaimed")
            elif slot.state == "idle" and now - slot.last_used >= IDLE_PROBE_AFTER_SEC:
                slot.state = "busy"
                try:
                    await self._clear(slot, timeout=self.config.clear_timeout_sec)
                except Exception as exc:  # noqa: BLE001
                    slot.last_error = f"idle_probe_failed:{type(exc).__name__}"
                    self._schedule_respawn(slot, reason="idle_probe_failed")
                    continue
                slot.last_used = time.monotonic()
                slot.state = "idle"

    # ---- debug surface ----------------------------------------------------

    def status(self) -> Dict[str, Any]:
        now = time.monotonic()
        return {
            "running": self.running,
            "relay_port": self.relay.port if self.relay else None,
            "wanted_model": self._wanted[0] if self._wanted else None,
            "wanted_n_ctx": self._wanted[1] if self._wanted else None,
            "counters": dict(self.counters),
            "slots": [
                {
                    "slot": s.slot_id,
                    "state": s.state,
                    "pid": getattr(s.proc, "pid", None),
                    "generation": s.generation,
                    "model": s.model_id,
                    "n_ctx": s.n_ctx,
                    "turns": s.turns,
                    "age_sec": round(now - s.spawned_at, 1) if s.spawned_at else None,
                    "last_error": s.last_error,
                    "relay": summarize_binding(self.registry.binding(s.slot_id)),
                }
                for s in self._slots
            ],
        }


_POOL: Optional[WarmPool] = None
# Why the last start failed, for /health (None = never failed or since recovered).
LAST_START_ERROR: Optional[str] = None


def get_warm_pool() -> Optional[WarmPool]:
    pool = _POOL
    return pool if pool is not None and pool.running else None


async def start_warm_pool(config: WarmPoolConfig) -> WarmPool:
    global _POOL, LAST_START_ERROR
    if _POOL is not None:
        await stop_warm_pool()
    pool = WarmPool(config)
    try:
        await pool.start()
    except BaseException as exc:
        LAST_START_ERROR = f"{type(exc).__name__}: {exc}"
        raise
    LAST_START_ERROR = None
    _POOL = pool
    return pool


async def stop_warm_pool() -> None:
    global _POOL
    pool, _POOL = _POOL, None
    if pool is not None:
        await pool.stop()


__all__ = [
    "RelayUpstream",
    "WarmPool",
    "WarmPoolConfig",
    "WarmTurn",
    "get_warm_pool",
    "start_warm_pool",
    "stop_warm_pool",
    "warm_signature",
    "write_turn_env_file",
]
